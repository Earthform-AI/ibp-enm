"""PyG graph data pipeline — convert ibp_enm proteins into PyTorch Geometric Data objects.

Phase B of the GNCA infrastructure.  Bridges the existing numpy-based
ENM pipeline with PyTorch Geometric's ``Data`` format for graph neural
network training.

Node features (per-residue)
---------------------------
- Cα coordinates (3) — also stored in ``data.pos``
- Experimental B-factor (1)
- Predicted B-factors: Laplacian, continuous, uniform (3)
- Fiedler vector value (1)
- Domain label (1)
- Hinge score (1)
- Stabiliser profiles: Laplacian, continuous (2)
- Node degree (1)
- Spectral gap value (1, broadcast to all nodes)
- Per-residue entropy contribution from top eigenvalues (1)

*Total node feature dimension: 15*

Edge features (per-edge)
------------------------
- Contact distance in Ångströms (1)
- Cross-domain flag (1)

*Total edge feature dimension: 2*

Targets
-------
- ``data.y``: integer archetype class label
- ``data.archetype_name``: string for convenience

Usage
-----
>>> from ibp_enm.graph_data import protein_to_pyg_data, corpus_to_dataset
>>>
>>> # Single protein
>>> data = protein_to_pyg_data("2LZM", "A", archetype="globin")
>>> data.x.shape   # (N, 15)
>>> data.edge_index.shape  # (2, 2*E)
>>>
>>> # Full corpus → DataLoader
>>> from ibp_enm.benchmark import LARGE_CORPUS
>>> dataset = corpus_to_dataset(LARGE_CORPUS, cache_dir="~/.ibp_enm_cache")
>>> loader = dataset_to_loader(dataset, batch_size=16)

Requires
--------
``pip install torch torch-geometric`` (or ``pip install ibp-enm[gnca]``)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Lazy imports for optional PyG deps ──────────────────────────

_TORCH_AVAILABLE = False
_PYG_AVAILABLE = False


def _check_torch():
    global _TORCH_AVAILABLE
    try:
        import torch  # noqa: F401
        _TORCH_AVAILABLE = True
    except ImportError:
        raise ImportError(
            "PyTorch is required for graph_data.  "
            "Install with: pip install torch  "
            "or: pip install ibp-enm[gnca]"
        )


def _check_pyg():
    global _PYG_AVAILABLE
    try:
        import torch_geometric  # noqa: F401
        _PYG_AVAILABLE = True
    except ImportError:
        raise ImportError(
            "PyTorch Geometric is required for graph_data.  "
            "Install with: pip install torch-geometric  "
            "or: pip install ibp-enm[gnca]"
        )


# ── Archetype label mapping ────────────────────────────────────

ARCHETYPE_NAMES: List[str] = [
    "dumbbell",
    "barrel",
    "globin",
    "enzyme_active",
    "allosteric",
]
"""Canonical archetype ordering (matches ARCHETYPES list in archetypes.py)."""

ARCHETYPE_TO_IDX: Dict[str, int] = {
    name: idx for idx, name in enumerate(ARCHETYPE_NAMES)
}

NUM_CLASSES: int = len(ARCHETYPE_NAMES)

# ── Feature extraction constants ───────────────────────────────

NODE_FEATURE_DIM: int = 15
THERMO_FEATURE_DIM: int = 7
EDGE_FEATURE_DIM: int = 2
CONTACT_CUTOFF: float = 8.0  # Ångströms — matches IBPProteinAnalyzer default


def get_input_dim(thermo_features: bool = False) -> int:
    """Return the node feature dimension for the given configuration.

    Base features: 15 (B-factors, Fiedler, hinge, stabilisers, etc.)
    Thermo features (+7): vibrational entropy, heat capacity, spectral
    entropy, mean IPR, entropy Gini, entropy CV, hinge occupation ratio.
    """
    return NODE_FEATURE_DIM + (THERMO_FEATURE_DIM if thermo_features else 0)


__all__ = [
    "ARCHETYPE_NAMES",
    "ARCHETYPE_TO_IDX",
    "NUM_CLASSES",
    "NODE_FEATURE_DIM",
    "THERMO_FEATURE_DIM",
    "EDGE_FEATURE_DIM",
    "get_input_dim",
    "protein_to_pyg_data",
    "corpus_to_dataset",
    "dataset_to_loader",
    "stratified_split",
    "DatasetStats",
    "compute_dataset_stats",
]


# ═══════════════════════════════════════════════════════════════════
# Core: protein → PyG Data
# ═══════════════════════════════════════════════════════════════════

def protein_to_pyg_data(
    pdb_id: str,
    chain: str = "A",
    archetype: Optional[str] = None,
    name: Optional[str] = None,
    *,
    cache_dir: str = "~/.ibp_enm_cache",
    use_cache: bool = True,
    normalize_features: bool = False,
    include_coords: bool = True,
    virtual_node: bool = False,
    thermo_features: bool = False,
) -> "torch_geometric.data.Data":
    """Convert a single protein into a PyG ``Data`` object.

    The function first attempts to load pre-computed ENM profiles from
    the cache (fast path).  If no cache entry exists or ``use_cache``
    is ``False``, it runs the full ENM pipeline from scratch (slow path,
    ~2 min per protein).

    Parameters
    ----------
    pdb_id : str
        4-char PDB accession code.
    chain : str
        Chain identifier (default ``"A"``).
    archetype : str or None
        Ground-truth archetype label.  If None, ``data.y`` is set to -1.
    name : str or None
        Human-readable name (stored as ``data.protein_name``).
    cache_dir : str
        Directory for cached profiles.
    use_cache : bool
        Whether to use cached profiles when available.
    normalize_features : bool
        If ``True``, z-score normalise each node feature column.
    include_coords : bool
        If ``True``, include absolute Cα coordinates in node features
        (adds 3 dims).  Set to ``False`` for rotationally-invariant
        features only (recommended for classification).
    virtual_node : bool
        If ``True``, append a virtual global node (index N) connected
        bidirectionally to all residue nodes.  The virtual node carries
        graph-level discriminative features (thermodynamic observables,
        spectral gap, etc.) mapped to the same feature space.  This
        enables local message-passing to access global topology info.
    thermo_features : bool
        If ``True``, append 7 thermodynamic band features as extra
        columns (broadcast to all nodes): vibrational entropy, heat
        capacity, spectral entropy, mean IPR, entropy Gini, entropy CV,
        and hinge occupation ratio.  Increases node feature dim from
        15 to 22.

    Returns
    -------
    torch_geometric.data.Data
        Graph with node features ``x``, edge connectivity ``edge_index``,
        edge features ``edge_attr``, positions ``pos``, label ``y``, and
        metadata attributes.
    """
    _check_torch()
    _check_pyg()
    import torch
    from torch_geometric.data import Data

    from .analyzer import IBPProteinAnalyzer
    from .band import _fetch_ca
    from .thermodynamics import per_residue_entropy_contribution

    name = name or pdb_id

    # ── Fetch coordinates ───────────────────────────────────────
    coords, bfactors = _fetch_ca(pdb_id, chain)
    N = len(coords)
    if N < 20:
        raise ValueError(
            f"Too few residues ({N}) for {pdb_id}:{chain}. "
            f"Minimum is 20."
        )

    # ── ENM analysis ────────────────────────────────────────────
    analyzer = IBPProteinAnalyzer(cutoff=CONTACT_CUTOFF)
    result = analyzer.analyze(coords, bfactors)
    contacts, degrees = analyzer._build_contacts(coords, N)

    if len(contacts) < N // 2:
        raise ValueError(
            f"Contact graph too sparse for {pdb_id}:{chain}: "
            f"{len(contacts)} contacts for {N} residues."
        )

    # ── Build edge_index and edge_attr ──────────────────────────
    src_list, dst_list = [], []
    dist_list = []
    for (i, j), _ in contacts.items():
        # Bidirectional edges
        src_list.extend([i, j])
        dst_list.extend([j, i])
        d = np.linalg.norm(coords[i] - coords[j])
        dist_list.extend([d, d])

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

    # Cross-domain flag
    domain_labels = result.domain_labels
    if domain_labels is None:
        domain_labels = np.zeros(N, dtype=int)

    cross_domain = []
    for s, d in zip(src_list, dst_list):
        cross_domain.append(float(domain_labels[s] != domain_labels[d]))

    edge_attr = torch.tensor(
        list(zip(dist_list, cross_domain)),
        dtype=torch.float,
    )

    # ── Build node features ─────────────────────────────────────
    # (N, 15) tensor

    # 1. B-factor (experimental)
    b_exp = bfactors if len(bfactors) == N else np.zeros(N)

    # 2–4. Predicted B-factors
    b_lap = result.b_laplacian if result.b_laplacian is not None else np.zeros(N)
    b_cont = result.b_continuous if result.b_continuous is not None else np.zeros(N)
    b_uni = result.b_uniform if result.b_uniform is not None else np.zeros(N)

    # 5. Fiedler vector
    fiedler = result.fiedler_vector if result.fiedler_vector is not None else np.zeros(N)

    # 6. Domain label (integer, but as float for feature vector)
    dom = domain_labels.astype(float)

    # 7. Hinge score
    hinge = result.hinge_scores if result.hinge_scores is not None else np.zeros(N)

    # 8–9. Stabiliser profiles
    stab_lap = result.stabilizer_laplacian if result.stabilizer_laplacian is not None else np.zeros(N)
    stab_cont = result.stabilizer_continuous if result.stabilizer_continuous is not None else np.zeros(N)

    # 10. Node degree
    deg = np.array(degrees if isinstance(degrees, (list, np.ndarray)) else
                   [degrees.get(i, 0) for i in range(N)], dtype=float)

    # 11. Spectral gap (broadcast)
    sg = result.spectral_gap if result.spectral_gap is not None else 0.0
    sg_arr = np.full(N, sg)

    # 12. Per-residue entropy contribution
    evals = result.laplacian_eigenvalues
    evecs = result.laplacian_eigenvectors
    if evals is not None and evecs is not None:
        try:
            per_res_ent = per_residue_entropy_contribution(evals, evecs)
        except Exception:
            per_res_ent = np.zeros(N)
    else:
        per_res_ent = np.zeros(N)

    # Stack features
    feature_cols = [
        b_exp,       # 0: experimental B-factor
        b_lap,       # 1: predicted B-factor (Laplacian)
        b_cont,      # 2: predicted B-factor (continuous)
        b_uni,       # 3: predicted B-factor (uniform)
        fiedler,     # 4: Fiedler vector
        dom,         # 5: domain label
        hinge,       # 6: hinge score
        stab_lap,    # 7: stabiliser (Laplacian)
        stab_cont,   # 8: stabiliser (continuous)
        deg,         # 9: node degree
        sg_arr,      # 10: spectral gap
        per_res_ent, # 11: per-residue entropy
    ]
    if include_coords:
        feature_cols.extend([
            coords[:, 0],  # 12: x-coordinate
            coords[:, 1],  # 13: y-coordinate
            coords[:, 2],  # 14: z-coordinate
        ])
    else:
        # Rotationally-invariant spatial features instead
        com = coords.mean(axis=0)
        d_com = np.linalg.norm(coords - com, axis=1)  # distance from centroid
        from scipy.spatial.distance import pdist, squareform
        dmat = squareform(pdist(coords))
        local_density = (dmat < 6.0).sum(axis=1) - 1  # neighbours within 6Å
        contact_density = (dmat < CONTACT_CUTOFF).sum(axis=1) - 1  # neighbours within 8Å
        feature_cols.extend([
            d_com.astype(float),             # 12: distance from centroid
            local_density.astype(float),     # 13: local density (6Å)
            contact_density.astype(float),   # 14: contact density (8Å)
        ])

    feature_matrix = np.column_stack(feature_cols)
    n_features = feature_matrix.shape[1]
    assert n_features == NODE_FEATURE_DIM, (
        f"Base feature shape mismatch: {n_features} != {NODE_FEATURE_DIM}"
    )

    # ── Z-score normalise base features (per-protein) ───────────
    # Done BEFORE appending thermo features (which are global constants
    # per protein and would be zeroed out by per-protein z-scoring).
    if normalize_features:
        feature_matrix = _z_normalize(feature_matrix)

    # ── Compute global thermodynamic observables ────────────────
    # Shared by Strategy 1 (virtual node) and Strategy 2 (thermo features).
    _thermo = None
    if virtual_node or thermo_features:
        from .thermodynamics import (
            vibrational_entropy as _vib_ent,
            heat_capacity as _heat_cap,
            helmholtz_free_energy as _helm_fe,
            spectral_entropy_shannon as _spec_ent,
            mean_ipr_low_modes as _mean_ipr,
            multimode_ipr as _mm_ipr,
            entropy_asymmetry_score as _ent_asym,
            hinge_occupation_ratio as _hor,
        )

        _s_vib = _vib_ent(evals) if evals is not None else 0.0
        _c_v = _heat_cap(evals) if evals is not None else 0.0
        _f_helm = _helm_fe(evals) if evals is not None else 0.0
        _s_spec = _spec_ent(evals) if evals is not None else 0.0
        _ipr_low = _mean_ipr(evecs) if evecs is not None else 0.0
        _mm_ipr_val = _mm_ipr(evecs) if evecs is not None else 0.0
        _asym = _ent_asym(per_res_ent)
        _e_gini = _asym["gini"]
        _e_cv = _asym["cv"]
        _e_kurt = _asym["kurtosis"]
        _hor_val = _hor(evecs, domain_labels) if evecs is not None else 1.0
        _n_dom = float(len(np.unique(domain_labels)))
        _max_hinge = float(np.max(hinge)) if len(hinge) > 0 else 0.0

        _thermo = {
            "s_vib": _s_vib, "c_v": _c_v, "f_helm": _f_helm,
            "s_spec": _s_spec, "ipr_low": _ipr_low, "mm_ipr": _mm_ipr_val,
            "e_gini": _e_gini, "e_cv": _e_cv, "e_kurt": _e_kurt,
            "hor": _hor_val, "n_dom": _n_dom, "max_hinge": _max_hinge,
        }

    # ── Strategy 2: Thermodynamic band features ────────────────
    # Append 7 global observables as extra columns broadcast to all nodes.
    # These survive z-normalization because they're added AFTER it.
    if thermo_features:
        g = _thermo
        thermo_cols = np.column_stack([
            np.full(N, g["s_vib"]),    # 15: vibrational entropy
            np.full(N, g["c_v"]),      # 16: heat capacity
            np.full(N, g["s_spec"]),   # 17: spectral entropy
            np.full(N, g["ipr_low"]),  # 18: mean IPR (low modes)
            np.full(N, g["e_gini"]),   # 19: entropy Gini coefficient
            np.full(N, g["e_cv"]),     # 20: entropy CV
            np.full(N, g["hor"]),      # 21: hinge occupation ratio
        ])
        feature_matrix = np.hstack([feature_matrix, thermo_cols])

    # ── Strategy 1: Virtual global node ────────────────────────
    # Append node N with graph-level discriminative features, connected
    # bidirectionally to all residue nodes.  Enables local message-passing
    # to access global topology information.
    if virtual_node:
        g = _thermo
        n_feat = feature_matrix.shape[1]
        vn_feat = np.zeros(n_feat)
        # Base 15 dims: global discriminative quantities
        vn_feat[0] = g["s_vib"]        # vibrational entropy
        vn_feat[1] = g["c_v"]         # heat capacity
        vn_feat[2] = g["f_helm"]      # Helmholtz free energy
        vn_feat[3] = g["s_spec"]      # spectral entropy
        vn_feat[4] = g["ipr_low"]     # mean IPR (low modes)
        vn_feat[5] = g["n_dom"]       # number of domains
        vn_feat[6] = g["max_hinge"]   # peak hinge score
        vn_feat[7] = g["hor"]         # hinge occupation ratio
        vn_feat[8] = g["e_gini"]      # entropy Gini
        vn_feat[9] = g["e_cv"]        # entropy CV
        vn_feat[10] = sg              # spectral gap
        vn_feat[11] = float(np.sum(per_res_ent))  # total entropy
        vn_feat[12] = float(np.mean(deg))  # average degree
        vn_feat[13] = g["mm_ipr"]     # multimode IPR
        vn_feat[14] = g["e_kurt"]     # entropy kurtosis
        # If thermo features enabled, fill dims 15-21 with same values
        if thermo_features:
            vn_feat[15] = g["s_vib"]
            vn_feat[16] = g["c_v"]
            vn_feat[17] = g["s_spec"]
            vn_feat[18] = g["ipr_low"]
            vn_feat[19] = g["e_gini"]
            vn_feat[20] = g["e_cv"]
            vn_feat[21] = g["hor"]

        feature_matrix = np.vstack([feature_matrix, vn_feat.reshape(1, -1)])

        # Add bidirectional edges: virtual node (index N) ↔ all residues
        vn_idx = N
        vn_src = list(range(N)) + [vn_idx] * N
        vn_dst = [vn_idx] * N + list(range(N))
        vn_edge_index = torch.tensor([vn_src, vn_dst], dtype=torch.long)
        edge_index = torch.cat([edge_index, vn_edge_index], dim=1)

        # Virtual edges: sentinel edge_attr [0, 0]
        vn_edge_attr = torch.zeros(2 * N, EDGE_FEATURE_DIM, dtype=torch.float)
        edge_attr = torch.cat([edge_attr, vn_edge_attr], dim=0)

    x = torch.tensor(feature_matrix, dtype=torch.float)
    pos = torch.tensor(coords, dtype=torch.float)
    if virtual_node:
        # Append centroid position for virtual node
        centroid = coords.mean(axis=0)
        vn_pos = torch.tensor(centroid, dtype=torch.float).unsqueeze(0)
        pos = torch.cat([pos, vn_pos], dim=0)

    # ── Label ───────────────────────────────────────────────────
    if archetype is not None:
        y = torch.tensor([ARCHETYPE_TO_IDX.get(archetype, -1)], dtype=torch.long)
    else:
        y = torch.tensor([-1], dtype=torch.long)

    # ── Assemble Data object ────────────────────────────────────
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=pos,
        y=y,
    )
    # Metadata (not used in training, but useful for debugging)
    data.pdb_id = pdb_id
    data.chain = chain
    data.protein_name = name
    data.archetype_name = archetype or "unknown"
    data.n_residues = N
    data.n_contacts = len(contacts)
    data.spectral_gap = float(sg)
    data.has_virtual_node = virtual_node
    data.has_thermo_features = thermo_features

    return data


def _z_normalize(features: np.ndarray) -> np.ndarray:
    """Z-score normalise each column of a feature matrix."""
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    std[std < 1e-10] = 1.0  # avoid division by zero for constant features
    return (features - mean) / std


# ═══════════════════════════════════════════════════════════════════
# Corpus → Dataset
# ═══════════════════════════════════════════════════════════════════

def corpus_to_dataset(
    corpus: Sequence,
    *,
    cache_dir: str = "~/.ibp_enm_cache",
    use_cache: bool = True,
    normalize_features: bool = False,
    include_coords: bool = True,
    virtual_node: bool = False,
    thermo_features: bool = False,
    max_proteins: Optional[int] = None,
    skip_errors: bool = True,
    verbose: bool = False,
) -> List["torch_geometric.data.Data"]:
    """Convert a list of ``ProteinEntry`` objects into PyG Data objects.

    Parameters
    ----------
    corpus : sequence of ProteinEntry
        Each entry must have ``.pdb_id``, ``.chain``, ``.archetype``, ``.name``.
    cache_dir : str
        Profile cache directory.
    use_cache : bool
        Use cached profiles when available.
    normalize_features : bool
        Z-score normalise node features per-protein.
    include_coords : bool
        If True, include absolute Cα coords (features 12-14).
        If False, replace with rotationally-invariant spatial features.
    virtual_node : bool
        If True, add a virtual global node connected to all residues.
    thermo_features : bool
        If True, append 7 thermodynamic band features as extra columns.
    max_proteins : int or None
        Process at most this many proteins (useful for debugging).
    skip_errors : bool
        If True, skip proteins that fail and log a warning.
        If False, raise on the first error.
    verbose : bool
        Print progress.

    Returns
    -------
    list of Data
        PyG graph data objects with labels.
    """
    _check_torch()
    _check_pyg()

    entries = list(corpus)
    if max_proteins is not None:
        entries = entries[:max_proteins]

    dataset: List = []
    errors: List[Tuple[str, str]] = []
    t0 = time.time()

    for i, entry in enumerate(entries):
        pdb_id = entry.pdb_id
        chain = entry.chain
        archetype = entry.archetype
        name = getattr(entry, "name", pdb_id)

        if verbose and (i % 10 == 0 or i == len(entries) - 1):
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 0.01)
            logger.info(
                f"[{i+1}/{len(entries)}] Processing {pdb_id}:{chain} "
                f"({rate:.1f} prot/s)"
            )
            print(
                f"  [{i+1}/{len(entries)}] {pdb_id}:{chain} "
                f"({elapsed:.0f}s elapsed, {rate:.1f}/s)",
                flush=True,
            )

        try:
            data = protein_to_pyg_data(
                pdb_id=pdb_id,
                chain=chain,
                archetype=archetype,
                name=name,
                cache_dir=cache_dir,
                use_cache=use_cache,
                normalize_features=normalize_features,
                include_coords=include_coords,
                virtual_node=virtual_node,
                thermo_features=thermo_features,
            )
            dataset.append(data)
        except Exception as exc:
            msg = f"{pdb_id}:{chain} — {exc}"
            errors.append((pdb_id, str(exc)))
            if skip_errors:
                logger.warning(f"Skipping {msg}")
                if verbose:
                    print(f"  ⚠ SKIP {msg}", flush=True)
            else:
                raise

    if verbose:
        elapsed = time.time() - t0
        print(
            f"\n  Dataset: {len(dataset)} graphs, "
            f"{len(errors)} errors, {elapsed:.1f}s total",
            flush=True,
        )

    return dataset


# ═══════════════════════════════════════════════════════════════════
# Train / Val / Test split
# ═══════════════════════════════════════════════════════════════════

def stratified_split(
    dataset: List["torch_geometric.data.Data"],
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[List, List, List]:
    """Split a dataset into train / val / test with archetype stratification.

    Ensures each archetype is proportionally represented in every split.

    Parameters
    ----------
    dataset : list of Data
        PyG Data objects with ``data.y`` labels.
    train_frac, val_frac, test_frac : float
        Split ratios (must sum to 1.0).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    train, val, test : lists of Data
    """
    _check_torch()
    import torch

    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, (
        f"Fractions must sum to 1.0, got {train_frac + val_frac + test_frac}"
    )

    rng = np.random.RandomState(seed)

    # Group by class
    class_buckets: Dict[int, List] = {}
    for data in dataset:
        label = int(data.y.item())
        class_buckets.setdefault(label, []).append(data)

    train, val, test = [], [], []
    for label in sorted(class_buckets.keys()):
        bucket = class_buckets[label]
        rng.shuffle(bucket)
        n = len(bucket)
        n_train = max(1, int(n * train_frac))
        n_val = max(1, int(n * val_frac))
        # Remaining goes to test
        train.extend(bucket[:n_train])
        val.extend(bucket[n_train:n_train + n_val])
        test.extend(bucket[n_train + n_val:])

    # Shuffle within each split
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return train, val, test


# ═══════════════════════════════════════════════════════════════════
# DataLoader helper
# ═══════════════════════════════════════════════════════════════════

def dataset_to_loader(
    dataset: List["torch_geometric.data.Data"],
    batch_size: int = 16,
    shuffle: bool = True,
    **kwargs,
) -> "torch_geometric.loader.DataLoader":
    """Wrap a list of Data objects in a PyG DataLoader.

    Parameters
    ----------
    dataset : list of Data
        PyG graph data objects.
    batch_size : int
        Batch size.
    shuffle : bool
        Whether to shuffle.

    Returns
    -------
    torch_geometric.loader.DataLoader
    """
    _check_torch()
    _check_pyg()
    from torch_geometric.loader import DataLoader
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)


# ═══════════════════════════════════════════════════════════════════
# Dataset statistics
# ═══════════════════════════════════════════════════════════════════

from dataclasses import dataclass, field


@dataclass
class DatasetStats:
    """Summary statistics for a PyG protein dataset."""
    n_graphs: int = 0
    n_nodes_total: int = 0
    n_edges_total: int = 0
    node_feature_dim: int = 0
    edge_feature_dim: int = 0
    n_classes: int = 0
    class_counts: Dict[str, int] = field(default_factory=dict)
    avg_nodes: float = 0.0
    avg_edges: float = 0.0
    min_nodes: int = 0
    max_nodes: int = 0
    feature_means: Optional[np.ndarray] = None
    feature_stds: Optional[np.ndarray] = None

    def summary(self) -> str:
        """Human-readable summary string."""
        lines = [
            f"PyG Protein Dataset Statistics",
            f"{'=' * 40}",
            f"Graphs:         {self.n_graphs}",
            f"Total nodes:    {self.n_nodes_total}",
            f"Total edges:    {self.n_edges_total}",
            f"Node features:  {self.node_feature_dim}",
            f"Edge features:  {self.edge_feature_dim}",
            f"Classes:        {self.n_classes}",
            f"",
            f"Node counts:    min={self.min_nodes}, "
            f"avg={self.avg_nodes:.1f}, max={self.max_nodes}",
            f"Avg edges/graph: {self.avg_edges:.1f}",
            f"",
            f"Class distribution:",
        ]
        for name, count in sorted(self.class_counts.items()):
            pct = 100 * count / max(self.n_graphs, 1)
            lines.append(f"  {name:20s}: {count:4d} ({pct:5.1f}%)")
        return "\n".join(lines)


def compute_dataset_stats(
    dataset: List["torch_geometric.data.Data"],
) -> DatasetStats:
    """Compute summary statistics for a PyG protein dataset.

    Parameters
    ----------
    dataset : list of Data

    Returns
    -------
    DatasetStats
    """
    _check_torch()
    import torch

    stats = DatasetStats()
    if not dataset:
        return stats

    stats.n_graphs = len(dataset)

    node_counts = []
    edge_counts = []
    all_features = []

    for data in dataset:
        n = data.x.shape[0]
        e = data.edge_index.shape[1]
        node_counts.append(n)
        edge_counts.append(e)
        stats.n_nodes_total += n
        stats.n_edges_total += e

        # Class distribution
        label = int(data.y.item())
        aname = data.archetype_name if hasattr(data, "archetype_name") else str(label)
        stats.class_counts[aname] = stats.class_counts.get(aname, 0) + 1

        # Feature statistics (sample to control memory)
        if data.x.shape[0] <= 500:
            all_features.append(data.x.numpy())

    stats.node_feature_dim = dataset[0].x.shape[1]
    stats.edge_feature_dim = (
        dataset[0].edge_attr.shape[1]
        if dataset[0].edge_attr is not None
        else 0
    )
    stats.n_classes = len(stats.class_counts)
    stats.avg_nodes = float(np.mean(node_counts))
    stats.avg_edges = float(np.mean(edge_counts))
    stats.min_nodes = int(min(node_counts))
    stats.max_nodes = int(max(node_counts))

    if all_features:
        cat = np.concatenate(all_features, axis=0)
        stats.feature_means = cat.mean(axis=0)
        stats.feature_stds = cat.std(axis=0)

    return stats
