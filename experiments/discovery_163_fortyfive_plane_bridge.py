#!/usr/bin/env python
"""D163: 45° Plane Bridge Signals.

D156 found 2016 "half-diagonal" kernel–subalgebra pairs at angle spectrum
{π/4, π/4, π/2, π/2}.  Each pair's two π/4 dimensions define a 2D "45° plane"
in R^16.  After deduplication: **210 unique planes**, each loading 0.5 on
exactly 4 basis elements (2 low-half e₁-e₇, 2 high-half e₈-e₁₅).

The low-half indices (e₁-e₇) map to the 7 instruments via OCTO_TO_CARVING.
This experiment tests whether these 210 planes provide supplementary
correction signals beyond the 168 contained channels that ZDPairSelector
currently uses.

Hypothesis: 45° planes provide a *weaker but broader* Fano correction signal.
- Contained channels: 4 per kernel, 168 total, 72 routes/line — strong, binary
- 45° planes: 24 per kernel, 210 unique — weaker (cos π/4 = 1/√2 damped)
  but cover ALL 35 subalgebras (incl. pure-low/high with 0 contained edges)

Phases:
  1. Enumerate 210 unique 45° planes from sedenion geometry
  2. Map each plane to Fano-line activation via low-half indices
  3. Compute per-archetype "45° plane score" for all 52 proteins
  4. Compare 45° signal vs contained signal (route_score) on LOST vs CORRECT
  5. Test whether 45° signals favour the correct archetype for LOSTs

0 free parameters — all geometry from sedenion algebra.

Prerequisites: D156 (enumeration), D157 (contained channels), D160 (production)
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import Counter, defaultdict

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ibp_enm.benchmark import EXPANDED_CORPUS, ProteinEntry
from ibp_enm.archetypes import ARCHETYPE_EXPECTATIONS
from ibp_enm.synthesis import AlgebraicFickBalancer
from ibp_enm.belief_algebra import (
    HammingBridge, SedenonBridge, ZDPairSelector, FANO_LINES,
    SYNDROME_RETENTION,
)
from ibp_enm.algebra import OCTO_TO_CARVING, CARVING_TO_OCTO, INSTRUMENT_NAMES
from ibp_enm.cache import profiles_from_json
from ibp_enm.lens_stack import build_default_stack

CACHE_DIR = Path.home() / ".ibp_enm_cache"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
ALL_ARCHS = list(ARCHETYPE_EXPECTATIONS.keys())


# ═══════════════════════════════════════════════════════════════════
# SEDENION ALGEBRA (from D156 — uses verified octonion library)
# ═══════════════════════════════════════════════════════════════════

# Import the verified Octonion implementation used by D156
sys.path.insert(0, str(Path("/home/josh/projects/CAExperimentsProject/CAExperiments/src")))
from octonion_ca.octonion import Octonion as _Octonion


class _Sedenion:
    """Sedenion = Cayley-Dickson doubling of Octonion. Matches D156 exactly."""
    __slots__ = ['c']
    def __init__(self, c=None):
        self.c = np.asarray(c, dtype=np.float64) if c is not None else np.zeros(16, dtype=np.float64)

    @property
    def left(self): return _Octonion(self.c[:8].copy())
    @property
    def right(self): return _Octonion(self.c[8:].copy())

    @staticmethod
    def unit(i):
        c = np.zeros(16); c[i] = 1.0; return _Sedenion(c)

    def conjugate(self):
        r = self.c.copy(); r[1:] *= -1; return _Sedenion(r)

    def __add__(self, o): return _Sedenion(self.c + o.c)
    def __sub__(self, o): return _Sedenion(self.c - o.c)

    def __mul__(self, o):
        if isinstance(o, (int, float)): return _Sedenion(self.c * o)
        a, b = self.left, self.right
        c, d = o.left, o.right
        # CD formula matching D156 exactly:
        # (a,b)(c,d) = (ac − db*, a*d + cb)
        return _Sedenion(np.concatenate([
            (a * c - d * b.conjugate()).components,
            (a.conjugate() * d + c * b).components]))

    def __rmul__(self, o):
        if isinstance(o, (int, float)): return _Sedenion(self.c * o)
        return NotImplemented


# ═══════════════════════════════════════════════════════════════════
# SEDENION INFRASTRUCTURE
# ═══════════════════════════════════════════════════════════════════

def _build_mult_tables():
    """Build sedenion multiplication index/sign tables."""
    mult_idx = np.zeros((16, 16), dtype=np.int8)
    mult_sign = np.zeros((16, 16), dtype=np.int8)
    for i in range(16):
        ei = _Sedenion.unit(i)
        for j in range(16):
            ej = _Sedenion.unit(j)
            prod = ei * ej
            idx = int(np.argmax(np.abs(prod.c)))
            mult_idx[i, j] = idx
            mult_sign[i, j] = 1 if prod.c[idx] > 0 else -1
    return mult_idx, mult_sign


def _build_left_mul_matrices(mult_idx, mult_sign):
    """Left multiplication matrices L[k] for each basis element e_k."""
    L = np.zeros((16, 16, 16), dtype=np.float64)
    for k in range(16):
        for j in range(16):
            L[k, mult_idx[k, j], j] = mult_sign[k, j]
    return L


def _zd_census(L_basis):
    """Find all zero-divisor pairs in the sedenions."""
    candidates = []
    signs = []
    for i in range(1, 16):
        for j in range(i + 1, 16):
            v = np.zeros(16); v[i] = 1.0; v[j] = 1.0
            candidates.append((f"e{i}+e{j}", v))
            signs.append((i, j, 1.0, 1.0))
            v = np.zeros(16); v[i] = 1.0; v[j] = -1.0
            candidates.append((f"e{i}-e{j}", v))
            signs.append((i, j, 1.0, -1.0))
    N = len(candidates)
    all_vecs = np.column_stack([v for _, v in candidates])
    zd_nodes = set()
    for idx_a in range(N):
        i, j, si, sj = signs[idx_a]
        L_a = si * L_basis[i] + sj * L_basis[j]
        prods = L_a @ all_vecs[:, idx_a + 1:]
        if prods.shape[1] == 0: continue
        for offset in np.where(np.all(np.abs(prods) < 1e-8, axis=0))[0]:
            idx_b = idx_a + 1 + offset
            zd_nodes.add(candidates[idx_a][0])
            zd_nodes.add(candidates[idx_b][0])
    return candidates, zd_nodes


def _find_quaternionic_subalgebras(mult_idx):
    """Find all quaternionic (3-imaginary) subalgebras of the sedenions."""
    found = set()
    for s1 in range(1, 16):
        for s2 in range(s1 + 1, 16):
            elements = {0, s1, s2}
            changed = True
            while changed:
                changed = False
                for i in list(elements):
                    for j in list(elements):
                        r = int(mult_idx[i, j])
                        if r not in elements:
                            elements.add(r); changed = True
            imag = frozenset(e for e in elements if e >= 1)
            if len(imag) == 3:
                found.add(imag)
    return found


def _kernel_subspace(vec, L_basis):
    """Compute kernel of left multiplication by vec."""
    nz = np.nonzero(vec)[0]
    L_a = sum(vec[k] * L_basis[k] for k in nz)
    U, s, Vt = np.linalg.svd(L_a, full_matrices=True)
    rank = np.sum(s > 1e-10)
    return Vt[rank:].T  # (16, ker_dim)


def _principal_angles(A, B):
    """Principal angles between column subspaces A and B."""
    QA, _ = np.linalg.qr(A, mode='reduced')
    QB, _ = np.linalg.qr(B, mode='reduced')
    M = QA.T @ QB
    cosines = np.linalg.svd(M, compute_uv=False)
    cosines = np.clip(cosines, -1.0, 1.0)
    return np.sort(np.arccos(cosines))


def _extract_45_plane(ker, sub_span):
    """Extract the 2D 45° subspace from a kernel-subalgebra pair.

    Returns the 16×2 basis matrix or None if no π/4 angles found.
    """
    QK, _ = np.linalg.qr(ker, mode='reduced')
    QS, _ = np.linalg.qr(sub_span, mode='reduced')
    M = QK.T @ QS
    UA, cosines, VBt = np.linalg.svd(M, full_matrices=False)
    theta = np.arccos(np.clip(cosines, -1, 1))
    mask = np.abs(theta - np.pi / 4) < 0.1
    if mask.sum() != 2:
        return None
    # UA columns correspond to cosines (reduced SVD ensures matching dims)
    dirs_K = QK @ UA[:, mask]
    return dirs_K


def _span_subalgebra(indices):
    """Build 16×4 span matrix for a quaternionic subalgebra."""
    cols = [0] + sorted(indices)
    M = np.zeros((16, len(cols)))
    for j, idx in enumerate(cols):
        M[idx, j] = 1.0
    return M


# ═══════════════════════════════════════════════════════════════════
# PHASE 1: ENUMERATE 210 UNIQUE 45° PLANES
# ═══════════════════════════════════════════════════════════════════

def enumerate_45_planes():
    """Reproduce D156 Phase 4: find all unique 45° planes.

    Returns:
        planes: list of 16×2 arrays (basis of each unique plane)
        plane_basis_indices: list of (low_pair, high_pair) tuples
            where low_pair = (i, j) are e₁-e₇ indices
            and high_pair = (k, l) are e₈-e₁₅ indices
    """
    print("  Building sedenion multiplication tables...")
    mult_idx, mult_sign = _build_mult_tables()
    L_basis = _build_left_mul_matrices(mult_idx, mult_sign)

    print("  Finding ZD nodes...")
    candidates, zd_nodes = _zd_census(L_basis)
    name_to_vec = {name: vec for name, vec in candidates}
    print(f"  ZD nodes: {len(zd_nodes)}")

    print("  Finding quaternionic subalgebras...")
    q_subs = _find_quaternionic_subalgebras(mult_idx)
    print(f"  Quaternionic subalgebras: {len(q_subs)}")

    # Classify subalgebras
    pure_low = [s for s in q_subs if all(e <= 7 for e in s)]
    pure_high = [s for s in q_subs if all(e >= 8 for e in s)]
    cross_half = [s for s in q_subs if not all(e <= 7 for e in s) and not all(e >= 8 for e in s)]
    print(f"  Subalgebra classes: {len(pure_low)} pure-low, {len(pure_high)} pure-high, {len(cross_half)} cross-half")

    # Build kernel equivalence classes
    zd_sorted = sorted(zd_nodes)
    kernels = {}
    for name in zd_sorted:
        kernels[name] = _kernel_subspace(name_to_vec[name], L_basis)

    visited = set()
    kernel_classes = {}
    for na in zd_sorted:
        if na in visited: continue
        group = [na]; visited.add(na)
        ker_a = kernels[na]
        for nb in zd_sorted:
            if nb in visited: continue
            ker_b = kernels[nb]
            if ker_a.shape[1] == ker_b.shape[1]:
                combined = np.hstack([ker_a, ker_b])
                ov = ker_a.shape[1] + ker_b.shape[1] - np.linalg.matrix_rank(combined, tol=1e-8)
                if ov == ker_a.shape[1]:
                    group.append(nb); visited.add(nb)
        kernel_classes[na] = group

    class_reps = sorted(kernel_classes.keys())
    print(f"  Kernel equivalence classes: {len(class_reps)}")

    # Build subalgebra spans
    q_spans = {tuple(sorted(s)): _span_subalgebra(s) for s in q_subs}

    # Extract 45° planes for all class reps × all subalgebras
    print("  Extracting 45° planes...")
    all_planes = []
    all_plane_meta = []

    for rep in class_reps:
        ker = kernels[rep]
        for sub_key, sub_span in q_spans.items():
            angles = _principal_angles(ker, sub_span)
            # Check if this is a half-diagonal pair
            quant = tuple(round(a / (np.pi / 8)) for a in angles)
            if quant == (2, 2, 4, 4):  # {π/4, π/4, π/2, π/2}
                plane = _extract_45_plane(ker, sub_span)
                if plane is not None and plane.shape[1] == 2:
                    all_planes.append(plane)
                    all_plane_meta.append((rep, sub_key))

    print(f"  Total 45° planes extracted: {len(all_planes)}")

    # Deduplicate via projector Frobenius distance
    unique_planes = []
    unique_meta = []
    plane_labels = []
    multiplicities = Counter()

    for P_basis, meta in zip(all_planes, all_plane_meta):
        P = P_basis @ P_basis.T  # rank-2 projector
        found_match = False
        for idx, (UP_basis, _) in enumerate(zip(unique_planes, unique_meta)):
            UP = UP_basis @ UP_basis.T
            d = np.linalg.norm(P - UP, 'fro')
            if d < 0.01:
                plane_labels.append(idx)
                multiplicities[idx] += 1
                found_match = True
                break
        if not found_match:
            plane_labels.append(len(unique_planes))
            multiplicities[len(unique_planes)] = 1
            unique_planes.append(P_basis)
            unique_meta.append(meta)

    print(f"  Unique 45° planes: {len(unique_planes)}")

    # Analyse basis element loadings for each unique plane
    plane_basis_info = []
    for idx, P_basis in enumerate(unique_planes):
        P = P_basis @ P_basis.T
        diag = np.diag(P)
        # Each plane should load 0.5 on exactly 4 basis elements
        top4 = np.argsort(-diag)[:4]
        low_indices = sorted([i for i in top4 if 1 <= i <= 7])
        high_indices = sorted([i for i in top4 if 8 <= i <= 15])
        plane_basis_info.append({
            "low": low_indices,   # e₁-e₇ (map to instruments)
            "high": high_indices,  # e₈-e₁₅
            "loadings": diag[top4].tolist(),
        })

    return (unique_planes, plane_basis_info, q_subs,
            pure_low, pure_high, cross_half, multiplicities)


# ═══════════════════════════════════════════════════════════════════
# PHASE 2: MAP PLANES TO FANO LINES
# ═══════════════════════════════════════════════════════════════════

def map_planes_to_fano(plane_basis_info):
    """Map each 45° plane's low-half indices to Fano-line activation.

    Each plane loads on 2 low-half indices (e₁-e₇). These are octonion
    points. Each pair of points lies on exactly 0 or 1 Fano lines.
    Via OCTO_TO_CARVING, we can map to instrument indices.

    Returns:
        plane_fano_map: list of dicts with keys:
            'instruments': pair of (0-6) instrument indices
            'fano_lines': list of Fano line indices activated
    """
    fano_line_sets = [set(line) for line in FANO_LINES]
    plane_fano_map = []

    for info in plane_basis_info:
        low = info["low"]
        # Convert e₁-e₇ (1-indexed) to octonion point indices (0-indexed)
        # In the sedenion, e₁-e₇ ARE octonion basis elements (0-indexed as 0-6)
        # But our naming is 1-indexed: e₁→point 0, e₂→point 1, ..., e₇→point 6
        octo_points = [i - 1 for i in low]  # e₁→0, ..., e₇→6

        # Map octonion points to CarvingIntent instrument indices
        instruments = [OCTO_TO_CARVING[p] for p in octo_points]

        # Find which Fano lines contain BOTH points
        activated = []
        for line_idx, ls in enumerate(fano_line_sets):
            if len(set(octo_points) & ls) >= 2:
                activated.append(line_idx)

        plane_fano_map.append({
            "octo_points": octo_points,
            "instruments": instruments,
            "fano_lines": activated,
        })

    return plane_fano_map


# ═══════════════════════════════════════════════════════════════════
# PHASE 3: 45° PLANE SCORE PER ARCHETYPE
# ═══════════════════════════════════════════════════════════════════

def compute_45_scores(
    carver_votes: List[Dict[str, float]],
    all_archs: List[str],
    plane_fano_map: List[Dict],
) -> Dict[str, float]:
    """Per-archetype "45° plane score" from instrument support.

    For each archetype, builds a binary support vector and counts
    how many of the 210 planes have at least one instrument supported.
    Planes with BOTH instruments supported get full weight (1.0);
    planes with exactly one get damped weight (1/√2 = CONTAINED_PURITY).

    This mirrors fano_activation but works through 210 planes instead
    of 7 lines.

    Returns dict mapping archetype → score ∈ [0, 1].
    """
    n_inst = min(len(carver_votes), 7)
    n_planes = len(plane_fano_map)
    scores = {}

    for arch in all_archs:
        # Build binary support for this archetype
        support = set()
        for i in range(n_inst):
            if max(carver_votes[i], key=carver_votes[i].get) == arch:
                support.add(i)

        total = 0.0
        for pmap in plane_fano_map:
            inst = set(pmap["instruments"])
            k = len(inst & support)
            if k >= 2:
                total += 1.0
            elif k == 1:
                total += SYNDROME_RETENTION  # 1/√2

        scores[arch] = total / n_planes if n_planes > 0 else 0.0

    return scores


def compute_45_fano_activation(
    support: np.ndarray,
    plane_fano_map: List[Dict],
) -> np.ndarray:
    """45° plane contribution to each Fano line.

    For each Fano line, sums the plane activations from planes that
    map to that line, damped by cos(π/4) = 1/√2.

    Returns array of shape (7,) — per-line activation from 45° planes.
    """
    active = set(int(i) for i in np.where(support > 0)[0])
    out = np.zeros(7)
    line_count = np.zeros(7)

    for pmap in plane_fano_map:
        inst = set(pmap["instruments"])
        k = len(inst & active)
        if k == 0:
            continue
        weight = 1.0 if k >= 2 else SYNDROME_RETENTION
        for line_idx in pmap["fano_lines"]:
            out[line_idx] += weight
            line_count[line_idx] += 1

    # Normalise per-line (average activation per plane mapping to that line)
    for i in range(7):
        if line_count[i] > 0:
            out[i] /= line_count[i]

    return out


# ═══════════════════════════════════════════════════════════════════
# PROTEIN PIPELINE
# ═══════════════════════════════════════════════════════════════════

def load_cached_profiles(pdb_id, chain):
    path = CACHE_DIR / f"{pdb_id.upper()}_{chain}.json"
    if not path.exists():
        return None, None
    text = path.read_text(encoding="utf-8")
    profiles, metadata = profiles_from_json(text)
    return profiles, metadata


def get_structural_data(pdb_id, chain):
    """Compute evals, evecs, domain_labels, contacts from PDB coords."""
    from ibp_enm.band import _fetch_ca, build_laplacian
    from ibp_enm.analyzer import IBPProteinAnalyzer
    coords, bfactors = _fetch_ca(pdb_id, chain)
    N = len(coords)
    analyzer = IBPProteinAnalyzer()
    result = analyzer.analyze(coords, bfactors)
    contacts, _ = analyzer._build_contacts(coords, N)
    L = build_laplacian(N, contacts)
    evals, evecs = np.linalg.eigh(L)
    domain_labels = result.domain_labels
    return evals, evecs, domain_labels, contacts, N


def run_protein(entry, plane_fano_map):
    """Run full pipeline on a protein, returning contained + 45° signals."""
    profiles, metadata = load_cached_profiles(entry.pdb_id, entry.chain)
    if profiles is None:
        return None

    try:
        evals, evecs, domain_labels, contacts, N = get_structural_data(
            entry.pdb_id, entry.chain)
    except Exception as e:
        print(f" structural error: {e}")
        return None

    carver_votes = [p.archetype_vote() for p in profiles]

    # Run production pipeline (AlgebraicFickBalancer + lens stack)
    balancer = AlgebraicFickBalancer()
    meta_state = balancer.compute_meta_fick_state(carver_votes)
    base_result = balancer.synthesize_identity(profiles, meta_state)
    alpha_0 = meta_state.get("alpha_0", 0.5)
    alpha_8 = meta_state.get("alpha_8", 0.0)

    # Get pre-lens scores from production
    consensus_scores = base_result["consensus_scores"]
    disagreement_scores = base_result["disagreement_scores"]
    context_boost = base_result["context_boost"]
    fano_bridge = base_result.get("fano_bridge", {})
    route_scores = base_result.get("route_scores", {})

    SQRT2 = np.sqrt(2)
    STRONG_WEIGHT = SQRT2 / (SQRT2 + 1)
    WEAK_WEIGHT = 1.0 / (SQRT2 + 1)
    bridge_weight = 0.5 * (1.0 - alpha_0) * alpha_8
    main_weight = 1.0 - bridge_weight

    pre_lens = {}
    for arch in ALL_ARCHS:
        strong = alpha_0 * consensus_scores.get(arch, 0)
        weak = (1 - alpha_0) * disagreement_scores.get(arch, 0)
        main = STRONG_WEIGHT * strong + WEAK_WEIGHT * weak
        rs = route_scores.get(arch, 0)
        bridge = (rs * context_boost.get(arch, 0)
                  + alpha_8 * fano_bridge.get(arch, 0))
        pre_lens[arch] = main_weight * main + bridge_weight * bridge

    total = sum(pre_lens.values())
    if total > 1e-10:
        pre_lens = {k: v / total for k, v in pre_lens.items()}

    # Apply lens stack
    stack = build_default_stack(
        evals=evals, evecs=evecs,
        domain_labels=domain_labels, contacts=contacts,
        pdb_id=entry.pdb_id, chain=entry.chain, n_residues=N)
    ctx = {
        "evals": evals, "evecs": evecs,
        "domain_labels": domain_labels, "contacts": contacts,
        "pdb_id": entry.pdb_id, "chain": entry.chain, "n_residues": N,
    }
    final_scores, traces = stack.apply(pre_lens, profiles, ctx)
    identity = max(final_scores, key=final_scores.get)

    # Standard route scores (contained channels)
    zdp = ZDPairSelector()
    n_inst = min(len(carver_votes), 7)

    per_arch_data = {}
    for arch in ALL_ARCHS:
        support = np.zeros(7, dtype=int)
        for i in range(n_inst):
            if max(carver_votes[i], key=carver_votes[i].get) == arch:
                support[i] = 1

        contained_act = zdp.fano_activation(support)
        route = zdp.route_score(support)
        plane45_act = compute_45_fano_activation(support, plane_fano_map)

        per_arch_data[arch] = {
            "contained_route_score": route,
            "contained_fano_activation": contained_act.tolist(),
            "plane45_fano_activation": plane45_act.tolist(),
            "plane45_mean": float(np.mean(plane45_act)),
            "support_count": int(support.sum()),
        }

    # 45° plane scores
    plane45_scores = compute_45_scores(carver_votes, ALL_ARCHS, plane_fano_map)

    return {
        "name": entry.name,
        "truth": entry.archetype,
        "identity": identity,
        "correct": identity == entry.archetype,
        "alpha_0": alpha_0,
        "alpha_8": alpha_8,
        "per_arch": per_arch_data,
        "plane45_scores": plane45_scores,
    }


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    T0 = time.time()
    PI = np.pi

    print("=" * 72)
    print("DISCOVERY 163: 45° PLANE BRIDGE SIGNALS")
    print("  D156's 2016 half-diagonal pairs → 210 unique 45° planes")
    print("  Test whether they provide correction signals for LOST proteins")
    print("=" * 72)

    # ── PHASE 1: Enumerate 45° planes ────────────────────────────
    print("\n" + "=" * 72)
    print("PHASE 1: ENUMERATE 210 UNIQUE 45° PLANES")
    print("=" * 72)

    t1 = time.time()
    (unique_planes, plane_basis_info,
     q_subs, pure_low, pure_high, cross_half,
     multiplicities) = enumerate_45_planes()
    print(f"  Phase 1 time: {time.time() - t1:.1f}s")

    # Show plane statistics
    n_planes = len(unique_planes)
    print(f"\n  Summary:")
    print(f"    Unique planes: {n_planes}")
    print(f"    Multiplicity range: {min(multiplicities.values())}-{max(multiplicities.values())}")

    # Low-half loading statistics
    low_pair_counts = Counter()
    for info in plane_basis_info:
        low_pair_counts[tuple(info["low"])] += 1
    print(f"    Distinct low-half pairs: {len(low_pair_counts)}")
    print(f"    Top low-half pairs: {low_pair_counts.most_common(5)}")

    # ── PHASE 2: Map planes to Fano lines ────────────────────────
    print("\n" + "=" * 72)
    print("PHASE 2: MAP 45° PLANES TO FANO LINES")
    print("=" * 72)

    plane_fano_map = map_planes_to_fano(plane_basis_info)

    # Count planes per Fano line
    planes_per_line = Counter()
    planes_no_line = 0
    planes_one_line = 0
    planes_multi_line = 0

    for pmap in plane_fano_map:
        n_lines = len(pmap["fano_lines"])
        if n_lines == 0:
            planes_no_line += 1
        elif n_lines == 1:
            planes_one_line += 1
        else:
            planes_multi_line += 1
        for lidx in pmap["fano_lines"]:
            planes_per_line[lidx] += 1

    print(f"\n  Plane-to-Fano mapping:")
    print(f"    Planes with 0 Fano lines: {planes_no_line}")
    print(f"    Planes with 1 Fano line: {planes_one_line}")
    print(f"    Planes with 2+ Fano lines: {planes_multi_line}")
    print(f"\n  Planes per Fano line:")
    for i in range(7):
        line_str = "{" + ",".join(str(x) for x in FANO_LINES[i]) + "}"
        print(f"    Line {i} {line_str}: {planes_per_line.get(i, 0)} planes")

    # Instrument pair coverage
    inst_pairs = Counter()
    for pmap in plane_fano_map:
        inst = tuple(sorted(pmap["instruments"]))
        inst_pairs[inst] += 1
    print(f"\n  Instrument pair coverage ({len(inst_pairs)} distinct pairs):")
    for pair, count in inst_pairs.most_common():
        names = [INSTRUMENT_NAMES[i] for i in pair]
        print(f"    {pair} ({names[0]}, {names[1]}): {count} planes")

    # ── PHASE 3: Run 52-protein benchmark ────────────────────────
    print("\n" + "=" * 72)
    print("PHASE 3: 52-PROTEIN BENCHMARK — CONTAINED vs 45° PLANE SIGNALS")
    print("=" * 72)

    results = []
    correct_names = []
    lost_names = []

    for entry in EXPANDED_CORPUS:
        print(f"  Processing {entry.name}...", end="", flush=True)
        r = run_protein(entry, plane_fano_map)
        if r is None:
            print(" SKIP")
            continue
        results.append(r)
        if r["correct"]:
            correct_names.append(r["name"])
            print(f" CORRECT ({r['identity']})")
        else:
            lost_names.append(r["name"])
            print(f" LOST ({r['identity']} != {r['truth']})")

    n_correct = len(correct_names)
    n_lost = len(lost_names)
    n_total = len(results)
    print(f"\n  Accuracy: {n_correct}/{n_total} ({100*n_correct/n_total:.1f}%)")

    # ── PHASE 4: Compare signals CORRECT vs LOST ─────────────────
    print("\n" + "=" * 72)
    print("PHASE 4: CONTAINED vs 45° SIGNAL — CORRECT vs LOST")
    print("=" * 72)

    # For each protein, compare the signal for the TRUE archetype
    contained_correct_scores = []
    contained_lost_scores = []
    plane45_correct_scores = []
    plane45_lost_scores = []

    # Also: does the 45° signal RANK the true archetype higher?
    contained_truth_rank_correct = []
    contained_truth_rank_lost = []
    plane45_truth_rank_correct = []
    plane45_truth_rank_lost = []

    for r in results:
        truth = r["truth"]
        truth_contained = r["per_arch"][truth]["contained_route_score"]
        truth_plane45 = r["plane45_scores"].get(truth, 0)

        # Rank truth archetype by contained route_score
        contained_ranking = sorted(
            ALL_ARCHS, key=lambda a: r["per_arch"][a]["contained_route_score"],
            reverse=True)
        c_rank = contained_ranking.index(truth) + 1

        # Rank truth archetype by 45° plane score
        plane45_ranking = sorted(
            ALL_ARCHS, key=lambda a: r["plane45_scores"].get(a, 0),
            reverse=True)
        p_rank = plane45_ranking.index(truth) + 1

        if r["correct"]:
            contained_correct_scores.append(truth_contained)
            plane45_correct_scores.append(truth_plane45)
            contained_truth_rank_correct.append(c_rank)
            plane45_truth_rank_correct.append(p_rank)
        else:
            contained_lost_scores.append(truth_contained)
            plane45_lost_scores.append(truth_plane45)
            contained_truth_rank_lost.append(c_rank)
            plane45_truth_rank_lost.append(p_rank)

    print(f"\n  Contained route_score for TRUE archetype:")
    print(f"    CORRECT group (n={n_correct}): mean={np.mean(contained_correct_scores):.4f}, "
          f"median={np.median(contained_correct_scores):.4f}")
    print(f"    LOST group (n={n_lost}): mean={np.mean(contained_lost_scores):.4f}, "
          f"median={np.median(contained_lost_scores):.4f}")
    print(f"    Separation: {np.mean(contained_correct_scores) - np.mean(contained_lost_scores):.4f}")

    print(f"\n  45° plane score for TRUE archetype:")
    print(f"    CORRECT group (n={n_correct}): mean={np.mean(plane45_correct_scores):.4f}, "
          f"median={np.median(plane45_correct_scores):.4f}")
    print(f"    LOST group (n={n_lost}): mean={np.mean(plane45_lost_scores):.4f}, "
          f"median={np.median(plane45_lost_scores):.4f}")
    print(f"    Separation: {np.mean(plane45_correct_scores) - np.mean(plane45_lost_scores):.4f}")

    print(f"\n  Rank of TRUE archetype (lower = better):")
    print(f"    Contained — CORRECT: mean rank={np.mean(contained_truth_rank_correct):.2f}, "
          f"LOST: mean rank={np.mean(contained_truth_rank_lost):.2f}")
    print(f"    45° plane — CORRECT: mean rank={np.mean(plane45_truth_rank_correct):.2f}, "
          f"LOST: mean rank={np.mean(plane45_truth_rank_lost):.2f}")

    # ── PHASE 5: Per-protein 45° signal analysis for LOSTs ───────
    print("\n" + "=" * 72)
    print("PHASE 5: PER-PROTEIN 45° CORRECTION POTENTIAL FOR LOSTs")
    print("=" * 72)

    correction_count = 0
    anti_count = 0

    for r in results:
        if r["correct"]:
            continue

        truth = r["truth"]
        pred = r["identity"]

        # Does the 45° signal favour the truth over the prediction?
        truth_p45 = r["plane45_scores"].get(truth, 0)
        pred_p45 = r["plane45_scores"].get(pred, 0)

        truth_contained = r["per_arch"][truth]["contained_route_score"]
        pred_contained = r["per_arch"][pred]["contained_route_score"]

        # "Correction": 45° ranks truth higher than contained does
        # Measure: (truth_p45 - pred_p45) vs (truth_contained - pred_contained)
        contained_margin = truth_contained - pred_contained
        plane45_margin = truth_p45 - pred_p45

        favours_truth = plane45_margin > contained_margin
        if favours_truth:
            correction_count += 1
        elif plane45_margin < contained_margin:
            anti_count += 1

        print(f"\n  {r['name']}: truth={truth}, pred={pred}")
        print(f"    Contained: truth={truth_contained:.4f}, pred={pred_contained:.4f}, "
              f"margin={contained_margin:+.4f}")
        print(f"    45° plane: truth={truth_p45:.4f}, pred={pred_p45:.4f}, "
              f"margin={plane45_margin:+.4f}")
        print(f"    45° {'FAVOURS TRUTH' if favours_truth else 'FAVOURS PRED'} "
              f"(relative to contained)")

        # Per-Fano-line comparison
        truth_c_act = r["per_arch"][truth]["contained_fano_activation"]
        truth_p_act = r["per_arch"][truth]["plane45_fano_activation"]
        pred_c_act = r["per_arch"][pred]["contained_fano_activation"]
        pred_p_act = r["per_arch"][pred]["plane45_fano_activation"]

        lines_with_45_truth_gain = []
        for li in range(7):
            c_diff = truth_c_act[li] - pred_c_act[li]
            p_diff = truth_p_act[li] - pred_p_act[li]
            if p_diff > c_diff + 0.01:
                lines_with_45_truth_gain.append(li)
        if lines_with_45_truth_gain:
            print(f"    Fano lines where 45° favours truth more: {lines_with_45_truth_gain}")

    print(f"\n  Summary of 45° correction potential:")
    print(f"    45° favours truth (vs contained): {correction_count}/{n_lost}")
    print(f"    45° favours pred (vs contained): {anti_count}/{n_lost}")
    print(f"    Neutral: {n_lost - correction_count - anti_count}/{n_lost}")

    # ── PHASE 6: Geometry summary ────────────────────────────────
    print("\n" + "=" * 72)
    print("PHASE 6: GEOMETRIC SUMMARY")
    print("=" * 72)

    # Compare coverage: contained vs 45° planes
    print(f"\n  Routing geometry comparison:")
    print(f"    Contained channels:  168 edges, 4/kernel, 72/line, 21 cross-half subs only")
    print(f"    45° planes:          {n_planes} planes, 24/kernel, all 35 subs")
    print(f"    Contained purity:    1/√2 ≈ {SYNDROME_RETENTION:.4f}")
    print(f"    45° strength:        cos(π/4) = 1/√2 ≈ {np.cos(PI/4):.4f} (same!)")

    # Check if 45° planes cover pure-low/high subs (which have 0 contained edges)
    planes_from_pure_low = sum(
        1 for info in plane_basis_info
        if all(1 <= i <= 7 for i in info["low"]) and all(8 <= i <= 15 for i in info["high"])
    )
    print(f"\n  All {n_planes} planes involve 2 low + 2 high basis elements (CD split)")
    print(f"  Pure-low subs (0 contained edges) → 45° planes provide the ONLY")
    print(f"  signal pathway from these subalgebras to the instrument space")

    # ── RESULTS ──────────────────────────────────────────────────
    elapsed = time.time() - T0
    print(f"\n{'=' * 72}")
    print(f"EXPERIMENT COMPLETE — {elapsed:.1f}s")
    print(f"{'=' * 72}")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "unique_45_planes": n_planes,
        "total_extracted": len(plane_basis_info),
        "planes_per_fano_line": {str(k): v for k, v in sorted(planes_per_line.items())},
        "planes_no_fano_line": planes_no_line,
        "planes_one_fano_line": planes_one_line,
        "planes_multi_fano_line": planes_multi_line,
        "instrument_pair_coverage": {
            str(pair): count for pair, count in inst_pairs.most_common()
        },
        "accuracy": n_correct,
        "total": n_total,
        "contained_signal": {
            "correct_mean": float(np.mean(contained_correct_scores)),
            "lost_mean": float(np.mean(contained_lost_scores)),
            "separation": float(np.mean(contained_correct_scores) - np.mean(contained_lost_scores)),
        },
        "plane45_signal": {
            "correct_mean": float(np.mean(plane45_correct_scores)),
            "lost_mean": float(np.mean(plane45_lost_scores)),
            "separation": float(np.mean(plane45_correct_scores) - np.mean(plane45_lost_scores)),
        },
        "truth_rank": {
            "contained_correct": float(np.mean(contained_truth_rank_correct)),
            "contained_lost": float(np.mean(contained_truth_rank_lost)),
            "plane45_correct": float(np.mean(plane45_truth_rank_correct)),
            "plane45_lost": float(np.mean(plane45_truth_rank_lost)),
        },
        "correction_potential": {
            "favours_truth": correction_count,
            "favours_pred": anti_count,
            "neutral": n_lost - correction_count - anti_count,
            "total_lost": n_lost,
        },
        "per_protein": [
            {
                "name": r["name"],
                "truth": r["truth"],
                "pred": r["identity"],
                "correct": r["correct"],
                "alpha_0": r["alpha_0"],
                "alpha_8": r["alpha_8"],
                "contained_truth_route": r["per_arch"][r["truth"]]["contained_route_score"],
                "plane45_truth_score": r["plane45_scores"].get(r["truth"], 0),
                "contained_pred_route": r["per_arch"][r["identity"]]["contained_route_score"],
                "plane45_pred_score": r["plane45_scores"].get(r["identity"], 0),
            }
            for r in results
        ],
    }

    out_path = RESULTS_DIR / "d163_fortyfive_plane_bridge.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
