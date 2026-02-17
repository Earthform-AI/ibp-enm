"""The Thermodynamic Band — 7 instruments, full physics.

This is the top-level orchestrator.  :class:`ThermodynamicBand` creates
7 independent carvers (each with its own contact-graph copy), lets
them each play a fixed number of beats, then fuses their votes via
:class:`~ibp_enm.synthesis.MetaFickBalancer` to produce a final
archetype identity.

Quick start
-----------
>>> from ibp_enm import ThermodynamicBand
>>> from ibp_enm.band import run_single_protein
>>>
>>> result = run_single_protein("2LZM", "A")
>>> print(result["band_identity"])
enzyme_active

Architecture
------------
::

    ThermodynamicBand
    ├─ 7 × ThermoInstrumentCarver  (independent contact-graph copies)
    │      algebraic   — max |Δgap|
    │      musical     — max mode_scatter
    │      fick        — FickBalancer
    │      thermal     — max |Δτ|        ← NEW in D109
    │      cooperative — max |Δβ|        ← NEW in D109
    │      propagative — max spatial_r   ← NEW in D109
    │      fragile     — max bus_mass    ← NEW in D109
    │
    ├─ MetaFickBalancer  (consensus / disagreement / context-boost fusion)
    │
    └─ SurgeonsHandbook  (initial snapshot diagnosis, NLP sentence)

Historical accuracy (12-protein benchmark):
    D106:  58% (7/12), barrel 0/2
    D107:  42% (5/12), barrel 2/2, +3 false reclassifications
    D108:  17% (2/12), barrel 2/2, +1 false reclassification
    D109:  83% (10/12), barrel 2/2, 0 false reclassifications  ★
"""

from __future__ import annotations

import time
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

from .analyzer import IBPProteinAnalyzer
from .archetypes import (
    ARCHETYPES,
    ARCHETYPE_EXPECTATIONS,
    GROUND_TRUTH,
    PROTEINS,
    SurgeonsHandbook,
)
from .carving import (
    CarvingIntent,
    FickBalancer,
    ShadowInspector,
    build_laplacian,
    spectral_gap,
)
from .instruments import (
    INSTRUMENTS,
    STEPS_PER_INSTRUMENT,
    ThermoInstrumentCarver,
    steps_for_protein,
)
from .synthesis import MetaFickBalancer, HingeLensSynthesis, SizeAwareHingeLens
from .lens_stack import LensStackSynthesizer, build_default_stack
from .cache import ProfileCache, profiles_to_json, profiles_from_json
from .thermodynamics import (
    heat_capacity,
    helmholtz_free_energy,
    mean_ipr_low_modes,
    vibrational_entropy,
)

__all__ = [
    "ThermodynamicBand",
    "run_single_protein",
]


# ═══════════════════════════════════════════════════════════════════
# ThermodynamicBand
# ═══════════════════════════════════════════════════════════════════

class ThermodynamicBand:
    """Seven-instrument carving band with full thermodynamic recording.

    Each instrument operates on its *own* contact-graph copy so
    perturbations do not interfere.  After all instruments play,
    :class:`MetaFickBalancer` fuses their archetype votes.

    Parameters
    ----------
    N : int
        Number of residues.
    contacts : dict
        ``{(i, j): distance}`` contact map.
    coords : ndarray (N, 3)
        Cα coordinates.
    bfactors : ndarray (N,)
        Crystallographic B-factors.
    fiedler : ndarray (N,)
        Fiedler vector from the Laplacian.
    domain_labels : ndarray (N,)
        Domain assignment per residue.
    gap_base : float
        Spectral gap of the intact protein.
    evals, evecs : ndarray
        Laplacian eigenvalues and eigenvectors.
    handbook : SurgeonsHandbook
        Diagnosis helper (optional — one is created if ``None``).
    predicted_bfactors : ndarray or None
        Predicted B-factors from the analyser.
    """

    def __init__(
        self,
        N: int,
        contacts: dict,
        coords: np.ndarray,
        bfactors: np.ndarray,
        fiedler: np.ndarray,
        domain_labels: np.ndarray,
        gap_base: float,
        evals: np.ndarray,
        evecs: np.ndarray,
        handbook: Optional[SurgeonsHandbook] = None,
        predicted_bfactors: Optional[np.ndarray] = None,
        thresholds=None,
    ):
        self.N = N
        self.contacts = contacts
        self.coords = coords
        self.bfactors = bfactors
        self.fiedler = fiedler
        self.domain_labels = domain_labels
        self.gap_base = gap_base
        self.evals = evals
        self.evecs = evecs
        self.handbook = handbook or SurgeonsHandbook()
        self.predicted_bfactors = predicted_bfactors

        # One carver per instrument
        self.carvers: Dict[str, ThermoInstrumentCarver] = {}
        for inst_name, intent_idx in INSTRUMENTS:
            self.carvers[inst_name] = ThermoInstrumentCarver(
                instrument=inst_name,
                intent_idx=intent_idx,
                N=N,
                contacts=contacts,
                coords=coords,
                bfactors=bfactors,
                fiedler=fiedler,
                domain_labels=domain_labels,
                gap_base=gap_base,
                evals=evals,
                evecs=evecs,
                predicted_bfactors=predicted_bfactors,
            )

        self.meta_fick = LensStackSynthesizer(
            evals=evals, evecs=evecs,
            domain_labels=domain_labels, contacts=contacts,
            thresholds=thresholds,
        )
        self.initial_diagnosis: Optional[Dict] = None

        # D113: dynamic carving steps based on protein size
        self._steps = steps_for_protein(N, len(contacts))

    # ── Phase 1: snapshot diagnosis ─────────────────────────────

    def diagnose_initial(self, max_probes: int = 80) -> Dict:
        """D106-style initial diagnosis for baseline comparison.

        Shadow-probes a sample of edges, aggregates bus vectors, and
        matches against archetypes via :class:`SurgeonsHandbook`.
        """
        inspector = ShadowInspector(
            self.N, self.contacts, self.coords, self.bfactors,
            self.fiedler, self.domain_labels)
        edges = list(self.contacts.keys())
        reactions = inspector.shadow_probe_batch(
            edges, max_probes=max_probes)

        balancer = FickBalancer()
        balancer.tau_prev = 1.0 / (self.evals[1] + 1e-10)
        fick_state = balancer.compute_fick_state(
            self.evals, self.evecs, self.N,
            self.contacts, self.fiedler)
        alpha = balancer.compute_alpha(fick_state)

        self.initial_diagnosis = self.handbook.diagnose(reactions, alpha)
        return self.initial_diagnosis

    # ── Phase 2: the band plays ─────────────────────────────────

    def play(self) -> Dict:
        """All 7 instruments play their beats.

        Returns a dict with keys:

        * ``identity`` — the synthesised archetype identity
        * ``meta_fick`` — meta-Fick consensus state
        * ``per_instrument`` — per-instrument summary dicts
        * ``band_log`` — raw step-by-step logs
        * ``fick_alpha_trajectory`` — Fick α over time
        """
        band_log: Dict[str, list] = {}

        for inst_name, _ in INSTRUMENTS:
            carver = self.carvers[inst_name]
            inst_log = []
            for step in range(1, self._steps + 1):
                result = carver.play_step(step)
                inst_log.append(result)
            band_log[inst_name] = inst_log

        # Meta-Fick synthesis
        profiles = [self.carvers[name].profile
                    for name, _ in INSTRUMENTS]
        final_votes = [p.archetype_vote() for p in profiles]
        meta_state = self.meta_fick.compute_meta_fick_state(final_votes)
        identity_result = self.meta_fick.synthesize_identity(
            profiles, meta_state)

        # Per-instrument summaries
        per_instrument: Dict[str, Dict] = {}
        for inst_name, _ in INSTRUMENTS:
            p = self.carvers[inst_name].profile
            per_instrument[inst_name] = {
                # Gap
                "gap_retained": p.gap_retained,
                "gap_flatness": p.gap_flatness,
                "gap_volatility": p.gap_volatility,
                "gap_trend": p.gap_trend,
                # Spectral
                "species_entropy": p.species_entropy,
                "reversible_frac": p.reversible_frac,
                "mean_scatter": p.mean_scatter,
                # Thermodynamic
                "entropy_change": p.entropy_change,
                "mean_delta_S": p.mean_delta_entropy,
                "entropy_volatility": p.entropy_volatility,
                "heat_cap_change": p.heat_cap_change,
                "free_energy_cost": p.free_energy_cost,
                "mean_ipr": p.mean_ipr,
                # Previously-ignored channels
                "mean_spatial_radius": p.mean_spatial_radius,
                "max_spatial_radius": p.max_spatial_radius,
                "mean_delta_beta": p.mean_delta_beta,
                "mean_bus_mass": p.mean_bus_mass,
                # Meta
                "intent_switches": p.intent_switches,
                "cuts_made": p.cuts_made,
                "species": dict(Counter(p.species_removed)),
                "vote": p.archetype_vote(),
            }

        fick_alphas = self.carvers["fick"].alpha_trajectory

        # Store profiles for caching / re-scoring
        self._last_profiles = profiles

        return {
            "identity": identity_result,
            "meta_fick": meta_state,
            "per_instrument": per_instrument,
            "band_log": band_log,
            "fick_alpha_trajectory": fick_alphas,
            "profiles": profiles,
        }

    # ── Profile caching ─────────────────────────────────────────

    def get_profiles(self) -> list:
        """Return the 7 ThermoReactionProfile objects from the last play()."""
        return getattr(self, "_last_profiles", [
            self.carvers[name].profile for name, _ in INSTRUMENTS
        ])

    def save_profiles(self, path: str, metadata: dict | None = None) -> None:
        """Save all 7 carving profiles to a JSON file.

        Use :meth:`rescore_from_profiles` or :func:`profiles_from_json`
        to reload and re-score without re-carving.
        """
        from pathlib import Path as P
        profiles = self.get_profiles()
        text = profiles_to_json(profiles, metadata)
        P(path).write_text(text, encoding="utf-8")

    @classmethod
    def rescore_from_profiles(
        cls,
        profiles: list,
        evals=None,
        evecs=None,
        domain_labels=None,
        contacts=None,
        thresholds=None,
    ) -> dict:
        """Re-score pre-computed profiles without re-carving.

        This is the fast path (~0.01s vs ~120s for full carving).
        Pass the eigenvalues/eigenvectors/domain_labels/contacts
        to enable the HingeLens and SizeAwareHingeLens.  Without
        them, only the base MetaFickBalancer and EnzymeLens fire.

        Parameters
        ----------
        profiles : list[ThermoReactionProfile]
            The 7 profiles from a previous :meth:`play` call.
        evals, evecs : ndarray, optional
            Laplacian eigenvalues/eigenvectors (enables hinge lens).
        domain_labels : ndarray, optional
            Domain assignment per residue.
        contacts : dict, optional
            Contact map.
        thresholds : ThresholdRegistry, optional
            Custom thresholds for the lens stack.  If ``None``,
            uses ``DEFAULT_THRESHOLDS``.

        Returns
        -------
        dict
            Same structure as :meth:`play`'s ``identity`` sub-dict.
        """
        synth = LensStackSynthesizer(
            evals=evals, evecs=evecs,
            domain_labels=domain_labels, contacts=contacts,
            thresholds=thresholds,
        )
        final_votes = [p.archetype_vote() for p in profiles]
        meta_state = synth.compute_meta_fick_state(final_votes)
        identity_result = synth.synthesize_identity(profiles, meta_state)
        return identity_result


# ═══════════════════════════════════════════════════════════════════
# Convenience: run a single protein end-to-end
# ═══════════════════════════════════════════════════════════════════

def _fetch_ca(pdb_id: str, chain: str = "A"):
    """Fetch Cα coordinates from RCSB mmCIF (same as D106 helper)."""
    import requests
    url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    coords, bfactors = [], []
    seen: set = set()
    for line in resp.text.split("\n"):
        if not line.startswith("ATOM"):
            continue
        parts = line.split()
        if len(parts) < 15:
            continue
        atom_name, chain_id, res_seq = parts[3], parts[6], parts[8]
        alt_id = parts[4] if len(parts) > 4 else "."
        if atom_name != "CA" or chain_id != chain:
            continue
        if alt_id not in (".", "?", "A", ""):
            continue
        key = (chain_id, res_seq)
        if key in seen:
            continue
        seen.add(key)
        try:
            x, y, z = float(parts[10]), float(parts[11]), float(parts[12])
            b = float(parts[14])
            coords.append([x, y, z])
            bfactors.append(b)
        except (ValueError, IndexError):
            continue
    return np.array(coords), np.array(bfactors)


def run_single_protein(
    pdb_id: str,
    chain: str = "A",
    name: Optional[str] = None,
    verbose: bool = False,
    thresholds=None,
) -> Dict:
    """Run the thermodynamic band on a single protein.

    Parameters
    ----------
    pdb_id : str
        PDB accession code (e.g. ``"2LZM"``).
    chain : str
        Chain identifier.
    name : str, optional
        Human-readable name.  Defaults to ``pdb_id``.
    verbose : bool
        If ``True``, print progress to stdout.

    Returns
    -------
    dict
        Keys include ``"band_identity"``, ``"initial_diagnosis"``,
        ``"band_result"``, ``"true_archetype"`` (if in the benchmark
        corpus), etc.
    """
    name = name or pdb_id
    log = print if verbose else (lambda *a, **k: None)
    t0 = time.perf_counter()

    log(f"Fetching {pdb_id}:{chain} …")
    coords, bfactors = _fetch_ca(pdb_id, chain)
    N = len(coords)
    if N < 20:
        raise ValueError(f"Too few residues ({N}) for ENM analysis")

    analyzer = IBPProteinAnalyzer()
    result = analyzer.analyze(coords, bfactors)
    contacts, _ = analyzer._build_contacts(coords, N)
    fiedler = result.fiedler_vector
    domain_labels = result.domain_labels

    sg = result.spectral_gap
    if sg is None:
        evals_ = result.laplacian_eigenvalues
        pos = evals_[evals_ > 1e-10]
        sg = float(pos[0] / pos[1]) if len(pos) >= 2 else 0.0

    L = build_laplacian(N, contacts)
    evals, evecs = np.linalg.eigh(L)
    predicted_bfactors = getattr(result, "b_laplacian", None)

    log(f"  N={N}, domains={result.n_domains}, "
        f"gap={sg:.4f}, contacts={len(contacts)}")

    # Initial thermo
    S0 = vibrational_entropy(evals)
    Cv0 = heat_capacity(evals)
    F0 = helmholtz_free_energy(evals)
    ipr0 = mean_ipr_low_modes(evecs)
    log(f"  Thermo: S={S0:.2f} Cv={Cv0:.2f} F={F0:.2f} IPR={ipr0:.4f}")

    handbook = SurgeonsHandbook(ARCHETYPES)

    band = ThermodynamicBand(
        N, contacts, coords, bfactors, fiedler,
        domain_labels, sg, evals, evecs,
        handbook=handbook,
        predicted_bfactors=predicted_bfactors,
        thresholds=thresholds,
    )

    initial_diag = band.diagnose_initial(max_probes=80)
    log(f"  Snapshot: {initial_diag['archetype']} "
        f"(score={initial_diag['archetype_score']:.3f})")

    band_result = band.play()
    band_identity = band_result["identity"]["identity"]
    hinge_identity = band_identity  # HingeLens is now the default
    enzyme_lens_activated = band_result["identity"].get(
        "enzyme_lens_activated", False)
    hinge_lens_activated = band_result["identity"].get(
        "hinge_lens_activated", False)
    log(f"  Band:     {band_identity}"
        f"  (enzyme_lens={'on' if enzyme_lens_activated else 'off'}"
        f", hinge_lens={'on' if hinge_lens_activated else 'off'})")

    # Ground truth (if available)
    true_arch = GROUND_TRUTH.get(name)
    initial_correct = (initial_diag["archetype"] == true_arch
                       if true_arch else None)
    band_correct = (band_identity == true_arch
                    if true_arch else None)

    # True rank
    sorted_scores = sorted(
        band_result["identity"]["scores"].items(),
        key=lambda x: -x[1])
    true_rank = (
        next((i + 1 for i, (a, _) in enumerate(sorted_scores)
              if a == true_arch), 5)
        if true_arch else None)

    dt = time.perf_counter() - t0
    log(f"  Time: {dt:.1f}s")

    return {
        "name": name,
        "pdb_id": pdb_id,
        "N": N,
        "spectral_gap": sg,
        "n_contacts": len(contacts),
        "initial_S_vib": S0,
        "initial_Cv": Cv0,
        "initial_F": F0,
        "initial_ipr": ipr0,
        "initial_diagnosis": initial_diag,
        "band_result": band_result,
        "band_identity": band_identity,
        "hinge_identity": hinge_identity,
        "enzyme_lens_activated": enzyme_lens_activated,
        "hinge_lens_activated": hinge_lens_activated,
        "true_archetype": true_arch,
        "initial_correct": initial_correct,
        "band_correct": band_correct,
        "true_rank": true_rank,
        "time_s": round(dt, 1),
        # ENM context for fast rescoring (D116)
        "evals": evals,
        "evecs": evecs,
        "domain_labels": domain_labels,
        "contacts": contacts,
    }
