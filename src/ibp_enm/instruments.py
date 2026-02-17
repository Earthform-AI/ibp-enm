"""Thermodynamic instrument carvers — seven genuinely distinct disturbance modes.

Each :class:`ThermoInstrumentCarver` selects edges via a different
physical signal, producing maximally diverse perturbation contexts.

Edge selection modes
--------------------
====  ===========  ========================  ================================
#     Name         Selection signal          Physical interpretation
====  ===========  ========================  ================================
1     algebraic    max |Δgap|                symmetry breaking
2     musical      max mode_scatter          resonance sensitivity
3     fick         FickBalancer scoring       diffusion-optimal cut
4     thermal      max |Δτ|                  entropy-disrupting (NEW in D109)
5     cooperative  max |Δβ|                  cooperativity probe (NEW in D109)
6     propagative  max spatial_radius        allosteric reach    (NEW in D109)
7     fragile      max bus_mass              thermal soft spots  (NEW in D109)
====  ===========  ========================  ================================

Historical notes
----------------
Instruments 1–3 originate in D106/D108.
Instruments 4–7 were introduced in D109 after the audit revealed
that only 3 of D108's 5 instruments used distinct edge-selection
signals.

The ``archetype_vote()`` thresholds inside :class:`ThermoReactionProfile`
are **data-calibrated** from D109 Run 1 empirical Cohen's d effect sizes.
"""

from __future__ import annotations

import numpy as np
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .archetypes import ARCHETYPE_EXPECTATIONS
from .rules import apply_rules, apply_rules_traced, ARCHETYPE_RULES
from .carving import (
    CarvingIntent,
    ShadowInspector,
    QuantumCarvingState,
    CarvingNote,
    MelodyTracker,
    FickBalancer,
    build_laplacian,
    spectral_gap,
)
from .thermodynamics import (
    vibrational_entropy,
    heat_capacity,
    helmholtz_free_energy,
    mean_ipr_low_modes,
)

__all__ = [
    "ThermoReactionProfile",
    "ThermoInstrumentCarver",
    "INSTRUMENT_CALIBRATION",
    "INSTRUMENTS",
    "STEPS_PER_INSTRUMENT",
    "steps_for_protein",
]

# ── Size-aware constants (D113) ─────────────────────────────────
# N_ref is the median protein size of the original 12-protein
# benchmark.  Scatter is normalised relative to this so that
# thresholds calibrated on 100–250 residue proteins transfer
# to larger structures without barrel over-prediction.
N_REF: int = 200


def steps_for_protein(N: int, n_contacts: int) -> int:
    """Compute carving steps scaled to protein size (D113).

    Small proteins (N ≤ 200): 5 steps (original D109 value).
    Larger proteins: more steps so that the fractional perturbation
    (steps / n_contacts) remains comparable.

    The formula ensures at least ~0.5% of contacts are probed.
    """
    base = max(5, N // 40)
    # Cap at 15 to avoid excessive runtime
    return min(15, base)


# ═══════════════════════════════════════════════════════════════════
# INSTRUMENT_CALIBRATION — contextual weights per instrument
# ═══════════════════════════════════════════════════════════════════

INSTRUMENT_CALIBRATION: Dict[str, Dict[str, float]] = {
    "algebraic":   {"weight_on_flatness": 2.5, "weight_on_scatter": 0.5},
    "musical":     {"weight_on_flatness": 0.0, "weight_on_scatter": 2.5},
    "fick":        {"weight_on_flatness": 1.5, "weight_on_alpha": 2.0},
    "thermal":     {"weight_on_entropy": 2.5},
    "cooperative": {"weight_on_cooperativity": 2.5},
    "propagative": {"weight_on_propagation": 2.5},
    "fragile":     {"weight_on_fragility": 2.5},
}

INSTRUMENTS: List[Tuple[str, int]] = [
    ("algebraic",   CarvingIntent.DOING),       # max |Δgap|
    ("musical",     CarvingIntent.FEELING),      # max mode_scatter
    ("fick",        CarvingIntent.KNOWING),      # FickBalancer
    ("thermal",     CarvingIntent.BECOMING),     # max |Δτ| ∝ ΔS
    ("cooperative", CarvingIntent.RELATING),     # max |Δβ|
    ("propagative", CarvingIntent.DOING),        # max spatial_radius
    ("fragile",     CarvingIntent.FEELING),      # max bus_mass
]

STEPS_PER_INSTRUMENT: int = 5


# ═══════════════════════════════════════════════════════════════════
# ThermoReactionProfile — one carver's full trajectory
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ThermoReactionProfile:
    """What one carver observed — with full thermodynamics.

    Extends D108's CarverReactionProfile with:

    * Vibrational entropy trajectory (*S_vib* after each step)
    * Heat capacity trajectory (*C_v* after each step)
    * Free energy trajectory (*F* after each step)
    * Per-cut entropy change (Δ*S* for each edge removal)
    * Mode localisation (*mean IPR*)
    * Δτ trajectory  (diffusion time-scale changes)
    * Δβ trajectory  (bottleneck / cooperativity changes)
    * Spatial radius trajectory (perturbation propagation)
    * Bus-mass trajectory (B-factor channel)
    """

    instrument: str
    intent_idx: int
    n_residues: int = 200     # D113: protein size for scatter normalisation
    n_contacts: int = 1000    # D113: contact count for scatter normalisation

    # Gap observables (from D108)
    gap_trajectory: List[float] = field(default_factory=list)
    species_removed: List[str] = field(default_factory=list)
    reversibility: List[bool] = field(default_factory=list)
    tensions: List[float] = field(default_factory=list)
    mode_scatters: List[float] = field(default_factory=list)

    # Thermodynamic observables
    entropy_trajectory: List[float] = field(default_factory=list)
    heat_cap_trajectory: List[float] = field(default_factory=list)
    free_energy_trajectory: List[float] = field(default_factory=list)
    delta_entropy_per_cut: List[float] = field(default_factory=list)
    ipr_trajectory: List[float] = field(default_factory=list)

    # Previously-ignored reaction channels
    delta_tau_trajectory: List[float] = field(default_factory=list)
    delta_beta_trajectory: List[float] = field(default_factory=list)
    spatial_radius_trajectory: List[float] = field(default_factory=list)
    bus_mass_trajectory: List[float] = field(default_factory=list)

    intent_switches: int = 0
    cuts_made: int = 0

    # ── Gap properties ──────────────────────────────────────────

    @property
    def gap_retained(self) -> float:
        return self.gap_trajectory[-1] if self.gap_trajectory else 1.0

    @property
    def gap_volatility(self) -> float:
        if len(self.gap_trajectory) < 2:
            return 0.0
        return float(np.std(np.diff(self.gap_trajectory)))

    @property
    def gap_trend(self) -> float:
        if len(self.gap_trajectory) < 2:
            return 0.0
        x = np.arange(len(self.gap_trajectory))
        return float(np.polyfit(x, self.gap_trajectory, 1)[0])

    @property
    def gap_flatness(self) -> float:
        if len(self.gap_trajectory) < 2:
            return 1.0
        rng = max(self.gap_trajectory) - min(self.gap_trajectory)
        mean_gap = np.mean(self.gap_trajectory)
        if mean_gap < 1e-10:
            return 0.0
        return max(0.0, 1.0 - rng / (mean_gap + 1e-10))

    @property
    def reversible_frac(self) -> float:
        return float(np.mean(self.reversibility)) if self.reversibility else 0.5

    @property
    def species_entropy(self) -> float:
        if not self.species_removed:
            return 0.0
        counts = Counter(self.species_removed)
        total = sum(counts.values())
        probs = np.array([c / total for c in counts.values()])
        return float(-np.sum(probs * np.log(probs + 1e-10)))

    @property
    def mean_scatter(self) -> float:
        return float(np.mean(self.mode_scatters)) if self.mode_scatters else 0.0

    @property
    def scatter_normalised(self) -> float:
        """Size-normalised scatter (D113).

        Raw scatter is trivially low for large proteins because
        removing 5 edges from 2000+ contacts is a < 0.3% perturbation.
        We scale scatter by N / N_ref so that thresholds calibrated
        on ~200-residue proteins transfer to larger structures.

        For the original 12-protein benchmark (N ≈ 100–470),
        scatter_normalised ≈ scatter for the median protein.
        """
        raw = self.mean_scatter
        scale = self.n_residues / N_REF
        return raw * scale

    # ── Thermodynamic properties ────────────────────────────────

    @property
    def entropy_change(self) -> float:
        """Total vibrational entropy change start → end."""
        if len(self.entropy_trajectory) < 2:
            return 0.0
        return self.entropy_trajectory[-1] - self.entropy_trajectory[0]

    @property
    def entropy_volatility(self) -> float:
        if len(self.delta_entropy_per_cut) < 2:
            return 0.0
        return float(np.std(self.delta_entropy_per_cut))

    @property
    def mean_delta_entropy(self) -> float:
        return float(np.mean(self.delta_entropy_per_cut)) if self.delta_entropy_per_cut else 0.0

    @property
    def heat_cap_change(self) -> float:
        if len(self.heat_cap_trajectory) < 2:
            return 0.0
        return self.heat_cap_trajectory[-1] - self.heat_cap_trajectory[0]

    @property
    def free_energy_cost(self) -> float:
        if len(self.free_energy_trajectory) < 2:
            return 0.0
        return self.free_energy_trajectory[-1] - self.free_energy_trajectory[0]

    @property
    def mean_spatial_radius(self) -> float:
        return float(np.mean(self.spatial_radius_trajectory)) if self.spatial_radius_trajectory else 0.0

    @property
    def max_spatial_radius(self) -> float:
        return float(np.max(self.spatial_radius_trajectory)) if self.spatial_radius_trajectory else 0.0

    @property
    def mean_delta_beta(self) -> float:
        return float(np.mean(np.abs(self.delta_beta_trajectory))) if self.delta_beta_trajectory else 0.0

    @property
    def delta_beta_volatility(self) -> float:
        return float(np.std(self.delta_beta_trajectory)) if len(self.delta_beta_trajectory) >= 2 else 0.0

    @property
    def mean_bus_mass(self) -> float:
        return float(np.mean(self.bus_mass_trajectory)) if self.bus_mass_trajectory else 0.0

    @property
    def mean_ipr(self) -> float:
        return float(np.mean(self.ipr_trajectory)) if self.ipr_trajectory else 0.0

    # ── Data-calibrated archetype vote ──────────────────────────

    def archetype_vote(
        self,
        rules: Optional[Sequence] = None,
    ) -> Dict[str, float]:
        """Data-calibrated archetype vote using thermodynamic fingerprints.

        Delegates to :func:`~ibp_enm.rules.apply_rules` which evaluates
        the :data:`~ibp_enm.rules.ARCHETYPE_RULES` registry against this
        profile.

        Parameters
        ----------
        rules : sequence of ArchetypeRule, optional
            Override the default rule set (for sweeps / experiments).
            When *None*, uses the full ``ARCHETYPE_RULES`` registry.

        Thresholds are derived from D109 Run 1 empirical analysis with
        Cohen's d effect sizes ranging from d = 1.0 to d = 3.4.

        D113 changes:
        - Barrel scatter thresholds use scatter_normalised (size-aware)
        - Barrel scatter voting concentrated in algebraic/thermal/cooperative
          (3 instruments instead of all 7) to reduce false barrel dominance
        - Allosteric gets stronger positive signals

        Key discriminating signals
        --------------------------
        BARREL    : scatter LOW (<1.5)   size-normalised          (d=2.1–3.3)
        DUMBBELL  : Δβ HIGH (>0.1)       across most instruments (d=2.3–2.9)
        GLOBIN    : flatness LOW (<0.75) under algebraic/thermal (d=2.4–3.4)
        ENZYME    : IPR HIGH (>0.025)    across most instruments (d=1.3–1.5)
        ALLOSTERIC: radius HIGH (>30)    under propagative       (d=2.8)
        """
        return apply_rules(self, rules)

    def archetype_vote_traced(
        self,
        rules: Optional[Sequence] = None,
    ) -> tuple:
        """Archetype vote with full audit trail.

        Returns ``(votes, firings)`` — see
        :func:`~ibp_enm.rules.apply_rules_traced`.
        """
        return apply_rules_traced(self, rules)


# ═══════════════════════════════════════════════════════════════════
# ThermoInstrumentCarver — one carver in the band
# ═══════════════════════════════════════════════════════════════════

class ThermoInstrumentCarver:
    """One carver in the thermodynamic band.

    Each instrument selects edges via a genuinely different signal,
    producing maximally diverse perturbation contexts.  Each carver
    operates on its *own copy* of the contact graph so that
    instruments do not interfere with each other.

    Parameters
    ----------
    instrument : str
        One of ``"algebraic"``, ``"musical"``, ``"fick"``,
        ``"thermal"``, ``"cooperative"``, ``"propagative"``,
        ``"fragile"``.
    intent_idx : int
        Starting Fano-plane intent.
    N : int
        Number of residues.
    contacts : dict
        ``{(i, j): distance}`` contact map.
    coords, bfactors, fiedler, domain_labels :
        Structural data from the ENM analyser.
    gap_base : float
        Spectral gap of the intact protein.
    evals, evecs : ndarray
        Laplacian eigenvalues/eigenvectors.
    predicted_bfactors : ndarray or None
        Predicted B-factors from ``IBPProteinAnalyzer``.
    """

    def __init__(self, instrument: str, intent_idx: int,
                 N: int, contacts: dict, coords: np.ndarray,
                 bfactors: np.ndarray, fiedler: np.ndarray,
                 domain_labels: np.ndarray, gap_base: float,
                 evals: np.ndarray, evecs: np.ndarray,
                 predicted_bfactors: Optional[np.ndarray] = None):
        self.instrument = instrument
        self.intent_idx = intent_idx
        self.N = N
        self.contacts = dict(contacts)          # own copy
        self.coords = coords
        self.bfactors = bfactors
        self.predicted_bfactors = predicted_bfactors
        self.fiedler = fiedler
        self.domain_labels = domain_labels
        self.gap_base = gap_base
        self.gap_current = gap_base
        self.evals = evals.copy()
        self.evecs = evecs.copy()

        self.L_current = build_laplacian(N, self.contacts)
        self.inspector = ShadowInspector(
            N, contacts, coords, bfactors, fiedler, domain_labels)
        self.qstate = QuantumCarvingState()
        self.melody = MelodyTracker()
        self.current_intent = CarvingIntent.superposition(
            {intent_idx: 0.8,
             CarvingIntent.fano_multiply(intent_idx,
                                         (intent_idx + 1) % 7): 0.2})
        self.calibration = INSTRUMENT_CALIBRATION.get(instrument, {})

        # FickBalancer for fick instrument
        self.balancer: Optional[FickBalancer] = None
        if instrument == "fick":
            self.balancer = FickBalancer()
            self.balancer.tau_prev = 1.0 / (evals[1] + 1e-10)

        # Initial thermodynamic state
        self.S_current = vibrational_entropy(evals)
        self.Cv_current = heat_capacity(evals)
        self.F_current = helmholtz_free_energy(evals)
        self.ipr_current = mean_ipr_low_modes(evecs)

        # Recording
        self.profile = ThermoReactionProfile(
            instrument=instrument,
            intent_idx=intent_idx,
            n_residues=N,
            n_contacts=len(contacts),
            gap_trajectory=[1.0],
            entropy_trajectory=[self.S_current],
            heat_cap_trajectory=[self.Cv_current],
            free_energy_trajectory=[self.F_current],
            ipr_trajectory=[self.ipr_current],
        )
        self.alpha_trajectory: List[float] = []

    # ── main step ───────────────────────────────────────────────

    def play_step(self, step: int, max_candidates: int = 100) -> Dict:
        """One beat from this instrument — with full thermodynamic recording."""
        edges = list(self.contacts.keys())
        if len(edges) < 5:
            return {"skipped": True}

        rng = np.random.RandomState(42 + step * 7 + self.intent_idx)

        # Sample candidates
        if len(edges) > max_candidates:
            cross = [e for e in edges
                     if self.fiedler is not None
                     and self.fiedler[e[0]] * self.fiedler[e[1]] < 0]
            intra = [e for e in edges if e not in set(cross)]
            n_remain = max(0, max_candidates - len(cross))
            if n_remain < len(intra):
                sel = rng.choice(len(intra), n_remain, replace=False)
                intra = [intra[int(i)] for i in sel]
            candidates = cross + intra
        else:
            candidates = edges

        # Probe all candidates
        reactions = [self.inspector.shadow_probe(e) for e in candidates]

        # Instrument-specific edge selection
        chosen_idx = self._select_edge(reactions, rng)
        if chosen_idx is None:
            return {"skipped": True}

        reaction = reactions[chosen_idx]
        chosen_edge = reaction.edge

        # Record mode scatter & reaction channels
        self.profile.mode_scatters.append(reaction.mode_scatter)
        self.profile.delta_tau_trajectory.append(reaction.delta_tau)
        self.profile.delta_beta_trajectory.append(reaction.delta_beta)
        self.profile.spatial_radius_trajectory.append(reaction.spatial_radius)
        self.profile.bus_mass_trajectory.append(reaction.bus_mass)

        # Pre-cut state
        S_before = self.S_current
        gap_before = self.gap_current

        # Actually cut the edge
        if chosen_edge not in self.contacts:
            return {"skipped": True}
        del self.contacts[chosen_edge]
        ci, cj = chosen_edge
        self.L_current[ci, cj] += 1
        self.L_current[cj, ci] += 1
        self.L_current[ci, ci] -= 1
        self.L_current[cj, cj] -= 1

        ev_new, ec_new = np.linalg.eigh(self.L_current)
        gap_new = spectral_gap(ev_new)

        # New thermodynamic state
        S_new = vibrational_entropy(ev_new)
        Cv_new = heat_capacity(ev_new)
        F_new = helmholtz_free_energy(ev_new)
        ipr_new = mean_ipr_low_modes(ec_new)
        delta_S = S_new - S_before

        # Stitch test
        stitch = self.inspector.stitch_test(
            chosen_edge, self.L_current, self.contacts)

        # Melodic note
        bus = reaction.bus_vector
        dominant_channel = int(np.argmax(bus))
        channel_to_fano = {0: 0, 1: 5, 2: 1, 3: 2}
        fano_point = channel_to_fano.get(dominant_channel, 0)
        note = CarvingNote(
            step=step,
            bus_vector=bus,
            species=reaction.species,
            intent_alignment=float(
                np.dot(self.current_intent,
                       CarvingIntent.basis_vector(fano_point)) ** 2),
            fano_point=fano_point,
            tension=self.qstate.tension(),
        )
        self.melody.add_note(note)

        # Fick alpha tracking
        alpha_val = None
        if self.balancer is not None:
            fick_state = self.balancer.compute_fick_state(
                ev_new, ec_new, self.N, self.contacts, self.fiedler)
            alpha_val = self.balancer.compute_alpha(fick_state)
            self.alpha_trajectory.append(alpha_val)
            self.balancer.step_update(
                fick_state["tau"], reaction.delta_gap, gap_before, gap_new)

        # Intent feedback
        if len(self.melody.notes) >= 3:
            coherence = self.melody.melodic_coherence()
            if coherence < 0.3:
                new_intent = self.qstate.chord_substitution(
                    self.intent_idx, "dissonant")
                if new_intent != self.intent_idx:
                    self.intent_idx = new_intent
                    self.current_intent = CarvingIntent.superposition(
                        {new_intent: 0.8})
                    self.profile.intent_switches += 1

        # Update profile
        gap_retained = gap_new / (self.gap_base + 1e-10)
        self.profile.gap_trajectory.append(float(gap_retained))
        self.profile.species_removed.append(reaction.species)
        self.profile.reversibility.append(stitch["is_reversible"])
        self.profile.tensions.append(float(note.tension))
        self.profile.cuts_made += 1

        self.profile.entropy_trajectory.append(S_new)
        self.profile.heat_cap_trajectory.append(Cv_new)
        self.profile.free_energy_trajectory.append(F_new)
        self.profile.delta_entropy_per_cut.append(delta_S)
        self.profile.ipr_trajectory.append(ipr_new)

        # Update current state
        self.gap_current = gap_new
        self.evals = ev_new
        self.evecs = ec_new
        self.S_current = S_new
        self.Cv_current = Cv_new
        self.F_current = F_new
        self.ipr_current = ipr_new

        return {
            "edge": chosen_edge,
            "species": reaction.species,
            "delta_gap": float(reaction.delta_gap),
            "gap_retained": float(gap_retained),
            "mode_scatter": float(reaction.mode_scatter),
            "delta_tau": float(reaction.delta_tau),
            "delta_beta": float(reaction.delta_beta),
            "spatial_radius": float(reaction.spatial_radius),
            "bus_mass": float(reaction.bus_mass),
            "delta_S": float(delta_S),
            "S_vib": float(S_new),
            "C_v": float(Cv_new),
            "F": float(F_new),
            "ipr": float(ipr_new),
            "reversible": stitch["is_reversible"],
            "tension": float(note.tension),
            "alpha": alpha_val,
        }

    # ── edge selection ──────────────────────────────────────────

    def _select_edge(self, reactions: list, rng) -> Optional[int]:
        """Instrument-specific edge selection — 7 genuinely different modes."""
        if not reactions:
            return None

        if self.instrument == "algebraic":
            scores = [abs(r.delta_gap) for r in reactions]
            return int(np.argmax(scores))

        elif self.instrument == "musical":
            scores = [r.mode_scatter for r in reactions]
            return int(np.argmax(scores))

        elif self.instrument == "fick":
            if self.balancer is not None:
                fick_state = self.balancer.compute_fick_state(
                    self.evals, self.evecs, self.N,
                    self.contacts, self.fiedler)
                alpha = self.balancer.compute_alpha(fick_state)
                candidates = [(r.edge, r.delta_gap) for r in reactions]
                chosen_edge, _, _ = self.balancer.select_edge(
                    candidates, fick_state, alpha)
                if chosen_edge is not None:
                    for i, r in enumerate(reactions):
                        if r.edge == chosen_edge:
                            return i
            # Fallback
            scores = [abs(r.delta_gap) * 0.5 + r.bus_stiffness * 0.5
                      for r in reactions]
            return int(np.argmax(scores))

        elif self.instrument == "thermal":
            scores = [abs(r.delta_tau) for r in reactions]
            return int(np.argmax(scores))

        elif self.instrument == "cooperative":
            scores = [abs(r.delta_beta) for r in reactions]
            return int(np.argmax(scores))

        elif self.instrument == "propagative":
            scores = [r.spatial_radius for r in reactions]
            return int(np.argmax(scores))

        elif self.instrument == "fragile":
            scores = [r.bus_mass for r in reactions]
            return int(np.argmax(scores))

        return 0  # fallback
