"""Carving primitives — the foundational objects for edge-removal analysis.

This module provides the core abstractions for the "carving" approach
to protein structural identification:

CarvingIntent       Seven intents on the Fano plane (BEING … BECOMING)
ReactionSignature   Full reaction to a shadow edge removal
ShadowInspector     Non-destructive edge probing ("palpate before cutting")
QuantumCarvingState |ψ⟩ on the Fano plane, Grover amplification
CarvingNote         One note on the information bus
MelodyTracker       Tracks the cut sequence as a musical melody
FickBalancer        Diffusion-guided carving balancer (α = gap vs flux)

Utility helpers:
    build_laplacian, spectral_gap, classify_edge_species, norm01

Historical notes
----------------
CarvingIntent, ReactionSignature, ShadowInspector, QuantumCarvingState
originate in D106 (The Quantum Carver).  FickBalancer originates in D105.
MelodyTracker and CarvingNote were introduced in D106.  All were validated
across D106–D109 and formalised here for reuse.
"""

from __future__ import annotations

import numpy as np
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

__all__ = [
    # Utilities
    "norm01",
    "build_laplacian",
    "spectral_gap",
    "classify_edge_species",
    # Core classes
    "CarvingIntent",
    "ReactionSignature",
    "QuantumCarvingState",
    "ShadowInspector",
    "CarvingNote",
    "MelodyTracker",
    "FickBalancer",
]


# ═══════════════════════════════════════════════════════════════════
# Utility helpers
# ═══════════════════════════════════════════════════════════════════

def norm01(x: np.ndarray) -> np.ndarray:
    """Normalise array to [0, 1]."""
    rng = x.max() - x.min()
    return np.zeros_like(x) if rng < 1e-10 else (x - x.min()) / rng


def build_laplacian(N: int, contacts: dict) -> np.ndarray:
    """Build the graph Laplacian from a contact dictionary.

    Parameters
    ----------
    N : int
        Number of nodes (residues).
    contacts : dict
        ``{(i, j): distance_or_weight, ...}`` for i < j.

    Returns
    -------
    L : (N, N) ndarray
        Unweighted graph Laplacian.
    """
    L = np.zeros((N, N))
    for (i, j) in contacts:
        L[i, j] = -1
        L[j, i] = -1
        L[i, i] += 1
        L[j, j] += 1
    return L


def spectral_gap(evals: np.ndarray) -> float:
    """Ratio λ₂ / λ₃ (the Fiedler gap)."""
    if len(evals) > 2 and evals[2] > 1e-10:
        return float(evals[1] / evals[2])
    return 0.0


def classify_edge_species(edge: Tuple[int, int],
                          fiedler: Optional[np.ndarray],
                          domain_labels: Optional[np.ndarray],
                          delta_gap: float,
                          gap_base: float) -> str:
    """Classify an edge into one of four species (D95 taxonomy).

    Bridge   — cross-domain, gap-sensitive
    Carved   — intra-domain, gap-sensitive
    Sibling  — intra-domain, gap-insensitive
    Soft     — gap-insensitive (< 0.1 % of gap)
    """
    i, j = edge
    cross = (fiedler[i] * fiedler[j] < 0) if fiedler is not None else False
    same = (domain_labels[i] == domain_labels[j]) if domain_labels is not None else True
    rel = abs(delta_gap) / gap_base if gap_base > 1e-10 else 0

    if cross and rel > 0.005:
        return "Bridge"
    elif rel < 0.001:
        return "Soft"
    elif rel < 0.01 and same:
        return "Sibling"
    else:
        return "Carved"


# ═══════════════════════════════════════════════════════════════════
# CarvingIntent — the Fano plane of carving intentions
# ═══════════════════════════════════════════════════════════════════

class CarvingIntent:
    """Seven carving intents arranged on the Fano plane.

    Each intent is a basis vector |k⟩ in a 7-dimensional Hilbert space.
    The Fano plane structure provides 7 *lines* (narrative arcs) where
    any 3 collinear intents form a consistent carving strategy.

    The 7 intents map to ChordSpeak's FanoPoint naming:

    ===  ========  ==========================================
    idx  name      description
    ===  ========  ==========================================
    0    BEING     structural maintenance — preserve the gap
    1    WANTING   exploration — rich path, low shortcut ratio
    2    FEELING   sensitivity — find fragile / reactive edges
    3    KNOWING   identity lock — confirm archetype
    4    DOING     aggressive carving — maximise depth
    5    RELATING  bridge protection — preserve domain links
    6    BECOMING  gap improvement — sculpt toward separation
    ===  ========  ==========================================
    """

    # Intent indices
    BEING     = 0
    WANTING   = 1
    FEELING   = 2
    KNOWING   = 3
    DOING     = 4
    RELATING  = 5
    BECOMING  = 6

    NAMES = [
        "BEING", "WANTING", "FEELING", "KNOWING",
        "DOING", "RELATING", "BECOMING",
    ]

    SENTENCES = [
        "preserve structural integrity",
        "explore carving richness",
        "detect sensitivity and fragility",
        "identify protein archetype",
        "carve aggressively for depth",
        "protect domain boundaries",
        "sculpt toward improved separation",
    ]

    # 7 Fano lines — each a consistent narrative arc
    LINES = [
        (0, 1, 3),   # BEING–WANTING–KNOWING       "motivation"
        (1, 2, 4),   # WANTING–FEELING–DOING        "intention"
        (2, 3, 5),   # FEELING–KNOWING–RELATING     "understanding"
        (3, 4, 6),   # KNOWING–DOING–BECOMING       "mastery"
        (4, 5, 0),   # DOING–RELATING–BEING         "identity"
        (5, 6, 1),   # RELATING–BECOMING–WANTING    "growth"
        (6, 0, 2),   # BECOMING–BEING–FEELING       "transformation"
    ]

    LINE_NAMES = [
        "MOTIVATION", "INTENTION", "UNDERSTANDING",
        "MASTERY", "IDENTITY", "GROWTH", "TRANSFORMATION",
    ]

    # Intrinsic tension per intent (from ChordSpeak POINT_TENSION)
    TENSION = [0.1, 0.7, 0.5, 0.2, 0.4, 0.3, 0.9]

    @staticmethod
    def fano_multiply(a: int, b: int) -> int:
        """Non-associative Fano plane product.

        Returns the third point on the unique line through *a* and *b*.
        """
        if a == b:
            return a
        for line in CarvingIntent.LINES:
            if a in line and b in line:
                for p in line:
                    if p != a and p != b:
                        return p
        return (a + b) % 7

    @staticmethod
    def basis_vector(k: int) -> np.ndarray:
        """Basis vector |k⟩ in the 7-dim intent space."""
        v = np.zeros(7)
        v[k] = 1.0
        return v

    @staticmethod
    def superposition(weights: Dict[int, float]) -> np.ndarray:
        """Normalised superposition Σ w_k |k⟩."""
        v = np.zeros(7)
        for k, w in weights.items():
            v[k] = w
        norm = np.linalg.norm(v)
        if norm > 1e-10:
            v /= norm
        return v


# ═══════════════════════════════════════════════════════════════════
# ReactionSignature — full reaction to an edge removal
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ReactionSignature:
    """Full reaction to a (shadow) edge removal.

    Contains 12 fields across 4 information channels:

    Spectral channel:
        delta_gap, delta_tau, mode_scatter, species

    Cooperativity channel:
        delta_beta, is_cross_domain

    Spatial channel:
        spatial_radius

    Bus channel (normalised to [0, 1]):
        bus_stiffness, bus_topology, bus_geometry, bus_mass
    """

    edge: Tuple[int, int]
    delta_gap: float
    delta_tau: float
    delta_beta: float
    species: str
    mode_scatter: float
    is_cross_domain: bool
    spatial_radius: float

    bus_stiffness: float
    bus_topology: float
    bus_geometry: float
    bus_mass: float

    @property
    def bus_vector(self) -> np.ndarray:
        return np.array([self.bus_stiffness, self.bus_topology,
                         self.bus_geometry, self.bus_mass])

    @property
    def magnitude(self) -> float:
        return float(np.linalg.norm(self.bus_vector))


# ═══════════════════════════════════════════════════════════════════
# QuantumCarvingState — the protein's state on the Fano plane
# ═══════════════════════════════════════════════════════════════════

class QuantumCarvingState:
    """The protein's carving state as a 7-dim vector on the Fano plane.

    |ψ⟩ = Σ c_k |k⟩ computed from observable metrics::

        c₀ (BEING)    = gap_retained
        c₁ (WANTING)  = 1 − shortcut_ratio
        c₂ (FEELING)  = sensitivity
        c₃ (KNOWING)  = archetype_confidence
        c₄ (DOING)    = carving_depth / max_depth
        c₅ (RELATING) = 1 − bridge_frac
        c₆ (BECOMING) = gap_improvement
    """

    def __init__(self):
        self.psi = np.ones(7) / np.sqrt(7)
        self.history: List[np.ndarray] = []

    def update_from_metrics(self, gap_retained: float,
                            shortcut_ratio: float,
                            sensitivity: float,
                            archetype_confidence: float,
                            carving_depth_frac: float,
                            bridge_frac: float,
                            gap_improvement: float) -> None:
        """Rebuild |ψ⟩ from current observables."""
        self.psi = np.array([
            max(0, gap_retained),
            max(0, 1 - shortcut_ratio),
            max(0, sensitivity),
            max(0, archetype_confidence),
            max(0, min(1, carving_depth_frac)),
            max(0, 1 - bridge_frac),
            max(0, gap_improvement),
        ])
        norm = np.linalg.norm(self.psi)
        if norm > 1e-10:
            self.psi /= norm
        self.history.append(self.psi.copy())

    def alignment(self, intent: np.ndarray) -> float:
        """Born probability |⟨ψ|I⟩|²."""
        return float(np.dot(self.psi, intent)) ** 2

    def grover_amplify(self, intent: np.ndarray,
                       n_iters: int = 1) -> np.ndarray:
        """Grover iterate: amplify the intent-aligned component."""
        psi = self.psi.copy()
        I_proj = np.outer(intent, intent)
        for _ in range(n_iters):
            oracle = 2 * I_proj - np.eye(7)
            diffusion = 2 * np.outer(psi, psi) - np.eye(7)
            psi = (diffusion @ oracle) @ psi
            norm = np.linalg.norm(psi)
            if norm > 1e-10:
                psi /= norm
        return psi

    def chord_substitution(self, current_intent_idx: int,
                           feedback: str) -> int:
        """Musical chord substitution via the Fano plane.

        "dissonant" → jump to opposite end of Fano line.
        "modulate"  → step one position along line.
        "consonant" → stay.
        """
        if feedback == "consonant":
            return current_intent_idx
        for line in CarvingIntent.LINES:
            if current_intent_idx in line:
                others = [p for p in line if p != current_intent_idx]
                if feedback == "dissonant":
                    return others[-1]
                elif feedback == "modulate":
                    return others[0]
        return current_intent_idx

    def dominant_intent(self) -> Tuple[int, float]:
        """Intent with highest amplitude."""
        idx = int(np.argmax(np.abs(self.psi)))
        return idx, float(self.psi[idx])

    def fano_line_coherence(self) -> Tuple[int, float]:
        """Which Fano line best describes the current state?"""
        best_line = 0
        best_prob = 0.0
        for li, (a, b, c) in enumerate(CarvingIntent.LINES):
            prob = self.psi[a]**2 + self.psi[b]**2 + self.psi[c]**2
            if prob > best_prob:
                best_prob = prob
                best_line = li
        return best_line, float(best_prob)

    def tension(self) -> float:
        """Weighted tension from ChordSpeak POINT_TENSION."""
        return float(np.dot(np.abs(self.psi)**2, CarvingIntent.TENSION))


# ═══════════════════════════════════════════════════════════════════
# ShadowInspector — non-destructive edge probing
# ═══════════════════════════════════════════════════════════════════

class ShadowInspector:
    """Probe edges without permanently altering the graph.

    *"A surgeon palpates before cutting."*

    Shadow mode
        Temporarily remove an edge, measure the full
        :class:`ReactionSignature`, then restore.  The protein is
        never permanently altered.

    Stitch mode
        After a *real* cut, stitch the edge back and measure how well
        the original state recovers.  Full recovery → reversible
        (Soft edge); incomplete recovery → permanent change (Carved /
        Bridge edge).
    """

    def __init__(self, N: int, contacts: dict, coords: np.ndarray,
                 bfactors: np.ndarray, fiedler: Optional[np.ndarray],
                 domain_labels: Optional[np.ndarray]):
        self.N = N
        self.contacts = dict(contacts)
        self.coords = coords
        self.bfactors = bfactors
        self.fiedler = fiedler
        self.domain_labels = domain_labels

        # Baseline spectral state
        self.L_base = build_laplacian(N, self.contacts)
        ev, ec = np.linalg.eigh(self.L_base)
        self.evals_base = ev
        self.evecs_base = ec
        self.gap_base = spectral_gap(ev)
        self.tau_base = 1.0 / (ev[1] + 1e-10)
        self.beta_base = self._compute_beta(ev, ec, self.contacts)

    # ── internal ────────────────────────────────────────────────

    def _compute_beta(self, evals: np.ndarray, evecs: np.ndarray,
                      contacts: dict) -> float:
        """Bottleneck factor β (D104)."""
        psi2 = evecs[:, 1]
        lam2 = evals[1]
        boundary_flux = total_flux = 0.0
        n_bound = n_total = 0
        for (i, j) in contacts:
            grad = abs(psi2[j] - psi2[i])
            flux = lam2 * grad
            total_flux += flux
            n_total += 1
            if self.fiedler is not None and self.fiedler[i] * self.fiedler[j] < 0:
                boundary_flux += flux
                n_bound += 1
        flux_share = boundary_flux / (total_flux + 1e-10)
        edge_share = n_bound / (n_total + 1e-10) if n_total > 0 else 0
        return flux_share / (edge_share + 1e-10) if edge_share > 0 else 1.0

    # ── public API ──────────────────────────────────────────────

    def shadow_probe(self, edge: Tuple[int, int]) -> ReactionSignature:
        """Probe *edge* without permanent removal."""
        ci, cj = edge

        # Temporarily remove
        L_red = self.L_base.copy()
        L_red[ci, cj] += 1
        L_red[cj, ci] += 1
        L_red[ci, ci] -= 1
        L_red[cj, cj] -= 1

        ev_red, ec_red = np.linalg.eigh(L_red)
        gap_red = spectral_gap(ev_red)
        tau_red = 1.0 / (ev_red[1] + 1e-10)

        delta_gap = gap_red - self.gap_base
        delta_tau = tau_red / (self.tau_base + 1e-10) - 1

        contacts_minus = {k: v for k, v in self.contacts.items()
                          if k != edge}
        beta_red = self._compute_beta(ev_red, ec_red, contacts_minus)
        delta_beta = beta_red - self.beta_base

        species = classify_edge_species(
            edge, self.fiedler, self.domain_labels,
            delta_gap, self.gap_base)

        n_check = min(10, len(ev_red) - 1)
        mode_shifts = np.abs(
            ev_red[1:n_check + 1] - self.evals_base[1:n_check + 1])
        mode_scatter = float(np.sum(mode_shifts > 0.01 * self.gap_base))

        is_cross = (bool(self.fiedler[ci] * self.fiedler[cj] < 0)
                    if self.fiedler is not None else False)

        # Spatial radius of perturbation
        delta_fiedler = np.abs(ec_red[:, 1] - self.evecs_base[:, 1])
        affected = np.where(delta_fiedler > 0.01)[0]
        if len(affected) > 1 and self.coords is not None:
            ac = self.coords[affected]
            centroid = ac.mean(axis=0)
            spatial_radius = float(np.max(
                np.linalg.norm(ac - centroid, axis=1)))
        else:
            spatial_radius = 0.0

        # Bus signals (normalised to [0, 1])
        bus_stiffness = min(1, abs(delta_gap) / (self.gap_base + 1e-10))
        bus_topology = 1.0 if is_cross else 0.0
        bus_geometry = min(1, spatial_radius / 20.0)
        bus_mass = 0.0
        if self.bfactors is not None and len(self.bfactors) > max(ci, cj):
            mean_b = (self.bfactors[ci] + self.bfactors[cj]) / 2
            bus_mass = min(1, mean_b / (np.mean(self.bfactors) + 1e-10))

        return ReactionSignature(
            edge=edge, delta_gap=delta_gap, delta_tau=delta_tau,
            delta_beta=delta_beta, species=species,
            mode_scatter=mode_scatter, is_cross_domain=is_cross,
            spatial_radius=spatial_radius,
            bus_stiffness=bus_stiffness, bus_topology=bus_topology,
            bus_geometry=bus_geometry, bus_mass=bus_mass,
        )

    def shadow_probe_batch(self, edges: List[Tuple[int, int]],
                           max_probes: int = 100
                           ) -> List[ReactionSignature]:
        """Probe multiple edges (sampled if *len(edges) > max_probes*)."""
        rng = np.random.RandomState(42)
        if len(edges) > max_probes:
            cross = [e for e in edges
                     if self.fiedler is not None
                     and self.fiedler[e[0]] * self.fiedler[e[1]] < 0]
            intra = [e for e in edges if e not in cross]
            n_remain = max(0, max_probes - len(cross))
            if n_remain < len(intra):
                sampled = [intra[i]
                           for i in rng.choice(len(intra), n_remain,
                                               replace=False)]
            else:
                sampled = intra
            edges = cross + sampled
        return [self.shadow_probe(e) for e in edges]

    def stitch_test(self, edge: Tuple[int, int],
                    L_current: np.ndarray,
                    contacts_current: dict) -> Dict:
        """After a real cut, stitch the edge back and measure recovery."""
        ci, cj = edge
        L_stitch = L_current.copy()
        L_stitch[ci, cj] -= 1
        L_stitch[cj, ci] -= 1
        L_stitch[ci, ci] += 1
        L_stitch[cj, cj] += 1

        ev_stitch, ec_stitch = np.linalg.eigh(L_stitch)
        gap_stitch = spectral_gap(ev_stitch)
        gap_recovery = gap_stitch / (self.gap_base + 1e-10)

        fiedler_stitch = ec_stitch[:, 1]
        if np.dot(fiedler_stitch, self.evecs_base[:, 1]) < 0:
            fiedler_stitch = -fiedler_stitch
        fiedler_diff = np.linalg.norm(
            fiedler_stitch - self.evecs_base[:, 1])
        fiedler_recovery = max(0, 1 - fiedler_diff)

        is_reversible = (abs(gap_recovery - 1.0) < 0.02
                         and fiedler_recovery > 0.95)
        return {
            "gap_recovery": float(gap_recovery),
            "fiedler_recovery": float(fiedler_recovery),
            "is_reversible": bool(is_reversible),
        }


# ═══════════════════════════════════════════════════════════════════
# CarvingNote + MelodyTracker — the musical metaphor
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CarvingNote:
    """A single note on the information bus — the reaction to one cut."""

    step: int
    bus_vector: np.ndarray
    species: str
    intent_alignment: float
    fano_point: int
    tension: float

    @property
    def pitch(self) -> str:
        """Map bus magnitude to a pitch class."""
        mag = np.linalg.norm(self.bus_vector)
        pitches = ["C", "D", "E", "F", "G", "A", "B"]
        return pitches[min(6, int(mag * 7))]


class MelodyTracker:
    """Track the sequence of cuts as a melody on the Fano plane.

    The melody should stay on a single Fano line (narrative arc).
    Deviations signal structural anomalies.
    """

    def __init__(self):
        self.notes: List[CarvingNote] = []
        self.expected_line: Optional[int] = None
        self.deviations = 0
        self.total_checks = 0

    def set_expected_line(self, line_idx: int) -> None:
        self.expected_line = line_idx

    def add_note(self, note: CarvingNote) -> None:
        self.notes.append(note)
        if self.expected_line is not None and len(self.notes) >= 2:
            line = CarvingIntent.LINES[self.expected_line]
            self.total_checks += 1
            if note.fano_point not in line:
                self.deviations += 1

    def melodic_coherence(self) -> float:
        """Fraction of notes that stay on the expected Fano line."""
        if self.total_checks == 0:
            return 1.0
        return 1 - self.deviations / self.total_checks

    def tension_arc(self) -> List[float]:
        return [n.tension for n in self.notes]

    def pitch_sequence(self) -> str:
        return " ".join(n.pitch for n in self.notes)

    def species_melody(self) -> str:
        return " → ".join(n.species for n in self.notes)

    def detect_anomaly(self, threshold: float = 0.4) -> bool:
        return self.melodic_coherence() < threshold


# ═══════════════════════════════════════════════════════════════════
# FickBalancer — diffusion-guided edge selection
# ═══════════════════════════════════════════════════════════════════

class FickBalancer:
    """Diffusion-guided carving balancer (D105).

    The Fick state at each step::

        F = {τ, β, ρ, Δτ, σ²}

    Balancer equation::

        α = sigmoid(w₁ · log Δτ  +  w₂ · (β − β₀)  +  w₃ · ρ)

    Edge scoring::

        score(e) = α · rank_gap(e)  +  (1 − α) · rank_flux(e)
    """

    def __init__(self, w1: float = -1.5, w2: float = -0.3,
                 w3: float = 0.5, beta0: float = 2.0):
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.beta0 = beta0
        self.tau_prev: Optional[float] = None
        self.rho_accum = 0.0
        self.step = 0
        self.history: List[dict] = []

    def reset(self) -> None:
        self.tau_prev = None
        self.rho_accum = 0.0
        self.step = 0
        self.history = []

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))

    def compute_fick_state(self, evals: np.ndarray, evecs: np.ndarray,
                           N: int, contacts: dict,
                           fiedler: Optional[np.ndarray]) -> Dict:
        """Compute the full Fick state from current spectral data."""
        tau = 1.0 / (evals[1] + 1e-10)
        psi2 = evecs[:, 1]
        lambda2 = evals[1]

        boundary_fluxes: List[float] = []
        interior_fluxes: List[float] = []
        edge_fluxes: Dict[Tuple[int, int], float] = {}

        for (i, j) in contacts:
            gradient = abs(psi2[j] - psi2[i])
            flux = lambda2 * gradient
            edge_fluxes[(i, j)] = flux
            if fiedler is not None and fiedler[i] * fiedler[j] < 0:
                boundary_fluxes.append(flux)
            else:
                interior_fluxes.append(flux)

        total_flux = sum(boundary_fluxes) + sum(interior_fluxes)
        n_bound = len(boundary_fluxes)
        n_total = n_bound + len(interior_fluxes)

        flux_share = sum(boundary_fluxes) / (total_flux + 1e-10)
        edge_share = n_bound / (n_total + 1e-10) if n_total > 0 else 0
        beta = flux_share / (edge_share + 1e-10) if edge_share > 0 else 1.0

        if self.tau_prev is not None and self.tau_prev > 0:
            delta_tau = tau / self.tau_prev
        else:
            delta_tau = 1.0

        n_check = min(10, len(evals) - 2)
        gaps = np.diff(evals[1:n_check + 2])
        sigma2 = float(np.var(gaps)) if len(gaps) > 0 else 0.0

        return {
            "tau": float(tau),
            "beta": float(beta),
            "delta_tau": float(delta_tau),
            "sigma2": float(sigma2),
            "rho": float(self.rho_accum),
            "edge_fluxes": edge_fluxes,
        }

    def compute_alpha(self, fick_state: Dict) -> float:
        """The Fick balancing parameter α ∈ [0, 1]."""
        log_dt = np.log(fick_state["delta_tau"] + 1e-10)
        z = (self.w1 * log_dt
             + self.w2 * (fick_state["beta"] - self.beta0)
             + self.w3 * fick_state["rho"])
        return float(self._sigmoid(z))

    def select_edge(self, candidates: List[Tuple],
                    fick_state: Dict, alpha: float
                    ) -> Tuple[Optional[Tuple], float, Dict]:
        """Score edges and return the best one.

        Each candidate is ``(edge, delta_gap)``.
        """
        edge_fluxes = fick_state["edge_fluxes"]
        n = len(candidates)
        if n == 0:
            return None, 0.0, {}

        sorted_by_gap = sorted(range(n), key=lambda i: candidates[i][1])
        rank_gap = np.zeros(n)
        for rank, idx in enumerate(sorted_by_gap):
            rank_gap[idx] = rank / (n - 1 + 1e-10)

        fluxes = [edge_fluxes.get(e, 0) for e, _ in candidates]
        sorted_by_flux = sorted(range(n), key=lambda i: fluxes[i])
        rank_flux = np.zeros(n)
        for rank, idx in enumerate(sorted_by_flux):
            rank_flux[idx] = rank / (n - 1 + 1e-10)

        scores = alpha * rank_gap + (1 - alpha) * rank_flux
        best = int(np.argmax(scores))

        return candidates[best][0], candidates[best][1], {
            "best_rank_gap": float(rank_gap[best]),
            "best_rank_flux": float(rank_flux[best]),
            "best_score": float(scores[best]),
            "alpha": float(alpha),
        }

    def step_update(self, tau_new: float, delta_gap: float,
                    gap_before: float, gap_after: float) -> None:
        """Update internal state after a real cut."""
        self.tau_prev = tau_new
        self.step += 1
        attr = abs(delta_gap) / gap_before if gap_before > 0 else 0
        self.rho_accum = max(self.rho_accum, attr)
