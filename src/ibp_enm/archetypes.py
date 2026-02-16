"""Protein archetypes and their expected thermodynamic fingerprints.

This module defines the 5 structural archetypes recognised by the
thermodynamic band, plus the 12-protein benchmark corpus used for
validation.

Archetypes
----------
======  ========  ====================================================
Name    Species   Structural character
======  ========  ====================================================
dumbbell  Soft    Fragile inter-domain bridge
barrel    Carved  Rigid cylindrical scaffold
globin    Sibling Helical bundle with hinge regions
enzyme    Carved  Functional core with localised active site
allosteric Bridge Signal-coupled domains, long-range propagation
======  ========  ====================================================

ArchetypeExpectation
    Per-step behaviour: gap shape, species fractions, reversibility,
    tension, regularity — used by the surprise accumulator and the
    per-instrument vote functions.

PROTEINS
    The 12-protein benchmark corpus with PDB codes and chain IDs.

Historical notes
----------------
ProteinArchetype and ARCHETYPES originate in D106.
ArchetypeExpectation and ARCHETYPE_EXPECTATIONS originate in D107.
Ground-truth assignment confirmed across D105–D109.
"""

from __future__ import annotations

import numpy as np
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

__all__ = [
    "ProteinArchetype",
    "ARCHETYPES",
    "ArchetypeExpectation",
    "ARCHETYPE_EXPECTATIONS",
    "SurgeonsHandbook",
    "PROTEINS",
    "GROUND_TRUTH",
]


# ═══════════════════════════════════════════════════════════════════
# ProteinArchetype — compact bus-vector description
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ProteinArchetype:
    """An archetype defined by its expected bus-vector profile and
    Fick α range.

    Parameters
    ----------
    name : str
        e.g. ``"barrel"``
    description : str
        NLP-style sentence (meta-IBP notation).
    expected_alpha_range : tuple of float
        Expected Fick α range from D105.
    dominant_species : str
        Most commonly removed edge species.
    gap_sensitivity : str
        ``"high"`` / ``"medium"`` / ``"low"``.
    probe_signature : ndarray
        Expected 4-channel bus vector.
    fano_line : int
        Characteristic narrative arc (index into CarvingIntent.LINES).
    """

    name: str
    description: str
    expected_alpha_range: Tuple[float, float]
    dominant_species: str
    gap_sensitivity: str
    probe_signature: np.ndarray
    fano_line: int

    def match_score(self, observed_bus: np.ndarray,
                    observed_alpha: float) -> float:
        """Cosine similarity (bus) + α-range match."""
        bus_sim = np.dot(self.probe_signature, observed_bus) / (
            np.linalg.norm(self.probe_signature)
            * np.linalg.norm(observed_bus) + 1e-10)

        mid = (self.expected_alpha_range[0]
               + self.expected_alpha_range[1]) / 2
        span = (self.expected_alpha_range[1]
                - self.expected_alpha_range[0]) / 2
        alpha_dist = abs(observed_alpha - mid) / (span + 0.1)
        alpha_match = max(0, 1 - alpha_dist)

        return 0.6 * bus_sim + 0.4 * alpha_match


ARCHETYPES: List[ProteinArchetype] = [
    ProteinArchetype(
        name="dumbbell",
        description="fragile inter-domain bridge; handle with extreme care",
        expected_alpha_range=(0.28, 0.35),
        dominant_species="Soft",
        gap_sensitivity="high",
        probe_signature=np.array([0.8, 0.9, 0.3, 0.2]),
        fano_line=6,
    ),
    ProteinArchetype(
        name="barrel",
        description="rigid cylindrical scaffold; carving is safe and progressive",
        expected_alpha_range=(0.49, 0.55),
        dominant_species="Carved",
        gap_sensitivity="low",
        probe_signature=np.array([0.3, 0.2, 0.5, 0.4]),
        fano_line=3,
    ),
    ProteinArchetype(
        name="globin",
        description="helical bundle with hinge regions; watch hinge mechanics",
        expected_alpha_range=(0.45, 0.50),
        dominant_species="Sibling",
        gap_sensitivity="medium",
        probe_signature=np.array([0.5, 0.4, 0.6, 0.5]),
        fano_line=0,
    ),
    ProteinArchetype(
        name="enzyme_active",
        description="functional core with active site; protect chemical locks",
        expected_alpha_range=(0.48, 0.58),
        dominant_species="Carved",
        gap_sensitivity="medium",
        probe_signature=np.array([0.6, 0.3, 0.4, 0.7]),
        fano_line=1,
    ),
    ProteinArchetype(
        name="allosteric",
        description="signal-coupled domains; cuts propagate unexpectedly",
        expected_alpha_range=(0.35, 0.48),
        dominant_species="Bridge",
        gap_sensitivity="high",
        probe_signature=np.array([0.7, 0.8, 0.5, 0.4]),
        fano_line=5,
    ),
]


# ═══════════════════════════════════════════════════════════════════
# ArchetypeExpectation — per-step expected behaviour
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ArchetypeExpectation:
    """Expected per-step behaviour for an archetype.

    Used by the surprise accumulator and by each instrument's
    vote function for species-diversity matching.
    """

    name: str
    gap_shape: str
    gap_tolerance: float
    gap_cumulative_drift: float
    expected_species: Dict[str, float]
    species_diversity: float
    expected_rev_frac: float
    rev_tolerance: float
    expected_switches_per_12: float
    switch_tolerance: float
    expected_tension_terminal: float
    expected_tension_range: float
    expected_regularity: float


ARCHETYPE_EXPECTATIONS: Dict[str, ArchetypeExpectation] = {
    "dumbbell": ArchetypeExpectation(
        name="dumbbell",
        gap_shape="volatile",
        gap_tolerance=0.03,
        gap_cumulative_drift=-0.02,
        expected_species={"Soft": 0.50, "Bridge": 0.15,
                          "Sibling": 0.25, "Carved": 0.10},
        species_diversity=1.2,
        expected_rev_frac=0.85,
        rev_tolerance=0.15,
        expected_switches_per_12=9,
        switch_tolerance=3,
        expected_tension_terminal=0.27,
        expected_tension_range=0.06,
        expected_regularity=0.3,
    ),
    "barrel": ArchetypeExpectation(
        name="barrel",
        gap_shape="flat",
        gap_tolerance=0.015,
        gap_cumulative_drift=0.005,
        expected_species={"Soft": 0.45, "Carved": 0.35,
                          "Bridge": 0.10, "Sibling": 0.10},
        species_diversity=0.9,
        expected_rev_frac=0.70,
        rev_tolerance=0.25,
        expected_switches_per_12=5,
        switch_tolerance=5,
        expected_tension_terminal=0.25,
        expected_tension_range=0.04,
        expected_regularity=0.6,
    ),
    "globin": ArchetypeExpectation(
        name="globin",
        gap_shape="drift_down",
        gap_tolerance=0.02,
        gap_cumulative_drift=-0.06,
        expected_species={"Carved": 0.40, "Soft": 0.25,
                          "Sibling": 0.15, "Bridge": 0.20},
        species_diversity=1.3,
        expected_rev_frac=0.42,
        rev_tolerance=0.15,
        expected_switches_per_12=10,
        switch_tolerance=2,
        expected_tension_terminal=0.26,
        expected_tension_range=0.05,
        expected_regularity=0.35,
    ),
    "enzyme_active": ArchetypeExpectation(
        name="enzyme_active",
        gap_shape="cliff",
        gap_tolerance=0.025,
        gap_cumulative_drift=-0.04,
        expected_species={"Carved": 0.40, "Soft": 0.30,
                          "Sibling": 0.15, "Bridge": 0.15},
        species_diversity=1.3,
        expected_rev_frac=0.55,
        rev_tolerance=0.20,
        expected_switches_per_12=10,
        switch_tolerance=2,
        expected_tension_terminal=0.26,
        expected_tension_range=0.06,
        expected_regularity=0.35,
    ),
    "allosteric": ArchetypeExpectation(
        name="allosteric",
        gap_shape="amplify",
        gap_tolerance=0.025,
        gap_cumulative_drift=0.05,
        expected_species={"Carved": 0.40, "Soft": 0.20,
                          "Sibling": 0.20, "Bridge": 0.20},
        species_diversity=1.35,
        expected_rev_frac=0.60,
        rev_tolerance=0.15,
        expected_switches_per_12=9,
        switch_tolerance=3,
        expected_tension_terminal=0.30,
        expected_tension_range=0.10,
        expected_regularity=0.10,
    ),
}


# ═══════════════════════════════════════════════════════════════════
# SurgeonsHandbook — snapshot archetype diagnosis
# ═══════════════════════════════════════════════════════════════════

class SurgeonsHandbook:
    """Identify protein archetype from shadow probes alone.

    The meta-IBP protocol: *"you are what you cut."*

    1. Shadow-probe a sample of edges
    2. Aggregate the bus-vector statistics
    3. Match against known archetypes
    4. Generate a diagnostic NLP sentence
    """

    def __init__(self, archetypes: Optional[List[ProteinArchetype]] = None):
        self.archetypes = archetypes or ARCHETYPES
        self.observations: List[dict] = []

    def diagnose(self, reactions, alpha_estimate: float) -> Dict:
        """Archetype diagnosis from a list of ReactionSignatures."""
        species_buses: Dict[str, list] = defaultdict(list)
        all_buses = []
        for r in reactions:
            species_buses[r.species].append(r.bus_vector)
            all_buses.append(r.bus_vector)

        if not all_buses:
            return {"archetype": "unknown", "confidence": 0,
                    "sentence": "???"}

        mean_bus = np.mean(all_buses, axis=0)
        species_counts = Counter(r.species for r in reactions)
        total = sum(species_counts.values())
        dominant_species = species_counts.most_common(1)[0][0]

        magnitudes = [r.magnitude for r in reactions]
        sensitivity = max(magnitudes) / (np.mean(magnitudes) + 1e-10)
        gap_sensitivity_score = float(
            np.std([abs(r.delta_gap) for r in reactions])
            / (np.mean([abs(r.delta_gap) for r in reactions]) + 1e-10))

        scores = [(a, a.match_score(mean_bus, alpha_estimate))
                  for a in self.archetypes]
        scores.sort(key=lambda x: -x[1])

        best_arch, best_score = scores[0]
        second_arch, second_score = (scores[1] if len(scores) > 1
                                     else (None, 0))
        confidence_gap = best_score - second_score
        if confidence_gap > 0.2:
            confidence = "CLEAR"
        elif confidence_gap > 0.05:
            confidence = "LIMINAL"
        else:
            confidence = "SUPERPOSITION"

        sentence = self._generate_sentence(
            best_arch, dominant_species, sensitivity,
            species_counts, total)

        diagnosis = {
            "archetype": best_arch.name,
            "archetype_score": float(best_score),
            "confidence": confidence,
            "confidence_gap": float(confidence_gap),
            "runner_up": second_arch.name if second_arch else None,
            "runner_up_score": float(second_score),
            "mean_bus": mean_bus.tolist(),
            "species_distribution": dict(species_counts),
            "dominant_species": dominant_species,
            "sensitivity": float(sensitivity),
            "gap_sensitivity_score": gap_sensitivity_score,
            "sentence": sentence,
            "all_scores": [(a.name, float(s)) for a, s in scores],
        }
        self.observations.append(diagnosis)
        return diagnosis

    @staticmethod
    def _generate_sentence(archetype, dominant_species,
                           sensitivity, species_counts, total) -> str:
        species_verbs = {
            "Soft": "drifts through noise",
            "Sibling": "resonates with neighbors",
            "Carved": "sculpts deep structure",
            "Bridge": "spans domain boundaries",
        }
        action = species_verbs.get(dominant_species, "responds")
        if sensitivity > 3.0:
            qual = "with sharp reactive spikes"
        elif sensitivity > 1.5:
            qual = "with moderate variability"
        else:
            qual = "with uniform gentleness"
        parts = [f"{sp}({c}/{total})"
                 for sp, c in species_counts.most_common()]
        mixture = ", ".join(parts)
        return (f"This {archetype.name} protein {action} {qual}. "
                f"Probe spectrum: [{mixture}]. "
                f"{archetype.description}.")

    def compare_diagnosis_to_truth(self, diagnosis: Dict,
                                   known_name: str) -> Dict:
        """Compare blind diagnosis to ground truth."""
        true_arch = GROUND_TRUTH.get(known_name, "unknown")
        predicted = diagnosis["archetype"]
        return {
            "protein": known_name,
            "true_archetype": true_arch,
            "predicted_archetype": predicted,
            "correct": predicted == true_arch,
            "confidence": diagnosis["confidence"],
            "match_score": diagnosis["archetype_score"],
        }


# ═══════════════════════════════════════════════════════════════════
# Benchmark corpus
# ═══════════════════════════════════════════════════════════════════

PROTEINS: List[Tuple[str, str, str]] = [
    ("T4_lysozyme",       "2LZM", "A"),
    ("HEWL",              "1LYZ", "A"),
    ("CaM_Ca_bound",      "3CLN", "A"),
    ("Myoglobin",         "1MBO", "A"),
    ("AdK_open",          "4AKE", "A"),
    ("DHFR",              "3DFR", "A"),
    ("Streptavidin",      "1STP", "A"),
    ("TIM_barrel",        "1TIM", "A"),
    ("LAO_binding",       "2LAO", "A"),
    ("HIV_protease",      "1HHP", "A"),
    ("Hemoglobin_alpha",  "2HHB", "A"),
    ("Citrate_synthase",  "5CSC", "A"),
]

GROUND_TRUTH: Dict[str, str] = {
    "CaM_Ca_bound":      "dumbbell",
    "LAO_binding":        "dumbbell",
    "TIM_barrel":         "barrel",
    "Citrate_synthase":   "barrel",
    "Myoglobin":          "globin",
    "Hemoglobin_alpha":   "globin",
    "DHFR":               "enzyme_active",
    "Streptavidin":       "enzyme_active",
    "T4_lysozyme":        "enzyme_active",
    "HEWL":               "enzyme_active",
    "AdK_open":           "allosteric",
    "HIV_protease":       "enzyme_active",
}
