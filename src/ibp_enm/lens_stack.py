"""LensStack composition — mix-and-match post-hoc scoring lenses.

Replaces the 4-deep inheritance tower
(``MetaFickBalancer → EnzymeLensSynthesis → HingeLensSynthesis
→ SizeAwareHingeLens``) with a flat composition:

::

    MetaFickBalancer  (base synthesizer — consensus/disagreement/context)
        ↓ scores
    LensStack([EnzymeLens, HingeLens, BarrelPenaltyLens])
        ↓ adjusted scores

Each :class:`Lens` has:

* ``should_activate(scores, profiles, context)`` — gate check
* ``apply(scores, profiles, context)`` — mutate scores, return trace

Lenses can be mixed, matched, reordered, or A/B-tested without
subclassing.

Usage
-----
>>> from ibp_enm.lens_stack import build_default_stack, LensStackSynthesizer
>>> synth = LensStackSynthesizer(evals=evals, evecs=evecs,
...                               domain_labels=dl, contacts=ct)
>>> result = synth.synthesize_identity(profiles, meta_state)

Or build a custom stack:

>>> from ibp_enm.lens_stack import LensStack, EnzymeLens, HingeLens
>>> stack = LensStack([EnzymeLens(evals=evals, evecs=evecs)])
>>> # No hinge lens, no barrel penalty — enzyme only

Historical notes
----------------
Lens extraction from synthesis.py hierarchy.
EnzymeLens: D110 (Enzyme Lens Experiment).
HingeLens: D111 (Multi-Mode Hinge Analysis).
BarrelPenaltyLens: D113 (Barrel Over-Prediction Fix).
LensStack composition: v0.5.0.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np

from .archetypes import ARCHETYPE_EXPECTATIONS
from .instruments import ThermoReactionProfile
from .thresholds import ThresholdRegistry, DEFAULT_THRESHOLDS
from .thermodynamics import (
    per_residue_entropy_contribution,
    entropy_asymmetry_score,
    multimode_ipr,
    hinge_occupation_ratio,
    domain_stiffness_asymmetry,
)
from .functional_sites import FunctionalSiteResolver, FunctionalAnnotation

__all__ = [
    "Lens",
    "LensTrace",
    "LensStack",
    "EnzymeLens",
    "HingeLens",
    "BarrelPenaltyLens",
    "AllostericLens",
    "FlowGrammarLens",
    "LensStackSynthesizer",
    "build_default_stack",
    "DEFAULT_THRESHOLDS",
]


# ═══════════════════════════════════════════════════════════════════
# LensTrace — per-lens audit record
# ═══════════════════════════════════════════════════════════════════

@dataclass
class LensTrace:
    """Record of one lens's activation within the stack."""
    lens_name: str
    activated: bool
    boost: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════
# Lens protocol — the composable unit
# ═══════════════════════════════════════════════════════════════════

@runtime_checkable
class Lens(Protocol):
    """Protocol for a post-hoc scoring lens.

    A lens inspects the current scores and carving profiles,
    decides whether to activate, and if so, adjusts scores.
    """

    @property
    def name(self) -> str:
        """Human-readable lens name."""
        ...

    def should_activate(
        self,
        scores: Dict[str, float],
        profiles: List[ThermoReactionProfile],
        context: Dict[str, Any],
    ) -> bool:
        """Return True if this lens should fire."""
        ...

    def apply(
        self,
        scores: Dict[str, float],
        profiles: List[ThermoReactionProfile],
        context: Dict[str, Any],
    ) -> Tuple[Dict[str, float], LensTrace]:
        """Apply the lens adjustment to scores.

        Parameters
        ----------
        scores : dict
            Current ``{archetype: score}`` dict (mutable copy).
        profiles : list of ThermoReactionProfile
            The 7 carving profiles.
        context : dict
            Shared context (evals, evecs, domain_labels, contacts, etc.).

        Returns
        -------
        (scores, trace)
            Updated scores dict and a :class:`LensTrace` record.
        """
        ...


# ═══════════════════════════════════════════════════════════════════
# LensStack — ordered composition
# ═══════════════════════════════════════════════════════════════════

class LensStack:
    """Ordered sequence of lenses applied after base synthesis.

    Parameters
    ----------
    lenses : sequence of Lens
        Lenses to apply in order.
    """

    def __init__(self, lenses: Sequence[Lens] = ()):
        self._lenses: List[Lens] = list(lenses)

    @property
    def lenses(self) -> Tuple[Lens, ...]:
        """Immutable view of the lens list."""
        return tuple(self._lenses)

    def __len__(self) -> int:
        return len(self._lenses)

    def __iter__(self):
        return iter(self._lenses)

    def __repr__(self) -> str:
        names = [l.name for l in self._lenses]
        return f"LensStack({names})"

    def apply(
        self,
        scores: Dict[str, float],
        profiles: List[ThermoReactionProfile],
        context: Dict[str, Any],
    ) -> Tuple[Dict[str, float], List[LensTrace]]:
        """Run all lenses in order, collecting traces.

        Parameters
        ----------
        scores : dict
            ``{archetype: score}`` from base synthesis.
        profiles : list of ThermoReactionProfile
        context : dict
            Shared context dict.

        Returns
        -------
        (final_scores, traces)
        """
        traces: List[LensTrace] = []
        current = dict(scores)  # work on a copy

        for lens in self._lenses:
            if lens.should_activate(current, profiles, context):
                current, trace = lens.apply(current, profiles, context)
                traces.append(trace)
            else:
                traces.append(LensTrace(
                    lens_name=lens.name,
                    activated=False,
                ))

        return current, traces

    def with_lens(self, lens: Lens) -> "LensStack":
        """Return a new stack with an additional lens appended."""
        return LensStack(list(self._lenses) + [lens])

    def without(self, lens_name: str) -> "LensStack":
        """Return a new stack with the named lens removed."""
        return LensStack([l for l in self._lenses if l.name != lens_name])

    def replace(self, lens_name: str, new_lens: Lens) -> "LensStack":
        """Return a new stack with the named lens replaced."""
        return LensStack([
            new_lens if l.name == lens_name else l
            for l in self._lenses
        ])


# ═══════════════════════════════════════════════════════════════════
# Normalisation helper
# ═══════════════════════════════════════════════════════════════════

def _renormalise(scores: Dict[str, float],
                 floor: float = 0.01) -> Dict[str, float]:
    """Renormalise scores with a floor (default 0.01)."""
    total = sum(max(floor, v) for v in scores.values())
    return {k: max(floor, v) / total for k, v in scores.items()}


# ═══════════════════════════════════════════════════════════════════
# EnzymeLens — extracted from EnzymeLensSynthesis (D110)
# ═══════════════════════════════════════════════════════════════════

class EnzymeLens:
    """Enzyme-specific entropy-asymmetry lens (D110).

    Targets the enzyme-vs-allosteric confusion.  Activates only when
    both enzyme and allosteric are in the top 3 archetypes and their
    scores are close (< 0.15 gap or top-2 gap < 0.10).

    Signal families:
    1. IPR (inverse participation ratio)
    2. Algebraic instrument enzyme vote
    3. Entropy asymmetry (gini, cv, top5%)
    4. Fragile instrument (high IPR + high reversibility)
    """

    def __init__(
        self,
        evals: Optional[np.ndarray] = None,
        evecs: Optional[np.ndarray] = None,
        thresholds: Optional[ThresholdRegistry] = None,
    ):
        self.initial_evals = evals
        self.initial_evecs = evecs
        self._t = thresholds or DEFAULT_THRESHOLDS

    @property
    def name(self) -> str:
        return "enzyme_lens"

    def should_activate(
        self,
        scores: Dict[str, float],
        profiles: List[ThermoReactionProfile],
        context: Dict[str, Any],
    ) -> bool:
        sorted_by_score = sorted(scores.items(), key=lambda x: -x[1])
        enzyme_in_top3 = "enzyme_active" in [a for a, _ in sorted_by_score[:3]]
        allosteric_in_top3 = "allosteric" in [a for a, _ in sorted_by_score[:3]]
        if not (enzyme_in_top3 and allosteric_in_top3):
            return False

        t = self._t
        is_close_call = (len(sorted_by_score) >= 2
                         and sorted_by_score[0][1] - sorted_by_score[1][1]
                         < t["enzyme_lens.close_call_gap"])
        enzyme_score = scores.get("enzyme_active", 0)
        allosteric_score = scores.get("allosteric", 0)
        enzyme_allosteric_close = (
            abs(enzyme_score - allosteric_score)
            < t["enzyme_lens.ea_proximity_gap"])

        return is_close_call or enzyme_allosteric_close

    def apply(
        self,
        scores: Dict[str, float],
        profiles: List[ThermoReactionProfile],
        context: Dict[str, Any],
    ) -> Tuple[Dict[str, float], LensTrace]:
        scores = dict(scores)  # copy
        signals = self._compute_signals(profiles, context)
        boost = self._compute_boost(signals, self._t)

        t = self._t
        if boost > 0:
            scores["enzyme_active"] += boost
            scores["allosteric"] -= boost * t["enzyme_lens.allosteric_counter_ratio"]
            scores = _renormalise(scores, t["renorm.floor"])

        trace = LensTrace(
            lens_name=self.name,
            activated=True,
            boost=boost,
            details={"enzyme_signals": signals},
        )
        return scores, trace

    def _compute_signals(
        self,
        profiles: List[ThermoReactionProfile],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract enzyme-specific signals from carving profiles."""
        evals = context.get("evals", self.initial_evals)
        evecs = context.get("evecs", self.initial_evecs)

        all_ipr = [p.mean_ipr for p in profiles]
        mean_ipr = float(np.mean(all_ipr))
        max_ipr = float(np.max(all_ipr))

        # Algebraic instrument vote
        algebraic_vote = None
        algebraic_enzyme_score = 0.0
        for p in profiles:
            if p.instrument == "algebraic":
                algebraic_vote = p.archetype_vote()
                algebraic_enzyme_score = algebraic_vote.get("enzyme_active", 0)
                break

        # Entropy asymmetry from initial spectrum
        asymmetry: dict = {}
        if evals is not None and evecs is not None:
            s_per_res = per_residue_entropy_contribution(evals, evecs)
            asymmetry = entropy_asymmetry_score(s_per_res)

        # Fragile instrument signals
        fragile_ipr = 0.0
        fragile_rev = 0.5
        for p in profiles:
            if p.instrument == "fragile":
                fragile_ipr = p.mean_ipr
                fragile_rev = p.reversible_frac
                break

        return {
            "mean_ipr": mean_ipr,
            "max_ipr": max_ipr,
            "algebraic_enzyme_score": algebraic_enzyme_score,
            "algebraic_vote": algebraic_vote,
            "entropy_asymmetry": asymmetry,
            "fragile_ipr": fragile_ipr,
            "fragile_rev": fragile_rev,
        }

    @staticmethod
    def _compute_boost(
        signals: Dict[str, Any],
        t: Optional[ThresholdRegistry] = None,
    ) -> float:
        """Compute enzyme boost from lens signals.

        Thresholds calibrated on D109/D110 empirical data.
        """
        if t is None:
            t = DEFAULT_THRESHOLDS
        boost = 0.0

        # IPR signal
        if signals["mean_ipr"] > t["enzyme_lens.ipr_strong"]:
            boost += t["enzyme_lens.ipr_strong_boost"]
        elif signals["mean_ipr"] > t["enzyme_lens.ipr_weak"]:
            boost += t["enzyme_lens.ipr_weak_boost"]

        # Algebraic instrument
        if signals["algebraic_enzyme_score"] > t["enzyme_lens.alg_strong"]:
            boost += t["enzyme_lens.alg_strong_boost"]
        elif signals["algebraic_enzyme_score"] > t["enzyme_lens.alg_weak"]:
            boost += t["enzyme_lens.alg_weak_boost"]

        # Entropy asymmetry (D110 discovery)
        asym = signals.get("entropy_asymmetry", {})
        if asym.get("gini", 0) > t["enzyme_lens.gini_thresh"]:
            boost += t["enzyme_lens.gini_boost"]
        if asym.get("cv", 0) > t["enzyme_lens.cv_thresh"]:
            boost += t["enzyme_lens.cv_boost"]
        if asym.get("top5_frac", 0) > t["enzyme_lens.top5_thresh"]:
            boost += t["enzyme_lens.top5_boost"]

        # Fragile: enzyme shows high rev + high IPR
        if (signals["fragile_ipr"] > t["enzyme_lens.fragile_ipr_thresh"]
                and signals["fragile_rev"] > t["enzyme_lens.fragile_rev_thresh"]):
            boost += t["enzyme_lens.fragile_combo_boost"]

        return boost


# ═══════════════════════════════════════════════════════════════════
# HingeLens — extracted from HingeLensSynthesis (D111)
# ═══════════════════════════════════════════════════════════════════

class HingeLens:
    """Multi-mode hinge lens for hinge enzymes (D111).

    Targets proteins whose catalytic site sits at the inter-domain
    hinge (e.g. T4 lysozyme).  Examines modes 2–5 to detect whether
    higher modes still concentrate amplitude at the domain boundary.

    Activation gate:
    1. allosteric in top 2
    2. enzyme_active has non-trivial score (> 0.05)
    3. allosteric score > 0.15
    4. N > 150
    5. hinge_R > 1.0
    """

    def __init__(
        self,
        evals: Optional[np.ndarray] = None,
        evecs: Optional[np.ndarray] = None,
        domain_labels: Optional[np.ndarray] = None,
        contacts: Optional[dict] = None,
        thresholds: Optional[ThresholdRegistry] = None,
    ):
        self.initial_evals = evals
        self.initial_evecs = evecs
        self.domain_labels = domain_labels
        self.contacts_for_hinge = contacts
        self._t = thresholds or DEFAULT_THRESHOLDS

    @property
    def name(self) -> str:
        return "hinge_lens"

    def _compute_hinge_signals(
        self, context: Dict[str, Any],
    ) -> Dict[str, float]:
        """Compute multi-mode hinge observables."""
        evals = context.get("evals", self.initial_evals)
        evecs = context.get("evecs", self.initial_evecs)
        domain_labels = context.get("domain_labels", self.domain_labels)
        contacts = context.get("contacts", self.contacts_for_hinge)

        signals: Dict[str, float] = {}
        if evals is not None and evecs is not None:
            signals["ipr_25"] = multimode_ipr(evecs)
            if domain_labels is not None:
                signals["hinge_r"] = hinge_occupation_ratio(evecs, domain_labels)
            else:
                signals["hinge_r"] = 1.0
            if contacts is not None and domain_labels is not None:
                signals["dom_stiff"] = domain_stiffness_asymmetry(
                    contacts, domain_labels)
            else:
                signals["dom_stiff"] = 0.0
        else:
            signals = {"ipr_25": 0.0, "hinge_r": 1.0, "dom_stiff": 0.0}
        return signals

    def should_activate(
        self,
        scores: Dict[str, float],
        profiles: List[ThermoReactionProfile],
        context: Dict[str, Any],
    ) -> bool:
        sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
        top2_archs = [a for a, _ in sorted_scores[:2]]

        t = self._t
        allosteric_in_top2 = "allosteric" in top2_archs
        enzyme_nontrivial = (scores.get("enzyme_active", 0)
                             > t["hinge_lens.enzyme_nontrivial"])
        allosteric_significant = (scores.get("allosteric", 0)
                                  > t["hinge_lens.allosteric_significant"])
        n_res = profiles[0].n_residues if profiles else 200
        protein_large_enough = n_res > t["hinge_lens.size_gate_n"]

        if not (allosteric_in_top2 and enzyme_nontrivial
                and allosteric_significant and protein_large_enough):
            return False

        hinge_signals = self._compute_hinge_signals(context)
        # Cache for apply()
        self._last_hinge_signals = hinge_signals

        if hinge_signals.get("hinge_r", 1.0) <= t["hinge_lens.hinge_r_gate"]:
            return False

        # D116 gate: dom_stiff — block when domain stiffness is too high
        # (non-enzymes show higher dom_stiff; default 999.0 = no gate)
        if hinge_signals.get("dom_stiff", 0.0) > t["hinge_lens.dom_stiff_max"]:
            return False

        # D116 gate: instrument consensus — require minimum fraction of
        # instruments voting enzyme_active (default 0 = no gate)
        enzyme_vote_min = t["hinge_lens.enzyme_vote_min"]
        if enzyme_vote_min > 0:
            n_enzyme = 0
            n_total = 0
            for p in profiles:
                votes = p.archetype_vote()
                if votes:
                    n_total += 1
                    if max(votes, key=votes.get) == "enzyme_active":
                        n_enzyme += 1
            frac = n_enzyme / n_total if n_total else 0
            if frac < enzyme_vote_min:
                return False

        return True

    def apply(
        self,
        scores: Dict[str, float],
        profiles: List[ThermoReactionProfile],
        context: Dict[str, Any],
    ) -> Tuple[Dict[str, float], LensTrace]:
        scores = dict(scores)
        signals = getattr(self, "_last_hinge_signals", None)
        if signals is None:
            signals = self._compute_hinge_signals(context)

        boost = self._hinge_boost(signals, self._t)

        t = self._t
        if boost > 0:
            scores["enzyme_active"] += boost
            scores["allosteric"] -= boost * t["hinge_lens.allosteric_counter_ratio"]
            scores = _renormalise(scores, t["renorm.floor"])

        trace = LensTrace(
            lens_name=self.name,
            activated=True,
            boost=boost,
            details={"hinge_signals": signals},
        )
        return scores, trace

    @staticmethod
    def _hinge_boost(
        signals: Dict[str, float],
        t: Optional[ThresholdRegistry] = None,
    ) -> float:
        """Compute enzyme boost from hinge occupation ratio.

        Calibrated on D111 empirical data:
        - T4 lysozyme: hinge_R = 1.091 → boost ≈ 0.273
        - AdK:         hinge_R = 0.952 → boost = 0
        """
        if t is None:
            t = DEFAULT_THRESHOLDS
        hinge_r = signals.get("hinge_r", 1.0)
        # D116: cap effective hinge_r to limit extreme boosts
        # (default 999.0 = no cap)
        hinge_r = min(hinge_r, t["hinge_lens.hinge_r_effective_cap"])
        if hinge_r > t["hinge_lens.hinge_r_gate"]:
            excess = hinge_r - t["hinge_lens.hinge_r_gate"]
            return min(t["hinge_lens.boost_cap"],
                       excess * t["hinge_lens.boost_multiplier"])
        return 0.0


# ═══════════════════════════════════════════════════════════════════
# BarrelPenaltyLens — extracted from SizeAwareHingeLens (D113)
# ═══════════════════════════════════════════════════════════════════

class BarrelPenaltyLens:
    """Barrel-penalty lens for large proteins (D113).

    Fires when barrel wins, the protein is large (N > 250), and
    size-normalised scatter is not extremely low.  Checks for
    dumbbell/allosteric signals that would override the barrel call.
    """

    def __init__(
        self,
        domain_labels: Optional[np.ndarray] = None,
        thresholds: Optional[ThresholdRegistry] = None,
    ):
        self.domain_labels = domain_labels
        self._t = thresholds or DEFAULT_THRESHOLDS

    @property
    def name(self) -> str:
        return "barrel_penalty"

    def should_activate(
        self,
        scores: Dict[str, float],
        profiles: List[ThermoReactionProfile],
        context: Dict[str, Any],
    ) -> bool:
        identity = max(scores, key=scores.get)
        if identity != "barrel":
            return False

        t = self._t
        n_residues = profiles[0].n_residues if profiles else 200
        if n_residues <= t["barrel_penalty.size_gate_n"]:
            return False

        all_scatter_norm = float(np.mean(
            [p.scatter_normalised for p in profiles]))
        return all_scatter_norm > t["barrel_penalty.scatter_gate"]

    def apply(
        self,
        scores: Dict[str, float],
        profiles: List[ThermoReactionProfile],
        context: Dict[str, Any],
    ) -> Tuple[Dict[str, float], LensTrace]:
        scores = dict(scores)
        domain_labels = context.get("domain_labels", self.domain_labels)

        t = self._t
        all_db = float(np.mean([p.mean_delta_beta for p in profiles]))
        all_radius = float(np.mean([p.mean_spatial_radius for p in profiles]))

        penalty = 0.0
        boost_target = None

        # Signal 1: high Δβ → probably dumbbell
        if all_db > t["barrel_penalty.db_thresh"]:
            penalty += t["barrel_penalty.db_penalty"]
            boost_target = "dumbbell"

        # Signal 2: high spatial radius → probably allosteric
        if all_radius > t["barrel_penalty.radius_thresh"]:
            penalty += t["barrel_penalty.radius_penalty"]
            if boost_target is None:
                boost_target = "allosteric"

        # Signal 3: multiple domains detected
        if domain_labels is not None and len(set(domain_labels)) > 1:
            n_domains = len(set(domain_labels))
            if n_domains >= int(t["barrel_penalty.domain_count_gate"]):
                penalty += t["barrel_penalty.domain_penalty"]
                if boost_target is None:
                    boost_target = "dumbbell"

        # Signal 4: gap NOT flat under algebraic → not barrel
        for p in profiles:
            if p.instrument == "algebraic":
                if p.gap_flatness < t["barrel_penalty.flatness_thresh"]:
                    penalty += t["barrel_penalty.flatness_penalty"]
                break

        activated = penalty > 0 and boost_target is not None
        if activated:
            scores["barrel"] -= penalty
            scores[boost_target] += penalty * t["barrel_penalty.boost_target_ratio"]
            scores = _renormalise(scores, t["renorm.floor"])

        trace = LensTrace(
            lens_name=self.name,
            activated=activated,
            boost=-penalty if activated else 0.0,
            details={
                "penalty": penalty,
                "boost_target": boost_target,
                "mean_delta_beta": round(all_db, 4),
                "mean_radius": round(all_radius, 4),
            },
        )
        return scores, trace


# ═══════════════════════════════════════════════════════════════════
# AllostericLens — Binding-Site-Aware TE (D128/D129)
# ═══════════════════════════════════════════════════════════════════

class AllostericLens:
    """Allosteric-specific lens using binding-site-aware transfer entropy.

    The allosteric archetype is the hardest to classify because its
    NLP sentence is *"signal-coupled domains; cuts propagate
    unexpectedly"* — it LOOKS like other things until you measure
    directed information flow between functional sites.

    This lens activates when the base synthesis produces an ambiguous
    result involving allosteric (e.g., allosteric scored but not winning,
    or allosteric close to the winner).  It then resolves functional
    annotations (regulatory, chemical, mechanical sites) and computes
    directed TE between them.

    Physics:
      - Allosteric proteins have NET information flow FROM regulatory
        sites TO active/chemical sites (regulatory drives catalysis).
      - Top TE drivers concentrate at regulatory/signalling sites.
      - The TE pathway is asymmetric (reg→act >> act→reg).

    Activation gate:
      1. Allosteric in top 3 (it's on the radar)
      2. Score confusion: top-2 gap < gate threshold, OR allosteric
         close to winner
      3. Eigendecomposition available (need TE computation)

    Signals (from D128):
      1. site_te_regulatory_to_active — TE from reg → chem sites
      2. site_net_reg_to_active — NET flow (should be positive)
      3. site_driver_enrichment_regulatory — drivers at reg sites
      4. site_te_pathway_asymmetry — directional flow
      5. site_functional_te_ratio — TE at sites vs everywhere

    NLP semantic mapping:
      Intent RELATING (protect domain boundaries) +
      Intent BECOMING (sculpt toward separation) =
      Fano line 5 (GROWTH) — the allosteric narrative arc.
      When RELATING amplitude is high in the QuantumCarvingState,
      the system already suspects signal-coupled domains.
      This lens makes that suspicion concrete.

    Historical notes:
      D128 — showed 5/7 allosteric correct with solo 12-feature classifier
      D129 — integration as composable lens in the LensStack
    """

    def __init__(
        self,
        evals: Optional[np.ndarray] = None,
        evecs: Optional[np.ndarray] = None,
        contacts: Optional[dict] = None,
        pdb_id: Optional[str] = None,
        chain: Optional[str] = None,
        n_residues: Optional[int] = None,
        resolver: Optional[FunctionalSiteResolver] = None,
        annotation: Optional[FunctionalAnnotation] = None,
        thresholds: Optional[ThresholdRegistry] = None,
    ):
        self.initial_evals = evals
        self.initial_evecs = evecs
        self.initial_contacts = contacts
        self._pdb_id = pdb_id
        self._chain = chain or "A"
        self._n_residues = n_residues
        self._resolver = resolver
        self._annotation = annotation   # pre-resolved, if available
        self._t = thresholds or DEFAULT_THRESHOLDS

    @property
    def name(self) -> str:
        return "allosteric_lens"

    def should_activate(
        self,
        scores: Dict[str, float],
        profiles: List[ThermoReactionProfile],
        context: Dict[str, Any],
    ) -> bool:
        """Gate: allosteric in top 3 and scores are ambiguous."""
        sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
        top3_archs = [a for a, _ in sorted_scores[:3]]

        if "allosteric" not in top3_archs:
            return False

        # Need eigendecomposition for TE computation
        evals = context.get("evals", self.initial_evals)
        if evals is None:
            return False

        t = self._t

        # Confusion check: is the classification ambiguous?
        top2_gap = (sorted_scores[0][1] - sorted_scores[1][1]
                    if len(sorted_scores) >= 2 else 1.0)
        allo_score = scores.get("allosteric", 0)
        winner_score = sorted_scores[0][1] if sorted_scores else 0

        # Gate 1: top-2 gap is small (genuine confusion)
        is_confused = top2_gap < t["allosteric_lens.confusion_gap"]

        # Gate 2: allosteric is close to the winner
        allo_proximity = (winner_score - allo_score
                          if winner_score > allo_score else 0)
        is_close = allo_proximity < t["allosteric_lens.proximity_gap"]

        # Gate 3: propagative instrument shows high spatial radius
        # (allosteric signature: perturbations propagate far)
        prop_high = False
        for p in profiles:
            if p.instrument == "propagative":
                if p.mean_spatial_radius > t[
                        "allosteric_lens.propagative_radius_gate"]:
                    prop_high = True
                break

        # Activate if confused/close AND has allosteric propagation signal
        return (is_confused or is_close) and prop_high

    def apply(
        self,
        scores: Dict[str, float],
        profiles: List[ThermoReactionProfile],
        context: Dict[str, Any],
    ) -> Tuple[Dict[str, float], LensTrace]:
        """Compute binding-site-aware TE and boost allosteric if confirmed."""
        scores = dict(scores)
        t = self._t

        signals = self._compute_allosteric_signals(profiles, context)

        boost = self._compute_boost(signals, t)

        if boost > 0:
            scores["allosteric"] += boost
            # Counter-suppress the current winner if it's not allosteric
            sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
            current_winner = sorted_scores[0][0]
            if current_winner != "allosteric":
                scores[current_winner] -= boost * t[
                    "allosteric_lens.counter_suppress_ratio"]
            scores = _renormalise(scores, t["renorm.floor"])

        trace = LensTrace(
            lens_name=self.name,
            activated=True,
            boost=boost,
            details={"allosteric_signals": signals},
        )
        return scores, trace

    def _compute_allosteric_signals(
        self,
        profiles: List[ThermoReactionProfile],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compute binding-site-aware TE signals.

        Uses FunctionalSiteResolver to get annotations, then computes
        GNM transfer entropy between functional site categories.
        """
        evals = context.get("evals", self.initial_evals)
        evecs = context.get("evecs", self.initial_evecs)
        contacts = context.get("contacts", self.initial_contacts)
        N = context.get("n_residues", self._n_residues)
        pdb_id = context.get("pdb_id", self._pdb_id)
        chain = context.get("chain", self._chain)

        signals: Dict[str, Any] = {
            "has_annotation": False,
            "n_regulatory": 0,
            "n_chemical": 0,
            "coverage": 0.0,
            "te_reg_to_active": 0.0,
            "te_active_to_reg": 0.0,
            "net_reg_to_active": 0.0,
            "driver_enrichment_reg": 0.0,
            "te_pathway_asymmetry": 0.0,
            "functional_te_ratio": 0.0,
        }

        if evals is None or evecs is None or N is None:
            return signals

        # ── Resolve functional annotation ──
        ann = self._annotation
        if ann is None and pdb_id and self._resolver:
            try:
                ann = self._resolver.resolve(pdb_id, chain=chain,
                                             n_residues=N)
            except Exception:
                pass  # annotation failure is non-fatal
        if ann is None:
            return signals

        # Functional site sets (clamped to valid range)
        reg_set = {r for r in ann.signalling if r < N}
        chem_set = {r for r in ann.chemical if r < N}

        signals["has_annotation"] = True
        signals["n_regulatory"] = len(reg_set)
        signals["n_chemical"] = len(chem_set)
        signals["coverage"] = ann.coverage

        # Minimum site count gate: need enough residues in BOTH
        # categories for statistically meaningful TE computation.
        # With < 3 sites in either set, the mean TE is based on
        # too few residue pairs to be reliable.
        min_sites = int(self._t.get("allosteric_lens.min_site_count", 3))
        if len(reg_set) < min_sites or len(chem_set) < min_sites:
            return signals

        # ── Compute GNM correlation and TE ──
        pos_mask = evals > 1e-8
        pos_evals = evals[pos_mask]
        if len(pos_evals) < 3:
            return signals

        try:
            from .carving import build_laplacian
            lambda2 = pos_evals[0]
            t_star = 1.0 / lambda2

            # GNM correlation at t=0 and t=t*
            C0 = self._gnm_correlation(N, evals, evecs, 0.0)
            Ct = self._gnm_correlation(N, evals, evecs, t_star)

            # Transfer entropy matrix
            TE, NET = self._transfer_entropy_matrix(C0, Ct, N)

            # Driver scores
            driver_scores = np.abs(np.sum(NET, axis=1))

            # ── Site-specific TE signals ──

            # 1. Mean TE from regulatory → chemical
            te_reg_act = self._mean_te_between(TE, reg_set, chem_set)
            signals["te_reg_to_active"] = te_reg_act

            # 2. Mean TE from chemical → regulatory
            te_act_reg = self._mean_te_between(TE, chem_set, reg_set)
            signals["te_active_to_reg"] = te_act_reg

            # 3. NET flow from regulatory → chemical
            net_r2a = self._mean_net_between(NET, reg_set, chem_set)
            signals["net_reg_to_active"] = net_r2a

            # 4. Driver enrichment at regulatory sites
            n_top = max(N // 10, 3)
            top_drivers = set(np.argsort(driver_scores)[-n_top:])
            obs_reg = (len(top_drivers & reg_set)
                       / max(len(top_drivers), 1))
            exp_reg = len(reg_set) / N
            signals["driver_enrichment_reg"] = (
                obs_reg / max(exp_reg, 1e-10)
            )

            # 5. TE pathway asymmetry
            te_sum = te_reg_act + te_act_reg
            if te_sum > 1e-15:
                signals["te_pathway_asymmetry"] = (
                    abs(te_reg_act - te_act_reg) / te_sum
                )

            # 6. Functional TE ratio
            all_func = {r for r in ann.all_functional if r < N}
            if len(all_func) >= 2:
                triu_i, triu_j = np.triu_indices(N, k=1)
                mean_te_all = float(np.mean(
                    0.5 * (TE[triu_i, triu_j] + TE[triu_j, triu_i])))
                if mean_te_all > 1e-15:
                    func_list = sorted(all_func)
                    func_vals = []
                    for ii in range(len(func_list)):
                        for jj in range(ii + 1, len(func_list)):
                            a, b = func_list[ii], func_list[jj]
                            func_vals.append(
                                0.5 * (TE[a, b] + TE[b, a]))
                    if func_vals:
                        signals["functional_te_ratio"] = (
                            float(np.mean(func_vals)) / mean_te_all
                        )

        except Exception:
            pass  # TE computation failure is non-fatal

        return signals

    @staticmethod
    def _gnm_correlation(N: int, evals: np.ndarray,
                         evecs: np.ndarray, t: float) -> np.ndarray:
        """GNM cross-correlation at pseudo-time t.

        C_ij(t) = Σ_k (1/λ_k) · exp(-λ_k·t) · u_ki · u_kj
        """
        pos = evals > 1e-8
        lam = evals[pos]
        U = evecs[:, pos]
        if t > 0:
            weights = (1.0 / lam) * np.exp(-lam * t)
        else:
            weights = 1.0 / lam
        C = (U * weights[None, :]) @ U.T
        return C

    @staticmethod
    def _transfer_entropy_matrix(C0: np.ndarray, Ct: np.ndarray,
                                 N: int) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorised TE and NET matrices.

        TE_{j→i}(t) = -½ ln(1 - α_ij²/β_ij)
        where α_ij = C_ij(t) - C_ij(0) · C_jj(t)/C_jj(0)
              β_ij = C_ii(t) · (1 - C_ij(0)²/(C_ii(0)·C_jj(0)))
        """
        diag0 = np.diag(C0).copy()
        diag0[diag0 < 1e-12] = 1e-12
        diagt = np.diag(Ct).copy()
        diagt[diagt < 1e-12] = 1e-12

        alpha = Ct - (C0 / diag0[None, :]) * diagt[:, None]

        r0 = C0 / np.sqrt(np.outer(diag0, diag0))
        r0 = np.clip(r0, -0.999, 0.999)
        beta = diagt[:, None] * (1.0 - r0 ** 2)
        beta = np.maximum(beta, 1e-15)

        ratio = np.clip(alpha ** 2 / beta, 0, 0.999)
        TE = -0.5 * np.log(1.0 - ratio)
        np.fill_diagonal(TE, 0)

        NET = TE.T - TE  # NET(i,j) = TE_{i→j} - TE_{j→i}
        return TE, NET

    @staticmethod
    def _mean_te_between(TE: np.ndarray,
                         src: set, dst: set) -> float:
        """Mean TE from src → dst (directed)."""
        if not src or not dst:
            return 0.0
        total = 0.0
        count = 0
        for i in src:
            for j in dst:
                if i != j:
                    total += TE[j, i]
                    count += 1
        return total / max(count, 1)

    @staticmethod
    def _mean_net_between(NET: np.ndarray,
                          src: set, dst: set) -> float:
        """Mean NET flow from src → dst."""
        if not src or not dst:
            return 0.0
        total = 0.0
        count = 0
        for i in src:
            for j in dst:
                if i != j:
                    total += NET[i, j]
                    count += 1
        return total / max(count, 1)

    @staticmethod
    def _compute_boost(signals: Dict[str, Any],
                       t: ThresholdRegistry) -> float:
        """Compute allosteric boost from binding-site-aware TE signals.

        We use a multi-signal approach calibrated on D128 empirical data:
        each signal that exceeds its threshold contributes to the boost.
        The compound effect of multiple positive signals gives higher
        confidence.

        NLP interpretation:
          Each signal is a "word" in the allosteric sentence.
          Multiple positive signals compose a "phrase" that affirms:
          "This protein has directed regulatory → catalytic information flow."
        """
        if not signals.get("has_annotation"):
            return 0.0

        # ── NET direction gate ──
        # The strongest single discriminator from D129 calibration:
        # allosteric proteins have POSITIVE NET (regulatory drives
        # chemical), enzymes have NEGATIVE NET.  If NET is solidly
        # negative, this is NOT allosteric — abort immediately.
        net = signals.get("net_reg_to_active", 0.0)
        if net < -t["allosteric_lens.net_negative_gate"]:
            return 0.0

        boost = 0.0

        # Signal 1: TE from regulatory → chemical sites
        # (allosteric: high, others: low/zero)
        if signals["te_reg_to_active"] > t[
                "allosteric_lens.te_reg_act_strong"]:
            boost += t["allosteric_lens.te_reg_act_strong_boost"]
        elif signals["te_reg_to_active"] > t[
                "allosteric_lens.te_reg_act_weak"]:
            boost += t["allosteric_lens.te_reg_act_weak_boost"]

        # Signal 2: Positive NET flow regulatory → chemical
        # (allosteric: positive = regulatory DRIVES catalysis)
        if signals["net_reg_to_active"] > t[
                "allosteric_lens.net_positive_thresh"]:
            boost += t["allosteric_lens.net_positive_boost"]

        # Signal 3: Driver enrichment at regulatory sites
        # (allosteric: TE drivers concentrate at regulatory residues)
        if signals["driver_enrichment_reg"] > t[
                "allosteric_lens.driver_enrich_strong"]:
            boost += t["allosteric_lens.driver_enrich_strong_boost"]
        elif signals["driver_enrichment_reg"] > t[
                "allosteric_lens.driver_enrich_weak"]:
            boost += t["allosteric_lens.driver_enrich_weak_boost"]

        # Signal 4: Pathway asymmetry
        # (allosteric: strongly directional, reg→act >> act→reg)
        if signals["te_pathway_asymmetry"] > t[
                "allosteric_lens.asymmetry_thresh"]:
            boost += t["allosteric_lens.asymmetry_boost"]

        # Signal 5: Functional TE ratio
        # (allosteric: TE concentrated at functional sites relative
        #  to the overall TE — functional sites are the information
        #  highway, not random background)
        if signals["functional_te_ratio"] > t[
                "allosteric_lens.func_te_ratio_thresh"]:
            boost += t["allosteric_lens.func_te_ratio_boost"]

        # Compound bonus: if ≥3 signals fire, this is a strong
        # allosteric sentence — boost further
        signal_count = sum([
            signals["te_reg_to_active"] > t[
                "allosteric_lens.te_reg_act_weak"],
            signals["net_reg_to_active"] > t[
                "allosteric_lens.net_positive_thresh"],
            signals["driver_enrichment_reg"] > t[
                "allosteric_lens.driver_enrich_weak"],
            signals["te_pathway_asymmetry"] > t[
                "allosteric_lens.asymmetry_thresh"],
        ])
        if signal_count >= int(t["allosteric_lens.compound_min_signals"]):
            boost += t["allosteric_lens.compound_bonus"]

        # Cap total boost
        boost = min(boost, t["allosteric_lens.boost_cap"])

        return boost


# ═══════════════════════════════════════════════════════════════════
# Shared TE helpers (used by AllostericLens + FlowGrammarLens)
# ═══════════════════════════════════════════════════════════════════

def _gnm_correlation_matrix(
    evals: np.ndarray,
    evecs: np.ndarray,
    t: float,
) -> np.ndarray:
    """GNM cross-correlation at pseudo-time *t*.

    C_ij(t) = Σ_k (1/λ_k) · exp(-λ_k·t) · u_ki · u_kj

    Only uses positive eigenvalues (rigid-body zero modes excluded).
    """
    pos = evals > 1e-8
    lam = evals[pos]
    U = evecs[:, pos]
    if t > 0:
        weights = (1.0 / lam) * np.exp(-lam * t)
    else:
        weights = 1.0 / lam
    return (U * weights[None, :]) @ U.T


def _te_and_net_matrices(
    C0: np.ndarray,
    Ct: np.ndarray,
    N: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorised transfer entropy and net flow matrices.

    TE_{j→i}(t) = -½ ln(1 - α²/β)  where
      α_ij = C_ij(t) - C_ij(0)·C_jj(t)/C_jj(0)
      β_ij = C_ii(t)·(1 - r₀²)
      r₀   = C_ij(0) / √(C_ii(0)·C_jj(0))

    Returns ``(TE, NET)`` where ``NET(i,j) = TE(i→j) - TE(j→i)``.
    """
    diag0 = np.diag(C0).copy()
    diag0[diag0 < 1e-12] = 1e-12
    diagt = np.diag(Ct).copy()
    diagt[diagt < 1e-12] = 1e-12

    alpha = Ct - (C0 / diag0[None, :]) * diagt[:, None]

    r0 = C0 / np.sqrt(np.outer(diag0, diag0))
    r0 = np.clip(r0, -0.999, 0.999)
    beta = diagt[:, None] * (1.0 - r0 ** 2)
    beta = np.maximum(beta, 1e-15)

    ratio = np.clip(alpha ** 2 / beta, 0, 0.999)
    TE = -0.5 * np.log(1.0 - ratio)
    np.fill_diagonal(TE, 0)

    NET = TE.T - TE
    return TE, NET


# ═══════════════════════════════════════════════════════════════════
# FlowGrammarLens — Pre-Carving TE Flow Vocabulary (D130)
# ═══════════════════════════════════════════════════════════════════

def _flow_word(te_asymmetry: float, cross_enrichment: float) -> str:
    """Classify flow pattern into an NLP word.

    Returns one of: ``DIRECTING``, ``CHANNELING``, ``DIFFUSING``.
    """
    if cross_enrichment > 1.0 and te_asymmetry > 0.9:
        return "DIRECTING"
    elif cross_enrichment > 0.85:
        return "CHANNELING"
    return "DIFFUSING"


class FlowGrammarLens:
    """Pre-carving information flow lens (D130).

    Computes transfer entropy features from the *intact*
    eigendecomposition to detect allosteric-like information flow
    patterns **before** carving.

    Unlike :class:`AllostericLens` (which requires functional site
    annotations from PDB), ``FlowGrammarLens`` works on raw spectral
    data using the Fiedler partition to define domain structure.

    Three flow observables:

    1. **TE asymmetry** — ``std(NET) / mean(TE)``: directed information
       bias across the entire network.
    2. **Cross-domain enrichment** — ``|NET|_cross / |NET|_intra``:
       how much information flow crosses the Fiedler partition vs
       stays within domains.
    3. **Driver-sensor ratio** — ``n_drivers / n_sensors``: whether
       the network has a clear driving core.

    Three NLP flow words:

    * **DIRECTING** — asymmetric cross-domain flow (allosteric
      indicator).
    * **CHANNELING** — strong driver hierarchy.
    * **DIFFUSING** — symmetric or weak flow (default).

    The lens applies a targeted boost to the allosteric score only
    (positive for DIRECTING flow, negative for anti-allosteric
    DIFFUSING patterns).  Other archetype scores are not directly
    modified.  This implements a key D130 finding: flow features
    work as *targeted signals*, not symmetric archetype
    discriminators.

    Physics
    -------
    Allosteric proteins route information **across** their domain
    boundary (high cross-enrichment) with directional bias (high TE
    asymmetry).  Non-allosteric proteins with similar structural
    profiles (e.g., CheY looks like enzyme_active structurally) lack
    this cross-domain information routing.

    NLP semantic mapping
    --------------------
    The flow vocabulary extends the existing NLP layers:

    * **Bus vectors** describe WHAT the protein looks like.
    * **Fick α** describes HOW it diffuses.
    * **Flow words** describe WHERE information goes in the intact
      network.

    Together they compose the diagnostic sentence::

        "This [archetype] protein [DIRECTS/CHANNELS/DIFFUSES]
         information [across/within] domain boundaries."

    Activation gate
    ---------------
    1. Allosteric in top 3 (it's on the radar).
    2. Score confusion: top-2 gap < threshold **or** allosteric
       close to winner.
    3. Eigendecomposition available.

    Historical notes
    ----------------
    D130 — FlowGrammar experiment, CheY rescued from
    ``enzyme_active`` via DIRECTING flow (X/I=1.066, asym=1.129).
    Key finding: flow features work as targeted signals, not
    symmetric archetype discriminators.
    """

    def __init__(
        self,
        evals: Optional[np.ndarray] = None,
        evecs: Optional[np.ndarray] = None,
        thresholds: Optional[ThresholdRegistry] = None,
    ):
        self.initial_evals = evals
        self.initial_evecs = evecs
        self._t = thresholds or DEFAULT_THRESHOLDS

    @property
    def name(self) -> str:
        return "flow_grammar_lens"

    # ── Gate ────────────────────────────────────────────────────

    def should_activate(
        self,
        scores: Dict[str, float],
        profiles: List[ThermoReactionProfile],
        context: Dict[str, Any],
    ) -> bool:
        """Gate: allosteric in top 3, evals available, scores confused."""
        sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
        top3_archs = [a for a, _ in sorted_scores[:3]]

        if "allosteric" not in top3_archs:
            return False

        evals = context.get("evals", self.initial_evals)
        evecs = context.get("evecs", self.initial_evecs)
        if evals is None or evecs is None:
            return False

        t = self._t

        top2_gap = (sorted_scores[0][1] - sorted_scores[1][1]
                    if len(sorted_scores) >= 2 else 1.0)
        allo_score = scores.get("allosteric", 0)
        winner_score = sorted_scores[0][1] if sorted_scores else 0
        allo_proximity = (winner_score - allo_score
                          if winner_score > allo_score else 0)

        is_confused = top2_gap < t["flow_grammar_lens.confusion_gap"]
        is_close = allo_proximity < t["flow_grammar_lens.proximity_gap"]

        return is_confused or is_close

    # ── Apply ───────────────────────────────────────────────────

    def apply(
        self,
        scores: Dict[str, float],
        profiles: List[ThermoReactionProfile],
        context: Dict[str, Any],
    ) -> Tuple[Dict[str, float], LensTrace]:
        """Compute pre-carving flow features and adjust allosteric score."""
        scores = dict(scores)
        t = self._t

        evals = context.get("evals", self.initial_evals)
        evecs = context.get("evecs", self.initial_evecs)
        N = context.get("n_residues")
        if N is None and evecs is not None:
            N = evecs.shape[0]

        flow_signals = self._compute_flow_signals(evals, evecs, N)
        boost = self._compute_flow_boost(flow_signals, t)

        if abs(boost) > 1e-6:
            scores["allosteric"] += boost
            if boost > 0:
                sorted_scores = sorted(
                    scores.items(), key=lambda x: -x[1])
                current_winner = sorted_scores[0][0]
                if current_winner != "allosteric":
                    scores[current_winner] -= boost * t[
                        "flow_grammar_lens.counter_suppress_ratio"]
            scores = _renormalise(scores, t["renorm.floor"])

        trace = LensTrace(
            lens_name=self.name,
            activated=True,
            boost=boost,
            details={"flow_signals": flow_signals},
        )
        return scores, trace

    # ── Signal computation ──────────────────────────────────────

    @staticmethod
    def _compute_flow_signals(
        evals: Optional[np.ndarray],
        evecs: Optional[np.ndarray],
        N: Optional[int],
    ) -> Dict[str, Any]:
        """Compute pre-carving TE flow observables.

        Returns a dict with:
        - ``te_asymmetry``: std(NET) / mean(TE)
        - ``cross_enrichment``: |NET|_cross / |NET|_intra  (Fiedler)
        - ``driver_sensor_ratio``: n_drivers / n_sensors
        - ``flow_word``: DIRECTING / CHANNELING / DIFFUSING
        """
        signals: Dict[str, Any] = {
            "te_asymmetry": 0.0,
            "cross_enrichment": 0.0,
            "driver_sensor_ratio": 0.0,
            "flow_word": "DIFFUSING",
        }

        if evals is None or evecs is None or N is None or N < 3:
            return signals

        pos_idx = np.where(evals > 1e-8)[0]
        if len(pos_idx) < 2:
            return signals

        try:
            lambda2 = evals[pos_idx[0]]
            t_star = 1.0 / lambda2

            C0 = _gnm_correlation_matrix(evals, evecs, 0.0)
            Ct = _gnm_correlation_matrix(evals, evecs, t_star)
            TE, NET = _te_and_net_matrices(C0, Ct, N)

            # 1. TE asymmetry: std(NET) / mean(TE)
            te_positive = TE[TE > 0]
            te_mean = (float(np.mean(te_positive))
                       if len(te_positive) > 0 else 1e-15)
            te_asym = float(np.std(NET)) / (te_mean + 1e-15)

            # 2. Cross-domain enrichment via Fiedler partition
            fiedler = evecs[:, pos_idx[0]]
            signs = np.sign(fiedler)
            cross_mask = signs[:, None] != signs[None, :]
            triu_i, triu_j = np.triu_indices(N, k=1)
            net_abs_triu = np.abs(NET[triu_i, triu_j])
            cross_triu = cross_mask[triu_i, triu_j]

            cross_mean = (float(np.mean(net_abs_triu[cross_triu]))
                          if np.any(cross_triu) else 0.0)
            intra_mean = (float(np.mean(net_abs_triu[~cross_triu]))
                          if np.any(~cross_triu) else 1e-15)
            cross_enrich = cross_mean / (intra_mean + 1e-15)

            # 3. Driver-sensor ratio
            row_sums = NET.sum(axis=1)
            n_drv = int(np.sum(row_sums > 0.01))
            n_sns = int(np.sum(row_sums < -0.01))
            drv_ratio = n_drv / (n_sns + 1e-10)

            signals["te_asymmetry"] = te_asym
            signals["cross_enrichment"] = cross_enrich
            signals["driver_sensor_ratio"] = drv_ratio
            signals["flow_word"] = _flow_word(te_asym, cross_enrich)

        except Exception:
            pass  # TE computation failure is non-fatal

        return signals

    # ── Boost computation ───────────────────────────────────────

    @staticmethod
    def _compute_flow_boost(
        signals: Dict[str, Any],
        t: ThresholdRegistry,
    ) -> float:
        """Compute allosteric boost/penalty from flow signals.

        Returns a signed value:

        * ``> 0`` → evidence FOR allosteric (DIRECTING flow).
        * ``< 0`` → evidence AGAINST allosteric (anti-allosteric).
        * ``= 0`` → neutral / insufficient signal.

        NLP interpretation:

        * DIRECTING + cross-domain → *"This protein DIRECTS
          information across its domain boundary — a hallmark of
          allosteric coupling."*
        * DIFFUSING + intra-domain → *"Information DIFFUSES
          symmetrically; no allosteric signal detected."*
        """
        boost = 0.0
        cross_enrich = signals.get("cross_enrichment", 0.0)
        te_asym = signals.get("te_asymmetry", 0.0)
        ds_ratio = signals.get("driver_sensor_ratio", 0.0)

        # Cross-domain enrichment: strongest allosteric signal
        if cross_enrich > t["flow_grammar_lens.cross_enrich_boost_thresh"]:
            boost += t["flow_grammar_lens.cross_enrich_boost"]
        elif cross_enrich < t[
                "flow_grammar_lens.cross_enrich_penalty_thresh"]:
            boost += t["flow_grammar_lens.cross_enrich_penalty"]

        # TE asymmetry: directed information flow
        if te_asym > t["flow_grammar_lens.te_asym_boost_thresh"]:
            boost += t["flow_grammar_lens.te_asym_boost"]

        # Driver-sensor ratio: passive ≠ allosteric
        if ds_ratio < t["flow_grammar_lens.ds_penalty_thresh"]:
            boost += t["flow_grammar_lens.ds_penalty"]

        # Compound: BOTH cross-domain AND high asymmetry
        if (cross_enrich > t[
                "flow_grammar_lens.compound_enrich_thresh"]
                and te_asym > t[
                    "flow_grammar_lens.compound_asym_thresh"]):
            boost += t["flow_grammar_lens.compound_bonus"]

        # Clamp
        boost = max(t["flow_grammar_lens.penalty_floor"], boost)
        boost = min(t["flow_grammar_lens.boost_cap"], boost)

        return boost


# ═══════════════════════════════════════════════════════════════════
# Factory — default stack
# ═══════════════════════════════════════════════════════════════════

def build_default_stack(
    evals: Optional[np.ndarray] = None,
    evecs: Optional[np.ndarray] = None,
    domain_labels: Optional[np.ndarray] = None,
    contacts: Optional[dict] = None,
    thresholds: Optional[ThresholdRegistry] = None,
    *,
    pdb_id: Optional[str] = None,
    chain: Optional[str] = None,
    n_residues: Optional[int] = None,
    resolver: Optional[FunctionalSiteResolver] = None,
    annotation: Optional[FunctionalAnnotation] = None,
    include_allosteric: bool = True,
    include_flow_grammar: bool = False,
) -> LensStack:
    """Build the default lens stack (D110 + D111 + D113 + D129).

    .. note:: FlowGrammarLens (D130) is disabled by default.
       D132 showed it causes 7 regressions (all non-allosteric proteins
       incorrectly flipped to allosteric) with 0 independent improvements.
       Pass ``include_flow_grammar=True`` to re-enable for experimentation.

    Equivalent to the old ``SizeAwareHingeLens`` inheritance tower
    but composable.

    Parameters
    ----------
    evals, evecs : ndarray, optional
        Laplacian eigenvalues/eigenvectors (enables enzyme, hinge,
        flow grammar, and allosteric lenses).
    domain_labels : ndarray, optional
        Domain assignment per residue (enables hinge + barrel lenses).
    contacts : dict, optional
        Contact map (enables hinge lens domain stiffness).
    thresholds : ThresholdRegistry, optional
        Override threshold values (defaults to :data:`DEFAULT_THRESHOLDS`).
    pdb_id : str, optional
        PDB identifier (enables allosteric lens annotation lookup).
    chain : str, optional
        Chain identifier (default "A").
    n_residues : int, optional
        Number of residues (enables allosteric lens).
    resolver : FunctionalSiteResolver, optional
        Pre-built resolver (avoids re-creation per protein).
    annotation : FunctionalAnnotation, optional
        Pre-resolved annotation (skips API lookup).
    include_allosteric : bool
        Include the AllostericLens in the stack (default True).
    include_flow_grammar : bool
        Include the FlowGrammarLens in the stack (default False).
        D132 showed FGL causes net-negative accuracy.

    Returns
    -------
    LensStack
        ``[EnzymeLens, HingeLens, BarrelPenaltyLens,
        AllostericLens]``  (FlowGrammarLens excluded by default)
    """
    t = thresholds
    lenses: list = [
        EnzymeLens(evals=evals, evecs=evecs, thresholds=t),
        HingeLens(evals=evals, evecs=evecs,
                  domain_labels=domain_labels, contacts=contacts,
                  thresholds=t),
        BarrelPenaltyLens(domain_labels=domain_labels, thresholds=t),
    ]
    if include_flow_grammar:
        lenses.append(FlowGrammarLens(
            evals=evals, evecs=evecs, thresholds=t,
        ))
    if include_allosteric:
        lenses.append(AllostericLens(
            evals=evals, evecs=evecs, contacts=contacts,
            pdb_id=pdb_id, chain=chain, n_residues=n_residues,
            resolver=resolver, annotation=annotation,
            thresholds=t,
        ))
    return LensStack(lenses)


# ═══════════════════════════════════════════════════════════════════
# LensStackSynthesizer — MetaFickBalancer + LensStack
# ═══════════════════════════════════════════════════════════════════

class LensStackSynthesizer:
    """Flat composition: MetaFickBalancer + configurable LensStack.

    Drop-in replacement for ``SizeAwareHingeLens`` but without the
    4-deep inheritance tower.  Lenses can be swapped, reordered,
    or A/B tested at construction time.

    Parameters
    ----------
    stack : LensStack, optional
        Custom lens stack.  Defaults to :func:`build_default_stack`.
    evals, evecs, domain_labels, contacts :
        Passed to :func:`build_default_stack` if *stack* is None.
    **balancer_kwargs :
        Forwarded to :class:`MetaFickBalancer` (w1, w2, w3, beta0).
    """

    def __init__(
        self,
        stack: Optional[LensStack] = None,
        evals: Optional[np.ndarray] = None,
        evecs: Optional[np.ndarray] = None,
        domain_labels: Optional[np.ndarray] = None,
        contacts: Optional[dict] = None,
        thresholds: Optional[ThresholdRegistry] = None,
        *,
        pdb_id: Optional[str] = None,
        chain: Optional[str] = None,
        n_residues: Optional[int] = None,
        resolver: Optional[FunctionalSiteResolver] = None,
        annotation: Optional[FunctionalAnnotation] = None,
        **balancer_kwargs,
    ):
        self._t = thresholds or DEFAULT_THRESHOLDS
        from .synthesis import MetaFickBalancer
        self.balancer = MetaFickBalancer(**balancer_kwargs)
        self.stack = stack if stack is not None else build_default_stack(
            evals=evals, evecs=evecs,
            domain_labels=domain_labels, contacts=contacts,
            thresholds=self._t,
            pdb_id=pdb_id, chain=chain, n_residues=n_residues,
            resolver=resolver, annotation=annotation,
        )
        self._context: Dict[str, Any] = {
            "evals": evals,
            "evecs": evecs,
            "domain_labels": domain_labels,
            "contacts": contacts,
            "pdb_id": pdb_id,
            "chain": chain,
            "n_residues": n_residues,
        }

    @property
    def history(self) -> list:
        return self.balancer.history

    def compute_meta_fick_state(
        self,
        carver_votes: List[Dict[str, float]],
    ) -> Dict:
        """Delegate to the inner MetaFickBalancer."""
        return self.balancer.compute_meta_fick_state(carver_votes)

    def synthesize_identity(
        self,
        carver_profiles: List[ThermoReactionProfile],
        meta_state: Dict,
    ) -> Dict:
        """Synthesize identity: base MetaFickBalancer + LensStack.

        Returns the same dict structure as
        ``SizeAwareHingeLens.synthesize_identity()`` for backwards
        compatibility, plus a ``"lens_traces"`` key.
        """
        # Base synthesis (consensus + disagreement + context boost)
        result = self.balancer.synthesize_identity(
            carver_profiles, meta_state)
        scores = dict(result["scores"])

        # Run the lens stack
        final_scores, traces = self.stack.apply(
            scores, carver_profiles, self._context)

        # Update result
        result["scores"] = final_scores
        result["identity"] = max(final_scores, key=final_scores.get)
        result["lens_traces"] = traces

        # Backwards-compatible keys
        for trace in traces:
            if trace.lens_name == "enzyme_lens":
                result["enzyme_lens_activated"] = trace.activated
                result["enzyme_lens_boost"] = trace.boost
                if trace.activated:
                    result["enzyme_lens"] = trace.details.get(
                        "enzyme_signals", {})
            elif trace.lens_name == "hinge_lens":
                result["hinge_lens_activated"] = trace.activated
                result["hinge_boost"] = trace.boost
                if trace.activated:
                    result["hinge_signals"] = trace.details.get(
                        "hinge_signals", {})
            elif trace.lens_name == "barrel_penalty":
                result["barrel_penalty_activated"] = trace.activated
                result["barrel_penalty_signals"] = trace.details

        # Ensure keys exist even if lenses didn't fire
        result.setdefault("enzyme_lens_activated", False)
        result.setdefault("enzyme_lens_boost", 0.0)
        result.setdefault("hinge_lens_activated", False)
        result.setdefault("hinge_boost", 0.0)
        result.setdefault("barrel_penalty_activated", False)
        result.setdefault("hinge_signals", {})
        result.setdefault("enzyme_lens", {})
        result.setdefault("barrel_penalty_signals", {})

        # ── Build ClassificationTrace ───────────────────────────
        from .trace import ClassificationTrace, ContextSignals

        ctx_raw = result.get("context_signals", {})
        ctx_sig = ContextSignals(
            all_scatter=ctx_raw.get("all_scatter", 0.0),
            all_db=ctx_raw.get("all_db", 0.0),
            all_ipr=ctx_raw.get("all_ipr", 0.0),
            all_mass=ctx_raw.get("all_mass", 0.0),
            all_scatter_norm=ctx_raw.get("all_scatter_norm", 0.0),
            all_radius=ctx_raw.get("all_radius", 0.0),
            n_residues=int(ctx_raw.get("n_residues", 0)),
            propagative_radius=ctx_raw.get("propagative_radius", 0.0),
            propagative_scatter_norm=ctx_raw.get(
                "propagative_scatter_norm", 0.0),
        )

        # Gather per-instrument rule firings (traced votes)
        rule_firings: dict = {}
        per_instrument_votes: dict = result.get("per_carver_votes", {})
        for p in carver_profiles:
            votes, firings = p.archetype_vote_traced()
            rule_firings[p.instrument] = list(firings)

        n_res = (
            carver_profiles[0].n_residues
            if carver_profiles else 0
        )

        trace = ClassificationTrace(
            identity=result["identity"],
            scores=dict(result["scores"]),
            per_instrument_votes=per_instrument_votes,
            rule_firings=rule_firings,
            consensus_scores=result.get("consensus_scores", {}),
            disagreement_scores=result.get("disagreement_scores", {}),
            context_boost=result.get("context_boost", {}),
            context_signals=ctx_sig,
            alpha_meta=result.get("alpha_meta", 0.5),
            meta_state=dict(meta_state),
            lens_traces=list(traces),
            thresholds_name=self._t.name,
            n_residues=n_res,
            n_instruments=len(carver_profiles),
        )
        result["trace"] = trace

        return result
