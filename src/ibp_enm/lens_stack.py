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

__all__ = [
    "Lens",
    "LensTrace",
    "LensStack",
    "EnzymeLens",
    "HingeLens",
    "BarrelPenaltyLens",
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
        return hinge_signals.get("hinge_r", 1.0) > t["hinge_lens.hinge_r_gate"]

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
# Factory — default stack
# ═══════════════════════════════════════════════════════════════════

def build_default_stack(
    evals: Optional[np.ndarray] = None,
    evecs: Optional[np.ndarray] = None,
    domain_labels: Optional[np.ndarray] = None,
    contacts: Optional[dict] = None,
    thresholds: Optional[ThresholdRegistry] = None,
) -> LensStack:
    """Build the default lens stack (D110 + D111 + D113).

    Equivalent to the old ``SizeAwareHingeLens`` inheritance tower
    but composable.

    Parameters
    ----------
    evals, evecs : ndarray, optional
        Laplacian eigenvalues/eigenvectors (enables enzyme + hinge lenses).
    domain_labels : ndarray, optional
        Domain assignment per residue (enables hinge + barrel lenses).
    contacts : dict, optional
        Contact map (enables hinge lens domain stiffness).
    thresholds : ThresholdRegistry, optional
        Override threshold values (defaults to :data:`DEFAULT_THRESHOLDS`).

    Returns
    -------
    LensStack
        ``[EnzymeLens, HingeLens, BarrelPenaltyLens]``
    """
    t = thresholds
    return LensStack([
        EnzymeLens(evals=evals, evecs=evecs, thresholds=t),
        HingeLens(evals=evals, evecs=evecs,
                  domain_labels=domain_labels, contacts=contacts,
                  thresholds=t),
        BarrelPenaltyLens(domain_labels=domain_labels, thresholds=t),
    ])


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
        **balancer_kwargs,
    ):
        self._t = thresholds or DEFAULT_THRESHOLDS
        from .synthesis import MetaFickBalancer
        self.balancer = MetaFickBalancer(**balancer_kwargs)
        self.stack = stack if stack is not None else build_default_stack(
            evals=evals, evecs=evecs,
            domain_labels=domain_labels, contacts=contacts,
            thresholds=self._t,
        )
        self._context: Dict[str, Any] = {
            "evals": evals,
            "evecs": evecs,
            "domain_labels": domain_labels,
            "contacts": contacts,
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
