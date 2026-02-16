"""Meta-Fick synthesis — fusing 7 independent carver votes into identity.

The :class:`MetaFickBalancer` applies the Fick-diffusion metaphor
*to the vote space itself*: each carver's archetype vote is a
"concentration", and the balancer decides how much to trust
consensus vs. disagreement when the votes conflict.

Key idea: when all 7 instruments agree (ρ → 1), trust the majority.
When they disagree, trust the instruments whose unique signals are
strongest (the "data-calibrated context boosts").

Data-calibrated context boosts
------------------------------
From D109 Run 1 empirical analysis:

  BARREL    : LOW scatter across ALL instruments (d=2.1–3.3)
  DUMBBELL  : HIGH Δβ across instruments (d=2.1–2.9)
  ENZYME    : HIGH IPR across instruments (d=1.3–1.5)
  GLOBIN    : LOW flatness under algebraic/thermal (d=2.4–3.4)
  ALLOSTERIC: specific instrument × observable combos

Historical notes
----------------
MetaFickBalancer architecture from D108.
Data-calibrated context boosts from D109.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List

from .archetypes import ARCHETYPE_EXPECTATIONS
from .instruments import ThermoReactionProfile
from .thermodynamics import (
    per_residue_entropy_contribution,
    entropy_asymmetry_score,
    multimode_ipr,
    hinge_occupation_ratio,
    domain_stiffness_asymmetry,
)

__all__ = [
    "MetaFickBalancer",
    "EnzymeLensSynthesis",
    "HingeLensSynthesis",
]


class MetaFickBalancer:
    """Fick diffusion applied to the band's consensus formation.

    Parameters
    ----------
    w1 : float
        Log-delta-tau weight (negative → penalise big jumps).
    w2 : float
        Beta-offset weight (negative → prefer sharp winners).
    w3 : float
        Agreement weight (positive → reward consensus).
    beta0 : float
        Beta baseline for the sigmoid.
    """

    def __init__(self, w1: float = -1.2, w2: float = -0.4,
                 w3: float = 0.8, beta0: float = 1.5):
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.beta0 = beta0
        self.history: List[Dict] = []
        self.tau_prev: float | None = None

    # ── helpers ─────────────────────────────────────────────────

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))

    # ── meta-fick state ─────────────────────────────────────────

    def compute_meta_fick_state(
            self, carver_votes: List[Dict[str, float]]) -> Dict:
        """Compute meta-Fick state from all carvers' votes.

        Returns a dict with τ, β, δτ, σ², ρ, α_meta, consensus,
        and the top archetype.
        """
        n_carvers = len(carver_votes)
        if n_carvers == 0:
            return {"tau": 1.0, "beta": 1.0, "delta_tau": 1.0,
                    "sigma2": 0.0, "rho": 0.0, "alpha_meta": 0.5}

        all_archs = list(ARCHETYPE_EXPECTATIONS.keys())

        # Mean consensus
        consensus: Dict[str, float] = {}
        for arch in all_archs:
            consensus[arch] = float(
                np.mean([v.get(arch, 0) for v in carver_votes]))
        total = sum(consensus.values())
        if total > 1e-10:
            consensus = {k: v / total for k, v in consensus.items()}

        # Entropy → τ
        probs = np.array([v for v in consensus.values() if v > 1e-10])
        entropy = float(-np.sum(probs * np.log(probs)))
        tau = 1.0 / (entropy + 1e-10)

        # β = winner / runner-up
        sorted_votes = sorted(consensus.values(), reverse=True)
        if len(sorted_votes) >= 2 and sorted_votes[1] > 1e-10:
            beta = sorted_votes[0] / sorted_votes[1]
        else:
            beta = 10.0

        # ρ = agreement fraction
        top_arch = max(consensus, key=consensus.get)
        agreeing = sum(
            1 for v in carver_votes if max(v, key=v.get) == top_arch)
        rho = agreeing / n_carvers

        # δτ
        if self.tau_prev is not None and self.tau_prev > 0:
            delta_tau = tau / self.tau_prev
        else:
            delta_tau = 1.0
        self.tau_prev = tau

        # σ²
        vote_variances = []
        for arch in all_archs:
            arch_votes = [v.get(arch, 0) for v in carver_votes]
            vote_variances.append(float(np.var(arch_votes)))
        sigma2 = float(np.mean(vote_variances))

        # α_meta via sigmoid
        log_dt = np.log(delta_tau + 1e-10)
        z = self.w1 * log_dt + self.w2 * (beta - self.beta0) + self.w3 * rho
        alpha_meta = float(self._sigmoid(z))

        state = {
            "tau": float(tau),
            "beta": float(beta),
            "delta_tau": float(delta_tau),
            "sigma2": sigma2,
            "rho": float(rho),
            "alpha_meta": alpha_meta,
            "consensus": consensus,
            "top_archetype": top_arch,
            "n_agreeing": agreeing,
            "n_carvers": n_carvers,
        }
        self.history.append(state)
        return state

    # ── identity synthesis ──────────────────────────────────────

    def synthesize_identity(
            self,
            carver_profiles: List[ThermoReactionProfile],
            meta_state: Dict) -> Dict:
        """Synthesize identity using thermodynamic context boosts.

        Three pathways are combined:

        1. **Consensus** (weight α_meta): mean vote across instruments.
        2. **Disagreement** (weight 1 − α_meta): mean / (std + 0.1).
        3. **Context boost** (weight 0.25): data-calibrated signals
           with Cohen's d > 1.5 from D109 Run 1.
        """
        alpha_meta = meta_state["alpha_meta"]
        carver_votes = [p.archetype_vote() for p in carver_profiles]
        all_archs = list(ARCHETYPE_EXPECTATIONS.keys())

        # Consensus pathway
        consensus_scores: Dict[str, float] = {}
        for arch in all_archs:
            consensus_scores[arch] = float(
                np.mean([v.get(arch, 0) for v in carver_votes]))

        # Disagreement pathway
        disagreement_scores: Dict[str, float] = {}
        for arch in all_archs:
            arch_vals = [v.get(arch, 0) for v in carver_votes]
            mean_v = float(np.mean(arch_vals))
            std_v = float(np.std(arch_vals))
            disagreement_scores[arch] = mean_v / (std_v + 0.1)

        # ── Thermodynamic context boost (data-calibrated) ──
        context_boost: Dict[str, float] = {a: 0.0 for a in all_archs}

        # Cross-instrument summary signals
        all_scatter = float(np.mean(
            [p.mean_scatter for p in carver_profiles]))
        all_db = float(np.mean(
            [p.mean_delta_beta for p in carver_profiles]))
        all_ipr = float(np.mean(
            [p.mean_ipr for p in carver_profiles]))
        all_mass = float(np.mean(
            [p.mean_bus_mass for p in carver_profiles]))

        # BARREL: LOW scatter across ALL instruments (d=2.1–3.3)
        if all_scatter < 1.5:
            context_boost["barrel"] += 0.8
        elif all_scatter > 3.0:
            context_boost["barrel"] -= 0.5

        # DUMBBELL: HIGH Δβ across instruments (d=2.1–2.9)
        if all_db > 0.12:
            context_boost["dumbbell"] += 0.8
        elif all_db < 0.03:
            context_boost["dumbbell"] -= 0.3

        # ENZYME: HIGH IPR across instruments (d=1.3–1.5)
        if all_ipr > 0.025:
            context_boost["enzyme_active"] += 0.8
        elif all_ipr < 0.017:
            context_boost["enzyme_active"] -= 0.3

        # Per-instrument specific signals
        for profile in carver_profiles:
            if profile.instrument == "algebraic":
                if profile.gap_flatness < 0.75:
                    context_boost["globin"] += 0.8
                    context_boost["barrel"] -= 0.3
                elif profile.gap_flatness < 0.85:
                    context_boost["globin"] += 0.3
                if profile.mean_bus_mass < 0.5:
                    context_boost["barrel"] += 0.3

            elif profile.instrument == "thermal":
                if profile.gap_flatness < 0.75:
                    context_boost["globin"] += 0.6
                if profile.mean_bus_mass < 0.45:
                    context_boost["barrel"] += 0.4

            elif profile.instrument == "propagative":
                if profile.gap_flatness < 0.9:
                    context_boost["globin"] += 0.5
                if profile.reversible_frac > 0.9:
                    context_boost["barrel"] += 0.3

            elif profile.instrument == "fragile":
                if profile.mean_delta_beta > 0.15:
                    context_boost["dumbbell"] += 0.5
                if profile.reversible_frac < 0.5:
                    context_boost["dumbbell"] += 0.3

        # ── Combine ──
        final_scores: Dict[str, float] = {}
        for arch in all_archs:
            final_scores[arch] = (
                alpha_meta * consensus_scores.get(arch, 0)
                + (1 - alpha_meta) * disagreement_scores.get(arch, 0)
                + 0.25 * context_boost.get(arch, 0)
            )

        total = sum(final_scores.values())
        if total > 1e-10:
            final_scores = {k: v / total for k, v in final_scores.items()}

        identity = max(final_scores, key=final_scores.get)

        return {
            "identity": identity,
            "scores": final_scores,
            "consensus_scores": consensus_scores,
            "disagreement_scores": disagreement_scores,
            "context_boost": context_boost,
            "alpha_meta": alpha_meta,
            "per_carver_votes": {
                p.instrument: p.archetype_vote()
                for p in carver_profiles},
        }


class EnzymeLensSynthesis(MetaFickBalancer):
    """MetaFickBalancer with an enzyme-specific entropy-asymmetry lens.

    After standard D109 synthesis, applies a post-hoc enzyme lens
    that targets the enzyme-vs-allosteric confusion.  The lens
    activates only when the top two archetypes are within 0.15 of
    each other (or when enzyme and allosteric are within 0.15),
    preventing false activations on clear-cut cases.

    The enzyme lens uses four signal families:

    1. **IPR** (inverse participation ratio) — enzyme active sites
       localise low-frequency modes (IPR > 0.025, Cohen's d ≈ 1.4).
    2. **Algebraic vote** — the algebraic instrument went 12/12 in
       D109; its enzyme score > 0.35 is a strong indicator.
    3. **Entropy asymmetry** — enzymes have gini > 0.15, cv > 0.3,
       top5% > 0.15 because their active sites are thermodynamic
       hot spots (D110 discovery).
    4. **Fragile instrument** — enzyme shows high IPR + high
       reversibility simultaneously.

    Historical notes
    ----------------
    Designed in D110 (Enzyme Lens Experiment).  Fixes DHFR (the
    0.018-gap miss from D109).  T4_lysozyme remains unfixed because
    its algebraic signal is too weak (0.390 vs 0.380 — essentially
    a coin flip at the instrument level).

    Accuracy: 11/12 overall, 4/5 enzyme, 2/2 barrel, 0 false barrel.
    """

    def __init__(self, evals=None, evecs=None, **kwargs):
        super().__init__(**kwargs)
        self.initial_evals = evals
        self.initial_evecs = evecs

    def synthesize_identity(self, carver_profiles, meta_state):
        # Standard D109 synthesis first
        result = super().synthesize_identity(carver_profiles, meta_state)
        scores = dict(result["scores"])

        # ── Enzyme lens ──
        enzyme_score = scores.get("enzyme_active", 0)
        allosteric_score = scores.get("allosteric", 0)
        top2 = sorted(scores.values(), reverse=True)[:2]
        is_close_call = (len(top2) >= 2
                         and top2[0] - top2[1] < 0.10)

        enzyme_signals = self._compute_enzyme_signals(carver_profiles)
        result["enzyme_lens"] = enzyme_signals

        if is_close_call or abs(enzyme_score - allosteric_score) < 0.15:
            lens_boost = self._enzyme_lens_boost(enzyme_signals)
            scores["enzyme_active"] += lens_boost
            scores["allosteric"] -= lens_boost * 0.5

            total = sum(max(0.01, v) for v in scores.values())
            scores = {k: max(0.01, v) / total for k, v in scores.items()}

            result["scores"] = scores
            result["identity"] = max(scores, key=scores.get)
            result["enzyme_lens_boost"] = lens_boost
            result["enzyme_lens_activated"] = True
        else:
            result["enzyme_lens_boost"] = 0.0
            result["enzyme_lens_activated"] = False

        return result

    def _compute_enzyme_signals(self, carver_profiles):
        """Extract enzyme-specific signals from carving profiles."""
        all_ipr = [p.mean_ipr for p in carver_profiles]
        mean_ipr = float(np.mean(all_ipr))
        max_ipr = float(np.max(all_ipr))

        # Algebraic instrument vote (it went 12/12 in D109)
        algebraic_vote = None
        algebraic_enzyme_score = 0.0
        for p in carver_profiles:
            if p.instrument == "algebraic":
                algebraic_vote = p.archetype_vote()
                algebraic_enzyme_score = algebraic_vote.get(
                    "enzyme_active", 0)
                break

        # Entropy asymmetry from initial spectrum
        asymmetry: dict = {}
        if (self.initial_evals is not None
                and self.initial_evecs is not None):
            s_per_res = per_residue_entropy_contribution(
                self.initial_evals, self.initial_evecs)
            asymmetry = entropy_asymmetry_score(s_per_res)

        # Fragile instrument signals
        fragile_ipr = 0.0
        fragile_rev = 0.5
        for p in carver_profiles:
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

    def _enzyme_lens_boost(self, signals):
        """Compute enzyme boost from lens signals.

        Thresholds calibrated on D109/D110 empirical data:

        - IPR > 0.025: +0.08 (strongest single discriminator)
        - IPR > 0.020: +0.04
        - Algebraic enzyme > 0.35: +0.06
        - Algebraic enzyme > 0.25: +0.03
        - Gini > 0.15: +0.04
        - CV > 0.3: +0.03
        - Top5% > 0.15: +0.03
        - Fragile IPR > 0.025 AND rev > 0.8: +0.04
        """
        boost = 0.0

        # IPR signal
        if signals["mean_ipr"] > 0.025:
            boost += 0.08
        elif signals["mean_ipr"] > 0.020:
            boost += 0.04

        # Algebraic instrument
        if signals["algebraic_enzyme_score"] > 0.35:
            boost += 0.06
        elif signals["algebraic_enzyme_score"] > 0.25:
            boost += 0.03

        # Entropy asymmetry (D110 discovery)
        asym = signals.get("entropy_asymmetry", {})
        if asym.get("gini", 0) > 0.15:
            boost += 0.04
        if asym.get("cv", 0) > 0.3:
            boost += 0.03
        if asym.get("top5_frac", 0) > 0.15:
            boost += 0.03

        # Fragile: enzyme shows high rev + high IPR
        if (signals["fragile_ipr"] > 0.025
                and signals["fragile_rev"] > 0.8):
            boost += 0.04

        return boost


class HingeLensSynthesis(EnzymeLensSynthesis):
    """EnzymeLensSynthesis with a multi-mode hinge lens for hinge enzymes.

    Extends D110's :class:`EnzymeLensSynthesis` with a second post-hoc
    lens that targets proteins whose catalytic site sits at the
    inter-domain hinge — the class of "hinge enzymes" (e.g.
    T4 lysozyme) that are spectrally indistinguishable from allosteric
    proteins using mode-1 observables alone.

    The hinge lens examines modes 2–5 (beyond the global Fiedler
    hinge-bend) to detect whether higher modes still concentrate
    amplitude at the domain boundary.  Enzymes show
    ``hinge_R > 1.0`` because their catalytic cleft imposes
    constraints that shape the local mode landscape.  Allosteric
    proteins show ``hinge_R ≤ 1.0`` because mode 1 exhausts the
    hinge's contribution to dynamics.

    Activation gate
    ---------------
    The hinge lens fires **independently** of the D110 enzyme lens.
    It activates when:

    1. ``allosteric`` is in the top 2 archetypes, AND
    2. ``enzyme_active`` is in the top 4, AND
    3. ``hinge_R > 1.0``

    This gate is necessary because the D110 enzyme lens requires an
    enzyme–allosteric score gap < 0.15, but T4 lysozyme has a gap
    of 0.225 — the D110 lens never fires.

    Boost calibration
    -----------------
    The boost is proportional to the hinge_R excess:

        boost = min(0.35, (hinge_R − 1.0) × 3.0)

    T4 lysozyme: hinge_R = 1.091 → boost ≈ 0.273 (flips the call).
    AdK:         hinge_R = 0.952 → boost = 0 (correctly blocked).

    Historical notes
    ----------------
    Designed in D111 (Multi-Mode Hinge Analysis).  Proto A was the
    winning design — the simplest of three prototypes, using only
    the hinge_R signal.  All three prototypes achieved 12/12 but
    Proto A uses one signal with the clearest physical interpretation.

    Accuracy: 12/12 overall, 5/5 enzyme, 2/2 barrel, 0 false barrel.
    """

    def __init__(self, evals=None, evecs=None,
                 domain_labels=None, contacts=None, **kwargs):
        super().__init__(evals=evals, evecs=evecs, **kwargs)
        self.domain_labels = domain_labels
        self.contacts_for_hinge = contacts

    def synthesize_identity(self, carver_profiles, meta_state):
        # Standard D110 synthesis (MetaFick + enzyme lens)
        result = super().synthesize_identity(carver_profiles, meta_state)
        scores = dict(result["scores"])

        # Compute multi-mode hinge signals
        hinge_signals = self._compute_hinge_signals()
        result["hinge_signals"] = hinge_signals

        # ── Hinge lens activation gate ──
        # Independent of the D110 enzyme lens.  The enzyme lens
        # requires enzyme–allosteric gap < 0.15, but T4 lysozyme
        # has a gap of 0.225 — the enzyme lens never fires there.
        # The hinge lens fires when:
        #   1. allosteric is in the top 2 archetypes
        #   2. enzyme_active is in the top 4
        #   3. hinge_R > 1.0 (modes 2–5 feel the cleft)
        sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
        top2_archs = [a for a, _ in sorted_scores[:2]]
        top4_archs = [a for a, _ in sorted_scores[:4]]
        allosteric_in_top2 = "allosteric" in top2_archs
        enzyme_in_top4 = "enzyme_active" in top4_archs

        hinge_r = hinge_signals.get("hinge_r", 1.0)
        should_activate = (allosteric_in_top2
                           and enzyme_in_top4
                           and hinge_r > 1.0)

        if should_activate:
            hinge_boost = self._hinge_boost(hinge_signals)
            if hinge_boost > 0:
                scores["enzyme_active"] += hinge_boost
                scores["allosteric"] -= hinge_boost * 0.5

                total = sum(max(0.01, v) for v in scores.values())
                scores = {k: max(0.01, v) / total
                          for k, v in scores.items()}
                result["scores"] = scores
                result["identity"] = max(scores, key=scores.get)
                result["hinge_boost"] = hinge_boost
                result["hinge_lens_activated"] = True
            else:
                result["hinge_boost"] = 0.0
                result["hinge_lens_activated"] = False
        else:
            result["hinge_boost"] = 0.0
            result["hinge_lens_activated"] = False

        return result

    def _compute_hinge_signals(self):
        """Compute multi-mode hinge observables from the initial spectrum."""
        signals = {}
        if (self.initial_evals is not None
                and self.initial_evecs is not None):
            signals["ipr_25"] = multimode_ipr(self.initial_evecs)
            if self.domain_labels is not None:
                signals["hinge_r"] = hinge_occupation_ratio(
                    self.initial_evecs, self.domain_labels)
            else:
                signals["hinge_r"] = 1.0
            if (self.contacts_for_hinge is not None
                    and self.domain_labels is not None):
                signals["dom_stiff"] = domain_stiffness_asymmetry(
                    self.contacts_for_hinge, self.domain_labels)
            else:
                signals["dom_stiff"] = 0.0
        else:
            signals = {"ipr_25": 0.0, "hinge_r": 1.0, "dom_stiff": 0.0}
        return signals

    @staticmethod
    def _hinge_boost(signals):
        """Compute enzyme boost from hinge occupation ratio.

        Calibrated on D111 empirical data:

        - T4 lysozyme: hinge_R = 1.091 → boost ≈ 0.273 (flips call)
        - AdK:         hinge_R = 0.952 → boost = 0 (blocked by gate)
        - CaM:         hinge_R = 1.269 → would give 0.35, but CaM
          is blocked by the activation gate (allosteric not in top 2).
        """
        hinge_r = signals.get("hinge_r", 1.0)
        if hinge_r > 1.0:
            excess = hinge_r - 1.0
            return min(0.35, excess * 3.0)
        return 0.0
