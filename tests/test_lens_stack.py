"""Tests for the LensStack composition (v0.5.0).

Covers:
1. Lens protocol — all concrete lenses implement the protocol
2. LensStack — ordering, composition, traces
3. EnzymeLens — activation gate, boost computation
4. HingeLens — activation gate, hinge_R boost
5. BarrelPenaltyLens — activation gate, penalty signals
6. LensStackSynthesizer — full integration, backwards compat
7. Stack manipulation — with_lens, without, replace
8. Equivalence — LensStackSynthesizer matches old SizeAwareHingeLens
"""

import pytest
import numpy as np
from collections import Counter

from ibp_enm.instruments import ThermoReactionProfile
from ibp_enm.synthesis import MetaFickBalancer, SizeAwareHingeLens
from ibp_enm.lens_stack import (
    Lens,
    LensTrace,
    LensStack,
    EnzymeLens,
    HingeLens,
    BarrelPenaltyLens,
    AllostericLens,
    FlowGrammarLens,
    LensStackSynthesizer,
    build_default_stack,
)


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════

def _make_profile(instrument, **kwargs):
    """Helper: build a ThermoReactionProfile with defaults."""
    defaults = dict(
        instrument=instrument, intent_idx=0,
        n_residues=200, n_contacts=1000,
        gap_trajectory=[1.0, 0.99, 0.98, 0.97, 0.96, 0.95],
        mode_scatters=[2.0, 2.5, 1.8, 2.2, 2.1],
        delta_beta_trajectory=[0.05, 0.06, 0.04, 0.07, 0.05],
        bus_mass_trajectory=[0.6, 0.65, 0.58, 0.62, 0.6],
        spatial_radius_trajectory=[3.0, 3.5, 2.8, 3.2, 3.0],
        reversibility=[True, False, True, True, False],
        species_removed=["Carved", "Sibling", "Carved", "Soft", "Carved"],
        ipr_trajectory=[0.02, 0.022, 0.019, 0.021, 0.02, 0.02],
        entropy_trajectory=[10.0, 10.1, 10.05, 10.08, 10.12, 10.15],
        heat_cap_trajectory=[5.0, 5.1, 5.05, 5.08, 5.12, 5.15],
        free_energy_trajectory=[-8.0, -7.9, -7.95, -7.92, -7.88, -7.85],
        delta_entropy_per_cut=[0.1, -0.05, 0.03, 0.04, 0.03],
    )
    defaults.update(kwargs)
    return ThermoReactionProfile(**defaults)


@pytest.fixture
def seven_profiles():
    """Standard 7-instrument profile set."""
    names = ["algebraic", "musical", "fick", "thermal",
             "cooperative", "propagative", "fragile"]
    return [_make_profile(n) for n in names]


@pytest.fixture
def enzyme_contest_profiles():
    """Profiles where enzyme and allosteric are close → enzyme lens fires."""
    names = ["algebraic", "musical", "fick", "thermal",
             "cooperative", "propagative", "fragile"]
    profiles = []
    for n in names:
        p = _make_profile(n, ipr_trajectory=[0.03, 0.035, 0.028, 0.032, 0.03, 0.031])
        profiles.append(p)
    return profiles


@pytest.fixture
def barrel_winner_large():
    """Profiles where barrel wins but protein is large → barrel penalty fires."""
    names = ["algebraic", "musical", "fick", "thermal",
             "cooperative", "propagative", "fragile"]
    return [_make_profile(n, n_residues=400, n_contacts=3000,
                          mode_scatters=[0.3, 0.4, 0.35, 0.25, 0.3],
                          delta_beta_trajectory=[0.05, 0.06, 0.05, 0.04, 0.05],
                          bus_mass_trajectory=[0.3, 0.35, 0.28, 0.32, 0.3],
                          spatial_radius_trajectory=[15.0, 16.0, 14.0, 15.5, 14.5])
            for n in names]


@pytest.fixture
def default_context():
    """Default context dict for lens testing."""
    return {
        "evals": None,
        "evecs": None,
        "domain_labels": None,
        "contacts": None,
    }


# ═══════════════════════════════════════════════════════════════════
# 1. Lens protocol
# ═══════════════════════════════════════════════════════════════════

class TestLensProtocol:
    def test_enzyme_lens_is_lens(self):
        assert isinstance(EnzymeLens(), Lens)

    def test_hinge_lens_is_lens(self):
        assert isinstance(HingeLens(), Lens)

    def test_barrel_penalty_is_lens(self):
        assert isinstance(BarrelPenaltyLens(), Lens)

    def test_enzyme_lens_name(self):
        assert EnzymeLens().name == "enzyme_lens"

    def test_hinge_lens_name(self):
        assert HingeLens().name == "hinge_lens"

    def test_barrel_penalty_name(self):
        assert BarrelPenaltyLens().name == "barrel_penalty"


# ═══════════════════════════════════════════════════════════════════
# 2. LensStack composition
# ═══════════════════════════════════════════════════════════════════

class TestLensStack:
    def test_empty_stack(self):
        stack = LensStack()
        assert len(stack) == 0

    def test_default_stack_has_four_lenses(self):
        stack = build_default_stack()
        assert len(stack) == 4

    def test_default_stack_order(self):
        stack = build_default_stack()
        names = [l.name for l in stack]
        assert names == ["enzyme_lens", "hinge_lens", "barrel_penalty",
                         "allosteric_lens"]

    def test_default_stack_with_flow_grammar(self):
        stack = build_default_stack(include_flow_grammar=True)
        assert len(stack) == 5
        names = [l.name for l in stack]
        assert "flow_grammar_lens" in names

    def test_stack_repr(self):
        stack = build_default_stack()
        r = repr(stack)
        assert "enzyme_lens" in r
        assert "hinge_lens" in r
        assert "barrel_penalty" in r

    def test_empty_stack_passes_through(self, seven_profiles, default_context):
        scores = {"barrel": 0.3, "dumbbell": 0.2, "globin": 0.2,
                  "enzyme_active": 0.15, "allosteric": 0.15}
        stack = LensStack()
        result, traces = stack.apply(scores, seven_profiles, default_context)
        assert result == scores
        assert traces == []

    def test_stack_collects_traces(self, seven_profiles, default_context):
        scores = {"barrel": 0.3, "dumbbell": 0.2, "globin": 0.2,
                  "enzyme_active": 0.15, "allosteric": 0.15}
        stack = build_default_stack()
        _, traces = stack.apply(scores, seven_profiles, default_context)
        assert len(traces) == 4
        for t in traces:
            assert isinstance(t, LensTrace)

    def test_stack_iteration(self):
        stack = build_default_stack()
        lenses = list(stack)
        assert len(lenses) == 4

    def test_stack_lenses_property_immutable(self):
        stack = build_default_stack()
        tup = stack.lenses
        assert isinstance(tup, tuple)


# ═══════════════════════════════════════════════════════════════════
# 3. EnzymeLens
# ═══════════════════════════════════════════════════════════════════

class TestEnzymeLens:
    def test_no_activation_clear_winner(self, seven_profiles, default_context):
        """When barrel dominates, enzyme lens should NOT activate."""
        scores = {"barrel": 0.5, "dumbbell": 0.15, "globin": 0.15,
                  "enzyme_active": 0.1, "allosteric": 0.1}
        lens = EnzymeLens()
        assert not lens.should_activate(scores, seven_profiles, default_context)

    def test_activation_enzyme_allosteric_contest(self, enzyme_contest_profiles, default_context):
        """When enzyme and allosteric are close, lens should activate."""
        scores = {"barrel": 0.1, "dumbbell": 0.1, "globin": 0.1,
                  "enzyme_active": 0.35, "allosteric": 0.35}
        lens = EnzymeLens()
        assert lens.should_activate(scores, enzyme_contest_profiles, default_context)

    def test_apply_boosts_enzyme(self, enzyme_contest_profiles, default_context):
        scores = {"barrel": 0.1, "dumbbell": 0.1, "globin": 0.1,
                  "enzyme_active": 0.35, "allosteric": 0.35}
        lens = EnzymeLens()
        new_scores, trace = lens.apply(scores, enzyme_contest_profiles, default_context)
        assert trace.activated
        assert new_scores["enzyme_active"] >= scores["enzyme_active"]

    def test_boost_computation_ipr_high(self):
        signals = {"mean_ipr": 0.03, "algebraic_enzyme_score": 0.4,
                   "entropy_asymmetry": {"gini": 0.2, "cv": 0.4, "top5_frac": 0.2},
                   "fragile_ipr": 0.03, "fragile_rev": 0.9}
        boost = EnzymeLens._compute_boost(signals)
        assert boost > 0.15  # multiple signals fire


# ═══════════════════════════════════════════════════════════════════
# 4. HingeLens
# ═══════════════════════════════════════════════════════════════════

class TestHingeLens:
    def test_no_activation_without_evals(self, seven_profiles, default_context):
        """Without eigenvalues, hinge_R defaults to 1.0 → no activation."""
        scores = {"barrel": 0.1, "dumbbell": 0.1, "globin": 0.1,
                  "enzyme_active": 0.3, "allosteric": 0.4}
        lens = HingeLens()
        assert not lens.should_activate(scores, seven_profiles, default_context)

    def test_no_activation_allosteric_not_top2(self, seven_profiles, default_context):
        scores = {"barrel": 0.4, "dumbbell": 0.3, "globin": 0.15,
                  "enzyme_active": 0.1, "allosteric": 0.05}
        lens = HingeLens()
        assert not lens.should_activate(scores, seven_profiles, default_context)

    def test_hinge_boost_above_1(self):
        """hinge_R > 1.0 should give positive boost."""
        signals = {"hinge_r": 1.091, "ipr_25": 0.02, "dom_stiff": 0.1}
        boost = HingeLens._hinge_boost(signals)
        assert abs(boost - 0.273) < 0.01  # T4 lysozyme value

    def test_hinge_boost_below_1(self):
        """hinge_R < 1.0 should give zero boost."""
        signals = {"hinge_r": 0.952}
        boost = HingeLens._hinge_boost(signals)
        assert boost == 0.0

    def test_hinge_boost_capped(self):
        """Boost should be capped at 0.35."""
        signals = {"hinge_r": 2.0}
        boost = HingeLens._hinge_boost(signals)
        assert boost == 0.35


# ═══════════════════════════════════════════════════════════════════
# 5. BarrelPenaltyLens
# ═══════════════════════════════════════════════════════════════════

class TestBarrelPenaltyLens:
    def test_no_activation_non_barrel(self, seven_profiles, default_context):
        scores = {"barrel": 0.1, "dumbbell": 0.4, "globin": 0.2,
                  "enzyme_active": 0.15, "allosteric": 0.15}
        lens = BarrelPenaltyLens()
        assert not lens.should_activate(scores, seven_profiles, default_context)

    def test_no_activation_small_protein(self, seven_profiles, default_context):
        """Small proteins (N <= 250) should not trigger barrel penalty."""
        scores = {"barrel": 0.4, "dumbbell": 0.15, "globin": 0.15,
                  "enzyme_active": 0.15, "allosteric": 0.15}
        lens = BarrelPenaltyLens()
        # Default profiles have n_residues=200
        assert not lens.should_activate(scores, seven_profiles, default_context)

    def test_activation_large_barrel(self, barrel_winner_large, default_context):
        """Large protein winning barrel with moderate scatter should activate."""
        scores = {"barrel": 0.4, "dumbbell": 0.15, "globin": 0.15,
                  "enzyme_active": 0.15, "allosteric": 0.15}
        lens = BarrelPenaltyLens()
        assert lens.should_activate(scores, barrel_winner_large, default_context)

    def test_apply_penalises_barrel(self, barrel_winner_large, default_context):
        scores = {"barrel": 0.4, "dumbbell": 0.15, "globin": 0.15,
                  "enzyme_active": 0.15, "allosteric": 0.15}
        lens = BarrelPenaltyLens()
        new_scores, trace = lens.apply(scores, barrel_winner_large, default_context)
        assert new_scores["barrel"] < scores["barrel"]
        assert trace.activated


# ═══════════════════════════════════════════════════════════════════
# 6. LensStackSynthesizer integration
# ═══════════════════════════════════════════════════════════════════

class TestLensStackSynthesizer:
    def test_default_constructor(self):
        synth = LensStackSynthesizer()
        assert len(synth.stack) == 4

    def test_custom_stack(self):
        stack = LensStack([EnzymeLens()])
        synth = LensStackSynthesizer(stack=stack)
        assert len(synth.stack) == 1

    def test_synthesize_returns_identity(self, seven_profiles):
        synth = LensStackSynthesizer()
        votes = [p.archetype_vote() for p in seven_profiles]
        meta = synth.compute_meta_fick_state(votes)
        result = synth.synthesize_identity(seven_profiles, meta)
        assert "identity" in result
        assert "scores" in result
        assert len(result["scores"]) == 5
        assert abs(sum(result["scores"].values()) - 1.0) < 0.02

    def test_backwards_compat_keys(self, seven_profiles):
        """Result should have enzyme_lens_activated, hinge_lens_activated, etc."""
        synth = LensStackSynthesizer()
        votes = [p.archetype_vote() for p in seven_profiles]
        meta = synth.compute_meta_fick_state(votes)
        result = synth.synthesize_identity(seven_profiles, meta)
        assert "enzyme_lens_activated" in result
        assert "hinge_lens_activated" in result
        assert "barrel_penalty_activated" in result

    def test_lens_traces_present(self, seven_profiles):
        synth = LensStackSynthesizer()
        votes = [p.archetype_vote() for p in seven_profiles]
        meta = synth.compute_meta_fick_state(votes)
        result = synth.synthesize_identity(seven_profiles, meta)
        assert "lens_traces" in result
        assert len(result["lens_traces"]) == 4

    def test_empty_stack_same_as_base(self, seven_profiles):
        """LensStackSynthesizer with no lenses = plain MetaFickBalancer."""
        synth_stack = LensStackSynthesizer(stack=LensStack())
        synth_base = MetaFickBalancer()
        votes = [p.archetype_vote() for p in seven_profiles]
        meta_s = synth_stack.compute_meta_fick_state(votes)
        meta_b = synth_base.compute_meta_fick_state(votes)
        result_s = synth_stack.synthesize_identity(seven_profiles, meta_s)
        result_b = synth_base.synthesize_identity(seven_profiles, meta_b)
        # Scores should be very close (tiny float diff from renorm floor)
        for arch in result_s["scores"]:
            assert abs(result_s["scores"][arch] - result_b["scores"][arch]) < 0.02


# ═══════════════════════════════════════════════════════════════════
# 7. Stack manipulation
# ═══════════════════════════════════════════════════════════════════

class TestStackManipulation:
    def test_with_lens(self):
        stack = LensStack([EnzymeLens()])
        new_stack = stack.with_lens(HingeLens())
        assert len(new_stack) == 2
        assert len(stack) == 1  # original unchanged

    def test_without(self):
        stack = build_default_stack()
        new_stack = stack.without("hinge_lens")
        assert len(new_stack) == 3
        names = [l.name for l in new_stack]
        assert "hinge_lens" not in names

    def test_replace(self):
        stack = build_default_stack()
        new_enzyme = EnzymeLens()
        new_stack = stack.replace("enzyme_lens", new_enzyme)
        assert len(new_stack) == 4
        assert new_stack.lenses[0] is new_enzyme

    def test_with_lens_does_not_mutate_original(self):
        stack = build_default_stack()
        original_len = len(stack)
        _ = stack.with_lens(EnzymeLens())
        assert len(stack) == original_len

    def test_ab_test_enzyme_only_vs_full(self, seven_profiles):
        """A/B test: enzyme-only stack vs full stack."""
        full_synth = LensStackSynthesizer()
        enzyme_only = LensStackSynthesizer(
            stack=LensStack([EnzymeLens()]))
        votes = [p.archetype_vote() for p in seven_profiles]

        meta_f = full_synth.compute_meta_fick_state(votes)
        meta_e = enzyme_only.compute_meta_fick_state(votes)
        result_f = full_synth.synthesize_identity(seven_profiles, meta_f)
        result_e = enzyme_only.synthesize_identity(seven_profiles, meta_e)

        # Both should produce valid results (may differ in identity)
        assert "identity" in result_f
        assert "identity" in result_e


# ═══════════════════════════════════════════════════════════════════
# 8. Equivalence with old SizeAwareHingeLens
# ═══════════════════════════════════════════════════════════════════

class TestEquivalenceWithOldSynthesis:
    def test_same_identity_no_spectral_data(self, seven_profiles):
        """Without evals/evecs, both systems should produce same result.

        When no spectral data is available, neither the enzyme lens
        nor the hinge lens has entropy-asymmetry or hinge_R signals,
        so their behaviour should match.
        """
        old_synth = SizeAwareHingeLens(
            evals=None, evecs=None,
            domain_labels=None, contacts=None,
        )
        new_synth = LensStackSynthesizer(
            evals=None, evecs=None,
            domain_labels=None, contacts=None,
        )
        votes = [p.archetype_vote() for p in seven_profiles]

        old_meta = old_synth.compute_meta_fick_state(votes)
        new_meta = new_synth.compute_meta_fick_state(votes)

        old_result = old_synth.synthesize_identity(seven_profiles, old_meta)
        new_result = new_synth.synthesize_identity(seven_profiles, new_meta)

        # Identity should match
        assert old_result["identity"] == new_result["identity"]

    def test_same_identity_with_spectral_data(self):
        """With synthetic spectral data, both should match."""
        N = 50
        evals = np.sort(np.random.RandomState(42).rand(N))
        evals[0] = 0.0  # zero eigenvalue
        evecs = np.eye(N)
        domain_labels = np.array([0]*25 + [1]*25)

        profiles = []
        for name in ["algebraic", "musical", "fick", "thermal",
                      "cooperative", "propagative", "fragile"]:
            profiles.append(_make_profile(name))

        old_synth = SizeAwareHingeLens(
            evals=evals, evecs=evecs,
            domain_labels=domain_labels, contacts=None,
        )
        new_synth = LensStackSynthesizer(
            evals=evals, evecs=evecs,
            domain_labels=domain_labels, contacts=None,
        )
        votes = [p.archetype_vote() for p in profiles]

        old_meta = old_synth.compute_meta_fick_state(votes)
        new_meta = new_synth.compute_meta_fick_state(votes)

        old_result = old_synth.synthesize_identity(profiles, old_meta)
        new_result = new_synth.synthesize_identity(profiles, new_meta)

        assert old_result["identity"] == new_result["identity"]


# ═══════════════════════════════════════════════════════════════════
# 9. LensTrace
# ═══════════════════════════════════════════════════════════════════

class TestLensTrace:
    def test_trace_fields(self):
        t = LensTrace(lens_name="test", activated=True, boost=0.5)
        assert t.lens_name == "test"
        assert t.activated is True
        assert t.boost == 0.5
        assert t.details == {}

    def test_trace_with_details(self):
        t = LensTrace(lens_name="test", activated=True,
                      details={"key": "value"})
        assert t.details["key"] == "value"


# ═══════════════════════════════════════════════════════════════════
# 10. FlowGrammarLens — pre-carving TE flow vocabulary (D130)
# ═══════════════════════════════════════════════════════════════════

class TestFlowGrammarLens:
    """Tests for FlowGrammarLens integration."""

    def test_flow_grammar_lens_is_lens(self):
        lens = FlowGrammarLens()
        assert isinstance(lens, Lens)

    def test_flow_grammar_lens_name(self):
        lens = FlowGrammarLens()
        assert lens.name == "flow_grammar_lens"

    def test_no_activation_without_evals(self, seven_profiles, default_context):
        """FlowGrammarLens should not activate without evals."""
        scores = {"barrel": 0.2, "dumbbell": 0.2, "globin": 0.2,
                  "enzyme_active": 0.2, "allosteric": 0.2}
        lens = FlowGrammarLens()
        assert not lens.should_activate(scores, seven_profiles, default_context)

    def test_no_activation_allosteric_not_top3(self, seven_profiles):
        """FlowGrammarLens should not activate if allosteric not in top 3."""
        N = 50
        evals = np.sort(np.abs(np.random.RandomState(42).rand(N)))
        evals[0] = 0.0
        evecs = np.linalg.qr(np.random.RandomState(42).rand(N, N))[0]
        scores = {"barrel": 0.35, "dumbbell": 0.25, "globin": 0.20,
                  "enzyme_active": 0.15, "allosteric": 0.05}
        context = {"evals": evals, "evecs": evecs, "n_residues": N}
        lens = FlowGrammarLens(evals=evals, evecs=evecs)
        assert not lens.should_activate(scores, seven_profiles, context)

    def test_activation_confused_scores(self, seven_profiles):
        """FlowGrammarLens should activate when allosteric in top 3 and
        scores are confused."""
        N = 50
        evals = np.sort(np.abs(np.random.RandomState(42).rand(N)))
        evals[0] = 0.0
        evecs = np.linalg.qr(np.random.RandomState(42).rand(N, N))[0]
        # Allosteric is close to winner → should activate
        scores = {"barrel": 0.22, "dumbbell": 0.18, "globin": 0.18,
                  "enzyme_active": 0.21, "allosteric": 0.21}
        context = {"evals": evals, "evecs": evecs, "n_residues": N}
        lens = FlowGrammarLens(evals=evals, evecs=evecs)
        assert lens.should_activate(scores, seven_profiles, context)

    def test_apply_returns_scores_and_trace(self, seven_profiles):
        """apply() should return modified scores and a LensTrace."""
        N = 50
        evals = np.sort(np.abs(np.random.RandomState(42).rand(N)))
        evals[0] = 0.0
        evecs = np.linalg.qr(np.random.RandomState(42).rand(N, N))[0]
        scores = {"barrel": 0.22, "dumbbell": 0.18, "globin": 0.18,
                  "enzyme_active": 0.21, "allosteric": 0.21}
        context = {"evals": evals, "evecs": evecs, "n_residues": N}
        lens = FlowGrammarLens(evals=evals, evecs=evecs)
        new_scores, trace = lens.apply(scores, seven_profiles, context)
        assert isinstance(new_scores, dict)
        assert len(new_scores) == 5
        assert isinstance(trace, LensTrace)
        assert trace.lens_name == "flow_grammar_lens"
        assert trace.activated is True
        assert "flow_signals" in trace.details
        signals = trace.details["flow_signals"]
        assert "te_asymmetry" in signals
        assert "cross_enrichment" in signals
        assert "driver_sensor_ratio" in signals
        assert signals["flow_word"] in ("DIRECTING", "CHANNELING", "DIFFUSING")

    def test_flow_signals_computation(self):
        """_compute_flow_signals should return valid features from
        synthetic spectral data."""
        N = 30
        rng = np.random.RandomState(99)
        # Build a symmetric positive-definite Laplacian-like matrix
        A = rng.rand(N, N)
        A = (A + A.T) / 2
        np.fill_diagonal(A, 0)
        L = np.diag(A.sum(axis=1)) - A
        evals, evecs = np.linalg.eigh(L)
        signals = FlowGrammarLens._compute_flow_signals(evals, evecs, N)
        assert signals["te_asymmetry"] >= 0
        assert signals["cross_enrichment"] >= 0
        assert signals["driver_sensor_ratio"] >= 0
        assert signals["flow_word"] in ("DIRECTING", "CHANNELING", "DIFFUSING")

    def test_flow_boost_directing(self):
        """High cross_enrichment + high te_asymmetry should produce
        positive allosteric boost."""
        from ibp_enm.thresholds import DEFAULT_THRESHOLDS
        signals = {
            "te_asymmetry": 1.15,
            "cross_enrichment": 1.07,
            "driver_sensor_ratio": 1.25,
            "flow_word": "DIRECTING",
        }
        boost = FlowGrammarLens._compute_flow_boost(signals, DEFAULT_THRESHOLDS)
        assert boost > 0  # should be positive
        assert boost <= 0.18  # within cap

    def test_flow_boost_anti_allosteric(self):
        """Low cross_enrichment + low driver_sensor_ratio should produce
        negative allosteric penalty."""
        from ibp_enm.thresholds import DEFAULT_THRESHOLDS
        signals = {
            "te_asymmetry": 0.7,
            "cross_enrichment": 0.50,
            "driver_sensor_ratio": 0.80,
            "flow_word": "DIFFUSING",
        }
        boost = FlowGrammarLens._compute_flow_boost(signals, DEFAULT_THRESHOLDS)
        assert boost < 0  # should be negative (anti-allosteric)

    def test_flow_boost_neutral(self):
        """Mid-range flow features should produce near-zero boost."""
        from ibp_enm.thresholds import DEFAULT_THRESHOLDS
        signals = {
            "te_asymmetry": 0.85,
            "cross_enrichment": 0.80,
            "driver_sensor_ratio": 1.10,
            "flow_word": "CHANNELING",
        }
        boost = FlowGrammarLens._compute_flow_boost(signals, DEFAULT_THRESHOLDS)
        assert abs(boost) < 0.10  # should be small/neutral

    def test_build_default_stack_excludes_flow_grammar(self):
        """Default stack should NOT include FlowGrammarLens (D132: causes regressions)."""
        stack = build_default_stack()
        names = [l.name for l in stack]
        assert "flow_grammar_lens" not in names
        assert len(stack) == 4  # enzyme + hinge + barrel + allosteric

    def test_build_default_stack_include_flow_grammar(self):
        """include_flow_grammar=True should opt-in FlowGrammarLens."""
        stack = build_default_stack(include_flow_grammar=True)
        names = [l.name for l in stack]
        assert "flow_grammar_lens" in names
        assert len(stack) == 5

    def test_build_default_stack_exclude_both(self):
        """Both disabled should give 3 lenses."""
        stack = build_default_stack(
            include_flow_grammar=False, include_allosteric=False)
        names = [l.name for l in stack]
        assert len(stack) == 3
        assert names == ["enzyme_lens", "hinge_lens", "barrel_penalty"]
