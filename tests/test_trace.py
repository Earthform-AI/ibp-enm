"""Tests for ClassificationTrace (v0.7.0).

Covers:
1. ClassificationTrace — construction, properties, serialisation
2. ContextSignals — frozen dataclass, to_dict
3. Integration — trace is produced by LensStackSynthesizer
4. explain() — human-readable audit trail
"""

import pytest
import numpy as np

from ibp_enm.instruments import ThermoReactionProfile
from ibp_enm.lens_stack import (
    LensTrace,
    LensStackSynthesizer,
    build_default_stack,
)
from ibp_enm.rules import RuleFiring
from ibp_enm.thresholds import DEFAULT_THRESHOLDS
from ibp_enm.trace import ClassificationTrace, ContextSignals


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
def sample_trace():
    """A manually constructed ClassificationTrace for unit tests."""
    return ClassificationTrace(
        identity="enzyme_active",
        scores={"enzyme_active": 0.35, "allosteric": 0.30,
                "barrel": 0.15, "dumbbell": 0.10, "globin": 0.10},
        per_instrument_votes={
            "algebraic": {"enzyme_active": 0.40, "allosteric": 0.20,
                          "barrel": 0.15, "dumbbell": 0.15, "globin": 0.10},
            "thermal": {"enzyme_active": 0.30, "allosteric": 0.35,
                        "barrel": 0.15, "dumbbell": 0.10, "globin": 0.10},
        },
        rule_firings={
            "algebraic": [
                RuleFiring("alg_enzyme_ipr_high", "algebraic",
                           "enzyme_active", "mean_ipr", 0.030, 0.15,
                           "D109 d=1.4"),
                RuleFiring("alg_barrel_scatter_low", "algebraic",
                           "barrel", "scatter_normalised", 1.2, 0.10,
                           "D113 d=2.1"),
            ],
            "thermal": [
                RuleFiring("therm_enzyme_ipr_high", "thermal",
                           "enzyme_active", "mean_ipr", 0.028, 0.12,
                           "D109 d=1.3"),
            ],
        },
        consensus_scores={"enzyme_active": 0.30, "allosteric": 0.25,
                          "barrel": 0.20, "dumbbell": 0.15, "globin": 0.10},
        disagreement_scores={"enzyme_active": 2.5, "allosteric": 2.0,
                             "barrel": 1.5, "dumbbell": 1.0, "globin": 0.5},
        context_boost={"enzyme_active": 0.8, "allosteric": 0.0,
                       "barrel": -0.5, "dumbbell": 0.0, "globin": 0.0},
        context_signals=ContextSignals(
            all_scatter=2.0, all_db=0.05, all_ipr=0.025,
            all_mass=0.6, all_scatter_norm=1.8, all_radius=5.0,
            n_residues=200, propagative_radius=4.5,
            propagative_scatter_norm=1.9,
        ),
        alpha_meta=0.75,
        meta_state={"tau": 3.0, "beta": 2.5, "alpha_meta": 0.75},
        lens_traces=[
            LensTrace("enzyme_lens", True, 0.08, {"enzyme_signals": {}}),
            LensTrace("hinge_lens", False, 0.0, {}),
            LensTrace("barrel_penalty", False, 0.0, {}),
        ],
        thresholds_name="production",
        n_residues=200,
        n_instruments=7,
    )


# ═══════════════════════════════════════════════════════════════════
# 1. ContextSignals
# ═══════════════════════════════════════════════════════════════════

class TestContextSignals:

    def test_frozen(self):
        cs = ContextSignals(all_scatter=2.0)
        with pytest.raises(AttributeError):
            cs.all_scatter = 999.0

    def test_defaults(self):
        cs = ContextSignals()
        assert cs.all_scatter == 0.0
        assert cs.n_residues == 200

    def test_to_dict(self):
        cs = ContextSignals(all_scatter=2.123456789, n_residues=300)
        d = cs.to_dict()
        assert d["all_scatter"] == pytest.approx(2.123457, abs=1e-5)
        assert d["n_residues"] == 300
        assert isinstance(d, dict)

    def test_all_fields_in_to_dict(self):
        cs = ContextSignals()
        d = cs.to_dict()
        expected_keys = {
            "all_scatter", "all_db", "all_ipr", "all_mass",
            "all_scatter_norm", "all_radius", "n_residues",
            "propagative_radius", "propagative_scatter_norm",
        }
        assert set(d.keys()) == expected_keys


# ═══════════════════════════════════════════════════════════════════
# 2. ClassificationTrace — core properties
# ═══════════════════════════════════════════════════════════════════

class TestClassificationTraceProperties:

    def test_frozen(self, sample_trace):
        with pytest.raises(AttributeError):
            sample_trace.identity = "barrel"

    def test_activated_lenses(self, sample_trace):
        assert sample_trace.activated_lenses == ["enzyme_lens"]

    def test_total_lens_boost(self, sample_trace):
        assert sample_trace.total_lens_boost == pytest.approx(0.08)

    def test_n_rules_fired(self, sample_trace):
        assert sample_trace.n_rules_fired == 3

    def test_score_margin(self, sample_trace):
        assert sample_trace.score_margin == pytest.approx(0.05)

    def test_runner_up(self, sample_trace):
        assert sample_trace.runner_up == "allosteric"

    def test_top_rules_sorted_by_score(self, sample_trace):
        top = sample_trace.top_rules
        assert len(top) == 3
        # Sorted by |score| descending
        assert abs(top[0].score) >= abs(top[1].score)
        assert abs(top[1].score) >= abs(top[2].score)

    def test_repr(self, sample_trace):
        r = repr(sample_trace)
        assert "enzyme_active" in r
        assert "margin=" in r
        assert "rules=3" in r


# ═══════════════════════════════════════════════════════════════════
# 3. Serialisation
# ═══════════════════════════════════════════════════════════════════

class TestClassificationTraceSerialization:

    def test_to_dict_is_json_safe(self, sample_trace):
        import json
        d = sample_trace.to_dict()
        # Should not raise
        text = json.dumps(d)
        assert isinstance(text, str)

    def test_to_dict_has_all_top_level_keys(self, sample_trace):
        d = sample_trace.to_dict()
        expected_keys = {
            "identity", "scores", "per_instrument_votes",
            "rule_firings", "consensus_scores", "disagreement_scores",
            "context_boost", "context_signals", "alpha_meta",
            "lens_traces", "thresholds_name", "n_residues",
            "n_instruments", "n_rules_fired", "score_margin",
            "activated_lenses",
        }
        assert expected_keys.issubset(set(d.keys()))

    def test_to_dict_rule_firings_structure(self, sample_trace):
        d = sample_trace.to_dict()
        alg_firings = d["rule_firings"]["algebraic"]
        assert len(alg_firings) == 2
        assert alg_firings[0]["rule_name"] == "alg_enzyme_ipr_high"
        assert "value" in alg_firings[0]
        assert "score" in alg_firings[0]

    def test_to_dict_lens_traces(self, sample_trace):
        d = sample_trace.to_dict()
        assert len(d["lens_traces"]) == 3
        assert d["lens_traces"][0]["lens_name"] == "enzyme_lens"
        assert d["lens_traces"][0]["activated"] is True

    def test_summary(self, sample_trace):
        s = sample_trace.summary()
        assert "enzyme_active" in s
        assert "margin=" in s
        assert "enzyme_lens" in s

    def test_explain(self, sample_trace):
        text = sample_trace.explain()
        assert "Classification: enzyme_active" in text
        assert "Scores:" in text
        assert "Context boost:" in text
        assert "Lens effects:" in text
        assert "Top" in text
        assert "alg_enzyme_ipr_high" in text

    def test_explain_top_n(self, sample_trace):
        text = sample_trace.explain(top_n=1)
        # Should only show 1 rule
        lines = [l for l in text.split("\n") if "alg_" in l or "therm_" in l]
        assert len(lines) == 1


# ═══════════════════════════════════════════════════════════════════
# 4. Integration — LensStackSynthesizer produces trace
# ═══════════════════════════════════════════════════════════════════

class TestTraceIntegration:

    def test_synthesize_identity_returns_trace(self, seven_profiles):
        """LensStackSynthesizer.synthesize_identity produces a trace."""
        synth = LensStackSynthesizer()
        votes = [p.archetype_vote() for p in seven_profiles]
        meta_state = synth.compute_meta_fick_state(votes)
        result = synth.synthesize_identity(seven_profiles, meta_state)

        assert "trace" in result
        trace = result["trace"]
        assert isinstance(trace, ClassificationTrace)

    def test_trace_identity_matches_result(self, seven_profiles):
        synth = LensStackSynthesizer()
        votes = [p.archetype_vote() for p in seven_profiles]
        meta_state = synth.compute_meta_fick_state(votes)
        result = synth.synthesize_identity(seven_profiles, meta_state)

        trace = result["trace"]
        assert trace.identity == result["identity"]
        assert trace.scores == result["scores"]

    def test_trace_has_all_instruments(self, seven_profiles):
        synth = LensStackSynthesizer()
        votes = [p.archetype_vote() for p in seven_profiles]
        meta_state = synth.compute_meta_fick_state(votes)
        result = synth.synthesize_identity(seven_profiles, meta_state)

        trace = result["trace"]
        assert trace.n_instruments == 7
        assert len(trace.rule_firings) == 7

    def test_trace_has_rule_firings(self, seven_profiles):
        synth = LensStackSynthesizer()
        votes = [p.archetype_vote() for p in seven_profiles]
        meta_state = synth.compute_meta_fick_state(votes)
        result = synth.synthesize_identity(seven_profiles, meta_state)

        trace = result["trace"]
        # At least some rules should have fired
        assert trace.n_rules_fired > 0

    def test_trace_has_context_signals(self, seven_profiles):
        synth = LensStackSynthesizer()
        votes = [p.archetype_vote() for p in seven_profiles]
        meta_state = synth.compute_meta_fick_state(votes)
        result = synth.synthesize_identity(seven_profiles, meta_state)

        trace = result["trace"]
        cs = trace.context_signals
        assert isinstance(cs, ContextSignals)
        assert cs.n_residues == 200
        assert cs.all_ipr > 0  # our profiles have non-zero IPR

    def test_trace_has_lens_traces(self, seven_profiles):
        synth = LensStackSynthesizer()
        votes = [p.archetype_vote() for p in seven_profiles]
        meta_state = synth.compute_meta_fick_state(votes)
        result = synth.synthesize_identity(seven_profiles, meta_state)

        trace = result["trace"]
        assert len(trace.lens_traces) == 3  # enzyme, hinge, barrel

    def test_trace_thresholds_name(self, seven_profiles):
        synth = LensStackSynthesizer()
        votes = [p.archetype_vote() for p in seven_profiles]
        meta_state = synth.compute_meta_fick_state(votes)
        result = synth.synthesize_identity(seven_profiles, meta_state)

        trace = result["trace"]
        assert trace.thresholds_name == "production"

    def test_trace_custom_thresholds_name(self, seven_profiles):
        custom = DEFAULT_THRESHOLDS.replace(
            {"enzyme_lens.ipr_strong": 0.999}, name="sweep-42")
        synth = LensStackSynthesizer(thresholds=custom)
        votes = [p.archetype_vote() for p in seven_profiles]
        meta_state = synth.compute_meta_fick_state(votes)
        result = synth.synthesize_identity(seven_profiles, meta_state)

        trace = result["trace"]
        assert trace.thresholds_name == "sweep-42"

    def test_trace_serializes_cleanly(self, seven_profiles):
        """Full integration: synthesize → trace → JSON."""
        import json
        synth = LensStackSynthesizer()
        votes = [p.archetype_vote() for p in seven_profiles]
        meta_state = synth.compute_meta_fick_state(votes)
        result = synth.synthesize_identity(seven_profiles, meta_state)

        trace = result["trace"]
        d = trace.to_dict()
        text = json.dumps(d, indent=2)
        assert len(text) > 100  # non-trivial output

    def test_trace_explain_integration(self, seven_profiles):
        synth = LensStackSynthesizer()
        votes = [p.archetype_vote() for p in seven_profiles]
        meta_state = synth.compute_meta_fick_state(votes)
        result = synth.synthesize_identity(seven_profiles, meta_state)

        trace = result["trace"]
        text = trace.explain()
        assert "Classification:" in text
        assert "Scores:" in text

    def test_trace_backwards_compatible(self, seven_profiles):
        """Original result dict keys still present alongside trace."""
        synth = LensStackSynthesizer()
        votes = [p.archetype_vote() for p in seven_profiles]
        meta_state = synth.compute_meta_fick_state(votes)
        result = synth.synthesize_identity(seven_profiles, meta_state)

        # All legacy keys should still be present
        assert "identity" in result
        assert "scores" in result
        assert "enzyme_lens_activated" in result
        assert "hinge_lens_activated" in result
        assert "barrel_penalty_activated" in result
        assert "lens_traces" in result
        assert "trace" in result  # new


# ═══════════════════════════════════════════════════════════════════
# 5. Public API
# ═══════════════════════════════════════════════════════════════════

class TestPublicAPI:

    def test_imports_from_top_level(self):
        import ibp_enm
        assert hasattr(ibp_enm, "ClassificationTrace")
        assert hasattr(ibp_enm, "ContextSignals")

    def test_protein_result_has_trace_field(self):
        from ibp_enm.benchmark import ProteinResult, ProteinEntry
        entry = ProteinEntry("test", "1ABC", "A", "barrel")
        pr = ProteinResult(
            entry=entry, predicted="barrel",
            scores={"barrel": 1.0}, correct=True, time_s=0.1,
        )
        assert pr.trace is None  # default
