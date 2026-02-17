"""Tests for ibp_enm.algebra — algebraic structure analysis.

Tests cover:
  - Fingerprint construction and properties
  - Hamming distance and distance matrix
  - Bitwise lattice operations (AND, OR)
  - Instrument collinearity detection
  - Firing-pattern lattice construction
  - Threshold sensitivity analysis
"""

import numpy as np
import pytest

from ibp_enm.algebra import (
    Fingerprint,
    FiringLattice,
    InstrumentCollinearity,
    ThresholdSensitivity,
    build_firing_lattice,
    distance_matrix,
    fingerprint_from_trace,
    hamming_distance,
    check_collinearity,
    threshold_sensitivity,
    FANO_LINES,
    INSTRUMENT_NAMES,
)
from ibp_enm.lens_stack import LensTrace
from ibp_enm.rules import ARCHETYPE_RULES, RuleFiring
from ibp_enm.trace import ClassificationTrace, ContextSignals


# ═══════════════════════════════════════════════════════════════════
# Helpers — build minimal trace objects for testing
# ═══════════════════════════════════════════════════════════════════

def _make_trace(
    identity: str = "barrel",
    scores: dict | None = None,
    fired_rule_names: list[str] | None = None,
    per_instrument_votes: dict | None = None,
    margin: float = 0.15,
    alpha: float = 0.7,
) -> ClassificationTrace:
    """Build a minimal ClassificationTrace for testing."""
    if scores is None:
        scores = {
            "barrel": 0.4, "enzyme_active": 0.25,
            "globin": 0.15, "dumbbell": 0.1, "allosteric": 0.1,
        }
    if per_instrument_votes is None:
        per_instrument_votes = {
            inst: dict(scores) for inst in INSTRUMENT_NAMES
        }

    # Build rule_firings from fired_rule_names
    rule_firings: dict[str, list[RuleFiring]] = {}
    if fired_rule_names:
        for name in fired_rule_names:
            rf = RuleFiring(
                rule_name=name,
                instrument="algebraic",
                archetype=identity,
                metric="test",
                value=1.0,
                score=0.1,
                provenance="test",
            )
            rule_firings.setdefault("algebraic", []).append(rf)
    else:
        rule_firings = {"algebraic": []}

    return ClassificationTrace(
        identity=identity,
        scores=scores,
        per_instrument_votes=per_instrument_votes,
        rule_firings=rule_firings,
        consensus_scores=dict(scores),
        disagreement_scores={k: 0.05 for k in scores},
        context_boost={k: 0.0 for k in scores},
        context_signals=ContextSignals(),
        alpha_meta=alpha,
        meta_state={"alpha": alpha},
        lens_traces=[
            LensTrace(lens_name="EnzymeLens", activated=False),
            LensTrace(lens_name="HingeLens", activated=False),
            LensTrace(lens_name="BarrelPenaltyLens", activated=False),
        ],
        n_residues=200,
        n_instruments=7,
    )


def _sample_rule_names(n: int = 5) -> list[str]:
    """Get the first n rule names from the registry."""
    return [r.name for r in ARCHETYPE_RULES[:n]]


# ═══════════════════════════════════════════════════════════════════
# Fingerprint tests
# ═══════════════════════════════════════════════════════════════════

class TestFingerprint:
    """Tests for Fingerprint construction and properties."""

    def test_basic_construction(self):
        fp = Fingerprint(
            bits=(1, 0, 1, 0, 0),
            rule_names=("r1", "r2", "r3", "r4", "r5"),
            protein_name="test",
            identity="barrel",
        )
        assert fp.n_fired == 2
        assert fp.n_total == 5
        assert fp.density == pytest.approx(0.4)
        assert fp.fired_rules == ("r1", "r3")
        assert fp.as_set == frozenset({"r1", "r3"})

    def test_zero_fingerprint(self):
        fp = Fingerprint(
            bits=(0, 0, 0),
            rule_names=("r1", "r2", "r3"),
        )
        assert fp.n_fired == 0
        assert fp.density == 0.0
        assert fp.fired_rules == ()

    def test_full_fingerprint(self):
        fp = Fingerprint(
            bits=(1, 1, 1),
            rule_names=("r1", "r2", "r3"),
        )
        assert fp.n_fired == 3
        assert fp.density == pytest.approx(1.0)

    def test_to_array(self):
        fp = Fingerprint(bits=(1, 0, 1), rule_names=("a", "b", "c"))
        arr = fp.to_array()
        assert arr.dtype == np.int8
        np.testing.assert_array_equal(arr, [1, 0, 1])

    def test_summary(self):
        fp = Fingerprint(
            bits=(1, 0, 1),
            rule_names=("r1", "r2", "r3"),
            protein_name="Myoglobin",
            identity="globin",
        )
        s = fp.summary()
        assert "Myoglobin" in s
        assert "2/3" in s
        assert "globin" in s


class TestFingerprintLatticeOps:
    """Tests for bitwise AND/OR on fingerprints."""

    def test_and(self):
        fp1 = Fingerprint(bits=(1, 1, 0), rule_names=("a", "b", "c"))
        fp2 = Fingerprint(bits=(1, 0, 1), rule_names=("a", "b", "c"))
        result = fp1 & fp2
        assert result.bits == (1, 0, 0)

    def test_or(self):
        fp1 = Fingerprint(bits=(1, 1, 0), rule_names=("a", "b", "c"))
        fp2 = Fingerprint(bits=(1, 0, 1), rule_names=("a", "b", "c"))
        result = fp1 | fp2
        assert result.bits == (1, 1, 1)

    def test_and_mismatched_rules_raises(self):
        fp1 = Fingerprint(bits=(1,), rule_names=("a",))
        fp2 = Fingerprint(bits=(1,), rule_names=("b",))
        with pytest.raises(ValueError, match="different rule orderings"):
            _ = fp1 & fp2

    def test_or_mismatched_rules_raises(self):
        fp1 = Fingerprint(bits=(1,), rule_names=("a",))
        fp2 = Fingerprint(bits=(1,), rule_names=("b",))
        with pytest.raises(ValueError, match="different rule orderings"):
            _ = fp1 | fp2


class TestFingerprintFromTrace:
    """Tests for extracting fingerprints from ClassificationTrace."""

    def test_no_firings(self):
        trace = _make_trace(fired_rule_names=[])
        fp = fingerprint_from_trace(trace, protein_name="test")
        assert fp.n_fired == 0
        assert fp.n_total == len(ARCHETYPE_RULES)
        assert fp.identity == "barrel"

    def test_with_firings(self):
        names = _sample_rule_names(3)
        trace = _make_trace(fired_rule_names=names)
        fp = fingerprint_from_trace(trace, protein_name="test")
        assert fp.n_fired == 3
        assert set(fp.fired_rules) == set(names)

    def test_custom_rules(self):
        from ibp_enm.rules import ArchetypeRule
        custom_rules = (
            ArchetypeRule(
                instrument="algebraic", archetype="barrel",
                name="custom_1", metric="gap_drift",
                condition=lambda x: x > 0, score=0.1,
                provenance="test",
            ),
            ArchetypeRule(
                instrument="algebraic", archetype="barrel",
                name="custom_2", metric="gap_drift",
                condition=lambda x: x > 0, score=0.1,
                provenance="test",
            ),
        )
        trace = _make_trace(fired_rule_names=["custom_1"])
        fp = fingerprint_from_trace(trace, rules=custom_rules, protein_name="test")
        assert fp.bits == (1, 0)
        assert fp.rule_names == ("custom_1", "custom_2")


# ═══════════════════════════════════════════════════════════════════
# Hamming distance tests
# ═══════════════════════════════════════════════════════════════════

class TestHammingDistance:
    """Tests for Hamming distance computation."""

    def test_identical(self):
        fp = Fingerprint(bits=(1, 0, 1), rule_names=("a", "b", "c"))
        assert hamming_distance(fp, fp) == 0

    def test_opposite(self):
        fp1 = Fingerprint(bits=(1, 0, 1), rule_names=("a", "b", "c"))
        fp2 = Fingerprint(bits=(0, 1, 0), rule_names=("a", "b", "c"))
        assert hamming_distance(fp1, fp2) == 3

    def test_one_diff(self):
        fp1 = Fingerprint(bits=(1, 0, 1), rule_names=("a", "b", "c"))
        fp2 = Fingerprint(bits=(1, 0, 0), rule_names=("a", "b", "c"))
        assert hamming_distance(fp1, fp2) == 1

    def test_length_mismatch_raises(self):
        fp1 = Fingerprint(bits=(1,), rule_names=("a",))
        fp2 = Fingerprint(bits=(1, 0), rule_names=("a", "b"))
        with pytest.raises(ValueError, match="lengths differ"):
            hamming_distance(fp1, fp2)


class TestDistanceMatrix:
    """Tests for pairwise distance matrix."""

    def test_empty(self):
        mat, names = distance_matrix([])
        assert mat.shape == (0, 0)
        assert names == []

    def test_single(self):
        fp = Fingerprint(bits=(1, 0), rule_names=("a", "b"), protein_name="p1")
        mat, names = distance_matrix([fp])
        assert mat.shape == (1, 1)
        assert mat[0, 0] == 0
        assert names == ["p1"]

    def test_three_proteins(self):
        fps = [
            Fingerprint(bits=(1, 0, 0), rule_names=("a", "b", "c"), protein_name="p1"),
            Fingerprint(bits=(1, 1, 0), rule_names=("a", "b", "c"), protein_name="p2"),
            Fingerprint(bits=(0, 1, 1), rule_names=("a", "b", "c"), protein_name="p3"),
        ]
        mat, names = distance_matrix(fps)
        assert mat.shape == (3, 3)
        assert mat[0, 1] == 1  # p1 vs p2
        assert mat[0, 2] == 3  # p1 vs p3
        assert mat[1, 2] == 2  # p2 vs p3
        # Symmetric
        assert mat[1, 0] == mat[0, 1]
        assert names == ["p1", "p2", "p3"]


# ═══════════════════════════════════════════════════════════════════
# Instrument collinearity tests
# ═══════════════════════════════════════════════════════════════════

class TestInstrumentCollinearity:
    """Tests for instrument-triple collinearity detection."""

    def test_fano_lines_count(self):
        assert len(FANO_LINES) == 7
        for line in FANO_LINES:
            assert len(line) == 3

    def test_seven_instruments(self):
        assert len(INSTRUMENT_NAMES) == 7

    def test_empty_traces(self):
        result = check_collinearity([])
        assert result == []

    def test_all_agreeing(self):
        """All instruments always agree → high collinearity."""
        scores = {"barrel": 0.8, "enzyme_active": 0.1, "globin": 0.05,
                  "dumbbell": 0.03, "allosteric": 0.02}
        per_inst = {inst: dict(scores) for inst in INSTRUMENT_NAMES}
        traces = [
            _make_trace(per_instrument_votes=per_inst)
            for _ in range(10)
        ]
        results = check_collinearity(traces)
        assert len(results) == 35  # C(7,3) = 35 triples
        # All should have 100% agreement
        assert all(r.agreement_rate == 1.0 for r in results)

    def test_sorted_by_z_score(self):
        traces = [_make_trace() for _ in range(5)]
        results = check_collinearity(traces)
        z_scores = [r.z_score for r in results]
        assert z_scores == sorted(z_scores, reverse=True)

    def test_collinearity_summary(self):
        c = InstrumentCollinearity(
            triple=("algebraic", "musical", "fick"),
            agreement_rate=0.8,
            expected_rate=0.3,
            collinear=True,
            z_score=3.5,
        )
        s = c.summary()
        assert "●" in s  # collinear marker
        assert "algebraic" in s

    def test_non_collinear_summary(self):
        c = InstrumentCollinearity(
            triple=("algebraic", "musical", "fick"),
            agreement_rate=0.3,
            expected_rate=0.3,
            collinear=False,
            z_score=0.1,
        )
        s = c.summary()
        assert "○" in s  # non-collinear marker


# ═══════════════════════════════════════════════════════════════════
# Firing lattice tests
# ═══════════════════════════════════════════════════════════════════

class TestFiringLattice:
    """Tests for the firing-pattern lattice."""

    def test_empty(self):
        lattice = build_firing_lattice([])
        assert lattice.n_patterns == 0
        assert lattice.n_edges == 0

    def test_single_pattern(self):
        fp = Fingerprint(
            bits=(1, 0, 1), rule_names=("a", "b", "c"), protein_name="p1",
        )
        lattice = build_firing_lattice([fp])
        assert lattice.n_patterns == 1
        assert lattice.proteins_at(frozenset({"a", "c"})) == ["p1"]

    def test_subset_inclusion(self):
        fp1 = Fingerprint(
            bits=(1, 0, 0), rule_names=("a", "b", "c"), protein_name="p1",
        )
        fp2 = Fingerprint(
            bits=(1, 1, 0), rule_names=("a", "b", "c"), protein_name="p2",
        )
        fp3 = Fingerprint(
            bits=(1, 1, 1), rule_names=("a", "b", "c"), protein_name="p3",
        )
        lattice = build_firing_lattice([fp1, fp2, fp3])
        assert lattice.n_patterns == 3
        # Hasse edges: {a} → {a,b} → {a,b,c}
        assert lattice.n_edges == 2

    def test_shared_pattern(self):
        fp1 = Fingerprint(
            bits=(1, 1, 0), rule_names=("a", "b", "c"), protein_name="p1",
        )
        fp2 = Fingerprint(
            bits=(1, 1, 0), rule_names=("a", "b", "c"), protein_name="p2",
        )
        lattice = build_firing_lattice([fp1, fp2])
        assert lattice.n_patterns == 1
        assert set(lattice.proteins_at(frozenset({"a", "b"}))) == {"p1", "p2"}

    def test_lattice_properties(self):
        fp1 = Fingerprint(bits=(1, 0, 0), rule_names=("a", "b", "c"), protein_name="p1")
        fp2 = Fingerprint(bits=(0, 1, 0), rule_names=("a", "b", "c"), protein_name="p2")
        fp3 = Fingerprint(bits=(1, 1, 1), rule_names=("a", "b", "c"), protein_name="p3")
        lattice = build_firing_lattice([fp1, fp2, fp3])
        assert lattice.height >= 1
        assert lattice.width >= 1

    def test_summary(self):
        fp = Fingerprint(bits=(1, 0), rule_names=("a", "b"), protein_name="p1")
        lattice = build_firing_lattice([fp])
        s = lattice.summary()
        assert "FiringLattice" in s
        assert "patterns=1" in s


# ═══════════════════════════════════════════════════════════════════
# Threshold sensitivity tests
# ═══════════════════════════════════════════════════════════════════

class TestThresholdSensitivity:
    """Tests for threshold sensitivity analysis."""

    def test_empty_traces(self):
        result = threshold_sensitivity([])
        assert result.n_thresholds == 0
        assert result.mean_sensitivity == 0.0
        assert result.top_sensitive == []

    def test_single_trace(self):
        trace = _make_trace()
        result = threshold_sensitivity([trace])
        assert result.n_thresholds > 0
        assert result.mean_sensitivity >= 0.0
        assert len(result.top_sensitive) <= 20
        assert result.principal_components is None  # need >=2 traces for PCA

    def test_multiple_traces_with_pca(self):
        traces = [_make_trace(margin=m) for m in [0.05, 0.15, 0.3]]
        result = threshold_sensitivity(traces)
        assert result.n_thresholds > 0
        assert result.principal_components is not None
        assert result.explained_variance is not None
        assert len(result.principal_components) <= 3

    def test_active_lens_boosts_sensitivity(self):
        # Trace with active enzyme lens should have higher enzyme_lens sensitivity
        trace_active = _make_trace()
        # Replace lens traces with an active one
        trace_active = ClassificationTrace(
            identity=trace_active.identity,
            scores=trace_active.scores,
            per_instrument_votes=trace_active.per_instrument_votes,
            rule_firings=trace_active.rule_firings,
            consensus_scores=trace_active.consensus_scores,
            disagreement_scores=trace_active.disagreement_scores,
            context_boost=trace_active.context_boost,
            context_signals=trace_active.context_signals,
            alpha_meta=trace_active.alpha_meta,
            meta_state=trace_active.meta_state,
            lens_traces=[
                LensTrace(lens_name="EnzymeLens", activated=True, boost=0.08),
                LensTrace(lens_name="HingeLens", activated=False),
                LensTrace(lens_name="BarrelPenaltyLens", activated=False),
            ],
            n_residues=200,
            n_instruments=7,
        )
        trace_inactive = _make_trace()

        s_active = threshold_sensitivity([trace_active])
        s_inactive = threshold_sensitivity([trace_inactive])

        # enzyme_lens section should be more sensitive when lens is active
        assert s_active.sections.get("enzyme_lens", 0) >= s_inactive.sections.get("enzyme_lens", 0)

    def test_shape_signature(self):
        traces = [_make_trace() for _ in range(3)]
        result = threshold_sensitivity(traces)
        assert len(result.shape_signature) <= 5
        # Each element should be a valid threshold key
        from ibp_enm.thresholds import DEFAULT_THRESHOLDS
        for key in result.shape_signature:
            assert key in DEFAULT_THRESHOLDS

    def test_summary(self):
        trace = _make_trace()
        result = threshold_sensitivity([trace])
        s = result.summary()
        assert "ThresholdSensitivity" in s

    def test_sections_nonempty(self):
        trace = _make_trace()
        result = threshold_sensitivity([trace])
        assert len(result.sections) > 0
        # Should have at least the main sections
        expected_sections = {"meta_fick", "ctx_boost", "enzyme_lens",
                           "hinge_lens", "barrel_penalty", "renorm"}
        assert set(result.sections.keys()) == expected_sections


# ═══════════════════════════════════════════════════════════════════
# Integration tests
# ═══════════════════════════════════════════════════════════════════

class TestIntegration:
    """End-to-end tests combining multiple algebra components."""

    def test_fingerprint_to_lattice_roundtrip(self):
        """Extract fingerprints → build lattice → verify structure."""
        traces = [
            _make_trace(fired_rule_names=_sample_rule_names(3), identity="barrel"),
            _make_trace(fired_rule_names=_sample_rule_names(5), identity="enzyme_active"),
            _make_trace(fired_rule_names=_sample_rule_names(3), identity="barrel"),
        ]
        fps = [
            fingerprint_from_trace(t, protein_name=f"p{i}")
            for i, t in enumerate(traces)
        ]
        lattice = build_firing_lattice(fps)
        # Two fingerprints with 3 rules are identical → same pattern
        assert lattice.n_patterns <= 2

    def test_full_pipeline(self):
        """Full pipeline: trace → fingerprint → distance → lattice → sensitivity."""
        traces = [
            _make_trace(identity="barrel", margin=0.1),
            _make_trace(identity="enzyme_active", margin=0.2),
            _make_trace(identity="globin", margin=0.3),
        ]
        fps = [fingerprint_from_trace(t, protein_name=f"p{i}") for i, t in enumerate(traces)]
        mat, names = distance_matrix(fps)
        assert mat.shape == (3, 3)

        lattice = build_firing_lattice(fps)
        assert lattice.n_patterns >= 1

        sens = threshold_sensitivity(traces)
        assert sens.n_thresholds > 0

        collin = check_collinearity(traces)
        assert len(collin) == 35  # C(7,3)
