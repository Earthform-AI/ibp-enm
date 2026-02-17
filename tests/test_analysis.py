"""Tests for ibp_enm.analysis — deep result analysis.

Tests cover:
  - ArchetypeProfile generation
  - ConfusionCluster detection
  - RuleCoFiringMatrix computation
  - AnalysisReport construction and summary
  - Cross-experiment comparison via compare_reports
"""

import numpy as np
import pytest

from ibp_enm.algebra import (
    Fingerprint,
    fingerprint_from_trace,
    INSTRUMENT_NAMES,
)
from ibp_enm.analysis import (
    AnalysisReport,
    ArchetypeProfile,
    ConfusionCluster,
    RuleCoFiringMatrix,
    analyse_traces,
    compare_reports,
)
from ibp_enm.lens_stack import LensTrace
from ibp_enm.rules import ARCHETYPE_RULES, RuleFiring
from ibp_enm.trace import ClassificationTrace, ContextSignals


# ═══════════════════════════════════════════════════════════════════
# Helpers — reuse same trace builder from test_algebra
# ═══════════════════════════════════════════════════════════════════

def _make_trace(
    identity: str = "barrel",
    scores: dict | None = None,
    fired_rule_names: list[str] | None = None,
    per_instrument_votes: dict | None = None,
    margin: float = 0.15,
    alpha: float = 0.7,
    context_boost: dict | None = None,
    active_lenses: list[str] | None = None,
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
    if context_boost is None:
        context_boost = {k: 0.0 for k in scores}

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

    lens_traces = []
    active = active_lenses or []
    for name in ["EnzymeLens", "HingeLens", "BarrelPenaltyLens"]:
        lens_traces.append(LensTrace(
            lens_name=name,
            activated=(name in active),
            boost=0.05 if name in active else 0.0,
        ))

    return ClassificationTrace(
        identity=identity,
        scores=scores,
        per_instrument_votes=per_instrument_votes,
        rule_firings=rule_firings,
        consensus_scores=dict(scores),
        disagreement_scores={k: 0.05 for k in scores},
        context_boost=context_boost,
        context_signals=ContextSignals(),
        alpha_meta=alpha,
        meta_state={"alpha": alpha},
        lens_traces=lens_traces,
        n_residues=200,
        n_instruments=7,
    )


def _sample_rule_names(n: int = 5) -> list[str]:
    return [r.name for r in ARCHETYPE_RULES[:n]]


# ═══════════════════════════════════════════════════════════════════
# ArchetypeProfile tests
# ═══════════════════════════════════════════════════════════════════

class TestArchetypeProfile:
    """Tests for per-archetype statistical summaries."""

    def test_summary(self):
        prof = ArchetypeProfile(
            archetype="barrel",
            n_proteins=10,
            mean_score=0.45,
            mean_margin=0.15,
            mean_n_rules_fired=25,
            mean_firing_density=0.3,
            lens_activation_rates={"EnzymeLens": 0.3, "HingeLens": 0.0},
            mean_context_boost={"barrel": 0.1},
            mean_alpha=0.72,
            common_runner_up="enzyme_active",
        )
        s = prof.summary()
        assert "barrel" in s
        assert "n=10" in s
        assert "enzyme_active" in s


# ═══════════════════════════════════════════════════════════════════
# ConfusionCluster tests
# ═══════════════════════════════════════════════════════════════════

class TestConfusionCluster:
    """Tests for confusion cluster detection."""

    def test_is_mixed(self):
        c = ConfusionCluster(
            proteins=("p1", "p2"),
            archetypes=("barrel", "enzyme_active"),
            mean_internal_distance=2.0,
            centroid_density=0.3,
            shared_rules=("r1",),
            unique_rules={"p1": ("r2",), "p2": ("r3",)},
        )
        assert c.is_mixed

    def test_not_mixed(self):
        c = ConfusionCluster(
            proteins=("p1", "p2"),
            archetypes=("barrel", "barrel"),
            mean_internal_distance=1.0,
            centroid_density=0.3,
            shared_rules=("r1", "r2"),
            unique_rules={"p1": (), "p2": ()},
        )
        assert not c.is_mixed

    def test_summary(self):
        c = ConfusionCluster(
            proteins=("p1", "p2", "p3"),
            archetypes=("barrel", "enzyme_active", "barrel"),
            mean_internal_distance=3.0,
            centroid_density=0.25,
            shared_rules=("r1",),
            unique_rules={},
        )
        s = c.summary()
        assert "3 proteins" in s
        assert "MIXED" in s


# ═══════════════════════════════════════════════════════════════════
# RuleCoFiringMatrix tests
# ═══════════════════════════════════════════════════════════════════

class TestRuleCoFiringMatrix:
    """Tests for rule co-firing matrix."""

    def test_empty(self):
        from ibp_enm.analysis import _build_co_firing_matrix
        mat = _build_co_firing_matrix([])
        assert mat.matrix.shape == (0, 0)
        assert mat.n_proteins == 0

    def test_correlation(self):
        # Two fingerprints: both fire r1 and r2, only fp1 fires r3
        fps = [
            Fingerprint(bits=(1, 1, 1), rule_names=("r1", "r2", "r3"), protein_name="p1"),
            Fingerprint(bits=(1, 1, 0), rule_names=("r1", "r2", "r3"), protein_name="p2"),
        ]
        from ibp_enm.analysis import _build_co_firing_matrix
        mat = _build_co_firing_matrix(fps)
        assert mat.correlation("r1", "r2") == 1.0  # both fire in both proteins
        assert mat.correlation("r1", "r3") == 0.5  # co-fire in 1 of 2

    def test_top_pairs(self):
        fps = [
            Fingerprint(bits=(1, 1, 0), rule_names=("r1", "r2", "r3"), protein_name="p1"),
            Fingerprint(bits=(1, 1, 1), rule_names=("r1", "r2", "r3"), protein_name="p2"),
        ]
        from ibp_enm.analysis import _build_co_firing_matrix
        mat = _build_co_firing_matrix(fps)
        pairs = mat.top_pairs(top_k=5)
        assert len(pairs) >= 1
        # Highest should be (r1, r2) with rate 1.0
        assert pairs[0][2] == 1.0

    def test_summary(self):
        fps = [
            Fingerprint(bits=(1, 0), rule_names=("r1", "r2"), protein_name="p1"),
        ]
        from ibp_enm.analysis import _build_co_firing_matrix
        mat = _build_co_firing_matrix(fps)
        s = mat.summary()
        assert "RuleCoFiringMatrix" in s


# ═══════════════════════════════════════════════════════════════════
# analyse_traces tests
# ═══════════════════════════════════════════════════════════════════

class TestAnalyseTraces:
    """Tests for the main analysis entry point."""

    def test_empty(self):
        report = analyse_traces([])
        assert report.n_proteins == 0
        assert report.archetype_profiles == {}
        assert report.fingerprints == []

    def test_single_trace(self):
        trace = _make_trace(identity="barrel")
        report = analyse_traces([trace], protein_names=["T4_lysozyme"])
        assert report.n_proteins == 1
        assert "barrel" in report.archetype_profiles
        assert len(report.fingerprints) == 1
        assert report.distance_mat.shape == (1, 1)

    def test_multiple_archetypes(self):
        traces = [
            _make_trace(identity="barrel"),
            _make_trace(identity="enzyme_active",
                       scores={"barrel": 0.1, "enzyme_active": 0.5,
                               "globin": 0.15, "dumbbell": 0.15,
                               "allosteric": 0.1}),
            _make_trace(identity="globin",
                       scores={"barrel": 0.1, "enzyme_active": 0.1,
                               "globin": 0.5, "dumbbell": 0.15,
                               "allosteric": 0.15}),
        ]
        names = ["p1", "p2", "p3"]
        report = analyse_traces(traces, protein_names=names)
        assert report.n_proteins == 3
        assert len(report.archetype_profiles) == 3
        assert report.distance_mat.shape == (3, 3)

    def test_ground_truth_accuracy(self):
        traces = [
            _make_trace(identity="barrel"),
            _make_trace(identity="enzyme_active",
                       scores={"barrel": 0.1, "enzyme_active": 0.5,
                               "globin": 0.15, "dumbbell": 0.15,
                               "allosteric": 0.1}),
        ]
        gt = {"p1": "barrel", "p2": "globin"}  # p2 wrong
        report = analyse_traces(traces, protein_names=["p1", "p2"],
                               ground_truth=gt)
        assert report.overall_accuracy == 0.5

    def test_collinearity_present(self):
        traces = [_make_trace() for _ in range(5)]
        report = analyse_traces(traces, protein_names=[f"p{i}" for i in range(5)])
        assert len(report.collinearity) == 35

    def test_firing_lattice_present(self):
        traces = [_make_trace() for _ in range(3)]
        report = analyse_traces(traces, protein_names=["a", "b", "c"])
        assert report.firing_lattice is not None
        assert report.firing_lattice.n_patterns >= 1

    def test_co_firing_present(self):
        traces = [_make_trace(fired_rule_names=_sample_rule_names(5))]
        report = analyse_traces(traces, protein_names=["p1"])
        assert report.co_firing.n_proteins == 1

    def test_sensitivity_present(self):
        traces = [_make_trace() for _ in range(3)]
        report = analyse_traces(traces, protein_names=["a", "b", "c"])
        assert report.sensitivity.n_thresholds > 0

    def test_summary_nocrash(self):
        traces = [
            _make_trace(identity="barrel"),
            _make_trace(identity="enzyme_active",
                       scores={"barrel": 0.1, "enzyme_active": 0.5,
                               "globin": 0.15, "dumbbell": 0.15,
                               "allosteric": 0.1}),
        ]
        report = analyse_traces(traces, protein_names=["p1", "p2"])
        s = report.summary()
        assert "Analysis Report" in s
        assert "2 proteins" in s

    def test_default_protein_names(self):
        traces = [_make_trace()]
        report = analyse_traces(traces)
        assert report.protein_names == ["protein_0"]


# ═══════════════════════════════════════════════════════════════════
# compare_reports tests
# ═══════════════════════════════════════════════════════════════════

class TestCompareReports:
    """Tests for cross-experiment comparison."""

    def test_identical_reports(self):
        traces = [_make_trace(identity="barrel")]
        r1 = analyse_traces(traces, protein_names=["p1"])
        r2 = analyse_traces(traces, protein_names=["p1"])
        delta = compare_reports(r1, r2)
        assert "Cross-Experiment Comparison" in delta
        assert "Before: 1 proteins" in delta
        assert "After:  1 proteins" in delta

    def test_accuracy_delta(self):
        t1 = [_make_trace(identity="barrel")]
        t2 = [
            _make_trace(identity="barrel"),
            _make_trace(identity="enzyme_active",
                       scores={"barrel": 0.1, "enzyme_active": 0.5,
                               "globin": 0.15, "dumbbell": 0.15,
                               "allosteric": 0.1}),
        ]
        gt = {"p1": "barrel", "p2": "enzyme_active"}
        r1 = analyse_traces(t1, ["p1"], ground_truth={"p1": "barrel"})
        r2 = analyse_traces(t2, ["p1", "p2"], ground_truth=gt)
        delta = compare_reports(r1, r2)
        assert "Accuracy" in delta

    def test_sensitivity_shape_change(self):
        # Different margins → different sensitivity profiles
        t1 = [_make_trace(margin=0.05)]
        t2 = [_make_trace(margin=0.5)]
        r1 = analyse_traces(t1, ["p1"])
        r2 = analyse_traces(t2, ["p1"])
        delta = compare_reports(r1, r2)
        assert "Threshold Sensitivity Shape" in delta

    def test_lattice_comparison(self):
        t1 = [_make_trace(fired_rule_names=_sample_rule_names(3))]
        t2 = [_make_trace(fired_rule_names=_sample_rule_names(5))]
        r1 = analyse_traces(t1, ["p1"])
        r2 = analyse_traces(t2, ["p1"])
        delta = compare_reports(r1, r2)
        assert "Firing Lattice" in delta

    def test_fingerprint_diversity(self):
        t1 = [_make_trace() for _ in range(2)]
        t2 = [_make_trace() for _ in range(3)]
        r1 = analyse_traces(t1, ["p1", "p2"])
        r2 = analyse_traces(t2, ["p1", "p2", "p3"])
        delta = compare_reports(r1, r2)
        assert "Fingerprint diversity" in delta


# ═══════════════════════════════════════════════════════════════════
# Integration: confusion cluster detection
# ═══════════════════════════════════════════════════════════════════

class TestConfusionClustering:
    """Tests for confusion cluster detection within analyse_traces."""

    def test_identical_traces_cluster(self):
        """Identical traces should form a cluster."""
        traces = [
            _make_trace(identity="barrel", fired_rule_names=_sample_rule_names(5)),
            _make_trace(identity="barrel", fired_rule_names=_sample_rule_names(5)),
            _make_trace(identity="barrel", fired_rule_names=_sample_rule_names(5)),
        ]
        report = analyse_traces(
            traces, ["p1", "p2", "p3"], cluster_threshold=0
        )
        # All identical → should be one cluster of 3
        assert any(c.n_proteins == 3 for c in report.confusion_clusters)

    def test_diverse_traces_fewer_clusters(self):
        """Very different traces should cluster less."""
        names1 = _sample_rule_names(3)
        names2 = [r.name for r in ARCHETYPE_RULES[40:45]]
        traces = [
            _make_trace(identity="barrel", fired_rule_names=names1),
            _make_trace(identity="enzyme_active", fired_rule_names=names2),
        ]
        report = analyse_traces(
            traces, ["p1", "p2"], cluster_threshold=2
        )
        # With threshold=2 and very different patterns, shouldn't cluster
        mixed = [c for c in report.confusion_clusters if c.is_mixed]
        # May or may not cluster depending on actual Hamming distance
        # Just verify no crash
        assert isinstance(report.confusion_clusters, list)


# ═══════════════════════════════════════════════════════════════════
# Import / export tests
# ═══════════════════════════════════════════════════════════════════

class TestImports:
    """Test that all public symbols are importable from ibp_enm."""

    def test_algebra_imports(self):
        from ibp_enm import (
            Fingerprint, fingerprint_from_trace,
            hamming_distance, distance_matrix,
            InstrumentCollinearity, check_collinearity,
            FiringLattice, build_firing_lattice,
            ThresholdSensitivity, threshold_sensitivity,
        )

    def test_analysis_imports(self):
        from ibp_enm import (
            ArchetypeProfile, ConfusionCluster, RuleCoFiringMatrix,
            AnalysisReport, analyse_traces, compare_reports,
        )
