"""Tests for benchmark module (ibp_enm.benchmark).

Tests the corpus definitions, BenchmarkReport analysis methods,
serialisation, and delta comparison.  These are all offline tests
that do not require network access.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from ibp_enm.benchmark import (
    ProteinEntry,
    ProteinResult,
    BenchmarkReport,
    BenchmarkRunner,
    ORIGINAL_CORPUS,
    EXPANDED_CORPUS,
)


# ── Fixtures ────────────────────────────────────────────────────

def _make_result(name, pdb, chain, archetype, predicted,
                 time_s=1.0, error=None):
    """Helper to create a ProteinResult."""
    entry = ProteinEntry(name, pdb, chain, archetype)
    correct = predicted == archetype and error is None
    return ProteinResult(
        entry=entry,
        predicted=predicted,
        scores={archetype: 0.6, predicted: 0.4}
        if predicted != archetype else {archetype: 0.8},
        correct=correct,
        time_s=time_s,
        error=error,
    )


@pytest.fixture
def perfect_report():
    """A report where everything is correct."""
    results = [
        _make_result("P1", "1ABC", "A", "enzyme_active", "enzyme_active"),
        _make_result("P2", "2DEF", "A", "barrel", "barrel"),
        _make_result("P3", "3GHI", "A", "globin", "globin"),
        _make_result("P4", "4JKL", "A", "dumbbell", "dumbbell"),
        _make_result("P5", "5MNO", "A", "allosteric", "allosteric"),
    ]
    return BenchmarkReport(results=results, total_time_s=5.0)


@pytest.fixture
def mixed_report():
    """A report with some misclassifications and one error."""
    results = [
        _make_result("P1", "1ABC", "A", "enzyme_active", "enzyme_active"),
        _make_result("P2", "2DEF", "A", "barrel", "barrel"),
        _make_result("P3", "3GHI", "A", "globin", "enzyme_active"),  # wrong
        _make_result("P4", "4JKL", "A", "dumbbell", "barrel"),  # wrong
        _make_result("P5", "5MNO", "A", "allosteric", "allosteric"),
        _make_result("P6", "6PQR", "A", "enzyme_active", "error",
                     error="ValueError: Too few residues"),
    ]
    return BenchmarkReport(results=results, total_time_s=10.0)


# ═══════════════════════════════════════════════════════════════════
# Corpus tests
# ═══════════════════════════════════════════════════════════════════

class TestCorpus:
    """Test the benchmark corpus definitions."""

    def test_original_corpus_has_12(self):
        assert len(ORIGINAL_CORPUS) == 12

    def test_expanded_corpus_has_52(self):
        assert len(EXPANDED_CORPUS) == 52

    def test_expanded_contains_original(self):
        orig_names = {e.name for e in ORIGINAL_CORPUS}
        expanded_names = {e.name for e in EXPANDED_CORPUS}
        assert orig_names.issubset(expanded_names)

    def test_all_entries_are_protein_entry(self):
        for entry in EXPANDED_CORPUS:
            assert isinstance(entry, ProteinEntry)
            assert entry.name
            assert entry.pdb_id
            assert entry.chain
            assert entry.archetype in {
                "enzyme_active", "barrel", "globin",
                "dumbbell", "allosteric",
            }

    def test_five_archetypes_represented(self):
        archetypes = {e.archetype for e in EXPANDED_CORPUS}
        assert archetypes == {
            "enzyme_active", "barrel", "globin",
            "dumbbell", "allosteric",
        }

    def test_original_matches_ground_truth(self):
        """Original corpus entries should match archetypes.GROUND_TRUTH."""
        from ibp_enm.archetypes import GROUND_TRUTH
        for entry in ORIGINAL_CORPUS:
            if entry.name in GROUND_TRUTH:
                assert entry.archetype == GROUND_TRUTH[entry.name], \
                    f"{entry.name}: corpus says {entry.archetype}, " \
                    f"GROUND_TRUTH says {GROUND_TRUTH[entry.name]}"

    def test_protein_entry_is_frozen(self):
        entry = ORIGINAL_CORPUS[0]
        with pytest.raises(AttributeError):
            entry.name = "modified"


# ═══════════════════════════════════════════════════════════════════
# BenchmarkReport tests
# ═══════════════════════════════════════════════════════════════════

class TestBenchmarkReport:
    """Test BenchmarkReport analysis methods."""

    def test_perfect_accuracy(self, perfect_report):
        assert perfect_report.accuracy == 1.0
        assert perfect_report.n_correct == 5
        assert perfect_report.n_total == 5

    def test_mixed_accuracy(self, mixed_report):
        # 5 valid results (1 error excluded), 3 correct
        assert mixed_report.n_total == 5  # excludes error
        assert mixed_report.n_correct == 3
        assert abs(mixed_report.accuracy - 0.6) < 1e-10

    def test_errors_counted(self, mixed_report):
        assert len(mixed_report.errors) == 1

    def test_per_archetype_perfect(self, perfect_report):
        pa = perfect_report.per_archetype
        for arch, info in pa.items():
            assert info["accuracy"] == 1.0
            assert info["misclassified_as"] == {}

    def test_per_archetype_mixed(self, mixed_report):
        pa = mixed_report.per_archetype
        assert pa["globin"]["accuracy"] == 0.0  # 0/1 correct
        assert pa["dumbbell"]["accuracy"] == 0.0  # 0/1 correct
        assert pa["enzyme_active"]["accuracy"] == 1.0  # 1/1 correct

    def test_confusion_matrix_shape(self, mixed_report):
        mat, labels = mixed_report.confusion_matrix
        n = len(labels)
        assert mat.shape == (n, n)
        assert n == 5  # 5 archetypes

    def test_confusion_matrix_diagonal_for_perfect(self, perfect_report):
        mat, labels = perfect_report.confusion_matrix
        for i in range(len(labels)):
            assert mat[i, i] >= 0
        # Off-diagonal should be 0 for perfect
        np.fill_diagonal(mat, 0)
        assert mat.sum() == 0

    def test_false_predictions(self, mixed_report):
        fp = mixed_report.false_predictions
        # P3 (globin) → enzyme_active, P4 (dumbbell) → barrel
        assert "enzyme_active" in fp or "barrel" in fp

    def test_summary_string(self, mixed_report):
        s = mixed_report.summary()
        assert "3/5" in s
        assert "60" in s  # 60.0% or 60%

    # ── Delta comparison ────────────────────────────────────────

    def test_delta_same_report(self, perfect_report):
        d = perfect_report.delta(perfect_report)
        assert "0%" in d or "+0" in d

    def test_delta_improvement(self, mixed_report, perfect_report):
        d = mixed_report.delta(perfect_report)
        assert "Fixed" in d or "100%" in d

    def test_delta_regression(self, perfect_report, mixed_report):
        d = perfect_report.delta(mixed_report)
        assert "Regressed" in d

    # ── Serialisation ───────────────────────────────────────────

    def test_to_dict(self, mixed_report):
        d = mixed_report.to_dict()
        assert d["n_correct"] == 3
        assert d["n_total"] == 5
        assert len(d["results"]) == 6  # includes error

    def test_save_load_round_trip(self, mixed_report, tmp_path):
        path = tmp_path / "report.json"
        mixed_report.save(path)
        loaded = BenchmarkReport.load(path)
        assert loaded.n_correct == mixed_report.n_correct
        assert loaded.n_total == mixed_report.n_total
        assert abs(loaded.accuracy - mixed_report.accuracy) < 1e-10
        assert len(loaded.errors) == len(mixed_report.errors)

    def test_load_preserves_entries(self, mixed_report, tmp_path):
        path = tmp_path / "report.json"
        mixed_report.save(path)
        loaded = BenchmarkReport.load(path)
        for orig, loaded_r in zip(mixed_report.results, loaded.results):
            assert orig.entry.name == loaded_r.entry.name
            assert orig.entry.pdb_id == loaded_r.entry.pdb_id
            assert orig.predicted == loaded_r.predicted
            assert orig.correct == loaded_r.correct

    def test_timestamp_auto_generated(self, perfect_report):
        assert perfect_report.timestamp != ""
        assert "T" in perfect_report.timestamp  # ISO format


# ═══════════════════════════════════════════════════════════════════
# BenchmarkRunner tests (offline — no network)
# ═══════════════════════════════════════════════════════════════════

class TestBenchmarkRunner:
    """Test BenchmarkRunner construction and configuration."""

    def test_default_corpus_is_expanded(self):
        runner = BenchmarkRunner()
        assert len(runner.corpus) == 52

    def test_custom_corpus(self):
        corpus = ORIGINAL_CORPUS[:3]
        runner = BenchmarkRunner(corpus=corpus)
        assert len(runner.corpus) == 3

    def test_cache_dir_creates_cache(self, tmp_path):
        runner = BenchmarkRunner(
            corpus=[], cache_dir=tmp_path / "cache")
        assert runner.cache is not None

    def test_no_cache_by_default(self):
        runner = BenchmarkRunner(corpus=[])
        assert runner.cache is None

    def test_empty_corpus_run(self):
        """Running with empty corpus should return empty report."""
        runner = BenchmarkRunner(corpus=[])
        report = runner.run()
        assert report.n_total == 0
        assert report.accuracy == 0.0

    def test_hooks_called(self):
        """Hooks should be called during run."""
        started = []
        done = []
        errors = []

        runner = BenchmarkRunner(corpus=[])
        report = runner.run(
            on_protein_start=lambda e: started.append(e),
            on_protein_done=lambda e, r: done.append(e),
            on_error=lambda e, x: errors.append(e),
        )
        # Empty corpus → no hooks called
        assert started == []
        assert done == []
        assert errors == []
