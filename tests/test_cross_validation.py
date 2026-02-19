"""Tests for CrossValidator and CrossValidationResult (ibp_enm.benchmark).

All tests are offline — they mock ``run_single_protein`` to avoid
network access and long ENM computations.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from ibp_enm.benchmark import (
    CrossValidationResult,
    CrossValidator,
    ProteinEntry,
    EXPANDED_CORPUS,
    LARGE_CORPUS,
    ParameterUsefulnessResult,
    ParameterUsefulnessAnalyzer,
)
from ibp_enm.thresholds import ThresholdRegistry, DEFAULT_THRESHOLDS


# ── Helpers ─────────────────────────────────────────────────────

def _fake_profile(archetype: str):
    """Create a minimal mock ThermoReactionProfile."""
    p = MagicMock()
    p.archetype_vote.return_value = archetype
    return p


def _make_run_result(
    name: str,
    predicted: str,
    true_archetype: str,
    scores: dict | None = None,
):
    """Build a dict mimicking ``run_single_protein`` output.

    Includes ``profiles``, ``evals``, ``evecs``, ``domain_labels``,
    ``contacts`` so the perturbation analysis can run.
    """
    if scores is None:
        scores = {
            "allosteric": 0.1,
            "barrel": 0.1,
            "dumbbell": 0.1,
            "enzyme_active": 0.1,
            "globin": 0.1,
        }
        scores[predicted] = 0.8

    profiles = [_fake_profile(predicted) for _ in range(7)]

    return {
        "name": name,
        "pdb_id": "XXXX",
        "N": 100,
        "band_identity": predicted,
        "band_result": {
            "identity": {
                "identity": predicted,
                "scores": scores,
            },
            "profiles": profiles,
        },
        "evals": np.ones(10),
        "evecs": np.eye(10),
        "domain_labels": np.zeros(100, dtype=int),
        "contacts": {(0, 1): 1.0},
        "true_archetype": true_archetype,
    }


MINI_CORPUS = [
    ProteinEntry("Lysozyme", "2LZM", "A", "enzyme_active"),
    ProteinEntry("Hemoglobin_A", "1HHO", "A", "globin"),
    ProteinEntry("TIM_barrel", "7TIM", "A", "barrel"),
    ProteinEntry("Adenylate_kinase", "4AKE", "A", "dumbbell"),
    ProteinEntry("GroEL", "1GRL", "A", "allosteric"),
]


# ── CrossValidationResult tests ────────────────────────────────

class TestCrossValidationResult:
    def test_default_values(self):
        r = CrossValidationResult()
        assert r.n_proteins == 0
        assert r.accuracy == 0.0
        assert r.fragile_proteins == []
        assert r.stability_scores is None
        assert r.perturbation_accuracy is None

    def test_summary_format(self):
        r = CrossValidationResult(
            n_proteins=4,
            n_correct=3,
            accuracy=0.75,
            per_protein=[
                {
                    "name": "P1", "expected": "globin",
                    "predicted": "globin", "correct": True,
                    "margin": 0.2, "rank": 1,
                },
                {
                    "name": "P2", "expected": "barrel",
                    "predicted": "barrel", "correct": True,
                    "margin": 0.01, "rank": 1,
                },
                {
                    "name": "P3", "expected": "enzyme_active",
                    "predicted": "globin", "correct": False,
                    "margin": -0.15, "rank": 2,
                },
                {
                    "name": "P4", "expected": "dumbbell",
                    "predicted": "dumbbell", "correct": True,
                    "margin": 0.1, "rank": 1,
                },
            ],
            per_archetype={
                "globin": {
                    "correct": 1, "total": 1,
                    "accuracy": 1.0, "mean_margin": 0.2,
                },
                "barrel": {
                    "correct": 1, "total": 1,
                    "accuracy": 1.0, "mean_margin": 0.01,
                },
                "enzyme_active": {
                    "correct": 0, "total": 1,
                    "accuracy": 0.0, "mean_margin": -0.15,
                },
                "dumbbell": {
                    "correct": 1, "total": 1,
                    "accuracy": 1.0, "mean_margin": 0.1,
                },
            },
            fragile_proteins=["P2"],
            robust_proteins=["P1", "P4"],
        )
        summary = r.summary()
        assert "3/4" in summary
        assert "75" in summary  # 75%
        assert "Fragile" in summary
        assert "P2" in summary
        assert "Misclassified" in summary
        assert "P3" in summary

    def test_to_dict_roundtrip(self):
        r = CrossValidationResult(
            n_proteins=10, n_correct=8, accuracy=0.8,
            fragile_proteins=["A", "B"],
        )
        d = r.to_dict()
        assert d["n_proteins"] == 10
        assert d["n_correct"] == 8
        assert d["fragile_proteins"] == ["A", "B"]
        # JSON-serialisable
        json.dumps(d)

    def test_save_json(self, tmp_path):
        r = CrossValidationResult(
            n_proteins=5, n_correct=4, accuracy=0.8,
            per_protein=[],
            per_archetype={},
            fragile_proteins=["X"],
        )
        out = tmp_path / "cv_result.json"
        r.save(str(out))
        data = json.loads(out.read_text())
        assert data["n_proteins"] == 5
        assert data["fragile_proteins"] == ["X"]


# ── CrossValidator tests ────────────────────────────────────────

class TestCrossValidator:
    def test_default_corpus(self):
        """CrossValidator uses EXPANDED_CORPUS by default."""
        cv = CrossValidator()
        assert len(cv.corpus) == len(EXPANDED_CORPUS)

    def test_custom_corpus(self):
        cv = CrossValidator(corpus=MINI_CORPUS)
        assert len(cv.corpus) == 5

    @patch("ibp_enm.band.run_single_protein")
    def test_all_correct_no_fragile(self, mock_run):
        """When all proteins are correctly classified with margin,
        none should be fragile."""
        corpus = MINI_CORPUS[:3]

        def side_effect(pdb_id, chain, name=None, thresholds=None):
            entry = next(e for e in corpus if e.name == name)
            return _make_run_result(
                name, entry.archetype, entry.archetype,
            )

        mock_run.side_effect = side_effect

        cv = CrossValidator(corpus=corpus)
        result = cv.run()

        assert result.n_proteins == 3
        assert result.n_correct == 3
        assert result.accuracy == 1.0
        assert len(result.fragile_proteins) == 0
        assert len(result.robust_proteins) == 3

    @patch("ibp_enm.band.run_single_protein")
    def test_misclassified_reported(self, mock_run):
        """Misclassified proteins have negative margin."""
        corpus = [
            ProteinEntry("P1", "1ABC", "A", "globin"),
            ProteinEntry("P2", "2DEF", "A", "barrel"),
        ]

        def side_effect(pdb_id, chain, name=None, thresholds=None):
            if name == "P1":
                return _make_run_result("P1", "globin", "globin")
            else:
                # Misclassified: predicted enzyme_active, true barrel
                return _make_run_result("P2", "enzyme_active", "barrel")

        mock_run.side_effect = side_effect

        cv = CrossValidator(corpus=corpus)
        result = cv.run()

        assert result.n_correct == 1
        assert result.accuracy == 0.5

        wrong = [pp for pp in result.per_protein if not pp["correct"]]
        assert len(wrong) == 1
        assert wrong[0]["name"] == "P2"
        assert wrong[0]["margin"] < 0

    @patch("ibp_enm.band.run_single_protein")
    def test_fragile_detection(self, mock_run):
        """A protein with margin < 0.05 should be flagged fragile."""
        corpus = [
            ProteinEntry("P1", "1ABC", "A", "globin"),
        ]

        # Correct, but very close scores
        scores = {
            "allosteric": 0.35,
            "barrel": 0.30,
            "dumbbell": 0.20,
            "enzyme_active": 0.15,
            "globin": 0.38,  # wins by 0.03
        }

        def side_effect(pdb_id, chain, name=None, thresholds=None):
            return _make_run_result("P1", "globin", "globin", scores)

        mock_run.side_effect = side_effect

        cv = CrossValidator(corpus=corpus, fragile_margin=0.05)
        result = cv.run()

        assert result.n_correct == 1
        assert "P1" in result.fragile_proteins
        assert len(result.robust_proteins) == 0

    @patch("ibp_enm.band.run_single_protein")
    def test_error_handling(self, mock_run):
        """Proteins that raise exceptions are handled gracefully."""
        corpus = [
            ProteinEntry("Good", "1ABC", "A", "globin"),
            ProteinEntry("Bad", "FAIL", "A", "barrel"),
        ]

        def side_effect(pdb_id, chain, name=None, thresholds=None):
            if name == "Bad":
                raise ValueError("Network error")
            return _make_run_result("Good", "globin", "globin")

        mock_run.side_effect = side_effect

        cv = CrossValidator(corpus=corpus)
        result = cv.run()

        # Error protein excluded from accuracy
        assert result.n_proteins == 1
        assert result.n_correct == 1

        error_pp = [
            pp for pp in result.per_protein if pp["predicted"] == "error"
        ]
        assert len(error_pp) == 1
        assert "error" in error_pp[0]

    @patch("ibp_enm.band.run_single_protein")
    def test_per_archetype_stats(self, mock_run):
        """Per-archetype accuracy and margin stats are computed."""
        corpus = [
            ProteinEntry("G1", "1A", "A", "globin"),
            ProteinEntry("G2", "1B", "A", "globin"),
            ProteinEntry("B1", "2A", "A", "barrel"),
        ]

        def side_effect(pdb_id, chain, name=None, thresholds=None):
            entry = next(e for e in corpus if e.name == name)
            if name == "G2":
                # Misclassify G2
                return _make_run_result(name, "barrel", "globin")
            return _make_run_result(name, entry.archetype, entry.archetype)

        mock_run.side_effect = side_effect

        cv = CrossValidator(corpus=corpus)
        result = cv.run()

        assert "globin" in result.per_archetype
        assert result.per_archetype["globin"]["total"] == 2
        assert result.per_archetype["globin"]["correct"] == 1
        assert result.per_archetype["barrel"]["correct"] == 1

    @patch("ibp_enm.band.run_single_protein")
    def test_perturbation_analysis(self, mock_run):
        """Perturbation analysis runs and returns stability scores."""
        corpus = MINI_CORPUS[:2]

        def side_effect(pdb_id, chain, name=None, thresholds=None):
            entry = next(e for e in corpus if e.name == name)
            return _make_run_result(name, entry.archetype, entry.archetype)

        mock_run.side_effect = side_effect

        # Mock rescore_from_profiles to always return the expected archetype
        with patch(
            "ibp_enm.band.ThermodynamicBand.rescore_from_profiles"
        ) as mock_rescore:
            def rescore_side(profiles, evals=None, evecs=None,
                             domain_labels=None, contacts=None,
                             thresholds=None, **kw):
                return {
                    "identity": corpus[0].archetype,
                    "scores": {corpus[0].archetype: 0.8},
                }

            mock_rescore.side_effect = rescore_side

            cv = CrossValidator(corpus=corpus)
            result = cv.run(perturbation_pct=10, n_trials=5)

        assert result.n_perturbation_trials == 5
        assert result.stability_scores is not None
        assert result.perturbation_accuracy is not None
        assert 0 <= result.perturbation_accuracy <= 1.0

    @patch("ibp_enm.band.run_single_protein")
    def test_verbose_output(self, mock_run, capsys):
        """Verbose mode produces output without errors."""
        corpus = MINI_CORPUS[:1]

        def side_effect(pdb_id, chain, name=None, thresholds=None):
            entry = corpus[0]
            return _make_run_result(name, entry.archetype, entry.archetype)

        mock_run.side_effect = side_effect

        cv = CrossValidator(corpus=corpus)
        result = cv.run(verbose=True)

        captured = capsys.readouterr()
        assert "Phase 1" in captured.out
        assert "Lysozyme" in captured.out

    def test_summary_with_perturbation(self):
        """Summary includes perturbation info when available."""
        r = CrossValidationResult(
            n_proteins=3,
            n_correct=2,
            accuracy=0.667,
            per_protein=[],
            per_archetype={},
            fragile_proteins=[],
            robust_proteins=[],
            stability_scores={"P1": 0.9, "P2": 0.3, "P3": 0.85},
            n_perturbation_trials=50,
            perturbation_accuracy=0.62,
        )
        text = r.summary()
        assert "Perturbation analysis" in text
        assert "50 trials" in text
        assert "62" in text  # 62%

    def test_threshold_registry_replace(self):
        """Verify ThresholdRegistry.replace works as CrossValidator uses it."""
        keys = list(DEFAULT_THRESHOLDS.keys())
        assert len(keys) > 0

        # Replace one key
        first_key = keys[0]
        orig_val = DEFAULT_THRESHOLDS[first_key]
        new_reg = DEFAULT_THRESHOLDS.replace(
            {first_key: orig_val * 1.1},
            name="test-perturb",
        )
        assert new_reg[first_key] == pytest.approx(orig_val * 1.1)
        # Original unchanged
        assert DEFAULT_THRESHOLDS[first_key] == orig_val


# ── LARGE_CORPUS tests ─────────────────────────────────────────

class TestLargeCorpus:
    def test_large_corpus_size(self):
        """LARGE_CORPUS has 200 proteins."""
        assert len(LARGE_CORPUS) == 200

    def test_large_corpus_superset(self):
        """LARGE_CORPUS contains all EXPANDED_CORPUS entries."""
        expanded_names = {e.name for e in EXPANDED_CORPUS}
        large_names = {e.name for e in LARGE_CORPUS}
        assert expanded_names.issubset(large_names)

    def test_per_archetype_counts(self):
        """Each archetype has exactly 40 proteins."""
        from collections import Counter
        counts = Counter(e.archetype for e in LARGE_CORPUS)
        for arch in ["enzyme_active", "barrel", "globin",
                      "dumbbell", "allosteric"]:
            assert counts[arch] == 40, (
                f"{arch} has {counts[arch]}, expected 40"
            )

    def test_no_duplicate_names(self):
        """No duplicate protein names."""
        names = [e.name for e in LARGE_CORPUS]
        assert len(names) == len(set(names)), (
            f"Duplicate names: "
            f"{[n for n in names if names.count(n) > 1]}"
        )

    def test_no_duplicate_pdb_chain(self):
        """No duplicate (pdb_id, chain) pairs."""
        pairs = [(e.pdb_id, e.chain) for e in LARGE_CORPUS]
        # Some legitimate reuse: 2HHB has A, B, C chains
        # So check for exact pair duplicates only
        seen = {}
        for e in LARGE_CORPUS:
            key = (e.pdb_id, e.chain)
            if key in seen:
                pytest.fail(
                    f"Duplicate PDB:chain {key}: "
                    f"{seen[key]} and {e.name}"
                )
            seen[key] = e.name

    def test_all_valid_archetypes(self):
        """All entries have valid archetype labels."""
        valid = {"enzyme_active", "barrel", "globin",
                 "dumbbell", "allosteric"}
        for e in LARGE_CORPUS:
            assert e.archetype in valid, (
                f"{e.name} has invalid archetype {e.archetype}"
            )


# ── ParameterUsefulnessAnalyzer tests ──────────────────────────

class TestParameterUsefulnessResult:
    def test_default_values(self):
        r = ParameterUsefulnessResult(
            per_param={}, sections={},
        )
        assert r.n_proteins == 0
        assert r.total_params == 0
        assert r.zero_impact_params == []

    def test_summary_format(self):
        r = ParameterUsefulnessResult(
            per_param={
                "key1": {
                    "flip_count": 5, "flip_rate": 0.1,
                    "mean_margin_delta": -0.02,
                    "directional_bias": 0.5,
                },
                "key2": {
                    "flip_count": 0, "flip_rate": 0.0,
                    "mean_margin_delta": 0.0,
                    "directional_bias": 0.0,
                },
            },
            sections={
                "key": {
                    "n_params": 2, "total_flips": 5,
                    "mean_flip_rate": 0.05, "max_flip_rate": 0.1,
                    "n_zero": 1,
                }
            },
            n_proteins=25,
            perturbation_pct=10.0,
            total_params=2,
            zero_impact_params=["key2"],
            high_impact_params=["key1"],
        )
        text = r.summary()
        assert "Parameter Usefulness" in text
        assert "key1" in text
        assert "Zero-impact" in text
        assert "key2" in text

    def test_to_dict_json_safe(self):
        r = ParameterUsefulnessResult(
            per_param={"k": {"flip_count": 1, "flip_rate": 0.05}},
            sections={},
            n_proteins=10,
        )
        d = r.to_dict()
        json.dumps(d)  # Must not raise

    def test_heatmap_data(self):
        r = ParameterUsefulnessResult(
            per_param={
                "a.x": {
                    "flip_count": 3, "flip_rate": 0.15,
                    "mean_margin_delta": -0.01,
                    "directional_bias": 0.5,
                },
                "b.y": {
                    "flip_count": 0, "flip_rate": 0.0,
                    "mean_margin_delta": 0.0,
                    "directional_bias": 0.0,
                },
            },
            sections={},
        )
        matrix, names, metrics = r.heatmap_data()
        assert matrix.shape == (2, 3)
        assert names[0] == "a.x"  # higher flip rate first
        assert metrics == ["flip_rate", "mean_margin_delta",
                           "directional_bias"]

    def test_save_json(self, tmp_path):
        r = ParameterUsefulnessResult(
            per_param={"k": {"flip_count": 1, "flip_rate": 0.05}},
            sections={},
            n_proteins=5,
            zero_impact_params=["z"],
        )
        out = tmp_path / "param_use.json"
        r.save(str(out))
        data = json.loads(out.read_text())
        assert data["n_proteins"] == 5
        assert data["zero_impact_params"] == ["z"]


class TestParameterUsefulnessAnalyzer:
    def test_empty_cache(self):
        """Empty band_cache returns empty result."""
        analyzer = ParameterUsefulnessAnalyzer()
        result = analyzer.analyze(MINI_CORPUS, {})
        assert result.n_proteins == 0

    @patch("ibp_enm.band.ThermodynamicBand.rescore_from_profiles")
    def test_basic_analysis(self, mock_rescore):
        """Basic analysis returns per-param results."""
        corpus = MINI_CORPUS[:2]

        # Build a fake band_cache
        band_cache = {}
        for entry in corpus:
            band_cache[entry.name] = _make_run_result(
                entry.name, entry.archetype, entry.archetype,
            )

        # rescore always returns same result (stable)
        def side_effect(profiles, **kw):
            return {
                "identity": corpus[0].archetype,
                "scores": {corpus[0].archetype: 0.8,
                           "barrel": 0.1},
            }
        mock_rescore.side_effect = side_effect

        analyzer = ParameterUsefulnessAnalyzer(perturbation_pct=10)
        result = analyzer.analyze(corpus, band_cache)

        assert result.n_proteins == 2
        assert result.total_params > 0
        assert len(result.per_param) > 0

    @patch("ibp_enm.band.ThermodynamicBand.rescore_from_profiles")
    def test_flip_detection(self, mock_rescore):
        """Detects when a parameter perturbation flips classification."""
        corpus = [ProteinEntry("P1", "1ABC", "A", "globin")]
        band_cache = {
            "P1": _make_run_result("P1", "globin", "globin"),
        }

        call_count = [0]

        def side_effect(profiles, thresholds=None, **kw):
            call_count[0] += 1
            # First call flips, rest don't
            if call_count[0] == 1:
                return {
                    "identity": "barrel",
                    "scores": {"barrel": 0.6, "globin": 0.4},
                }
            return {
                "identity": "globin",
                "scores": {"globin": 0.8, "barrel": 0.1},
            }

        mock_rescore.side_effect = side_effect

        analyzer = ParameterUsefulnessAnalyzer(perturbation_pct=10)
        result = analyzer.analyze(corpus, band_cache)

        # At least one parameter should have flips
        total_flips = sum(
            v["flip_count"] for v in result.per_param.values()
        )
        assert total_flips >= 1
