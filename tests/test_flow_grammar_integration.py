"""Integration test: FlowGrammarLens produces correct boost/penalty
for the D130-calibrated proteins.

Uses synthetic Laplacian-like matrices (not real PDB data) to verify
the lens mechanics.  The D130 experiment validated on real proteins;
this test verifies the integration into ibp_enm.
"""

import numpy as np
import pytest

from ibp_enm.lens_stack import (
    FlowGrammarLens,
    _gnm_correlation_matrix,
    _te_and_net_matrices,
    _flow_word,
)
from ibp_enm.thresholds import DEFAULT_THRESHOLDS


class TestTEHelpers:
    """Test module-level TE helper functions."""

    def test_gnm_correlation_symmetric(self):
        N = 20
        rng = np.random.RandomState(42)
        A = rng.rand(N, N); A = (A + A.T) / 2
        np.fill_diagonal(A, 0)
        L = np.diag(A.sum(axis=1)) - A
        evals, evecs = np.linalg.eigh(L)

        C = _gnm_correlation_matrix(evals, evecs, 0.0)
        assert C.shape == (N, N)
        np.testing.assert_allclose(C, C.T, atol=1e-10)

    def test_gnm_correlation_decays_with_time(self):
        N = 15
        rng = np.random.RandomState(7)
        A = rng.rand(N, N); A = (A + A.T) / 2
        np.fill_diagonal(A, 0)
        L = np.diag(A.sum(axis=1)) - A
        evals, evecs = np.linalg.eigh(L)

        C0 = _gnm_correlation_matrix(evals, evecs, 0.0)
        Ct = _gnm_correlation_matrix(evals, evecs, 10.0)
        # Off-diagonal correlations should decay
        assert np.mean(np.abs(Ct)) < np.mean(np.abs(C0))

    def test_te_net_matrices_shapes(self):
        N = 10
        C0 = np.eye(N) + 0.1 * np.random.RandomState(1).rand(N, N)
        C0 = (C0 + C0.T) / 2
        Ct = np.eye(N) + 0.05 * np.random.RandomState(2).rand(N, N)
        Ct = (Ct + Ct.T) / 2

        TE, NET = _te_and_net_matrices(C0, Ct, N)
        assert TE.shape == (N, N)
        assert NET.shape == (N, N)
        # TE should be non-negative
        assert np.all(TE >= -1e-10)
        # TE diagonal should be zero
        np.testing.assert_allclose(np.diag(TE), 0, atol=1e-10)
        # NET should be antisymmetric: NET + NET^T = 0
        np.testing.assert_allclose(NET + NET.T, 0, atol=1e-10)


class TestFlowWord:
    def test_directing(self):
        assert _flow_word(1.2, 1.1) == "DIRECTING"

    def test_channeling(self):
        assert _flow_word(0.5, 0.90) == "CHANNELING"

    def test_diffusing(self):
        assert _flow_word(0.5, 0.5) == "DIFFUSING"

    def test_boundary_directing_needs_both(self):
        # High enrichment but low asymmetry → CHANNELING not DIRECTING
        assert _flow_word(0.5, 1.1) == "CHANNELING"


class TestFlowGrammarBoostCalibration:
    """Verify boost values match D130 calibration."""

    def test_chey_like_signals(self):
        """CheY-like flow (DIRECTING) should give positive boost."""
        signals = {
            "te_asymmetry": 1.129,
            "cross_enrichment": 1.066,
            "driver_sensor_ratio": 1.25,
            "flow_word": "DIRECTING",
        }
        boost = FlowGrammarLens._compute_flow_boost(
            signals, DEFAULT_THRESHOLDS)
        # D130: CheY got +0.18 (cap)
        assert boost == pytest.approx(0.18, abs=0.01)

    def test_groel_like_signals(self):
        """GroEL-like flow (anti-allosteric) should give negative boost."""
        signals = {
            "te_asymmetry": 1.000,
            "cross_enrichment": 0.518,
            "driver_sensor_ratio": 0.86,
            "flow_word": "DIFFUSING",
        }
        boost = FlowGrammarLens._compute_flow_boost(
            signals, DEFAULT_THRESHOLDS)
        # D130: GroEL got -0.10
        assert boost == pytest.approx(-0.10, abs=0.01)

    def test_neutral_signals(self):
        """Mid-range flow should give near-zero boost."""
        signals = {
            "te_asymmetry": 0.85,
            "cross_enrichment": 0.80,
            "driver_sensor_ratio": 1.10,
            "flow_word": "CHANNELING",
        }
        boost = FlowGrammarLens._compute_flow_boost(
            signals, DEFAULT_THRESHOLDS)
        assert abs(boost) < 0.05

    def test_boost_capped(self):
        """Boost should never exceed cap (0.18)."""
        signals = {
            "te_asymmetry": 5.0,
            "cross_enrichment": 5.0,
            "driver_sensor_ratio": 5.0,
            "flow_word": "DIRECTING",
        }
        boost = FlowGrammarLens._compute_flow_boost(
            signals, DEFAULT_THRESHOLDS)
        assert boost <= 0.18

    def test_penalty_floored(self):
        """Penalty should never exceed floor (-0.15)."""
        signals = {
            "te_asymmetry": 0.0,
            "cross_enrichment": 0.0,
            "driver_sensor_ratio": 0.0,
            "flow_word": "DIFFUSING",
        }
        boost = FlowGrammarLens._compute_flow_boost(
            signals, DEFAULT_THRESHOLDS)
        assert boost >= -0.15
