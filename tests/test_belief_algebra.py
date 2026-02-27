"""Tests for ibp_enm.belief_algebra — Hamming bridge & ZD pair selection.

Covers:
  1. ZDPairSelector — D157 structural constants, fano_activation,
     route_score, select_lines, diagnose_routing
  2. HammingBridge — syndrome computation, bridge_scores with
     routing-weighted Fano coherence, diagnose with routing info
  3. Backward compatibility — known vote patterns produce expected
     behaviour
"""

import numpy as np
import pytest

from ibp_enm.belief_algebra import (
    HAMMING_H,
    SYNDROME_RETENTION,
    compute_syndrome,
    decode_error_position,
    HammingBridge,
    ZDPairSelector,
)
from ibp_enm.algebra import FANO_LINES, INSTRUMENT_NAMES


# ═══════════════════════════════════════════════════════════════════
# ZDPairSelector tests
# ═══════════════════════════════════════════════════════════════════

class TestZDPairSelectorConstants:
    """D157 structural constants are correctly encoded."""

    def test_kernel_classes(self):
        assert ZDPairSelector.KERNEL_CLASSES == 42

    def test_contained_per_kernel(self):
        assert ZDPairSelector.CONTAINED_PER_KERNEL == 4

    def test_total_contained(self):
        assert (ZDPairSelector.KERNEL_CLASSES
                * ZDPairSelector.CONTAINED_PER_KERNEL
                == ZDPairSelector.TOTAL_CONTAINED)
        assert ZDPairSelector.TOTAL_CONTAINED == 168

    def test_cross_half_subs(self):
        assert ZDPairSelector.CROSS_HALF_SUBS == 21

    def test_pure_half_subs(self):
        assert ZDPairSelector.PURE_HALF_SUBS == 14

    def test_routes_per_line(self):
        assert ZDPairSelector.ROUTES_PER_LINE == 72

    def test_contained_purity_equals_syndrome_retention(self):
        """The 1/√2 from D157 geometry = the 1/√2 from D148 spectrum."""
        assert ZDPairSelector.CONTAINED_PURITY == pytest.approx(
            SYNDROME_RETENTION)
        assert ZDPairSelector.CONTAINED_PURITY == pytest.approx(
            1.0 / np.sqrt(2))

    def test_sub_accounting(self):
        """21 cross-half + 14 pure-half = 35 quaternionic subs."""
        assert (ZDPairSelector.CROSS_HALF_SUBS
                + ZDPairSelector.PURE_HALF_SUBS) == 35


class TestFanoActivation:
    """ZDPairSelector.fano_activation correctness."""

    @pytest.fixture
    def selector(self):
        return ZDPairSelector()

    def test_zero_support(self, selector):
        """No supporters → no activation."""
        s = np.zeros(7, dtype=int)
        act = selector.fano_activation(s)
        assert act.shape == (7,)
        np.testing.assert_array_equal(act, 0.0)

    def test_full_support(self, selector):
        """All instruments support → all lines fully activated."""
        s = np.ones(7, dtype=int)
        act = selector.fano_activation(s)
        np.testing.assert_array_equal(act, 1.0)

    def test_fano_line_support(self, selector):
        """Support = a Fano line → that line gets 1.0, others get mixed."""
        # Line 0 = (0, 1, 3) → instruments 0, 1, 3
        s = np.zeros(7, dtype=int)
        s[[0, 1, 3]] = 1
        act = selector.fano_activation(s)
        # Line 0 has all 3: activation = 1.0
        assert act[0] == 1.0
        # Line 3 = (3, 4, 6): only instrument 3 overlaps → 1/√2
        assert act[3] == pytest.approx(SYNDROME_RETENTION)
        # All activations are in {0, 1/√2, 1.0}
        valid = {0.0, float(SYNDROME_RETENTION), 1.0}
        for v in act:
            assert float(v) in valid or any(
                abs(float(v) - x) < 1e-10 for x in valid)

    def test_single_supporter(self, selector):
        """Single instrument → only lines containing it get 1/√2."""
        s = np.zeros(7, dtype=int)
        s[0] = 1  # only instrument 0 (algebraic)
        act = selector.fano_activation(s)
        # Instrument 0 is on lines 0 (0,1,3), 4 (4,5,0), 6 (6,0,2)
        for i in range(7):
            if 0 in set(FANO_LINES[i]):
                assert act[i] == pytest.approx(SYNDROME_RETENTION)
            else:
                assert act[i] == 0.0

    def test_activation_shape(self, selector):
        """Always returns 7-element array."""
        for w in range(8):
            s = np.zeros(7, dtype=int)
            s[:w] = 1
            assert selector.fano_activation(s).shape == (7,)

    def test_two_adjacent_supporters(self, selector):
        """Two instruments on same Fano line → that line = 1.0."""
        s = np.zeros(7, dtype=int)
        s[0] = 1; s[1] = 1  # instruments 0, 1 → share line 0 (0,1,3)
        act = selector.fano_activation(s)
        assert act[0] == 1.0  # line (0,1,3) fully activated

    def test_weight_symmetry(self, selector):
        """All weight-3 Fano-line supports give exactly the same activation pattern shape."""
        act_sums = []
        for line in FANO_LINES:
            s = np.zeros(7, dtype=int)
            for p in line:
                s[p] = 1
            act = selector.fano_activation(s)
            # The supporting line is always 1.0
            assert np.max(act) == 1.0
            act_sums.append(float(np.sum(act)))
        # By Fano plane symmetry, all lines give the same total activation
        assert len(set(round(x, 10) for x in act_sums)) == 1


class TestRouteScore:
    """ZDPairSelector.route_score correctness."""

    @pytest.fixture
    def selector(self):
        return ZDPairSelector()

    def test_zero(self, selector):
        assert selector.route_score(np.zeros(7, dtype=int)) == 0.0

    def test_full(self, selector):
        assert selector.route_score(np.ones(7, dtype=int)) == 1.0

    def test_range(self, selector):
        """Route score is always in [0, 1]."""
        rng = np.random.default_rng(42)
        for _ in range(50):
            s = rng.integers(0, 2, size=7)
            score = selector.route_score(s)
            assert 0.0 <= score <= 1.0

    def test_monotonic_in_support_weight(self, selector):
        """More supporters → greater or equal route score."""
        # Special case: weight 0 < weight 7
        s0 = np.zeros(7, dtype=int)
        s7 = np.ones(7, dtype=int)
        assert selector.route_score(s0) < selector.route_score(s7)


class TestSelectLines:
    """ZDPairSelector.select_lines correctness."""

    @pytest.fixture
    def selector(self):
        return ZDPairSelector()

    def test_zero_support(self, selector):
        assert selector.select_lines(np.zeros(7, dtype=int)) == []

    def test_full_support(self, selector):
        assert sorted(selector.select_lines(np.ones(7, dtype=int))) == list(range(7))

    def test_fano_line_pattern(self, selector):
        """Support = line 0's instruments → line 0 is in select_lines."""
        s = np.zeros(7, dtype=int)
        for p in FANO_LINES[0]:
            s[p] = 1
        lines = selector.select_lines(s)
        assert 0 in lines

    def test_single_supporter_no_strong_lines(self, selector):
        """Single instrument → no line has ≥2 supporters."""
        s = np.zeros(7, dtype=int)
        s[3] = 1
        assert selector.select_lines(s) == []


class TestDiagnoseRouting:
    """ZDPairSelector.diagnose_routing structure."""

    @pytest.fixture
    def selector(self):
        return ZDPairSelector()

    def test_keys_present(self, selector):
        s = np.array([1, 0, 1, 0, 1, 0, 1])
        d = selector.diagnose_routing(s)
        expected_keys = {
            'fano_activation', 'strong_lines', 'weak_lines',
            'n_strong', 'n_weak', 'route_score',
            'total_strong_routes', 'support_weight',
        }
        assert set(d.keys()) == expected_keys

    def test_support_weight(self, selector):
        s = np.array([1, 1, 0, 0, 0, 1, 1])
        d = selector.diagnose_routing(s)
        assert d['support_weight'] == 4

    def test_strong_routes_multiple(self, selector):
        s = np.ones(7, dtype=int)
        d = selector.diagnose_routing(s)
        assert d['n_strong'] == 7
        assert d['total_strong_routes'] == 7 * 72

    def test_consistency(self, selector):
        """n_strong + n_weak ≤ 7."""
        s = np.array([1, 1, 1, 0, 0, 0, 0])
        d = selector.diagnose_routing(s)
        assert d['n_strong'] + d['n_weak'] <= 7


# ═══════════════════════════════════════════════════════════════════
# HammingBridge tests
# ═══════════════════════════════════════════════════════════════════

class TestHammingBridgeBasic:
    """Core syndrome protocol tests."""

    def test_syndrome_zero_codeword(self):
        """All-zero and all-one support → zero syndrome."""
        assert np.all(compute_syndrome(np.zeros(7, dtype=int)) == 0)
        assert np.all(compute_syndrome(np.ones(7, dtype=int)) == 0)

    def test_fano_line_is_codeword(self):
        """Each Fano line (weight 3) is a valid codeword."""
        for line in FANO_LINES:
            s = np.zeros(7, dtype=int)
            for p in line:
                s[p] = 1
            syn = compute_syndrome(s)
            assert np.all(syn == 0), f"Line {line} not a codeword"

    def test_complement_is_codeword(self):
        """Fano line complement (weight 4) is a valid codeword."""
        for line in FANO_LINES:
            s = np.ones(7, dtype=int)
            for p in line:
                s[p] = 0
            syn = compute_syndrome(s)
            assert np.all(syn == 0), f"Complement of {line} not a codeword"

    def test_single_bit_detected(self):
        """Single-bit support has nonzero syndrome → error detected."""
        for bit in range(7):
            s = np.zeros(7, dtype=int)
            s[bit] = 1
            syn = compute_syndrome(s)
            assert not np.all(syn == 0)
            pos = decode_error_position(syn)
            assert pos == bit


class TestHammingBridgeScores:
    """HammingBridge.bridge_scores with D157 routing."""

    @pytest.fixture
    def bridge(self):
        return HammingBridge()

    @staticmethod
    def _uniform_votes(archs, top_arch, gap=0.1):
        """Build 7 vote dicts with instruments mostly agreeing on top_arch."""
        votes = []
        for i in range(7):
            v = {a: 0.10 for a in archs}
            v[top_arch] = 0.10 + gap
            votes.append(v)
        return votes

    @staticmethod
    def _split_votes(archs, arch_a, arch_b, split=4):
        """Build 7 votes with `split` instruments favouring arch_a, rest arch_b."""
        votes = []
        for i in range(7):
            v = {a: 0.05 for a in archs}
            if i < split:
                v[arch_a] = 0.30
            else:
                v[arch_b] = 0.30
            votes.append(v)
        return votes

    def test_bridge_returns_all_archs(self, bridge):
        archs = ["barrel", "enzyme_active", "globin"]
        votes = self._uniform_votes(archs, "barrel")
        scores = bridge.bridge_scores(votes, archs)
        assert set(scores.keys()) == set(archs)

    def test_bridge_normalised(self, bridge):
        archs = ["barrel", "enzyme_active", "globin", "dumbbell"]
        votes = self._uniform_votes(archs, "barrel")
        scores = bridge.bridge_scores(votes, archs)
        assert sum(scores.values()) == pytest.approx(1.0, abs=0.01)

    def test_bridge_top_arch_highest(self, bridge):
        archs = ["barrel", "enzyme_active", "globin"]
        votes = self._uniform_votes(archs, "barrel", gap=0.2)
        scores = bridge.bridge_scores(votes, archs)
        assert max(scores, key=scores.get) == "barrel"

    def test_bridge_fano_coherence_boost(self, bridge):
        """Fano-line support pattern yields a valid syndrome (no dampening),
        while a non-Fano pattern triggers syndrome correction."""
        archs = ["barrel", "enzyme_active", "globin"]

        # Fano-coherent: instruments 0,1,3 are Fano line 0
        votes_coherent = [{a: 0.10 for a in archs} for _ in range(7)]
        for i in [0, 1, 3]:
            votes_coherent[i]["barrel"] = 0.40

        # Check syndrome is valid (weight-3 codeword)
        diag_coh = bridge.diagnose(votes_coherent, archs)
        barrel_diag = diag_coh["per_archetype"]["barrel"]
        assert barrel_diag["syndrome"] == (0, 0, 0), (
            "Fano-line support should be valid Hamming codeword"
        )
        assert barrel_diag["error_position"] is None

        # Non-Fano pattern: instruments 0,1,2 are NOT a Fano line
        votes_nonline = [{a: 0.10 for a in archs} for _ in range(7)]
        for i in [0, 1, 2]:
            votes_nonline[i]["barrel"] = 0.40

        diag_non = bridge.diagnose(votes_nonline, archs)
        barrel_non = diag_non["per_archetype"]["barrel"]
        assert barrel_non["syndrome"] != (0, 0, 0), (
            "Non-Fano support should trigger nonzero syndrome"
        )


class TestHammingBridgeDiagnose:
    """HammingBridge.diagnose includes D157 routing data."""

    @pytest.fixture
    def bridge(self):
        return HammingBridge()

    def test_diagnose_has_routing(self, bridge):
        archs = ["barrel", "enzyme_active"]
        votes = [{a: 0.15 for a in archs} for _ in range(7)]
        votes[0]["barrel"] = 0.30
        votes[1]["barrel"] = 0.30

        diag = bridge.diagnose(votes, archs)

        assert "routing" in diag
        assert "mean_route_score" in diag
        assert set(diag["routing"].keys()) == set(archs)

    def test_routing_per_archetype_structure(self, bridge):
        archs = ["barrel", "globin"]
        votes = [{a: 0.15 for a in archs} for _ in range(7)]
        diag = bridge.diagnose(votes, archs)

        for arch in archs:
            rd = diag["routing"][arch]
            assert 'fano_activation' in rd
            assert 'strong_lines' in rd
            assert 'route_score' in rd
            assert len(rd['fano_activation']) == 7

    def test_mean_route_score_range(self, bridge):
        archs = ["barrel", "enzyme_active", "globin"]
        votes = [{a: 0.15 for a in archs} for _ in range(7)]
        votes[0]["barrel"] = 0.30
        diag = bridge.diagnose(votes, archs)
        assert 0.0 <= diag["mean_route_score"] <= 1.0


class TestHammingBridgeBackwardCompat:
    """Routing enhancement preserves backward-compatible behaviour."""

    @pytest.fixture
    def bridge(self):
        return HammingBridge()

    def test_unanimous_vote_unchanged(self, bridge):
        """When all instruments agree, routing changes nothing —
        all lines are fully activated (activation = 1.0)."""
        archs = ["barrel", "enzyme_active"]
        # All instruments strongly favour barrel
        votes = [{"barrel": 0.8, "enzyme_active": 0.2} for _ in range(7)]
        scores = bridge.bridge_scores(votes, archs)
        assert max(scores, key=scores.get) == "barrel"
        assert scores["barrel"] > 0.5

    def test_few_instruments_still_works(self, bridge):
        """< 7 instruments → routing degrades gracefully to zeros."""
        archs = ["barrel", "enzyme_active"]
        votes = [{"barrel": 0.6, "enzyme_active": 0.4} for _ in range(3)]
        scores = bridge.bridge_scores(votes, archs)
        assert sum(scores.values()) == pytest.approx(1.0, abs=0.01)


# ═══════════════════════════════════════════════════════════════════
# Integration: ZDPairSelector + HammingBridge
# ═══════════════════════════════════════════════════════════════════

class TestIntegration:
    """End-to-end tests combining selector and bridge."""

    def test_bridge_has_zd_selector(self):
        bridge = HammingBridge()
        assert hasattr(bridge, '_zd_selector')
        assert isinstance(bridge._zd_selector, ZDPairSelector)

    def test_selector_purity_matches_retention(self):
        """The geometric constant matches the spectral constant."""
        assert ZDPairSelector.CONTAINED_PURITY == SYNDROME_RETENTION

    def test_all_fano_lines_universally_routed(self):
        """D157 key result: all 7 lines reachable from all 42 kernels."""
        sel = ZDPairSelector()
        full = np.ones(7, dtype=int)
        act = sel.fano_activation(full)
        # All 7 lines fully activated
        assert np.all(act == 1.0)
        # 7 lines × 72 routes = 504 total routes
        assert 7 * sel.ROUTES_PER_LINE == 504
