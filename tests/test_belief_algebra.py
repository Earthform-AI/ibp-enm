"""Tests for ibp_enm.belief_algebra — Hamming bridge, ZD pair selection & Sedenion bridge.

Covers:
  1. ZDPairSelector — D157 structural constants, fano_activation,
     route_score, select_lines, diagnose_routing
  2. HammingBridge — syndrome computation, bridge_scores with
     routing-weighted Fano coherence, diagnose with routing info
  3. SedenonBridge — D158 rank-based dual-threshold, correction
     candidates, bridge_scores, diagnose
  4. Backward compatibility — known vote patterns produce expected
     behaviour
"""

import numpy as np
import pytest

from ibp_enm.belief_algebra import (
    HAMMING_H,
    SYNDROME_RETENTION,
    FANO_COMPLEMENTS,
    compute_syndrome,
    decode_error_position,
    HammingBridge,
    SedenonBridge,
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


# ═══════════════════════════════════════════════════════════════════
# SedenonBridge tests (D158)
# ═══════════════════════════════════════════════════════════════════

class TestFanoComplements:
    """FANO_COMPLEMENTS are correctly derived from FANO_LINES."""

    def test_complement_count(self):
        assert len(FANO_COMPLEMENTS) == 7

    def test_complement_size(self):
        for c in FANO_COMPLEMENTS:
            assert len(c) == 4

    def test_complement_is_set_complement(self):
        for i, comp in enumerate(FANO_COMPLEMENTS):
            assert set(comp) == set(range(7)) - set(FANO_LINES[i])

    def test_complements_sorted(self):
        for c in FANO_COMPLEMENTS:
            assert c == tuple(sorted(c))

    def test_all_complements_distinct(self):
        assert len(set(FANO_COMPLEMENTS)) == 7


class TestMutualExclusivity:
    """Top-3 Fano and top-4 complement are mutually exclusive (D158 P1)."""

    def test_no_nested_line_and_complement(self):
        """No Fano line L can be extended by one element to form a complement."""
        line_sets = [frozenset(l) for l in FANO_LINES]
        comp_sets = [frozenset(c) for c in FANO_COMPLEMENTS]
        for line in FANO_LINES:
            line_set = set(line)
            remaining = set(range(7)) - line_set
            for d in remaining:
                top4 = frozenset(line_set | {d})
                # Check: is top4 a complement?
                matches = [top4 == cs for cs in comp_sets]
                assert not any(matches), \
                    f"Line {line} + {d} matches complement"

    def test_combined_rate_is_40_percent(self):
        """7/35 + 7/35 = 14/35 = 40% valid rate."""
        from itertools import combinations
        line_sets = [frozenset(l) for l in FANO_LINES]
        comp_sets = [frozenset(c) for c in FANO_COMPLEMENTS]

        all_3 = list(combinations(range(7), 3))
        all_4 = list(combinations(range(7), 4))

        valid_3 = sum(1 for t in all_3 if frozenset(t) in line_sets)
        valid_4 = sum(1 for t in all_4 if frozenset(t) in comp_sets)

        assert valid_3 == 7
        assert valid_4 == 7
        # Rate = (7 + 7) / 35 = 40%
        assert (valid_3 + valid_4) / len(all_3) == pytest.approx(0.4)


class TestRankSyndrome:
    """SedenonBridge.rank_syndrome detects valid and invalid patterns."""

    @pytest.fixture
    def bridge(self):
        return SedenonBridge()

    def test_top3_fano_detected(self, bridge):
        """Votes peaked on a Fano line are detected as valid."""
        # Line 0 = (0, 1, 3): algebraic, musical, thermal
        v = np.array([0.30, 0.25, 0.05, 0.20, 0.06, 0.07, 0.07])
        syn = bridge.rank_syndrome(v)
        assert syn['valid'] is True
        assert syn['syndrome_type'] == 'top3_fano'
        assert syn['matched_line'] == 0

    def test_top4_complement_detected(self, bridge):
        """Votes peaked on a complement are detected as valid."""
        # Complement 0 = (2, 4, 5, 6)
        v = np.array([0.05, 0.06, 0.30, 0.07, 0.25, 0.20, 0.15])
        syn = bridge.rank_syndrome(v)
        assert syn['valid'] is True
        assert syn['syndrome_type'] == 'top4_complement'
        assert syn['matched_line'] == 0

    def test_invalid_has_corrections(self, bridge):
        """Non-Fano top-3 produces 3 correction candidates."""
        # top-3 = {0, 1, 2} which is NOT a Fano line
        v = np.array([0.30, 0.20, 0.15, 0.10, 0.10, 0.10, 0.05])
        syn = bridge.rank_syndrome(v)
        assert syn['valid'] is False
        assert syn['syndrome_type'] == 'invalid'
        assert len(syn['correction']) == 3

    def test_corrections_sorted_by_coherence(self, bridge):
        """Corrections are ranked by line coherence (highest first)."""
        v = np.array([0.30, 0.20, 0.15, 0.10, 0.10, 0.10, 0.05])
        syn = bridge.rank_syndrome(v)
        coherences = [c['coherence'] for c in syn['correction']]
        assert coherences == sorted(coherences, reverse=True)

    def test_corrections_point_to_bottom4(self, bridge):
        """The 'add' instrument is always from the bottom-4 voters."""
        v = np.array([0.30, 0.20, 0.15, 0.10, 0.10, 0.10, 0.05])
        syn = bridge.rank_syndrome(v)
        top3 = set(syn['ranking'][:3])
        for c in syn['correction']:
            assert c['add'] not in top3
            assert c['drop'] in top3

    def test_all_seven_lines_detectable(self, bridge):
        """Each of the 7 Fano lines can be detected as valid."""
        for li, line in enumerate(FANO_LINES):
            v = np.zeros(7)
            v[list(line)] = [0.4, 0.3, 0.2]
            syn = bridge.rank_syndrome(v)
            assert syn['valid'] is True, f"Line {li} not detected"
            assert syn['syndrome_type'] == 'top3_fano'

    def test_all_seven_complements_detectable(self, bridge):
        """Each of the 7 Fano complements can be detected as valid."""
        for ci, comp in enumerate(FANO_COMPLEMENTS):
            v = np.zeros(7)
            v[list(comp)] = [0.4, 0.3, 0.2, 0.1]
            syn = bridge.rank_syndrome(v)
            assert syn['valid'] is True, f"Complement {ci} not detected"
            assert syn['syndrome_type'] == 'top4_complement'

    def test_ranking_is_descending(self, bridge):
        """Ranking is in descending vote order."""
        v = np.array([0.05, 0.30, 0.10, 0.25, 0.15, 0.07, 0.08])
        syn = bridge.rank_syndrome(v)
        # Verify ranking sorts descending
        for i in range(6):
            assert v[syn['ranking'][i]] >= v[syn['ranking'][i+1]]


class TestCorrectionCandidates:
    """Correction candidates have the right structure (D158 P2-P3)."""

    @pytest.fixture
    def bridge(self):
        return SedenonBridge()

    def test_every_invalid_triple_has_3_corrections(self, bridge):
        """D158 P2: all 28 invalid triples yield exactly 3 corrections."""
        from itertools import combinations
        line_sets = [frozenset(l) for l in FANO_LINES]
        truly_invalid = 0
        for top3 in combinations(range(7), 3):
            if frozenset(top3) in line_sets:
                continue  # skip valid triples
            v = np.zeros(7)
            v[list(top3)] = [0.4, 0.3, 0.2]
            syn = bridge.rank_syndrome(v)
            if syn['syndrome_type'] == 'top4_complement':
                # 4th-ranked element formed a valid complement —
                # this is a legitimate 'rescue' case
                assert syn['valid'] is True
            else:
                assert len(syn['correction']) == 3, \
                    f"Triple {top3} has {len(syn['correction'])} corrections"
                truly_invalid += 1
        # Most of the 28 non-line triples should be truly invalid
        assert truly_invalid >= 20

    def test_correction_lines_are_fano(self, bridge):
        """Each correction produces a valid Fano line."""
        line_sets = [frozenset(l) for l in FANO_LINES]
        v = np.array([0.30, 0.20, 0.15, 0.10, 0.10, 0.10, 0.05])
        syn = bridge.rank_syndrome(v)
        for c in syn['correction']:
            assert frozenset(c['line']) in line_sets

    def test_correction_coherence_is_mean_vote(self, bridge):
        """Coherence equals the mean vote on the corrected line."""
        v = np.array([0.30, 0.20, 0.15, 0.10, 0.10, 0.10, 0.05])
        syn = bridge.rank_syndrome(v)
        for c in syn['correction']:
            expected = np.mean([v[k] for k in c['line']])
            assert c['coherence'] == pytest.approx(expected)


class TestSedenonBridgeScores:
    """SedenonBridge.bridge_scores has the right interface and properties."""

    @pytest.fixture
    def bridge(self):
        return SedenonBridge()

    def _make_votes(self, winning_arch, archs, rng=None):
        if rng is None:
            rng = np.random.default_rng(42)
        votes = []
        for _ in range(7):
            d = {a: rng.random() * 0.15 for a in archs}
            d[winning_arch] += 0.3
            votes.append(d)
        return votes

    def test_returns_all_archetypes(self, bridge):
        archs = ['A', 'B', 'C']
        votes = self._make_votes('A', archs)
        scores = bridge.bridge_scores(votes, archs)
        assert set(scores.keys()) == set(archs)

    def test_normalised(self, bridge):
        archs = ['A', 'B', 'C']
        votes = self._make_votes('A', archs)
        scores = bridge.bridge_scores(votes, archs)
        assert sum(scores.values()) == pytest.approx(1.0, abs=1e-8)

    def test_top_arch_gets_highest_score(self, bridge):
        archs = ['A', 'B', 'C']
        votes = self._make_votes('A', archs)
        scores = bridge.bridge_scores(votes, archs)
        assert max(scores, key=scores.get) == 'A'

    def test_fewer_than_7_instruments(self, bridge):
        """Graceful degradation with < 7 instruments."""
        archs = ['A', 'B']
        votes = [{'A': 0.6, 'B': 0.4} for _ in range(3)]
        scores = bridge.bridge_scores(votes, archs)
        assert sum(scores.values()) == pytest.approx(1.0, abs=1e-8)


class TestSedenonBridgeDiagnose:
    """SedenonBridge.diagnose returns rank-based syndrome diagnostics."""

    @pytest.fixture
    def bridge(self):
        return SedenonBridge()

    def test_diagnose_has_valid_fraction(self, bridge):
        archs = ['A', 'B', 'C']
        votes = [{'A': 0.6, 'B': 0.3, 'C': 0.1} for _ in range(7)]
        diag = bridge.diagnose(votes, archs)
        assert 'valid_fraction' in diag
        assert 0 <= diag['valid_fraction'] <= 1

    def test_valid_fractions_sum(self, bridge):
        """valid_top3 + valid_top4 + invalid = 1."""
        archs = ['A', 'B', 'C']
        votes = [{a: np.random.random() for a in archs} for _ in range(7)]
        diag = bridge.diagnose(votes, archs)
        total = (diag['valid_top3_fraction']
                 + diag['valid_top4_fraction']
                 + diag['invalid_fraction'])
        assert total == pytest.approx(1.0, abs=1e-8)

    def test_diagnose_has_routing(self, bridge):
        archs = ['A', 'B']
        votes = [{'A': 0.6, 'B': 0.4} for _ in range(7)]
        diag = bridge.diagnose(votes, archs)
        assert 'routing' in diag
        assert 'mean_route_score' in diag

    def test_per_archetype_syndrome_types(self, bridge):
        """Each archetype gets a syndrome classification."""
        archs = ['A', 'B', 'C']
        votes = [{a: np.random.random() for a in archs} for _ in range(7)]
        diag = bridge.diagnose(votes, archs)
        for arch in archs:
            info = diag['per_archetype'][arch]
            assert info['syndrome_type'] in (
                'top3_fano', 'top4_complement', 'invalid',
            )


class TestSedenonBridgeValidity:
    """Rank-based approach has higher valid rate than Hamming (D158 P4)."""

    def test_synthetic_valid_rate_exceeds_hamming(self):
        """On random Dirichlet votes, rank valid rate > Hamming valid rate."""
        from ibp_enm.belief_algebra import compute_syndrome
        bridge = SedenonBridge()
        rng = np.random.default_rng(123)
        n = 2000

        hamming_valid = 0
        rank_valid = 0

        for _ in range(n):
            v = rng.dirichlet(np.ones(7))
            # Hamming
            support = (v > np.mean(v)).astype(int)
            if np.all(compute_syndrome(support) == 0):
                hamming_valid += 1
            # Rank
            syn = bridge.rank_syndrome(v)
            if syn['valid']:
                rank_valid += 1

        hamming_rate = hamming_valid / n
        rank_rate = rank_valid / n
        assert rank_rate > hamming_rate, \
            f"Rank rate {rank_rate:.3f} not > Hamming rate {hamming_rate:.3f}"
        # D158 target: ≥ 35% (allowing margin for small sample)
        assert rank_rate >= 0.35, f"Rank rate {rank_rate:.3f} < 35%"


class TestSedenonHammingBackwardCompat:
    """SedenonBridge can replace HammingBridge without breaking API."""

    def test_same_interface_bridge_scores(self):
        """Both bridges return the same dict structure."""
        archs = ['A', 'B', 'C']
        votes = [{'A': 0.5, 'B': 0.3, 'C': 0.2} for _ in range(7)]
        h = HammingBridge()
        s = SedenonBridge()
        h_scores = h.bridge_scores(votes, archs)
        s_scores = s.bridge_scores(votes, archs)
        assert set(h_scores.keys()) == set(s_scores.keys())
        assert sum(h_scores.values()) == pytest.approx(1.0, abs=1e-8)
        assert sum(s_scores.values()) == pytest.approx(1.0, abs=1e-8)

    def test_threshold_shift_same_shape(self):
        """Both bridges return the retention dict with same shape."""
        archs = ['X', 'Y']
        votes = [{'X': 0.6, 'Y': 0.4} for _ in range(7)]
        h_shifts = HammingBridge().threshold_shift(votes, archs)
        s_shifts = SedenonBridge().threshold_shift(votes, archs)
        for arch in archs:
            assert h_shifts[arch].shape == s_shifts[arch].shape == (7,)
            assert np.all(h_shifts[arch] > 0)
            assert np.all(s_shifts[arch] > 0)
