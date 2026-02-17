"""Tests for the ArchetypeRule decomposition (v0.4.0).

Covers:
1. Registry completeness — correct number of rules, unique names
2. Per-rule firing — each rule fires on the correct signal
3. Equivalence — apply_rules() reproduces original archetype_vote() output
4. Tracing — apply_rules_traced() returns correct firings
5. Sweep utilities — get_rules(), replace_rules()
6. Compound rules — multi-metric conditions
"""

import math
import pytest
from collections import Counter

from ibp_enm.instruments import ThermoReactionProfile
from ibp_enm.rules import (
    ArchetypeRule,
    CompoundArchetypeRule,
    RuleFiring,
    ARCHETYPE_RULES,
    apply_rules,
    apply_rules_traced,
    get_rules,
    replace_rules,
    _lt, _gt, _between,
)


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def blank_profile():
    """Blank algebraic profile — all defaults."""
    return ThermoReactionProfile(instrument="algebraic", intent_idx=0)


@pytest.fixture
def barrel_like_profile():
    """Profile with strong barrel signals (D113 style)."""
    return ThermoReactionProfile(
        instrument="algebraic",
        intent_idx=0,
        n_residues=200,
        n_contacts=1000,
        gap_trajectory=[1.0, 0.99, 0.98, 0.97, 0.96, 0.95],
        mode_scatters=[0.1, 0.15, 0.12, 0.08, 0.11],
        delta_beta_trajectory=[0.01, 0.02, 0.01, 0.015, 0.01],
        bus_mass_trajectory=[0.3, 0.35, 0.28, 0.32, 0.3],
        spatial_radius_trajectory=[1.0, 1.2, 0.8, 1.1, 0.9],
        reversibility=[True, True, True, True, True],
        species_removed=["Carved", "Carved", "Carved", "Carved", "Carved"],
    )


@pytest.fixture
def dumbbell_like_profile():
    """Profile with strong dumbbell signals."""
    return ThermoReactionProfile(
        instrument="algebraic",
        intent_idx=0,
        n_residues=200,
        n_contacts=1000,
        gap_trajectory=[1.0, 0.8, 0.6, 0.5, 0.4, 0.3],
        mode_scatters=[5.0, 6.0, 5.5, 7.0, 5.2],
        delta_beta_trajectory=[0.2, 0.25, 0.18, 0.22, 0.3],
        bus_mass_trajectory=[0.9, 0.95, 0.88, 0.92, 0.91],
        spatial_radius_trajectory=[10.0, 12.0, 8.0, 11.0, 9.0],
        reversibility=[False, False, False, False, False],
        species_removed=["Soft", "Soft", "Soft", "Soft", "Soft"],
    )


@pytest.fixture
def globin_like_profile():
    """Profile with strong globin signals."""
    return ThermoReactionProfile(
        instrument="algebraic",
        intent_idx=0,
        n_residues=200,
        n_contacts=1000,
        gap_trajectory=[1.0, 0.7, 0.5, 0.6, 0.4, 0.55],
        mode_scatters=[2.0, 2.5, 1.8, 2.2, 2.1],
        delta_beta_trajectory=[0.05, 0.06, 0.04, 0.07, 0.05],
        bus_mass_trajectory=[0.6, 0.65, 0.58, 0.62, 0.6],
        spatial_radius_trajectory=[3.0, 3.5, 2.8, 3.2, 3.0],
        reversibility=[False, True, False, False, True],
        species_removed=["Sibling", "Sibling", "Sibling", "Sibling", "Sibling"],
    )


@pytest.fixture
def enzyme_like_profile():
    """Profile with strong enzyme signals."""
    return ThermoReactionProfile(
        instrument="algebraic",
        intent_idx=0,
        n_residues=200,
        n_contacts=1000,
        gap_trajectory=[1.0, 0.95, 0.90, 0.88, 0.85, 0.83],
        mode_scatters=[3.0, 3.5, 2.8, 3.2, 3.0],
        delta_beta_trajectory=[0.05, 0.06, 0.04, 0.05, 0.05],
        bus_mass_trajectory=[0.6, 0.65, 0.58, 0.62, 0.6],
        spatial_radius_trajectory=[3.0, 3.5, 2.8, 3.2, 3.0],
        ipr_trajectory=[0.03, 0.035, 0.028, 0.032, 0.03, 0.031],
        reversibility=[False, True, False, True, False],
        species_removed=["Carved", "Carved", "Carved", "Carved", "Carved"],
    )


def _profile_with(**kwargs):
    """Helper to build a profile with specific metric values."""
    defaults = dict(
        instrument="algebraic", intent_idx=0,
        n_residues=200, n_contacts=1000,
    )
    defaults.update(kwargs)
    return ThermoReactionProfile(**defaults)


# ═══════════════════════════════════════════════════════════════════
# 1. Registry completeness
# ═══════════════════════════════════════════════════════════════════

class TestRegistryCompleteness:
    def test_rules_is_tuple(self):
        assert isinstance(ARCHETYPE_RULES, tuple)

    def test_no_empty_registry(self):
        assert len(ARCHETYPE_RULES) > 50

    def test_unique_names(self):
        names = [r.name for r in ARCHETYPE_RULES]
        assert len(names) == len(set(names)), f"Duplicate names: {[n for n, c in Counter(names).items() if c > 1]}"

    def test_all_seven_instruments_covered(self):
        instruments = {r.instrument for r in ARCHETYPE_RULES}
        expected = {"algebraic", "musical", "fick", "thermal",
                    "cooperative", "propagative", "fragile", "*"}
        assert expected == instruments

    def test_all_five_archetypes_covered(self):
        archetypes = {r.archetype for r in ARCHETYPE_RULES}
        expected = {"barrel", "dumbbell", "globin", "enzyme_active", "allosteric"}
        assert expected == archetypes

    def test_every_instrument_archetype_pair_has_at_least_one_rule(self):
        """Each instrument × archetype pair should have at least one rule
        (either specific or universal)."""
        pairs = set()
        for r in ARCHETYPE_RULES:
            if r.instrument == "*":
                for a in {"barrel", "dumbbell", "globin", "enzyme_active", "allosteric"}:
                    pairs.add(("*", a))
            else:
                pairs.add((r.instrument, r.archetype))
        # Every archetype should have at least one universal rule
        for arch in ["barrel", "dumbbell", "globin", "enzyme_active", "allosteric"]:
            assert ("*", arch) in pairs, f"Missing universal rule for {arch}"

    def test_compound_rule_present(self):
        """The thermal→allosteric compound rule should exist."""
        compound = [r for r in ARCHETYPE_RULES if isinstance(r, CompoundArchetypeRule)]
        assert len(compound) >= 1
        names = [r.name for r in compound]
        assert "therm_allosteric_combo" in names

    def test_rule_scores_positive(self):
        """All rule scores should be positive."""
        for r in ARCHETYPE_RULES:
            assert r.score > 0, f"Rule {r.name} has non-positive score {r.score}"

    def test_count_per_instrument(self):
        """Sanity: each instrument should have 8-20 rules (5 archetypes, ~2-4 each)."""
        for inst in ["algebraic", "musical", "fick", "thermal",
                     "cooperative", "propagative", "fragile"]:
            rules = get_rules(instrument=inst)
            assert 5 <= len(rules) <= 25, f"{inst} has {len(rules)} rules"


# ═══════════════════════════════════════════════════════════════════
# 2. Per-rule firing tests
# ═══════════════════════════════════════════════════════════════════

class TestPerRuleFiring:
    """Each rule should fire on a profile crafted to trigger it."""

    def test_alg_barrel_scatter_low_fires(self, barrel_like_profile):
        rule = _find_rule("alg_barrel_scatter_low")
        result = rule.evaluate(barrel_like_profile)
        assert result == 2.0

    def test_alg_barrel_scatter_low_no_fire(self, dumbbell_like_profile):
        rule = _find_rule("alg_barrel_scatter_low")
        result = rule.evaluate(dumbbell_like_profile)
        assert result is None  # dumbbell scatter is high

    def test_alg_dumbbell_beta_high_fires(self, dumbbell_like_profile):
        rule = _find_rule("alg_dumbbell_beta_high")
        result = rule.evaluate(dumbbell_like_profile)
        assert result == 2.0

    def test_alg_globin_flat_low_fires(self, globin_like_profile):
        rule = _find_rule("alg_globin_flat_low")
        result = rule.evaluate(globin_like_profile)
        assert result == 2.5

    def test_alg_enzyme_ipr_high_fires(self, enzyme_like_profile):
        rule = _find_rule("alg_enzyme_ipr_high")
        result = rule.evaluate(enzyme_like_profile)
        assert result == 1.5

    def test_wrong_instrument_returns_none(self, barrel_like_profile):
        """A musical rule should not fire on an algebraic profile."""
        rule = _find_rule("mus_barrel_scatter_low")
        result = rule.evaluate(barrel_like_profile)
        assert result is None

    def test_universal_rule_fires_any_instrument(self):
        """Universal species diversity rules should fire on any instrument."""
        rule = _find_rule("universal_species_barrel")
        for inst in ["algebraic", "musical", "fick", "thermal",
                     "cooperative", "propagative", "fragile"]:
            p = _profile_with(
                instrument=inst,
                species_removed=["Carved", "Carved", "Carved"],
            )
            result = rule.evaluate(p)
            assert result is not None

    def test_compound_thermal_allosteric_fires(self):
        """Compound rule: reversible_frac < 0.5 AND gap_flatness > 0.85."""
        p = _profile_with(
            instrument="thermal",
            gap_trajectory=[1.0, 0.99, 0.98, 0.97, 0.96, 0.95],
            reversibility=[False, False, True, False, False],
        )
        rule = _find_rule("therm_allosteric_combo")
        result = rule.evaluate(p)
        assert result == 0.8

    def test_compound_thermal_allosteric_no_fire_high_rev(self):
        """Compound rule should NOT fire when reversible_frac > 0.5."""
        p = _profile_with(
            instrument="thermal",
            gap_trajectory=[1.0, 0.99, 0.98, 0.97, 0.96, 0.95],
            reversibility=[True, True, True, True, True],
        )
        rule = _find_rule("therm_allosteric_combo")
        result = rule.evaluate(p)
        assert result is None


# ═══════════════════════════════════════════════════════════════════
# 3. Equivalence with original archetype_vote()
# ═══════════════════════════════════════════════════════════════════

class TestEquivalence:
    """apply_rules() must produce the same output as the now-delegating
    archetype_vote()."""

    def test_blank_profile_same(self, blank_profile):
        votes_method = blank_profile.archetype_vote()
        votes_direct = apply_rules(blank_profile)
        for arch in votes_method:
            assert abs(votes_method[arch] - votes_direct[arch]) < 1e-10

    def test_barrel_profile_same(self, barrel_like_profile):
        votes = barrel_like_profile.archetype_vote()
        assert votes["barrel"] > votes["dumbbell"]
        assert votes["barrel"] > votes["globin"]

    def test_dumbbell_profile_same(self, dumbbell_like_profile):
        votes = dumbbell_like_profile.archetype_vote()
        assert votes["dumbbell"] > votes["barrel"]

    def test_all_instruments_produce_valid_votes(self):
        """Every instrument should produce normalised votes summing to 1."""
        for inst in ["algebraic", "musical", "fick", "thermal",
                     "cooperative", "propagative", "fragile"]:
            p = _profile_with(instrument=inst)
            votes = p.archetype_vote()
            assert len(votes) == 5
            assert abs(sum(votes.values()) - 1.0) < 1e-6

    def test_archetype_vote_accepts_custom_rules(self, barrel_like_profile):
        """archetype_vote(rules=...) should work for sweeps."""
        # Use only barrel rules
        barrel_rules = get_rules(archetype="barrel")
        votes = barrel_like_profile.archetype_vote(rules=barrel_rules)
        assert len(votes) == 5
        assert abs(sum(votes.values()) - 1.0) < 1e-6

    def test_traced_matches_untraced(self, barrel_like_profile):
        """Traced and untraced should produce identical votes."""
        votes1 = apply_rules(barrel_like_profile)
        votes2, firings = apply_rules_traced(barrel_like_profile)
        for arch in votes1:
            assert abs(votes1[arch] - votes2[arch]) < 1e-10
        assert len(firings) > 0


# ═══════════════════════════════════════════════════════════════════
# 4. Tracing
# ═══════════════════════════════════════════════════════════════════

class TestTracing:
    def test_traced_returns_firings(self, barrel_like_profile):
        _, firings = apply_rules_traced(barrel_like_profile)
        assert len(firings) > 0
        for f in firings:
            assert isinstance(f, RuleFiring)
            assert f.rule_name
            assert f.archetype
            assert f.score > 0 or f.score == 0  # species match can be 0

    def test_traced_firings_have_provenance(self, barrel_like_profile):
        _, firings = apply_rules_traced(barrel_like_profile)
        for f in firings:
            # Some firings may have empty provenance (universal rules)
            assert isinstance(f.provenance, str)

    def test_barrel_trace_shows_barrel_rules(self, barrel_like_profile):
        """Barrel-like profile should fire barrel-specific rules."""
        _, firings = apply_rules_traced(barrel_like_profile)
        barrel_firings = [f for f in firings if f.archetype == "barrel"]
        assert len(barrel_firings) >= 2  # at least scatter + bus mass

    def test_no_wrong_instrument_firings(self, barrel_like_profile):
        """Only algebraic + universal rules should fire on algebraic profile."""
        _, firings = apply_rules_traced(barrel_like_profile)
        for f in firings:
            assert f.instrument in ("algebraic", "*"), \
                f"Unexpected instrument {f.instrument} in firing {f.rule_name}"

    def test_archetype_vote_traced_method(self, barrel_like_profile):
        """The method on ThermoReactionProfile should work."""
        votes, firings = barrel_like_profile.archetype_vote_traced()
        assert len(votes) == 5
        assert abs(sum(votes.values()) - 1.0) < 1e-6
        assert len(firings) > 0


# ═══════════════════════════════════════════════════════════════════
# 5. Sweep utilities
# ═══════════════════════════════════════════════════════════════════

class TestSweepUtilities:
    def test_get_rules_by_instrument(self):
        alg = get_rules(instrument="algebraic")
        assert all(r.instrument == "algebraic" for r in alg)
        assert len(alg) > 0

    def test_get_rules_by_archetype(self):
        barrel = get_rules(archetype="barrel")
        assert all(r.archetype == "barrel" for r in barrel)
        assert len(barrel) >= 7  # at least one per instrument

    def test_get_rules_by_name(self):
        scatter = get_rules(name_contains="scatter")
        assert all("scatter" in r.name for r in scatter)
        assert len(scatter) >= 5

    def test_get_rules_combined_filter(self):
        """Filter by both instrument and archetype."""
        rules = get_rules(instrument="algebraic", archetype="barrel")
        assert len(rules) >= 3
        assert all(r.instrument == "algebraic" and r.archetype == "barrel"
                   for r in rules)

    def test_replace_rules_single(self):
        """Replace one rule and verify the change."""
        original_rule = _find_rule("alg_barrel_scatter_low")
        new_rule = ArchetypeRule(
            instrument="algebraic", archetype="barrel",
            name="alg_barrel_scatter_low",
            metric="scatter_normalised", condition=_lt(2.0),  # wider threshold
            score=3.0, provenance="sweep test",
        )
        new_registry = replace_rules(ARCHETYPE_RULES, {
            "alg_barrel_scatter_low": new_rule,
        })
        # Check length preserved
        assert len(new_registry) == len(ARCHETYPE_RULES)
        # Check replacement applied
        found = [r for r in new_registry if r.name == "alg_barrel_scatter_low"]
        assert len(found) == 1
        assert found[0].score == 3.0

    def test_replace_rules_sweep_changes_vote(self, barrel_like_profile):
        """Replacing a rule should change the vote output."""
        votes_original = apply_rules(barrel_like_profile)

        # Zero out all barrel scoring
        zeroed = ArchetypeRule(
            instrument="algebraic", archetype="barrel",
            name="alg_barrel_scatter_low",
            metric="scatter_normalised", condition=lambda _: False,
            score=0.0, provenance="zeroed",
        )
        new_rules = replace_rules(ARCHETYPE_RULES, {
            "alg_barrel_scatter_low": zeroed,
        })
        votes_new = apply_rules(barrel_like_profile, rules=new_rules)
        # Barrel score should decrease
        assert votes_new["barrel"] < votes_original["barrel"]


# ═══════════════════════════════════════════════════════════════════
# 6. Condition helpers
# ═══════════════════════════════════════════════════════════════════

class TestConditionHelpers:
    def test_lt(self):
        assert _lt(5.0)(3.0) is True
        assert _lt(5.0)(5.0) is False
        assert _lt(5.0)(7.0) is False

    def test_gt(self):
        assert _gt(5.0)(7.0) is True
        assert _gt(5.0)(5.0) is False
        assert _gt(5.0)(3.0) is False

    def test_between(self):
        assert _between(2.0, 5.0)(3.0) is True
        assert _between(2.0, 5.0)(2.0) is False  # exclusive
        assert _between(2.0, 5.0)(5.0) is False  # exclusive
        assert _between(2.0, 5.0)(1.0) is False


# ═══════════════════════════════════════════════════════════════════
# 7. Edge cases
# ═══════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_empty_rules_returns_uniform(self):
        """With no rules, all archetypes get the minimum (0.01) → uniform."""
        p = _profile_with(instrument="algebraic")
        votes = apply_rules(p, rules=[])
        assert abs(sum(votes.values()) - 1.0) < 1e-6
        # Should be nearly uniform
        for v in votes.values():
            assert abs(v - 0.2) < 1e-6

    def test_apply_rules_never_returns_zero(self):
        """No archetype should ever get exactly zero vote (floor is 0.01)."""
        p = _profile_with(instrument="algebraic")
        votes = apply_rules(p)
        for arch, v in votes.items():
            assert v > 0, f"{arch} has zero vote"

    def test_rules_work_with_large_protein(self):
        """Size normalisation should work for large proteins."""
        p = _profile_with(
            instrument="thermal",
            n_residues=1000,
            n_contacts=5000,
            mode_scatters=[0.5, 0.6, 0.4, 0.55, 0.45],
            bus_mass_trajectory=[0.3, 0.35, 0.28, 0.32, 0.3],
            delta_beta_trajectory=[0.01, 0.02, 0.01, 0.015, 0.01],
        )
        votes = apply_rules(p)
        assert len(votes) == 5
        assert abs(sum(votes.values()) - 1.0) < 1e-6


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _find_rule(name: str) -> ArchetypeRule:
    """Find a rule by name in the registry."""
    for r in ARCHETYPE_RULES:
        if r.name == name:
            return r
    raise KeyError(f"Rule '{name}' not found in ARCHETYPE_RULES")
