"""Tests for the ThresholdRegistry (v0.6.0).

Covers:
1. ThresholdRegistry — read, immutability, replace, diff, section
2. DEFAULT_THRESHOLDS — structure, types, section coverage
3. Integration — registry wiring into lenses and MetaFickBalancer
"""

import pytest
from ibp_enm.thresholds import ThresholdRegistry, DEFAULT_THRESHOLDS


# ═══════════════════════════════════════════════════════════════════
# 1. ThresholdRegistry core behaviour
# ═══════════════════════════════════════════════════════════════════

class TestThresholdRegistryRead:
    """Reading keys, contains, len, iter."""

    def test_getitem(self):
        reg = ThresholdRegistry({"a.b": 1.0, "a.c": 2.0})
        assert reg["a.b"] == 1.0
        assert reg["a.c"] == 2.0

    def test_getitem_missing_raises(self):
        reg = ThresholdRegistry({"a.b": 1.0})
        with pytest.raises(KeyError):
            _ = reg["z.z"]

    def test_get_with_default(self):
        reg = ThresholdRegistry({"a.b": 1.0})
        assert reg.get("a.b") == 1.0
        assert reg.get("z.z") == 0.0
        assert reg.get("z.z", -1.0) == -1.0

    def test_contains(self):
        reg = ThresholdRegistry({"a.b": 1.0})
        assert "a.b" in reg
        assert "z.z" not in reg

    def test_len(self):
        reg = ThresholdRegistry({"a.b": 1.0, "a.c": 2.0, "x.y": 3.0})
        assert len(reg) == 3

    def test_iter_yields_keys(self):
        reg = ThresholdRegistry({"a.b": 1.0, "a.c": 2.0})
        assert sorted(reg) == ["a.b", "a.c"]

    def test_keys_values_items(self):
        data = {"a.b": 1.0, "a.c": 2.0}
        reg = ThresholdRegistry(data)
        assert set(reg.keys()) == {"a.b", "a.c"}
        assert set(reg.values()) == {1.0, 2.0}
        assert dict(reg.items()) == data

    def test_to_dict_returns_copy(self):
        reg = ThresholdRegistry({"a.b": 1.0})
        d = reg.to_dict()
        assert d == {"a.b": 1.0}
        d["a.b"] = 999.0  # mutate the copy
        assert reg["a.b"] == 1.0  # original unchanged

    def test_repr(self):
        reg = ThresholdRegistry({"a.b": 1.0}, name="test")
        assert "test" in repr(reg)
        assert "1 keys" in repr(reg)

    def test_name_property(self):
        reg = ThresholdRegistry({}, name="my-reg")
        assert reg.name == "my-reg"

    def test_default_name(self):
        reg = ThresholdRegistry({})
        assert reg.name == "custom"


class TestThresholdRegistryImmutability:
    """Registry is read-only."""

    def test_setitem_raises(self):
        reg = ThresholdRegistry({"a.b": 1.0})
        with pytest.raises(TypeError, match="immutable"):
            reg["a.b"] = 2.0

    def test_setitem_new_key_raises(self):
        reg = ThresholdRegistry({"a.b": 1.0})
        with pytest.raises(TypeError, match="immutable"):
            reg["z.z"] = 2.0

    def test_constructor_does_not_alias(self):
        """Mutating the original dict doesn't affect the registry."""
        data = {"a.b": 1.0}
        reg = ThresholdRegistry(data)
        data["a.b"] = 999.0
        assert reg["a.b"] == 1.0


class TestThresholdRegistryReplace:
    """replace() returns a new registry with overrides."""

    def test_replace_single_key(self):
        reg = ThresholdRegistry({"a.b": 1.0, "a.c": 2.0}, name="orig")
        new = reg.replace({"a.b": 99.0})
        assert new["a.b"] == 99.0
        assert new["a.c"] == 2.0  # unchanged
        assert reg["a.b"] == 1.0  # original unchanged

    def test_replace_multiple_keys(self):
        reg = ThresholdRegistry({"a.b": 1.0, "a.c": 2.0, "x.y": 3.0})
        new = reg.replace({"a.b": 10.0, "x.y": 30.0})
        assert new["a.b"] == 10.0
        assert new["a.c"] == 2.0
        assert new["x.y"] == 30.0

    def test_replace_unknown_key_raises(self):
        reg = ThresholdRegistry({"a.b": 1.0})
        with pytest.raises(KeyError, match="Unknown threshold key"):
            reg.replace({"z.z": 99.0})

    def test_replace_auto_name(self):
        reg = ThresholdRegistry({"a.b": 1.0}, name="base")
        new = reg.replace({"a.b": 2.0})
        assert new.name == "base+"

    def test_replace_custom_name(self):
        reg = ThresholdRegistry({"a.b": 1.0}, name="base")
        new = reg.replace({"a.b": 2.0}, name="sweep-42")
        assert new.name == "sweep-42"

    def test_replace_empty_overrides(self):
        reg = ThresholdRegistry({"a.b": 1.0})
        new = reg.replace({})
        assert new == reg


class TestThresholdRegistryDiff:
    """diff() returns keys that differ between two registries."""

    def test_diff_no_change(self):
        reg = ThresholdRegistry({"a.b": 1.0, "a.c": 2.0})
        assert reg.diff(reg) == {}

    def test_diff_one_changed(self):
        r1 = ThresholdRegistry({"a.b": 1.0, "a.c": 2.0})
        r2 = r1.replace({"a.b": 99.0})
        d = r1.diff(r2)
        assert d == {"a.b": (1.0, 99.0)}

    def test_diff_extra_key_in_other(self):
        r1 = ThresholdRegistry({"a.b": 1.0})
        r2 = ThresholdRegistry({"a.b": 1.0, "a.c": 2.0})
        d = r1.diff(r2)
        assert "a.c" in d
        assert d["a.c"][0] is None  # missing in self
        assert d["a.c"][1] == 2.0

    def test_diff_extra_key_in_self(self):
        r1 = ThresholdRegistry({"a.b": 1.0, "a.c": 2.0})
        r2 = ThresholdRegistry({"a.b": 1.0})
        d = r1.diff(r2)
        assert d["a.c"] == (2.0, None)


class TestThresholdRegistryEquality:
    """Equality checks."""

    def test_equal_registries(self):
        r1 = ThresholdRegistry({"a.b": 1.0})
        r2 = ThresholdRegistry({"a.b": 1.0})
        assert r1 == r2

    def test_unequal_registries(self):
        r1 = ThresholdRegistry({"a.b": 1.0})
        r2 = ThresholdRegistry({"a.b": 2.0})
        assert r1 != r2

    def test_not_equal_to_non_registry(self):
        reg = ThresholdRegistry({"a.b": 1.0})
        assert reg != {"a.b": 1.0}


class TestThresholdRegistrySections:
    """Section-based access."""

    def test_section_returns_matching_keys(self):
        reg = ThresholdRegistry({
            "alpha.x": 1.0, "alpha.y": 2.0,
            "beta.x": 3.0,
        })
        assert reg.section("alpha") == {"alpha.x": 1.0, "alpha.y": 2.0}
        assert reg.section("beta") == {"beta.x": 3.0}

    def test_section_no_match(self):
        reg = ThresholdRegistry({"alpha.x": 1.0})
        assert reg.section("gamma") == {}

    def test_sections_property(self):
        reg = ThresholdRegistry({
            "a.x": 1.0, "b.x": 2.0, "c.x": 3.0,
        })
        assert reg.sections == ("a", "b", "c")

    def test_sections_empty(self):
        reg = ThresholdRegistry({"noDot": 1.0})
        assert reg.sections == ()


# ═══════════════════════════════════════════════════════════════════
# 2. DEFAULT_THRESHOLDS validation
# ═══════════════════════════════════════════════════════════════════

class TestDefaultThresholds:
    """DEFAULT_THRESHOLDS has the right shape and content."""

    def test_is_registry(self):
        assert isinstance(DEFAULT_THRESHOLDS, ThresholdRegistry)

    def test_name(self):
        assert DEFAULT_THRESHOLDS.name == "production"

    def test_has_expected_sections(self):
        expected = {
            "meta_fick", "ctx_boost", "enzyme_lens",
            "hinge_lens", "barrel_penalty", "renorm",
        }
        assert set(DEFAULT_THRESHOLDS.sections) == expected

    def test_at_least_80_keys(self):
        """We set out to extract ~90 magic numbers."""
        assert len(DEFAULT_THRESHOLDS) >= 80

    def test_all_values_are_numeric(self):
        for key, val in DEFAULT_THRESHOLDS.items():
            assert isinstance(val, (int, float)), (
                f"{key} has type {type(val)}, expected float"
            )

    def test_all_keys_have_section_prefix(self):
        """Every key must have a dotted section prefix."""
        for key in DEFAULT_THRESHOLDS:
            assert "." in key, f"Key {key!r} lacks section prefix"

    # Spot-check specific thresholds for regression
    @pytest.mark.parametrize("key,expected", [
        ("meta_fick.w1", -1.2),
        ("meta_fick.beta_fallback", 10.0),
        ("meta_fick.context_boost_weight", 0.25),
        ("ctx_boost.barrel_scatter_low", 1.5),
        ("ctx_boost.dumbbell_db_high", 0.12),
        ("ctx_boost.enzyme_ipr_high", 0.025),
        ("enzyme_lens.ipr_strong", 0.025),
        ("enzyme_lens.close_call_gap", 0.10),
        ("hinge_lens.boost_cap", 0.35),
        ("barrel_penalty.size_gate_n", 250.0),
        ("renorm.floor", 0.01),
    ])
    def test_production_values(self, key, expected):
        assert DEFAULT_THRESHOLDS[key] == pytest.approx(expected)

    def test_section_sizes(self):
        """Each section should have a reasonable number of keys."""
        for sec in DEFAULT_THRESHOLDS.sections:
            size = len(DEFAULT_THRESHOLDS.section(sec))
            assert size >= 1, f"Section {sec} is empty"


# ═══════════════════════════════════════════════════════════════════
# 3. Integration — registry affects lens behaviour
# ═══════════════════════════════════════════════════════════════════

class TestRegistryIntegration:
    """Verify that changing registry values actually changes behaviour."""

    def test_custom_thresholds_roundtrip(self):
        """replace + diff roundtrip."""
        custom = DEFAULT_THRESHOLDS.replace(
            {"enzyme_lens.ipr_strong": 0.999},
            name="test-sweep",
        )
        diff = custom.diff(DEFAULT_THRESHOLDS)
        assert len(diff) == 1
        assert "enzyme_lens.ipr_strong" in diff
        assert diff["enzyme_lens.ipr_strong"] == (0.999, 0.025)

    def test_sweep_pattern(self):
        """Simulating a threshold sweep creates distinct registries."""
        import numpy as np
        sweep_values = np.linspace(0.01, 0.05, 5)
        registries = [
            DEFAULT_THRESHOLDS.replace(
                {"enzyme_lens.ipr_strong": float(v)},
                name=f"sweep-{i}",
            )
            for i, v in enumerate(sweep_values)
        ]
        # All should be distinct
        for i in range(len(registries)):
            for j in range(i + 1, len(registries)):
                assert registries[i] != registries[j]

    def test_build_default_stack_accepts_thresholds(self):
        """build_default_stack can take a custom ThresholdRegistry."""
        from ibp_enm.lens_stack import build_default_stack
        custom = DEFAULT_THRESHOLDS.replace(
            {"enzyme_lens.close_call_gap": 0.50}
        )
        stack = build_default_stack(thresholds=custom)
        assert len(stack) == 3

    def test_lens_stack_synthesizer_accepts_thresholds(self):
        """LensStackSynthesizer can be constructed with custom thresholds."""
        from ibp_enm.lens_stack import LensStackSynthesizer
        custom = DEFAULT_THRESHOLDS.replace(
            {"hinge_lens.boost_cap": 0.99}
        )
        synth = LensStackSynthesizer(thresholds=custom)
        assert synth._t is custom

    def test_meta_fick_accepts_thresholds(self):
        """MetaFickBalancer stores the registry."""
        from ibp_enm.synthesis import MetaFickBalancer
        custom = DEFAULT_THRESHOLDS.replace(
            {"meta_fick.beta_fallback": 42.0}
        )
        mfb = MetaFickBalancer(thresholds=custom)
        assert mfb._t["meta_fick.beta_fallback"] == 42.0

    def test_meta_fick_default_thresholds(self):
        """MetaFickBalancer uses DEFAULT_THRESHOLDS when none given."""
        from ibp_enm.synthesis import MetaFickBalancer
        mfb = MetaFickBalancer()
        assert mfb._t is DEFAULT_THRESHOLDS

    def test_public_api_exports(self):
        """ThresholdRegistry and DEFAULT_THRESHOLDS in ibp_enm.__init__."""
        import ibp_enm
        assert hasattr(ibp_enm, "ThresholdRegistry")
        assert hasattr(ibp_enm, "DEFAULT_THRESHOLDS")
        assert ibp_enm.DEFAULT_THRESHOLDS is DEFAULT_THRESHOLDS
