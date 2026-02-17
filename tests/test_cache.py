"""Tests for profile caching (ibp_enm.cache).

Verifies serialisation round-tripping and the ProfileCache
disk-backed cache.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from ibp_enm.cache import (
    ProfileCache,
    profile_to_dict,
    profile_from_dict,
    profiles_to_json,
    profiles_from_json,
)
from ibp_enm.instruments import ThermoReactionProfile


# ── Fixtures ────────────────────────────────────────────────────

@pytest.fixture
def sample_profile():
    """A realistic ThermoReactionProfile with all fields populated."""
    return ThermoReactionProfile(
        instrument="algebraic",
        intent_idx=0,
        n_residues=164,
        n_contacts=523,
        gap_trajectory=[1.0, 0.98, 0.95, 0.93, 0.91],
        species_removed=["bridge", "carved", "sibling", "bridge", "soft"],
        reversibility=[True, False, True, True, False],
        tensions=[0.1, 0.15, 0.2, 0.18, 0.22],
        mode_scatters=[2.5, 3.1, 2.8, 4.0, 1.9],
        entropy_trajectory=[12.5, 12.3, 12.1, 11.9, 11.7, 11.5],
        heat_cap_trajectory=[3.2, 3.1, 3.0, 2.9, 2.8, 2.7],
        free_energy_trajectory=[-5.0, -4.9, -4.8, -4.7, -4.6, -4.5],
        delta_entropy_per_cut=[-0.2, -0.2, -0.2, -0.2, -0.2],
        ipr_trajectory=[0.03, 0.031, 0.029, 0.032, 0.028, 0.030],
        delta_tau_trajectory=[0.5, 0.3, 0.4, 0.6, 0.2],
        delta_beta_trajectory=[0.02, 0.03, 0.01, 0.05, 0.02],
        spatial_radius_trajectory=[5.0, 6.0, 4.5, 7.0, 5.5],
        bus_mass_trajectory=[0.8, 0.7, 0.9, 0.6, 0.85],
        intent_switches=1,
        cuts_made=5,
    )


@pytest.fixture
def seven_profiles(sample_profile):
    """Seven profiles, one per instrument."""
    instruments = [
        "algebraic", "musical", "fick", "thermal",
        "cooperative", "propagative", "fragile",
    ]
    profiles = []
    for i, inst in enumerate(instruments):
        p = ThermoReactionProfile(
            instrument=inst,
            intent_idx=i,
            n_residues=164,
            n_contacts=523,
            gap_trajectory=[1.0, 0.98, 0.95],
            species_removed=["bridge", "carved", "sibling"],
            reversibility=[True, False, True],
            tensions=[0.1, 0.15, 0.2],
            mode_scatters=[2.5, 3.1, 2.8],
            entropy_trajectory=[12.5, 12.3, 12.1, 11.9],
            heat_cap_trajectory=[3.2, 3.1, 3.0, 2.9],
            free_energy_trajectory=[-5.0, -4.9, -4.8, -4.7],
            delta_entropy_per_cut=[-0.2, -0.2, -0.2],
            ipr_trajectory=[0.03, 0.031, 0.029, 0.032],
            delta_tau_trajectory=[0.5, 0.3, 0.4],
            delta_beta_trajectory=[0.02, 0.03, 0.01],
            spatial_radius_trajectory=[5.0, 6.0, 4.5],
            bus_mass_trajectory=[0.8, 0.7, 0.9],
            intent_switches=0,
            cuts_made=3,
        )
        profiles.append(p)
    return profiles


# ═══════════════════════════════════════════════════════════════════
# Serialisation round-trip tests
# ═══════════════════════════════════════════════════════════════════

class TestProfileSerialisation:
    """Test profile → dict → profile round-tripping."""

    def test_to_dict_returns_dict(self, sample_profile):
        d = profile_to_dict(sample_profile)
        assert isinstance(d, dict)

    def test_to_dict_contains_instrument(self, sample_profile):
        d = profile_to_dict(sample_profile)
        assert d["instrument"] == "algebraic"

    def test_to_dict_contains_all_trajectories(self, sample_profile):
        d = profile_to_dict(sample_profile)
        for key in ["gap_trajectory", "entropy_trajectory",
                     "mode_scatters", "delta_tau_trajectory",
                     "delta_beta_trajectory", "spatial_radius_trajectory",
                     "bus_mass_trajectory"]:
            assert key in d, f"Missing key: {key}"
            assert isinstance(d[key], list)

    def test_round_trip_preserves_instrument(self, sample_profile):
        d = profile_to_dict(sample_profile)
        p2 = profile_from_dict(d)
        assert p2.instrument == sample_profile.instrument

    def test_round_trip_preserves_n_residues(self, sample_profile):
        d = profile_to_dict(sample_profile)
        p2 = profile_from_dict(d)
        assert p2.n_residues == sample_profile.n_residues

    def test_round_trip_preserves_gap_trajectory(self, sample_profile):
        d = profile_to_dict(sample_profile)
        p2 = profile_from_dict(d)
        assert p2.gap_trajectory == sample_profile.gap_trajectory

    def test_round_trip_preserves_species(self, sample_profile):
        d = profile_to_dict(sample_profile)
        p2 = profile_from_dict(d)
        assert p2.species_removed == sample_profile.species_removed

    def test_round_trip_preserves_reversibility(self, sample_profile):
        d = profile_to_dict(sample_profile)
        p2 = profile_from_dict(d)
        assert p2.reversibility == sample_profile.reversibility

    def test_round_trip_preserves_cuts_made(self, sample_profile):
        d = profile_to_dict(sample_profile)
        p2 = profile_from_dict(d)
        assert p2.cuts_made == sample_profile.cuts_made

    def test_round_trip_preserves_computed_properties(self, sample_profile):
        """Computed properties (gap_flatness etc.) should match after round-trip."""
        d = profile_to_dict(sample_profile)
        p2 = profile_from_dict(d)
        assert abs(p2.gap_flatness - sample_profile.gap_flatness) < 1e-10
        assert abs(p2.mean_scatter - sample_profile.mean_scatter) < 1e-10
        assert abs(p2.mean_ipr - sample_profile.mean_ipr) < 1e-10
        assert abs(p2.scatter_normalised - sample_profile.scatter_normalised) < 1e-10

    def test_dict_is_json_serialisable(self, sample_profile):
        d = profile_to_dict(sample_profile)
        text = json.dumps(d)  # should not raise
        assert isinstance(text, str)

    def test_numpy_scalar_conversion(self):
        """numpy scalars in trajectories should be converted to Python floats."""
        p = ThermoReactionProfile(
            instrument="thermal",
            intent_idx=3,
            gap_trajectory=[np.float64(1.0), np.float64(0.95)],
            mode_scatters=[np.float32(2.5)],
        )
        d = profile_to_dict(p)
        assert isinstance(d["gap_trajectory"][0], float)
        assert isinstance(d["mode_scatters"][0], float)


class TestProfilesJson:
    """Test multi-profile JSON serialisation."""

    def test_json_round_trip(self, seven_profiles):
        text = profiles_to_json(seven_profiles, metadata={"name": "test"})
        profiles2, meta = profiles_from_json(text)
        assert len(profiles2) == 7
        assert meta["name"] == "test"

    def test_json_preserves_instruments(self, seven_profiles):
        text = profiles_to_json(seven_profiles)
        profiles2, _ = profiles_from_json(text)
        instruments = [p.instrument for p in profiles2]
        assert instruments == [p.instrument for p in seven_profiles]

    def test_json_version_field(self, seven_profiles):
        text = profiles_to_json(seven_profiles)
        payload = json.loads(text)
        assert payload["version"] == 2
        assert payload["n_profiles"] == 7

    def test_vote_consistency_after_round_trip(self, seven_profiles):
        """archetype_vote() should return identical results after round-trip."""
        original_votes = [p.archetype_vote() for p in seven_profiles]
        text = profiles_to_json(seven_profiles)
        profiles2, _ = profiles_from_json(text)
        round_trip_votes = [p.archetype_vote() for p in profiles2]
        for orig, rt in zip(original_votes, round_trip_votes):
            for arch in orig:
                assert abs(orig[arch] - rt[arch]) < 1e-10, \
                    f"Vote mismatch for {arch}"


# ═══════════════════════════════════════════════════════════════════
# ProfileCache tests
# ═══════════════════════════════════════════════════════════════════

class TestProfileCache:
    """Test the disk-backed profile cache."""

    @pytest.fixture
    def cache(self, tmp_path):
        return ProfileCache(tmp_path / "test_cache")

    def test_cache_empty_initially(self, cache):
        assert cache.list_cached() == []
        assert not cache.has("2LZM", "A")

    def test_save_creates_file(self, cache, seven_profiles):
        path = cache.save("2LZM", "A", seven_profiles)
        assert path.exists()
        assert cache.has("2LZM", "A")

    def test_load_returns_profiles(self, cache, seven_profiles):
        cache.save("2LZM", "A", seven_profiles, metadata={"N": 164})
        profiles, meta = cache.load("2LZM", "A")
        assert len(profiles) == 7
        assert meta["N"] == 164

    def test_load_missing_raises(self, cache):
        with pytest.raises(FileNotFoundError):
            cache.load("XXXX", "A")

    def test_list_cached(self, cache, seven_profiles):
        cache.save("2LZM", "A", seven_profiles)
        cache.save("1LYZ", "A", seven_profiles)
        cached = cache.list_cached()
        assert len(cached) == 2
        keys = {(pdb, ch) for pdb, ch in cached}
        assert ("2LZM", "A") in keys
        assert ("1LYZ", "A") in keys

    def test_clear(self, cache, seven_profiles):
        cache.save("2LZM", "A", seven_profiles)
        cache.save("1LYZ", "A", seven_profiles)
        removed = cache.clear()
        assert removed == 2
        assert cache.list_cached() == []

    def test_case_insensitive_pdb_id(self, cache, seven_profiles):
        cache.save("2lzm", "A", seven_profiles)
        assert cache.has("2LZM", "A")

    def test_repr(self, cache, seven_profiles):
        cache.save("2LZM", "A", seven_profiles)
        r = repr(cache)
        assert "ProfileCache" in r
        assert "1 proteins" in r

    def test_round_trip_votes_match(self, cache, seven_profiles):
        """Votes must be identical before and after cache round-trip."""
        original_votes = [p.archetype_vote() for p in seven_profiles]
        cache.save("2LZM", "A", seven_profiles)
        loaded, _ = cache.load("2LZM", "A")
        for orig, loaded_p in zip(original_votes, loaded):
            loaded_vote = loaded_p.archetype_vote()
            for arch in orig:
                assert abs(orig[arch] - loaded_vote[arch]) < 1e-10
