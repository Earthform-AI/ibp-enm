"""Profile caching — skip re-carving when tuning scoring rules.

The most expensive operation in the classification pipeline is
:meth:`ThermodynamicBand.play` (~2 min per protein × 52 proteins
= ~100 min).  But when iterating on scoring thresholds, lens
gates, or context boosts, only the *interpretation* of the
carving profiles changes — not the profiles themselves.

This module provides serialisation for :class:`ThermoReactionProfile`
and a :class:`ProfileCache` that stores/retrieves pre-computed
profiles keyed by ``(pdb_id, chain)``.

Workflow
--------
>>> cache = ProfileCache("~/.ibp_enm_cache")
>>>
>>> # First run: carve and cache
>>> result = run_single_protein("2LZM", "A")
>>> cache.save("2LZM", "A", band)           # saves 7 profiles
>>>
>>> # Later: re-score in <0.1s without re-carving
>>> profiles = cache.load("2LZM", "A")
>>> new_scores = my_new_synthesiser.synthesize_identity(profiles, meta)
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .instruments import ThermoReactionProfile

__all__ = [
    "ProfileCache",
    "profile_to_dict",
    "profile_from_dict",
    "profiles_to_json",
    "profiles_from_json",
]


# ═══════════════════════════════════════════════════════════════════
# Serialisation helpers
# ═══════════════════════════════════════════════════════════════════

def profile_to_dict(profile: ThermoReactionProfile) -> Dict[str, Any]:
    """Convert a ThermoReactionProfile to a JSON-serialisable dict."""
    d = asdict(profile)
    # asdict handles lists of floats/strings/bools natively.
    # numpy scalars need explicit conversion.
    return _numpy_safe(d)


def _numpy_safe(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays to native Python types."""
    if isinstance(obj, dict):
        return {k: _numpy_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_numpy_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [_numpy_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def profile_from_dict(d: Dict[str, Any]) -> ThermoReactionProfile:
    """Reconstruct a ThermoReactionProfile from a dict."""
    return ThermoReactionProfile(**d)


def profiles_to_json(
    profiles: List[ThermoReactionProfile],
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Serialise a list of profiles (one per instrument) to JSON string."""
    payload = {
        "version": 2,
        "n_profiles": len(profiles),
        "profiles": [profile_to_dict(p) for p in profiles],
    }
    if metadata:
        payload["metadata"] = _numpy_safe(metadata)
    return json.dumps(payload, indent=2)


def profiles_from_json(text: str) -> Tuple[List[ThermoReactionProfile], Dict]:
    """Deserialise profiles from JSON string.

    Returns
    -------
    profiles : list[ThermoReactionProfile]
    metadata : dict
    """
    payload = json.loads(text)
    profiles = [profile_from_dict(d) for d in payload["profiles"]]
    metadata = payload.get("metadata", {})
    return profiles, metadata


# ═══════════════════════════════════════════════════════════════════
# ProfileCache — disk-backed cache keyed by (pdb_id, chain)
# ═══════════════════════════════════════════════════════════════════

class ProfileCache:
    """Disk cache for pre-computed carving profiles.

    Stores one JSON file per protein under ``cache_dir``.

    Parameters
    ----------
    cache_dir : str or Path
        Directory for cached profiles.  Created on first write.
    """

    def __init__(self, cache_dir: str | Path = "~/.ibp_enm_cache"):
        self.cache_dir = Path(cache_dir).expanduser()

    def _key(self, pdb_id: str, chain: str) -> str:
        return f"{pdb_id.upper()}_{chain}"

    def _path(self, pdb_id: str, chain: str) -> Path:
        return self.cache_dir / f"{self._key(pdb_id, chain)}.json"

    def has(self, pdb_id: str, chain: str) -> bool:
        """Check whether profiles are cached for this protein."""
        return self._path(pdb_id, chain).exists()

    def is_complete(self, pdb_id: str, chain: str) -> bool:
        """Check whether the cache entry has actual profile data.

        Returns ``False`` if:
        * the file does not exist,
        * the profiles list is empty, or
        * ``per_instrument`` metadata is missing.

        Use this instead of :meth:`has` to detect stale entries
        created by older benchmark versions that saved ``profiles=[]``.
        """
        path = self._path(pdb_id, chain)
        if not path.exists():
            return False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return False
        if payload.get("n_profiles", 0) > 0:
            return True
        # Summary-level cache: check metadata completeness
        meta = payload.get("metadata", {})
        return bool(meta.get("per_instrument")) and bool(
            meta.get("identity_result"))

    def list_stale(self) -> list[tuple[str, str]]:
        """Return ``(pdb_id, chain)`` pairs for incomplete cache entries."""
        result = []
        for pdb_id, chain in self.list_cached():
            if not self.is_complete(pdb_id, chain):
                result.append((pdb_id, chain))
        return result

    def save(
        self,
        pdb_id: str,
        chain: str,
        profiles: List[ThermoReactionProfile],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save profiles to the cache.  Returns the file path."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._path(pdb_id, chain)
        text = profiles_to_json(profiles, metadata)
        path.write_text(text, encoding="utf-8")
        return path

    def load(
        self, pdb_id: str, chain: str
    ) -> Tuple[List[ThermoReactionProfile], Dict]:
        """Load profiles from the cache.

        Raises
        ------
        FileNotFoundError
            If no cached profiles exist for this protein.
        """
        path = self._path(pdb_id, chain)
        if not path.exists():
            raise FileNotFoundError(
                f"No cached profiles for {pdb_id}:{chain} at {path}")
        text = path.read_text(encoding="utf-8")
        return profiles_from_json(text)

    def list_cached(self) -> List[Tuple[str, str]]:
        """Return a list of ``(pdb_id, chain)`` tuples currently cached."""
        if not self.cache_dir.exists():
            return []
        result = []
        for p in self.cache_dir.glob("*.json"):
            stem = p.stem
            if "_" in stem:
                parts = stem.rsplit("_", 1)
                result.append((parts[0], parts[1]))
        return result

    def clear(self) -> int:
        """Remove all cached profiles.  Returns count of files removed."""
        if not self.cache_dir.exists():
            return 0
        count = 0
        for p in self.cache_dir.glob("*.json"):
            p.unlink()
            count += 1
        return count

    def invalidate(self, pdb_id: str, chain: str) -> bool:
        """Remove a specific cache entry.  Returns True if file existed."""
        path = self._path(pdb_id, chain)
        if path.exists():
            path.unlink()
            return True
        return False

    def repair(self, *, verbose: bool = False) -> int:
        """Fill in missing ``per_instrument`` metadata from stored profiles.

        Entries that have serialised :class:`ThermoReactionProfile`
        objects but lack ``per_instrument`` or ``identity_result``
        get those fields recomputed and written back to disk.

        Returns the number of entries repaired.
        """
        from collections import Counter

        repaired = 0
        for pdb_id, chain in self.list_cached():
            path = self._path(pdb_id, chain)
            payload = json.loads(path.read_text(encoding="utf-8"))

            profiles_raw = payload.get("profiles", [])
            meta = payload.get("metadata", {})
            has_pi = bool(meta.get("per_instrument"))
            has_ir = bool(meta.get("identity_result"))

            if not profiles_raw or (has_pi and has_ir):
                continue  # nothing to repair

            # Deserialise profiles
            profiles = [profile_from_dict(d) for d in profiles_raw]

            if not has_pi:
                per_instrument = {}
                for p in profiles:
                    per_instrument[p.instrument] = {
                        "gap_retained": p.gap_retained,
                        "gap_flatness": p.gap_flatness,
                        "gap_volatility": p.gap_volatility,
                        "gap_trend": p.gap_trend,
                        "species_entropy": p.species_entropy,
                        "reversible_frac": p.reversible_frac,
                        "mean_scatter": p.mean_scatter,
                        "entropy_change": p.entropy_change,
                        "mean_delta_S": p.mean_delta_entropy,
                        "entropy_volatility": p.entropy_volatility,
                        "heat_cap_change": p.heat_cap_change,
                        "free_energy_cost": p.free_energy_cost,
                        "mean_ipr": p.mean_ipr,
                        "mean_spatial_radius": p.mean_spatial_radius,
                        "max_spatial_radius": p.max_spatial_radius,
                        "mean_delta_beta": p.mean_delta_beta,
                        "mean_bus_mass": p.mean_bus_mass,
                        "intent_switches": p.intent_switches,
                        "cuts_made": p.cuts_made,
                        "species": dict(Counter(p.species_removed)),
                        "vote": p.archetype_vote(),
                    }
                meta["per_instrument"] = _numpy_safe(per_instrument)

            if not has_ir:
                # Re-synthesize identity from profiles
                try:
                    from .lens_stack import LensStackSynthesizer
                    synth = LensStackSynthesizer(
                        evals=None, evecs=None,
                        domain_labels=None, contacts=None,
                    )
                    final_votes = [p.archetype_vote() for p in profiles]
                    meta_state = synth.compute_meta_fick_state(final_votes)
                    identity = synth.synthesize_identity(
                        profiles, meta_state)
                    meta["identity_result"] = _numpy_safe({
                        k: v for k, v in identity.items()
                        if k not in (
                            "per_carver_votes", "trace", "lens_traces")
                    })
                except Exception:
                    pass  # leave identity_result empty

            payload["metadata"] = meta
            path.write_text(
                json.dumps(payload, indent=2), encoding="utf-8")
            repaired += 1
            if verbose:
                import sys
                print(f"  [cache] repaired {pdb_id}_{chain}",
                      file=sys.stderr, flush=True)

        return repaired

    def __repr__(self) -> str:
        cached = self.list_cached()
        stale = sum(1 for pid, ch in cached
                    if not self.is_complete(pid, ch))
        return (f"ProfileCache({self.cache_dir!s}, "
                f"{len(cached)} proteins, {stale} stale)")
