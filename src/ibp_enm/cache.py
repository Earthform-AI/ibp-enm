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

    def __repr__(self) -> str:
        n = len(self.list_cached())
        return f"ProfileCache({self.cache_dir!s}, {n} proteins)"
