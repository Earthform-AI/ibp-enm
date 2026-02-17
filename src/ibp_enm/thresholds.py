"""ThresholdRegistry — every tunable number in one place.

Extracts the ~90 magic numbers from the lens stack and the
MetaFickBalancer context-boost section into a typed, immutable
registry that can be:

* **inspected** — ``registry["enzyme_lens.ipr_strong"]``
* **overridden** — ``registry.replace({"enzyme_lens.ipr_strong": 0.030})``
* **diffed** — ``registry.diff(other)``
* **swept** — build 100 registries with one threshold varying

The registry does *not* cover:

* **ArchetypeRule** thresholds — those are already sweepable via
  :func:`~ibp_enm.rules.replace_rules`.
* **Instrument weights** — structural constants in
  :mod:`~ibp_enm.instruments`.
* **Legacy synthesis classes** — preserved for backwards compatibility
  but no longer on the active code path.

Usage
-----
>>> from ibp_enm.thresholds import ThresholdRegistry, DEFAULT_THRESHOLDS
>>> reg = DEFAULT_THRESHOLDS
>>> reg["enzyme_lens.ipr_strong"]           # 0.025
>>> custom = reg.replace({"enzyme_lens.ipr_strong": 0.030})
>>> diff = custom.diff(reg)                 # {'enzyme_lens.ipr_strong': (0.030, 0.025)}

Historical notes
----------------
v0.6.0 — Step 5 of the architectural plan.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple

__all__ = [
    "ThresholdRegistry",
    "DEFAULT_THRESHOLDS",
]


# ═══════════════════════════════════════════════════════════════════
# ThresholdRegistry
# ═══════════════════════════════════════════════════════════════════

class ThresholdRegistry:
    """Immutable typed mapping of dotted threshold keys → float values.

    Parameters
    ----------
    data : dict[str, float]
        ``{"section.name": value, ...}``.
    name : str, optional
        Human-readable label (e.g. ``"production"``, ``"sweep-042"``).

    Notes
    -----
    * Read-only: ``__setitem__`` raises ``TypeError``.
    * ``replace()`` returns a new registry.
    * ``diff()`` compares two registries.
    * Iteration yields ``(key, value)`` pairs.
    """

    def __init__(self, data: Dict[str, float], *, name: str = "custom"):
        self._data: Dict[str, float] = dict(data)
        self._name = name

    # ── read ────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return self._name

    def __getitem__(self, key: str) -> float:
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __repr__(self) -> str:
        return f"ThresholdRegistry({self._name!r}, {len(self._data)} keys)"

    def get(self, key: str, default: float = 0.0) -> float:
        """Return value for *key*, or *default* if missing."""
        return self._data.get(key, default)

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def to_dict(self) -> Dict[str, float]:
        """Return a mutable copy of the data."""
        return dict(self._data)

    # ── immutable mutation ──────────────────────────────────────

    def __setitem__(self, key: str, value: float):
        raise TypeError(
            "ThresholdRegistry is immutable — use .replace() instead")

    def replace(
        self,
        overrides: Dict[str, float],
        *,
        name: Optional[str] = None,
    ) -> "ThresholdRegistry":
        """Return a new registry with selected keys overridden.

        Parameters
        ----------
        overrides : dict
            ``{key: new_value}`` for keys to change.
        name : str, optional
            Name for the new registry.  Defaults to ``self.name + "+"``.

        Raises
        ------
        KeyError
            If any key in *overrides* is not in the registry.
        """
        for k in overrides:
            if k not in self._data:
                raise KeyError(
                    f"Unknown threshold key {k!r}. "
                    f"Valid keys: {sorted(self._data.keys())}"
                )
        merged = dict(self._data)
        merged.update(overrides)
        return ThresholdRegistry(
            merged,
            name=name or (self._name + "+"),
        )

    # ── comparison ──────────────────────────────────────────────

    def diff(
        self, other: "ThresholdRegistry",
    ) -> Dict[str, Tuple[float, float]]:
        """Return ``{key: (self_value, other_value)}`` for differing keys."""
        result = {}
        all_keys = set(self._data) | set(other._data)
        for k in sorted(all_keys):
            v_self = self._data.get(k)
            v_other = other._data.get(k)
            if v_self != v_other:
                result[k] = (v_self, v_other)
        return result

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ThresholdRegistry):
            return NotImplemented
        return self._data == other._data

    # ── section access ──────────────────────────────────────────

    def section(self, prefix: str) -> Dict[str, float]:
        """Return all keys starting with *prefix* as a flat dict.

        >>> reg.section("enzyme_lens")
        {'enzyme_lens.ipr_strong': 0.025, ...}
        """
        return {
            k: v for k, v in self._data.items()
            if k.startswith(prefix + ".")
        }

    @property
    def sections(self) -> Tuple[str, ...]:
        """Return sorted tuple of all section prefixes."""
        prefixes = set()
        for k in self._data:
            dot = k.find(".")
            if dot > 0:
                prefixes.add(k[:dot])
        return tuple(sorted(prefixes))


# ═══════════════════════════════════════════════════════════════════
# DEFAULT_THRESHOLDS — the production config
# ═══════════════════════════════════════════════════════════════════
#
# Naming convention: section.descriptive_name
#   section ∈ {meta_fick, ctx_boost, enzyme_lens, hinge_lens,
#              barrel_penalty, renorm}
#
# Every value below has a comment with provenance (Dxxx = discovery
# experiment number).
# ═══════════════════════════════════════════════════════════════════

_DEFAULT_DATA: Dict[str, float] = {

    # ── meta_fick — MetaFickBalancer defaults (D108) ────────────
    "meta_fick.w1": -1.2,               # log-Δτ weight
    "meta_fick.w2": -0.4,               # β-offset weight
    "meta_fick.w3": 0.8,                # agreement weight
    "meta_fick.beta0": 1.5,             # β baseline for sigmoid
    "meta_fick.beta_fallback": 10.0,    # β when runner-up ≈ 0
    "meta_fick.disagree_epsilon": 0.1,  # disagreement smoothing
    "meta_fick.context_boost_weight": 0.25,  # context-boost weight

    # ── ctx_boost — context-boost thresholds (D109/D113) ────────

    # Barrel
    "ctx_boost.barrel_scatter_low": 1.5,
    "ctx_boost.barrel_scatter_low_boost": 0.8,
    "ctx_boost.barrel_scatter_high": 3.0,
    "ctx_boost.barrel_scatter_high_penalty": -0.5,
    "ctx_boost.barrel_large_n": 350.0,
    "ctx_boost.barrel_large_scatter_gate": 1.2,
    "ctx_boost.barrel_large_penalty": -0.6,
    "ctx_boost.barrel_large_dumbbell_boost": 0.3,
    "ctx_boost.barrel_large_allosteric_boost": 0.3,

    # Dumbbell
    "ctx_boost.dumbbell_db_high": 0.12,
    "ctx_boost.dumbbell_db_high_boost": 0.8,
    "ctx_boost.dumbbell_db_low": 0.03,
    "ctx_boost.dumbbell_db_low_penalty": -0.3,
    "ctx_boost.dumbbell_size_gate_n": 250.0,
    "ctx_boost.dumbbell_size_db_thresh": 0.05,
    "ctx_boost.dumbbell_size_boost": 0.4,

    # Enzyme
    "ctx_boost.enzyme_ipr_high": 0.025,
    "ctx_boost.enzyme_ipr_high_boost": 0.8,
    "ctx_boost.enzyme_ipr_low": 0.017,
    "ctx_boost.enzyme_ipr_low_penalty": -0.3,

    # Allosteric
    "ctx_boost.allosteric_prop_radius_high": 20.0,
    "ctx_boost.allosteric_size_gate_n": 200.0,
    "ctx_boost.allosteric_prop_radius_boost": 0.8,
    "ctx_boost.allosteric_barrel_suppress_scatter": 0.8,
    "ctx_boost.allosteric_barrel_suppress_penalty": -0.3,
    "ctx_boost.allosteric_all_radius_thresh": 15.0,
    "ctx_boost.allosteric_all_radius_boost": 0.4,

    # Per-instrument: algebraic
    "ctx_boost.alg_globin_flat_strong": 0.75,
    "ctx_boost.alg_globin_flat_strong_boost": 0.8,
    "ctx_boost.alg_barrel_flat_penalty": -0.3,
    "ctx_boost.alg_globin_flat_mid": 0.85,
    "ctx_boost.alg_globin_flat_mid_boost": 0.3,
    "ctx_boost.alg_barrel_bus_thresh": 0.5,
    "ctx_boost.alg_barrel_bus_boost": 0.3,

    # Per-instrument: thermal
    "ctx_boost.therm_globin_flat_thresh": 0.75,
    "ctx_boost.therm_globin_flat_boost": 0.6,
    "ctx_boost.therm_barrel_bus_thresh": 0.45,
    "ctx_boost.therm_barrel_bus_boost": 0.4,

    # Per-instrument: propagative
    "ctx_boost.prop_globin_flat_thresh": 0.9,
    "ctx_boost.prop_globin_flat_boost": 0.5,
    "ctx_boost.prop_barrel_rev_thresh": 0.9,
    "ctx_boost.prop_barrel_rev_boost": 0.3,

    # Per-instrument: fragile
    "ctx_boost.frag_dumbbell_db_thresh": 0.15,
    "ctx_boost.frag_dumbbell_db_boost": 0.5,
    "ctx_boost.frag_dumbbell_rev_thresh": 0.5,
    "ctx_boost.frag_dumbbell_rev_boost": 0.3,

    # ── enzyme_lens — EnzymeLens thresholds (D110) ──────────────

    # Activation gate
    "enzyme_lens.close_call_gap": 0.10,
    "enzyme_lens.ea_proximity_gap": 0.15,
    "enzyme_lens.allosteric_counter_ratio": 0.5,

    # IPR signal
    "enzyme_lens.ipr_strong": 0.025,
    "enzyme_lens.ipr_strong_boost": 0.08,
    "enzyme_lens.ipr_weak": 0.020,
    "enzyme_lens.ipr_weak_boost": 0.04,

    # Algebraic instrument
    "enzyme_lens.alg_strong": 0.35,
    "enzyme_lens.alg_strong_boost": 0.06,
    "enzyme_lens.alg_weak": 0.25,
    "enzyme_lens.alg_weak_boost": 0.03,

    # Entropy asymmetry
    "enzyme_lens.gini_thresh": 0.15,
    "enzyme_lens.gini_boost": 0.04,
    "enzyme_lens.cv_thresh": 0.3,
    "enzyme_lens.cv_boost": 0.03,
    "enzyme_lens.top5_thresh": 0.15,
    "enzyme_lens.top5_boost": 0.03,

    # Fragile combo
    "enzyme_lens.fragile_ipr_thresh": 0.025,
    "enzyme_lens.fragile_rev_thresh": 0.8,
    "enzyme_lens.fragile_combo_boost": 0.04,

    # ── hinge_lens — HingeLens thresholds (D111) ────────────────

    # Activation gate
    "hinge_lens.enzyme_nontrivial": 0.05,
    "hinge_lens.allosteric_significant": 0.15,
    "hinge_lens.size_gate_n": 150.0,
    "hinge_lens.hinge_r_gate": 1.0,
    "hinge_lens.allosteric_counter_ratio": 0.5,

    # Boost computation
    "hinge_lens.boost_cap": 0.35,
    "hinge_lens.boost_multiplier": 3.0,

    # ── barrel_penalty — BarrelPenaltyLens thresholds (D113) ────

    # Activation gate
    "barrel_penalty.size_gate_n": 250.0,
    "barrel_penalty.scatter_gate": 0.5,

    # Signal 1: Δβ → dumbbell
    "barrel_penalty.db_thresh": 0.04,
    "barrel_penalty.db_penalty": 0.08,

    # Signal 2: spatial radius → allosteric
    "barrel_penalty.radius_thresh": 12.0,
    "barrel_penalty.radius_penalty": 0.06,

    # Signal 3: multi-domain
    "barrel_penalty.domain_count_gate": 2.0,
    "barrel_penalty.domain_penalty": 0.05,

    # Signal 4: gap flatness
    "barrel_penalty.flatness_thresh": 0.90,
    "barrel_penalty.flatness_penalty": 0.04,

    # Boost allocation
    "barrel_penalty.boost_target_ratio": 0.7,

    # ── renorm — shared renormalisation ─────────────────────────
    "renorm.floor": 0.01,
}


DEFAULT_THRESHOLDS: ThresholdRegistry = ThresholdRegistry(
    _DEFAULT_DATA, name="production",
)
"""The production threshold registry.

Contains all tunable numbers from the LensStack and
MetaFickBalancer context-boost section.
"""
