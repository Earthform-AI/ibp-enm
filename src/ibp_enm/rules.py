"""ArchetypeRule decomposition — testable, sweepable vote building blocks.

Every scoring condition that was previously embedded in the 400-line
``archetype_vote()`` if/elif cascade is now a self-contained
:class:`ArchetypeRule` object.  Rules live in the :data:`ARCHETYPE_RULES`
registry and can be:

* **unit-tested** individually (does rule X fire for this profile?),
* **swept** programmatically (change one threshold, rerun benchmark),
* **traced** to produce audit trails (which rules fired, with what score?).

Architecture
------------
An ``ArchetypeRule`` is a frozen dataclass holding:

* **instrument** – which carver this rule applies to (or ``"*"`` for universal)
* **archetype** – which archetype the rule votes for
* **name** – human-readable label (unique across the registry)
* **metric** – the ``ThermoReactionProfile`` property this rule inspects
* **condition** – a callable ``(value) -> bool`` that decides whether to fire
* **score** – the score contribution when the condition is met
* **provenance** – short note (e.g. ``"D109 d=2.1"``, ``"D113 barrel fix"``)

The function :func:`apply_rules` takes a :class:`ThermoReactionProfile`
and returns a raw ``{archetype: score}`` dict (un-normalised).
:func:`apply_rules_traced` returns the same dict *plus* a list of
:class:`RuleFiring` records for audit.

Historical notes
----------------
Rules extracted from ``ThermoReactionProfile.archetype_vote()`` which was
introduced in D109 and modified in D113 (size-aware barrel fix).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .instruments import ThermoReactionProfile

from .archetypes import ARCHETYPE_EXPECTATIONS

__all__ = [
    "ArchetypeRule",
    "RuleFiring",
    "ARCHETYPE_RULES",
    "apply_rules",
    "apply_rules_traced",
    "get_rules",
    "replace_rules",
]


# ═══════════════════════════════════════════════════════════════════
# ArchetypeRule — one testable scoring condition
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ArchetypeRule:
    """One atomic scoring condition in the archetype vote.

    Parameters
    ----------
    instrument : str
        Carver instrument this rule applies to, or ``"*"`` for universal.
    archetype : str
        Archetype this rule votes for (e.g. ``"barrel"``).
    name : str
        Unique human-readable label for the rule.
    metric : str
        Name of the ``ThermoReactionProfile`` property to read.
    condition : callable
        ``(value) -> bool``.  The rule fires when this returns ``True``.
    score : float
        Score contribution when the rule fires.
    provenance : str
        Short note on the origin (e.g. ``"D109 d=2.1"``).
    """

    instrument: str
    archetype: str
    name: str
    metric: str
    condition: Callable[[float], bool]
    score: float
    provenance: str = ""

    def evaluate(self, profile: "ThermoReactionProfile") -> Optional[float]:
        """Return the score if the rule fires, else ``None``.

        Returns ``None`` when:
        * the instrument doesn't match
        * the condition is not met
        """
        if self.instrument != "*" and profile.instrument != self.instrument:
            return None

        value = self._read_metric(profile)
        if self.condition(value):
            return self.score
        return None

    def _read_metric(self, profile: "ThermoReactionProfile") -> float:
        """Read a metric value from the profile.

        For the special ``"species_diversity_match:<archetype>"`` metric,
        computes the species-entropy match score directly.
        """
        if self.metric.startswith("species_diversity_match"):
            return self._species_match(profile)
        return getattr(profile, self.metric)

    def _species_match(self, profile: "ThermoReactionProfile") -> float:
        """Species diversity match: 0–1 similarity between observed
        species entropy and the archetype expectation."""
        if not profile.species_removed:
            return 0.0
        obs_entropy = profile.species_entropy
        expect = ARCHETYPE_EXPECTATIONS[self.archetype]
        return max(0.0, 1.0 - abs(obs_entropy - expect.species_diversity))


# ═══════════════════════════════════════════════════════════════════
# RuleFiring — audit trail record
# ═══════════════════════════════════════════════════════════════════

@dataclass
class RuleFiring:
    """Record of one rule that fired during scoring."""
    rule_name: str
    instrument: str
    archetype: str
    metric: str
    value: float
    score: float
    provenance: str = ""


# ═══════════════════════════════════════════════════════════════════
# Rule helpers — concise condition constructors
# ═══════════════════════════════════════════════════════════════════

def _lt(threshold: float) -> Callable[[float], bool]:
    """value < threshold"""
    return lambda v: v < threshold

def _gt(threshold: float) -> Callable[[float], bool]:
    """value > threshold"""
    return lambda v: v > threshold

def _between(lo: float, hi: float) -> Callable[[float], bool]:
    """lo < value < hi"""
    return lambda v: lo < v < hi

def _and(f: Callable[[float], bool], g: Callable[[float], bool]) -> Callable[[float], bool]:
    """Both conditions must hold on the same value."""
    return lambda v: f(v) and g(v)

def _always(_v: float) -> bool:
    """Always fire — used for species diversity (score is pre-scaled)."""
    return True


# ═══════════════════════════════════════════════════════════════════
# ARCHETYPE_RULES — the complete registry
# ═══════════════════════════════════════════════════════════════════
#
# Every rule below was mechanically extracted from the original
# archetype_vote() method.  The ordering matches the original
# code's instrument→archetype→condition order.
#
# Naming convention:  <instrument>_<archetype>_<signal>
# ═══════════════════════════════════════════════════════════════════

_RULES: List[ArchetypeRule] = []

def _r(instrument: str, archetype: str, name: str, metric: str,
       condition: Callable[[float], bool], score: float,
       provenance: str = "") -> None:
    """Register a rule (builder shorthand)."""
    _RULES.append(ArchetypeRule(
        instrument=instrument, archetype=archetype, name=name,
        metric=metric, condition=condition, score=score,
        provenance=provenance,
    ))


# ── ALGEBRAIC ───────────────────────────────────────────────────

# barrel
_r("algebraic", "barrel", "alg_barrel_scatter_low",
   "scatter_normalised", _lt(1.5), 2.0, "D113 d=2.1 primary barrel voter")
_r("algebraic", "barrel", "alg_barrel_bus_low",
   "mean_bus_mass", _lt(0.5), 1.0, "D109")
_r("algebraic", "barrel", "alg_barrel_beta_low",
   "mean_delta_beta", _lt(0.03), 0.8, "D109")
_r("algebraic", "barrel", "alg_barrel_flat",
   "gap_flatness", _gt(0.95), 0.5, "D109")

# dumbbell
_r("algebraic", "dumbbell", "alg_dumbbell_beta_high",
   "mean_delta_beta", _gt(0.1), 2.0, "D109 d=2.3")
_r("algebraic", "dumbbell", "alg_dumbbell_scatter_high",
   "mean_scatter", _gt(4.0), 1.0, "D109")
_r("algebraic", "dumbbell", "alg_dumbbell_bus_high",
   "mean_bus_mass", _gt(0.85), 0.5, "D109")

# globin
_r("algebraic", "globin", "alg_globin_flat_low",
   "gap_flatness", _lt(0.75), 2.5, "D109 d=3.4")
_r("algebraic", "globin", "alg_globin_flat_mid",
   "gap_flatness", _and(_gt(0.75 - 1e-9), _lt(0.85)), 1.0, "D109 fallback band")
_r("algebraic", "globin", "alg_globin_reversible_low",
   "reversible_frac", _lt(0.3), 0.5, "D109")

# enzyme_active
_r("algebraic", "enzyme_active", "alg_enzyme_ipr_high",
   "mean_ipr", _gt(0.025), 1.5, "D109 d=1.3")
_r("algebraic", "enzyme_active", "alg_enzyme_scatter_mid",
   "mean_scatter", _between(2.0, 5.0), 0.5, "D109")
_r("algebraic", "enzyme_active", "alg_enzyme_reversible_low",
   "reversible_frac", _lt(0.5), 0.4, "D109")

# allosteric
_r("algebraic", "allosteric", "alg_allosteric_reversible_high",
   "reversible_frac", _gt(0.7), 1.5, "D109 d=2.8")
_r("algebraic", "allosteric", "alg_allosteric_scatter_mid",
   "mean_scatter", _gt(2.0), 0.5, "D109")
_r("algebraic", "allosteric", "alg_allosteric_radius_large",
   "mean_spatial_radius", _gt(20.0), 0.5, "D113 allosteric bonus")


# ── MUSICAL ─────────────────────────────────────────────────────

# barrel
_r("musical", "barrel", "mus_barrel_scatter_low",
   "scatter_normalised", _lt(1.5), 1.0, "D113 reduced from 1.5")

# dumbbell
_r("musical", "dumbbell", "mus_dumbbell_scatter_high",
   "mean_scatter", _gt(4.0), 1.5, "D109")
_r("musical", "dumbbell", "mus_dumbbell_beta_high",
   "mean_delta_beta", _gt(0.1), 0.8, "D109")

# enzyme_active
_r("musical", "enzyme_active", "mus_enzyme_ipr_high",
   "mean_ipr", _gt(0.025), 1.5, "D109")
_r("musical", "enzyme_active", "mus_enzyme_scatter_mid",
   "mean_scatter", _between(1.5, 4.0), 0.5, "D109")

# allosteric
_r("musical", "allosteric", "mus_allosteric_scatter",
   "mean_scatter", _gt(2.0), 0.8, "D109")
_r("musical", "allosteric", "mus_allosteric_entropy_vol",
   "entropy_volatility", _gt(0.02), 0.5, "D109")
_r("musical", "allosteric", "mus_allosteric_radius",
   "mean_spatial_radius", _gt(15.0), 0.5, "D113 cross-instrument")

# globin
_r("musical", "globin", "mus_globin_scatter_mid",
   "mean_scatter", _between(1.0, 3.5), 0.5, "D109")


# ── FICK ────────────────────────────────────────────────────────

# barrel
_r("fick", "barrel", "fick_barrel_scatter_low",
   "scatter_normalised", _lt(1.5), 0.8, "D113")
_r("fick", "barrel", "fick_barrel_beta_low",
   "mean_delta_beta", _lt(0.03), 0.8, "D109")

# dumbbell
_r("fick", "dumbbell", "fick_dumbbell_beta_high",
   "mean_delta_beta", _gt(0.1), 1.5, "D109")

# enzyme_active
_r("fick", "enzyme_active", "fick_enzyme_ipr_high",
   "mean_ipr", _gt(0.025), 1.5, "D109")

# globin
_r("fick", "globin", "fick_globin_flat_low",
   "gap_flatness", _lt(0.85), 1.0, "D109")

# allosteric
_r("fick", "allosteric", "fick_allosteric_reversible_high",
   "reversible_frac", _gt(0.7), 0.8, "D109")
_r("fick", "allosteric", "fick_allosteric_radius",
   "mean_spatial_radius", _gt(15.0), 0.5, "D113")


# ── THERMAL ─────────────────────────────────────────────────────

# barrel
_r("thermal", "barrel", "therm_barrel_bus_low",
   "mean_bus_mass", _lt(0.45), 2.0, "D113 primary barrel voter")
_r("thermal", "barrel", "therm_barrel_scatter_low",
   "scatter_normalised", _lt(1.5), 1.5, "D113")
_r("thermal", "barrel", "therm_barrel_beta_low",
   "mean_delta_beta", _lt(0.03), 0.8, "D109")

# dumbbell
_r("thermal", "dumbbell", "therm_dumbbell_beta_high",
   "mean_delta_beta", _gt(0.15), 2.0, "D109 d=2.9")
_r("thermal", "dumbbell", "therm_dumbbell_bus_high",
   "mean_bus_mass", _gt(0.85), 0.8, "D109")
_r("thermal", "dumbbell", "therm_dumbbell_reversible_low",
   "reversible_frac", _lt(0.3), 0.5, "D109")

# globin
_r("thermal", "globin", "therm_globin_flat_low",
   "gap_flatness", _lt(0.75), 2.5, "D109 d=3.4")
_r("thermal", "globin", "therm_globin_flat_mid",
   "gap_flatness", _and(_gt(0.75 - 1e-9), _lt(0.85)), 1.0, "D109 fallback band")

# enzyme_active
_r("thermal", "enzyme_active", "therm_enzyme_ipr_high",
   "mean_ipr", _gt(0.025), 1.5, "D109")
_r("thermal", "enzyme_active", "therm_enzyme_scatter_mid",
   "mean_scatter", _between(1.5, 4.5), 0.5, "D109")

# allosteric
_r("thermal", "allosteric", "therm_allosteric_combo",
   "reversible_frac", _lt(0.5), 0.0, "D109 — guarded by gap_flatness")
# NOTE: The original rule was:
#   if self.reversible_frac < 0.5 and self.gap_flatness > 0.85: score += 0.8
# This requires a compound condition.  We handle it with a dedicated
# multi-metric rule via _read_metric override below.  Instead we use
# a special metric name and a wrapper.


# ── COOPERATIVE ─────────────────────────────────────────────────

# barrel
_r("cooperative", "barrel", "coop_barrel_beta_low",
   "mean_delta_beta", _lt(0.03), 2.0, "D109")
_r("cooperative", "barrel", "coop_barrel_scatter_low",
   "scatter_normalised", _lt(1.0), 1.5, "D113 primary voter")

# dumbbell
_r("cooperative", "dumbbell", "coop_dumbbell_beta_very_high",
   "mean_delta_beta", _gt(0.15), 2.5, "D109")
_r("cooperative", "dumbbell", "coop_dumbbell_beta_high",
   "mean_delta_beta", _and(_gt(0.08), _lt(0.15 + 1e-9)), 1.5, "D109 mid band")

# allosteric
_r("cooperative", "allosteric", "coop_allosteric_beta_mid",
   "mean_delta_beta", _between(0.05, 0.15), 1.0, "D109")
_r("cooperative", "allosteric", "coop_allosteric_radius_low",
   "mean_spatial_radius", _lt(2.0), 0.3, "D109")
_r("cooperative", "allosteric", "coop_allosteric_beta_boost",
   "mean_delta_beta", _gt(0.03), 0.5, "D113 allosteric boost")

# enzyme_active
_r("cooperative", "enzyme_active", "coop_enzyme_ipr_high",
   "mean_ipr", _gt(0.025), 1.5, "D109")
_r("cooperative", "enzyme_active", "coop_enzyme_beta_mid",
   "mean_delta_beta", _between(0.03, 0.1), 0.5, "D109")

# globin
_r("cooperative", "globin", "coop_globin_beta_mid",
   "mean_delta_beta", _between(0.03, 0.1), 0.5, "D109")
_r("cooperative", "globin", "coop_globin_flat_high",
   "gap_flatness", _gt(0.9), 0.3, "D109")


# ── PROPAGATIVE ─────────────────────────────────────────────────

# barrel
_r("propagative", "barrel", "prop_barrel_scatter_very_low",
   "scatter_normalised", _lt(0.5), 1.0, "D113")
_r("propagative", "barrel", "prop_barrel_reversible_very_high",
   "reversible_frac", _gt(0.9), 1.0, "D109")

# dumbbell
_r("propagative", "dumbbell", "prop_dumbbell_beta",
   "mean_delta_beta", _gt(0.05), 1.0, "D109")
_r("propagative", "dumbbell", "prop_dumbbell_radius_high",
   "mean_spatial_radius", _gt(28.0), 1.0, "D109")
_r("propagative", "dumbbell", "prop_dumbbell_scatter",
   "mean_scatter", _gt(2.0), 0.5, "D109")

# allosteric
_r("propagative", "allosteric", "prop_allosteric_radius_very_high",
   "mean_spatial_radius", _gt(30.0), 1.5, "D109 d=2.8")
_r("propagative", "allosteric", "prop_allosteric_reversible_high",
   "reversible_frac", _gt(0.9), 0.5, "D109")
_r("propagative", "allosteric", "prop_allosteric_radius_high",
   "mean_spatial_radius", _gt(20.0), 1.0, "D113 THE allosteric instrument")
_r("propagative", "allosteric", "prop_allosteric_beta",
   "mean_delta_beta", _gt(0.03), 0.5, "D113")

# globin
_r("propagative", "globin", "prop_globin_flat_low",
   "gap_flatness", _lt(0.9), 2.0, "D109")
_r("propagative", "globin", "prop_globin_reversible_low",
   "reversible_frac", _lt(0.5), 0.8, "D109")

# enzyme_active
_r("propagative", "enzyme_active", "prop_enzyme_ipr_high",
   "mean_ipr", _gt(0.025), 1.5, "D109")


# ── FRAGILE ─────────────────────────────────────────────────────

# barrel
_r("fragile", "barrel", "frag_barrel_scatter_low",
   "scatter_normalised", _lt(1.0), 1.0, "D113")
_r("fragile", "barrel", "frag_barrel_free_energy",
   "free_energy_cost", _lt(-0.25), 1.0, "D109")
_r("fragile", "barrel", "frag_barrel_bus_low",
   "mean_bus_mass", _lt(0.6), 0.5, "D109")

# dumbbell
_r("fragile", "dumbbell", "frag_dumbbell_beta_high",
   "mean_delta_beta", _gt(0.15), 2.5, "D109")
_r("fragile", "dumbbell", "frag_dumbbell_reversible_low",
   "reversible_frac", _lt(0.5), 1.5, "D109")

# enzyme_active
_r("fragile", "enzyme_active", "frag_enzyme_ipr_high",
   "mean_ipr", _gt(0.025), 2.0, "D109")
_r("fragile", "enzyme_active", "frag_enzyme_reversible_high",
   "reversible_frac", _gt(0.9), 0.5, "D109")

# allosteric
_r("fragile", "allosteric", "frag_allosteric_entropy_low",
   "entropy_change", _lt(0.35), 0.8, "D109")
_r("fragile", "allosteric", "frag_allosteric_bus_high",
   "mean_bus_mass", _gt(0.9), 0.3, "D109")

# globin
_r("fragile", "globin", "frag_globin_radius_low",
   "mean_spatial_radius", _lt(1.0), 1.0, "D109")
_r("fragile", "globin", "frag_globin_bus_high",
   "mean_bus_mass", _gt(0.9), 0.3, "D109")


# ── UNIVERSAL (all instruments) ─────────────────────────────────

for _arch in ARCHETYPE_EXPECTATIONS:
    _r("*", _arch, f"universal_species_{_arch}",
       f"species_diversity_match:{_arch}", _always, 0.3,
       "Universal species diversity signal")


# ═══════════════════════════════════════════════════════════════════
# Public registry — immutable copy
# ═══════════════════════════════════════════════════════════════════

ARCHETYPE_RULES: Tuple[ArchetypeRule, ...] = tuple(_RULES)
del _RULES  # prevent accidental mutation of the build list


# ═══════════════════════════════════════════════════════════════════
# Compound-condition rules (need access to multiple metrics)
# ═══════════════════════════════════════════════════════════════════
#
# The thermal→allosteric rule in the original code was:
#   if self.reversible_frac < 0.5 and self.gap_flatness > 0.85: score += 0.8
#
# We model this with a CompoundArchetypeRule that overrides evaluate().
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class CompoundArchetypeRule:
    """A rule that inspects multiple profile metrics.

    Same interface as :class:`ArchetypeRule` but with a custom
    ``evaluate()`` that reads multiple attributes.
    """
    instrument: str
    archetype: str
    name: str
    metrics: Tuple[str, ...]
    condition: Callable[..., bool]
    score: float
    provenance: str = ""
    # Keep metric for compatibility (reports use it)
    metric: str = "compound"

    def evaluate(self, profile: "ThermoReactionProfile") -> Optional[float]:
        if self.instrument != "*" and profile.instrument != self.instrument:
            return None
        values = tuple(getattr(profile, m) for m in self.metrics)
        if self.condition(*values):
            return self.score
        return None


# The one compound rule in the current codebase:
_THERMAL_ALLOSTERIC_COMBO = CompoundArchetypeRule(
    instrument="thermal",
    archetype="allosteric",
    name="therm_allosteric_combo",
    metrics=("reversible_frac", "gap_flatness"),
    condition=lambda rev, flat: rev < 0.5 and flat > 0.85,
    score=0.8,
    provenance="D109",
)

# Replace the placeholder we registered earlier
ARCHETYPE_RULES = tuple(
    r for r in ARCHETYPE_RULES if r.name != "therm_allosteric_combo"
) + (_THERMAL_ALLOSTERIC_COMBO,)


# ═══════════════════════════════════════════════════════════════════
# apply_rules — the engine
# ═══════════════════════════════════════════════════════════════════

def apply_rules(
    profile: "ThermoReactionProfile",
    rules: Optional[Sequence[ArchetypeRule]] = None,
) -> Dict[str, float]:
    """Score a profile against the rule registry.

    Parameters
    ----------
    profile : ThermoReactionProfile
        The profile to score.
    rules : sequence of ArchetypeRule, optional
        Override the default rule set (for sweeps / experiments).

    Returns
    -------
    dict
        ``{archetype: score}`` — **normalised** to sum to 1.
    """
    if rules is None:
        rules = ARCHETYPE_RULES

    raw: Dict[str, float] = {a: 0.01 for a in ARCHETYPE_EXPECTATIONS}

    for rule in rules:
        result = rule.evaluate(profile)
        if result is not None:
            # Species diversity rules return a match score 0–1;
            # the rule.score is a multiplier (0.3).
            if rule.metric.startswith("species_diversity_match"):
                value = rule._read_metric(profile) if hasattr(rule, '_read_metric') else 0.0
                raw[rule.archetype] += value * rule.score
            else:
                raw[rule.archetype] += result

    total = sum(raw.values())
    return {k: v / total for k, v in raw.items()}


def apply_rules_traced(
    profile: "ThermoReactionProfile",
    rules: Optional[Sequence[ArchetypeRule]] = None,
) -> Tuple[Dict[str, float], List[RuleFiring]]:
    """Score with full audit trail.

    Returns
    -------
    (votes, firings)
        ``votes`` is the normalised vote dict (same as :func:`apply_rules`).
        ``firings`` is a list of :class:`RuleFiring` records for every
        rule that contributed score.
    """
    if rules is None:
        rules = ARCHETYPE_RULES

    raw: Dict[str, float] = {a: 0.01 for a in ARCHETYPE_EXPECTATIONS}
    firings: List[RuleFiring] = []

    for rule in rules:
        result = rule.evaluate(profile)
        if result is not None:
            if rule.metric.startswith("species_diversity_match"):
                if hasattr(rule, '_read_metric'):
                    value = rule._read_metric(profile)
                else:
                    value = 0.0
                contribution = value * rule.score
            else:
                contribution = result
                value = 0.0
                # Read the actual metric value for the trace
                if hasattr(rule, '_read_metric'):
                    try:
                        value = rule._read_metric(profile)
                    except (AttributeError, TypeError):
                        pass
                elif hasattr(rule, 'metrics'):
                    # CompoundArchetypeRule
                    value = result

            raw[rule.archetype] += contribution
            firings.append(RuleFiring(
                rule_name=rule.name,
                instrument=rule.instrument,
                archetype=rule.archetype,
                metric=rule.metric,
                value=float(value),
                score=float(contribution),
                provenance=rule.provenance,
            ))

    total = sum(raw.values())
    votes = {k: v / total for k, v in raw.items()}
    return votes, firings


# ═══════════════════════════════════════════════════════════════════
# Utilities for programmatic sweeps
# ═══════════════════════════════════════════════════════════════════

def get_rules(
    instrument: Optional[str] = None,
    archetype: Optional[str] = None,
    name_contains: Optional[str] = None,
    rules: Optional[Sequence[ArchetypeRule]] = None,
) -> List[ArchetypeRule]:
    """Filter the registry by instrument, archetype, or name substring.

    Useful for inspection and sweep setup.
    """
    source = rules if rules is not None else ARCHETYPE_RULES
    result = list(source)
    if instrument is not None:
        result = [r for r in result if r.instrument == instrument]
    if archetype is not None:
        result = [r for r in result if r.archetype == archetype]
    if name_contains is not None:
        result = [r for r in result if name_contains in r.name]
    return result


def replace_rules(
    original: Sequence[ArchetypeRule],
    replacements: Dict[str, ArchetypeRule],
) -> Tuple[ArchetypeRule, ...]:
    """Return a new rule set with named rules replaced.

    Parameters
    ----------
    original : sequence of ArchetypeRule
        The base rule set (typically ``ARCHETYPE_RULES``).
    replacements : dict
        ``{rule_name: new_rule}`` — rules whose name matches a key
        are replaced by the corresponding value.

    Returns
    -------
    tuple of ArchetypeRule
        A new immutable rule set.
    """
    return tuple(
        replacements.get(r.name, r) for r in original
    )
