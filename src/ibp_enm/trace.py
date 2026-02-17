"""ClassificationTrace — full audit trail for one protein.

Captures every intermediate result produced during classification:
per-instrument rule firings, MetaFick consensus state, context-boost
signals, lens traces, final scores, and timing — in a single frozen
dataclass suitable for debugging, serialisation, and post-hoc analysis.

Usage
-----
>>> from ibp_enm import LensStackSynthesizer
>>> synth = LensStackSynthesizer(evals=evals, evecs=evecs,
...                               domain_labels=dl, contacts=ct)
>>> result = synth.synthesize_identity(profiles, meta_state)
>>> trace = result["trace"]           # ClassificationTrace
>>> trace.identity                    # "enzyme_active"
>>> trace.rule_firings["algebraic"]   # [RuleFiring(…), …]
>>> trace.context_signals["all_ipr"]  # 0.027
>>> trace.to_dict()                   # JSON-safe dict

Historical notes
----------------
v0.7.0 — Step 6 of the architectural plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .lens_stack import LensTrace
from .rules import RuleFiring
from .thresholds import ThresholdRegistry, DEFAULT_THRESHOLDS

__all__ = [
    "ClassificationTrace",
    "ContextSignals",
]


# ═══════════════════════════════════════════════════════════════════
# ContextSignals — cross-instrument summary values
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ContextSignals:
    """Cross-instrument summary signals used by the context-boost.

    These are the intermediate values computed inside
    :meth:`MetaFickBalancer.synthesize_identity` that were previously
    discarded as local variables.  Now captured for audit.
    """

    all_scatter: float = 0.0
    all_db: float = 0.0
    all_ipr: float = 0.0
    all_mass: float = 0.0
    all_scatter_norm: float = 0.0
    all_radius: float = 0.0
    n_residues: int = 200
    propagative_radius: float = 0.0
    propagative_scatter_norm: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Return a JSON-safe dict of all signal values."""
        return {
            "all_scatter": round(self.all_scatter, 6),
            "all_db": round(self.all_db, 6),
            "all_ipr": round(self.all_ipr, 6),
            "all_mass": round(self.all_mass, 6),
            "all_scatter_norm": round(self.all_scatter_norm, 6),
            "all_radius": round(self.all_radius, 6),
            "n_residues": self.n_residues,
            "propagative_radius": round(self.propagative_radius, 6),
            "propagative_scatter_norm": round(
                self.propagative_scatter_norm, 6),
        }


# ═══════════════════════════════════════════════════════════════════
# ClassificationTrace — the full audit trail
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ClassificationTrace:
    """Complete audit trail for one protein classification.

    Every intermediate result from the pipeline is captured here:

    1. **identity** — the winning archetype
    2. **scores** — final normalised ``{archetype: score}``
    3. **per_instrument_votes** — ``{instrument: {archetype: score}}``
    4. **rule_firings** — ``{instrument: [RuleFiring, …]}`` — which
       rules fired for each instrument and how much score each contributed
    5. **consensus_scores** — pre-lens consensus scores
    6. **disagreement_scores** — pre-lens disagreement-weighted scores
    7. **context_boost** — per-archetype context boost values
    8. **context_signals** — :class:`ContextSignals` — the cross-instrument
       summary values that drove the context boost
    9. **alpha_meta** — the MetaFick α (consensus vs disagreement weight)
    10. **meta_state** — full MetaFick state dict
    11. **lens_traces** — ``[LensTrace, …]`` — per-lens activation records
    12. **thresholds_name** — which :class:`ThresholdRegistry` was used
    13. **n_residues** — protein size
    14. **n_instruments** — number of instruments (typically 7)

    The trace is frozen (immutable) to prevent accidental mutation.

    Serialisation
    -------------
    Call :meth:`to_dict` for a JSON-safe representation.  Call
    :meth:`summary` for a human-readable one-line description.
    """

    # Final result
    identity: str
    scores: Dict[str, float]

    # Per-instrument detail
    per_instrument_votes: Dict[str, Dict[str, float]]
    rule_firings: Dict[str, List[RuleFiring]]

    # MetaFick synthesis
    consensus_scores: Dict[str, float]
    disagreement_scores: Dict[str, float]
    context_boost: Dict[str, float]
    context_signals: ContextSignals
    alpha_meta: float
    meta_state: Dict[str, Any]

    # Lens stack
    lens_traces: List[LensTrace]

    # Config
    thresholds_name: str = "production"
    n_residues: int = 0
    n_instruments: int = 7

    # ── Derived properties ──────────────────────────────────────

    @property
    def activated_lenses(self) -> List[str]:
        """Names of lenses that actually fired."""
        return [t.lens_name for t in self.lens_traces if t.activated]

    @property
    def total_lens_boost(self) -> float:
        """Sum of all lens boosts (may be negative for penalties)."""
        return sum(t.boost for t in self.lens_traces)

    @property
    def top_rules(self) -> List[RuleFiring]:
        """All rule firings across all instruments, sorted by |score|."""
        all_firings = []
        for firings in self.rule_firings.values():
            all_firings.extend(firings)
        return sorted(all_firings, key=lambda r: abs(r.score), reverse=True)

    @property
    def n_rules_fired(self) -> int:
        """Total number of rules that fired across all instruments."""
        return sum(len(f) for f in self.rule_firings.values())

    @property
    def score_margin(self) -> float:
        """Gap between top-1 and top-2 scores."""
        sorted_s = sorted(self.scores.values(), reverse=True)
        if len(sorted_s) >= 2:
            return sorted_s[0] - sorted_s[1]
        return sorted_s[0] if sorted_s else 0.0

    @property
    def runner_up(self) -> str:
        """Second-place archetype."""
        sorted_items = sorted(
            self.scores.items(), key=lambda x: -x[1])
        return sorted_items[1][0] if len(sorted_items) >= 2 else ""

    # ── Serialisation ───────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe dict of the full trace."""
        return {
            "identity": self.identity,
            "scores": {k: round(v, 6) for k, v in self.scores.items()},
            "per_instrument_votes": {
                inst: {k: round(v, 6) for k, v in votes.items()}
                for inst, votes in self.per_instrument_votes.items()
            },
            "rule_firings": {
                inst: [
                    {
                        "rule_name": r.rule_name,
                        "instrument": r.instrument,
                        "archetype": r.archetype,
                        "metric": r.metric,
                        "value": round(r.value, 6),
                        "score": round(r.score, 6),
                        "provenance": r.provenance,
                    }
                    for r in firings
                ]
                for inst, firings in self.rule_firings.items()
            },
            "consensus_scores": {
                k: round(v, 6) for k, v in self.consensus_scores.items()
            },
            "disagreement_scores": {
                k: round(v, 6)
                for k, v in self.disagreement_scores.items()
            },
            "context_boost": {
                k: round(v, 6) for k, v in self.context_boost.items()
            },
            "context_signals": self.context_signals.to_dict(),
            "alpha_meta": round(self.alpha_meta, 6),
            "lens_traces": [
                {
                    "lens_name": t.lens_name,
                    "activated": t.activated,
                    "boost": round(t.boost, 6),
                    "details": t.details,
                }
                for t in self.lens_traces
            ],
            "thresholds_name": self.thresholds_name,
            "n_residues": self.n_residues,
            "n_instruments": self.n_instruments,
            "n_rules_fired": self.n_rules_fired,
            "score_margin": round(self.score_margin, 6),
            "activated_lenses": self.activated_lenses,
        }

    def summary(self) -> str:
        """One-line human-readable summary."""
        lenses = ", ".join(self.activated_lenses) or "none"
        return (
            f"{self.identity} "
            f"(margin={self.score_margin:.3f}, "
            f"rules={self.n_rules_fired}, "
            f"lenses=[{lenses}], "
            f"α={self.alpha_meta:.3f})"
        )

    def explain(self, top_n: int = 10) -> str:
        """Multi-line explanation of why this identity was chosen.

        Shows the top rules, context-boost breakdown, and lens effects.
        """
        lines = [
            f"Classification: {self.identity}",
            f"  Score margin:  {self.score_margin:.4f} "
            f"(runner-up: {self.runner_up})",
            f"  MetaFick α:    {self.alpha_meta:.4f}",
            f"  Rules fired:   {self.n_rules_fired} "
            f"across {self.n_instruments} instruments",
            "",
            "Scores:",
        ]
        for arch, score in sorted(
            self.scores.items(), key=lambda x: -x[1]
        ):
            marker = " ←" if arch == self.identity else ""
            lines.append(f"  {arch:<16s} {score:.4f}{marker}")

        # Context boost
        nonzero_ctx = {
            k: v for k, v in self.context_boost.items() if abs(v) > 0.001
        }
        if nonzero_ctx:
            lines.append("")
            lines.append("Context boost:")
            for arch, boost in sorted(
                nonzero_ctx.items(), key=lambda x: -abs(x[1])
            ):
                lines.append(f"  {arch:<16s} {boost:+.4f}")

        # Lens effects
        active = [t for t in self.lens_traces if t.activated]
        if active:
            lines.append("")
            lines.append("Lens effects:")
            for t in active:
                lines.append(
                    f"  {t.lens_name:<20s} boost={t.boost:+.4f}")

        # Top rules
        top = self.top_rules[:top_n]
        if top:
            lines.append("")
            lines.append(f"Top {min(top_n, len(top))} rules:")
            for r in top:
                lines.append(
                    f"  {r.rule_name:<35s} {r.archetype:<16s} "
                    f"{r.score:+.4f}  "
                    f"({r.metric}={r.value:.4f})")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"ClassificationTrace({self.identity!r}, "
            f"margin={self.score_margin:.3f}, "
            f"rules={self.n_rules_fired})"
        )
