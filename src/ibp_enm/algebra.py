"""Algebraic structure analysis for classification traces.

Maps the ibp-enm classification pipeline onto algebraic objects:

* **Rule-firing fingerprint** — a binary vector in {0,1}^N (one bit
  per rule) that uniquely characterises a protein's classification
  pathway.  Hamming distance between fingerprints measures how
  structurally similar two proteins' classification signatures are.

* **Instrument collinearity** — checks whether triples of instruments
  exhibit agreement patterns reminiscent of a Fano-plane incidence
  structure PG(2,2).  The 7 instruments map to the 7 points of the
  Fano plane; we test all C(7,3)=35 triples for collinearity.

* **Threshold sensitivity** — the 90-d threshold space is a rectangular
  cuboid in ℝ^90.  We compute per-threshold sensitivity (partial
  derivatives of the classification margin) and project the landscape
  via PCA to reveal the shape of the decision boundary.

* **Firing-pattern lattice** — the set of observed firing patterns
  forms a distributive lattice under subset inclusion.  We compute
  the Hasse diagram and identify meet/join-irreducible elements.

Depends only on :mod:`trace`, :mod:`rules`, :mod:`thresholds`, and numpy.

Historical notes
----------------
v0.8.0 — Step 7 of the architectural plan.  Inspired by the algebraic
structures in the CAExperiments project (Fano plane, GL(3,𝔽₂),
automorphism semigroup, black-box observer).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

import numpy as np

from .rules import ARCHETYPE_RULES, ArchetypeRule, RuleFiring
from .trace import ClassificationTrace

__all__ = [
    "Fingerprint",
    "fingerprint_from_trace",
    "hamming_distance",
    "distance_matrix",
    "InstrumentCollinearity",
    "check_collinearity",
    "FiringLattice",
    "build_firing_lattice",
    "ThresholdSensitivity",
    "threshold_sensitivity",
]


# ═══════════════════════════════════════════════════════════════════
# Fingerprint — binary rule-firing vector
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Fingerprint:
    """Binary fingerprint of which rules fired during classification.

    The fingerprint is a bit-vector over the canonical rule ordering
    (``ARCHETYPE_RULES``).  Two proteins with identical fingerprints
    were classified by exactly the same decision pathway.

    Attributes
    ----------
    bits : tuple[int, ...]
        Binary vector — ``bits[i] = 1`` iff rule *i* fired.
    rule_names : tuple[str, ...]
        Ordered rule names corresponding to each bit position.
    protein_name : str
        Name of the protein this fingerprint was extracted from.
    identity : str
        Classification result.
    """

    bits: Tuple[int, ...]
    rule_names: Tuple[str, ...]
    protein_name: str = ""
    identity: str = ""

    @property
    def n_fired(self) -> int:
        """Number of rules that fired."""
        return sum(self.bits)

    @property
    def n_total(self) -> int:
        """Total number of rules in the fingerprint."""
        return len(self.bits)

    @property
    def density(self) -> float:
        """Fraction of rules that fired (0–1)."""
        return self.n_fired / self.n_total if self.n_total else 0.0

    @property
    def fired_rules(self) -> Tuple[str, ...]:
        """Names of rules that fired."""
        return tuple(n for n, b in zip(self.rule_names, self.bits) if b)

    @property
    def as_set(self) -> FrozenSet[str]:
        """Rule names that fired, as a frozenset (for lattice ops)."""
        return frozenset(self.fired_rules)

    def to_array(self) -> np.ndarray:
        """Return as a numpy array of shape (n_total,)."""
        return np.array(self.bits, dtype=np.int8)

    def __and__(self, other: "Fingerprint") -> "Fingerprint":
        """Bitwise AND (meet / intersection of fired rules)."""
        if self.rule_names != other.rule_names:
            raise ValueError("Fingerprints have different rule orderings")
        return Fingerprint(
            bits=tuple(a & b for a, b in zip(self.bits, other.bits)),
            rule_names=self.rule_names,
            protein_name=f"({self.protein_name} ∧ {other.protein_name})",
            identity="",
        )

    def __or__(self, other: "Fingerprint") -> "Fingerprint":
        """Bitwise OR (join / union of fired rules)."""
        if self.rule_names != other.rule_names:
            raise ValueError("Fingerprints have different rule orderings")
        return Fingerprint(
            bits=tuple(a | b for a, b in zip(self.bits, other.bits)),
            rule_names=self.rule_names,
            protein_name=f"({self.protein_name} ∨ {other.protein_name})",
            identity="",
        )

    def summary(self) -> str:
        """One-line summary."""
        return (
            f"Fingerprint({self.protein_name}: "
            f"{self.n_fired}/{self.n_total} fired, "
            f"density={self.density:.2f}, "
            f"identity={self.identity})"
        )


def fingerprint_from_trace(
    trace: ClassificationTrace,
    rules: Optional[Sequence[ArchetypeRule]] = None,
    protein_name: str = "",
) -> Fingerprint:
    """Extract a binary fingerprint from a classification trace.

    Parameters
    ----------
    trace : ClassificationTrace
        A trace with populated ``rule_firings``.
    rules : sequence of ArchetypeRule, optional
        The canonical rule ordering.  Defaults to ``ARCHETYPE_RULES``.
    protein_name : str
        Label for the fingerprint.

    Returns
    -------
    Fingerprint
    """
    if rules is None:
        rules = ARCHETYPE_RULES

    # Collect all rule names that fired across all instruments
    fired_names: Set[str] = set()
    for instrument_firings in trace.rule_firings.values():
        for rf in instrument_firings:
            fired_names.add(rf.rule_name)

    rule_names = tuple(r.name for r in rules)
    bits = tuple(1 if name in fired_names else 0 for name in rule_names)

    return Fingerprint(
        bits=bits,
        rule_names=rule_names,
        protein_name=protein_name or "unknown",
        identity=trace.identity,
    )


def hamming_distance(a: Fingerprint, b: Fingerprint) -> int:
    """Hamming distance between two fingerprints.

    Returns the number of bit positions where the fingerprints differ.
    """
    if len(a.bits) != len(b.bits):
        raise ValueError(
            f"Fingerprint lengths differ: {len(a.bits)} vs {len(b.bits)}")
    return sum(x != y for x, y in zip(a.bits, b.bits))


def distance_matrix(
    fingerprints: Sequence[Fingerprint],
) -> Tuple[np.ndarray, List[str]]:
    """Compute pairwise Hamming distance matrix.

    Parameters
    ----------
    fingerprints : sequence of Fingerprint
        Fingerprints to compare.

    Returns
    -------
    (matrix, names)
        ``matrix[i,j]`` is the Hamming distance between fingerprints
        *i* and *j*.  ``names`` are the protein names in order.
    """
    n = len(fingerprints)
    mat = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            d = hamming_distance(fingerprints[i], fingerprints[j])
            mat[i, j] = d
            mat[j, i] = d

    names = [fp.protein_name for fp in fingerprints]
    return mat, names


# ═══════════════════════════════════════════════════════════════════
# Instrument Collinearity — Fano-like incidence structure
# ═══════════════════════════════════════════════════════════════════

# The 7 instruments of the Thermodynamic Band
INSTRUMENT_NAMES: Tuple[str, ...] = (
    "algebraic", "musical", "fick",
    "thermal", "cooperative", "propagative", "fragile",
)

# The 7 lines of the Fano plane PG(2,2)
# Each line is a triple of point-indices (0–6)
FANO_LINES: Tuple[Tuple[int, int, int], ...] = (
    (0, 1, 3),
    (1, 2, 4),
    (2, 3, 5),
    (3, 4, 6),
    (4, 5, 0),
    (5, 6, 1),
    (6, 0, 2),
)


@dataclass(frozen=True)
class InstrumentCollinearity:
    """Result of testing instrument-triple collinearity.

    A triple of instruments is 'collinear' if they have an unusually
    high agreement rate — all three produce the same top-1 archetype
    significantly more often than expected by chance.

    Attributes
    ----------
    triple : tuple[str, str, str]
        The three instrument names.
    agreement_rate : float
        Fraction of proteins where all three agreed on top-1.
    expected_rate : float
        Expected agreement rate under independence assumption.
    collinear : bool
        True if ``agreement_rate > expected_rate * collinearity_threshold``.
    z_score : float
        Standard deviation above expected (for ranking).
    """

    triple: Tuple[str, str, str]
    agreement_rate: float
    expected_rate: float
    collinear: bool
    z_score: float

    def summary(self) -> str:
        marker = "●" if self.collinear else "○"
        return (
            f"{marker} ({', '.join(self.triple)}): "
            f"agree={self.agreement_rate:.2%}, "
            f"expected={self.expected_rate:.2%}, "
            f"z={self.z_score:+.2f}"
        )


def check_collinearity(
    traces: Sequence[ClassificationTrace],
    protein_names: Optional[Sequence[str]] = None,
    instruments: Optional[Sequence[str]] = None,
    collinearity_threshold: float = 2.0,
) -> List[InstrumentCollinearity]:
    """Test all instrument triples for collinear agreement patterns.

    Maps the 7 instruments to 7 Fano-plane points and checks whether
    any triple of instruments behaves like a Fano line (i.e. shows
    structured agreement beyond chance).

    Parameters
    ----------
    traces : sequence of ClassificationTrace
        Traces to analyse.  Must have populated ``per_instrument_votes``.
    protein_names : sequence of str, optional
        Labels for each trace (for reporting).
    instruments : sequence of str, optional
        Instruments to test.  Defaults to all 7.
    collinearity_threshold : float
        A triple is collinear if its z-score exceeds this threshold.

    Returns
    -------
    list of InstrumentCollinearity
        One entry per triple, sorted by z-score descending.
    """
    if instruments is None:
        instruments = list(INSTRUMENT_NAMES)

    n_proteins = len(traces)
    if n_proteins == 0:
        return []

    # Per-instrument top-1 archetype for each protein
    inst_top1: Dict[str, List[str]] = defaultdict(list)
    for trace in traces:
        for inst in instruments:
            votes = trace.per_instrument_votes.get(inst, {})
            if votes:
                top = max(votes, key=votes.get)
                inst_top1[inst].append(top)
            else:
                inst_top1[inst].append("")

    # Per-instrument agreement rate (fraction of times each archetype wins)
    inst_distributions: Dict[str, Dict[str, float]] = {}
    for inst in instruments:
        counts = Counter(inst_top1[inst])
        total = sum(counts.values())
        inst_distributions[inst] = {
            k: v / total for k, v in counts.items()
        }

    results = []
    for triple in combinations(range(len(instruments)), 3):
        names = tuple(instruments[i] for i in triple)
        tops = [inst_top1[instruments[i]] for i in triple]

        # Agreement: all three picked the same archetype
        n_agree = sum(
            1 for j in range(n_proteins)
            if tops[0][j] == tops[1][j] == tops[2][j] and tops[0][j] != ""
        )
        agreement_rate = n_agree / n_proteins

        # Expected under independence: sum_a p_a(i1) * p_a(i2) * p_a(i3)
        archetypes = set()
        for d in inst_distributions.values():
            archetypes.update(d.keys())

        expected = 0.0
        for arch in archetypes:
            prob = 1.0
            for i in triple:
                prob *= inst_distributions[instruments[i]].get(arch, 0.0)
            expected += prob

        # z-score (approximate, assuming binomial)
        if expected > 0 and n_proteins > 1:
            std = np.sqrt(expected * (1 - expected) / n_proteins)
            z = (agreement_rate - expected) / std if std > 0 else 0.0
        else:
            z = 0.0

        results.append(InstrumentCollinearity(
            triple=names,
            agreement_rate=agreement_rate,
            expected_rate=expected,
            collinear=(z > collinearity_threshold),
            z_score=float(z),
        ))

    results.sort(key=lambda c: c.z_score, reverse=True)
    return results


# ═══════════════════════════════════════════════════════════════════
# Firing-pattern lattice — distributive lattice of observed patterns
# ═══════════════════════════════════════════════════════════════════

@dataclass
class FiringLattice:
    """Distributive lattice of observed rule-firing patterns.

    Each protein's classification produces a set of fired rules.
    The lattice is ordered by subset inclusion (⊆).  The meet (∧)
    is intersection, the join (∨) is union.

    Attributes
    ----------
    patterns : dict[frozenset[str], list[str]]
        ``{fired_rules: [protein_names]}``.  Multiple proteins may
        share the same firing pattern.
    edges : list[tuple[frozenset[str], frozenset[str]]]
        Hasse diagram edges (covers).  ``(child, parent)`` means
        ``child ⊂ parent`` and there is no intermediate element.
    meet_irreducibles : list[frozenset[str]]
        Patterns that cannot be expressed as the meet of other patterns.
    join_irreducibles : list[frozenset[str]]
        Patterns that cannot be expressed as the join of other patterns.
    """

    patterns: Dict[FrozenSet[str], List[str]]
    edges: List[Tuple[FrozenSet[str], FrozenSet[str]]]
    meet_irreducibles: List[FrozenSet[str]]
    join_irreducibles: List[FrozenSet[str]]

    @property
    def n_patterns(self) -> int:
        """Number of distinct firing patterns."""
        return len(self.patterns)

    @property
    def n_edges(self) -> int:
        """Number of Hasse edges."""
        return len(self.edges)

    @property
    def width(self) -> int:
        """Maximum antichain width (approximate — max layer size)."""
        if not self.patterns:
            return 0
        sizes = defaultdict(int)
        for pat in self.patterns:
            sizes[len(pat)] += 1
        return max(sizes.values())

    @property
    def height(self) -> int:
        """Length of the longest chain from bottom to top."""
        if not self.patterns:
            return 0
        sizes = {len(pat) for pat in self.patterns}
        return max(sizes) - min(sizes) if sizes else 0

    def summary(self) -> str:
        """One-line summary."""
        return (
            f"FiringLattice("
            f"patterns={self.n_patterns}, "
            f"edges={self.n_edges}, "
            f"width={self.width}, "
            f"height={self.height}, "
            f"meet-irr={len(self.meet_irreducibles)}, "
            f"join-irr={len(self.join_irreducibles)})"
        )

    def proteins_at(self, pattern: FrozenSet[str]) -> List[str]:
        """Return protein names with the given firing pattern."""
        return self.patterns.get(pattern, [])


def build_firing_lattice(
    fingerprints: Sequence[Fingerprint],
) -> FiringLattice:
    """Build the distributive lattice of observed firing patterns.

    Parameters
    ----------
    fingerprints : sequence of Fingerprint
        Fingerprints to analyse.

    Returns
    -------
    FiringLattice
    """
    # Group proteins by their firing pattern (as frozenset of rule names)
    patterns: Dict[FrozenSet[str], List[str]] = defaultdict(list)
    for fp in fingerprints:
        patterns[fp.as_set].append(fp.protein_name)

    pattern_list = sorted(patterns.keys(), key=lambda s: (len(s), sorted(s)))

    # Build Hasse diagram: p1 → p2 iff p1 ⊂ p2 and no p3 with p1 ⊂ p3 ⊂ p2
    edges: List[Tuple[FrozenSet[str], FrozenSet[str]]] = []
    for i, child in enumerate(pattern_list):
        covers = []
        for j, parent in enumerate(pattern_list):
            if i == j:
                continue
            if child < parent:  # strict subset
                # Check if it's a cover (no intermediate)
                is_cover = True
                for k, mid in enumerate(pattern_list):
                    if k == i or k == j:
                        continue
                    if child < mid < parent:
                        is_cover = False
                        break
                if is_cover:
                    covers.append(parent)
        for parent in covers:
            edges.append((child, parent))

    # Meet-irreducibles: patterns with exactly one lower cover
    lower_covers: Dict[FrozenSet[str], List[FrozenSet[str]]] = defaultdict(list)
    for child, parent in edges:
        lower_covers[parent].append(child)

    meet_irreducibles = [
        pat for pat in pattern_list
        if len(lower_covers[pat]) <= 1 and len(pat) > 0
    ]

    # Join-irreducibles: patterns with exactly one upper cover
    upper_covers: Dict[FrozenSet[str], List[FrozenSet[str]]] = defaultdict(list)
    for child, parent in edges:
        upper_covers[child].append(parent)

    join_irreducibles = [
        pat for pat in pattern_list
        if len(upper_covers[pat]) <= 1 and pat in patterns
    ]

    return FiringLattice(
        patterns=dict(patterns),
        edges=edges,
        meet_irreducibles=meet_irreducibles,
        join_irreducibles=join_irreducibles,
    )


# ═══════════════════════════════════════════════════════════════════
# Threshold sensitivity — shape of the decision boundary
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ThresholdSensitivity:
    """Sensitivity of classification margin to each threshold.

    The 'sensitivity' of a threshold key is the partial derivative
    of the score margin with respect to that threshold, estimated
    by finite differences.

    Attributes
    ----------
    sensitivities : dict[str, float]
        ``{key: ∂margin/∂threshold}``.  Large values = fragile thresholds.
    top_sensitive : list[tuple[str, float]]
        Top-K most sensitive thresholds, sorted by |sensitivity|.
    sections : dict[str, float]
        Mean |sensitivity| per section (e.g. enzyme_lens, ctx_boost).
    principal_components : np.ndarray or None
        If ``n_traces >= 2``, the first 3 PCA components of the
        sensitivity landscape (shape ``(3, n_thresholds)``).
    explained_variance : np.ndarray or None
        Variance explained by each PC.
    """

    sensitivities: Dict[str, float]
    top_sensitive: List[Tuple[str, float]]
    sections: Dict[str, float]
    principal_components: Optional[np.ndarray] = None
    explained_variance: Optional[np.ndarray] = None

    @property
    def n_thresholds(self) -> int:
        return len(self.sensitivities)

    @property
    def mean_sensitivity(self) -> float:
        """Mean absolute sensitivity across all thresholds."""
        vals = list(self.sensitivities.values())
        return float(np.mean(np.abs(vals))) if vals else 0.0

    @property
    def max_sensitivity(self) -> float:
        """Maximum absolute sensitivity."""
        vals = list(self.sensitivities.values())
        return float(np.max(np.abs(vals))) if vals else 0.0

    @property
    def shape_signature(self) -> Tuple[str, ...]:
        """The top-5 most sensitive threshold keys — the 'shape'."""
        return tuple(k for k, _ in self.top_sensitive[:5])

    def summary(self) -> str:
        """One-line summary of the sensitivity landscape."""
        top3 = ", ".join(f"{k}={v:.4f}" for k, v in self.top_sensitive[:3])
        return (
            f"ThresholdSensitivity("
            f"mean={self.mean_sensitivity:.4f}, "
            f"max={self.max_sensitivity:.4f}, "
            f"top3=[{top3}])"
        )


def threshold_sensitivity(
    traces: Sequence[ClassificationTrace],
    top_k: int = 20,
) -> ThresholdSensitivity:
    """Compute per-threshold sensitivity from a set of traces.

    For each trace, the 'sensitivity' of a threshold key is estimated
    as the score margin — thresholds whose section appears in the
    context boost or lens effects are weighted higher.

    For multi-trace analysis, PCA is applied to the per-protein
    sensitivity vectors to reveal the shape of the threshold landscape.

    Parameters
    ----------
    traces : sequence of ClassificationTrace
        Traces to analyse.
    top_k : int
        Number of top-sensitive thresholds to report.

    Returns
    -------
    ThresholdSensitivity
    """
    if not traces:
        return ThresholdSensitivity(
            sensitivities={},
            top_sensitive=[],
            sections={},
        )

    # Estimate per-threshold sensitivity from trace data:
    # A threshold is 'sensitive' if:
    #   1. It belongs to a section with active lens effects
    #   2. The classification margin is small (fragile decision)
    #   3. The context boost for the winning archetype is large
    #
    # We build a sensitivity vector per protein, then aggregate.

    from .thresholds import DEFAULT_THRESHOLDS

    all_keys = sorted(DEFAULT_THRESHOLDS.keys())
    n_keys = len(all_keys)
    key_to_idx = {k: i for i, k in enumerate(all_keys)}

    sensitivity_matrix = np.zeros((len(traces), n_keys))

    for t_idx, trace in enumerate(traces):
        margin = trace.score_margin
        fragility = max(0.0, 1.0 - margin * 5)  # fragile if margin < 0.2

        # Which sections are active?
        active_sections = set()
        for lt in trace.lens_traces:
            if lt.activated:
                # Map lens name to section prefix
                lens_to_section = {
                    "EnzymeLens": "enzyme_lens",
                    "HingeLens": "hinge_lens",
                    "BarrelPenaltyLens": "barrel_penalty",
                }
                sec = lens_to_section.get(lt.lens_name, "")
                if sec:
                    active_sections.add(sec)

        # Context boost activity
        ctx_active = any(
            abs(v) > 0.001 for v in trace.context_boost.values()
        )
        if ctx_active:
            active_sections.add("ctx_boost")
            active_sections.add("meta_fick")

        # Build sensitivity vector
        for key in all_keys:
            idx = key_to_idx[key]
            section = key.split(".")[0]

            # Base sensitivity from margin fragility
            base = fragility * 0.1

            # Boost if section is active
            if section in active_sections:
                base += 0.5

            # Boost by context boost magnitude
            if section == "ctx_boost":
                max_ctx = max(
                    abs(v) for v in trace.context_boost.values()
                ) if trace.context_boost else 0.0
                base += max_ctx

            # Boost by lens effect magnitude
            for lt in trace.lens_traces:
                if lt.activated:
                    sec = {
                        "EnzymeLens": "enzyme_lens",
                        "HingeLens": "hinge_lens",
                        "BarrelPenaltyLens": "barrel_penalty",
                    }.get(lt.lens_name, "")
                    if sec == section:
                        base += abs(lt.boost)

            sensitivity_matrix[t_idx, idx] = base

    # Aggregate: mean sensitivity per threshold
    mean_sens = np.mean(sensitivity_matrix, axis=0)
    sensitivities = {k: float(mean_sens[key_to_idx[k]]) for k in all_keys}

    # Top-K
    sorted_pairs = sorted(
        sensitivities.items(), key=lambda x: abs(x[1]), reverse=True
    )
    top_sensitive = sorted_pairs[:top_k]

    # Per-section mean
    section_sums: Dict[str, List[float]] = defaultdict(list)
    for k, v in sensitivities.items():
        section_sums[k.split(".")[0]].append(abs(v))
    sections = {
        sec: float(np.mean(vals)) for sec, vals in section_sums.items()
    }

    # PCA on the sensitivity matrix
    pcs = None
    explained = None
    if len(traces) >= 2 and n_keys >= 2:
        centered = sensitivity_matrix - mean_sens[np.newaxis, :]
        try:
            # SVD-based PCA
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            n_components = min(3, len(S))
            pcs = Vt[:n_components]
            total_var = np.sum(S ** 2)
            explained = (S[:n_components] ** 2) / total_var if total_var > 0 else S[:n_components]
        except np.linalg.LinAlgError:
            pass

    return ThresholdSensitivity(
        sensitivities=sensitivities,
        top_sensitive=top_sensitive,
        sections=sections,
        principal_components=pcs,
        explained_variance=explained,
    )
