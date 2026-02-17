"""Deep result analysis — cross-protein pattern comparison.

Takes a set of :class:`ClassificationTrace` objects (typically from a
:class:`BenchmarkReport`) and produces:

* **Archetype profile** — per-archetype statistical summaries (mean scores,
  rule-firing density, lens activation frequency, context-boost magnitude).

* **Confusion clusters** — groups of proteins that are frequently confused
  with each other, with shared characteristics identified.

* **Threshold cascade fingerprint** — the 'shape' of the decision boundary
  for each protein, showing which threshold sections dominated.

* **Cross-experiment comparison** — given two :class:`AnalysisReport` objects,
  identify what changed between them (new patterns, disappeared patterns,
  threshold sensitivity shifts).

* **Rule co-firing matrix** — which rules tend to fire together.

Usage
-----
>>> from ibp_enm.analysis import analyse_traces, compare_reports
>>> report = analyse_traces(traces, protein_names)
>>> print(report.summary())
>>>
>>> report2 = analyse_traces(traces2, protein_names)
>>> delta = compare_reports(report, report2)

Historical notes
----------------
v0.8.0 — Step 7 of the architectural plan.  The user's three interests:
1. Does the threshold cascade make a fingerprint or shape?
2. A formal result deep analysis tool comparing experiment results.
3. Algebraic structure analysis from CAExperiments.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

import numpy as np

from .algebra import (
    Fingerprint,
    FiringLattice,
    InstrumentCollinearity,
    ThresholdSensitivity,
    build_firing_lattice,
    distance_matrix,
    fingerprint_from_trace,
    check_collinearity,
    threshold_sensitivity,
)
from .rules import ARCHETYPE_RULES
from .trace import ClassificationTrace

__all__ = [
    "ArchetypeProfile",
    "ConfusionCluster",
    "RuleCoFiringMatrix",
    "AnalysisReport",
    "analyse_traces",
    "compare_reports",
]


# ═══════════════════════════════════════════════════════════════════
# ArchetypeProfile — statistical summary per archetype
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ArchetypeProfile:
    """Statistical summary for one archetype across multiple proteins.

    Attributes
    ----------
    archetype : str
        Archetype name.
    n_proteins : int
        Number of proteins classified as this archetype.
    mean_score : float
        Mean winning score.
    mean_margin : float
        Mean score margin (gap to runner-up).
    mean_n_rules_fired : float
        Mean number of rules that fired.
    mean_firing_density : float
        Mean fraction of rules that fired.
    lens_activation_rates : dict[str, float]
        Fraction of proteins where each lens activated.
    mean_context_boost : dict[str, float]
        Mean per-archetype context boost.
    mean_alpha : float
        Mean MetaFick α.
    common_runner_up : str
        Most frequent runner-up archetype.
    """

    archetype: str
    n_proteins: int
    mean_score: float
    mean_margin: float
    mean_n_rules_fired: float
    mean_firing_density: float
    lens_activation_rates: Dict[str, float]
    mean_context_boost: Dict[str, float]
    mean_alpha: float
    common_runner_up: str

    def summary(self) -> str:
        lenses = ", ".join(
            f"{k}={v:.0%}" for k, v in self.lens_activation_rates.items()
            if v > 0
        )
        return (
            f"{self.archetype} (n={self.n_proteins}): "
            f"score={self.mean_score:.3f}, "
            f"margin={self.mean_margin:.3f}, "
            f"rules={self.mean_n_rules_fired:.0f}, "
            f"α={self.mean_alpha:.3f}, "
            f"lenses=[{lenses}], "
            f"runner_up={self.common_runner_up}"
        )


# ═══════════════════════════════════════════════════════════════════
# ConfusionCluster — proteins that get confused with each other
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ConfusionCluster:
    """A group of proteins with similar classification profiles.

    Proteins in the same cluster have small Hamming distances between
    their rule-firing fingerprints, meaning they activate similar
    decision pathways regardless of their true archetype.

    Attributes
    ----------
    proteins : tuple[str, ...]
        Protein names in this cluster.
    archetypes : tuple[str, ...]
        True archetypes of proteins in this cluster.
    mean_internal_distance : float
        Mean pairwise Hamming distance within the cluster.
    centroid_density : float
        Firing density of the cluster centroid.
    shared_rules : tuple[str, ...]
        Rules that fired for ALL proteins in the cluster.
    unique_rules : dict[str, tuple[str, ...]]
        Rules unique to each protein (not shared by others in cluster).
    """

    proteins: Tuple[str, ...]
    archetypes: Tuple[str, ...]
    mean_internal_distance: float
    centroid_density: float
    shared_rules: Tuple[str, ...]
    unique_rules: Dict[str, Tuple[str, ...]]

    @property
    def n_proteins(self) -> int:
        return len(self.proteins)

    @property
    def is_mixed(self) -> bool:
        """True if the cluster contains multiple archetypes."""
        return len(set(self.archetypes)) > 1

    def summary(self) -> str:
        archs = ", ".join(sorted(set(self.archetypes)))
        mixed = " [MIXED]" if self.is_mixed else ""
        return (
            f"Cluster({self.n_proteins} proteins{mixed}): "
            f"archetypes=[{archs}], "
            f"mean_dist={self.mean_internal_distance:.1f}, "
            f"shared_rules={len(self.shared_rules)}"
        )


# ═══════════════════════════════════════════════════════════════════
# Rule Co-Firing Matrix
# ═══════════════════════════════════════════════════════════════════

@dataclass
class RuleCoFiringMatrix:
    """Which rules tend to fire together.

    Attributes
    ----------
    matrix : np.ndarray
        Co-firing matrix of shape ``(n_rules, n_rules)``.
        ``matrix[i, j]`` = number of proteins where both rules fired.
    rule_names : tuple[str, ...]
        Rule names corresponding to rows/columns.
    n_proteins : int
        Total number of proteins analysed.
    """

    matrix: np.ndarray
    rule_names: Tuple[str, ...]
    n_proteins: int

    def correlation(self, rule_a: str, rule_b: str) -> float:
        """Co-firing rate between two rules (0–1)."""
        if rule_a not in self.rule_names or rule_b not in self.rule_names:
            return 0.0
        ia = self.rule_names.index(rule_a)
        ib = self.rule_names.index(rule_b)
        return float(self.matrix[ia, ib]) / self.n_proteins if self.n_proteins else 0.0

    def top_pairs(self, top_k: int = 20) -> List[Tuple[str, str, float]]:
        """Top co-firing rule pairs by frequency."""
        pairs = []
        n = len(self.rule_names)
        for i in range(n):
            for j in range(i + 1, n):
                rate = float(self.matrix[i, j]) / self.n_proteins if self.n_proteins else 0.0
                pairs.append((self.rule_names[i], self.rule_names[j], rate))
        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs[:top_k]

    def summary(self) -> str:
        n = len(self.rule_names)
        total_cells = n * (n - 1) // 2
        active = sum(
            1 for i in range(n)
            for j in range(i + 1, n)
            if self.matrix[i, j] > 0
        )
        return (
            f"RuleCoFiringMatrix({n} rules, "
            f"{active}/{total_cells} non-zero pairs, "
            f"{self.n_proteins} proteins)"
        )


# ═══════════════════════════════════════════════════════════════════
# AnalysisReport — the comprehensive analysis output
# ═══════════════════════════════════════════════════════════════════

@dataclass
class AnalysisReport:
    """Comprehensive analysis of classification traces.

    Produced by :func:`analyse_traces`.  Contains all derived
    algebraic structures and statistical summaries.

    Attributes
    ----------
    n_proteins : int
        Number of proteins analysed.
    protein_names : list[str]
        Protein names in order.
    archetype_profiles : dict[str, ArchetypeProfile]
        Per-archetype statistical summaries.
    fingerprints : list[Fingerprint]
        Binary rule-firing fingerprints.
    distance_mat : np.ndarray
        Pairwise Hamming distance matrix.
    confusion_clusters : list[ConfusionCluster]
        Groups of proteins with similar decision pathways.
    collinearity : list[InstrumentCollinearity]
        Instrument-triple collinearity results.
    firing_lattice : FiringLattice
        Distributive lattice of observed firing patterns.
    sensitivity : ThresholdSensitivity
        Threshold sensitivity landscape.
    co_firing : RuleCoFiringMatrix
        Rule co-firing matrix.
    overall_accuracy : float
        Classification accuracy (if ground truth available).
    """

    n_proteins: int
    protein_names: List[str]
    archetype_profiles: Dict[str, ArchetypeProfile]
    fingerprints: List[Fingerprint]
    distance_mat: np.ndarray
    confusion_clusters: List[ConfusionCluster]
    collinearity: List[InstrumentCollinearity]
    firing_lattice: FiringLattice
    sensitivity: ThresholdSensitivity
    co_firing: RuleCoFiringMatrix
    overall_accuracy: float = 0.0

    def summary(self) -> str:
        """Multi-line summary of all analyses."""
        lines = [
            f"Analysis Report — {self.n_proteins} proteins",
            "=" * 55,
            "",
            "Archetype Profiles:",
        ]
        for arch, prof in sorted(self.archetype_profiles.items()):
            lines.append(f"  {prof.summary()}")

        lines.append("")
        lines.append(f"Fingerprints: {self.firing_lattice.summary()}")

        collinear = [c for c in self.collinearity if c.collinear]
        lines.append(f"Collinear instrument triples: {len(collinear)}")
        for c in collinear[:5]:
            lines.append(f"  {c.summary()}")

        mixed = [c for c in self.confusion_clusters if c.is_mixed]
        lines.append(f"Confusion clusters: {len(self.confusion_clusters)} "
                     f"({len(mixed)} mixed-archetype)")
        for c in mixed[:3]:
            lines.append(f"  {c.summary()}")

        lines.append(f"Sensitivity: {self.sensitivity.summary()}")
        lines.append(f"Co-firing: {self.co_firing.summary()}")

        # Threshold cascade shape
        if self.sensitivity.shape_signature:
            lines.append("")
            lines.append("Threshold cascade fingerprint (top-5):")
            for key in self.sensitivity.shape_signature:
                val = self.sensitivity.sensitivities[key]
                lines.append(f"  {key}: {val:.4f}")

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# analyse_traces — the main entry point
# ═══════════════════════════════════════════════════════════════════

def analyse_traces(
    traces: Sequence[ClassificationTrace],
    protein_names: Optional[Sequence[str]] = None,
    ground_truth: Optional[Dict[str, str]] = None,
    cluster_threshold: float = 5.0,
) -> AnalysisReport:
    """Run the full analysis pipeline on a set of classification traces.

    Parameters
    ----------
    traces : sequence of ClassificationTrace
        Traces from the classification pipeline.
    protein_names : sequence of str, optional
        Labels for each trace.  If None, uses "protein_0", "protein_1", etc.
    ground_truth : dict[str, str], optional
        ``{protein_name: true_archetype}`` for accuracy calculation.
    cluster_threshold : float
        Maximum mean internal Hamming distance for a confusion cluster.

    Returns
    -------
    AnalysisReport
    """
    n = len(traces)
    if protein_names is None:
        protein_names = [f"protein_{i}" for i in range(n)]
    names = list(protein_names)

    # 1. Fingerprints
    fingerprints = [
        fingerprint_from_trace(trace, protein_name=name)
        for trace, name in zip(traces, names)
    ]

    # 2. Distance matrix
    dist_mat, _ = distance_matrix(fingerprints) if n > 0 else (np.zeros((0, 0)), [])

    # 3. Archetype profiles
    by_archetype: Dict[str, List[Tuple[ClassificationTrace, Fingerprint]]] = defaultdict(list)
    for trace, fp in zip(traces, fingerprints):
        by_archetype[trace.identity].append((trace, fp))

    archetype_profiles = {}
    for arch, items in by_archetype.items():
        arch_traces = [t for t, _ in items]
        arch_fps = [f for _, f in items]

        mean_score = float(np.mean([t.scores.get(arch, 0.0) for t in arch_traces]))
        mean_margin = float(np.mean([t.score_margin for t in arch_traces]))
        mean_n_rules = float(np.mean([t.n_rules_fired for t in arch_traces]))
        mean_density = float(np.mean([fp.density for fp in arch_fps]))
        mean_alpha = float(np.mean([t.alpha_meta for t in arch_traces]))

        # Lens activation rates
        lens_counts: Dict[str, int] = Counter()
        for t in arch_traces:
            for lt in t.lens_traces:
                if lt.activated:
                    lens_counts[lt.lens_name] += 1
        lens_rates = {
            k: v / len(arch_traces) for k, v in lens_counts.items()
        }

        # Mean context boost
        ctx_keys = set()
        for t in arch_traces:
            ctx_keys.update(t.context_boost.keys())
        mean_ctx = {}
        for k in ctx_keys:
            mean_ctx[k] = float(np.mean([
                t.context_boost.get(k, 0.0) for t in arch_traces
            ]))

        # Common runner-up
        runner_ups = Counter(t.runner_up for t in arch_traces)
        common_runner = runner_ups.most_common(1)[0][0] if runner_ups else ""

        archetype_profiles[arch] = ArchetypeProfile(
            archetype=arch,
            n_proteins=len(items),
            mean_score=mean_score,
            mean_margin=mean_margin,
            mean_n_rules_fired=mean_n_rules,
            mean_firing_density=mean_density,
            lens_activation_rates=lens_rates,
            mean_context_boost=mean_ctx,
            mean_alpha=mean_alpha,
            common_runner_up=common_runner,
        )

    # 4. Confusion clusters (greedy single-linkage with threshold)
    confusion_clusters = _build_confusion_clusters(
        fingerprints, dist_mat, names, traces, cluster_threshold
    )

    # 5. Instrument collinearity
    collinearity = check_collinearity(traces, names)

    # 6. Firing lattice
    firing_lattice = build_firing_lattice(fingerprints)

    # 7. Threshold sensitivity
    sensitivity = threshold_sensitivity(traces)

    # 8. Rule co-firing matrix
    co_firing = _build_co_firing_matrix(fingerprints)

    # 9. Accuracy
    accuracy = 0.0
    if ground_truth:
        correct = sum(
            1 for trace, name in zip(traces, names)
            if ground_truth.get(name) == trace.identity
        )
        accuracy = correct / n if n else 0.0

    return AnalysisReport(
        n_proteins=n,
        protein_names=names,
        archetype_profiles=archetype_profiles,
        fingerprints=fingerprints,
        distance_mat=dist_mat,
        confusion_clusters=confusion_clusters,
        collinearity=collinearity,
        firing_lattice=firing_lattice,
        sensitivity=sensitivity,
        co_firing=co_firing,
        overall_accuracy=accuracy,
    )


# ═══════════════════════════════════════════════════════════════════
# compare_reports — cross-experiment comparison
# ═══════════════════════════════════════════════════════════════════

def compare_reports(
    before: AnalysisReport,
    after: AnalysisReport,
) -> str:
    """Compare two analysis reports to identify what changed.

    Returns a multi-line human-readable delta report.

    Parameters
    ----------
    before : AnalysisReport
        The baseline report.
    after : AnalysisReport
        The new report.

    Returns
    -------
    str
        Multi-line comparison.
    """
    lines = [
        "Cross-Experiment Comparison",
        "=" * 55,
        f"Before: {before.n_proteins} proteins",
        f"After:  {after.n_proteins} proteins",
        "",
    ]

    # Accuracy delta
    if before.overall_accuracy or after.overall_accuracy:
        d = after.overall_accuracy - before.overall_accuracy
        lines.append(
            f"Accuracy: {before.overall_accuracy:.1%} → "
            f"{after.overall_accuracy:.1%} ({d:+.1%})"
        )
        lines.append("")

    # Archetype profile changes
    all_archs = sorted(
        set(before.archetype_profiles.keys()) |
        set(after.archetype_profiles.keys())
    )
    lines.append("Archetype Profile Deltas:")
    for arch in all_archs:
        bp = before.archetype_profiles.get(arch)
        ap = after.archetype_profiles.get(arch)
        if bp and ap:
            d_score = ap.mean_score - bp.mean_score
            d_margin = ap.mean_margin - bp.mean_margin
            d_rules = ap.mean_n_rules_fired - bp.mean_n_rules_fired
            d_alpha = ap.mean_alpha - bp.mean_alpha
            lines.append(
                f"  {arch}: score {d_score:+.3f}, "
                f"margin {d_margin:+.3f}, "
                f"rules {d_rules:+.0f}, "
                f"α {d_alpha:+.3f}"
            )
        elif ap:
            lines.append(f"  {arch}: NEW (n={ap.n_proteins})")
        elif bp:
            lines.append(f"  {arch}: REMOVED (was n={bp.n_proteins})")

    # Sensitivity shape shift
    lines.append("")
    lines.append("Threshold Sensitivity Shape:")
    before_shape = before.sensitivity.shape_signature
    after_shape = after.sensitivity.shape_signature
    if before_shape == after_shape:
        lines.append(f"  Unchanged: {', '.join(before_shape)}")
    else:
        lines.append(f"  Before: {', '.join(before_shape) if before_shape else '(empty)'}")
        lines.append(f"  After:  {', '.join(after_shape) if after_shape else '(empty)'}")
        entered = set(after_shape) - set(before_shape)
        exited = set(before_shape) - set(after_shape)
        if entered:
            lines.append(f"  Entered top-5: {', '.join(entered)}")
        if exited:
            lines.append(f"  Exited top-5: {', '.join(exited)}")

    # Sensitivity magnitude
    d_mean = after.sensitivity.mean_sensitivity - before.sensitivity.mean_sensitivity
    d_max = after.sensitivity.max_sensitivity - before.sensitivity.max_sensitivity
    lines.append(f"  Mean sensitivity: {d_mean:+.4f}")
    lines.append(f"  Max sensitivity:  {d_max:+.4f}")

    # Lattice shape
    lines.append("")
    lines.append("Firing Lattice:")
    lines.append(f"  Before: {before.firing_lattice.summary()}")
    lines.append(f"  After:  {after.firing_lattice.summary()}")

    # Collinearity changes
    before_col = {c.triple for c in before.collinearity if c.collinear}
    after_col = {c.triple for c in after.collinearity if c.collinear}
    new_col = after_col - before_col
    lost_col = before_col - after_col
    if new_col or lost_col:
        lines.append("")
        lines.append("Collinearity changes:")
        for c in new_col:
            lines.append(f"  NEW collinear: ({', '.join(c)})")
        for c in lost_col:
            lines.append(f"  LOST collinear: ({', '.join(c)})")

    # Confusion cluster changes
    before_mixed = len([c for c in before.confusion_clusters if c.is_mixed])
    after_mixed = len([c for c in after.confusion_clusters if c.is_mixed])
    lines.append("")
    lines.append(
        f"Confusion clusters: {len(before.confusion_clusters)} → "
        f"{len(after.confusion_clusters)} "
        f"(mixed: {before_mixed} → {after_mixed})"
    )

    # Fingerprint diversity
    before_unique = len(set(fp.bits for fp in before.fingerprints))
    after_unique = len(set(fp.bits for fp in after.fingerprints))
    lines.append("")
    lines.append(
        f"Fingerprint diversity: {before_unique} → {after_unique} "
        f"unique patterns"
    )

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════

def _build_confusion_clusters(
    fingerprints: Sequence[Fingerprint],
    dist_mat: np.ndarray,
    names: List[str],
    traces: Sequence[ClassificationTrace],
    threshold: float,
) -> List[ConfusionCluster]:
    """Greedy single-linkage clustering by Hamming distance."""
    n = len(fingerprints)
    if n == 0:
        return []

    assigned = [False] * n
    clusters = []

    # Sort all pairs by distance
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((int(dist_mat[i, j]), i, j))
    pairs.sort()

    # Union-Find for single-linkage
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for dist, i, j in pairs:
        if dist <= threshold:
            union(i, j)

    # Group by cluster
    groups: Dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    for members in groups.values():
        if len(members) < 2:
            continue

        member_fps = [fingerprints[i] for i in members]
        member_names = tuple(names[i] for i in members)
        member_archs = tuple(traces[i].identity for i in members)

        # Mean internal distance
        dists = []
        for a, b in zip(range(len(members)), range(1, len(members))):
            for bb in range(a + 1, len(members)):
                dists.append(int(dist_mat[members[a], members[bb]]))
        mean_dist = float(np.mean(dists)) if dists else 0.0

        # Shared rules (fired in ALL members)
        shared = member_fps[0].as_set
        for fp in member_fps[1:]:
            shared = shared & fp.as_set

        # Unique rules per protein
        unique_rules = {}
        for i, fp in zip(members, member_fps):
            others = frozenset()
            for j, fp2 in zip(members, member_fps):
                if j != i:
                    others = others | fp2.as_set
            unique = fp.as_set - others
            unique_rules[names[i]] = tuple(sorted(unique))

        # Centroid density
        centroid_density = float(np.mean([fp.density for fp in member_fps]))

        clusters.append(ConfusionCluster(
            proteins=member_names,
            archetypes=member_archs,
            mean_internal_distance=mean_dist,
            centroid_density=centroid_density,
            shared_rules=tuple(sorted(shared)),
            unique_rules=unique_rules,
        ))

    # Sort by size (largest first), then by mixed-ness
    clusters.sort(key=lambda c: (-c.n_proteins, -int(c.is_mixed)))
    return clusters


def _build_co_firing_matrix(
    fingerprints: Sequence[Fingerprint],
) -> RuleCoFiringMatrix:
    """Build the rule co-firing matrix."""
    if not fingerprints:
        return RuleCoFiringMatrix(
            matrix=np.zeros((0, 0), dtype=int),
            rule_names=(),
            n_proteins=0,
        )

    rule_names = fingerprints[0].rule_names
    n_rules = len(rule_names)
    n_proteins = len(fingerprints)

    # Stack all fingerprint bit vectors
    bit_matrix = np.array([fp.bits for fp in fingerprints], dtype=np.int8)

    # Co-firing: transpose dot product
    co_fire = bit_matrix.T @ bit_matrix  # (n_rules, n_rules)

    return RuleCoFiringMatrix(
        matrix=co_fire,
        rule_names=rule_names,
        n_proteins=n_proteins,
    )
