"""Benchmark harness for the thermodynamic band classification pipeline.

Promotes the 52-protein corpus from D112 and the benchmark runner
pattern from D112/D113 into a reusable, cacheable, hookable module.

Quick start
-----------
>>> from ibp_enm.benchmark import BenchmarkRunner, EXPANDED_CORPUS
>>>
>>> runner = BenchmarkRunner(EXPANDED_CORPUS, cache_dir="~/.ibp_enm_cache")
>>> report = runner.run(verbose=True)
>>> print(report.summary())
>>>
>>> # Re-score with different synthesis (uses cached profiles):
>>> report2 = runner.run(rescore_only=True)
>>> print(report.delta(report2))

Corpus conventions
------------------
Each entry is a :class:`ProteinEntry` with ``(name, pdb_id, chain,
archetype)``.  The ``archetype`` field is the ground-truth label
used for accuracy calculations.

Hooks
-----
The runner accepts optional callbacks:

* ``on_protein_start(entry)`` — called before each protein
* ``on_protein_done(entry, result)`` — called after each protein
* ``on_error(entry, exception)`` — called on failure
"""

from __future__ import annotations

import json
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .archetypes import ARCHETYPE_EXPECTATIONS, GROUND_TRUTH, PROTEINS
from .cache import ProfileCache, profile_to_dict

__all__ = [
    "ProteinEntry",
    "ProteinResult",
    "BenchmarkReport",
    "BenchmarkRunner",
    "ORIGINAL_CORPUS",
    "EXPANDED_CORPUS",
]


# ═══════════════════════════════════════════════════════════════════
# Data types
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ProteinEntry:
    """One protein in the benchmark corpus."""
    name: str
    pdb_id: str
    chain: str
    archetype: str

    def __repr__(self) -> str:
        return f"ProteinEntry({self.name!r}, {self.pdb_id}:{self.chain}, {self.archetype})"


@dataclass
class ProteinResult:
    """Classification result for one protein."""
    entry: ProteinEntry
    predicted: str
    scores: Dict[str, float]
    correct: bool
    time_s: float
    error: Optional[str] = None

    # Diagnostic fields (populated when available)
    n_residues: int = 0
    n_contacts: int = 0
    enzyme_lens_activated: bool = False
    hinge_lens_activated: bool = False
    barrel_penalty_activated: bool = False
    initial_diagnosis: Optional[str] = None
    true_rank: int = 0

    @property
    def archetype(self) -> str:
        return self.entry.archetype


# ═══════════════════════════════════════════════════════════════════
# BenchmarkReport — structured results with analysis
# ═══════════════════════════════════════════════════════════════════

@dataclass
class BenchmarkReport:
    """Structured benchmark results with confusion matrix and analysis."""

    results: List[ProteinResult]
    timestamp: str = ""
    total_time_s: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            import datetime
            self.timestamp = datetime.datetime.now().isoformat()

    # ── Accuracy ────────────────────────────────────────────────

    @property
    def valid_results(self) -> List[ProteinResult]:
        """Results excluding errors."""
        return [r for r in self.results if r.error is None]

    @property
    def errors(self) -> List[ProteinResult]:
        return [r for r in self.results if r.error is not None]

    @property
    def n_correct(self) -> int:
        return sum(1 for r in self.valid_results if r.correct)

    @property
    def n_total(self) -> int:
        return len(self.valid_results)

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n_total if self.n_total else 0.0

    # ── Per-archetype ───────────────────────────────────────────

    @property
    def per_archetype(self) -> Dict[str, Dict[str, Any]]:
        """Per-archetype accuracy breakdown."""
        by_arch: Dict[str, List[ProteinResult]] = defaultdict(list)
        for r in self.valid_results:
            by_arch[r.archetype].append(r)
        result = {}
        for arch, rs in sorted(by_arch.items()):
            correct = sum(1 for r in rs if r.correct)
            result[arch] = {
                "correct": correct,
                "total": len(rs),
                "accuracy": correct / len(rs) if rs else 0.0,
                "misclassified_as": Counter(
                    r.predicted for r in rs if not r.correct),
            }
        return result

    # ── Confusion matrix ────────────────────────────────────────

    @property
    def confusion_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """Confusion matrix as (matrix, labels).

        Rows = true archetype, columns = predicted archetype.
        """
        labels = sorted(ARCHETYPE_EXPECTATIONS.keys())
        idx = {a: i for i, a in enumerate(labels)}
        n = len(labels)
        mat = np.zeros((n, n), dtype=int)
        for r in self.valid_results:
            if r.archetype in idx and r.predicted in idx:
                mat[idx[r.archetype], idx[r.predicted]] += 1
        return mat, labels

    # ── False predictions ───────────────────────────────────────

    @property
    def false_predictions(self) -> Dict[str, List[str]]:
        """Proteins falsely predicted as each archetype.

        Returns e.g. ``{"barrel": ["Lactoferrin", "ATCase_cat", ...]}``.
        """
        fp: Dict[str, List[str]] = defaultdict(list)
        for r in self.valid_results:
            if not r.correct:
                fp[r.predicted].append(r.entry.name)
        return dict(fp)

    # ── Summary ─────────────────────────────────────────────────

    def summary(self) -> str:
        """Human-readable summary string."""
        lines = [
            f"Benchmark Report — {self.timestamp}",
            f"{'=' * 55}",
            f"Overall: {self.n_correct}/{self.n_total}"
            f" ({self.accuracy:.1%})",
            f"Errors:  {len(self.errors)}",
            f"Time:    {self.total_time_s:.0f}s",
            "",
            "Per-archetype:",
        ]
        for arch, info in self.per_archetype.items():
            line = (f"  {arch:<15s} "
                    f"{info['correct']:>2d}/{info['total']:<2d} "
                    f"({info['accuracy']:.0%})")
            if info["misclassified_as"]:
                misses = ", ".join(
                    f"{k}×{v}" for k, v in
                    info["misclassified_as"].most_common())
                line += f"  [{misses}]"
            lines.append(line)

        fp = self.false_predictions
        if fp:
            lines.append("")
            lines.append("False predictions:")
            for arch, names in sorted(fp.items()):
                lines.append(f"  false {arch}: {len(names)}"
                             f"  ({', '.join(names[:5])}"
                             f"{'…' if len(names) > 5 else ''})")

        return "\n".join(lines)

    # ── Delta comparison ────────────────────────────────────────

    def delta(self, other: "BenchmarkReport") -> str:
        """Compare this report against another, showing what changed."""
        lines = [
            f"Delta: {self.accuracy:.1%} → {other.accuracy:.1%}"
            f" ({other.accuracy - self.accuracy:+.1%})",
            "",
        ]

        # Per-archetype delta
        pa_self = self.per_archetype
        pa_other = other.per_archetype
        all_archs = sorted(
            set(pa_self.keys()) | set(pa_other.keys()))
        lines.append("Per-archetype delta:")
        for arch in all_archs:
            a_self = pa_self.get(arch, {"accuracy": 0, "total": 0})
            a_other = pa_other.get(arch, {"accuracy": 0, "total": 0})
            d = a_other["accuracy"] - a_self["accuracy"]
            lines.append(
                f"  {arch:<15s} "
                f"{a_self['accuracy']:.0%} → {a_other['accuracy']:.0%}"
                f" ({d:+.0%})")

        # Individual protein changes
        self_by_name = {r.entry.name: r for r in self.valid_results}
        other_by_name = {r.entry.name: r for r in other.valid_results}

        fixed = []
        regressed = []
        for name in sorted(set(self_by_name) & set(other_by_name)):
            was = self_by_name[name]
            now = other_by_name[name]
            if not was.correct and now.correct:
                fixed.append(
                    f"  ✅ {name}: {was.predicted} → {now.predicted}"
                    f" (true: {was.archetype})")
            elif was.correct and not now.correct:
                regressed.append(
                    f"  ❌ {name}: {was.predicted} → {now.predicted}"
                    f" (true: {was.archetype})")

        if fixed:
            lines.append("")
            lines.append(f"Fixed ({len(fixed)}):")
            lines.extend(fixed)
        if regressed:
            lines.append("")
            lines.append(f"Regressed ({len(regressed)}):")
            lines.extend(regressed)

        # False prediction deltas
        fp_self = self.false_predictions
        fp_other = other.false_predictions
        all_fp_archs = sorted(
            set(fp_self.keys()) | set(fp_other.keys()))
        if all_fp_archs:
            lines.append("")
            lines.append("False prediction counts:")
            for arch in all_fp_archs:
                n_self = len(fp_self.get(arch, []))
                n_other = len(fp_other.get(arch, []))
                d = n_other - n_self
                lines.append(
                    f"  false {arch}: {n_self} → {n_other} ({d:+d})")

        return "\n".join(lines)

    # ── Serialisation ───────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-compatible dict."""
        return {
            "timestamp": self.timestamp,
            "total_time_s": self.total_time_s,
            "accuracy": self.accuracy,
            "n_correct": self.n_correct,
            "n_total": self.n_total,
            "results": [
                {
                    "name": r.entry.name,
                    "pdb_id": r.entry.pdb_id,
                    "chain": r.entry.chain,
                    "expected": r.entry.archetype,
                    "predicted": r.predicted,
                    "correct": r.correct,
                    "scores": r.scores,
                    "time_s": r.time_s,
                    "error": r.error,
                    "n_residues": r.n_residues,
                    "enzyme_lens_activated": r.enzyme_lens_activated,
                    "hinge_lens_activated": r.hinge_lens_activated,
                    "barrel_penalty_activated": r.barrel_penalty_activated,
                    "initial_diagnosis": r.initial_diagnosis,
                    "true_rank": r.true_rank,
                }
                for r in self.results
            ],
        }

    def save(self, path: str | Path) -> None:
        """Save report to JSON file."""
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2),
            encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BenchmarkReport":
        """Load report from JSON file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        results = []
        for r in data["results"]:
            entry = ProteinEntry(
                name=r["name"],
                pdb_id=r["pdb_id"],
                chain=r["chain"],
                archetype=r["expected"],
            )
            results.append(ProteinResult(
                entry=entry,
                predicted=r["predicted"],
                scores=r.get("scores", {}),
                correct=r["correct"],
                time_s=r.get("time_s", 0),
                error=r.get("error"),
                n_residues=r.get("n_residues", 0),
                enzyme_lens_activated=r.get("enzyme_lens_activated", False),
                hinge_lens_activated=r.get("hinge_lens_activated", False),
                barrel_penalty_activated=r.get(
                    "barrel_penalty_activated", False),
                initial_diagnosis=r.get("initial_diagnosis"),
                true_rank=r.get("true_rank", 0),
            ))
        return cls(
            results=results,
            timestamp=data.get("timestamp", ""),
            total_time_s=data.get("total_time_s", 0),
        )


# ═══════════════════════════════════════════════════════════════════
# Corpora
# ═══════════════════════════════════════════════════════════════════

ORIGINAL_CORPUS: List[ProteinEntry] = [
    ProteinEntry("T4_lysozyme",       "2LZM", "A", "enzyme_active"),
    ProteinEntry("HEWL",              "1LYZ", "A", "enzyme_active"),
    ProteinEntry("CaM_Ca_bound",      "3CLN", "A", "dumbbell"),
    ProteinEntry("Myoglobin",         "1MBO", "A", "globin"),
    ProteinEntry("AdK_open",          "4AKE", "A", "allosteric"),
    ProteinEntry("DHFR",              "3DFR", "A", "enzyme_active"),
    ProteinEntry("Streptavidin",      "1STP", "A", "enzyme_active"),
    ProteinEntry("TIM_barrel",        "1TIM", "A", "barrel"),
    ProteinEntry("LAO_binding",       "2LAO", "A", "dumbbell"),
    ProteinEntry("HIV_protease",      "1HHP", "A", "enzyme_active"),
    ProteinEntry("Hemoglobin_alpha",  "2HHB", "A", "globin"),
    ProteinEntry("Citrate_synthase",  "5CSC", "A", "barrel"),
]

# ── Expanded corpus (from D112) ──

_NEW_ENZYME = [
    ProteinEntry("Chymotrypsin",       "4CHA", "A", "enzyme_active"),
    ProteinEntry("Trypsin",            "2PTN", "A", "enzyme_active"),
    ProteinEntry("RNase_A",            "7RSA", "A", "enzyme_active"),
    ProteinEntry("Carbonic_anhyd_II",  "2CBA", "A", "enzyme_active"),
    ProteinEntry("Subtilisin",         "1SBT", "A", "enzyme_active"),
    ProteinEntry("Thermolysin",        "4TMN", "A", "enzyme_active"),
    ProteinEntry("Papain",             "9PAP", "A", "enzyme_active"),
    ProteinEntry("Elastase",           "3EST", "A", "enzyme_active"),
]

_NEW_BARREL = [
    ProteinEntry("Aldolase_A",         "1ALD", "A", "barrel"),
    ProteinEntry("Enolase",            "2ONE", "A", "barrel"),
    ProteinEntry("Xylose_isomerase",   "1XIS", "A", "barrel"),
    ProteinEntry("Rubisco_large",      "1RCX", "A", "barrel"),
    ProteinEntry("Glycolate_oxidase",  "1GOX", "A", "barrel"),
    ProteinEntry("Tryptophan_synth",   "1WDV", "A", "barrel"),
    ProteinEntry("KDPG_aldolase",      "1EUA", "A", "barrel"),
    ProteinEntry("Mandelate_racemase", "2MNR", "A", "barrel"),
]

_NEW_GLOBIN = [
    ProteinEntry("Leghemoglobin",      "1BIN", "A", "globin"),
    ProteinEntry("Neuroglobin",        "1OJ6", "A", "globin"),
    ProteinEntry("Cytochrome_b5",      "3B5C", "A", "globin"),
    ProteinEntry("Cytochrome_c",       "1HRC", "A", "globin"),
    ProteinEntry("Hemoglobin_beta",    "2HHB", "B", "globin"),
    ProteinEntry("Erythrocruorin",     "1ECA", "A", "globin"),
    ProteinEntry("Myoglobin_whale",    "1A6M", "A", "globin"),
    ProteinEntry("Truncated_Hb",       "1DLW", "A", "globin"),
]

_NEW_DUMBBELL = [
    ProteinEntry("Lactoferrin",        "1LFG", "A", "dumbbell"),
    ProteinEntry("Transferrin",        "1D3K", "A", "dumbbell"),
    ProteinEntry("Pyruvate_kinase",    "1PKN", "A", "dumbbell"),
    ProteinEntry("Immunoglobulin",     "1IGT", "A", "dumbbell"),
    ProteinEntry("MBP",                "1OMP", "A", "dumbbell"),
    ProteinEntry("Phosphoglycerate_k", "3PGK", "A", "dumbbell"),
    ProteinEntry("HSP70_NBD",          "1S3X", "A", "dumbbell"),
    ProteinEntry("Glutamine_BP",       "1GGG", "A", "dumbbell"),
]

_NEW_ALLOSTERIC = [
    ProteinEntry("ATCase_cat",         "1D09", "A", "allosteric"),
    ProteinEntry("Hemoglobin_T",       "2HHB", "C", "allosteric"),
    ProteinEntry("Phosphofructokinase","3PFK", "A", "allosteric"),
    ProteinEntry("Glycogen_phosph",    "1GPB", "A", "allosteric"),
    ProteinEntry("Protein_kinase_A",   "1ATP", "E", "allosteric"),
    ProteinEntry("ABP_open",           "1ABP", "A", "allosteric"),
    ProteinEntry("CheY",              "3CHY", "A", "allosteric"),
    ProteinEntry("GroEL_subunit",      "1GRL", "A", "allosteric"),
]

EXPANDED_CORPUS: List[ProteinEntry] = (
    list(ORIGINAL_CORPUS)
    + _NEW_ENZYME
    + _NEW_BARREL
    + _NEW_GLOBIN
    + _NEW_DUMBBELL
    + _NEW_ALLOSTERIC
)


# ═══════════════════════════════════════════════════════════════════
# BenchmarkRunner
# ═══════════════════════════════════════════════════════════════════

class BenchmarkRunner:
    """Reusable benchmark harness with caching, hooks, and reporting.

    Parameters
    ----------
    corpus : list[ProteinEntry]
        Proteins to benchmark.
    cache_dir : str or Path or None
        If given, profile caching is enabled.  Cached profiles
        allow re-scoring in <1s per protein instead of ~2 min.
    """

    def __init__(
        self,
        corpus: List[ProteinEntry] | None = None,
        cache_dir: str | Path | None = None,
    ):
        self.corpus = list(corpus) if corpus is not None else list(EXPANDED_CORPUS)
        self.cache = ProfileCache(cache_dir) if cache_dir else None

    def run(
        self,
        *,
        rescore_only: bool = False,
        verbose: bool = False,
        on_protein_start: Optional[Callable] = None,
        on_protein_done: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
    ) -> BenchmarkReport:
        """Run the full benchmark.

        Parameters
        ----------
        rescore_only : bool
            If True, skip carving and only re-score cached profiles.
            Requires all proteins to have cached profiles.
        verbose : bool
            Print progress to stdout.
        on_protein_start : callable, optional
            ``f(entry)`` called before each protein.
        on_protein_done : callable, optional
            ``f(entry, result)`` called after each protein.
        on_error : callable, optional
            ``f(entry, exception)`` called on failure.

        Returns
        -------
        BenchmarkReport
        """
        # Lazy import to avoid circular dependency
        from .band import run_single_protein

        results: List[ProteinResult] = []
        t_total = time.perf_counter()

        for i, entry in enumerate(self.corpus):
            if on_protein_start:
                on_protein_start(entry)

            if verbose:
                print(f"[{i+1}/{len(self.corpus)}] {entry.name} "
                      f"({entry.pdb_id}:{entry.chain}) …", end=" ",
                      flush=True)

            t0 = time.perf_counter()
            try:
                if rescore_only:
                    pr = self._rescore(entry)
                else:
                    pr = self._run_protein(entry)
                dt = time.perf_counter() - t0

                predicted = pr["band_identity"]
                correct = (predicted == entry.archetype)
                scores = pr["band_result"]["identity"]["scores"]

                # True rank
                sorted_scores = sorted(
                    scores.items(), key=lambda x: -x[1])
                true_rank = next(
                    (i + 1 for i, (a, _) in enumerate(sorted_scores)
                     if a == entry.archetype), len(sorted_scores))

                result = ProteinResult(
                    entry=entry,
                    predicted=predicted,
                    scores=scores,
                    correct=correct,
                    time_s=round(dt, 1),
                    n_residues=pr.get("N", 0),
                    n_contacts=pr.get("n_contacts", 0),
                    enzyme_lens_activated=pr.get(
                        "enzyme_lens_activated", False),
                    hinge_lens_activated=pr.get(
                        "hinge_lens_activated", False),
                    barrel_penalty_activated=pr.get(
                        "band_result", {}).get("identity", {}).get(
                        "barrel_penalty_activated", False),
                    initial_diagnosis=pr.get(
                        "initial_diagnosis", {}).get("archetype"),
                    true_rank=true_rank,
                )

                if verbose:
                    mark = "✓" if correct else "✗"
                    print(f"{mark} {predicted}"
                          f" (true={entry.archetype}) [{dt:.1f}s]")

                if on_protein_done:
                    on_protein_done(entry, result)

            except Exception as exc:
                dt = time.perf_counter() - t0
                result = ProteinResult(
                    entry=entry,
                    predicted="error",
                    scores={},
                    correct=False,
                    time_s=round(dt, 1),
                    error=f"{type(exc).__name__}: {exc}",
                )

                if verbose:
                    print(f"ERROR: {exc}")

                if on_error:
                    on_error(entry, exc)

            results.append(result)

        total_time = time.perf_counter() - t_total

        return BenchmarkReport(
            results=results,
            total_time_s=round(total_time, 1),
        )

    def _run_protein(self, entry: ProteinEntry) -> Dict:
        """Run a single protein through the full pipeline, caching profiles."""
        from .band import run_single_protein, ThermodynamicBand
        from .instruments import INSTRUMENTS

        pr = run_single_protein(
            entry.pdb_id, entry.chain, name=entry.name)

        # Cache profiles if cache is configured
        if self.cache is not None:
            band_result = pr.get("band_result", {})
            per_inst = band_result.get("per_instrument", {})
            identity = band_result.get("identity", {})

            # We need to save enough metadata to re-score
            metadata = {
                "name": entry.name,
                "pdb_id": entry.pdb_id,
                "chain": entry.chain,
                "archetype": entry.archetype,
                "N": pr.get("N", 0),
                "n_contacts": pr.get("n_contacts", 0),
                "initial_diagnosis": pr.get(
                    "initial_diagnosis", {}).get("archetype"),
            }

            # Extract profiles from the band_result per-instrument
            # votes.  The full ThermoReactionProfile data is in
            # the per_instrument summaries, but for cache we need
            # the actual profile objects.  Since run_single_protein
            # doesn't expose them directly, we save what we can from
            # the summary data.
            #
            # For true profile caching, we store the per_instrument
            # dict and the identity result so rescore can replay it.
            metadata["per_instrument"] = per_inst
            metadata["identity_result"] = {
                k: v for k, v in identity.items()
                if k != "per_carver_votes"  # redundant with per_inst
            }

            self.cache.save(
                entry.pdb_id, entry.chain,
                [],  # full profile caching added in step 2 integration
                metadata=metadata,
            )

        return pr

    def _rescore(self, entry: ProteinEntry) -> Dict:
        """Re-score from cached profiles (no carving)."""
        if self.cache is None:
            raise RuntimeError(
                "Cannot rescore without a cache. "
                "Pass cache_dir to BenchmarkRunner.")

        profiles, metadata = self.cache.load(entry.pdb_id, entry.chain)

        if not profiles:
            # If we only have metadata from a summary-level cache,
            # we can still return the cached identity.
            identity = metadata.get("identity_result", {})
            return {
                "band_identity": identity.get("identity", "unknown"),
                "band_result": {"identity": identity,
                                "per_instrument": metadata.get(
                                    "per_instrument", {})},
                "N": metadata.get("N", 0),
                "n_contacts": metadata.get("n_contacts", 0),
                "initial_diagnosis": {"archetype": metadata.get(
                    "initial_diagnosis")},
                "enzyme_lens_activated": identity.get(
                    "enzyme_lens_activated", False),
                "hinge_lens_activated": identity.get(
                    "hinge_lens_activated", False),
            }

        # Full profile-level rescore
        from .lens_stack import LensStackSynthesizer

        synth = LensStackSynthesizer(
            evals=None, evecs=None,
            domain_labels=None, contacts=None,
        )
        final_votes = [p.archetype_vote() for p in profiles]
        meta_state = synth.compute_meta_fick_state(final_votes)
        identity_result = synth.synthesize_identity(profiles, meta_state)

        return {
            "band_identity": identity_result["identity"],
            "band_result": {"identity": identity_result},
            "N": metadata.get("N", 0),
            "n_contacts": metadata.get("n_contacts", 0),
            "initial_diagnosis": {"archetype": metadata.get(
                "initial_diagnosis")},
            "enzyme_lens_activated": identity_result.get(
                "enzyme_lens_activated", False),
            "hinge_lens_activated": identity_result.get(
                "hinge_lens_activated", False),
        }
