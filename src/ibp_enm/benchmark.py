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
    "CrossValidationResult",
    "CrossValidator",
    "ORIGINAL_CORPUS",
    "EXPANDED_CORPUS",
    "LARGE_CORPUS",
    "ParameterUsefulnessResult",
    "ParameterUsefulnessAnalyzer",
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

    # Full audit trail (v0.7.0)
    trace: Any = None

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
        from .cache import _numpy_safe
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
                    "scores": _numpy_safe(r.scores),
                    "time_s": r.time_s,
                    "error": r.error,
                    "n_residues": r.n_residues,
                    "enzyme_lens_activated": r.enzyme_lens_activated,
                    "hinge_lens_activated": r.hinge_lens_activated,
                    "barrel_penalty_activated": r.barrel_penalty_activated,
                    "initial_diagnosis": r.initial_diagnosis,
                    "true_rank": r.true_rank,
                    "trace": (r.trace.to_dict()
                              if hasattr(r.trace, "to_dict") else None),
                }
                for r in self.results
            ],
        }

    def save(self, path: str | Path) -> None:
        """Save report to JSON file."""
        from .cache import _numpy_safe
        Path(path).write_text(
            json.dumps(_numpy_safe(self.to_dict()), indent=2),
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
    ProteinEntry("Chymotrypsin",       "2CGA", "A", "enzyme_active"),  # chymotrypsinogen; 4CHA is multi-chain (11 res on chain A)
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
    ProteinEntry("Cytochrome_b5",      "2I96", "A", "globin"),  # 3B5C is all-HETATM (1980s entry)
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
    ProteinEntry("Protein_kinase_A",   "1ATP", "A", "allosteric"),  # label_asym_id=A (auth=E)
    ProteinEntry("ABP_open",           "8ABP", "A", "allosteric"),  # 1ABP is all-HETATM (1970s entry)
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


# ── Large corpus (200 proteins, ~40 per archetype) ──────────────

_LARGE_ENZYME = [
    ProteinEntry("Barnase",             "1BNI", "A", "enzyme_active"),  # ribonuclease, N=108
    ProteinEntry("Staph_nuclease",      "1STN", "A", "enzyme_active"),  # SNase, N=136
    ProteinEntry("Phospholipase_A2",    "1BP2", "A", "enzyme_active"),  # PLA2, N=123
    ProteinEntry("Carboxypeptidase_A",  "5CPA", "A", "enzyme_active"),  # metalloprotease, N=307
    ProteinEntry("Pepsin",              "4PEP", "A", "enzyme_active"),  # aspartic protease, N=325
    ProteinEntry("Penicillopepsin",     "3APP", "A", "enzyme_active"),  # fungal protease, N=323
    ProteinEntry("Actinidin",           "2ACT", "A", "enzyme_active"),  # cysteine protease, N=217
    ProteinEntry("Beta_lactamase_TEM",  "1BTL", "A", "enzyme_active"),  # TEM-1, N=263
    ProteinEntry("Cyclophilin_A",       "2CPL", "A", "enzyme_active"),  # PPIase, N=164
    ProteinEntry("Thioredoxin",         "2TRX", "A", "enzyme_active"),  # oxidoreductase, N=108
    ProteinEntry("Cutinase",            "1CEX", "A", "enzyme_active"),  # serine esterase, N=197
    ProteinEntry("Cu_Zn_SOD",           "2SOD", "A", "enzyme_active"),  # superoxide dismutase, N=151
    ProteinEntry("Lysozyme_human",      "1LZ1", "A", "enzyme_active"),  # human lysozyme, N=130
    ProteinEntry("Xylanase_Bcirculans", "1XNB", "A", "enzyme_active"),  # GH11, N=185
    ProteinEntry("Ribonuclease_Sa",     "1RGG", "A", "enzyme_active"),  # RNase Sa, N=96
    ProteinEntry("CytC_peroxidase",     "2CYP", "A", "enzyme_active"),  # CcP, N=293
    ProteinEntry("Alpha_lytic_protease","2ALP", "A", "enzyme_active"),  # serine protease, N=198
    ProteinEntry("Proteinase_K",        "2PKC", "A", "enzyme_active"),  # subtilase, N=279
    ProteinEntry("Glutathione_Strans",  "2GST", "A", "enzyme_active"),  # GST, N=217
    ProteinEntry("Galactose_oxidase",   "1GOF", "A", "enzyme_active"),  # copper oxidase, N=639
    ProteinEntry("Carbonic_anhydrase_I","2CAB", "A", "enzyme_active"),  # CA-I, N=256
    ProteinEntry("Trypsinogen",         "1TGN", "A", "enzyme_active"),  # zymogen, N=222
    ProteinEntry("Phosphoglycerate_mut","3PGM", "A", "enzyme_active"),  # PGM, N=230
    ProteinEntry("Endoglucanase",       "1EGZ", "A", "enzyme_active"),  # cellulase, N=291
    ProteinEntry("Savinase",            "1SVN", "A", "enzyme_active"),  # subtilisin variant, N=269
    ProteinEntry("Lipase_Rhizomucor",   "4TGL", "A", "enzyme_active"),  # lipase, N=265
    ProteinEntry("Dihydroorotase",      "1J79", "A", "enzyme_active"),  # DHO, N=342
]

_LARGE_BARREL = [
    ProteinEntry("HisF",               "1THF", "A", "barrel"),  # TIM barrel, N=253
    ProteinEntry("NAL_lyase",          "1NAL", "A", "barrel"),  # TIM barrel, N=291
    ProteinEntry("Indole3GP_synthase", "1IGS", "A", "barrel"),  # TIM barrel, N=247
    ProteinEntry("PRAI",              "1PII", "A", "barrel"),  # TIM barrel, N=452
    ProteinEntry("Old_yellow_enzyme",  "1OYA", "A", "barrel"),  # TIM barrel, N=399
    ProteinEntry("PI_PLC",            "1PTG", "A", "barrel"),  # TIM barrel, N=296
    ProteinEntry("Orotidine_decarb",  "1DQX", "A", "barrel"),  # TIM barrel, N=267
    ProteinEntry("Alpha_amylase",     "1SMD", "A", "barrel"),  # TIM barrel domain, N=495
    ProteinEntry("Deoxyribose_P_ald", "1JCJ", "A", "barrel"),  # TIM barrel, N=252
    ProteinEntry("GFP",               "1GFL", "A", "barrel"),  # β-barrel, N=230
    ProteinEntry("Retinol_BP",        "1RBP", "A", "barrel"),  # lipocalin β-barrel, N=174
    ProteinEntry("Avidin",            "1AVD", "A", "barrel"),  # β-barrel, N=123
    ProteinEntry("FABP_intestinal",   "1IFB", "A", "barrel"),  # β-barrel, N=131
    ProteinEntry("Concanavalin_A",    "3CNA", "A", "barrel"),  # jelly-roll, N=237
    ProteinEntry("Beta_glucosidase",  "1BGA", "A", "barrel"),  # TIM barrel, N=447
    ProteinEntry("Alanine_racemase",  "1BD0", "A", "barrel"),  # TIM barrel, N=381
    ProteinEntry("Muconate_lactonize","1MUC", "A", "barrel"),  # TIM barrel, N=360
    ProteinEntry("Thiamin_P_synthase","2TPS", "A", "barrel"),  # TIM barrel, N=226
    ProteinEntry("Methylmalonyl_mut", "1REQ", "A", "barrel"),  # TIM barrel domain, N=727
    ProteinEntry("Glycerophosphodiesterase","1YMQ","A","barrel"),  # TIM barrel, N=260
    ProteinEntry("Cellulase_Cel5A",   "1CEN", "A", "barrel"),  # TIM barrel, N=334
    ProteinEntry("Dihydropteroate_syn","1AJ0","A", "barrel"),  # TIM barrel, N=282
    ProteinEntry("IGPS",              "1A53", "A", "barrel"),  # TIM barrel, N=247
    ProteinEntry("Phosphotriesterase","1HZY", "A", "barrel"),  # TIM barrel, N=331
    ProteinEntry("Transaldolase",     "1ONR", "A", "barrel"),  # TIM barrel, N=316
    ProteinEntry("Flavocytochrome_b2","1FCB", "A", "barrel"),  # TIM barrel domain, N=494
    ProteinEntry("Luciferase_bact",   "1LUC", "A", "barrel"),  # TIM barrel, N=326
    ProteinEntry("Chitinase_A1",      "1CTN", "A", "barrel"),  # TIM barrel, N=538
    ProteinEntry("Neopullulanase",    "1J0H", "A", "barrel"),  # TIM barrel, N=588
    ProteinEntry("Pyruvate_oxidase",  "1POW", "A", "barrel"),  # TIM barrel, N=585
]

_LARGE_GLOBIN = [
    ProteinEntry("Hemoglobin_lamprey",  "2LHB", "A", "globin"),  # lamprey Hb, N=149
    ProteinEntry("Cytochrome_b562",     "256B", "A", "globin"),  # 4-helix bundle, N=106
    ProteinEntry("Cytochrome_c2",       "1C2R", "A", "globin"),  # cyt c family, N=116
    ProteinEntry("Cytochrome_c551",     "351C", "A", "globin"),  # cyt c, N=82
    ProteinEntry("Myohemerythrin",      "2MHR", "A", "globin"),  # 4-helix bundle, N=118
    ProteinEntry("Cytoglobin",          "1V5H", "A", "globin"),  # human cytoglobin, N=151
    ProteinEntry("Hemoglobin_Ascaris",  "1ASH", "A", "globin"),  # nematode Hb, N=147
    ProteinEntry("Phycocyanin",         "1CPC", "A", "globin"),  # globin-like, N=162
    ProteinEntry("Ferritin",            "1FHA", "A", "globin"),  # 4-helix bundle subunit, N=172
    ProteinEntry("Bacterioferritin",    "1BFR", "A", "globin"),  # 4-helix bundle, N=158
    ProteinEntry("Hemoglobin_Chironomus","1ECD","A", "globin"),  # midge Hb, N=136
    ProteinEntry("Hemoglobin_sickle",   "2HBS", "A", "globin"),  # HbS, N=141
    ProteinEntry("Hemoglobin_Scapharca","4SDH", "A", "globin"),  # clam Hb, N=145
    ProteinEntry("Interleukin_4",       "1HIK", "A", "globin"),  # 4-helix bundle cytokine, N=129
    ProteinEntry("Growth_hormone",      "1HGU", "A", "globin"),  # 4-helix bundle, N=186
    ProteinEntry("GM_CSF",              "2GMF", "A", "globin"),  # 4-helix bundle, N=121
    ProteinEntry("EPO",                 "1BUY", "A", "globin"),  # 4-helix bundle, N=166
    ProteinEntry("Interferon_beta",     "1AU1", "A", "globin"),  # 4-helix bundle, N=166
    ProteinEntry("ROP_protein",         "1ROP", "A", "globin"),  # 4-helix bundle, N=56
    ProteinEntry("Cyt_c3",             "2CDV", "A", "globin"),  # multi-heme cyt, N=107
    ProteinEntry("Hemerythrin",         "1HMD", "A", "globin"),  # 4-helix bundle, N=113
    ProteinEntry("Myoglobin_horse",     "1WLA", "A", "globin"),  # horse Mb, N=153
    ProteinEntry("Hemoglobin_fetal_G",  "1FDH", "A", "globin"),  # fetal Hb, N=141
    ProteinEntry("Cytochrome_c6",       "1CYJ", "A", "globin"),  # cyt c6, N=90
    ProteinEntry("Leghemoglobin_lupin", "1GDI", "A", "globin"),  # plant Hb, N=153
    ProteinEntry("Hemoglobin_sea_lamp", "1HBG", "A", "globin"),  # sea lamprey Hb, N=147
    ProteinEntry("Cyt_c_oxidase_sub2",  "2OCC", "A", "globin"),  # CcO subunit, N=514
    ProteinEntry("Flavodoxin",          "1FLV", "A", "globin"),  # FMN-binding, N=168
    ProteinEntry("Ferredoxin",          "1FDX", "A", "globin"),  # iron-sulfur, N=54
    ProteinEntry("Apomyoglobin",        "1U7S", "A", "globin"),  # apo-Mb, N=153
]

_LARGE_DUMBBELL = [
    ProteinEntry("Galactose_BP",        "2GBP", "A", "dumbbell"),  # periplasmic BP, N=309
    ProteinEntry("Ribose_BP",           "2DRI", "A", "dumbbell"),  # periplasmic BP, N=271
    ProteinEntry("Sulfate_BP",          "1SBP", "A", "dumbbell"),  # periplasmic BP, N=309
    ProteinEntry("Maltodextrin_BP",     "1ANF", "A", "dumbbell"),  # MBP, N=370
    ProteinEntry("Dipeptide_BP",        "1DPP", "A", "dumbbell"),  # OppA, N=507
    ProteinEntry("Histidine_BP",        "1HSL", "A", "dumbbell"),  # periplasmic BP, N=238
    ProteinEntry("Ferric_BP",           "1MRP", "A", "dumbbell"),  # iron BP, N=309
    ProteinEntry("Ovotransferrin",      "1OVT", "A", "dumbbell"),  # two-lobe, N=682
    ProteinEntry("Hexokinase_yeast",    "2YHX", "A", "dumbbell"),  # hinge kinase, N=457
    ProteinEntry("Actin",               "1J6Z", "A", "dumbbell"),  # two-domain, N=368
    ProteinEntry("EF_Tu",               "1TTT", "A", "dumbbell"),  # multi-domain GTPase, N=405
    ProteinEntry("Serum_albumin",       "1AO6", "A", "dumbbell"),  # 3-domain, N=578
    ProteinEntry("Glutathione_reductase","3GRS","A", "dumbbell"),  # multi-domain, N=461
    ProteinEntry("GluR2_LBD",           "1FTJ", "A", "dumbbell"),  # clamshell, N=258
    ProteinEntry("Aspartate_aminotrans","7AAT", "A", "dumbbell"),  # 2-domain, N=401
    ProteinEntry("DNA_pol_beta",        "1BPY", "A", "dumbbell"),  # 2-domain, N=326
    ProteinEntry("Leucine_BP",          "1USG", "A", "dumbbell"),  # periplasmic BP, N=345
    ProteinEntry("Phosphate_BP",        "1A54", "A", "dumbbell"),  # periplasmic BP, N=321
    ProteinEntry("Hsp90_ATPase",        "1YES", "A", "dumbbell"),  # N-terminal domain, N=213
    ProteinEntry("Aminoacyl_tRNA_synth","1EUY", "A", "dumbbell"),  # multi-domain, N=529
    ProteinEntry("Protein_disulfide_iso","1MEK","A", "dumbbell"),  # PDI, N=120
    ProteinEntry("Catalase_HPII",       "1IPH", "A", "dumbbell"),  # 4-domain, N=727
    ProteinEntry("Aconitase",           "7ACN", "A", "dumbbell"),  # 4-domain, N=753
    ProteinEntry("Glutamine_synthetase","2GLS", "A", "dumbbell"),  # multi-domain, N=468
    ProteinEntry("Phosphoenolpyr_CK",   "1KHB", "A", "dumbbell"),  # multi-domain, N=603
    ProteinEntry("Citrate_lyase",       "1K6W", "A", "dumbbell"),  # multi-domain, N=423
    ProteinEntry("G6PD",                "1QKI", "A", "dumbbell"),  # multi-domain, N=488
    ProteinEntry("Aldehyde_dehydrog",   "1BXS", "A", "dumbbell"),  # multi-domain, N=494
    ProteinEntry("Alcohol_dehydrog",    "2OHX", "A", "dumbbell"),  # 2-domain, N=374
    ProteinEntry("Dihydropteridine_red","1DHR", "A", "dumbbell"),  # reductase, N=236
]

_LARGE_ALLOSTERIC = [
    ProteinEntry("Ras_p21",             "5P21", "A", "allosteric"),  # GTPase switch, N=166
    ProteinEntry("G_alpha_i",           "1GIA", "A", "allosteric"),  # het. G protein, N=310
    ProteinEntry("Transducin_alpha",    "1TAG", "A", "allosteric"),  # G protein, N=314
    ProteinEntry("Cdc42",               "1AN0", "A", "allosteric"),  # Rho GTPase, N=187
    ProteinEntry("Ran_GTPase",          "1BYU", "A", "allosteric"),  # nuclear GTPase, N=202
    ProteinEntry("NtrC_receiver",       "1NTR", "A", "allosteric"),  # response reg, N=124
    ProteinEntry("FixJ_receiver",       "1D5W", "A", "allosteric"),  # response reg, N=122
    ProteinEntry("FBPase",              "1FBP", "A", "allosteric"),  # allosteric enzyme, N=316
    ProteinEntry("Isocitrate_DH",       "1AI2", "A", "allosteric"),  # regulated enzyme, N=414
    ProteinEntry("ERK2",                "2ERK", "A", "allosteric"),  # MAP kinase, N=351
    ProteinEntry("CDK2_cycA",           "1FIN", "A", "allosteric"),  # CDK, N=298
    ProteinEntry("Src_kinase",          "2SRC", "A", "allosteric"),  # regulated kinase, N=449
    ProteinEntry("DnaK_ATPase",         "1DKG", "A", "allosteric"),  # Hsp70 ATPase, N=158
    ProteinEntry("CRP",                 "2CGP", "A", "allosteric"),  # cAMP receptor, N=200
    ProteinEntry("Lac_repressor",       "1LBI", "A", "allosteric"),  # allosteric TF, N=296
    ProteinEntry("Spo0F",               "1SRR", "A", "allosteric"),  # response reg, N=119
    ProteinEntry("PhoB_receiver",       "1B00", "A", "allosteric"),  # response reg, N=122
    ProteinEntry("Calcineurin_A",       "1AUI", "A", "allosteric"),  # PP2B, N=378
    ProteinEntry("Arf1",                "1HUR", "A", "allosteric"),  # GTPase, N=180
    ProteinEntry("Hemoglobin_R",        "1HHO", "A", "allosteric"),  # R-state Hb, N=141
    ProteinEntry("Threonine_deaminase", "1TDJ", "A", "allosteric"),  # allosteric enzyme, N=494
    ProteinEntry("PGDH",                "1PSD", "A", "allosteric"),  # V-type allosteric, N=404
    ProteinEntry("Aspartate_kinase",    "2J0W", "A", "allosteric"),  # allosteric enzyme, N=447
    ProteinEntry("GDH_bovine",          "1HWZ", "A", "allosteric"),  # allosteric GDH, N=501
    ProteinEntry("IMPDH",               "1JR1", "A", "allosteric"),  # allosteric, N=436
    ProteinEntry("Ribonuc_reductase_R1","1RLR", "A", "allosteric"),  # allosteric, N=737
    ProteinEntry("Pyruvate_dehydrog_E1","1W85", "A", "allosteric"),  # regulated, N=358
    ProteinEntry("RNR_R2",              "1RIB", "A", "allosteric"),  # RNR subunit, N=340
    ProteinEntry("Tryptophan_repressor","2OZ9", "A", "allosteric"),  # allosteric TF, N=104
    ProteinEntry("ArcA_receiver",       "1XHE", "A", "allosteric"),  # response reg, N=121
    ProteinEntry("OmpR_receiver",       "1ODD", "A", "allosteric"),  # response reg, N=100
]

LARGE_CORPUS: List[ProteinEntry] = (
    list(EXPANDED_CORPUS)
    + _LARGE_ENZYME
    + _LARGE_BARREL
    + _LARGE_GLOBIN
    + _LARGE_DUMBBELL
    + _LARGE_ALLOSTERIC
)
"""200-protein corpus: 52 from EXPANDED_CORPUS + 148 new entries.

Per-archetype counts:
  enzyme_active: 13 + 27 = 40
  barrel:        10 + 30 = 40
  globin:        10 + 30 = 40
  dumbbell:      10 + 30 = 40
  allosteric:     9 + 31 = 40
  TOTAL:                  200
"""


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
        thresholds: "ThresholdRegistry | None" = None,
    ):
        self.corpus = list(corpus) if corpus is not None else list(EXPANDED_CORPUS)
        self.cache = ProfileCache(cache_dir) if cache_dir else None
        self.thresholds = thresholds  # None → DEFAULT_THRESHOLDS

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
                    trace=pr.get("band_result", {}).get(
                        "identity", {}).get("trace"),
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
            entry.pdb_id, entry.chain, name=entry.name,
            thresholds=self.thresholds)

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
                if k not in ("per_carver_votes", "trace", "lens_traces")
                # per_carver_votes is redundant with per_inst;
                # trace contains ClassificationTrace objects;
                # lens_traces contains LensTrace objects —
                # none of these are JSON-serializable.
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
            thresholds=self.thresholds,
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


# ═══════════════════════════════════════════════════════════════════
# CrossValidator — margin analysis and perturbation stability
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CrossValidationResult:
    """Result of cross-validation analysis on the benchmark corpus.

    Attributes
    ----------
    n_proteins : int
        Total proteins evaluated (excluding errors).
    n_correct : int
        Number correctly classified.
    accuracy : float
        Fraction correct.
    per_protein : list[dict]
        Per-protein detail: name, expected, predicted, correct,
        margin, rank, stability (if perturbation was run).
    per_archetype : dict
        Per-archetype accuracy breakdown with mean/min margins.
    fragile_proteins : list[str]
        Proteins classified correctly but with margin < threshold
        (likely to flip with threshold changes).
    robust_proteins : list[str]
        Proteins classified correctly with comfortable margin.
    stability_scores : dict[str, float] | None
        If perturbation analysis was run, maps protein name to the
        fraction of perturbation trials where classification was
        correct.
    n_perturbation_trials : int
        Number of perturbation trials (0 if not run).
    perturbation_accuracy : float | None
        Mean accuracy across all perturbation trials.
    total_time_s : float
        Wall-clock time.
    """

    n_proteins: int = 0
    n_correct: int = 0
    accuracy: float = 0.0
    per_protein: List[Dict[str, Any]] = field(default_factory=list)
    per_archetype: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    fragile_proteins: List[str] = field(default_factory=list)
    robust_proteins: List[str] = field(default_factory=list)
    stability_scores: Optional[Dict[str, float]] = None
    n_perturbation_trials: int = 0
    perturbation_accuracy: Optional[float] = None
    total_time_s: float = 0.0

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "Cross-Validation Report",
            "=" * 60,
            f"Overall:  {self.n_correct}/{self.n_proteins}"
            f" ({self.accuracy:.1%})",
            f"Fragile:  {len(self.fragile_proteins)} proteins"
            f" (correct but margin < 0.05)",
            f"Robust:   {len(self.robust_proteins)} proteins"
            f" (correct with margin >= 0.05)",
            "",
        ]

        # Per-archetype
        lines.append("Per-archetype:")
        for arch in sorted(self.per_archetype):
            info = self.per_archetype[arch]
            lines.append(
                f"  {arch:<16s} "
                f"{info['correct']:>2d}/{info['total']:<2d} "
                f"({info['accuracy']:.0%})"
                f"  mean_margin={info['mean_margin']:+.4f}"
            )

        # Fragile proteins
        if self.fragile_proteins:
            lines.append("")
            lines.append("Fragile (correct but at risk):")
            for pp in self.per_protein:
                if pp["name"] in self.fragile_proteins:
                    stab = ""
                    if (self.stability_scores
                            and pp["name"] in self.stability_scores):
                        stab = (f"  stability="
                                f"{self.stability_scores[pp['name']]:.0%}")
                    lines.append(
                        f"  {pp['name']:25s}  margin={pp['margin']:+.4f}"
                        f"  rank={pp['rank']}{stab}"
                    )

        # Misclassified proteins
        wrong = [pp for pp in self.per_protein if not pp["correct"]]
        if wrong:
            lines.append("")
            lines.append(f"Misclassified ({len(wrong)}):")
            for pp in wrong:
                stab = ""
                if (self.stability_scores
                        and pp["name"] in self.stability_scores):
                    stab = (f"  stability="
                            f"{self.stability_scores[pp['name']]:.0%}")
                lines.append(
                    f"  {pp['name']:25s}  "
                    f"true={pp['expected']:<16s}  "
                    f"pred={pp['predicted']:<16s}  "
                    f"margin={pp['margin']:+.4f}{stab}"
                )

        # Perturbation summary
        if self.perturbation_accuracy is not None:
            lines.append("")
            lines.append(
                f"Perturbation analysis "
                f"({self.n_perturbation_trials} trials):"
            )
            lines.append(
                f"  Mean accuracy under perturbation: "
                f"{self.perturbation_accuracy:.1%}"
            )
            if self.stability_scores:
                n_stable = sum(
                    1 for v in self.stability_scores.values() if v >= 0.8
                )
                n_unstable = sum(
                    1 for v in self.stability_scores.values() if v < 0.5
                )
                lines.append(f"  Stable proteins (>=80%):   {n_stable}")
                lines.append(f"  Unstable proteins (<50%):  {n_unstable}")

        lines.append(f"\nTime: {self.total_time_s:.0f}s")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-compatible dict."""
        return {
            "n_proteins": self.n_proteins,
            "n_correct": self.n_correct,
            "accuracy": self.accuracy,
            "per_protein": self.per_protein,
            "per_archetype": self.per_archetype,
            "fragile_proteins": self.fragile_proteins,
            "robust_proteins": self.robust_proteins,
            "stability_scores": self.stability_scores,
            "n_perturbation_trials": self.n_perturbation_trials,
            "perturbation_accuracy": self.perturbation_accuracy,
            "total_time_s": self.total_time_s,
        }

    def save(self, path: str | Path) -> None:
        """Save to JSON file."""
        from .cache import _numpy_safe
        Path(path).write_text(
            json.dumps(_numpy_safe(self.to_dict()), indent=2),
            encoding="utf-8",
        )


class CrossValidator:
    """Margin analysis and perturbation-based cross-validation.

    Since thresholds and rules in ibp_enm are hand-tuned (not learned
    from training data), standard LOOCV where you retrain on N-1
    samples doesn't apply directly.  Instead, this class provides
    two complementary analyses:

    1. **Margin analysis** — for each protein, measure how far the
       correct archetype's score is from the winner.  Proteins with
       small positive margins are *fragile* (likely benefiting from
       threshold overfitting).  Proteins with negative margins are
       misclassified.

    2. **Perturbation stability** — jitter all 134 threshold
       parameters by ±N% and re-score every protein using the
       perturbed thresholds (via ``rescore_from_profiles``, so no
       re-carving needed — seconds, not hours).  The fraction of
       trials where each classification stays correct is its
       *stability score*.  This directly quantifies overfitting:
       a protein that's only correct under the exact production
       thresholds but flips under small perturbations is almost
       certainly fitted, not genuinely discriminated.

    Quick start
    -----------
    >>> from ibp_enm.benchmark import CrossValidator, EXPANDED_CORPUS
    >>> cv = CrossValidator(EXPANDED_CORPUS)
    >>> result = cv.run(verbose=True)
    >>> print(result.summary())
    >>>
    >>> # With perturbation analysis (fast — uses cached profiles):
    >>> result = cv.run(perturbation_pct=10, n_trials=50, verbose=True)
    >>> print(result.summary())

    Parameters
    ----------
    corpus : list[ProteinEntry]
        Proteins to evaluate.
    thresholds : ThresholdRegistry, optional
        Base thresholds.  Defaults to ``DEFAULT_THRESHOLDS``.
    fragile_margin : float
        Margin threshold below which a correct prediction is
        considered fragile.  Default 0.05.
    """

    def __init__(
        self,
        corpus: Optional[List[ProteinEntry]] = None,
        thresholds: "ThresholdRegistry | None" = None,
        fragile_margin: float = 0.05,
    ):
        self.corpus = (
            list(corpus) if corpus is not None
            else list(EXPANDED_CORPUS)
        )
        self.thresholds = thresholds
        self.fragile_margin = fragile_margin

    def run(
        self,
        *,
        perturbation_pct: float = 0,
        n_trials: int = 50,
        seed: int = 42,
        verbose: bool = False,
    ) -> CrossValidationResult:
        """Run cross-validation analysis.

        Parameters
        ----------
        perturbation_pct : float
            If > 0, run perturbation stability analysis with this
            percentage jitter (e.g. ``10`` means each threshold is
            independently multiplied by ``U(0.90, 1.10)``).
        n_trials : int
            Number of perturbation trials.
        seed : int
            Random seed for reproducibility.
        verbose : bool
            Print progress.

        Returns
        -------
        CrossValidationResult
        """
        from .band import run_single_protein, ThermodynamicBand
        from .thresholds import DEFAULT_THRESHOLDS

        base_thresholds = self.thresholds or DEFAULT_THRESHOLDS

        t_total = time.perf_counter()
        per_protein: List[Dict[str, Any]] = []

        # ── Phase 1: run all proteins with base thresholds ──
        if verbose:
            print("Phase 1: Full band classification …")
            print()

        # Cache the full run result so we can re-score cheaply
        band_cache: Dict[str, Dict] = {}

        for i, entry in enumerate(self.corpus):
            if verbose:
                print(
                    f"  [{i+1:2d}/{len(self.corpus)}] "
                    f"{entry.name:25s} "
                    f"({entry.pdb_id}:{entry.chain}) ",
                    end="", flush=True,
                )

            t0 = time.perf_counter()
            try:
                pr = run_single_protein(
                    entry.pdb_id, entry.chain, name=entry.name,
                    thresholds=base_thresholds,
                )
                dt = time.perf_counter() - t0

                predicted = pr["band_identity"]
                scores = pr["band_result"]["identity"]["scores"]
                correct = predicted == entry.archetype

                # Margin: score gap between true archetype and winner
                true_score = float(scores.get(entry.archetype, 0.0))
                winner_score = float(max(scores.values()))
                if correct:
                    # When correct: distance to runner-up
                    sorted_vals = sorted(scores.values(), reverse=True)
                    runner_up = (
                        float(sorted_vals[1]) if len(sorted_vals) > 1
                        else 0.0
                    )
                    margin = true_score - runner_up
                else:
                    margin = true_score - winner_score  # negative

                # True rank
                sorted_vals = sorted(scores.values(), reverse=True)
                rank = sorted_vals.index(true_score) + 1

                detail = {
                    "name": entry.name,
                    "pdb_id": entry.pdb_id,
                    "chain": entry.chain,
                    "expected": entry.archetype,
                    "predicted": predicted,
                    "correct": correct,
                    "scores": {
                        k: round(float(v), 4)
                        for k, v in scores.items()
                    },
                    "margin": round(float(margin), 4),
                    "rank": rank,
                    "n_residues": pr.get("N", 0),
                    "time_s": round(dt, 1),
                }
                per_protein.append(detail)
                band_cache[entry.name] = pr

                if verbose:
                    mark = "✓" if correct else "✗"
                    frag_label = ""
                    if correct and margin < self.fragile_margin:
                        frag_label = " [FRAGILE]"
                    print(
                        f"{mark} {predicted:16s} "
                        f"margin={margin:+.4f} [{dt:.0f}s]"
                        f"{frag_label}"
                    )

            except Exception as exc:
                dt = time.perf_counter() - t0
                per_protein.append({
                    "name": entry.name,
                    "pdb_id": entry.pdb_id,
                    "chain": entry.chain,
                    "expected": entry.archetype,
                    "predicted": "error",
                    "correct": False,
                    "scores": {},
                    "margin": -1.0,
                    "rank": 0,
                    "n_residues": 0,
                    "time_s": round(dt, 1),
                    "error": str(exc),
                })
                if verbose:
                    print(f"ERROR: {exc}")

        # ── Phase 2: Compute per-archetype stats ──
        valid = [
            pp for pp in per_protein
            if pp["predicted"] != "error"
        ]
        n_correct = sum(1 for pp in valid if pp["correct"])
        n_total = len(valid)

        by_arch: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"correct": 0, "total": 0, "margins": []}
        )
        for pp in valid:
            arch = pp["expected"]
            by_arch[arch]["total"] += 1
            by_arch[arch]["margins"].append(pp["margin"])
            if pp["correct"]:
                by_arch[arch]["correct"] += 1

        per_archetype: Dict[str, Dict[str, Any]] = {}
        for arch in sorted(by_arch):
            d = by_arch[arch]
            per_archetype[arch] = {
                "correct": d["correct"],
                "total": d["total"],
                "accuracy": (
                    d["correct"] / d["total"] if d["total"] else 0
                ),
                "mean_margin": (
                    float(np.mean(d["margins"]))
                    if d["margins"] else 0
                ),
                "min_margin": (
                    float(np.min(d["margins"]))
                    if d["margins"] else 0
                ),
            }

        # ── Phase 3: Fragile / Robust classification ──
        fragile = [
            pp["name"] for pp in valid
            if pp["correct"] and pp["margin"] < self.fragile_margin
        ]
        robust = [
            pp["name"] for pp in valid
            if pp["correct"] and pp["margin"] >= self.fragile_margin
        ]

        # ── Phase 4: Perturbation stability analysis ──
        stability_scores: Optional[Dict[str, float]] = None
        perturbation_accuracy: Optional[float] = None

        if perturbation_pct > 0 and band_cache:
            if verbose:
                print(
                    f"\nPhase 2: Perturbation stability "
                    f"(±{perturbation_pct}%, {n_trials} trials) …"
                )

            stability_scores, perturbation_accuracy = (
                self._perturbation_analysis(
                    band_cache,
                    base_thresholds,
                    perturbation_pct,
                    n_trials,
                    seed,
                    verbose,
                )
            )

        total_time = time.perf_counter() - t_total

        return CrossValidationResult(
            n_proteins=n_total,
            n_correct=n_correct,
            accuracy=n_correct / n_total if n_total else 0,
            per_protein=per_protein,
            per_archetype=per_archetype,
            fragile_proteins=fragile,
            robust_proteins=robust,
            stability_scores=stability_scores,
            n_perturbation_trials=(
                n_trials if perturbation_pct > 0 else 0
            ),
            perturbation_accuracy=perturbation_accuracy,
            total_time_s=round(total_time, 1),
        )

    def _perturbation_analysis(
        self,
        band_cache: Dict[str, Dict],
        base_thresholds: "ThresholdRegistry",
        perturbation_pct: float,
        n_trials: int,
        seed: int,
        verbose: bool,
    ) -> Tuple[Dict[str, float], float]:
        """Run perturbation stability analysis.

        For each trial, jitter ALL thresholds by ±pct%, then re-score
        every cached protein using the perturbed thresholds.  This
        uses ``ThermodynamicBand.rescore_from_profiles`` so only the
        synthesis + lens stack is re-run — no PDB fetching or ENM
        carving — making each trial < 1s for the full corpus.

        Returns
        -------
        stability_scores : dict[str, float]
            Protein name → fraction of trials where classification
            was correct.
        mean_accuracy : float
            Mean accuracy across all trials.
        """
        from .band import ThermodynamicBand

        rng = np.random.RandomState(seed)
        all_keys = list(base_thresholds.keys())
        base_values = np.array(
            [base_thresholds[k] for k in all_keys]
        )

        # Track per-protein correctness across trials
        correct_counts: Dict[str, int] = defaultdict(int)
        trial_accuracies: List[float] = []

        # Pre-extract profiles from band_cache
        protein_data: Dict[str, Tuple] = {}
        for entry in self.corpus:
            if entry.name not in band_cache:
                continue
            pr = band_cache[entry.name]
            br = pr.get("band_result", {})
            profiles = br.get("profiles")
            if profiles is None:
                continue
            protein_data[entry.name] = (
                entry,
                profiles,
                pr.get("evals"),
                pr.get("evecs"),
                pr.get("domain_labels"),
                pr.get("contacts"),
            )

        n_proteins = len(protein_data)
        if n_proteins == 0:
            if verbose:
                print("  No cached profiles for perturbation.")
            return {}, 0.0

        if verbose:
            print(f"  {n_proteins} proteins with cached profiles")

        for trial in range(n_trials):
            # Jitter each threshold independently by ±pct%
            jitter = 1.0 + rng.uniform(
                -perturbation_pct / 100.0,
                perturbation_pct / 100.0,
                size=len(all_keys),
            )
            perturbed_values = base_values * jitter
            overrides = {
                k: float(v)
                for k, v in zip(all_keys, perturbed_values)
            }
            perturbed = base_thresholds.replace(
                overrides, name=f"perturb-{trial}",
            )

            # Re-score each protein with perturbed thresholds
            trial_correct = 0
            for name, (entry, profiles, evals, evecs, dlabels, contacts) in protein_data.items():
                try:
                    identity = ThermodynamicBand.rescore_from_profiles(
                        profiles,
                        evals=evals, evecs=evecs,
                        domain_labels=dlabels, contacts=contacts,
                        thresholds=perturbed,
                    )
                    pred = identity["identity"]
                    if pred == entry.archetype:
                        correct_counts[name] += 1
                        trial_correct += 1
                except Exception:
                    pass  # skip on error

            trial_accuracies.append(trial_correct / n_proteins)

            if verbose and (trial + 1) % 10 == 0:
                mean_so_far = float(np.mean(trial_accuracies))
                print(
                    f"  Trial {trial+1:3d}/{n_trials}: "
                    f"mean acc = {mean_so_far:.1%}"
                )

        stability: Dict[str, float] = {
            name: correct_counts[name] / n_trials
            for name in protein_data
        }

        mean_accuracy = float(np.mean(trial_accuracies))

        return stability, mean_accuracy


# ═══════════════════════════════════════════════════════════════════
# ParameterUsefulnessAnalyzer — heatmap of threshold usefulness
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ParameterUsefulnessResult:
    """Result of per-parameter usefulness analysis.

    Attributes
    ----------
    per_param : dict[str, dict]
        For each threshold key: flip_count, flip_rate, mean_margin_delta,
        affected_proteins, directional sensitivity (+/-).
    sections : dict[str, dict]
        Aggregated stats per threshold section (prefix before '.').
    n_proteins : int
        Number of proteins analysed.
    perturbation_pct : float
        Perturbation percentage used.
    total_params : int
        Total number of threshold parameters.
    zero_impact_params : list[str]
        Parameters that caused zero flips when perturbed.
    high_impact_params : list[str]
        Parameters with flip_rate > 5% (top-impact).
    """

    per_param: Dict[str, Dict[str, Any]]
    sections: Dict[str, Dict[str, Any]]
    n_proteins: int = 0
    perturbation_pct: float = 10.0
    total_params: int = 0
    zero_impact_params: List[str] = field(default_factory=list)
    high_impact_params: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable summary of parameter usefulness."""
        lines = [
            "Parameter Usefulness Report",
            "=" * 60,
            f"Proteins: {self.n_proteins}   "
            f"Params: {self.total_params}   "
            f"Perturbation: ±{self.perturbation_pct}%",
            f"Zero-impact:  {len(self.zero_impact_params)} params "
            f"({100*len(self.zero_impact_params)/self.total_params:.0f}%)"
            if self.total_params else "",
            f"High-impact:  {len(self.high_impact_params)} params",
            "",
        ]

        # Top 20 most impactful parameters
        sorted_params = sorted(
            self.per_param.items(),
            key=lambda x: x[1].get("flip_rate", 0),
            reverse=True,
        )
        lines.append("Top 20 most impactful parameters:")
        for i, (key, info) in enumerate(sorted_params[:20]):
            lines.append(
                f"  {i+1:2d}. {key:40s} "
                f"flip_rate={info['flip_rate']:.1%}  "
                f"flips={info['flip_count']:2d}  "
                f"Δmargin={info['mean_margin_delta']:+.4f}"
            )

        # Per-section summary
        lines.append("")
        lines.append("Per-section summary:")
        for sec in sorted(self.sections):
            info = self.sections[sec]
            lines.append(
                f"  {sec:25s}  "
                f"params={info['n_params']:3d}  "
                f"mean_flip_rate={info['mean_flip_rate']:.1%}  "
                f"zero={info['n_zero']:2d}"
            )

        # Zero-impact params
        if self.zero_impact_params:
            lines.append("")
            lines.append(
                f"Zero-impact parameters "
                f"({len(self.zero_impact_params)}) — candidates for removal:"
            )
            for k in sorted(self.zero_impact_params):
                lines.append(f"  {k}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to JSON-compatible dict."""
        return {
            "n_proteins": self.n_proteins,
            "perturbation_pct": self.perturbation_pct,
            "total_params": self.total_params,
            "n_zero_impact": len(self.zero_impact_params),
            "n_high_impact": len(self.high_impact_params),
            "zero_impact_params": self.zero_impact_params,
            "high_impact_params": self.high_impact_params,
            "per_param": self.per_param,
            "sections": self.sections,
        }

    def heatmap_data(
        self,
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        """Return matrix for heatmap visualisation.

        Returns
        -------
        matrix : np.ndarray
            Shape ``(n_params, 3)`` with columns:
            [flip_rate, mean_margin_delta, directional_bias].
        param_names : list[str]
            Row labels (parameter names), sorted by flip_rate desc.
        metric_names : list[str]
            Column labels.
        """
        sorted_params = sorted(
            self.per_param.items(),
            key=lambda x: x[1].get("flip_rate", 0),
            reverse=True,
        )
        param_names = [k for k, _ in sorted_params]
        metrics = ["flip_rate", "mean_margin_delta", "directional_bias"]
        matrix = np.zeros((len(param_names), 3))
        for i, (_, info) in enumerate(sorted_params):
            matrix[i, 0] = info.get("flip_rate", 0)
            matrix[i, 1] = info.get("mean_margin_delta", 0)
            matrix[i, 2] = info.get("directional_bias", 0)
        return matrix, param_names, metrics

    def save(self, path: str | Path) -> None:
        """Save to JSON file."""
        from .cache import _numpy_safe
        Path(path).write_text(
            json.dumps(_numpy_safe(self.to_dict()), indent=2),
            encoding="utf-8",
        )


class ParameterUsefulnessAnalyzer:
    """Empirical analysis of which threshold parameters matter.

    For each of the ~134 threshold parameters, independently perturb
    it by ±N% while keeping all others fixed, then re-score every
    protein.  Count how many classifications flip.  This reveals:

    - **Zero-impact parameters** — can be removed without affecting
      any classification (candidates for pruning).
    - **High-impact parameters** — control many decisions, likely
      the most important to get right.
    - **Directional bias** — whether increasing vs decreasing the
      parameter has asymmetric effects.

    This uses ``rescore_from_profiles``, so it requires cached band
    results (from a prior CrossValidator or BenchmarkRunner run).

    Quick start
    -----------
    >>> from ibp_enm.benchmark import (
    ...     ParameterUsefulnessAnalyzer, CrossValidator, LARGE_CORPUS
    ... )
    >>> cv = CrossValidator(LARGE_CORPUS)
    >>> result = cv.run()  # get band_cache
    >>> analyzer = ParameterUsefulnessAnalyzer()
    >>> usefulness = analyzer.analyze_from_cv(cv, result)
    >>> print(usefulness.summary())
    >>> usefulness.save("param_usefulness.json")

    Parameters
    ----------
    perturbation_pct : float
        Percentage to perturb each parameter (default 10 = ±10%).
    """

    def __init__(self, perturbation_pct: float = 10.0):
        self.perturbation_pct = perturbation_pct

    def analyze(
        self,
        corpus: List[ProteinEntry],
        band_cache: Dict[str, Dict],
        thresholds: "ThresholdRegistry | None" = None,
        *,
        verbose: bool = False,
    ) -> ParameterUsefulnessResult:
        """Run per-parameter sensitivity analysis.

        Parameters
        ----------
        corpus : list[ProteinEntry]
            The protein corpus (needed for ground truth labels).
        band_cache : dict[str, dict]
            Mapping from protein name to ``run_single_protein`` output
            (must include ``band_result.profiles``, ``evals``, etc.).
        thresholds : ThresholdRegistry, optional
            Base thresholds.  Defaults to ``DEFAULT_THRESHOLDS``.
        verbose : bool
            Print progress.

        Returns
        -------
        ParameterUsefulnessResult
        """
        from .band import ThermodynamicBand
        from .thresholds import DEFAULT_THRESHOLDS

        base = thresholds or DEFAULT_THRESHOLDS
        all_keys = sorted(base.keys())
        n_keys = len(all_keys)

        # Build protein data lookup
        entry_by_name = {e.name: e for e in corpus}
        protein_data: Dict[str, Tuple] = {}
        base_predictions: Dict[str, str] = {}
        base_margins: Dict[str, float] = {}

        for name, pr in band_cache.items():
            if name not in entry_by_name:
                continue
            br = pr.get("band_result", {})
            profiles = br.get("profiles")
            if profiles is None:
                continue
            identity = br.get("identity", {})
            base_predictions[name] = identity.get("identity", "")
            scores = identity.get("scores", {})
            entry = entry_by_name[name]
            true_score = float(scores.get(entry.archetype, 0))
            sorted_vals = sorted(scores.values(), reverse=True)
            if base_predictions[name] == entry.archetype:
                runner_up = float(sorted_vals[1]) if len(sorted_vals) > 1 else 0
                base_margins[name] = true_score - runner_up
            else:
                base_margins[name] = true_score - float(max(scores.values()))

            protein_data[name] = (
                entry,
                profiles,
                pr.get("evals"),
                pr.get("evecs"),
                pr.get("domain_labels"),
                pr.get("contacts"),
            )

        n_proteins = len(protein_data)
        if n_proteins == 0:
            return ParameterUsefulnessResult(
                per_param={}, sections={},
                n_proteins=0,
                perturbation_pct=self.perturbation_pct,
                total_params=n_keys,
            )

        if verbose:
            print(
                f"Parameter usefulness analysis: "
                f"{n_keys} params × {n_proteins} proteins "
                f"(±{self.perturbation_pct}%)"
            )

        per_param: Dict[str, Dict[str, Any]] = {}
        pct = self.perturbation_pct / 100.0

        for k_idx, key in enumerate(all_keys):
            base_val = base[key]
            flip_count = 0
            margin_deltas: List[float] = []
            affected: List[str] = []
            up_flips = 0
            down_flips = 0

            for direction, mult in [("up", 1 + pct), ("down", 1 - pct)]:
                overrides = {key: base_val * mult}
                perturbed = base.replace(overrides, name=f"sens-{key}")

                for name, (entry, profiles, evals, evecs, dlabels, contacts) in protein_data.items():
                    try:
                        identity = ThermodynamicBand.rescore_from_profiles(
                            profiles,
                            evals=evals, evecs=evecs,
                            domain_labels=dlabels, contacts=contacts,
                            thresholds=perturbed,
                        )
                        new_pred = identity["identity"]
                        old_pred = base_predictions[name]
                        if new_pred != old_pred:
                            if direction == "up":
                                up_flips += 1
                            else:
                                down_flips += 1
                            flip_count += 1
                            affected.append(name)

                        # Margin change
                        new_scores = identity.get("scores", {})
                        new_true = float(
                            new_scores.get(entry.archetype, 0)
                        )
                        new_sorted = sorted(
                            new_scores.values(), reverse=True
                        )
                        if new_pred == entry.archetype:
                            new_runner = (
                                float(new_sorted[1])
                                if len(new_sorted) > 1 else 0
                            )
                            new_margin = new_true - new_runner
                        else:
                            new_margin = new_true - float(
                                max(new_scores.values())
                            )
                        margin_deltas.append(
                            new_margin - base_margins.get(name, 0)
                        )

                    except Exception:
                        pass

            # Deduplicate affected (may appear for both up and down)
            unique_affected = sorted(set(affected))
            total_trials = 2 * n_proteins  # up + down
            flip_rate = flip_count / total_trials if total_trials else 0
            mean_margin_delta = (
                float(np.mean(margin_deltas)) if margin_deltas else 0
            )
            # Directional bias: +1 = only up-flips, -1 = only down-flips
            total_flips = up_flips + down_flips
            directional_bias = (
                (up_flips - down_flips) / total_flips
                if total_flips > 0 else 0
            )

            per_param[key] = {
                "flip_count": flip_count,
                "flip_rate": round(flip_rate, 4),
                "mean_margin_delta": round(mean_margin_delta, 6),
                "affected_proteins": unique_affected,
                "up_flips": up_flips,
                "down_flips": down_flips,
                "directional_bias": round(directional_bias, 3),
            }

            if verbose and (k_idx + 1) % 20 == 0:
                print(
                    f"  [{k_idx+1:3d}/{n_keys}] "
                    f"done … {flip_count} flips so far for {key}"
                )

        # ── Aggregate by section ──
        section_data: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "n_params": 0, "total_flips": 0,
                "flip_rates": [], "n_zero": 0,
            }
        )
        for key, info in per_param.items():
            sec = key.split(".")[0]
            section_data[sec]["n_params"] += 1
            section_data[sec]["total_flips"] += info["flip_count"]
            section_data[sec]["flip_rates"].append(info["flip_rate"])
            if info["flip_count"] == 0:
                section_data[sec]["n_zero"] += 1

        sections: Dict[str, Dict[str, Any]] = {}
        for sec, d in sorted(section_data.items()):
            sections[sec] = {
                "n_params": d["n_params"],
                "total_flips": d["total_flips"],
                "mean_flip_rate": float(np.mean(d["flip_rates"])),
                "max_flip_rate": float(np.max(d["flip_rates"])),
                "n_zero": d["n_zero"],
            }

        # Classify parameters
        zero_impact = sorted(
            k for k, v in per_param.items() if v["flip_count"] == 0
        )
        high_impact = sorted(
            k for k, v in per_param.items()
            if v["flip_rate"] > 0.05  # >5% flip rate
        )

        if verbose:
            print(
                f"\nDone: {len(zero_impact)} zero-impact, "
                f"{len(high_impact)} high-impact"
            )

        return ParameterUsefulnessResult(
            per_param=per_param,
            sections=sections,
            n_proteins=n_proteins,
            perturbation_pct=self.perturbation_pct,
            total_params=n_keys,
            zero_impact_params=zero_impact,
            high_impact_params=high_impact,
        )
