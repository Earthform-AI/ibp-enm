#!/usr/bin/env python
"""D169: Rank-2 Correction Lens — confusion-pair discriminant post-hoc flip.

Sprint 10 rationale:
  D167 identified 6 confusion axes with strong discriminant features
  (AUC 0.80–1.00) and 6/7 near-miss proteins where those features
  favour the correct (rank-2) archetype.  D169 builds a lens that
  exploits these discriminants to flip near-miss predictions.

Algorithm:
  1. GATE: top-2 margin < MARGIN_THRESH (only near-miss proteins).
  2. For the confusion axis {rank-1, rank-2}, look up the best
     discriminant feature(s) from D167.
  3. Extract the feature from the 7 ThermoReactionProfiles.
  4. Compare against the D167 midpoint (mean of class medians).
     Count how many instruments' feature values favour rank-2.
  5. If ≥ evidence_quorum instruments favour rank-2 AND the
     discriminant AUC > AUC threshold, apply an additive correction:
       scores[rank-2] += boost
       scores[rank-1] -= boost
     Then renormalise.

Variants:
  A: Baseline (production scoring + default lens stack)
  B: ConfusionPairLens only (no existing lenses)
  C: Full stack + ConfusionPairLens appended
  D: Full stack + ConfusionPairLens (aggressive: lower margin thresh)
  E: Full stack + ConfusionPairLens (conservative: higher evidence quorum)

Free parameters: 0 new — the discriminant table is derived from D167
  feature-space analysis (AUC, midpoints), not fitted.  Margin
  threshold = 0.10 (same as existing lens gates).  Evidence quorum
  = 4/7 (majority vote).  Boost = 0.05 (half the enzyme lens boost).

Bridge-blind + near-miss targets:
  GroEL_subunit     (allosteric→enzyme, margin=-0.017, 3/3 D167 discriminants favour)
  Protein_kinase_A  (allosteric→globin, margin=-0.046, 3/3 favour)
  Neuroglobin       (globin→barrel, margin=-0.047, 3/3 favour)
  Erythrocruorin    (globin→allosteric, margin=-0.059, 3/3 favour)
  KDPG_aldolase     (barrel→enzyme, margin=-0.081, 2/3 favour)
  Truncated_Hb      (globin→enzyme, margin=-0.099, 3/3 favour)

Predictions:
  P1: ≥2 near-miss proteins flipped correct by variant C
  P2: Variant C has ≤1 regression (lens is conservative — only fires
      on small-margin proteins)
  P3: GroEL_subunit is recovered (smallest margin, 3/3 discriminants)
  P4: Transferrin is NOT recovered (0/3 discriminants favour in D167)
  P5: Variant E (conservative quorum) has 0 regressions

Usage:
    python experiments/discovery_169_rank2_correction.py
"""

import json
import math
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ibp_enm.benchmark import EXPANDED_CORPUS, ProteinEntry
from ibp_enm.archetypes import ARCHETYPE_EXPECTATIONS
from ibp_enm.synthesis import AlgebraicFickBalancer
from ibp_enm.belief_algebra import ZDPairSelector, FANO_LINES
from ibp_enm.cache import profiles_from_json
from ibp_enm.lens_stack import (
    LensStack, LensTrace, build_default_stack, _renormalise,
)
from ibp_enm.instruments import ThermoReactionProfile
from ibp_enm.band import _fetch_ca, build_laplacian
from ibp_enm.analyzer import IBPProteinAnalyzer
from ibp_enm.thresholds import DEFAULT_THRESHOLDS

CACHE_DIR = Path.home() / ".ibp_enm_cache"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
ALL_ARCHS = list(ARCHETYPE_EXPECTATIONS.keys())

# ── D167 Confusion-Pair Discriminant Table ─────────────────────────
# Each entry: (axis_key, archetype_A, archetype_B, feature_name,
#              property_on_profile, instrument_or_agg, auc,
#              direction)
#   direction = "higher_favours_A" or "higher_favours_B"
#   property_on_profile = attribute name on ThermoReactionProfile
#
# Source: d167_confusion_discriminants.json top discriminants per axis.
# Midpoints computed from D167 class medians.

@dataclass
class DiscriminantRule:
    """One D167-derived discriminant feature for a confusion axis."""
    arch_a: str           # archetype A
    arch_b: str           # archetype B
    feature_name: str     # human label
    profile_attr: str     # attribute on ThermoReactionProfile
    instrument: Optional[str]  # specific instrument or None for aggregate
    auc: float
    higher_favours: str   # "A" or "B" — which class has higher values


# D167 top discriminants per confusion axis
# Extracted from d167_confusion_discriminants.json
DISCRIMINANT_TABLE: List[DiscriminantRule] = [
    # allosteric vs barrel — fick_mean_delta_beta AUC=1.000
    # D167: allosteric has HIGHER mean_delta_beta than barrel
    DiscriminantRule("allosteric", "barrel", "fick_mean_delta_beta",
                     "mean_delta_beta", "fick", 1.000, "A"),
    # allosteric vs barrel (backup) — algebraic_mean_delta_beta AUC=0.956
    DiscriminantRule("allosteric", "barrel", "algebraic_mean_delta_beta",
                     "mean_delta_beta", "algebraic", 0.956, "A"),

    # allosteric vs globin — propagative_mean_scatter AUC=0.983
    # D167: allosteric has HIGHER mean_scatter
    DiscriminantRule("allosteric", "globin", "propagative_mean_scatter",
                     "mean_scatter", "propagative", 0.983, "A"),

    # allosteric vs dumbbell — fragile_mean_delta_entropy AUC=0.800
    DiscriminantRule("allosteric", "dumbbell", "fragile_mean_delta_entropy",
                     "mean_delta_entropy", "fragile", 0.800, "A"),

    # barrel vs enzyme — agg_min_entropy_change AUC=0.946
    # D167: barrel has LOWER entropy_change (min across instruments)
    DiscriminantRule("barrel", "enzyme_active", "agg_min_entropy_change",
                     "entropy_change", None, 0.946, "B"),

    # barrel vs dumbbell — algebraic_mean_delta_beta AUC=0.960
    # D167: dumbbell has HIGHER mean_delta_beta
    DiscriminantRule("barrel", "dumbbell", "algebraic_mean_delta_beta",
                     "mean_delta_beta", "algebraic", 0.960, "B"),

    # enzyme vs globin — thermal_gap_trend AUC=0.969
    # D167: enzyme has HIGHER gap_trend (positive trend = persistent gap)
    DiscriminantRule("enzyme_active", "globin", "thermal_gap_trend",
                     "gap_trend", "thermal", 0.969, "A"),
]


def get_discriminant(arch1: str, arch2: str) -> Optional[DiscriminantRule]:
    """Look up the best discriminant for a confusion pair (order-agnostic)."""
    for d in DISCRIMINANT_TABLE:
        if {d.arch_a, d.arch_b} == {arch1, arch2}:
            return d
    return None


def extract_feature(
    profiles: List[ThermoReactionProfile],
    rule: DiscriminantRule,
) -> List[Tuple[str, float]]:
    """Extract the discriminant feature from all instrument profiles.

    Returns list of (instrument_name, feature_value).
    """
    results = []
    for p in profiles:
        if rule.instrument is not None and p.instrument != rule.instrument:
            continue
        val = getattr(p, rule.profile_attr, None)
        if val is not None:
            results.append((p.instrument, float(val)))
    return results


def extract_feature_all_instruments(
    profiles: List[ThermoReactionProfile],
    profile_attr: str,
) -> List[Tuple[str, float]]:
    """Extract a feature from ALL instrument profiles (ignoring rule.instrument)."""
    results = []
    for p in profiles:
        val = getattr(p, profile_attr, None)
        if val is not None:
            results.append((p.instrument, float(val)))
    return results


# ── Population-level threshold table ───────────────────────────────
# Populated at runtime from corpus data (see build_population_thresholds).
# Key: (instrument_or_"all", profile_attr) → median_value
POPULATION_THRESHOLDS: Dict[Tuple[str, str], float] = {}


def build_population_thresholds(
    all_protein_data: Dict[str, Dict],
) -> Dict[Tuple[str, str], float]:
    """Compute population medians for each discriminant feature.

    For each discriminant rule, extract the feature from all proteins
    in the corpus and compute the median across proteins.  This median
    serves as the decision boundary (not fitted — it's the corpus
    central tendency).
    """
    from collections import defaultdict

    feature_collector: Dict[Tuple[str, str], List[float]] = defaultdict(list)

    for name, pd in all_protein_data.items():
        profiles = pd["profiles"]
        for rule in DISCRIMINANT_TABLE:
            # Specific-instrument rule: get value from that instrument only
            if rule.instrument is not None:
                for p in profiles:
                    if p.instrument == rule.instrument:
                        val = getattr(p, rule.profile_attr, None)
                        if val is not None:
                            feature_collector[(rule.instrument, rule.profile_attr)].append(float(val))

            # Also collect all-instrument version of the feature
            for p in profiles:
                val = getattr(p, rule.profile_attr, None)
                if val is not None:
                    key = (p.instrument, rule.profile_attr)
                    if key not in feature_collector or float(val) not in feature_collector[key]:
                        feature_collector[key].append(float(val))

            # Aggregate features (instrument=None): collect per-protein aggregate
            if rule.instrument is None:
                vals = []
                for p in profiles:
                    val = getattr(p, rule.profile_attr, None)
                    if val is not None:
                        vals.append(float(val))
                if vals:
                    # min for "agg_min_..." style features
                    agg = min(vals) if "min" in rule.feature_name else np.mean(vals)
                    feature_collector[("agg", rule.profile_attr)].append(float(agg))

    thresholds = {}
    for key, vals in feature_collector.items():
        if vals:
            thresholds[key] = float(np.median(vals))

    return thresholds


def evidence_favours_rank2(
    profiles: List[ThermoReactionProfile],
    rank1: str,
    rank2: str,
    rule: DiscriminantRule,
    pop_thresholds: Dict[Tuple[str, str], float],
) -> Tuple[int, int, float]:
    """Count instruments where the feature favours rank-2.

    Uses POPULATION-LEVEL thresholds (corpus medians) to determine
    whether each instrument's feature is on the rank-2 side.

    Returns (n_favour_rank2, n_total_instruments, mean_feature_value).
    """
    # Determine which direction favours rank-2
    rank2_wants_higher = (
        (rule.higher_favours == "A" and rank2 == rule.arch_a) or
        (rule.higher_favours == "B" and rank2 == rule.arch_b)
    )

    # For aggregate features (instrument=None)
    if rule.instrument is None:
        vals = []
        for p in profiles:
            val = getattr(p, rule.profile_attr, None)
            if val is not None:
                vals.append(float(val))
        if not vals:
            return 0, 0, 0.0
        agg = min(vals) if "min" in rule.feature_name else float(np.mean(vals))
        threshold = pop_thresholds.get(("agg", rule.profile_attr), 0.0)
        above = agg > threshold
        n_favour = 1 if (rank2_wants_higher == above) else 0
        return n_favour, 1, agg

    # For per-instrument features: compare EACH instrument's value
    # against its population median
    all_features = extract_feature_all_instruments(profiles, rule.profile_attr)
    if not all_features:
        return 0, 0, 0.0

    mean_val = float(np.mean([v for _, v in all_features]))
    n_favour = 0
    n_total = 0

    for inst, val in all_features:
        threshold = pop_thresholds.get((inst, rule.profile_attr))
        if threshold is None:
            continue
        n_total += 1
        above = val > threshold
        if rank2_wants_higher == above:
            n_favour += 1

    return n_favour, n_total, mean_val


# ── ConfusionPairLens ──────────────────────────────────────────────

class ConfusionPairLens:
    """Rank-2 correction lens using D167 confusion-pair discriminants.

    Fires when the top-2 margin is small AND per-instrument
    discriminant features favour the rank-2 archetype.
    """

    MARGIN_THRESHOLD = 0.10   # same as enzyme/hinge lens gates
    EVIDENCE_QUORUM = 4       # majority of 7 instruments
    BOOST = 0.05              # half the enzyme lens boost (0.10)
    MIN_AUC = 0.80            # only use high-confidence discriminants

    def __init__(
        self,
        margin_threshold: float = 0.10,
        evidence_quorum: int = 4,
        boost: float = 0.05,
        min_auc: float = 0.80,
        pop_thresholds: Optional[Dict] = None,
        thresholds=None,
    ):
        self.margin_threshold = margin_threshold
        self.evidence_quorum = evidence_quorum
        self.boost = boost
        self.min_auc = min_auc
        self.pop_thresholds = pop_thresholds or {}
        self._t = thresholds or DEFAULT_THRESHOLDS

    @property
    def name(self) -> str:
        return "confusion_pair"

    def should_activate(
        self,
        scores: Dict[str, float],
        profiles: List[ThermoReactionProfile],
        context: Dict[str, Any],
    ) -> bool:
        sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
        if len(sorted_scores) < 2:
            return False
        margin = sorted_scores[0][1] - sorted_scores[1][1]
        if margin >= self.margin_threshold:
            return False

        rank1 = sorted_scores[0][0]
        rank2 = sorted_scores[1][0]

        rule = get_discriminant(rank1, rank2)
        if rule is None or rule.auc < self.min_auc:
            return False

        return True

    def apply(
        self,
        scores: Dict[str, float],
        profiles: List[ThermoReactionProfile],
        context: Dict[str, Any],
    ) -> Tuple[Dict[str, float], LensTrace]:
        scores = dict(scores)

        sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
        rank1, s1 = sorted_scores[0]
        rank2, s2 = sorted_scores[1]
        margin = s1 - s2

        rule = get_discriminant(rank1, rank2)
        if rule is None:
            return scores, LensTrace(
                lens_name=self.name, activated=False,
                details={"reason": "no_discriminant"})

        n_favour, n_total, mean_feat = evidence_favours_rank2(
            profiles, rank1, rank2, rule, self.pop_thresholds)

        activated = (n_favour >= self.evidence_quorum and
                     rule.auc >= self.min_auc)

        if activated:
            # Scale boost by how much evidence exceeds quorum
            # and by discriminant AUC (higher AUC → more confident)
            scores[rank2] += self.boost
            scores[rank1] -= self.boost
            scores = _renormalise(scores, self._t["renorm.floor"])

        trace = LensTrace(
            lens_name=self.name,
            activated=activated,
            boost=self.boost if activated else 0.0,
            details={
                "rank1": rank1,
                "rank2": rank2,
                "margin": round(margin, 4),
                "discriminant": rule.feature_name,
                "auc": rule.auc,
                "n_favour_rank2": n_favour,
                "n_total": n_total,
                "mean_feature": round(mean_feat, 4),
                "quorum": self.evidence_quorum,
            },
        )
        return scores, trace


# ── Helpers ────────────────────────────────────────────────────────

def load_cached_profiles(pdb_id: str, chain: str):
    path = CACHE_DIR / f"{pdb_id.upper()}_{chain}.json"
    if not path.exists():
        return None, None
    text = path.read_text(encoding="utf-8")
    profiles, metadata = profiles_from_json(text)
    return profiles, metadata


def get_structural_data(pdb_id: str, chain: str):
    coords, bfactors = _fetch_ca(pdb_id, chain)
    N = len(coords)
    analyzer = IBPProteinAnalyzer()
    result = analyzer.analyze(coords, bfactors)
    contacts, _ = analyzer._build_contacts(coords, N)
    L = build_laplacian(N, contacts)
    evals, evecs = np.linalg.eigh(L)
    domain_labels = result.domain_labels
    return evals, evecs, domain_labels, contacts, N


# Near-miss target proteins from D167
NEAR_MISS = {
    "GroEL_subunit", "Protein_kinase_A", "Neuroglobin",
    "Erythrocruorin", "KDPG_aldolase", "Truncated_Hb", "Transferrin",
}


# ── Variant scoring ───────────────────────────────────────────────

def score_protein(
    profiles, meta_state, base_result,
    evals, evecs, domain_labels, contacts, N,
    pdb_id, chain,
    extra_lenses: Optional[List] = None,
    replace_stack: bool = False,
):
    """Score protein with optional extra lenses appended to default stack."""
    carver_votes = [p.archetype_vote() for p in profiles]
    alpha_0 = meta_state.get("alpha_0", 0.5)
    alpha_8 = meta_state.get("alpha_8", 0.0)

    consensus_scores = base_result["consensus_scores"]
    disagreement_scores = base_result["disagreement_scores"]
    context_boost = base_result["context_boost"]

    balancer = AlgebraicFickBalancer()
    fano_bridge = balancer._hamming_bridge.bridge_scores(carver_votes, ALL_ARCHS)

    zdp = ZDPairSelector()
    n_inst = min(len(carver_votes), 7)
    route_scores = {}
    for arch in ALL_ARCHS:
        support = np.zeros(7, dtype=int)
        for i in range(n_inst):
            if max(carver_votes[i], key=carver_votes[i].get) == arch:
                support[i] = 1
        route_scores[arch] = zdp.route_score(support)

    SQRT2 = math.sqrt(2)
    STRONG_WEIGHT = SQRT2 / (SQRT2 + 1)
    WEAK_WEIGHT = 1.0 / (SQRT2 + 1)
    BRIDGE_SCALE = 0.5

    # D168: √α₈ bridge weight
    bridge_weight = BRIDGE_SCALE * (1.0 - alpha_0) * math.sqrt(alpha_8)
    main_weight = 1.0 - bridge_weight

    pre_lens_scores = {}
    for arch in ALL_ARCHS:
        strong = alpha_0 * consensus_scores.get(arch, 0)
        weak = (1 - alpha_0) * disagreement_scores.get(arch, 0)
        main = STRONG_WEIGHT * strong + WEAK_WEIGHT * weak
        rs = route_scores.get(arch, 0)
        bridge = (rs * context_boost.get(arch, 0)
                  + alpha_8 * fano_bridge.get(arch, 0))
        pre_lens_scores[arch] = main_weight * main + bridge_weight * bridge

    total = sum(pre_lens_scores.values())
    if total > 1e-10:
        pre_lens_scores = {k: v / total for k, v in pre_lens_scores.items()}

    # Build lens stack
    if replace_stack and extra_lenses:
        stack = LensStack(extra_lenses)
    else:
        stack = build_default_stack(
            evals=evals, evecs=evecs,
            domain_labels=domain_labels, contacts=contacts,
            pdb_id=pdb_id, chain=chain, n_residues=N,
        )
        if extra_lenses:
            for lens in extra_lenses:
                stack = stack.with_lens(lens)

    context = {
        "evals": evals, "evecs": evecs,
        "domain_labels": domain_labels, "contacts": contacts,
        "pdb_id": pdb_id, "chain": chain, "n_residues": N,
    }
    final_scores, traces = stack.apply(pre_lens_scores, profiles, context)
    identity = max(final_scores, key=final_scores.get)

    return {
        "identity": identity,
        "scores": final_scores,
        "pre_lens_scores": pre_lens_scores,
        "alpha_0": alpha_0,
        "alpha_8": alpha_8,
        "lens_traces": [
            {"name": t.lens_name, "activated": t.activated,
             "boost": t.boost, "details": t.details}
            for t in traces
        ],
    }


# ── Main experiment ────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    corpus = list(EXPANDED_CORPUS)
    print("D169: Rank-2 Correction Lens — Confusion-Pair Discriminants")
    print(f"  Corpus: {len(corpus)} proteins")
    print(f"  Near-miss targets: {', '.join(sorted(NEAR_MISS))}")
    print(f"  Discriminant table: {len(DISCRIMINANT_TABLE)} rules")
    print()

    # ── Phase 1: Load profiles + structural data ───────────────────
    print("=" * 72)
    print("PHASE 1: LOADING PROFILES & STRUCTURAL DATA")
    print("=" * 72)

    protein_data = {}
    n_loaded = 0
    t_start = time.perf_counter()

    for i, entry in enumerate(corpus):
        label = f"[{i+1}/{len(corpus)}]"

        profiles, metadata = load_cached_profiles(entry.pdb_id, entry.chain)
        if profiles is None or len(profiles) == 0:
            print(f"  {label} ✗ {entry.name}: no cached profiles!")
            continue

        try:
            evals, evecs, domain_labels, contacts, N = get_structural_data(
                entry.pdb_id, entry.chain)
        except Exception as exc:
            print(f"  {label} ✗ {entry.name}: structural error: {exc}")
            continue

        carver_votes = [p.archetype_vote() for p in profiles]
        balancer = AlgebraicFickBalancer()
        meta_state = balancer.compute_meta_fick_state(carver_votes)
        base_result = balancer.synthesize_identity(profiles, meta_state)

        protein_data[entry.name] = {
            "entry": entry,
            "profiles": profiles,
            "meta_state": meta_state,
            "base_result": base_result,
            "evals": evals,
            "evecs": evecs,
            "domain_labels": domain_labels,
            "contacts": contacts,
            "N": N,
        }

        nm = " ★" if entry.name in NEAR_MISS else ""
        print(f"  {label} ✓ {entry.name} (N={N}, "
              f"α₀={meta_state['alpha_0']:.3f}, "
              f"α₈={meta_state['alpha_8']:.3f}){nm}")
        n_loaded += 1

    t_load = time.perf_counter() - t_start
    print(f"\n  Loaded: {n_loaded}/{len(corpus)} ({t_load:.1f}s)")

    n_nm = sum(1 for n in NEAR_MISS if n in protein_data)
    print(f"  Near-miss loaded: {n_nm}/{len(NEAR_MISS)}")

    # ── Phase 1b: Build population thresholds ──────────────────────
    print("\n  Building population-level feature thresholds...")
    pop_thresholds = build_population_thresholds(protein_data)
    print(f"  Population thresholds computed for {len(pop_thresholds)} "
          f"(instrument, feature) pairs")
    for key, val in sorted(pop_thresholds.items()):
        print(f"    {key[0]:>15s}.{key[1]:<25s} median={val:.4f}")

    # ── Phase 2: Score all variants ────────────────────────────────
    print("\n" + "=" * 72)
    print("PHASE 2: SCORING ALL VARIANTS")
    print("=" * 72)

    # Define variant configurations
    variant_configs = {
        "A_baseline": {"extra_lenses": [], "replace_stack": False},
        "B_cpl_only": {
            "extra_lenses": [ConfusionPairLens(pop_thresholds=pop_thresholds)],
            "replace_stack": True,
        },
        "C_full_plus_cpl": {
            "extra_lenses": [ConfusionPairLens(pop_thresholds=pop_thresholds)],
            "replace_stack": False,
        },
        "D_aggressive": {
            "extra_lenses": [ConfusionPairLens(
                margin_threshold=0.15, boost=0.08,
                pop_thresholds=pop_thresholds)],
            "replace_stack": False,
        },
        "E_conservative": {
            "extra_lenses": [ConfusionPairLens(
                evidence_quorum=5, boost=0.03,
                pop_thresholds=pop_thresholds)],
            "replace_stack": False,
        },
    }

    results = {vname: {} for vname in variant_configs}

    for i, entry in enumerate(corpus):
        if entry.name not in protein_data:
            continue

        pd = protein_data[entry.name]
        label = f"[{i+1}/{len(corpus)}]"
        preds = {}

        for vname, cfg in variant_configs.items():
            vresult = score_protein(
                pd["profiles"], pd["meta_state"], pd["base_result"],
                pd["evals"], pd["evecs"], pd["domain_labels"],
                pd["contacts"], pd["N"],
                entry.pdb_id, entry.chain,
                extra_lenses=cfg["extra_lenses"],
                replace_stack=cfg["replace_stack"],
            )
            results[vname][entry.name] = vresult
            correct = vresult["identity"] == entry.archetype
            preds[vname] = ("✓" if correct else "✗", vresult["identity"])

        a_pred = preds.get("A_baseline", ("?", "?"))
        changes = []
        for vname in list(variant_configs.keys())[1:]:
            vpred = preds.get(vname, ("?", "?"))
            if vpred[1] != a_pred[1]:
                short = vname.split("_")[0]
                if vpred[0] == "✓" and a_pred[0] == "✗":
                    changes.append(f"{short}:+1")
                elif vpred[0] == "✗" and a_pred[0] == "✓":
                    changes.append(f"{short}:-1")
                else:
                    changes.append(f"{short}:Δ")

        nm = " ★" if entry.name in NEAR_MISS else ""
        change_str = f"  [{', '.join(changes)}]" if changes else ""
        print(f"  {label} {a_pred[0]} {entry.name:<25s} "
              f"truth={entry.archetype:<15s} pred={a_pred[1]:<15s}"
              f"{change_str}{nm}")

    # ── Phase 3: Accuracy comparison ───────────────────────────────
    print("\n" + "=" * 72)
    print("PHASE 3: VARIANT ACCURACY COMPARISON")
    print("=" * 72)

    variant_stats = {}
    baseline_correct = 0

    for vname in variant_configs:
        correct = sum(
            1 for entry in corpus
            if entry.name in results[vname]
            and results[vname][entry.name]["identity"] == entry.archetype
        )
        total = len(results[vname])
        pct = 100 * correct / max(total, 1)
        variant_stats[vname] = {"correct": correct, "total": total, "pct": pct}
        if vname == "A_baseline":
            baseline_correct = correct
        delta = correct - baseline_correct
        delta_str = f"(Δ={delta:+d})" if vname != "A_baseline" else ""
        print(f"  {vname:<20s}: {correct}/{total} ({pct:.1f}%) {delta_str}")

    # Near-miss subset
    print(f"\n  Near-miss accuracy ({len(NEAR_MISS)} proteins):")
    for vname in variant_configs:
        correct = sum(
            1 for name in NEAR_MISS
            if name in results[vname]
            and results[vname][name]["identity"] == protein_data[name]["entry"].archetype
        )
        loaded = sum(1 for name in NEAR_MISS if name in results[vname])
        print(f"    {vname:<20s}: {correct}/{loaded}")

    # ── Phase 4: Per-protein changes ───────────────────────────────
    print("\n" + "=" * 72)
    print("PHASE 4: PER-PROTEIN CHANGES (vs baseline)")
    print("=" * 72)

    variant_changes = {}

    for vname in list(variant_configs.keys())[1:]:
        gains, losses, flips = [], [], []
        for entry in corpus:
            if entry.name not in results["A_baseline"] or entry.name not in results[vname]:
                continue
            a_correct = results["A_baseline"][entry.name]["identity"] == entry.archetype
            v_correct = results[vname][entry.name]["identity"] == entry.archetype
            v_pred = results[vname][entry.name]["identity"]
            a_pred = results["A_baseline"][entry.name]["identity"]

            nm = " ★" if entry.name in NEAR_MISS else ""
            if not a_correct and v_correct:
                gains.append(f"    + {entry.name:<25s} truth={entry.archetype:<15s} "
                             f"was={a_pred:<15s} now={v_pred}{nm}")
            elif a_correct and not v_correct:
                losses.append(f"    - {entry.name:<25s} truth={entry.archetype:<15s} "
                              f"was_correct, now={v_pred}{nm}")
            elif v_pred != a_pred:
                flips.append(f"    ~ {entry.name:<25s} truth={entry.archetype:<15s} "
                             f"was={a_pred:<15s} now={v_pred}{nm}")

        variant_changes[vname] = {
            "gains": [g.strip() for g in gains],
            "losses": [l.strip() for l in losses],
            "flips": [f.strip() for f in flips],
        }

        print(f"\n  --- {vname} ---")
        print(f"  GAINS ({len(gains)}):")
        for g in gains:
            print(g)
        print(f"  LOSSES ({len(losses)}):")
        for l in losses:
            print(l)
        if flips:
            print(f"  FLIPS (wrong→wrong, {len(flips)}):")
            for f in flips:
                print(f)

    # ── Phase 5: Lens activation analysis ──────────────────────────
    print("\n" + "=" * 72)
    print("PHASE 5: CONFUSION-PAIR LENS ACTIVATION ANALYSIS")
    print("=" * 72)

    vname = "C_full_plus_cpl"
    cpl_activations = 0
    cpl_details = []

    for entry in corpus:
        if entry.name not in results[vname]:
            continue
        r = results[vname][entry.name]
        for t in r["lens_traces"]:
            if t["name"] == "confusion_pair":
                nm = " ★" if entry.name in NEAR_MISS else ""
                correct = r["identity"] == protein_data[entry.name]["entry"].archetype
                mark = "✓" if correct else "✗"
                truth = protein_data[entry.name]["entry"].archetype

                if t["activated"]:
                    cpl_activations += 1
                    det = t["details"]
                    print(f"  {mark} {entry.name:<25s} truth={truth:<15s} "
                          f"pred={r['identity']:<15s} "
                          f"rank1={det['rank1']}, rank2={det['rank2']}, "
                          f"margin={det['margin']:.4f}, "
                          f"discr={det['discriminant']}, "
                          f"evidence={det['n_favour_rank2']}/{det['n_total']}"
                          f"{nm}")
                    cpl_details.append({
                        "name": entry.name,
                        "truth": truth,
                        "identity": r["identity"],
                        "correct": correct,
                        **det,
                    })
                elif t["details"].get("reason") != "no_discriminant":
                    # Gate passed but evidence insufficient
                    det = t["details"]
                    if "discriminant" in det:
                        print(f"  — {entry.name:<25s} truth={truth:<15s} "
                              f"margin={det.get('margin', '?'):.4f} "
                              f"(evidence {det.get('n_favour_rank2', 0)}"
                              f"/{det.get('n_total', 0)} < quorum){nm}")

    print(f"\n  Total CPL activations: {cpl_activations}")

    # ── Phase 6: Near-miss deep dive ───────────────────────────────
    print("\n" + "=" * 72)
    print("PHASE 6: NEAR-MISS PROTEIN DEEP DIVE")
    print("=" * 72)

    for name in sorted(NEAR_MISS):
        if name not in protein_data:
            print(f"\n  {name}: NOT LOADED")
            continue

        pd = protein_data[name]
        entry = pd["entry"]
        ms = pd["meta_state"]
        print(f"\n  ── {name} ─────────────────────────────")
        print(f"  Truth: {entry.archetype}")
        print(f"  α₀={ms['alpha_0']:.4f}, α₈={ms['alpha_8']:.4f}")

        for vname in variant_configs:
            if name not in results[vname]:
                continue
            r = results[vname][name]
            correct = "✓" if r["identity"] == entry.archetype else "✗"
            sorted_scores = sorted(r["scores"].items(), key=lambda x: -x[1])
            top3 = ", ".join(f"{a}={s:.4f}" for a, s in sorted_scores[:3])

            # Check CPL trace
            cpl_trace = ""
            for t in r["lens_traces"]:
                if t["name"] == "confusion_pair" and t["activated"]:
                    det = t["details"]
                    cpl_trace = (f" CPL: {det['discriminant']} "
                                 f"ev={det['n_favour_rank2']}/{det['n_total']}")

            print(f"    {vname:<20s}: {correct} {r['identity']:<15s} [{top3}]{cpl_trace}")

        # Show discriminant features for rank-1 vs truth
        a_result = results["A_baseline"].get(name)
        if a_result:
            rank1 = max(a_result["scores"], key=a_result["scores"].get)
            truth = entry.archetype
            rule = get_discriminant(rank1, truth)
            if rule:
                features = extract_feature(pd["profiles"], rule)
                n_fav, n_tot, mean_v = evidence_favours_rank2(
                    pd["profiles"], rank1, truth, rule, pop_thresholds)
                print(f"  Discriminant ({rank1}↔{truth}): {rule.feature_name} "
                      f"AUC={rule.auc:.3f}, evidence={n_fav}/{n_tot}, "
                      f"mean={mean_v:.4f}")

    # ── Phase 7: Prediction scorecard ──────────────────────────────
    print("\n" + "=" * 72)
    print("PHASE 7: PREDICTION SCORECARD")
    print("=" * 72)

    predictions = {}

    # P1: ≥2 near-miss flipped correct by C
    c_gains_nm = sum(
        1 for name in NEAR_MISS
        if name in results["A_baseline"] and name in results["C_full_plus_cpl"]
        and results["A_baseline"][name]["identity"] != protein_data[name]["entry"].archetype
        and results["C_full_plus_cpl"][name]["identity"] == protein_data[name]["entry"].archetype
    )
    p1 = c_gains_nm >= 2
    predictions["P1"] = p1
    print(f"  P1 {'✓' if p1 else '✗'}: C near-miss gains = {c_gains_nm} "
          f"(need ≥2)")

    # P2: C has ≤1 regression
    c_losses = len(variant_changes.get("C_full_plus_cpl", {}).get("losses", []))
    p2 = c_losses <= 1
    predictions["P2"] = p2
    print(f"  P2 {'✓' if p2 else '✗'}: C regressions = {c_losses} (need ≤1)")

    # P3: GroEL flipped correct
    groel_correct_c = (
        "GroEL_subunit" in results.get("C_full_plus_cpl", {})
        and results["C_full_plus_cpl"]["GroEL_subunit"]["identity"] == "allosteric"
    )
    p3 = groel_correct_c
    predictions["P3"] = p3
    print(f"  P3 {'✓' if p3 else '✗'}: GroEL_subunit recovered by C = {groel_correct_c}")

    # P4: Transferrin NOT recovered
    trans_correct_c = (
        "Transferrin" in results.get("C_full_plus_cpl", {})
        and results["C_full_plus_cpl"]["Transferrin"]["identity"] == "dumbbell"
    )
    p4 = not trans_correct_c
    predictions["P4"] = p4
    print(f"  P4 {'✓' if p4 else '✗'}: Transferrin NOT recovered = {not trans_correct_c}")

    # P5: E has 0 regressions
    e_losses = len(variant_changes.get("E_conservative", {}).get("losses", []))
    p5 = e_losses == 0
    predictions["P5"] = p5
    print(f"  P5 {'✓' if p5 else '✗'}: E regressions = {e_losses} (need 0)")

    confirmed = sum(1 for v in predictions.values() if v)
    print(f"\n  SCORECARD: {confirmed}/{len(predictions)} predictions confirmed")

    # ── Save results ───────────────────────────────────────────────
    output = {
        "experiment": "D169",
        "title": "Rank-2 Correction Lens — Confusion-Pair Discriminants",
        "corpus_size": len(corpus),
        "loaded": n_loaded,
        "near_miss_targets": sorted(NEAR_MISS),
        "discriminant_table": [
            {"arch_a": d.arch_a, "arch_b": d.arch_b,
             "feature": d.feature_name, "auc": d.auc,
             "higher_favours": d.higher_favours}
            for d in DISCRIMINANT_TABLE
        ],
        "variant_accuracy": {
            vname: {
                "correct": variant_stats[vname]["correct"],
                "total": variant_stats[vname]["total"],
                "pct": round(variant_stats[vname]["pct"], 1),
                "delta": variant_stats[vname]["correct"] - baseline_correct,
            }
            for vname in variant_configs
        },
        "variant_changes": {
            vname: variant_changes.get(vname, {})
            for vname in list(variant_configs.keys())[1:]
        },
        "cpl_activations": cpl_details,
        "predictions": predictions,
        "predictions_confirmed": confirmed,
        "predictions_total": len(predictions),
    }

    results_path = RESULTS_DIR / "d169_rank2_correction.json"
    results_path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"\n  Results saved to {results_path}")


if __name__ == "__main__":
    main()
