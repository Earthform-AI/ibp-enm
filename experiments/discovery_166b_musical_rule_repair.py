#!/usr/bin/env python
"""D166b: Musical Rule Repair — Fix the 0% Globin Catastrophe.

Background
----------
D166 identified 3 structural problems in musical's rule set:

1. **Globin catastrophe**: 0/10 globin proteins correct.  All 10 predicted
   as dumbbell because `mus_dumbbell_scatter_high` (+1.5 at >4.0) dominates
   `mus_globin_scatter_mid` (+0.5 at 1.0-3.5).

2. **Allosteric scatter too broad**: `mus_allosteric_scatter` (+0.8 at >2.0)
   fires on 15/21 LOSTs, creating allosteric bias.

3. **Dumbbell scatter overlap**: `mus_dumbbell_scatter_high` (+1.5 at >4.0)
   catches globin/enzyme proteins whose mode scatter is high due to
   resonance sensitivity, not actual dumbbell structure.

Method
------
Phase 1: Measure musical metrics on all 10 globin proteins.  Find which
    metrics distinguish globin from dumbbell in musical's feature space.

Phase 2: Test rule variants (all 0 new free parameters — we're adjusting
    thresholds/scores within the existing rule infrastructure):
    A. Baseline (current rules)
    B. Boost globin score (+0.5 -> +1.2)
    C. Widen globin scatter range (1.0-3.5 -> 1.0-7.0)
    D. Add new globin rule: entropy_volatility < 0.04 -> globin +0.8
    E. Tighten allosteric scatter threshold (>2.0 -> >3.5)
    F. Combined best: B+C+D+E
    G. Reduce dumbbell scatter score (+1.5 -> +0.8)

Phase 3: Production accuracy check for each variant.

Predictions
-----------
P1: At least one variant gives musical globin accuracy > 30%.
P2: Widening globin range (C) alone flips >=2 globin proteins.
P3: Combined variant (F) achieves musical globin accuracy >= 50%.
P4: No variant causes overall accuracy regression below baseline.
P5: The best variant improves overall accuracy to >= 32/52.

Usage:
    python experiments/discovery_166b_musical_rule_repair.py
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ibp_enm.benchmark import EXPANDED_CORPUS
from ibp_enm.archetypes import ARCHETYPE_EXPECTATIONS
from ibp_enm.synthesis import AlgebraicFickBalancer
from ibp_enm.belief_algebra import ZDPairSelector, FANO_LINES
from ibp_enm.cache import profiles_from_json
from ibp_enm.lens_stack import build_default_stack
from ibp_enm.band import _fetch_ca, build_laplacian
from ibp_enm.analyzer import IBPProteinAnalyzer
from ibp_enm.rules import (
    ArchetypeRule, ARCHETYPE_RULES, apply_rules,
    _lt, _gt, _between,
)

CACHE_DIR = Path.home() / ".ibp_enm_cache"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
ALL_ARCHS = list(ARCHETYPE_EXPECTATIONS.keys())
MUSICAL_IDX = 1

SQRT2 = np.sqrt(2)
STRONG_WEIGHT = SQRT2 / (SQRT2 + 1)
WEAK_WEIGHT = 1.0 / (SQRT2 + 1)
BRIDGE_SCALE = 0.5


# ── Helpers ────────────────────────────────────────────────────

def load_cached_profiles(pdb_id, chain):
    path = CACHE_DIR / f"{pdb_id.upper()}_{chain}.json"
    if not path.exists():
        return None, None
    profiles, metadata = profiles_from_json(path.read_text(encoding="utf-8"))
    return profiles, metadata


def get_structural_data(pdb_id, chain):
    coords, bfactors = _fetch_ca(pdb_id, chain)
    N = len(coords)
    analyzer = IBPProteinAnalyzer()
    result = analyzer.analyze(coords, bfactors)
    contacts, _ = analyzer._build_contacts(coords, N)
    L = build_laplacian(N, contacts)
    evals, evecs = np.linalg.eigh(L)
    return evals, evecs, result.domain_labels, contacts, N


def patch_profiles(profiles, rules):
    """Monkey-patch archetype_vote on each profile to use custom rules.

    AlgebraicFickBalancer.synthesize_identity() calls p.archetype_vote()
    internally with default rules.  We replace archetype_vote on each
    profile instance so it delegates to apply_rules(profile, rules).
    """
    for p in profiles:
        p._orig_vote = p.archetype_vote
        p.archetype_vote = lambda _rules=rules, _p=p: apply_rules(_p, _rules)


def unpatch_profiles(profiles):
    """Restore original archetype_vote on each profile."""
    for p in profiles:
        if hasattr(p, '_orig_vote'):
            p.archetype_vote = p._orig_vote
            del p._orig_vote


def classify_full_pipeline(profiles, carver_votes, meta_state, base_result,
                           evals, evecs, domain_labels, contacts, N,
                           pdb_id, chain):
    """Run full production pipeline: synthesis -> lens stack."""
    alpha_0 = meta_state.get("alpha_0", 0.5)
    alpha_8 = meta_state.get("alpha_8", 0.0)

    consensus_scores = base_result["consensus_scores"]
    disagreement_scores = base_result["disagreement_scores"]
    context_boost = base_result["context_boost"]
    fano_bridge = base_result["fano_bridge"]

    zdp = ZDPairSelector()
    route_scores = {}
    for arch in ALL_ARCHS:
        support = np.zeros(7, dtype=int)
        for i in range(min(len(carver_votes), 7)):
            if max(carver_votes[i], key=carver_votes[i].get) == arch:
                support[i] = 1
        route_scores[arch] = zdp.route_score(support)

    bridge_weight = BRIDGE_SCALE * (1.0 - alpha_0) * alpha_8
    main_weight = 1.0 - bridge_weight
    pre_lens = {}
    for arch in ALL_ARCHS:
        strong = alpha_0 * consensus_scores.get(arch, 0)
        weak = (1 - alpha_0) * disagreement_scores.get(arch, 0)
        main = STRONG_WEIGHT * strong + WEAK_WEIGHT * weak
        rs = route_scores.get(arch, 0)
        bridge = (rs * context_boost.get(arch, 0)
                  + alpha_8 * fano_bridge.get(arch, 0))
        pre_lens[arch] = main_weight * main + bridge_weight * bridge
    total = sum(pre_lens.values())
    if total > 1e-10:
        pre_lens = {k: v / total for k, v in pre_lens.items()}

    stack = build_default_stack(
        evals=evals, evecs=evecs,
        domain_labels=domain_labels, contacts=contacts,
        pdb_id=pdb_id, chain=chain, n_residues=N)
    ctx = {
        "evals": evals, "evecs": evecs,
        "domain_labels": domain_labels, "contacts": contacts,
        "pdb_id": pdb_id, "chain": chain, "n_residues": N,
    }
    final_scores, _traces = stack.apply(pre_lens, profiles, ctx)
    return {"identity": max(final_scores, key=final_scores.get),
            "scores": final_scores}


def classify_with_variant(profiles, variant_rules, balancer,
                          evals, evecs, domain_labels, contacts, N,
                          pdb_id, chain):
    """Classify a protein using a variant rule set through the full pipeline.

    Monkey-patches archetype_vote so synthesize_identity's internal calls
    go through the variant rules.
    """
    try:
        patch_profiles(profiles, variant_rules)
        carver_votes = [p.archetype_vote() for p in profiles]
        meta = balancer.compute_meta_fick_state(carver_votes)
        base_result = balancer.synthesize_identity(profiles, meta)
        result = classify_full_pipeline(
            profiles, carver_votes, meta, base_result,
            evals, evecs, domain_labels, contacts, N,
            pdb_id, chain)
        return result, carver_votes
    finally:
        unpatch_profiles(profiles)


# ── Variant rule builders ──────────────────────────────────────

def build_musical_rules_variant(variant):
    """Build modified ARCHETYPE_RULES list for a given variant."""
    non_musical = [r for r in ARCHETYPE_RULES if r.instrument != "musical"]

    if variant == "A_baseline":
        return non_musical + [r for r in ARCHETYPE_RULES
                              if r.instrument == "musical"]

    elif variant == "B_boost_globin":
        musical = []
        for r in ARCHETYPE_RULES:
            if r.instrument != "musical":
                continue
            if r.name == "mus_globin_scatter_mid":
                musical.append(ArchetypeRule(
                    instrument="musical", archetype="globin",
                    name="mus_globin_scatter_mid",
                    metric="mean_scatter", condition=_between(1.0, 3.5),
                    score=1.2, provenance="D166b boosted from 0.5"))
            else:
                musical.append(r)
        return non_musical + musical

    elif variant == "C_widen_globin":
        musical = []
        for r in ARCHETYPE_RULES:
            if r.instrument != "musical":
                continue
            if r.name == "mus_globin_scatter_mid":
                musical.append(ArchetypeRule(
                    instrument="musical", archetype="globin",
                    name="mus_globin_scatter_wide",
                    metric="mean_scatter", condition=_between(1.0, 7.0),
                    score=0.5, provenance="D166b widened from 3.5 to 7.0"))
            else:
                musical.append(r)
        return non_musical + musical

    elif variant == "D_new_globin_evol":
        musical = [r for r in ARCHETYPE_RULES if r.instrument == "musical"]
        musical.append(ArchetypeRule(
            instrument="musical", archetype="globin",
            name="mus_globin_entropy_vol_low",
            metric="entropy_volatility", condition=_lt(0.04),
            score=0.8, provenance="D166b new globin rule"))
        return non_musical + musical

    elif variant == "E_tighten_allosteric":
        musical = []
        for r in ARCHETYPE_RULES:
            if r.instrument != "musical":
                continue
            if r.name == "mus_allosteric_scatter":
                musical.append(ArchetypeRule(
                    instrument="musical", archetype="allosteric",
                    name="mus_allosteric_scatter",
                    metric="mean_scatter", condition=_gt(3.5),
                    score=0.8, provenance="D166b tightened from 2.0 to 3.5"))
            else:
                musical.append(r)
        return non_musical + musical

    elif variant == "F_combined":
        musical = []
        for r in ARCHETYPE_RULES:
            if r.instrument != "musical":
                continue
            if r.name == "mus_globin_scatter_mid":
                musical.append(ArchetypeRule(
                    instrument="musical", archetype="globin",
                    name="mus_globin_scatter_wide",
                    metric="mean_scatter", condition=_between(1.0, 7.0),
                    score=1.2, provenance="D166b combined: widen+boost"))
            elif r.name == "mus_allosteric_scatter":
                musical.append(ArchetypeRule(
                    instrument="musical", archetype="allosteric",
                    name="mus_allosteric_scatter",
                    metric="mean_scatter", condition=_gt(3.5),
                    score=0.8, provenance="D166b combined: tighten"))
            else:
                musical.append(r)
        musical.append(ArchetypeRule(
            instrument="musical", archetype="globin",
            name="mus_globin_entropy_vol_low",
            metric="entropy_volatility", condition=_lt(0.04),
            score=0.8, provenance="D166b combined: new globin rule"))
        return non_musical + musical

    elif variant == "G_reduce_dumbbell":
        musical = []
        for r in ARCHETYPE_RULES:
            if r.instrument != "musical":
                continue
            if r.name == "mus_dumbbell_scatter_high":
                musical.append(ArchetypeRule(
                    instrument="musical", archetype="dumbbell",
                    name="mus_dumbbell_scatter_high",
                    metric="mean_scatter", condition=_gt(4.0),
                    score=0.8, provenance="D166b reduced from 1.5"))
            else:
                musical.append(r)
        return non_musical + musical

    raise ValueError(f"Unknown variant: {variant}")


# ── Logging ────────────────────────────────────────────────────

def log(msg="", end="\n"):
    print(msg, end=end)

def log_section(title):
    log()
    log("=" * 72)
    log(title)
    log("=" * 72)

def log_subsection(title):
    log()
    log(f"-- {title} " + "-" * max(1, 66 - len(title)))


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    log_section("D166b: Musical Rule Repair")
    log(f"Corpus: {len(EXPANDED_CORPUS)} proteins")

    VARIANTS = [
        "A_baseline", "B_boost_globin", "C_widen_globin",
        "D_new_globin_evol", "E_tighten_allosteric",
        "F_combined", "G_reduce_dumbbell",
    ]

    # ── Load corpus ────────────────────────────────────────────
    log("\nLoading corpus...")
    corpus_data = []
    skipped = []
    for entry in EXPANDED_CORPUS:
        profiles, metadata = load_cached_profiles(entry.pdb_id, entry.chain)
        if profiles is None:
            skipped.append(entry.name)
            continue
        try:
            evals, evecs, domain_labels, contacts, N = \
                get_structural_data(entry.pdb_id, entry.chain)
        except Exception as e:
            skipped.append(f"{entry.name}(struct:{e})")
            continue
        corpus_data.append({
            "entry": entry, "profiles": profiles, "metadata": metadata,
            "evals": evals, "evecs": evecs,
            "domain_labels": domain_labels, "contacts": contacts, "N": N,
        })
    n = len(corpus_data)
    log(f"Loaded: {n} proteins ({len(skipped)} skipped)")

    # ───────────────────────────────────────────────────────────
    # PHASE 1: Musical metrics on globin proteins
    # ───────────────────────────────────────────────────────────
    log_section("Phase 1: Musical Metrics on Globin Proteins")

    globin_data = [d for d in corpus_data
                   if d["entry"].archetype == "globin"]
    dumbbell_data = [d for d in corpus_data
                     if d["entry"].archetype == "dumbbell"]

    metrics = [
        "mean_scatter", "scatter_normalised", "mean_delta_beta",
        "mean_ipr", "entropy_volatility", "mean_spatial_radius",
        "gap_flatness", "gap_volatility", "mean_bus_mass",
    ]

    log(f"\n  {'Protein':<20}", end="")
    for m in metrics:
        log(f" {m[:12]:>12}", end="")
    log()
    log("  " + "-" * (20 + 13 * len(metrics)))

    for d in globin_data:
        prof = d["profiles"][MUSICAL_IDX]
        log(f"  {d['entry'].name:<20}", end="")
        for m in metrics:
            try:
                log(f" {getattr(prof, m):>12.4f}", end="")
            except (AttributeError, TypeError):
                log(f" {'N/A':>12}", end="")
        log()

    log_subsection("Dumbbell comparison (reference)")
    for d in dumbbell_data:
        prof = d["profiles"][MUSICAL_IDX]
        log(f"  {d['entry'].name:<20}", end="")
        for m in metrics:
            try:
                log(f" {getattr(prof, m):>12.4f}", end="")
            except (AttributeError, TypeError):
                log(f" {'N/A':>12}", end="")
        log()

    # Summary stats
    log_subsection("Metric distribution: Globin vs Dumbbell (musical)")
    log(f"  {'Metric':<25} {'Globin mean':>12} {'Dumbbell mean':>14} {'Sep':>6}")
    log("  " + "-" * 60)
    for m in metrics:
        g_vals, d_vals = [], []
        for d in globin_data:
            try:
                g_vals.append(getattr(d["profiles"][MUSICAL_IDX], m))
            except (AttributeError, TypeError):
                pass
        for d in dumbbell_data:
            try:
                d_vals.append(getattr(d["profiles"][MUSICAL_IDX], m))
            except (AttributeError, TypeError):
                pass
        if g_vals and d_vals:
            g_m, d_m = np.mean(g_vals), np.mean(d_vals)
            g_s, d_s = np.std(g_vals), np.std(d_vals)
            sep = abs(g_m - d_m) / (0.5 * (g_s + d_s) + 1e-10)
            log(f"    {m:<23} {g_m:>12.4f} {d_m:>14.4f} {sep:>6.2f}")

    # ───────────────────────────────────────────────────────────
    # PHASE 2: Test rule variants
    # ───────────────────────────────────────────────────────────
    log_section("Phase 2: Rule Variant Testing")

    # Pre-compute baseline predictions
    log("\n  Computing baseline predictions...")
    baseline_rules = build_musical_rules_variant("A_baseline")
    baseline_preds = {}
    for d in corpus_data:
        entry = d["entry"]
        balancer = AlgebraicFickBalancer()
        result, votes = classify_with_variant(
            d["profiles"], baseline_rules, balancer,
            d["evals"], d["evecs"], d["domain_labels"],
            d["contacts"], d["N"], entry.pdb_id, entry.chain)
        baseline_preds[entry.name] = {
            "pred": result["identity"],
            "musical_top": max(votes[MUSICAL_IDX],
                               key=votes[MUSICAL_IDX].get),
        }
    log("  Baseline done.")

    variant_results = {}

    for variant in VARIANTS:
        log_subsection(f"Variant {variant}")

        rules = build_musical_rules_variant(variant)
        correct_total = 0
        musical_correct_total = 0
        musical_correct_globin = 0
        musical_total_globin = 0
        changes = []
        per_arch_musical = {a: {"correct": 0, "total": 0} for a in ALL_ARCHS}

        for d in corpus_data:
            entry = d["entry"]
            truth = entry.archetype

            balancer = AlgebraicFickBalancer()
            result, carver_votes = classify_with_variant(
                d["profiles"], rules, balancer,
                d["evals"], d["evecs"], d["domain_labels"],
                d["contacts"], d["N"], entry.pdb_id, entry.chain)
            pred = result["identity"]

            if pred == truth:
                correct_total += 1

            # Musical's own vote
            musical_top = max(carver_votes[MUSICAL_IDX],
                              key=carver_votes[MUSICAL_IDX].get)
            per_arch_musical[truth]["total"] += 1
            if musical_top == truth:
                musical_correct_total += 1
                per_arch_musical[truth]["correct"] += 1
            if truth == "globin":
                musical_total_globin += 1
                if musical_top == truth:
                    musical_correct_globin += 1

            # Compare to baseline
            if variant != "A_baseline":
                baseline_pred = baseline_preds[entry.name]["pred"]
                if pred != baseline_pred:
                    changes.append({
                        "name": entry.name, "truth": truth,
                        "baseline": baseline_pred, "variant": pred,
                        "gain": pred == truth and baseline_pred != truth,
                        "loss": pred != truth and baseline_pred == truth,
                    })

        musical_acc = 100 * musical_correct_total / n
        musical_globin_acc = (100 * musical_correct_globin /
                              musical_total_globin
                              if musical_total_globin > 0 else 0)
        gains = sum(1 for c in changes if c["gain"])
        losses = sum(1 for c in changes if c["loss"])

        log(f"    Overall accuracy:       {correct_total}/{n} "
            f"({100*correct_total/n:.1f}%)")
        log(f"    Musical accuracy:       {musical_correct_total}/{n} "
            f"({musical_acc:.1f}%)")
        log(f"    Musical globin:         {musical_correct_globin}/"
            f"{musical_total_globin} ({musical_globin_acc:.1f}%)")
        log(f"    Classification changes: {len(changes)} "
            f"(gains={gains}, losses={losses}, net={gains-losses:+d})")

        log(f"    Per-arch musical: ", end="")
        for a in ALL_ARCHS:
            s = per_arch_musical[a]
            pct = 100 * s["correct"] / s["total"] if s["total"] > 0 else 0
            log(f"{a[:4]}={s['correct']}/{s['total']}({pct:.0f}%) ", end="")
        log()

        for c in changes:
            tag = "GAIN" if c["gain"] else "LOSS" if c["loss"] else "SWAP"
            log(f"      {tag}: {c['name']} truth={c['truth']} "
                f"base={c['baseline']} -> {c['variant']}")

        variant_results[variant] = {
            "overall_accuracy": correct_total,
            "musical_accuracy": musical_correct_total,
            "musical_globin_accuracy": musical_correct_globin,
            "musical_globin_total": musical_total_globin,
            "changes": changes, "gains": gains, "losses": losses,
            "net": gains - losses,
            "per_archetype_musical": {
                a: dict(per_arch_musical[a]) for a in ALL_ARCHS},
        }

    # ───────────────────────────────────────────────────────────
    # PHASE 3: Summary & Best Variant
    # ───────────────────────────────────────────────────────────
    log_section("Phase 3: Variant Comparison")

    log(f"\n  {'Variant':<22} {'Accuracy':>9} {'Musical':>8} "
        f"{'Glob':>6} {'Net':>5}")
    log("  " + "-" * 55)
    for v in VARIANTS:
        r = variant_results[v]
        log(f"    {v:<20} {r['overall_accuracy']:>5}/{n:<3} "
            f"{r['musical_accuracy']:>4}/{n:<3} "
            f"{r['musical_globin_accuracy']:>3}/"
            f"{r['musical_globin_total']:<3} "
            f"{r['net']:>+4}")

    best = max(VARIANTS, key=lambda v: (
        variant_results[v]["overall_accuracy"],
        variant_results[v]["musical_accuracy"],
        variant_results[v]["musical_globin_accuracy"],
    ))
    best_r = variant_results[best]
    baseline_r = variant_results["A_baseline"]

    log(f"\n  Best variant: {best}")
    log(f"    Overall: {best_r['overall_accuracy']}/{n} "
        f"(baseline: {baseline_r['overall_accuracy']}/{n})")
    log(f"    Musical: {best_r['musical_accuracy']}/{n} "
        f"(baseline: {baseline_r['musical_accuracy']}/{n})")
    log(f"    Globin:  {best_r['musical_globin_accuracy']}/"
        f"{best_r['musical_globin_total']} "
        f"(baseline: {baseline_r['musical_globin_accuracy']}/"
        f"{baseline_r['musical_globin_total']})")

    # ───────────────────────────────────────────────────────────
    # PHASE 4: Prediction Scorecard
    # ───────────────────────────────────────────────────────────
    log_section("Phase 4: Prediction Scorecard")

    best_globin = max(variant_results[v]["musical_globin_accuracy"]
                      for v in VARIANTS)
    best_globin_total = max(
        variant_results[v]["musical_globin_total"]
        for v in VARIANTS
        if variant_results[v]["musical_globin_total"] > 0)
    best_globin_pct = 100 * best_globin / best_globin_total
    p1 = best_globin_pct > 30
    log(f"\n  P1: Any variant musical globin > 30%")
    log(f"      Best: {best_globin}/{best_globin_total} "
        f"({best_globin_pct:.1f}%) -> "
        f"{'CONFIRMED' if p1 else 'REFUTED'}")

    c_globin = variant_results["C_widen_globin"]["musical_globin_accuracy"]
    a_globin = variant_results["A_baseline"]["musical_globin_accuracy"]
    p2_gain = c_globin - a_globin
    p2 = p2_gain >= 2
    log(f"\n  P2: Variant C (widen) flips >=2 globin")
    log(f"      C globin={c_globin}, baseline={a_globin}, gain={p2_gain} "
        f"-> {'CONFIRMED' if p2 else 'REFUTED'}")

    f_globin = variant_results["F_combined"]["musical_globin_accuracy"]
    f_total = variant_results["F_combined"]["musical_globin_total"]
    f_pct = 100 * f_globin / f_total if f_total > 0 else 0
    p3 = f_pct >= 50
    log(f"\n  P3: Combined (F) musical globin >= 50%")
    log(f"      F: {f_globin}/{f_total} ({f_pct:.1f}%) "
        f"-> {'CONFIRMED' if p3 else 'REFUTED'}")

    min_acc = min(variant_results[v]["overall_accuracy"] for v in VARIANTS)
    p4 = min_acc >= baseline_r["overall_accuracy"]
    log(f"\n  P4: No variant regresses below baseline "
        f"{baseline_r['overall_accuracy']}/{n}")
    log(f"      Minimum accuracy: {min_acc}/{n} "
        f"-> {'CONFIRMED' if p4 else 'REFUTED'}")

    p5 = best_r["overall_accuracy"] >= 32
    log(f"\n  P5: Best variant >= 32/{n}")
    log(f"      Best: {best_r['overall_accuracy']}/{n} "
        f"-> {'CONFIRMED' if p5 else 'REFUTED'}")

    n_confirmed = sum([p1, p2, p3, p4, p5])
    log(f"\n  Score: {n_confirmed}/5 predictions confirmed")

    # ───────────────────────────────────────────────────────────
    # PHASE 5: Recommendation
    # ───────────────────────────────────────────────────────────
    log_section("Phase 5: Recommendation")

    if best_r["overall_accuracy"] > baseline_r["overall_accuracy"]:
        log(f"\n  RECOMMEND: Apply variant {best} to production.")
        log(f"    Accuracy: {baseline_r['overall_accuracy']} -> "
            f"{best_r['overall_accuracy']}")
        log(f"    Musical: {baseline_r['musical_accuracy']} -> "
            f"{best_r['musical_accuracy']}")
        log(f"    Changes: +{best_r['gains']} "
            f"-{best_r['losses']} = net {best_r['net']:+d}")
    elif (best_r["overall_accuracy"] == baseline_r["overall_accuracy"]
          and best_r["musical_accuracy"] > baseline_r["musical_accuracy"]):
        log(f"\n  RECOMMEND: Apply variant {best} -- same accuracy "
            f"but improves musical votes")
        log(f"    Musical: {baseline_r['musical_accuracy']} -> "
            f"{best_r['musical_accuracy']} "
            f"(+{best_r['musical_accuracy'] - baseline_r['musical_accuracy']})")
    else:
        log(f"\n  NO CHANGE: No variant improves accuracy or musical votes.")

    # ── Save ───────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "experiment": "D166b_musical_rule_repair",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "corpus_size": n,
        "variants": variant_results,
        "best_variant": best,
        "predictions": {
            "P1_any_globin_gt_30": {"confirmed": p1,
                                    "value": round(best_globin_pct, 1)},
            "P2_widen_flips_2": {"confirmed": p2, "value": p2_gain},
            "P3_combined_globin_50": {"confirmed": p3,
                                     "value": round(f_pct, 1)},
            "P4_no_regression": {"confirmed": p4, "value": min_acc},
            "P5_best_ge_32": {"confirmed": p5,
                              "value": best_r["overall_accuracy"]},
        },
        "n_confirmed": n_confirmed,
    }
    out_path = RESULTS_DIR / "d166b_musical_rule_repair.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"\n  Results saved to {out_path}")

    elapsed = time.time() - t0
    log(f"\n  Total time: {elapsed:.1f}s")
    return results


if __name__ == "__main__":
    main()
