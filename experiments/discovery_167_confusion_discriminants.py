#!/usr/bin/env python
"""D167: Confusion-Pair Discriminants.

Background
----------
32/52 (61.5%) accuracy with 20 wrong predictions.  The errors scatter
across multiple confusion axes.  D167 systematically identifies which
features best discriminate each confused archetype pair, computing
ROC-AUC on the raw thermodynamic features from per-instrument metadata.

Method
------
Phase 1: Build feature matrix from cached per-instrument metadata:
    52 proteins × (N_features × 7 instruments + cross-instrument aggregates).

Phase 2: For each confusion axis (A↔B where ≥2 errors cross the boundary),
    compute AUC of every feature using only proteins of type A or B.
    Report top-10 features per axis.

Phase 3: Near-miss analysis.  For proteins with margin < 0.10,
    check whether the top discriminant features favour the correct archetype.

Phase 4: Aggregate: which features appear in the top-5 across multiple
    axes?  These are candidates for new lens rules in D169.

Predictions
-----------
P1: At least one confusion axis has a feature with AUC > 0.85.
P2: Cross-instrument aggregate features (mean/std) outperform per-instrument
    features on at least 2 confusion axes.
P3: For ≥ 2 of the near-miss proteins, the top discriminant correctly
    separates them from the predicted class.
P4: gap_flatness or scatter_normalised appears in top-5 on ≥ 3 axes.
P5: A previously unused feature (not in current rules) appears in the
    top-3 on at least one axis (potential new rule discovery).

Usage:
    python experiments/discovery_167_confusion_discriminants.py
"""

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ibp_enm.benchmark import EXPANDED_CORPUS
from ibp_enm.archetypes import ARCHETYPE_EXPECTATIONS
from ibp_enm.algebra import INSTRUMENT_NAMES

CACHE_DIR = Path.home() / ".ibp_enm_cache"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
ALL_ARCHS = list(ARCHETYPE_EXPECTATIONS.keys())

# Features available in per-instrument metadata
# Map from our feature name -> key in per_instrument dict
FEATURE_MAP = {
    "gap_retained": "gap_retained",
    "gap_volatility": "gap_volatility",
    "gap_trend": "gap_trend",
    "gap_flatness": "gap_flatness",
    "reversible_frac": "reversible_frac",
    "species_entropy": "species_entropy",
    "mean_scatter": "mean_scatter",
    "entropy_change": "entropy_change",
    "entropy_volatility": "entropy_volatility",
    "mean_delta_entropy": "mean_delta_S",  # renamed in cache
    "heat_cap_change": "heat_cap_change",
    "free_energy_cost": "free_energy_cost",
    "mean_spatial_radius": "mean_spatial_radius",
    "max_spatial_radius": "max_spatial_radius",
    "mean_delta_beta": "mean_delta_beta",
    "mean_bus_mass": "mean_bus_mass",
    "mean_ipr": "mean_ipr",
}

SCALAR_FEATURES = list(FEATURE_MAP.keys())

# Features currently used in rules (to detect novel discoveries)
FEATURES_IN_RULES = {
    "scatter_normalised", "mean_scatter", "mean_delta_beta",
    "mean_ipr", "entropy_volatility", "mean_spatial_radius",
    "gap_flatness", "mean_bus_mass", "reversible_frac",
}


def load_protein_data(pdb_id, chain):
    """Load cached per-instrument features and identity result."""
    path = CACHE_DIR / f"{pdb_id.upper()}_{chain}.json"
    if not path.exists():
        return None
    with open(path) as f:
        payload = json.load(f)
    meta = payload.get("metadata", {})
    per_inst = meta.get("per_instrument", {})
    identity = meta.get("identity_result", {})
    if not per_inst or not identity:
        return None
    return {
        "per_instrument": per_inst,
        "identity": identity.get("identity", "unknown"),
        "scores": identity.get("scores", {}),
        "N": meta.get("N", 0),
    }


def compute_auc(values_a, values_b):
    """AUC via Mann-Whitney U statistic.

    Returns max(AUC, 1-AUC) and direction.
    """
    na, nb = len(values_a), len(values_b)
    if na == 0 or nb == 0:
        return 0.5, "none"
    count = 0
    for a in values_a:
        for b in values_b:
            if a > b:
                count += 1
            elif a == b:
                count += 0.5
    raw_auc = count / (na * nb)
    if raw_auc >= 0.5:
        return raw_auc, "A>B"
    else:
        return 1 - raw_auc, "B>A"


def cohens_d(values_a, values_b):
    """Cohen's d effect size."""
    if len(values_a) < 2 or len(values_b) < 2:
        return 0.0
    ma, mb = np.mean(values_a), np.mean(values_b)
    sa, sb = np.std(values_a, ddof=1), np.std(values_b, ddof=1)
    pooled = np.sqrt((sa**2 + sb**2) / 2)
    if pooled < 1e-10:
        return 0.0
    return abs(ma - mb) / pooled


def log(msg="", end="\n"):
    print(msg, end=end, flush=True)


def log_section(title):
    log()
    log("=" * 72)
    log(title)
    log("=" * 72)


def log_subsection(title):
    log()
    log(f"-- {title} " + "-" * max(1, 66 - len(title)))


def main():
    t0 = time.time()
    log_section("D167: Confusion-Pair Discriminants")
    log(f"Corpus: {len(EXPANDED_CORPUS)} proteins")
    log(f"Features: {len(SCALAR_FEATURES)} scalars x 7 instruments + aggregates")

    # -- Load corpus from cached metadata ----------------------------
    log("\nLoading corpus from cache...")
    corpus = []
    skipped = []
    for entry in EXPANDED_CORPUS:
        data = load_protein_data(entry.pdb_id, entry.chain)
        if data is None:
            skipped.append(entry.name)
            continue

        pred = data["identity"]
        scores = data["scores"]
        truth = entry.archetype
        margin = scores.get(truth, 0) - max(
            v for k, v in scores.items() if k != truth)

        corpus.append({
            "entry": entry,
            "per_instrument": data["per_instrument"],
            "truth": truth,
            "predicted": pred,
            "scores": scores,
            "correct": pred == truth,
            "margin": margin,
            "N": data["N"],
        })

    n = len(corpus)
    n_correct = sum(1 for d in corpus if d["correct"])
    log(f"Loaded: {n} proteins, {n_correct}/{n} correct "
        f"({100*n_correct/n:.1f}%), {len(skipped)} skipped")

    # -- Build feature matrix ----------------------------------------
    log("\nBuilding feature matrix...")

    instrument_names = list(INSTRUMENT_NAMES[:7])
    feature_names = []
    for inst_name in instrument_names:
        for feat in SCALAR_FEATURES:
            feature_names.append(f"{inst_name}_{feat}")

    # Cross-instrument aggregates
    for stat in ["mean", "std", "min", "max"]:
        for feat in SCALAR_FEATURES:
            feature_names.append(f"agg_{stat}_{feat}")

    n_features = len(feature_names)
    log(f"Feature matrix: {n} x {n_features}")

    feature_matrix = np.zeros((n, n_features))
    for i, d in enumerate(corpus):
        pi = d["per_instrument"]
        col = 0
        per_inst_values = {feat: [] for feat in SCALAR_FEATURES}

        for inst_name in instrument_names:
            inst_data = pi.get(inst_name, {})
            for feat in SCALAR_FEATURES:
                cache_key = FEATURE_MAP[feat]
                val = inst_data.get(cache_key, 0.0)
                try:
                    val = float(val)
                except (TypeError, ValueError):
                    val = 0.0
                if not np.isfinite(val):
                    val = 0.0
                feature_matrix[i, col] = val
                per_inst_values[feat].append(val)
                col += 1

        # Aggregates
        for stat in ["mean", "std", "min", "max"]:
            for feat in SCALAR_FEATURES:
                vals = per_inst_values[feat]
                if stat == "mean":
                    feature_matrix[i, col] = np.mean(vals) if vals else 0
                elif stat == "std":
                    feature_matrix[i, col] = np.std(vals) if len(vals) > 1 else 0
                elif stat == "min":
                    feature_matrix[i, col] = np.min(vals) if vals else 0
                elif stat == "max":
                    feature_matrix[i, col] = np.max(vals) if vals else 0
                col += 1

    log(f"Feature matrix built: shape {feature_matrix.shape}")

    # -- Confusion matrix --------------------------------------------
    log_section("Phase 1: Confusion Matrix")

    confusion = defaultdict(list)
    for d in corpus:
        if not d["correct"]:
            axis = f"{d['truth']} -> {d['predicted']}"
            confusion[axis].append(d["entry"].name)

    log(f"\n  {'Axis':<30} {'Count':>5}  Proteins")
    log("  " + "-" * 70)
    for axis in sorted(confusion, key=lambda x: -len(confusion[x])):
        names = ", ".join(confusion[axis])
        log(f"  {axis:<30} {len(confusion[axis]):>5}  {names}")

    # -- Identify confusion axes with >= 2 crossings -----------------
    pair_crossings = defaultdict(list)
    for d in corpus:
        if not d["correct"]:
            pair = tuple(sorted([d["truth"], d["predicted"]]))
            pair_crossings[pair].append(d)

    directional_axes = defaultdict(list)
    for d in corpus:
        if not d["correct"]:
            directional_axes[(d["truth"], d["predicted"])].append(d)

    axes_to_test = {}
    for pair, items in pair_crossings.items():
        if len(items) >= 2:
            axes_to_test[pair] = items
    for (truth, pred), items in directional_axes.items():
        pair = tuple(sorted([truth, pred]))
        if len(items) >= 2 and pair not in axes_to_test:
            axes_to_test[pair] = pair_crossings[pair]

    log(f"\n  Confusion axes with >= 2 crossings: {len(axes_to_test)}")
    for pair, items in sorted(axes_to_test.items(),
                              key=lambda x: -len(x[1])):
        names = [d["entry"].name for d in items]
        log(f"    {pair[0]} <-> {pair[1]}: {len(items)} crossings "
            f"({', '.join(names)})")

    # -- Phase 2: Per-axis AUC computation ---------------------------
    log_section("Phase 2: Per-Axis Feature AUC")

    axis_results = {}

    for pair in sorted(axes_to_test.keys()):
        a_arch, b_arch = pair
        log_subsection(f"{a_arch} vs {b_arch}")

        a_indices = [i for i, d in enumerate(corpus)
                     if d["truth"] == a_arch]
        b_indices = [i for i, d in enumerate(corpus)
                     if d["truth"] == b_arch]

        log(f"  {a_arch}: {len(a_indices)} proteins, "
            f"{b_arch}: {len(b_indices)} proteins")

        auc_results = []
        for fi, fname in enumerate(feature_names):
            vals_a = feature_matrix[a_indices, fi]
            vals_b = feature_matrix[b_indices, fi]

            if np.std(np.concatenate([vals_a, vals_b])) < 1e-10:
                continue

            auc, direction = compute_auc(vals_a, vals_b)
            d_val = cohens_d(vals_a, vals_b)

            auc_results.append({
                "feature": fname,
                "auc": auc,
                "direction": direction,
                "cohens_d": d_val,
                "mean_a": float(np.mean(vals_a)),
                "mean_b": float(np.mean(vals_b)),
                "std_a": float(np.std(vals_a)),
                "std_b": float(np.std(vals_b)),
            })

        auc_results.sort(key=lambda x: -x["auc"])
        top10 = auc_results[:10]

        log(f"\n  Top 10 discriminating features:")
        log(f"  {'#':>3} {'Feature':<35} {'AUC':>6} {'d':>6} "
            f"{'Dir':>5} {'Mean_A':>8} {'Mean_B':>8}")
        log("  " + "-" * 78)
        for rank, r in enumerate(top10, 1):
            fname = r["feature"]
            if fname.startswith("agg_"):
                parts = fname.split("_", 2)
                base = parts[2] if len(parts) > 2 else fname
            else:
                parts = fname.split("_", 1)
                base = parts[1] if len(parts) > 1 else fname
            novel = "*" if base not in FEATURES_IN_RULES else " "
            is_agg = fname.startswith("agg_")
            tag = "AGG" if is_agg else "   "
            log(f"  {rank:>3} {fname:<35} {r['auc']:>6.3f} "
                f"{r['cohens_d']:>6.2f} {r['direction']:>5} "
                f"{r['mean_a']:>8.4f} {r['mean_b']:>8.4f} {tag}{novel}")

        axis_results[f"{a_arch}_vs_{b_arch}"] = {
            "n_a": len(a_indices), "n_b": len(b_indices),
            "top_features": top10,
            "all_features": auc_results[:30],
        }

    # -- Phase 3: Near-miss analysis ---------------------------------
    log_section("Phase 3: Near-Miss Proteins")

    near_misses = [d for d in corpus
                   if not d["correct"] and abs(d["margin"]) < 0.10]
    near_misses.sort(key=lambda d: abs(d["margin"]))

    log(f"\n  Near-miss proteins (|margin| < 0.10): {len(near_misses)}")
    log(f"  {'Protein':<25} {'Truth':<15} {'Predicted':<15} {'Margin':>8}")
    log("  " + "-" * 68)
    for d in near_misses:
        log(f"  {d['entry'].name:<25} {d['truth']:<15} "
            f"{d['predicted']:<15} {d['margin']:>+8.4f}")

    near_miss_analysis = []
    for d in near_misses:
        pair = tuple(sorted([d["truth"], d["predicted"]]))
        axis_key = f"{pair[0]}_vs_{pair[1]}"

        if axis_key not in axis_results:
            log(f"\n  {d['entry'].name}: axis {axis_key} not in main analysis "
                f"(< 2 crossings), computing ad-hoc...")
            a_arch, b_arch = pair
            a_indices = [i for i, dd in enumerate(corpus)
                         if dd["truth"] == a_arch]
            b_indices = [i for i, dd in enumerate(corpus)
                         if dd["truth"] == b_arch]

            top_feats_adhoc = []
            for fi, fname in enumerate(feature_names):
                vals_a = feature_matrix[a_indices, fi]
                vals_b = feature_matrix[b_indices, fi]
                if np.std(np.concatenate([vals_a, vals_b])) < 1e-10:
                    continue
                auc, direction = compute_auc(vals_a, vals_b)
                top_feats_adhoc.append({
                    "feature": fname, "auc": auc, "direction": direction})
            top_feats_adhoc.sort(key=lambda x: -x["auc"])
            axis_results[axis_key] = {
                "n_a": len(a_indices), "n_b": len(b_indices),
                "top_features": top_feats_adhoc[:10],
            }

        top_feats = axis_results[axis_key]["top_features"][:3]
        protein_idx = next(i for i, dd in enumerate(corpus)
                           if dd["entry"].name == d["entry"].name)

        log(f"\n  {d['entry'].name}: truth={d['truth']}, "
            f"pred={d['predicted']}, margin={d['margin']:+.4f}")
        log(f"    Top discriminants for {axis_key}:")

        favours_correct = 0
        for f in top_feats:
            fname = f["feature"]
            fi = feature_names.index(fname)
            val = feature_matrix[protein_idx, fi]

            a_arch, b_arch = pair
            a_indices = [i for i, dd in enumerate(corpus)
                         if dd["truth"] == a_arch]
            b_indices = [i for i, dd in enumerate(corpus)
                         if dd["truth"] == b_arch]
            mean_a = np.mean(feature_matrix[a_indices, fi])
            mean_b = np.mean(feature_matrix[b_indices, fi])
            dist_a = abs(val - mean_a)
            dist_b = abs(val - mean_b)
            favours = a_arch if dist_a < dist_b else b_arch
            correct = favours == d["truth"]
            if correct:
                favours_correct += 1
            log(f"      {fname:<35} val={val:>8.4f} "
                f"mean_{a_arch[:4]}={mean_a:>8.4f} "
                f"mean_{b_arch[:4]}={mean_b:>8.4f} "
                f"-> favours {favours} "
                f"({'YES' if correct else 'NO'})")

        rescue = favours_correct >= 2
        near_miss_analysis.append({
            "name": d["entry"].name,
            "truth": d["truth"],
            "predicted": d["predicted"],
            "margin": d["margin"],
            "favours_correct": favours_correct,
            "total_checked": len(top_feats),
            "rescuable": rescue,
        })
        log(f"    -> {favours_correct}/{len(top_feats)} features "
            f"favour truth -> {'RESCUABLE' if rescue else 'STUCK'}")

    # -- Phase 4: Cross-axis feature frequency -----------------------
    log_section("Phase 4: Cross-Axis Feature Frequency")

    feature_counts = Counter()
    feature_max_auc = {}
    per_inst_wins = 0
    agg_wins = 0

    for axis_key, result in axis_results.items():
        top5 = result["top_features"][:5]
        for r in top5:
            fname = r["feature"]
            feature_counts[fname] += 1
            prev = feature_max_auc.get(fname, 0)
            feature_max_auc[fname] = max(prev, r["auc"])
            if fname.startswith("agg_"):
                agg_wins += 1
            else:
                per_inst_wins += 1

    log(f"\n  Features appearing in top-5 across multiple axes:")
    log(f"  {'Feature':<40} {'Axes':>5} {'MaxAUC':>7}")
    log("  " + "-" * 55)
    for fname, count in feature_counts.most_common(20):
        if fname.startswith("agg_"):
            parts = fname.split("_", 2)
            base = parts[2] if len(parts) > 2 else fname
        else:
            parts = fname.split("_", 1)
            base = parts[1] if len(parts) > 1 else fname
        is_novel = base not in FEATURES_IN_RULES
        tag = " *NEW*" if is_novel else ""
        log(f"  {fname:<40} {count:>5} {feature_max_auc[fname]:>7.3f}{tag}")

    log(f"\n  Top-5 composition: {per_inst_wins} per-instrument, "
        f"{agg_wins} aggregate")

    # Base features across axes regardless of instrument
    base_feature_axes = defaultdict(set)
    for axis_key, result in axis_results.items():
        for r in result["top_features"][:5]:
            fname = r["feature"]
            if fname.startswith("agg_"):
                parts = fname.split("_", 2)
                base = parts[2] if len(parts) > 2 else fname
            else:
                parts = fname.split("_", 1)
                base = parts[1] if len(parts) > 1 else fname
            base_feature_axes[base].add(axis_key)

    log(f"\n  Base features (ignoring instrument) across axes:")
    log(f"  {'Base feature':<30} {'Axes':>5} {'In rules?':>10}")
    log("  " + "-" * 50)
    for base, axes in sorted(base_feature_axes.items(),
                             key=lambda x: -len(x[1])):
        in_rules = "YES" if base in FEATURES_IN_RULES else "NO"
        log(f"  {base:<30} {len(axes):>5} {in_rules:>10}")

    # -- Phase 5: Prediction Scorecard -------------------------------
    log_section("Phase 5: Prediction Scorecard")

    # P1: At least one axis with AUC > 0.85
    max_auc_overall = 0
    max_auc_axis = ""
    max_auc_feat = ""
    for axis_key, result in axis_results.items():
        if result["top_features"]:
            top = result["top_features"][0]
            if top["auc"] > max_auc_overall:
                max_auc_overall = top["auc"]
                max_auc_axis = axis_key
                max_auc_feat = top["feature"]
    p1 = max_auc_overall > 0.85
    log(f"\n  P1: Any axis with AUC > 0.85")
    log(f"      Best: {max_auc_feat} on {max_auc_axis} "
        f"AUC={max_auc_overall:.3f} -> "
        f"{'CONFIRMED' if p1 else 'REFUTED'}")

    # P2: Aggregate features outperform per-instrument on >= 2 axes
    agg_better_count = 0
    for axis_key, result in axis_results.items():
        if not result["top_features"]:
            continue
        best = result["top_features"][0]
        if best["feature"].startswith("agg_"):
            agg_better_count += 1
    p2 = agg_better_count >= 2
    log(f"\n  P2: Aggregate features #1 on >= 2 axes")
    log(f"      Aggregate #1 on {agg_better_count} axes -> "
        f"{'CONFIRMED' if p2 else 'REFUTED'}")

    # P3: >= 2 near-miss proteins rescuable
    n_rescuable = sum(1 for r in near_miss_analysis if r["rescuable"])
    p3 = n_rescuable >= 2
    log(f"\n  P3: >= 2 near-miss proteins rescuable")
    log(f"      {n_rescuable} rescuable -> "
        f"{'CONFIRMED' if p3 else 'REFUTED'}")

    # P4: gap_flatness or scatter_normalised in top-5 on >= 3 axes
    key_feats = {"gap_flatness", "scatter_normalised"}
    key_axis_count = 0
    for axis_key, result in axis_results.items():
        top5_bases = set()
        for r in result["top_features"][:5]:
            fname = r["feature"]
            if fname.startswith("agg_"):
                parts = fname.split("_", 2)
                base = parts[2] if len(parts) > 2 else fname
            else:
                parts = fname.split("_", 1)
                base = parts[1] if len(parts) > 1 else fname
            top5_bases.add(base)
        if key_feats & top5_bases:
            key_axis_count += 1
    p4 = key_axis_count >= 3
    log(f"\n  P4: gap_flatness or scatter_normalised in top-5 on >= 3 axes")
    log(f"      Found on {key_axis_count} axes -> "
        f"{'CONFIRMED' if p4 else 'REFUTED'}")

    # P5: Previously unused feature in top-3 on any axis
    novel_top3 = []
    for axis_key, result in axis_results.items():
        for r in result["top_features"][:3]:
            fname = r["feature"]
            if fname.startswith("agg_"):
                parts = fname.split("_", 2)
                base = parts[2] if len(parts) > 2 else fname
            else:
                parts = fname.split("_", 1)
                base = parts[1] if len(parts) > 1 else fname
            if base not in FEATURES_IN_RULES:
                novel_top3.append({
                    "feature": fname, "base": base,
                    "axis": axis_key, "auc": r["auc"],
                })
    p5 = len(novel_top3) > 0
    log(f"\n  P5: Novel feature in top-3 on any axis")
    if novel_top3:
        for nf in novel_top3[:5]:
            log(f"      {nf['feature']} (base={nf['base']}) on "
                f"{nf['axis']} AUC={nf['auc']:.3f}")
    log(f"      {len(novel_top3)} found -> "
        f"{'CONFIRMED' if p5 else 'REFUTED'}")

    n_confirmed = sum([p1, p2, p3, p4, p5])
    log(f"\n  Score: {n_confirmed}/5 predictions confirmed")

    # -- Save --------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    save_axes = {}
    for k, v in axis_results.items():
        save_axes[k] = {
            "n_a": v["n_a"], "n_b": v["n_b"],
            "top_features": v["top_features"][:10],
        }

    results = {
        "experiment": "D167_confusion_pair_discriminants",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "corpus_size": n,
        "n_correct": n_correct,
        "n_features": n_features,
        "confusion_axes": save_axes,
        "near_miss_analysis": near_miss_analysis,
        "cross_axis_frequency": [
            {"feature": f, "count": c, "max_auc": feature_max_auc[f]}
            for f, c in feature_counts.most_common(30)
        ],
        "base_feature_axes": {
            base: list(axes)
            for base, axes in sorted(base_feature_axes.items(),
                                     key=lambda x: -len(x[1]))
        },
        "predictions": {
            "P1_auc_gt_085": {
                "confirmed": p1,
                "value": round(max_auc_overall, 3),
                "feature": max_auc_feat,
                "axis": max_auc_axis,
            },
            "P2_agg_outperform_2axes": {
                "confirmed": p2,
                "value": agg_better_count,
            },
            "P3_near_miss_rescuable_2": {
                "confirmed": p3,
                "value": n_rescuable,
            },
            "P4_key_feats_3axes": {
                "confirmed": p4,
                "value": key_axis_count,
            },
            "P5_novel_feature_top3": {
                "confirmed": p5,
                "value": len(novel_top3),
            },
        },
        "n_confirmed": n_confirmed,
    }

    out_path = RESULTS_DIR / "d167_confusion_discriminants.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"\n  Results saved to {out_path}")

    elapsed = time.time() - t0
    log(f"  Total time: {elapsed:.1f}s")
    return results


if __name__ == "__main__":
    main()
