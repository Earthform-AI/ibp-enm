#!/usr/bin/env python
"""D170: Hinge + Enzyme Lens Gate Tightening.

Sprint 10 rationale:
  D169's full lens trace audit revealed that the hinge lens is the
  #1 false-positive creator on the expanded corpus:
    - Transferrin:    allosteric→enzyme_active (truth=dumbbell, 0/7 enz vote)
    - MBP:            allosteric→enzyme_active (truth=dumbbell, 0/7 enz vote)
    - GroEL_subunit:  globin→enzyme_active    (truth=allosteric, 0/7 enz vote)
    - ABP_open:       dumbbell nearly→enzyme  (truth=allosteric, 0/7 enz vote)
  The enzyme lens creates 1 additional false positive:
    - KDPG_aldolase:  barrel→enzyme_active    (truth=barrel, 2/7 enz vote)

  All 4 hinge false positives have enzyme_vote=0/7.  The existing
  `hinge_lens.enzyme_vote_min` gate is disabled (0.0).  Enabling it
  at 1/7 blocks all 4 with zero cost — no protein benefits from the
  hinge lens in the expanded corpus.

Algorithm: Vary gate thresholds on hinge and enzyme lenses. No new
  code — just threshold overrides on existing lens infrastructure.

Variants:
  A: Baseline (production thresholds)
  B: hinge_lens.enzyme_vote_min = 0.14 (1/7) — require ≥1 instrument
     to vote enzyme for hinge lens to fire
  C: B + enzyme_lens.close_call_gap = 0.06 — tighten enzyme lens gate
  D: B + hinge_lens.boost_cap = 0.20 (was 0.35) — reduce max boost
  E: B + enzyme_lens gate: require alg_enzyme > 0.40 (raise alg_strong)
  F: B + C + D combined — all tightenings

Predictions:
  P1: Variant B recovers ≥3 proteins with 0 regressions
  P2: KDPG_aldolase is recovered by variant C or E
  P3: Variant F achieves ≥19/32 (currently 16/32)
  P4: No variant loses Rubisco, Enolase, or other barrel proteins
  P5: GroEL_subunit returns to pre-lens prediction (globin) in variant B

Usage:
    python experiments/discovery_170_lens_tightening.py
"""

import json
import math
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ibp_enm.benchmark import EXPANDED_CORPUS, ProteinEntry
from ibp_enm.archetypes import ARCHETYPE_EXPECTATIONS
from ibp_enm.synthesis import AlgebraicFickBalancer
from ibp_enm.belief_algebra import ZDPairSelector
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

# Proteins affected by lens false positives (from D169 trace audit)
HINGE_FP = {"Transferrin", "MBP", "GroEL_subunit", "ABP_open"}
ENZYME_FP = {"KDPG_aldolase"}
ALL_FP = HINGE_FP | ENZYME_FP


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


def compute_pre_lens(profiles, meta_state, base_result):
    """Compute pre-lens scores (synthesis + bridge, no lenses)."""
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

    return pre_lens_scores


def score_with_thresholds(
    pre_lens_scores, profiles, evals, evecs, domain_labels,
    contacts, N, pdb_id, chain, threshold_overrides=None,
):
    """Score with optional threshold overrides for the lens stack."""
    from ibp_enm.thresholds import ThresholdRegistry

    # Build custom thresholds
    t = dict(DEFAULT_THRESHOLDS)
    if threshold_overrides:
        t.update(threshold_overrides)
    custom_t = ThresholdRegistry(t)

    stack = build_default_stack(
        evals=evals, evecs=evecs,
        domain_labels=domain_labels, contacts=contacts,
        pdb_id=pdb_id, chain=chain, n_residues=N,
        thresholds=custom_t,
    )

    context = {
        "evals": evals, "evecs": evecs,
        "domain_labels": domain_labels, "contacts": contacts,
        "pdb_id": pdb_id, "chain": chain, "n_residues": N,
    }
    final_scores, traces = stack.apply(dict(pre_lens_scores), profiles, context)
    identity = max(final_scores, key=final_scores.get)

    return {
        "identity": identity,
        "scores": final_scores,
        "pre_lens_scores": pre_lens_scores,
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
    print("D170: Hinge + Enzyme Lens Gate Tightening")
    print(f"  Corpus: {len(corpus)} proteins")
    print(f"  Hinge FP targets: {', '.join(sorted(HINGE_FP))}")
    print(f"  Enzyme FP targets: {', '.join(sorted(ENZYME_FP))}")
    print()

    # ── Phase 1: Load ──────────────────────────────────────────────
    print("=" * 72)
    print("PHASE 1: LOADING PROFILES & STRUCTURAL DATA")
    print("=" * 72)

    protein_data = {}
    t_start = time.perf_counter()

    for i, entry in enumerate(corpus):
        label = f"[{i+1}/{len(corpus)}]"
        profiles, metadata = load_cached_profiles(entry.pdb_id, entry.chain)
        if profiles is None or len(profiles) == 0:
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
        pre_lens = compute_pre_lens(profiles, meta_state, base_result)

        protein_data[entry.name] = {
            "entry": entry,
            "profiles": profiles,
            "meta_state": meta_state,
            "base_result": base_result,
            "pre_lens": pre_lens,
            "evals": evals,
            "evecs": evecs,
            "domain_labels": domain_labels,
            "contacts": contacts,
            "N": N,
        }

        fp = " ★" if entry.name in ALL_FP else ""
        pre_id = max(pre_lens, key=pre_lens.get)
        pre_mark = "✓" if pre_id == entry.archetype else "✗"
        print(f"  {label} {pre_mark} {entry.name} (N={N}, pre={pre_id}){fp}")

    t_load = time.perf_counter() - t_start
    n_loaded = len(protein_data)
    print(f"\n  Loaded: {n_loaded}/{len(corpus)} ({t_load:.1f}s)")

    n_fp = sum(1 for n in ALL_FP if n in protein_data)
    print(f"  FP targets loaded: {n_fp}/{len(ALL_FP)}")

    # Pre-lens accuracy
    pre_correct = sum(
        1 for n, pd in protein_data.items()
        if max(pd["pre_lens"], key=pd["pre_lens"].get) == pd["entry"].archetype
    )
    print(f"  Pre-lens accuracy: {pre_correct}/{n_loaded}")

    # ── Phase 2: Score all variants ────────────────────────────────
    print("\n" + "=" * 72)
    print("PHASE 2: SCORING ALL VARIANTS")
    print("=" * 72)

    variant_configs = {
        "A_baseline": {},
        "B_enz_vote_min": {
            "hinge_lens.enzyme_vote_min": 1/7,  # ≈0.143
        },
        "C_B_plus_tight_enz": {
            "hinge_lens.enzyme_vote_min": 1/7,
            "enzyme_lens.close_call_gap": 0.06,
        },
        "D_B_plus_low_cap": {
            "hinge_lens.enzyme_vote_min": 1/7,
            "hinge_lens.boost_cap": 0.20,
        },
        "E_B_plus_alg_strict": {
            "hinge_lens.enzyme_vote_min": 1/7,
            "enzyme_lens.alg_strong": 0.40,
        },
        "F_all_tight": {
            "hinge_lens.enzyme_vote_min": 1/7,
            "enzyme_lens.close_call_gap": 0.06,
            "hinge_lens.boost_cap": 0.20,
            "enzyme_lens.alg_strong": 0.40,
        },
    }

    results = {vname: {} for vname in variant_configs}

    for name, pd in sorted(protein_data.items(), key=lambda x: x[0]):
        entry = pd["entry"]

        for vname, overrides in variant_configs.items():
            vresult = score_with_thresholds(
                pd["pre_lens"], pd["profiles"],
                pd["evals"], pd["evecs"], pd["domain_labels"],
                pd["contacts"], pd["N"],
                entry.pdb_id, entry.chain,
                threshold_overrides=overrides,
            )
            results[vname][name] = vresult

        # Show per-protein comparison
        a_pred = results["A_baseline"][name]["identity"]
        a_correct = a_pred == entry.archetype
        changes = []
        for vname in list(variant_configs.keys())[1:]:
            vpred = results[vname][name]["identity"]
            if vpred != a_pred:
                v_correct = vpred == entry.archetype
                short = vname.split("_")[0]
                if v_correct and not a_correct:
                    changes.append(f"{short}:+1")
                elif not v_correct and a_correct:
                    changes.append(f"{short}:-1")
                else:
                    changes.append(f"{short}:Δ")

        fp = " ★" if name in ALL_FP else ""
        change_str = f"  [{', '.join(changes)}]" if changes else ""
        mark = "✓" if a_correct else "✗"
        print(f"  {mark} {name:<25s} truth={entry.archetype:<15s} "
              f"pred={a_pred:<15s}{change_str}{fp}")

    # ── Phase 3: Accuracy comparison ───────────────────────────────
    print("\n" + "=" * 72)
    print("PHASE 3: VARIANT ACCURACY COMPARISON")
    print("=" * 72)

    variant_stats = {}
    baseline_correct = 0

    for vname in variant_configs:
        correct = sum(
            1 for name in protein_data
            if results[vname][name]["identity"] == protein_data[name]["entry"].archetype
        )
        total = len(results[vname])
        pct = 100 * correct / max(total, 1)
        variant_stats[vname] = {"correct": correct, "total": total, "pct": pct}
        if vname == "A_baseline":
            baseline_correct = correct
        delta = correct - baseline_correct
        delta_str = f"(Δ={delta:+d})" if vname != "A_baseline" else ""
        print(f"  {vname:<25s}: {correct}/{total} ({pct:.1f}%) {delta_str}")

    # FP subset
    print(f"\n  False-positive targets ({len(ALL_FP)} proteins):")
    for vname in variant_configs:
        correct = sum(
            1 for name in ALL_FP
            if name in results[vname]
            and results[vname][name]["identity"] == protein_data[name]["entry"].archetype
        )
        loaded = sum(1 for name in ALL_FP if name in results[vname])
        print(f"    {vname:<25s}: {correct}/{loaded}")

    # ── Phase 4: Per-protein changes ───────────────────────────────
    print("\n" + "=" * 72)
    print("PHASE 4: PER-PROTEIN CHANGES (vs baseline)")
    print("=" * 72)

    variant_changes = {}

    for vname in list(variant_configs.keys())[1:]:
        gains, losses, flips = [], [], []
        for name in sorted(protein_data.keys()):
            entry = protein_data[name]["entry"]
            a_correct = results["A_baseline"][name]["identity"] == entry.archetype
            v_correct = results[vname][name]["identity"] == entry.archetype
            a_pred = results["A_baseline"][name]["identity"]
            v_pred = results[vname][name]["identity"]

            fp = " ★" if name in ALL_FP else ""
            if not a_correct and v_correct:
                gains.append(f"    + {name:<25s} truth={entry.archetype:<15s} "
                             f"was={a_pred:<15s} now={v_pred}{fp}")
            elif a_correct and not v_correct:
                losses.append(f"    - {name:<25s} truth={entry.archetype:<15s} "
                              f"was_correct, now={v_pred}{fp}")
            elif v_pred != a_pred:
                flips.append(f"    ~ {name:<25s} truth={entry.archetype:<15s} "
                             f"was={a_pred:<15s} now={v_pred}{fp}")

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

    # ── Phase 5: Lens activation audit ─────────────────────────────
    print("\n" + "=" * 72)
    print("PHASE 5: LENS ACTIVATION AUDIT (Variant B)")
    print("=" * 72)

    vname = "B_enz_vote_min"
    for name in sorted(protein_data.keys()):
        r = results[vname][name]
        entry = protein_data[name]["entry"]
        for t in r["lens_traces"]:
            if t["activated"] and t["boost"] != 0:
                correct = "✓" if r["identity"] == entry.archetype else "✗"
                fp = " ★" if name in ALL_FP else ""
                print(f"  {correct} {name:<25s} lens={t['name']:<20s} "
                      f"boost={t['boost']:.3f}{fp}")

    # Also show baseline activations for comparison
    print("\n  Baseline (A) active lenses:")
    for name in sorted(protein_data.keys()):
        r = results["A_baseline"][name]
        entry = protein_data[name]["entry"]
        for t in r["lens_traces"]:
            if t["activated"] and t["boost"] != 0:
                correct = "✓" if r["identity"] == entry.archetype else "✗"
                fp = " ★" if name in ALL_FP else ""
                print(f"  {correct} {name:<25s} lens={t['name']:<20s} "
                      f"boost={t['boost']:.3f}{fp}")

    # ── Phase 6: Prediction scorecard ──────────────────────────────
    print("\n" + "=" * 72)
    print("PHASE 6: PREDICTION SCORECARD")
    print("=" * 72)

    predictions = {}

    # P1: Variant B recovers ≥3 proteins with 0 regressions
    b_gains = len(variant_changes.get("B_enz_vote_min", {}).get("gains", []))
    b_losses = len(variant_changes.get("B_enz_vote_min", {}).get("losses", []))
    p1 = b_gains >= 3 and b_losses == 0
    predictions["P1"] = p1
    print(f"  P1 {'✓' if p1 else '✗'}: B gains={b_gains} losses={b_losses} "
          f"(need ≥3 gains, 0 losses)")

    # P2: KDPG_aldolase recovered by C or E
    kdpg_c = (
        "KDPG_aldolase" in results.get("C_B_plus_tight_enz", {})
        and results["C_B_plus_tight_enz"]["KDPG_aldolase"]["identity"] == "barrel"
    )
    kdpg_e = (
        "KDPG_aldolase" in results.get("E_B_plus_alg_strict", {})
        and results["E_B_plus_alg_strict"]["KDPG_aldolase"]["identity"] == "barrel"
    )
    p2 = kdpg_c or kdpg_e
    predictions["P2"] = p2
    print(f"  P2 {'✓' if p2 else '✗'}: KDPG recovered by C={kdpg_c} E={kdpg_e}")

    # P3: Variant F achieves ≥19/32
    f_correct = variant_stats.get("F_all_tight", {}).get("correct", 0)
    p3 = f_correct >= 19
    predictions["P3"] = p3
    print(f"  P3 {'✓' if p3 else '✗'}: F accuracy={f_correct}/32 (need ≥19)")

    # P4: No variant loses barrel proteins (Rubisco, Enolase, etc.)
    barrel_proteins = {"Rubisco_large", "Enolase", "Aldolase_A",
                       "Glycolate_oxidase", "Tryptophan_synth", "Mandelate_racemase"}
    barrel_lost = False
    for vname in variant_configs:
        for bp in barrel_proteins:
            if bp in results[vname]:
                if results[vname][bp]["identity"] != "barrel":
                    barrel_lost = True
                    break
    p4 = not barrel_lost
    predictions["P4"] = p4
    print(f"  P4 {'✓' if p4 else '✗'}: No barrel regressions = {not barrel_lost}")

    # P5: GroEL returns to pre-lens prediction in B
    groel_b = results.get("B_enz_vote_min", {}).get("GroEL_subunit", {})
    groel_pre = max(
        protein_data.get("GroEL_subunit", {}).get("pre_lens", {}),
        key=protein_data.get("GroEL_subunit", {}).get("pre_lens", {}).get,
        default="?"
    )
    groel_b_id = groel_b.get("identity", "?")
    p5 = groel_b_id == groel_pre
    predictions["P5"] = p5
    print(f"  P5 {'✓' if p5 else '✗'}: GroEL B={groel_b_id} pre={groel_pre}")

    confirmed = sum(1 for v in predictions.values() if v)
    print(f"\n  SCORECARD: {confirmed}/{len(predictions)} predictions confirmed")

    # ── Phase 7: Recommended production change ─────────────────────
    print("\n" + "=" * 72)
    print("PHASE 7: RECOMMENDED PRODUCTION CHANGE")
    print("=" * 72)

    # Find the best variant (max correct, 0 losses preferred)
    best = None
    best_score = 0
    for vname in variant_configs:
        vc = variant_stats[vname]
        losses = len(variant_changes.get(vname, {}).get("losses", []))
        score = vc["correct"] * 100 - losses * 1000  # heavily penalize losses
        if score > best_score:
            best_score = score
            best = vname

    if best and best != "A_baseline":
        vc = variant_stats[best]
        losses = len(variant_changes.get(best, {}).get("losses", []))
        print(f"  Best variant: {best}")
        print(f"  Accuracy: {vc['correct']}/{vc['total']} "
              f"(Δ={vc['correct']-baseline_correct:+d})")
        print(f"  Regressions: {losses}")
        print(f"  Threshold overrides: {json.dumps(variant_configs[best], indent=4)}")
    else:
        print(f"  No improvement found — baseline remains best")

    # ── Save results ───────────────────────────────────────────────
    output = {
        "experiment": "D170",
        "title": "Hinge + Enzyme Lens Gate Tightening",
        "corpus_size": len(corpus),
        "loaded": n_loaded,
        "pre_lens_accuracy": pre_correct,
        "fp_targets": sorted(ALL_FP),
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
        "variant_configs": {
            vname: {k: round(v, 4) for k, v in cfg.items()}
            for vname, cfg in variant_configs.items()
        },
        "predictions": predictions,
        "predictions_confirmed": confirmed,
        "predictions_total": len(predictions),
        "recommended": best,
    }

    results_path = RESULTS_DIR / "d170_lens_tightening.json"
    results_path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"\n  Results saved to {results_path}")


if __name__ == "__main__":
    main()
