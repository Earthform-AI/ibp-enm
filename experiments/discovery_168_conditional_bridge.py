#!/usr/bin/env python
"""D168: Conditional Bridge — Surgical α₈ De-gating.

The α₈ "double-gating" issue (documented D159/D160):
  bridge_weight = 0.5 × (1-α₀) × α₈          # α₈ gates the WEIGHT
  bridge = rs × context_boost + α₈ × fano_bridge  # α₈ gates fano_bridge

  → context_boost effective scaling ∝ α₈
  → fano_bridge effective scaling ∝ α₈²
  → 4 proteins with α₈ ≤ 1/7 are "bridge-blind"

D160 Variant C (remove α₈ from bridge_weight entirely) caused 5 regressions
because bridge_weight ≈ 0.5 for low-α₀ proteins, overwhelming the main signal.

D168 tests 6 SURGICAL approaches that increase bridge for low-α₈ proteins
without over-boosting the majority:

Variants:
  A: Baseline (production: route-gated, α₈ in bridge_weight)
  B: Floor gate — clamp α₈ ≥ 0.15 in bridge_weight only
     bw = 0.5 × (1-α₀) × max(α₈, 0.15)
  C: Soft floor — clamp α₈ ≥ 0.15 everywhere (bw AND bridge signal)
     bw = 0.5 × (1-α₀) × max(α₈, 0.15)
     bridge = rs × context_boost + max(α₈, 0.15) × fano_bridge
  D: Margin-conditional — apply floor only when α₀ < 0.05 (confused votes)
     if α₀ < 0.05: α₈_eff = max(α₈, 0.15) else: α₈_eff = α₈
  E: Sqrt gate — √α₈ in bridge_weight (compressive: 0.14→0.37, 0.43→0.66)
     bw = 0.5 × (1-α₀) × √α₈
  F: Sqrt gate + soft floor — √max(α₈, 0.15) everywhere
     bw = 0.5 × (1-α₀) × √max(α₈, 0.15)
     bridge = rs × context_boost + √max(α₈, 0.15) × fano_bridge

All variants use 0 new free parameters (floor 0.15 = 1/7 rounded up,
the minimum non-zero α₈ for a protein with exactly 1 Fano line covered).

Bridge-blind targets (α₈ ≤ 1/7):
  Neuroglobin       (α₈=0.143, truth=globin,     pred=barrel)
  Cytochrome_b5     (α₈=0.000, truth=globin,     pred=enzyme_active)
  Erythrocruorin    (α₈=0.143, truth=globin,     pred=allosteric)
  GroEL_subunit     (α₈=0.143, truth=allosteric, pred=enzyme_active)

Predictions:
  P1: ≥1 variant recovers ≥1 bridge-blind protein (Neuroglobin or
      Erythrocruorin most likely — small margin-to-truth)
  P2: Sqrt gate (E) causes fewer regressions than floor gate (B)
      because √ is compressive — less distortion at high α₈
  P3: Margin-conditional (D) has 0 regressions — only fires on
      confused proteins where main signal is already unreliable
  P4: No variant exceeds 33/52 — bridge alone can't fix rule-level
      confusion (Chymotrypsin, Subtilisin, Glycogen_phosph)
  P5: Cytochrome_b5 (α₈=0.000) is NOT recovered by floor gate
      because even with floor, its context_boost is structurally
      misaligned (truth=globin, D159 margin-to-truth = -0.226)

Usage:
    python experiments/discovery_168_conditional_bridge.py
"""

import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ibp_enm.benchmark import EXPANDED_CORPUS, ProteinEntry
from ibp_enm.archetypes import ARCHETYPE_EXPECTATIONS
from ibp_enm.synthesis import AlgebraicFickBalancer
from ibp_enm.belief_algebra import ZDPairSelector, FANO_LINES
from ibp_enm.cache import profiles_from_json
from ibp_enm.lens_stack import build_default_stack
from ibp_enm.band import _fetch_ca, build_laplacian
from ibp_enm.analyzer import IBPProteinAnalyzer

CACHE_DIR = Path.home() / ".ibp_enm_cache"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
ALL_ARCHS = list(ARCHETYPE_EXPECTATIONS.keys())

# The 4 bridge-blind proteins
BRIDGE_BLIND = {"Neuroglobin", "Cytochrome_b5", "Erythrocruorin", "GroEL_subunit"}

# Floor: 1/7 rounded up — minimum non-zero α₈
ALPHA8_FLOOR = 1 / 7  # ≈ 0.1429

# ── Constants from AlgebraicFickBalancer ───────────────────────────
SQRT2 = math.sqrt(2)
STRONG_WEIGHT = SQRT2 / (SQRT2 + 1)   # ≈ 0.5858
WEAK_WEIGHT = 1.0 / (SQRT2 + 1)       # ≈ 0.4142
BRIDGE_SCALE = 0.5


# ── Helper functions ───────────────────────────────────────────────

def load_cached_profiles(pdb_id: str, chain: str):
    """Load profiles + metadata from cache."""
    path = CACHE_DIR / f"{pdb_id.upper()}_{chain}.json"
    if not path.exists():
        return None, None
    text = path.read_text(encoding="utf-8")
    profiles, metadata = profiles_from_json(text)
    return profiles, metadata


def get_structural_data(pdb_id: str, chain: str):
    """Compute evals, evecs, domain_labels, contacts from PDB coords."""
    coords, bfactors = _fetch_ca(pdb_id, chain)
    N = len(coords)
    analyzer = IBPProteinAnalyzer()
    result = analyzer.analyze(coords, bfactors)
    contacts, _ = analyzer._build_contacts(coords, N)
    L = build_laplacian(N, contacts)
    evals, evecs = np.linalg.eigh(L)
    domain_labels = result.domain_labels
    return evals, evecs, domain_labels, contacts, N


def compute_per_arch_route_scores(
    carver_votes: List[Dict[str, float]],
    all_archs: List[str],
) -> Dict[str, float]:
    """Per-archetype route_score from instrument votes."""
    zdp = ZDPairSelector()
    n = min(len(carver_votes), 7)
    route_scores = {}
    for arch in all_archs:
        support = np.zeros(7, dtype=int)
        for i in range(n):
            top = max(carver_votes[i], key=carver_votes[i].get)
            if top == arch:
                support[i] = 1
        route_scores[arch] = zdp.route_score(support)
    return route_scores


# ── Scoring variants ───────────────────────────────────────────────

def score_variant_a(
    consensus_scores, disagreement_scores, context_boost,
    fano_bridge, alpha_0, alpha_8, route_scores,
):
    """A: Baseline — production (route-gated, α₈ in bridge_weight)."""
    bridge_weight = BRIDGE_SCALE * (1.0 - alpha_0) * alpha_8
    main_weight = 1.0 - bridge_weight

    scores = {}
    for arch in ALL_ARCHS:
        strong = alpha_0 * consensus_scores.get(arch, 0)
        weak = (1 - alpha_0) * disagreement_scores.get(arch, 0)
        main = STRONG_WEIGHT * strong + WEAK_WEIGHT * weak
        rs = route_scores.get(arch, 0)
        bridge = (rs * context_boost.get(arch, 0)
                  + alpha_8 * fano_bridge.get(arch, 0))
        scores[arch] = main_weight * main + bridge_weight * bridge

    total = sum(scores.values())
    if total > 1e-10:
        scores = {k: v / total for k, v in scores.items()}
    return scores


def score_variant_b(
    consensus_scores, disagreement_scores, context_boost,
    fano_bridge, alpha_0, alpha_8, route_scores,
):
    """B: Floor gate — clamp α₈ ≥ floor in bridge_weight only.

    bw = 0.5 × (1-α₀) × max(α₈, floor)
    bridge signal unchanged (uses raw α₈).
    """
    a8_bw = max(alpha_8, ALPHA8_FLOOR)
    bridge_weight = BRIDGE_SCALE * (1.0 - alpha_0) * a8_bw
    main_weight = 1.0 - bridge_weight

    scores = {}
    for arch in ALL_ARCHS:
        strong = alpha_0 * consensus_scores.get(arch, 0)
        weak = (1 - alpha_0) * disagreement_scores.get(arch, 0)
        main = STRONG_WEIGHT * strong + WEAK_WEIGHT * weak
        rs = route_scores.get(arch, 0)
        bridge = (rs * context_boost.get(arch, 0)
                  + alpha_8 * fano_bridge.get(arch, 0))
        scores[arch] = main_weight * main + bridge_weight * bridge

    total = sum(scores.values())
    if total > 1e-10:
        scores = {k: v / total for k, v in scores.items()}
    return scores


def score_variant_c(
    consensus_scores, disagreement_scores, context_boost,
    fano_bridge, alpha_0, alpha_8, route_scores,
):
    """C: Soft floor — clamp α₈ ≥ floor EVERYWHERE.

    bw = 0.5 × (1-α₀) × max(α₈, floor)
    bridge = rs × context_boost + max(α₈, floor) × fano_bridge
    """
    a8_eff = max(alpha_8, ALPHA8_FLOOR)
    bridge_weight = BRIDGE_SCALE * (1.0 - alpha_0) * a8_eff
    main_weight = 1.0 - bridge_weight

    scores = {}
    for arch in ALL_ARCHS:
        strong = alpha_0 * consensus_scores.get(arch, 0)
        weak = (1 - alpha_0) * disagreement_scores.get(arch, 0)
        main = STRONG_WEIGHT * strong + WEAK_WEIGHT * weak
        rs = route_scores.get(arch, 0)
        bridge = (rs * context_boost.get(arch, 0)
                  + a8_eff * fano_bridge.get(arch, 0))
        scores[arch] = main_weight * main + bridge_weight * bridge

    total = sum(scores.values())
    if total > 1e-10:
        scores = {k: v / total for k, v in scores.items()}
    return scores


def score_variant_d(
    consensus_scores, disagreement_scores, context_boost,
    fano_bridge, alpha_0, alpha_8, route_scores,
):
    """D: Margin-conditional — floor only when α₀ < 0.05.

    For confused proteins (low vote margin), apply soft floor.
    For confident proteins, use production formula unchanged.
    """
    if alpha_0 < 0.05:
        a8_eff = max(alpha_8, ALPHA8_FLOOR)
    else:
        a8_eff = alpha_8

    bridge_weight = BRIDGE_SCALE * (1.0 - alpha_0) * a8_eff
    main_weight = 1.0 - bridge_weight

    scores = {}
    for arch in ALL_ARCHS:
        strong = alpha_0 * consensus_scores.get(arch, 0)
        weak = (1 - alpha_0) * disagreement_scores.get(arch, 0)
        main = STRONG_WEIGHT * strong + WEAK_WEIGHT * weak
        rs = route_scores.get(arch, 0)
        bridge = (rs * context_boost.get(arch, 0)
                  + a8_eff * fano_bridge.get(arch, 0))
        scores[arch] = main_weight * main + bridge_weight * bridge

    total = sum(scores.values())
    if total > 1e-10:
        scores = {k: v / total for k, v in scores.items()}
    return scores


def score_variant_e(
    consensus_scores, disagreement_scores, context_boost,
    fano_bridge, alpha_0, alpha_8, route_scores,
):
    """E: Sqrt gate — √α₈ in bridge_weight only.

    Compressive: 0.14→0.37, 0.43→0.66, 0.71→0.84, 1.0→1.0
    Less distortion at high α₈ than a floor gate.
    """
    a8_bw = math.sqrt(alpha_8)
    bridge_weight = BRIDGE_SCALE * (1.0 - alpha_0) * a8_bw
    main_weight = 1.0 - bridge_weight

    scores = {}
    for arch in ALL_ARCHS:
        strong = alpha_0 * consensus_scores.get(arch, 0)
        weak = (1 - alpha_0) * disagreement_scores.get(arch, 0)
        main = STRONG_WEIGHT * strong + WEAK_WEIGHT * weak
        rs = route_scores.get(arch, 0)
        bridge = (rs * context_boost.get(arch, 0)
                  + alpha_8 * fano_bridge.get(arch, 0))
        scores[arch] = main_weight * main + bridge_weight * bridge

    total = sum(scores.values())
    if total > 1e-10:
        scores = {k: v / total for k, v in scores.items()}
    return scores


def score_variant_f(
    consensus_scores, disagreement_scores, context_boost,
    fano_bridge, alpha_0, alpha_8, route_scores,
):
    """F: Sqrt gate + soft floor — √max(α₈, floor) everywhere.

    Combines compressive sqrt with floor guarantee.
    bw = 0.5 × (1-α₀) × √max(α₈, floor)
    bridge = rs × context_boost + √max(α₈, floor) × fano_bridge
    """
    a8_eff = math.sqrt(max(alpha_8, ALPHA8_FLOOR))
    bridge_weight = BRIDGE_SCALE * (1.0 - alpha_0) * a8_eff
    main_weight = 1.0 - bridge_weight

    scores = {}
    for arch in ALL_ARCHS:
        strong = alpha_0 * consensus_scores.get(arch, 0)
        weak = (1 - alpha_0) * disagreement_scores.get(arch, 0)
        main = STRONG_WEIGHT * strong + WEAK_WEIGHT * weak
        rs = route_scores.get(arch, 0)
        bridge = (rs * context_boost.get(arch, 0)
                  + a8_eff * fano_bridge.get(arch, 0))
        scores[arch] = main_weight * main + bridge_weight * bridge

    total = sum(scores.values())
    if total > 1e-10:
        scores = {k: v / total for k, v in scores.items()}
    return scores


VARIANTS = {
    "A_baseline": score_variant_a,
    "B_floor_gate": score_variant_b,
    "C_soft_floor": score_variant_c,
    "D_margin_cond": score_variant_d,
    "E_sqrt_gate": score_variant_e,
    "F_sqrt_floor": score_variant_f,
}


# ── Re-scoring with lens stack ─────────────────────────────────────

def rescore_protein(
    profiles, meta_state, base_result,
    evals, evecs, domain_labels, contacts, N,
    pdb_id, chain, variant_fn,
):
    """Re-score using variant scoring function + lens stack."""
    carver_votes = [p.archetype_vote() for p in profiles]

    alpha_0 = meta_state.get("alpha_0", 0.5)
    alpha_8 = meta_state.get("alpha_8", 0.0)

    consensus_scores = base_result["consensus_scores"]
    disagreement_scores = base_result["disagreement_scores"]
    context_boost = base_result["context_boost"]

    balancer = AlgebraicFickBalancer()
    fano_bridge = balancer._hamming_bridge.bridge_scores(carver_votes, ALL_ARCHS)

    route_scores = compute_per_arch_route_scores(carver_votes, ALL_ARCHS)

    pre_lens_scores = variant_fn(
        consensus_scores, disagreement_scores, context_boost,
        fano_bridge, alpha_0, alpha_8, route_scores,
    )

    stack = build_default_stack(
        evals=evals, evecs=evecs,
        domain_labels=domain_labels, contacts=contacts,
        pdb_id=pdb_id, chain=chain, n_residues=N,
    )
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
        "route_scores": route_scores,
        "fano_bridge": fano_bridge,
        "context_boost": context_boost,
        "lens_traces": traces,
    }


# ── Main experiment ────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    corpus = list(EXPANDED_CORPUS)
    print("D168: Conditional Bridge — Surgical α₈ De-gating")
    print(f"  Corpus: {len(corpus)} proteins")
    print(f"  Variants: {', '.join(VARIANTS.keys())}")
    print(f"  α₈ floor: {ALPHA8_FLOOR:.4f} (1/7)")
    print(f"  Bridge-blind targets: {', '.join(sorted(BRIDGE_BLIND))}")
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
            print(f"  {label} ✗ {entry.name}: structural data error: {exc}")
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

        blind_marker = " ★" if entry.name in BRIDGE_BLIND else ""
        print(f"  {label} ✓ {entry.name} (N={N}, "
              f"α₀={meta_state['alpha_0']:.3f}, "
              f"α₈={meta_state['alpha_8']:.3f}){blind_marker}")
        n_loaded += 1

    t_load = time.perf_counter() - t_start
    print(f"\n  Loaded: {n_loaded}/{len(corpus)} ({t_load:.1f}s)")

    n_blind_loaded = sum(1 for n in BRIDGE_BLIND if n in protein_data)
    print(f"  Bridge-blind loaded: {n_blind_loaded}/{len(BRIDGE_BLIND)}")

    # ── Phase 2: Score all variants ────────────────────────────────
    print("\n" + "=" * 72)
    print("PHASE 2: SCORING ALL VARIANTS")
    print("=" * 72)

    results = {vname: {} for vname in VARIANTS}

    for i, entry in enumerate(corpus):
        if entry.name not in protein_data:
            continue

        pd = protein_data[entry.name]
        label = f"[{i+1}/{len(corpus)}]"
        preds = {}

        for vname, vfn in VARIANTS.items():
            vresult = rescore_protein(
                pd["profiles"], pd["meta_state"], pd["base_result"],
                pd["evals"], pd["evecs"], pd["domain_labels"],
                pd["contacts"], pd["N"],
                entry.pdb_id, entry.chain, vfn,
            )
            results[vname][entry.name] = vresult
            correct = vresult["identity"] == entry.archetype
            preds[vname] = ("✓" if correct else "✗", vresult["identity"])

        a_pred = preds.get("A_baseline", ("?", "?"))
        changes = []
        for vname in list(VARIANTS.keys())[1:]:
            vpred = preds.get(vname, ("?", "?"))
            if vpred[1] != a_pred[1]:
                short = vname.split("_")[0]
                if vpred[0] == "✓" and a_pred[0] == "✗":
                    changes.append(f"{short}:+1")
                elif vpred[0] == "✗" and a_pred[0] == "✓":
                    changes.append(f"{short}:-1")
                else:
                    changes.append(f"{short}:Δ")

        blind_marker = " ★" if entry.name in BRIDGE_BLIND else ""
        change_str = f"  [{', '.join(changes)}]" if changes else ""
        print(f"  {label} {a_pred[0]} {entry.name:<25s} "
              f"truth={entry.archetype:<15s} pred={a_pred[1]:<15s}"
              f"{change_str}{blind_marker}")

    # ── Phase 3: Accuracy comparison ───────────────────────────────
    print("\n" + "=" * 72)
    print("PHASE 3: VARIANT ACCURACY COMPARISON")
    print("=" * 72)

    variant_stats = {}
    baseline_correct = 0

    for vname in VARIANTS:
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

    # Bridge-blind subset accuracy
    print(f"\n  Bridge-blind accuracy ({len(BRIDGE_BLIND)} proteins):")
    for vname in VARIANTS:
        correct = sum(
            1 for name in BRIDGE_BLIND
            if name in results[vname]
            and results[vname][name]["identity"] == protein_data[name]["entry"].archetype
        )
        loaded = sum(1 for name in BRIDGE_BLIND if name in results[vname])
        print(f"    {vname:<20s}: {correct}/{loaded}")

    # ── Phase 4: Per-protein changes vs baseline ───────────────────
    print("\n" + "=" * 72)
    print("PHASE 4: PER-PROTEIN CHANGES (vs baseline)")
    print("=" * 72)

    variant_changes = {}

    for vname in list(VARIANTS.keys())[1:]:
        gains, losses, flips = [], [], []
        for entry in corpus:
            if entry.name not in results["A_baseline"] or entry.name not in results[vname]:
                continue
            a_correct = results["A_baseline"][entry.name]["identity"] == entry.archetype
            v_correct = results[vname][entry.name]["identity"] == entry.archetype
            v_pred = results[vname][entry.name]["identity"]
            a_pred = results["A_baseline"][entry.name]["identity"]

            blind = " ★" if entry.name in BRIDGE_BLIND else ""
            if not a_correct and v_correct:
                gains.append(f"    + {entry.name:<25s} truth={entry.archetype:<15s} "
                             f"was={a_pred:<15s} now={v_pred}{blind}")
            elif a_correct and not v_correct:
                losses.append(f"    - {entry.name:<25s} truth={entry.archetype:<15s} "
                              f"was_correct, now={v_pred}{blind}")
            elif v_pred != a_pred:
                flips.append(f"    ~ {entry.name:<25s} truth={entry.archetype:<15s} "
                             f"was={a_pred:<15s} now={v_pred}{blind}")

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

    # ── Phase 5: Bridge-blind deep dive ────────────────────────────
    print("\n" + "=" * 72)
    print("PHASE 5: BRIDGE-BLIND PROTEIN ANALYSIS")
    print("=" * 72)

    for name in sorted(BRIDGE_BLIND):
        if name not in protein_data:
            print(f"\n  {name}: NOT LOADED")
            continue

        pd = protein_data[name]
        entry = pd["entry"]
        ms = pd["meta_state"]
        print(f"\n  ── {name} ─────────────────────────────")
        print(f"  Truth: {entry.archetype}")
        print(f"  α₀={ms['alpha_0']:.4f}, α₈={ms['alpha_8']:.4f}")

        # Show which variants change its prediction
        for vname in VARIANTS:
            if name not in results[vname]:
                continue
            r = results[vname][name]
            correct = "✓" if r["identity"] == entry.archetype else "✗"

            # Top-3 scores
            sorted_scores = sorted(r["scores"].items(), key=lambda x: -x[1])
            top3 = ", ".join(f"{a}={s:.4f}" for a, s in sorted_scores[:3])

            # Bridge weight for this variant
            a8 = ms["alpha_8"]
            a0 = ms["alpha_0"]
            if vname == "A_baseline":
                bw = BRIDGE_SCALE * (1 - a0) * a8
            elif vname == "B_floor_gate":
                bw = BRIDGE_SCALE * (1 - a0) * max(a8, ALPHA8_FLOOR)
            elif vname in ("C_soft_floor", "D_margin_cond"):
                a8_eff = max(a8, ALPHA8_FLOOR) if (vname == "C_soft_floor" or a0 < 0.05) else a8
                bw = BRIDGE_SCALE * (1 - a0) * a8_eff
            elif vname == "E_sqrt_gate":
                bw = BRIDGE_SCALE * (1 - a0) * math.sqrt(a8)
            elif vname == "F_sqrt_floor":
                bw = BRIDGE_SCALE * (1 - a0) * math.sqrt(max(a8, ALPHA8_FLOOR))
            else:
                bw = 0

            print(f"    {vname:<20s}: {correct} {r['identity']:<15s} "
                  f"bw={bw:.4f}  [{top3}]")

        # Context boost for truth archetype
        if name in results["A_baseline"]:
            cb_truth = results["A_baseline"][name]["context_boost"].get(entry.archetype, 0)
            fb_truth = results["A_baseline"][name]["fano_bridge"].get(entry.archetype, 0)
            rs_truth = results["A_baseline"][name]["route_scores"].get(entry.archetype, 0)
            print(f"  context_boost[{entry.archetype}]={cb_truth:.4f}, "
                  f"fano_bridge={fb_truth:.4f}, route_score={rs_truth:.4f}")

    # ── Phase 6: α₈ distribution + bridge_weight comparison ───────
    print("\n" + "=" * 72)
    print("PHASE 6: α₈ DISTRIBUTION & BRIDGE WEIGHT IMPACT")
    print("=" * 72)

    a8_values = []
    for name, pd in protein_data.items():
        a8 = pd["meta_state"]["alpha_8"]
        a0 = pd["meta_state"]["alpha_0"]
        a8_values.append((name, a8, a0))
    a8_values.sort(key=lambda x: x[1])

    print(f"\n  α₈ distribution ({len(a8_values)} proteins):")
    print(f"    min={a8_values[0][1]:.3f} ({a8_values[0][0]})")
    print(f"    max={a8_values[-1][1]:.3f} ({a8_values[-1][0]})")
    print(f"    mean={np.mean([x[1] for x in a8_values]):.3f}")
    print(f"    median={np.median([x[1] for x in a8_values]):.3f}")

    # Count proteins where each variant changes bridge_weight significantly
    print(f"\n  Bridge weight comparison (baseline vs variants):")
    print(f"    Proteins where variant increases bw by >0.01:")
    for vname in list(VARIANTS.keys())[1:]:
        count = 0
        for name, a8, a0 in a8_values:
            bw_base = BRIDGE_SCALE * (1 - a0) * a8

            if vname == "B_floor_gate":
                bw_var = BRIDGE_SCALE * (1 - a0) * max(a8, ALPHA8_FLOOR)
            elif vname == "C_soft_floor":
                bw_var = BRIDGE_SCALE * (1 - a0) * max(a8, ALPHA8_FLOOR)
            elif vname == "D_margin_cond":
                a8_eff = max(a8, ALPHA8_FLOOR) if a0 < 0.05 else a8
                bw_var = BRIDGE_SCALE * (1 - a0) * a8_eff
            elif vname == "E_sqrt_gate":
                bw_var = BRIDGE_SCALE * (1 - a0) * math.sqrt(a8)
            elif vname == "F_sqrt_floor":
                bw_var = BRIDGE_SCALE * (1 - a0) * math.sqrt(max(a8, ALPHA8_FLOOR))
            else:
                bw_var = bw_base

            if bw_var - bw_base > 0.01:
                count += 1
        print(f"      {vname:<20s}: {count}")

    # ── Phase 7: Prediction scorecard ──────────────────────────────
    print("\n" + "=" * 72)
    print("PHASE 7: PREDICTION SCORECARD")
    print("=" * 72)

    predictions = {}

    # P1: ≥1 variant recovers ≥1 bridge-blind protein
    blind_recovered = {}
    for vname in list(VARIANTS.keys())[1:]:
        recovered = set()
        for name in BRIDGE_BLIND:
            if name not in results["A_baseline"] or name not in results[vname]:
                continue
            a_correct = results["A_baseline"][name]["identity"] == protein_data[name]["entry"].archetype
            v_correct = results[vname][name]["identity"] == protein_data[name]["entry"].archetype
            if not a_correct and v_correct:
                recovered.add(name)
        if recovered:
            blind_recovered[vname] = recovered

    p1 = len(blind_recovered) > 0
    predictions["P1"] = p1
    if p1:
        for vname, recovered in blind_recovered.items():
            print(f"  P1 ✓: {vname} recovers {', '.join(sorted(recovered))}")
    else:
        print("  P1 ✗: No variant recovers any bridge-blind protein")

    # P2: Sqrt (E) causes fewer regressions than floor (B)
    e_losses = len(variant_changes.get("E_sqrt_gate", {}).get("losses", []))
    b_losses = len(variant_changes.get("B_floor_gate", {}).get("losses", []))
    p2 = e_losses < b_losses
    predictions["P2"] = p2
    print(f"  P2 {'✓' if p2 else '✗'}: Sqrt regressions ({e_losses}) "
          f"{'<' if p2 else '>='} floor regressions ({b_losses})")

    # P3: Margin-conditional (D) has 0 regressions
    d_losses = len(variant_changes.get("D_margin_cond", {}).get("losses", []))
    p3 = d_losses == 0
    predictions["P3"] = p3
    print(f"  P3 {'✓' if p3 else '✗'}: D_margin_cond regressions = {d_losses}")

    # P4: No variant exceeds 33/52
    max_correct = max(variant_stats[v]["correct"] for v in VARIANTS)
    p4 = max_correct <= 33
    predictions["P4"] = p4
    best_variant = max(VARIANTS.keys(), key=lambda v: variant_stats[v]["correct"])
    print(f"  P4 {'✓' if p4 else '✗'}: Best = {best_variant} "
          f"({variant_stats[best_variant]['correct']}/52), threshold=33")

    # P5: Cytochrome_b5 NOT recovered by floor gate
    cyto_recovered = False
    for vname in ["B_floor_gate", "C_soft_floor"]:
        if "Cytochrome_b5" in results.get(vname, {}):
            if results[vname]["Cytochrome_b5"]["identity"] == "globin":
                cyto_recovered = True
    p5 = not cyto_recovered
    predictions["P5"] = p5
    print(f"  P5 {'✓' if p5 else '✗'}: Cytochrome_b5 "
          f"{'NOT' if p5 else ''} recovered by floor variants")

    confirmed = sum(1 for v in predictions.values() if v)
    total_pred = len(predictions)
    print(f"\n  SCORECARD: {confirmed}/{total_pred} predictions confirmed")

    # ── Save results ───────────────────────────────────────────────
    output = {
        "experiment": "D168",
        "title": "Conditional Bridge — Surgical α₈ De-gating",
        "corpus_size": len(corpus),
        "loaded": n_loaded,
        "alpha8_floor": ALPHA8_FLOOR,
        "bridge_blind": sorted(BRIDGE_BLIND),
        "variant_accuracy": {
            vname: {
                "correct": variant_stats[vname]["correct"],
                "total": variant_stats[vname]["total"],
                "pct": round(variant_stats[vname]["pct"], 1),
                "delta": variant_stats[vname]["correct"] - baseline_correct,
            }
            for vname in VARIANTS
        },
        "variant_changes": {
            vname: {
                "gains": variant_changes.get(vname, {}).get("gains", []),
                "losses": variant_changes.get(vname, {}).get("losses", []),
                "flips": variant_changes.get(vname, {}).get("flips", []),
            }
            for vname in list(VARIANTS.keys())[1:]
        },
        "bridge_blind_results": {},
        "predictions": {k: v for k, v in predictions.items()},
        "predictions_confirmed": confirmed,
        "predictions_total": total_pred,
    }

    # Bridge-blind detail
    for name in sorted(BRIDGE_BLIND):
        if name not in protein_data:
            continue
        entry = protein_data[name]["entry"]
        ms = protein_data[name]["meta_state"]
        output["bridge_blind_results"][name] = {
            "truth": entry.archetype,
            "alpha_0": ms["alpha_0"],
            "alpha_8": ms["alpha_8"],
            "variant_predictions": {
                vname: results[vname][name]["identity"]
                for vname in VARIANTS if name in results[vname]
            },
            "variant_scores": {
                vname: {k: round(v, 4) for k, v in results[vname][name]["scores"].items()}
                for vname in VARIANTS if name in results[vname]
            },
        }

    results_path = RESULTS_DIR / "d168_conditional_bridge.json"
    results_path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"\n  Results saved to {results_path}")


if __name__ == "__main__":
    main()
