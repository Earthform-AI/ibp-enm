#!/usr/bin/env python
"""D162: Benchmark Routing-Enhanced Bridge (SedenonBridge vs HammingBridge).

Runs the full 52-protein benchmark comparing the D153 HammingBridge
(mean-threshold Hamming(7,4) syndrome) against the D158 SedenonBridge
(rank-based dual-threshold) through the complete production pipeline
(AlgebraicFickBalancer + D160 route-gating + lens stack).

The task pre-dates D160 deployment: SPRINT predicted SedenonBridge ≥ 30/52.
Production now uses SedenonBridge at 31/52. This experiment confirms the
prediction and measures the exact bridge contribution.

Variants (0 new free parameters):
  A: HammingBridge (D153 — original mean-threshold syndrome)
  B: SedenonBridge (D158 — rank-based dual-threshold, current production)

For each variant, the bridge is swapped inside AlgebraicFickBalancer
while keeping all other pipeline components identical (route-gating,
lens stack, α₀/α₈).

Usage:
    python experiments/discovery_162_bridge_benchmark.py
"""

import json
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
from ibp_enm.belief_algebra import (
    HammingBridge, SedenonBridge, ZDPairSelector, FANO_LINES,
)
from ibp_enm.cache import profiles_from_json
from ibp_enm.lens_stack import build_default_stack
from ibp_enm.band import _fetch_ca, build_laplacian
from ibp_enm.analyzer import IBPProteinAnalyzer

CACHE_DIR = Path.home() / ".ibp_enm_cache"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
ALL_ARCHS = list(ARCHETYPE_EXPECTATIONS.keys())

# Scoring constants (from AlgebraicFickBalancer)
SQRT2 = np.sqrt(2)
STRONG_WEIGHT = SQRT2 / (SQRT2 + 1)
WEAK_WEIGHT = 1.0 / (SQRT2 + 1)
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


# ── Scoring function (D160 production formula) ────────────────────

def score_with_bridge(
    consensus_scores, disagreement_scores, context_boost,
    fano_bridge, alpha_0, alpha_8, route_scores,
):
    """Production scoring: D160 route-gated context boost."""
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


# ── Per-protein pipeline ───────────────────────────────────────────

def run_protein_with_bridge(
    bridge_obj,
    profiles, meta_state, base_result,
    evals, evecs, domain_labels, contacts, N,
    pdb_id, chain, carver_votes,
):
    """Run the full pipeline with a specific bridge implementation.

    The bridge_obj is either HammingBridge() or SedenonBridge().
    Everything else (route-gating, lens stack, α₀/α₈) is identical.
    """
    alpha_0 = meta_state.get("alpha_0", 0.5)
    alpha_8 = meta_state.get("alpha_8", 0.0)

    consensus_scores = base_result["consensus_scores"]
    disagreement_scores = base_result["disagreement_scores"]
    context_boost = base_result["context_boost"]

    # Fano bridge using the specified bridge implementation
    fano_bridge = bridge_obj.bridge_scores(carver_votes, ALL_ARCHS)

    # Route scores (same for both variants — independent of bridge)
    route_scores = compute_per_arch_route_scores(carver_votes, ALL_ARCHS)

    # Pre-lens scores
    pre_lens_scores = score_with_bridge(
        consensus_scores, disagreement_scores, context_boost,
        fano_bridge, alpha_0, alpha_8, route_scores)

    # Lens stack
    stack = build_default_stack(
        evals=evals, evecs=evecs,
        domain_labels=domain_labels, contacts=contacts,
        pdb_id=pdb_id, chain=chain, n_residues=N)
    ctx = {
        "evals": evals, "evecs": evecs,
        "domain_labels": domain_labels, "contacts": contacts,
        "pdb_id": pdb_id, "chain": chain, "n_residues": N,
    }

    final_scores, traces = stack.apply(pre_lens_scores, profiles, ctx)
    identity = max(final_scores, key=final_scores.get)

    # Bridge diagnostics
    diag = bridge_obj.diagnose(carver_votes, ALL_ARCHS)

    return {
        "identity": identity,
        "scores": final_scores,
        "pre_lens_scores": pre_lens_scores,
        "fano_bridge": fano_bridge,
        "route_scores": route_scores,
        "bridge_diag": diag,
        "alpha_0": alpha_0,
        "alpha_8": alpha_8,
    }


# ── Main experiment ────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    corpus = list(EXPANDED_CORPUS)
    print("D162: Benchmark Routing-Enhanced Bridge")
    print(f"  Corpus: {len(corpus)} proteins")
    print(f"  Variants: A (HammingBridge/D153), B (SedenonBridge/D158)")
    print(f"  Pipeline: AlgebraicFickBalancer + D160 route-gating + lens stack")
    print(f"  Prediction: SedenonBridge ≥ 30/52 (target 31+)")
    print()

    # ── Phase 1: Load data ─────────────────────────────────────
    print("=" * 72)
    print("PHASE 1: LOADING PROFILES & STRUCTURAL DATA")
    print("=" * 72)

    protein_data = {}
    t_start = time.perf_counter()

    for i, entry in enumerate(corpus):
        label = f"[{i+1}/{len(corpus)}]"

        profiles, metadata = load_cached_profiles(entry.pdb_id, entry.chain)
        if profiles is None:
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
            "carver_votes": carver_votes,
            "evals": evals,
            "evecs": evecs,
            "domain_labels": domain_labels,
            "contacts": contacts,
            "N": N,
        }
        print(f"  {label} ✓ {entry.name} (N={N}, "
              f"α₀={meta_state['alpha_0']:.3f}, α₈={meta_state['alpha_8']:.3f})")

    t_load = time.perf_counter() - t_start
    print(f"\n  Loaded: {len(protein_data)}/{len(corpus)} ({t_load:.1f}s)")

    # ── Phase 2: Benchmark both bridges ────────────────────────
    print("\n" + "=" * 72)
    print("PHASE 2: BENCHMARKING BOTH BRIDGES")
    print("=" * 72)

    hamming = HammingBridge()
    sedenon = SedenonBridge()

    results_a = {}  # HammingBridge
    results_b = {}  # SedenonBridge

    for i, entry in enumerate(corpus):
        if entry.name not in protein_data:
            continue

        pd = protein_data[entry.name]
        label = f"[{i+1}/{len(corpus)}]"

        r_a = run_protein_with_bridge(
            hamming,
            pd["profiles"], pd["meta_state"], pd["base_result"],
            pd["evals"], pd["evecs"], pd["domain_labels"],
            pd["contacts"], pd["N"],
            entry.pdb_id, entry.chain, pd["carver_votes"])
        results_a[entry.name] = r_a

        r_b = run_protein_with_bridge(
            sedenon,
            pd["profiles"], pd["meta_state"], pd["base_result"],
            pd["evals"], pd["evecs"], pd["domain_labels"],
            pd["contacts"], pd["N"],
            entry.pdb_id, entry.chain, pd["carver_votes"])
        results_b[entry.name] = r_b

        a_ok = "✓" if r_a["identity"] == entry.archetype else "✗"
        b_ok = "✓" if r_b["identity"] == entry.archetype else "✗"
        changed = ""
        if r_a["identity"] != r_b["identity"]:
            if b_ok == "✓" and a_ok == "✗":
                changed = "  SED+1"
            elif b_ok == "✗" and a_ok == "✓":
                changed = "  SED-1"
            else:
                changed = "  DIFF"

        print(f"  {label} H:{a_ok} S:{b_ok} {entry.name:<25s} "
              f"truth={entry.archetype:<15s} "
              f"H={r_a['identity']:<15s} S={r_b['identity']:<15s}{changed}")

    # ── Phase 3: Accuracy comparison ───────────────────────────
    print("\n" + "=" * 72)
    print("PHASE 3: ACCURACY COMPARISON")
    print("=" * 72)

    correct_a = sum(
        1 for e in corpus if e.name in results_a
        and results_a[e.name]["identity"] == e.archetype)
    correct_b = sum(
        1 for e in corpus if e.name in results_b
        and results_b[e.name]["identity"] == e.archetype)
    total = len(results_a)

    pct_a = 100 * correct_a / max(total, 1)
    pct_b = 100 * correct_b / max(total, 1)
    delta = correct_b - correct_a

    print(f"\n  A (HammingBridge/D153):   {correct_a}/{total} ({pct_a:.1f}%)")
    print(f"  B (SedenonBridge/D158):   {correct_b}/{total} ({pct_b:.1f}%)")
    print(f"  Delta (B - A):            {delta:+d}")
    print(f"\n  SPRINT prediction (P5): SedenonBridge ≥ 30/52 → "
          f"{'CONFIRMED' if correct_b >= 30 else 'REFUTED'} ({correct_b}/52)")
    print(f"  SPRINT target 31+: "
          f"{'ACHIEVED' if correct_b >= 31 else 'NOT YET'} ({correct_b}/52)")

    # ── Phase 4: Per-protein changes ───────────────────────────
    print("\n" + "=" * 72)
    print("PHASE 4: PER-PROTEIN CHANGES (SedenonBridge vs HammingBridge)")
    print("=" * 72)

    gains = []
    losses = []
    diffs = []

    for entry in corpus:
        if entry.name not in results_a or entry.name not in results_b:
            continue
        ra, rb = results_a[entry.name], results_b[entry.name]
        a_ok = ra["identity"] == entry.archetype
        b_ok = rb["identity"] == entry.archetype

        if not a_ok and b_ok:
            gains.append((entry.name, entry.archetype,
                          ra["identity"], rb["identity"]))
        elif a_ok and not b_ok:
            losses.append((entry.name, entry.archetype,
                           ra["identity"], rb["identity"]))
        elif ra["identity"] != rb["identity"]:
            diffs.append((entry.name, entry.archetype,
                          ra["identity"], rb["identity"]))

    print(f"\n  SedenonBridge GAINS ({len(gains)}):")
    for name, truth, h_pred, s_pred in gains:
        print(f"    + {name:<25s} truth={truth:<15s} "
              f"Hamming→{h_pred:<15s} Sedenon→{s_pred}")

    print(f"\n  SedenonBridge LOSSES ({len(losses)}):")
    for name, truth, h_pred, s_pred in losses:
        print(f"    - {name:<25s} truth={truth:<15s} "
              f"Hamming→{h_pred:<15s} Sedenon→{s_pred}")

    print(f"\n  DIFFERENT but both wrong ({len(diffs)}):")
    for name, truth, h_pred, s_pred in diffs:
        print(f"    ~ {name:<25s} truth={truth:<15s} "
              f"Hamming→{h_pred:<15s} Sedenon→{s_pred}")

    # ── Phase 5: Syndrome statistics comparison ────────────────
    print("\n" + "=" * 72)
    print("PHASE 5: SYNDROME STATISTICS COMPARISON")
    print("=" * 72)

    # Aggregate syndrome stats from diagnose() output
    h_valid = 0
    h_total = 0
    s_valid_top3 = 0
    s_valid_top4 = 0
    s_total = 0
    s_invalid = 0

    for entry in corpus:
        if entry.name not in results_a:
            continue

        # HammingBridge diagnostics
        h_diag = results_a[entry.name]["bridge_diag"]
        h_total += h_diag.get("total_syndromes", 0)
        h_valid += h_diag.get("valid_syndromes", 0)

        # SedenonBridge diagnostics
        s_diag = results_b[entry.name]["bridge_diag"]
        s_total += s_diag.get("total_syndromes", 0)
        for arch_data in s_diag.get("per_archetype", {}).values():
            stype = arch_data.get("syndrome_type", "")
            if stype == "top3_fano":
                s_valid_top3 += 1
            elif stype == "top4_complement":
                s_valid_top4 += 1
            elif stype == "invalid":
                s_invalid += 1

    s_valid = s_valid_top3 + s_valid_top4

    print(f"\n  HammingBridge (D153):")
    print(f"    Total syndromes: {h_total}")
    print(f"    Valid syndromes: {h_valid} "
          f"({100*h_valid/max(h_total,1):.1f}%)")

    print(f"\n  SedenonBridge (D158):")
    print(f"    Total syndromes: {s_total}")
    print(f"    Valid (top-3 Fano):      {s_valid_top3}")
    print(f"    Valid (top-4 complement): {s_valid_top4}")
    print(f"    Valid total:             {s_valid} "
          f"({100*s_valid/max(s_total,1):.1f}%)")
    print(f"    Invalid:                 {s_invalid}")

    # ── Phase 6: Bridge score divergence ───────────────────────
    print("\n" + "=" * 72)
    print("PHASE 6: BRIDGE SCORE DIVERGENCE")
    print("=" * 72)

    max_divergences = []
    for entry in corpus:
        if entry.name not in results_a or entry.name not in results_b:
            continue
        fb_a = results_a[entry.name]["fano_bridge"]
        fb_b = results_b[entry.name]["fano_bridge"]
        max_div = 0
        max_arch = ""
        for arch in ALL_ARCHS:
            div = abs(fb_b.get(arch, 0) - fb_a.get(arch, 0))
            if div > max_div:
                max_div = div
                max_arch = arch
        max_divergences.append((entry.name, max_div, max_arch))

    max_divergences.sort(key=lambda x: -x[1])
    print(f"\n  Top-10 largest fano_bridge divergences (any archetype):")
    for name, div, arch in max_divergences[:10]:
        entry = next(e for e in corpus if e.name == name)
        ra = results_a[name]
        rb = results_b[name]
        h_ok = "✓" if ra["identity"] == entry.archetype else "✗"
        s_ok = "✓" if rb["identity"] == entry.archetype else "✗"
        changed = "*CHANGED*" if ra["identity"] != rb["identity"] else ""
        print(f"    {name:<25s} max_div={div:.4f} on {arch:<15s} "
              f"H:{h_ok} S:{s_ok} {changed}")

    # ── Phase 7: Summary ───────────────────────────────────────
    print("\n" + "=" * 72)
    print("PHASE 7: SUMMARY")
    print("=" * 72)

    print(f"""
  ROUTING-ENHANCED BRIDGE BENCHMARK (D162):

  Pipeline: AlgebraicFickBalancer + D160 route-gating + lens stack

  | Bridge | Accuracy | Syndrome Valid Rate |
  |--------|----------|---------------------|
  | HammingBridge (D153) | {correct_a}/{total} ({pct_a:.1f}%) | {100*h_valid/max(h_total,1):.1f}% |
  | SedenonBridge (D158) | {correct_b}/{total} ({pct_b:.1f}%) | {100*s_valid/max(s_total,1):.1f}% |
  | Delta (Sedenon−Hamming) | {delta:+d} | +{100*s_valid/max(s_total,1) - 100*h_valid/max(h_total,1):.1f}pp |

  Gains: {len(gains)} proteins improved by SedenonBridge
  Losses: {len(losses)} proteins regressed by SedenonBridge
  Net: {len(gains) - len(losses):+d}

  SPRINT P5 (SedenonBridge ≥ 30/52): {'CONFIRMED ✓' if correct_b >= 30 else 'REFUTED ✗'}
  Target 31+: {'ACHIEVED ✓' if correct_b >= 31 else 'NOT YET ✗'}
""")

    # Save results
    out = {
        "accuracy_hamming": correct_a,
        "accuracy_sedenon": correct_b,
        "total": total,
        "delta": delta,
        "p5_confirmed": correct_b >= 30,
        "target_31_achieved": correct_b >= 31,
        "gains": [{"name": n, "truth": t, "hamming": h, "sedenon": s}
                  for n, t, h, s in gains],
        "losses": [{"name": n, "truth": t, "hamming": h, "sedenon": s}
                   for n, t, h, s in losses],
        "diffs": [{"name": n, "truth": t, "hamming": h, "sedenon": s}
                  for n, t, h, s in diffs],
        "syndrome_stats": {
            "hamming": {"total": h_total, "valid": h_valid},
            "sedenon": {
                "total": s_total,
                "valid_top3": s_valid_top3,
                "valid_top4": s_valid_top4,
                "valid_total": s_valid,
                "invalid": s_invalid,
            },
        },
        "per_protein": {},
    }
    for entry in corpus:
        if entry.name not in results_a:
            continue
        ra, rb = results_a[entry.name], results_b[entry.name]
        out["per_protein"][entry.name] = {
            "truth": entry.archetype,
            "hamming_pred": ra["identity"],
            "sedenon_pred": rb["identity"],
            "hamming_correct": ra["identity"] == entry.archetype,
            "sedenon_correct": rb["identity"] == entry.archetype,
            "alpha_0": ra["alpha_0"],
            "alpha_8": ra["alpha_8"],
        }

    json_path = RESULTS_DIR / "d162_bridge_benchmark.json"
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"  Results saved to {json_path}")


if __name__ == "__main__":
    main()
