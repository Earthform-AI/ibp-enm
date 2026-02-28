#!/usr/bin/env python
"""D160: Route-Score Gated Context-Boost Corrections.

Tests 4 scoring variants on the full 52-protein EXPANDED_CORPUS using
cached profiles + freshly-computed structural data for the lens stack.

Variants (all with 0 new free parameters):
  A: Baseline (current AlgebraicFickBalancer formula)
  B: Route-score gated context_boost
     bridge = rs[arch] × context_boost + α₈ × fano_bridge
     (damps structurally-unmotivated context boosts)
  C: De-doubled α₈ (remove α₈ from bridge_weight)
     bridge_weight = 0.5 × (1-α₀)
     (recovers bridge-blind proteins)
  D: Combined (B + C)

Key finding from D159 error analysis:
  - α₈ appears TWICE: bridge_weight ∝ α₈ AND bridge ∝ α₈×fano_bridge
  - This double-gates: fano_bridge scales as α₈², context_boost as α₈
  - 4 bridge-blind proteins (α₈<0.2) get no context_boost at all
  - Route score is the most actionable improvement signal

Usage:
    python experiments/discovery_160_route_gated_corrections.py
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ibp_enm.benchmark import EXPANDED_CORPUS, ProteinEntry
from ibp_enm.archetypes import ARCHETYPE_EXPECTATIONS
from ibp_enm.synthesis import AlgebraicFickBalancer
from ibp_enm.belief_algebra import ZDPairSelector, FANO_LINES
from ibp_enm.cache import profiles_from_json
from ibp_enm.lens_stack import LensStackSynthesizer, build_default_stack
from ibp_enm.band import _fetch_ca, build_laplacian, ThermodynamicBand
from ibp_enm.analyzer import IBPProteinAnalyzer

CACHE_DIR = Path.home() / ".ibp_enm_cache"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
ALL_ARCHS = list(ARCHETYPE_EXPECTATIONS.keys())

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
    """Compute per-archetype route_score from instrument votes.

    For each archetype, builds a binary support vector (1 = instrument's
    top vote is that archetype) and computes ZDPairSelector.route_score().
    """
    zdp = ZDPairSelector()
    n = min(len(carver_votes), 7)
    route_scores = {}

    for arch in all_archs:
        # Binary support: which instruments have this archetype as their top?
        support = np.zeros(7, dtype=int)
        for i in range(n):
            top = max(carver_votes[i], key=carver_votes[i].get)
            if top == arch:
                support[i] = 1
        route_scores[arch] = zdp.route_score(support)

    return route_scores


# ── Scoring variants ───────────────────────────────────────────────

SQRT2 = np.sqrt(2)
STRONG_WEIGHT = SQRT2 / (SQRT2 + 1)   # ≈ 0.5858
WEAK_WEIGHT = 1.0 / (SQRT2 + 1)       # ≈ 0.4142
BRIDGE_SCALE = 0.5


def score_variant_a(
    consensus_scores, disagreement_scores, context_boost,
    fano_bridge, alpha_0, alpha_8, route_scores,
):
    """A: Baseline — current AlgebraicFickBalancer formula."""
    bridge_weight = BRIDGE_SCALE * (1.0 - alpha_0) * alpha_8
    main_weight = 1.0 - bridge_weight

    scores = {}
    for arch in ALL_ARCHS:
        strong = alpha_0 * consensus_scores.get(arch, 0)
        weak = (1 - alpha_0) * disagreement_scores.get(arch, 0)
        main = STRONG_WEIGHT * strong + WEAK_WEIGHT * weak
        bridge = (context_boost.get(arch, 0)
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
    """B: Route-score gated context_boost.

    bridge = rs[arch] × context_boost + α₈ × fano_bridge
    bridge_weight unchanged.
    """
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


def score_variant_c(
    consensus_scores, disagreement_scores, context_boost,
    fano_bridge, alpha_0, alpha_8, route_scores,
):
    """C: De-doubled α₈ — remove α₈ from bridge_weight.

    bridge_weight = 0.5 × (1-α₀)  [no α₈ gating]
    bridge = context_boost + α₈ × fano_bridge  [unchanged]
    """
    bridge_weight = BRIDGE_SCALE * (1.0 - alpha_0)
    main_weight = 1.0 - bridge_weight

    scores = {}
    for arch in ALL_ARCHS:
        strong = alpha_0 * consensus_scores.get(arch, 0)
        weak = (1 - alpha_0) * disagreement_scores.get(arch, 0)
        main = STRONG_WEIGHT * strong + WEAK_WEIGHT * weak
        bridge = (context_boost.get(arch, 0)
                  + alpha_8 * fano_bridge.get(arch, 0))
        scores[arch] = main_weight * main + bridge_weight * bridge

    total = sum(scores.values())
    if total > 1e-10:
        scores = {k: v / total for k, v in scores.items()}
    return scores


def score_variant_d(
    consensus_scores, disagreement_scores, context_boost,
    fano_bridge, alpha_0, alpha_8, route_scores,
):
    """D: Combined — route-score gated + de-doubled α₈.

    bridge_weight = 0.5 × (1-α₀)  [no α₈ gating]
    bridge = rs × context_boost + α₈ × fano_bridge
    """
    bridge_weight = BRIDGE_SCALE * (1.0 - alpha_0)
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


VARIANTS = {
    "A_baseline": score_variant_a,
    "B_route_gated": score_variant_b,
    "C_de_doubled": score_variant_c,
    "D_combined": score_variant_d,
}


def rescore_protein(
    profiles, meta_state, base_result,
    evals, evecs, domain_labels, contacts, N,
    pdb_id, chain, variant_fn,
):
    """Re-score using a variant scoring function + lens stack.

    1. Extract intermediate values from base (MetaFick) synthesis
    2. Compute variant-specific pre-lens scores
    3. Run lens stack on top for final identity
    """
    carver_votes = [p.archetype_vote() for p in profiles]

    # Get AlgebraicFickBalancer intermediates
    alpha_0 = meta_state.get("alpha_0", 0.5)
    alpha_8 = meta_state.get("alpha_8", 0.0)

    consensus_scores = base_result["consensus_scores"]
    disagreement_scores = base_result["disagreement_scores"]
    context_boost = base_result["context_boost"]

    # Compute Fano bridge (SedenonBridge)
    balancer = AlgebraicFickBalancer()
    fano_bridge = balancer._hamming_bridge.bridge_scores(carver_votes, ALL_ARCHS)

    # Compute per-archetype route scores
    route_scores = compute_per_arch_route_scores(carver_votes, ALL_ARCHS)

    # Variant-specific scoring (pre-lens)
    pre_lens_scores = variant_fn(
        consensus_scores, disagreement_scores, context_boost,
        fano_bridge, alpha_0, alpha_8, route_scores,
    )

    # Build and apply lens stack
    stack = build_default_stack(
        evals=evals, evecs=evecs,
        domain_labels=domain_labels, contacts=contacts,
        pdb_id=pdb_id, chain=chain, n_residues=N,
    )
    context = {
        "evals": evals,
        "evecs": evecs,
        "domain_labels": domain_labels,
        "contacts": contacts,
        "pdb_id": pdb_id,
        "chain": chain,
        "n_residues": N,
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
    print(f"D160: Route-Score Gated Context-Boost Corrections")
    print(f"  Corpus: {len(corpus)} proteins")
    print(f"  Variants: {', '.join(VARIANTS.keys())}")
    print(f"  Cache: {CACHE_DIR}")
    print()

    # Phase 1: Load profiles + structural data, compute base synthesis
    print("=" * 72)
    print("PHASE 1: LOADING PROFILES & STRUCTURAL DATA")
    print("=" * 72)

    protein_data = {}
    n_cached = 0
    n_fetched = 0
    t_start = time.perf_counter()

    for i, entry in enumerate(corpus):
        label = f"[{i+1}/{len(corpus)}]"

        # 1. Load cached profiles
        profiles, metadata = load_cached_profiles(entry.pdb_id, entry.chain)
        if profiles is None:
            print(f"  {label} ✗ {entry.name}: no cached profiles!")
            continue
        n_cached += 1

        # 2. Fetch structural data for lens stack
        try:
            evals, evecs, domain_labels, contacts, N = get_structural_data(
                entry.pdb_id, entry.chain)
            n_fetched += 1
        except Exception as exc:
            print(f"  {label} ✗ {entry.name}: structural data error: {exc}")
            continue

        # 3. Compute base (MetaFickBalancer) synthesis for intermediates
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
        print(f"  {label} ✓ {entry.name} (N={N}, α₀={meta_state['alpha_0']:.3f}, α₈={meta_state['alpha_8']:.3f})")

    t_load = time.perf_counter() - t_start
    print(f"\n  Loaded: {n_cached} cached, {n_fetched} structural ({t_load:.1f}s)")

    # Phase 2: Score all variants
    print("\n" + "=" * 72)
    print("PHASE 2: SCORING ALL VARIANTS")
    print("=" * 72)

    results = {}  # {variant_name: {protein_name: result_dict}}
    for vname in VARIANTS:
        results[vname] = {}

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

        # Show baseline result + any variant changes
        a_pred = preds.get("A_baseline", ("?", "?"))
        changes = []
        for vname in ["B_route_gated", "C_de_doubled", "D_combined"]:
            vpred = preds.get(vname, ("?", "?"))
            if vpred[1] != a_pred[1]:
                if vpred[0] == "✓" and a_pred[0] == "✗":
                    changes.append(f"{vname[0]}:+1")
                elif vpred[0] == "✗" and a_pred[0] == "✓":
                    changes.append(f"{vname[0]}:-1")

        change_str = f"  [{', '.join(changes)}]" if changes else ""
        print(f"  {label} {a_pred[0]} {entry.name:<25s} "
              f"truth={entry.archetype:<15s} pred={a_pred[1]:<15s}{change_str}")

    # Phase 3: Accuracy comparison
    print("\n" + "=" * 72)
    print("PHASE 3: VARIANT ACCURACY COMPARISON")
    print("=" * 72)

    variant_correct = {}
    for vname in VARIANTS:
        correct = sum(
            1 for entry in corpus
            if entry.name in results[vname]
            and results[vname][entry.name]["identity"] == entry.archetype
        )
        total = len(results[vname])
        variant_correct[vname] = (correct, total)
        pct = 100 * correct / max(total, 1)
        print(f"  {vname:<20s}: {correct}/{total} ({pct:.1f}%)")

    baseline_correct = variant_correct["A_baseline"][0]
    print()
    print(f"  Baseline: {baseline_correct}/{variant_correct['A_baseline'][1]}")
    for vname in ["B_route_gated", "C_de_doubled", "D_combined"]:
        c, t = variant_correct[vname]
        delta = c - baseline_correct
        print(f"  {vname:<20s}: {c}/{t} (Δ={delta:+d})")

    # Phase 4: Per-variant changes
    print("\n" + "=" * 72)
    print("PHASE 4: PER-PROTEIN CHANGES (vs baseline)")
    print("=" * 72)

    for vname in ["B_route_gated", "C_de_doubled", "D_combined"]:
        gains, losses = [], []
        for entry in corpus:
            if entry.name not in results["A_baseline"] or entry.name not in results[vname]:
                continue
            a_correct = results["A_baseline"][entry.name]["identity"] == entry.archetype
            v_correct = results[vname][entry.name]["identity"] == entry.archetype
            v_pred = results[vname][entry.name]["identity"]
            a_pred = results["A_baseline"][entry.name]["identity"]

            if not a_correct and v_correct:
                gains.append(f"    + {entry.name:<25s} truth={entry.archetype:<15s} was={a_pred:<15s} now={v_pred}")
            elif a_correct and not v_correct:
                losses.append(f"    - {entry.name:<25s} truth={entry.archetype:<15s} was_correct, now={v_pred}")

        print(f"\n  --- {vname} ---")
        print(f"  GAINS ({len(gains)}):")
        for g in gains:
            print(g)
        print(f"  LOSSES ({len(losses)}):")
        for l in losses:
            print(l)

    # Phase 5: Target protein analysis
    print("\n" + "=" * 72)
    print("PHASE 5: TARGET PROTEIN ANALYSIS")
    print("=" * 72)

    targets = {
        "easy_flips": ["Rubisco_large", "Protein_kinase_A"],
        "bridge_blind": ["Neuroglobin", "Cytochrome_b5", "Erythrocruorin", "GroEL_subunit"],
        "high_conf_wrong": ["Chymotrypsin", "Subtilisin", "Glycogen_phosph"],
    }

    for category, names in targets.items():
        print(f"\n  === {category.upper()} ===\n")
        for name in names:
            if name not in results["A_baseline"]:
                print(f"  {name}: not found")
                continue

            entry = next(e for e in corpus if e.name == name)
            print(f"  {name}:")

            for vname in VARIANTS:
                if name not in results[vname]:
                    continue
                vr = results[vname][name]
                correct = "✓" if vr["identity"] == entry.archetype else "✗"
                truth_score = vr["scores"].get(entry.archetype, 0)
                pred_score = vr["scores"].get(vr["identity"], 0)
                margin = truth_score - pred_score if vr["identity"] != entry.archetype else 0

                # Top-2 scores
                sorted_s = sorted(vr["scores"].items(), key=lambda x: -x[1])
                top2 = ", ".join(f"{a}:{s:.3f}" for a, s in sorted_s[:2])

                # Truth rank
                truth_rank = next(
                    (i + 1 for i, (a, _) in enumerate(sorted_s)
                     if a == entry.archetype), 5)

                print(f"    {vname:<20s}: {correct} pred={vr['identity']:<15s} "
                      f"truth_rank={truth_rank} margin={margin:+.4f} top2=({top2})")

    # Phase 6: D159 prediction scorecard
    print("\n" + "=" * 72)
    print("PHASE 6: PREDICTION SCORECARD")
    print("=" * 72)

    best_variant = max(
        ["B_route_gated", "C_de_doubled", "D_combined"],
        key=lambda v: variant_correct[v][0]
    )
    bc, bt = variant_correct[best_variant]
    bl, _ = variant_correct["A_baseline"]
    print(f"\n  Best variant: {best_variant} ({bc}/{bt})")
    print(f"  Baseline: {bl}/{bt}")
    print(f"  Net gain: {bc - bl:+d}")

    # P1: Route score is most actionable signal
    b_gain = variant_correct["B_route_gated"][0] - bl
    print(f"\n  P1: Route-score gating improves accuracy: {'CONFIRMED' if b_gain > 0 else 'REFUTED'}")
    print(f"      B_route_gated Δ={b_gain:+d}")

    # P2: Easy flips within 0.03
    flipped = 0
    for name in ["Rubisco_large", "Protein_kinase_A"]:
        if name in results[best_variant]:
            entry = next(e for e in corpus if e.name == name)
            if results[best_variant][name]["identity"] == entry.archetype:
                flipped += 1
    print(f"\n  P2: Easy flips recovered: {flipped}/2")

    # P3: Bridge-blind improvement
    bridge_flipped = 0
    for name in ["Neuroglobin", "Cytochrome_b5", "Erythrocruorin", "GroEL_subunit"]:
        if name in results[best_variant]:
            entry = next(e for e in corpus if e.name == name)
            if results[best_variant][name]["identity"] == entry.archetype:
                bridge_flipped += 1
    print(f"\n  P3: Bridge-blind recovered: {bridge_flipped}/4")

    # P4: High-confidence wrong remain unfixed
    hcw_fixed = 0
    for name in ["Chymotrypsin", "Subtilisin", "Glycogen_phosph"]:
        if name in results[best_variant]:
            entry = next(e for e in corpus if e.name == name)
            if results[best_variant][name]["identity"] == entry.archetype:
                hcw_fixed += 1
    print(f"\n  P4: High-conf wrong (rule-level): {hcw_fixed}/3 fixed "
          f"({'EXPECTED' if hcw_fixed == 0 else 'UNEXPECTED'})")

    # P5: No regression from baseline on the 30 correct proteins
    baseline_correct_names = set(
        entry.name for entry in corpus
        if entry.name in results["A_baseline"]
        and results["A_baseline"][entry.name]["identity"] == entry.archetype
    )
    best_regressions = [
        name for name in baseline_correct_names
        if name in results[best_variant]
        and results[best_variant][name]["identity"]
            != next(e for e in corpus if e.name == name).archetype
    ]
    print(f"\n  P5: Regressions from best variant: {len(best_regressions)}")
    for r in best_regressions:
        entry = next(e for e in corpus if e.name == r)
        bp = results[best_variant][r]["identity"]
        print(f"      {r}: was correct, now={bp} (truth={entry.archetype})")

    # Save results
    out = {
        "variant_accuracy": {v: {"correct": c, "total": t} for v, (c, t) in variant_correct.items()},
        "best_variant": best_variant,
        "predictions": {
            "P1_route_score_improves": b_gain > 0,
            "P2_easy_flips": flipped,
            "P3_bridge_blind": bridge_flipped,
            "P4_hcw_fixed": hcw_fixed,
            "P5_regressions": len(best_regressions),
        },
        "changes": {},
    }
    for vname in ["B_route_gated", "C_de_doubled", "D_combined"]:
        gains, losses = [], []
        for entry in corpus:
            if entry.name not in results["A_baseline"] or entry.name not in results[vname]:
                continue
            a_correct = results["A_baseline"][entry.name]["identity"] == entry.archetype
            v_correct = results[vname][entry.name]["identity"] == entry.archetype
            if not a_correct and v_correct:
                gains.append(entry.name)
            elif a_correct and not v_correct:
                losses.append(entry.name)
        out["changes"][vname] = {"gains": gains, "losses": losses}

    json_path = RESULTS_DIR / "d160_route_gated.json"
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  Results saved to {json_path}")


if __name__ == "__main__":
    main()
