#!/usr/bin/env python
"""D161: Pivot Instrument Validation — Cooperative (BECOMING).

Tests whether instrument 4 (cooperative → BECOMING/e₅) actually
pivots classification on proteins where its Fano lines carry
conflicting archetype signals.

Background (D150, D157):
  - Cooperative sits on Fano lines 1, 3, 4:
      Line 1 = {1,2,4} = musical, fick, cooperative
      Line 3 = {3,4,6} = thermal, cooperative, fragile
      Line 4 = {4,5,0} = cooperative, propagative, algebraic
  - In carving space, cooperative → BECOMING(6), which lies on
    BOTH the secure (TRANSFORMATION = {6,0,2}) and vulnerable
    (MASTERY = {3,4,6}) lines — it is the pivot point.
  - D160 route-score gating damps context_boost when Fano support
    is weak, but doesn't address whether cooperative itself is
    pivoting correctly.

Experiment structure:
  Phase 1: Load 52-protein corpus (cached profiles + structural data)
  Phase 2: Cooperative diagnostic — per-protein analysis of
           instrument 4's role in routing and support vectors
  Phase 3: Ablation — classify with/without cooperative to identify
           proteins where it pivots the result
  Phase 4: Fano line conflict analysis — for cooperative's 3 lines,
           determine when they carry contradictory signals
  Phase 5: Pivot scorecard — net impact of cooperative as a pivot
  Phase 6: Summary & actionable findings

Variants (0 new free parameters):
  A: Baseline (current production, D160 route-gated)
  B: Cooperative ablated (instrument 4 vote zeroed)

Usage:
    python experiments/discovery_161_pivot_instrument_validation.py
"""

import json
import sys
import time
from collections import Counter, defaultdict
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

COOP_IDX = 4  # Instrument index for cooperative
COOP_LINES = [1, 3, 4]  # Fano lines containing cooperative
INSTRUMENT_NAMES = (
    "algebraic", "musical", "fick",
    "thermal", "cooperative", "propagative", "fragile",
)

# Scoring constants (from AlgebraicFickBalancer)
SQRT2 = np.sqrt(2)
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


def compute_per_arch_support(
    carver_votes: List[Dict[str, float]],
    all_archs: List[str],
) -> Dict[str, np.ndarray]:
    """Per-archetype binary support vectors."""
    n = min(len(carver_votes), 7)
    supports = {}
    for arch in all_archs:
        support = np.zeros(7, dtype=int)
        for i in range(n):
            top = max(carver_votes[i], key=carver_votes[i].get)
            if top == arch:
                support[i] = 1
        supports[arch] = support
    return supports


# ── Scoring functions ──────────────────────────────────────────────

def score_production(
    consensus_scores, disagreement_scores, context_boost,
    fano_bridge, alpha_0, alpha_8, route_scores,
):
    """A: Baseline — current production (D160 route-gated)."""
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


def score_ablated(
    consensus_scores, disagreement_scores, context_boost,
    fano_bridge_ablated, alpha_0, alpha_8, route_scores_ablated,
):
    """B: Cooperative ablated — same formula, but instrument 4 zeroed."""
    bridge_weight = BRIDGE_SCALE * (1.0 - alpha_0) * alpha_8
    main_weight = 1.0 - bridge_weight

    scores = {}
    for arch in ALL_ARCHS:
        strong = alpha_0 * consensus_scores.get(arch, 0)
        weak = (1 - alpha_0) * disagreement_scores.get(arch, 0)
        main = STRONG_WEIGHT * strong + WEAK_WEIGHT * weak
        rs = route_scores_ablated.get(arch, 0)
        bridge = (rs * context_boost.get(arch, 0)
                  + alpha_8 * fano_bridge_ablated.get(arch, 0))
        scores[arch] = main_weight * main + bridge_weight * bridge

    total = sum(scores.values())
    if total > 1e-10:
        scores = {k: v / total for k, v in scores.items()}
    return scores


# ── Per-protein pipeline ───────────────────────────────────────────

def run_protein(
    profiles, meta_state, base_result,
    evals, evecs, domain_labels, contacts, N,
    pdb_id, chain, carver_votes,
):
    """Run both variants and full diagnostic for one protein.

    Returns dict with variant results + cooperative diagnostic.
    """
    alpha_0 = meta_state.get("alpha_0", 0.5)
    alpha_8 = meta_state.get("alpha_8", 0.0)

    consensus_scores = base_result["consensus_scores"]
    disagreement_scores = base_result["disagreement_scores"]
    context_boost = base_result["context_boost"]

    # Full-set Fano bridge + route scores
    balancer = AlgebraicFickBalancer()
    fano_bridge_full = balancer._hamming_bridge.bridge_scores(
        carver_votes, ALL_ARCHS)
    route_scores_full = compute_per_arch_route_scores(carver_votes, ALL_ARCHS)

    # Ablated: zero instrument 4's votes
    ablated_votes = []
    for i, v in enumerate(carver_votes):
        if i == COOP_IDX:
            ablated_votes.append({arch: 0.0 for arch in v})
        else:
            ablated_votes.append(dict(v))

    fano_bridge_ablated = balancer._hamming_bridge.bridge_scores(
        ablated_votes, ALL_ARCHS)
    route_scores_ablated = compute_per_arch_route_scores(
        ablated_votes, ALL_ARCHS)

    # Pre-lens scores: production (A) and ablated (B)
    pre_lens_a = score_production(
        consensus_scores, disagreement_scores, context_boost,
        fano_bridge_full, alpha_0, alpha_8, route_scores_full)
    pre_lens_b = score_ablated(
        consensus_scores, disagreement_scores, context_boost,
        fano_bridge_ablated, alpha_0, alpha_8, route_scores_ablated)

    # Lens stack for both
    stack = build_default_stack(
        evals=evals, evecs=evecs,
        domain_labels=domain_labels, contacts=contacts,
        pdb_id=pdb_id, chain=chain, n_residues=N)
    ctx = {
        "evals": evals, "evecs": evecs,
        "domain_labels": domain_labels, "contacts": contacts,
        "pdb_id": pdb_id, "chain": chain, "n_residues": N,
    }

    final_a, traces_a = stack.apply(pre_lens_a, profiles, ctx)
    identity_a = max(final_a, key=final_a.get)

    final_b, traces_b = stack.apply(pre_lens_b, profiles, ctx)
    identity_b = max(final_b, key=final_b.get)

    # ── Cooperative diagnostic ──────────────────────────────────
    zdp = ZDPairSelector()
    supports_full = compute_per_arch_support(carver_votes, ALL_ARCHS)
    supports_ablated = compute_per_arch_support(ablated_votes, ALL_ARCHS)

    # What cooperative says (its top archetype)
    coop_vote = carver_votes[COOP_IDX] if len(carver_votes) > COOP_IDX else {}
    coop_top_arch = max(coop_vote, key=coop_vote.get) if coop_vote else None
    coop_top_score = coop_vote.get(coop_top_arch, 0) if coop_top_arch else 0

    # Per-line conflict analysis for cooperative's lines
    line_arch_map = {}  # line_idx -> which arch activates it most
    for line_idx in COOP_LINES:
        line_members = set(FANO_LINES[line_idx])
        # For each archetype, count how many of this line's members
        # support it (i.e., have it as their top vote)
        line_support = {}
        for arch in ALL_ARCHS:
            count = sum(1 for m in line_members
                        if supports_full[arch][m] > 0)
            if count > 0:
                line_support[arch] = count
        line_arch_map[line_idx] = line_support

    # Detect conflict: cooperative's lines favour different archetypes
    line_top_archs = {}
    for line_idx, lsup in line_arch_map.items():
        if lsup:
            line_top_archs[line_idx] = max(lsup, key=lsup.get)
        else:
            line_top_archs[line_idx] = None

    top_archs_on_lines = set(a for a in line_top_archs.values() if a)
    has_conflict = len(top_archs_on_lines) > 1

    # Fano activation difference (with vs without cooperative)
    # for truth archetype
    diag = {
        "coop_top_arch": coop_top_arch,
        "coop_top_score": float(coop_top_score),
        "coop_vote_all": {k: float(v) for k, v in coop_vote.items()}
            if coop_vote else {},
        "line_arch_support": {
            str(li): {a: c for a, c in ls.items()}
            for li, ls in line_arch_map.items()
        },
        "line_top_archs": {str(li): a for li, a in line_top_archs.items()},
        "has_line_conflict": has_conflict,
        "conflicting_archs": sorted(top_archs_on_lines),
        "route_scores_full": route_scores_full,
        "route_scores_ablated": route_scores_ablated,
    }

    return {
        "identity_a": identity_a,
        "scores_a": final_a,
        "identity_b": identity_b,
        "scores_b": final_b,
        "diagnosis": diag,
        "alpha_0": alpha_0,
        "alpha_8": alpha_8,
    }


# ── Main experiment ────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    corpus = list(EXPANDED_CORPUS)
    print("D161: Pivot Instrument Validation — Cooperative (BECOMING)")
    print(f"  Corpus: {len(corpus)} proteins")
    print(f"  Cooperative = instrument {COOP_IDX} ({INSTRUMENT_NAMES[COOP_IDX]})")
    print(f"  Fano lines: {COOP_LINES} = "
          + ", ".join(f"L{l}={set(FANO_LINES[l])}" for l in COOP_LINES))
    print(f"  Variants: A (production), B (cooperative ablated)")
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
        print(f"  {label} ✓ {entry.name} (N={N})")

    t_load = time.perf_counter() - t_start
    print(f"\n  Loaded: {len(protein_data)}/{len(corpus)} ({t_load:.1f}s)")

    # ── Phase 2: Run all proteins ──────────────────────────────
    print("\n" + "=" * 72)
    print("PHASE 2: COOPERATIVE DIAGNOSTIC + ABLATION")
    print("=" * 72)

    results = {}
    n_pivot = 0
    n_pivot_correct = 0
    n_pivot_wrong = 0
    n_conflict = 0

    for i, entry in enumerate(corpus):
        if entry.name not in protein_data:
            continue

        pd = protein_data[entry.name]
        label = f"[{i+1}/{len(corpus)}]"

        r = run_protein(
            pd["profiles"], pd["meta_state"], pd["base_result"],
            pd["evals"], pd["evecs"], pd["domain_labels"],
            pd["contacts"], pd["N"],
            entry.pdb_id, entry.chain, pd["carver_votes"])
        results[entry.name] = r

        # Classify pivot behaviour
        identity_changed = r["identity_a"] != r["identity_b"]
        a_correct = r["identity_a"] == entry.archetype
        b_correct = r["identity_b"] == entry.archetype
        conflict = r["diagnosis"]["has_line_conflict"]

        pivot_tag = ""
        if identity_changed:
            n_pivot += 1
            if a_correct and not b_correct:
                pivot_tag = " PIVOT-SAVES ✓"
                n_pivot_correct += 1
            elif not a_correct and b_correct:
                pivot_tag = " PIVOT-HURTS ✗"
                n_pivot_wrong += 1
            else:
                pivot_tag = " PIVOT-NEUTRAL"

        conflict_tag = " CONFLICT" if conflict else ""

        coop_top = r["diagnosis"]["coop_top_arch"] or "?"
        coop_agrees = "✓" if coop_top == entry.archetype else "✗"

        correct_tag = "✓" if a_correct else "✗"
        print(f"  {label} {correct_tag} {entry.name:<25s} "
              f"truth={entry.archetype:<15s} A={r['identity_a']:<15s} "
              f"B={r['identity_b']:<15s} coop→{coop_top}({coop_agrees})"
              f"{pivot_tag}{conflict_tag}")

        if conflict:
            n_conflict += 1

    print(f"\n  Pivot summary: {n_pivot} pivots "
          f"({n_pivot_correct} save, {n_pivot_wrong} hurt, "
          f"{n_pivot - n_pivot_correct - n_pivot_wrong} neutral)")
    print(f"  Line conflicts: {n_conflict}/{len(results)}")

    # ── Phase 3: Accuracy comparison ───────────────────────────
    print("\n" + "=" * 72)
    print("PHASE 3: ACCURACY COMPARISON")
    print("=" * 72)

    correct_a = sum(
        1 for entry in corpus if entry.name in results
        and results[entry.name]["identity_a"] == entry.archetype)
    correct_b = sum(
        1 for entry in corpus if entry.name in results
        and results[entry.name]["identity_b"] == entry.archetype)
    total = len(results)

    print(f"  A (production):         {correct_a}/{total} "
          f"({100*correct_a/max(total,1):.1f}%)")
    print(f"  B (coop ablated):       {correct_b}/{total} "
          f"({100*correct_b/max(total,1):.1f}%)")
    print(f"  Delta (A - B):          {correct_a - correct_b:+d}")
    print(f"\n  → Cooperative net value: "
          f"{'POSITIVE' if correct_a > correct_b else 'NEGATIVE' if correct_a < correct_b else 'ZERO'}")

    # ── Phase 4: Fano line conflict analysis ───────────────────
    print("\n" + "=" * 72)
    print("PHASE 4: FANO LINE CONFLICT ANALYSIS")
    print("=" * 72)

    # Group proteins by conflict pattern
    conflict_proteins = []
    no_conflict_proteins = []

    for entry in corpus:
        if entry.name not in results:
            continue
        r = results[entry.name]
        if r["diagnosis"]["has_line_conflict"]:
            conflict_proteins.append(entry)
        else:
            no_conflict_proteins.append(entry)

    print(f"\n  Proteins WITH line conflict: {len(conflict_proteins)}")
    print(f"  Proteins WITHOUT conflict:   {len(no_conflict_proteins)}")

    # Accuracy among conflict vs non-conflict
    conflict_correct_a = sum(
        1 for e in conflict_proteins
        if results[e.name]["identity_a"] == e.archetype)
    conflict_correct_b = sum(
        1 for e in conflict_proteins
        if results[e.name]["identity_b"] == e.archetype)
    noconflict_correct_a = sum(
        1 for e in no_conflict_proteins
        if results[e.name]["identity_a"] == e.archetype)
    noconflict_correct_b = sum(
        1 for e in no_conflict_proteins
        if results[e.name]["identity_b"] == e.archetype)

    nc = len(conflict_proteins) or 1
    nn = len(no_conflict_proteins) or 1
    print(f"\n  CONFLICT group accuracy:")
    print(f"    A (production):     {conflict_correct_a}/{len(conflict_proteins)} "
          f"({100*conflict_correct_a/nc:.1f}%)")
    print(f"    B (coop ablated):   {conflict_correct_b}/{len(conflict_proteins)} "
          f"({100*conflict_correct_b/nc:.1f}%)")
    print(f"    Coop net:           {conflict_correct_a - conflict_correct_b:+d}")

    print(f"\n  NO-CONFLICT group accuracy:")
    print(f"    A (production):     {noconflict_correct_a}/{len(no_conflict_proteins)} "
          f"({100*noconflict_correct_a/nn:.1f}%)")
    print(f"    B (coop ablated):   {noconflict_correct_b}/{len(no_conflict_proteins)} "
          f"({100*noconflict_correct_b/nn:.1f}%)")
    print(f"    Coop net:           {noconflict_correct_a - noconflict_correct_b:+d}")

    # Detail: per-protein conflict analysis
    print(f"\n  --- Conflict proteins detail ---")
    for entry in conflict_proteins:
        r = results[entry.name]
        d = r["diagnosis"]
        a_ok = "✓" if r["identity_a"] == entry.archetype else "✗"
        b_ok = "✓" if r["identity_b"] == entry.archetype else "✗"
        changed = "PIVOT" if r["identity_a"] != r["identity_b"] else "same"

        line_detail = []
        for li in COOP_LINES:
            ls = d["line_arch_support"].get(str(li), {})
            top = d["line_top_archs"].get(str(li), "?")
            members = FANO_LINES[li]
            line_detail.append(
                f"L{li}({','.join(INSTRUMENT_NAMES[m][:4] for m in members)})→{top}")

        print(f"    {a_ok}→{b_ok} {entry.name:<25s} "
              f"truth={entry.archetype:<15s} {changed:<8s} "
              f"coop→{d['coop_top_arch']}  "
              + "  ".join(line_detail))

    # ── Phase 5: Pivot scorecard ───────────────────────────────
    print("\n" + "=" * 72)
    print("PHASE 5: PIVOT SCORECARD")
    print("=" * 72)

    # Cooperative agreement rate
    coop_agrees_total = sum(
        1 for e in corpus if e.name in results
        and results[e.name]["diagnosis"]["coop_top_arch"] == e.archetype)
    print(f"\n  Cooperative agrees with truth: {coop_agrees_total}/{total} "
          f"({100*coop_agrees_total/max(total,1):.1f}%)")

    # Pivot proteins (where ablation changes prediction)
    pivot_saves = []
    pivot_hurts = []
    pivot_neutral = []

    for entry in corpus:
        if entry.name not in results:
            continue
        r = results[entry.name]
        if r["identity_a"] == r["identity_b"]:
            continue
        a_ok = r["identity_a"] == entry.archetype
        b_ok = r["identity_b"] == entry.archetype
        if a_ok and not b_ok:
            pivot_saves.append(entry.name)
        elif not a_ok and b_ok:
            pivot_hurts.append(entry.name)
        else:
            pivot_neutral.append(entry.name)

    print(f"\n  Pivot proteins (classification changes on ablation):")
    print(f"    Total pivots:  {len(pivot_saves) + len(pivot_hurts) + len(pivot_neutral)}")
    print(f"    SAVES (coop→correct):   {len(pivot_saves)}")
    for name in pivot_saves:
        r = results[name]
        entry = next(e for e in corpus if e.name == name)
        print(f"      {name:<25s} truth={entry.archetype:<15s} "
              f"A={r['identity_a']:<15s} B(no_coop)={r['identity_b']}")

    print(f"    HURTS (coop→wrong):     {len(pivot_hurts)}")
    for name in pivot_hurts:
        r = results[name]
        entry = next(e for e in corpus if e.name == name)
        print(f"      {name:<25s} truth={entry.archetype:<15s} "
              f"A={r['identity_a']:<15s} B(no_coop)={r['identity_b']}")

    print(f"    NEUTRAL (both wrong):   {len(pivot_neutral)}")
    for name in pivot_neutral:
        r = results[name]
        entry = next(e for e in corpus if e.name == name)
        print(f"      {name:<25s} truth={entry.archetype:<15s} "
              f"A={r['identity_a']:<15s} B(no_coop)={r['identity_b']}")

    print(f"\n  NET PIVOT VALUE: {len(pivot_saves) - len(pivot_hurts):+d} proteins")

    # ── Route score shifts on ablation ──
    print(f"\n  --- Route score shifts (truth archetype) ---")
    rs_shifts = []
    for entry in corpus:
        if entry.name not in results:
            continue
        r = results[entry.name]
        rs_full = r["diagnosis"]["route_scores_full"].get(entry.archetype, 0)
        rs_abl = r["diagnosis"]["route_scores_ablated"].get(entry.archetype, 0)
        rs_shifts.append((entry.name, rs_full, rs_abl, rs_full - rs_abl))

    rs_shifts.sort(key=lambda x: -abs(x[3]))
    print(f"  Top-10 largest route_score shifts (truth archetype):")
    for name, full, abl, delta in rs_shifts[:10]:
        entry = next(e for e in corpus if e.name == name)
        r = results[name]
        pivot_tag = ""
        if r["identity_a"] != r["identity_b"]:
            pivot_tag = " *PIVOT*"
        print(f"    {name:<25s} full={full:.4f} ablated={abl:.4f} "
              f"Δ={delta:+.4f}{pivot_tag}")

    # ── Phase 6: Summary ───────────────────────────────────────
    print("\n" + "=" * 72)
    print("PHASE 6: SUMMARY & ACTIONABLE FINDINGS")
    print("=" * 72)

    print(f"""
  COOPERATIVE (instrument 4 / BECOMING / e₅) PIVOT ANALYSIS:

  1. Agreement rate:      {coop_agrees_total}/{total} ({100*coop_agrees_total/max(total,1):.1f}%) proteins
  2. Line conflicts:      {n_conflict}/{total} ({100*n_conflict/max(total,1):.1f}%) proteins
  3. Classification pivots: {n_pivot} proteins changed on ablation
     - Saves (coop→correct):  {len(pivot_saves)}
     - Hurts (coop→wrong):    {len(pivot_hurts)}
     - Neutral (both wrong):  {len(pivot_neutral)}
  4. Net pivot value:     {len(pivot_saves) - len(pivot_hurts):+d}
  5. Accuracy A (prod):   {correct_a}/{total}
  6. Accuracy B (ablated): {correct_b}/{total}

  CONCLUSION: Cooperative is {"A NET POSITIVE PIVOT" if len(pivot_saves) > len(pivot_hurts) else "A NET NEGATIVE PIVOT" if len(pivot_saves) < len(pivot_hurts) else "NEUTRAL"} ({len(pivot_saves) - len(pivot_hurts):+d}).
""")

    if len(pivot_hurts) > 0:
        print("  ACTIONABLE: Proteins where cooperative HURTS are candidates")
        print("  for targeted damping or routing correction in D162+:")
        for name in pivot_hurts:
            r = results[name]
            d = r["diagnosis"]
            entry = next(e for e in corpus if e.name == name)
            print(f"    - {name}: coop→{d['coop_top_arch']}, truth={entry.archetype}, "
                  f"conflict={'yes' if d['has_line_conflict'] else 'no'}")

    if len(pivot_saves) > 0:
        print("\n  VALIDATES: Proteins where cooperative SAVES classification:")
        for name in pivot_saves:
            r = results[name]
            d = r["diagnosis"]
            entry = next(e for e in corpus if e.name == name)
            print(f"    - {name}: coop→{d['coop_top_arch']}, truth={entry.archetype}, "
                  f"conflict={'yes' if d['has_line_conflict'] else 'no'}")

    # Save results
    out = {
        "accuracy_a": correct_a,
        "accuracy_b": correct_b,
        "total": total,
        "delta": correct_a - correct_b,
        "n_pivots": n_pivot,
        "pivot_saves": pivot_saves,
        "pivot_hurts": pivot_hurts,
        "pivot_neutral": pivot_neutral,
        "net_pivot_value": len(pivot_saves) - len(pivot_hurts),
        "n_conflicts": n_conflict,
        "coop_agrees_with_truth": coop_agrees_total,
        "conflict_accuracy_a": conflict_correct_a,
        "conflict_accuracy_b": conflict_correct_b,
        "no_conflict_accuracy_a": noconflict_correct_a,
        "no_conflict_accuracy_b": noconflict_correct_b,
        "per_protein": {},
    }
    for entry in corpus:
        if entry.name not in results:
            continue
        r = results[entry.name]
        d = r["diagnosis"]
        out["per_protein"][entry.name] = {
            "truth": entry.archetype,
            "identity_a": r["identity_a"],
            "identity_b": r["identity_b"],
            "coop_top_arch": d["coop_top_arch"],
            "has_line_conflict": d["has_line_conflict"],
            "conflicting_archs": d["conflicting_archs"],
            "pivot": r["identity_a"] != r["identity_b"],
            "alpha_0": r["alpha_0"],
            "alpha_8": r["alpha_8"],
        }

    json_path = RESULTS_DIR / "d161_pivot_validation.json"
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  Results saved to {json_path}")


if __name__ == "__main__":
    main()
