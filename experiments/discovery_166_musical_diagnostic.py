#!/usr/bin/env python
"""D166: Musical Instrument Diagnostic — Full Autopsy.

Background
----------
D165 identified the musical instrument (index 1, e₂, max mode_scatter)
as the root cause of transitivity error.  Musical has anomalously low
agreement with fick (0.173) and propagative (0.269), creating 82% of
the total transitivity error concentrated on Fano lines 0 and 1:
  - Line 0: (algebraic, musical, thermal) — error 0.212
  - Line 1: (musical, fick, cooperative) — error 0.192

Musical sits on 3 Fano lines: (0,1,3), (1,2,4), (5,6,1) — lines 0, 1,
and 5.  Its low agreement propagates through these lines.

Sprint 10 needs to decide: is musical's disagreement INFORMATIVE
(detects something others miss, e.g. a protein's resonance sensitivity
distinguishes archetypes that scatter/gap metrics cannot) or NOISE
(musical is simply wrong more often, contributing negative signal)?

Method
------
Phase 1: Per-instrument accuracy on all 52 proteins.
         Baseline: what fraction of proteins does each instrument's
         argmax(vote) match truth?  Compare musical to the other 6.
Phase 2: Per-instrument accuracy split by CORRECT vs LOST proteins.
         Does musical get the right answer less often on LOSTs?
Phase 3: Per-archetype accuracy of musical.  Does musical systematically
         fail on specific archetypes (e.g. barrel↔enzyme confusion)?
Phase 4: Musical-specific rule audit.  For each LOST protein, trace
         which musical rules fire and what metric values drive them.
         Compare musical's metric distributions for correct vs confused.
Phase 5: Ablation — classify all 52 proteins without musical.
         Does accuracy improve, stay, or drop?
Phase 6: Fano-line partner analysis — for each of musical's 3 Fano
         lines, measure per-protein agreement with partners.  Identify
         proteins where musical disagrees with BOTH partners.
Phase 7: Informative disagreement test — for the proteins where musical
         alone is correct (its argmax = truth, others' argmax ≠ truth),
         identify what features musical detects.

Predictions
-----------
P1: Musical accuracy < 35% on LOSTs (vs ~40% overall from D161).
P2: Musical's errors cluster on barrel↔enzyme confusion axis.
P3: Musical agrees with fick on < 25% of LOSTs.
P4: Removing musical improves accuracy ≥ 31/52 (net non-negative).
P5: Musical's top-vote is correct when all 3 Fano-line partners agree.

Usage:
    python experiments/discovery_166_musical_diagnostic.py
"""

import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ibp_enm.benchmark import EXPANDED_CORPUS, ProteinEntry
from ibp_enm.archetypes import ARCHETYPE_EXPECTATIONS
from ibp_enm.synthesis import AlgebraicFickBalancer
from ibp_enm.belief_algebra import ZDPairSelector, FANO_LINES, HammingBridge
from ibp_enm.cache import profiles_from_json
from ibp_enm.lens_stack import LensStackSynthesizer, build_default_stack
from ibp_enm.band import _fetch_ca, build_laplacian
from ibp_enm.analyzer import IBPProteinAnalyzer
from ibp_enm.algebra import INSTRUMENT_NAMES
from ibp_enm.rules import apply_rules_traced, RuleFiring

CACHE_DIR = Path.home() / ".ibp_enm_cache"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
ALL_ARCHS = list(ARCHETYPE_EXPECTATIONS.keys())

MUSICAL_IDX = 1  # musical is instrument index 1

# Musical's 3 Fano lines: lines that contain instrument index 1
MUSICAL_LINES = [(i, line) for i, line in enumerate(FANO_LINES) if MUSICAL_IDX in line]
# Should be: [(0, (0,1,3)), (1, (1,2,4)), (5, (5,6,1))]

# Scoring constants (from AlgebraicFickBalancer)
SQRT2 = np.sqrt(2)
STRONG_WEIGHT = SQRT2 / (SQRT2 + 1)
WEAK_WEIGHT = 1.0 / (SQRT2 + 1)
BRIDGE_SCALE = 0.5


# ── Helpers ────────────────────────────────────────────────────────

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


def classify_full_pipeline(profiles, carver_votes, meta_state, base_result,
                           evals, evecs, domain_labels, contacts, N,
                           pdb_id, chain):
    """Run full production pipeline: AlgebraicFickBalancer + lens stack."""
    alpha_0 = meta_state.get("alpha_0", 0.5)
    alpha_8 = meta_state.get("alpha_8", 0.0)

    consensus_scores = base_result["consensus_scores"]
    disagreement_scores = base_result["disagreement_scores"]
    context_boost = base_result["context_boost"]
    fano_bridge = base_result["fano_bridge"]

    # Route scores
    zdp = ZDPairSelector()
    route_scores = {}
    for arch in ALL_ARCHS:
        support = np.zeros(7, dtype=int)
        for i in range(min(len(carver_votes), 7)):
            if max(carver_votes[i], key=carver_votes[i].get) == arch:
                support[i] = 1
        route_scores[arch] = zdp.route_score(support)

    # Pre-lens scores (D160 production formula)
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

    # Apply lens stack
    stack = build_default_stack(
        evals=evals, evecs=evecs,
        domain_labels=domain_labels, contacts=contacts,
        pdb_id=pdb_id, chain=chain, n_residues=N)
    ctx = {
        "evals": evals, "evecs": evecs,
        "domain_labels": domain_labels, "contacts": contacts,
        "pdb_id": pdb_id, "chain": chain, "n_residues": N,
    }
    final_scores, traces = stack.apply(pre_lens, profiles, ctx)
    identity = max(final_scores, key=final_scores.get)
    return {"identity": identity, "scores": final_scores}


def classify_without_instrument(profiles, carver_votes_orig, drop_idx,
                                evals, evecs, domain_labels, contacts, N,
                                pdb_id, chain):
    """Classify with one instrument zeroed out.

    We zero the dropped instrument's votes (equal across archetypes)
    rather than removing it, preserving the 7-instrument Fano geometry.
    """
    # Copy and replace dropped instrument with uniform votes
    carver_votes = list(carver_votes_orig)
    n_arch = len(ALL_ARCHS)
    carver_votes[drop_idx] = {a: 1.0 / n_arch for a in ALL_ARCHS}

    balancer = AlgebraicFickBalancer()
    meta = balancer.compute_meta_fick_state(carver_votes)
    base_result = balancer.synthesize_identity(profiles, meta)

    return classify_full_pipeline(
        profiles, carver_votes, meta, base_result,
        evals, evecs, domain_labels, contacts, N,
        pdb_id, chain)


def log(msg="", end="\n"):
    print(msg, end=end)


def log_section(title):
    log()
    log("=" * 72)
    log(title)
    log("=" * 72)


def log_subsection(title):
    log()
    log(f"── {title} {'─' * max(1, 66 - len(title))}")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    log_section("D166: Musical Instrument Diagnostic")
    log(f"Corpus: {len(EXPANDED_CORPUS)} proteins")
    log(f"Musical index: {MUSICAL_IDX} ({INSTRUMENT_NAMES[MUSICAL_IDX]})")
    log(f"Musical's Fano lines: {MUSICAL_LINES}")
    log()

    # ── Load all proteins ──────────────────────────────────────────
    corpus_data = []
    skipped = []
    for entry in EXPANDED_CORPUS:
        profiles, metadata = load_cached_profiles(entry.pdb_id, entry.chain)
        if profiles is None:
            skipped.append(entry.name)
            continue

        # Structural data for lens stack
        try:
            evals, evecs, domain_labels, contacts, N = get_structural_data(
                entry.pdb_id, entry.chain)
        except Exception as e:
            skipped.append(f"{entry.name}(struct:{e})")
            continue

        # Per-instrument votes
        carver_votes = [p.archetype_vote() for p in profiles]

        # Base synthesis (without lens stack — just for the intermediate values)
        balancer = AlgebraicFickBalancer()
        meta = balancer.compute_meta_fick_state(carver_votes)
        base_result = balancer.synthesize_identity(profiles, meta)

        # Full production classification
        full_result = classify_full_pipeline(
            profiles, carver_votes, meta, base_result,
            evals, evecs, domain_labels, contacts, N,
            entry.pdb_id, entry.chain)
        predicted = full_result["identity"]
        truth = entry.archetype

        # Traced musical audit
        musical_profile = profiles[MUSICAL_IDX]
        musical_votes, musical_firings = musical_profile.archetype_vote_traced()

        corpus_data.append({
            "entry": entry,
            "profiles": profiles,
            "metadata": metadata,
            "evals": evals,
            "evecs": evecs,
            "domain_labels": domain_labels,
            "contacts": contacts,
            "N": N,
            "carver_votes": carver_votes,
            "meta": meta,
            "base_result": base_result,
            "predicted": predicted,
            "truth": truth,
            "correct": predicted == truth,
            "musical_votes": musical_votes,
            "musical_firings": musical_firings,
        })

    n = len(corpus_data)
    n_correct = sum(1 for d in corpus_data if d["correct"])
    n_lost = n - n_correct
    log(f"Loaded: {n} proteins ({len(skipped)} skipped)")
    log(f"Production accuracy: {n_correct}/{n} ({100*n_correct/n:.1f}%)")
    if skipped:
        log(f"  Skipped: {', '.join(skipped)}")

    # ──────────────────────────────────────────────────────────────
    # PHASE 1: Per-instrument accuracy (argmax = truth?)
    # ──────────────────────────────────────────────────────────────
    log_section("Phase 1: Per-Instrument Accuracy (All 52)")

    inst_accuracy = {name: {"correct": 0, "total": 0} for name in INSTRUMENT_NAMES}
    for d in corpus_data:
        truth = d["truth"]
        for i, name in enumerate(INSTRUMENT_NAMES):
            votes = d["carver_votes"][i]
            top_vote = max(votes, key=votes.get)
            inst_accuracy[name]["total"] += 1
            if top_vote == truth:
                inst_accuracy[name]["correct"] += 1

    log(f"\n{'Instrument':<14} {'Correct':>8} {'Total':>6} {'Accuracy':>9}")
    log("-" * 40)
    inst_acc_pcts = {}
    for name in INSTRUMENT_NAMES:
        c = inst_accuracy[name]["correct"]
        t = inst_accuracy[name]["total"]
        pct = 100 * c / t if t > 0 else 0
        inst_acc_pcts[name] = pct
        marker = " <<<" if name == "musical" else ""
        log(f"  {name:<12} {c:>8} {t:>6} {pct:>8.1f}%{marker}")

    musical_acc_all = inst_acc_pcts["musical"]
    mean_acc_all = np.mean(list(inst_acc_pcts.values()))
    log(f"\n  Musical: {musical_acc_all:.1f}%  Mean: {mean_acc_all:.1f}%  "
        f"Delta: {musical_acc_all - mean_acc_all:+.1f}pp")

    # ──────────────────────────────────────────────────────────────
    # PHASE 2: Accuracy split CORRECT vs LOST
    # ──────────────────────────────────────────────────────────────
    log_section("Phase 2: Per-Instrument Accuracy (CORRECT vs LOST)")

    for subset_name, filter_fn in [("CORRECT", lambda d: d["correct"]),
                                    ("LOST", lambda d: not d["correct"])]:
        subset = [d for d in corpus_data if filter_fn(d)]
        log_subsection(f"{subset_name} proteins ({len(subset)})")
        log(f"  {'Instrument':<14} {'Correct':>8} {'Total':>6} {'Accuracy':>9}")
        log("  " + "-" * 38)
        for name in INSTRUMENT_NAMES:
            idx = list(INSTRUMENT_NAMES).index(name)
            c = sum(1 for d in subset
                    if max(d["carver_votes"][idx], key=d["carver_votes"][idx].get) == d["truth"])
            t = len(subset)
            pct = 100 * c / t if t > 0 else 0
            marker = " <<<" if name == "musical" else ""
            log(f"    {name:<12} {c:>8} {t:>6} {pct:>8.1f}%{marker}")

    musical_lost_correct = sum(
        1 for d in corpus_data if not d["correct"]
        and max(d["musical_votes"], key=d["musical_votes"].get) == d["truth"]
    )
    musical_acc_lost = 100 * musical_lost_correct / n_lost if n_lost > 0 else 0
    log(f"\n  Musical accuracy on LOSTs: {musical_acc_lost:.1f}%")

    # ──────────────────────────────────────────────────────────────
    # PHASE 3: Per-archetype accuracy of musical
    # ──────────────────────────────────────────────────────────────
    log_section("Phase 3: Musical Per-Archetype Accuracy")

    arch_stats = {a: {"correct": 0, "total": 0, "predicted_as": Counter()}
                  for a in ALL_ARCHS}
    for d in corpus_data:
        truth = d["truth"]
        musical_top = max(d["musical_votes"], key=d["musical_votes"].get)
        arch_stats[truth]["total"] += 1
        arch_stats[truth]["predicted_as"][musical_top] += 1
        if musical_top == truth:
            arch_stats[truth]["correct"] += 1

    log(f"\n  {'Truth Arch':<18} {'Musical Correct':>15} {'Accuracy':>9} {'Confusions'}")
    log("  " + "-" * 70)
    for arch in ALL_ARCHS:
        s = arch_stats[arch]
        pct = 100 * s["correct"] / s["total"] if s["total"] > 0 else 0
        confusions = {k: v for k, v in s["predicted_as"].items() if k != arch}
        conf_str = ", ".join(f"{k}:{v}" for k, v in sorted(confusions.items(),
                                                              key=lambda x: -x[1]))
        log(f"    {arch:<16} {s['correct']:>6}/{s['total']:<6} {pct:>8.1f}%  {conf_str}")

    # Musical confusion matrix
    log_subsection("Musical Confusion Matrix")
    log(f"  {'Truth \\ Pred':<18}", end="")
    for a in ALL_ARCHS:
        log(f" {a[:8]:>8}", end="")
    log()
    for truth_arch in ALL_ARCHS:
        log(f"    {truth_arch:<16}", end="")
        for pred_arch in ALL_ARCHS:
            cnt = arch_stats[truth_arch]["predicted_as"].get(pred_arch, 0)
            marker = "*" if truth_arch == pred_arch else " "
            log(f" {cnt:>7}{marker}", end="")
        log()

    # ──────────────────────────────────────────────────────────────
    # PHASE 4: Musical rule audit on LOSTs
    # ──────────────────────────────────────────────────────────────
    log_section("Phase 4: Musical Rule Audit on LOSTs")

    lost_data = [d for d in corpus_data if not d["correct"]]

    # Aggregate which rules fire most on LOSTs
    rule_fire_counts = Counter()
    rule_scores = defaultdict(float)
    for d in lost_data:
        for firing in d["musical_firings"]:
            if firing.instrument in ("musical", "*"):
                rule_fire_counts[firing.rule_name] += 1
                rule_scores[firing.rule_name] += firing.score

    log(f"\n  Musical rules firing on {n_lost} LOSTs:")
    log(f"  {'Rule':<35} {'Fires':>6} {'Total Score':>12} {'Archetype'}")
    log("  " + "-" * 65)
    for rule_name, count in rule_fire_counts.most_common(20):
        # Find archetype from first firing
        arch = ""
        for d in lost_data:
            for f in d["musical_firings"]:
                if f.rule_name == rule_name:
                    arch = f.archetype
                    break
            if arch:
                break
        log(f"    {rule_name:<33} {count:>6} {rule_scores[rule_name]:>11.2f}  {arch}")

    # Per-LOST detail for musical
    log_subsection("Per-LOST Musical Diagnostic")
    for d in lost_data:
        entry = d["entry"]
        truth = d["truth"]
        pred = d["predicted"]
        mvotes = d["musical_votes"]
        mtop = max(mvotes, key=mvotes.get)
        musical_correct = mtop == truth

        log(f"\n  {entry.name} (truth={truth}, pred={pred})")
        log(f"    Musical top: {mtop} (score={mvotes[mtop]:.3f})"
            f"{'  ✓' if musical_correct else '  ✗'}")
        log(f"    Musical votes: {', '.join(f'{a}={v:.3f}' for a, v in sorted(mvotes.items(), key=lambda x: -x[1]))}")

        # Key musical rules that fired
        musical_rules = [f for f in d["musical_firings"]
                         if f.instrument in ("musical", "*") and f.score > 0.05]
        if musical_rules:
            log(f"    Key rules: ", end="")
            log(", ".join(f"{f.rule_name}→{f.archetype}(+{f.score:.2f})"
                          for f in sorted(musical_rules, key=lambda x: -x.score)[:5]))

    # ──────────────────────────────────────────────────────────────
    # PHASE 5: Ablation — classify without musical
    # ──────────────────────────────────────────────────────────────
    log_section("Phase 5: Ablation — Without Musical")

    ablation_results = {}
    for name_idx, name in enumerate(INSTRUMENT_NAMES):
        correct_count = 0
        changes = []
        for d in corpus_data:
            abl_result = classify_without_instrument(
                d["profiles"], d["carver_votes"], name_idx,
                d["evals"], d["evecs"], d["domain_labels"],
                d["contacts"], d["N"],
                d["entry"].pdb_id, d["entry"].chain)
            abl_pred = abl_result["identity"]
            truth = d["truth"]
            if abl_pred == truth:
                correct_count += 1
            if abl_pred != d["predicted"]:
                changes.append({
                    "name": d["entry"].name,
                    "truth": truth,
                    "prod_pred": d["predicted"],
                    "abl_pred": abl_pred,
                    "gain": abl_pred == truth and d["predicted"] != truth,
                    "loss": abl_pred != truth and d["predicted"] == truth,
                })
        ablation_results[name] = {
            "accuracy": correct_count,
            "n_changes": len(changes),
            "gains": sum(1 for c in changes if c["gain"]),
            "losses": sum(1 for c in changes if c["loss"]),
            "net": sum(1 for c in changes if c["gain"]) - sum(1 for c in changes if c["loss"]),
            "changes": changes,
        }

    log(f"\n  {'Instrument':<14} {'Ablated Acc':>12} {'Changes':>8} {'Gains':>6} {'Losses':>7} {'Net':>5}")
    log("  " + "-" * 55)
    for name in INSTRUMENT_NAMES:
        r = ablation_results[name]
        marker = " <<<" if name == "musical" else ""
        log(f"    {name:<12} {r['accuracy']:>5}/{n:<4} {r['n_changes']:>8} "
            f"{r['gains']:>6} {r['losses']:>7} {r['net']:>+5}{marker}")

    # Musical-specific ablation detail
    musical_abl = ablation_results["musical"]
    log_subsection("Musical Ablation Changes")
    for c in musical_abl["changes"]:
        direction = "GAIN" if c["gain"] else "LOSS" if c["loss"] else "SWAP"
        log(f"    {direction}: {c['name']} — truth={c['truth']}, "
            f"prod={c['prod_pred']}, ablated={c['abl_pred']}")

    # ──────────────────────────────────────────────────────────────
    # PHASE 6: Fano-line partner analysis for musical
    # ──────────────────────────────────────────────────────────────
    log_section("Phase 6: Fano-Line Partner Agreement (Musical's Lines)")

    for line_idx, line in MUSICAL_LINES:
        partners = [i for i in line if i != MUSICAL_IDX]
        partner_names = [INSTRUMENT_NAMES[i] for i in partners]
        log_subsection(f"Line {line_idx}: {tuple(INSTRUMENT_NAMES[i] for i in line)}")

        # Per-protein: does musical agree with each partner?
        agree_with_both = 0
        agree_with_one = 0
        agree_with_none = 0
        disagree_all_wrong = []  # musical disagrees with both AND all are wrong

        for d in corpus_data:
            votes = d["carver_votes"]
            musical_top = max(votes[MUSICAL_IDX], key=votes[MUSICAL_IDX].get)
            partner_tops = [max(votes[p], key=votes[p].get) for p in partners]

            agrees = [musical_top == pt for pt in partner_tops]
            if all(agrees):
                agree_with_both += 1
            elif any(agrees):
                agree_with_one += 1
            else:
                agree_with_none += 1
                # When musical disagrees with BOTH partners
                truth = d["truth"]
                if musical_top != truth and all(pt != truth for pt in partner_tops):
                    disagree_all_wrong.append(d["entry"].name)

        log(f"    Agrees with both partners: {agree_with_both}/{n} ({100*agree_with_both/n:.1f}%)")
        log(f"    Agrees with one partner:   {agree_with_one}/{n} ({100*agree_with_one/n:.1f}%)")
        log(f"    Disagrees with both:        {agree_with_none}/{n} ({100*agree_with_none/n:.1f}%)")
        if disagree_all_wrong:
            log(f"    All-wrong disagreements: {', '.join(disagree_all_wrong)}")

        # Check: when musical disagrees, who is right?
        musical_right_partner_wrong = 0
        partner_right_musical_wrong = 0
        both_wrong = 0
        both_right = 0

        for d in corpus_data:
            votes = d["carver_votes"]
            truth = d["truth"]
            musical_top = max(votes[MUSICAL_IDX], key=votes[MUSICAL_IDX].get)
            partner_tops = [max(votes[p], key=votes[p].get) for p in partners]

            m_correct = musical_top == truth
            # "partners correct" = majority of partners correct
            p_correct = sum(pt == truth for pt in partner_tops) > 0

            if m_correct and not p_correct:
                musical_right_partner_wrong += 1
            elif not m_correct and p_correct:
                partner_right_musical_wrong += 1
            elif not m_correct and not p_correct:
                both_wrong += 1
            else:
                both_right += 1

        log(f"    Musical right, partners wrong: {musical_right_partner_wrong}")
        log(f"    Partners right, musical wrong: {partner_right_musical_wrong}")
        log(f"    Both right:                    {both_right}")
        log(f"    Both wrong:                    {both_wrong}")

    # ──────────────────────────────────────────────────────────────
    # PHASE 7: Informative disagreement — musical uniquely correct
    # ──────────────────────────────────────────────────────────────
    log_section("Phase 7: Musical Uniquely Correct (Informative Disagreement)")

    # Find proteins where musical's top vote = truth but the consensus
    # (majority of other 6 instruments) would be wrong
    musical_unique = []
    for d in corpus_data:
        votes = d["carver_votes"]
        truth = d["truth"]
        musical_top = max(votes[MUSICAL_IDX], key=votes[MUSICAL_IDX].get)

        if musical_top != truth:
            continue

        # Check if other 6 instruments' majority vote ≠ truth
        other_tops = []
        for i, name in enumerate(INSTRUMENT_NAMES):
            if i == MUSICAL_IDX:
                continue
            other_tops.append(max(votes[i], key=votes[i].get))

        majority = Counter(other_tops).most_common(1)[0][0]
        if majority != truth:
            musical_unique.append({
                "name": d["entry"].name,
                "truth": truth,
                "musical_top": musical_top,
                "other_majority": majority,
                "other_tops": {INSTRUMENT_NAMES[i]: max(votes[i], key=votes[i].get)
                               for i in range(7) if i != MUSICAL_IDX},
                "correct": d["correct"],
            })

    log(f"\n  Proteins where musical alone is correct ({len(musical_unique)}):")
    for mu in musical_unique:
        others_str = ", ".join(f"{k}={v}" for k, v in mu["other_tops"].items())
        prod_marker = "✓" if mu["correct"] else "✗"
        log(f"    {mu['name']}: truth={mu['truth']}, musical={mu['musical_top']}, "
            f"other_majority={mu['other_majority']} [{prod_marker}]")
        log(f"      Others: {others_str}")

    # Also: proteins where musical is the SOLE dissenter (wrong when all agree)
    log_subsection("Musical as Sole Dissenter (wrong when all others agree)")
    sole_dissenter = []
    for d in corpus_data:
        votes = d["carver_votes"]
        truth = d["truth"]
        musical_top = max(votes[MUSICAL_IDX], key=votes[MUSICAL_IDX].get)

        other_tops = [max(votes[i], key=votes[i].get)
                      for i in range(7) if i != MUSICAL_IDX]
        # All 6 others agree
        if len(set(other_tops)) == 1 and other_tops[0] != musical_top:
            sole_dissenter.append({
                "name": d["entry"].name,
                "truth": truth,
                "musical": musical_top,
                "others": other_tops[0],
                "musical_correct": musical_top == truth,
                "others_correct": other_tops[0] == truth,
            })

    log(f"  Musical sole dissenter: {len(sole_dissenter)} proteins")
    for sd in sole_dissenter:
        m_mark = "✓" if sd["musical_correct"] else "✗"
        o_mark = "✓" if sd["others_correct"] else "✗"
        log(f"    {sd['name']}: truth={sd['truth']}, "
            f"musical={sd['musical']}{m_mark}, "
            f"others={sd['others']}{o_mark}")

    # ──────────────────────────────────────────────────────────────
    # PHASE 8: Key metric distributions for musical (CORRECT vs LOST)
    # ──────────────────────────────────────────────────────────────
    log_section("Phase 8: Musical Feature Distributions (CORRECT vs LOST)")

    metrics_to_check = [
        "scatter_normalised", "mean_scatter", "mean_delta_beta",
        "mean_ipr", "entropy_volatility", "mean_spatial_radius",
    ]

    correct_data = [d for d in corpus_data if d["correct"]]

    for metric_name in metrics_to_check:
        correct_vals = []
        lost_vals = []
        for d in corpus_data:
            prof = d["profiles"][MUSICAL_IDX]
            try:
                val = getattr(prof, metric_name)
                if val is not None:
                    if d["correct"]:
                        correct_vals.append(val)
                    else:
                        lost_vals.append(val)
            except (AttributeError, TypeError):
                pass

        if correct_vals and lost_vals:
            c_mean = np.mean(correct_vals)
            l_mean = np.mean(lost_vals)
            c_std = np.std(correct_vals)
            l_std = np.std(lost_vals)
            sep = abs(c_mean - l_mean) / (0.5 * (c_std + l_std) + 1e-10)
            log(f"  {metric_name:<25} CORRECT: {c_mean:.4f}±{c_std:.4f}  "
                f"LOST: {l_mean:.4f}±{l_std:.4f}  sep={sep:.2f}")
        else:
            log(f"  {metric_name:<25} (insufficient data)")

    # ──────────────────────────────────────────────────────────────
    # PHASE 9: Agreement matrix (from D165) + musical's row
    # ──────────────────────────────────────────────────────────────
    log_section("Phase 9: Musical Agreement Row (Pairwise)")

    # Compute 7×7 agreement matrix
    agree_matrix = np.zeros((7, 7))
    for d in corpus_data:
        tops = [max(d["carver_votes"][i], key=d["carver_votes"][i].get)
                for i in range(7)]
        for i in range(7):
            for j in range(7):
                if tops[i] == tops[j]:
                    agree_matrix[i, j] += 1
    agree_matrix /= n

    # Show musical's row
    log(f"\n  Agreement of musical with each instrument:")
    log(f"  {'Instrument':<14} {'Agreement':>10}")
    log("  " + "-" * 26)
    for j, name in enumerate(INSTRUMENT_NAMES):
        marker = " <<<" if j == MUSICAL_IDX else ""
        log(f"    {name:<12} {agree_matrix[MUSICAL_IDX, j]:>9.3f}{marker}")

    # Mean agreement for each instrument (excluding self)
    log(f"\n  Mean pairwise agreement (excluding self):")
    for i, name in enumerate(INSTRUMENT_NAMES):
        others = [agree_matrix[i, j] for j in range(7) if j != i]
        marker = " <<<" if i == MUSICAL_IDX else ""
        log(f"    {name:<12} {np.mean(others):>9.3f}{marker}")

    # ──────────────────────────────────────────────────────────────
    # SUMMARY & PREDICTIONS
    # ──────────────────────────────────────────────────────────────
    log_section("Summary & Prediction Scorecard")

    # Compute summary stats
    musical_acc_lost_pct = musical_acc_lost
    musical_abl_acc = ablation_results["musical"]["accuracy"]
    musical_abl_net = ablation_results["musical"]["net"]

    # P1: Musical accuracy < 35% on LOSTs
    p1 = musical_acc_lost_pct < 35.0
    log(f"\n  P1: Musical accuracy < 35% on LOSTs")
    log(f"      Musical LOST accuracy = {musical_acc_lost_pct:.1f}% → {'CONFIRMED' if p1 else 'REFUTED'}")

    # P2: Musical's errors cluster on barrel↔enzyme confusion
    barrel_enzyme_confusions = 0
    total_musical_errors = 0
    for d in corpus_data:
        truth = d["truth"]
        mtop = max(d["musical_votes"], key=d["musical_votes"].get)
        if mtop != truth:
            total_musical_errors += 1
            if (truth in ("barrel", "enzyme_active") and
                    mtop in ("barrel", "enzyme_active")):
                barrel_enzyme_confusions += 1
    p2_frac = barrel_enzyme_confusions / total_musical_errors if total_musical_errors > 0 else 0
    p2 = p2_frac > 0.20  # "cluster" = at least 20% of errors are on this axis
    log(f"\n  P2: Musical errors cluster on barrel↔enzyme")
    log(f"      barrel↔enzyme confusions: {barrel_enzyme_confusions}/{total_musical_errors} "
        f"({100*p2_frac:.1f}%) → {'CONFIRMED' if p2 else 'REFUTED'}")

    # P3: Musical agrees with fick on < 25% of LOSTs
    fick_idx = list(INSTRUMENT_NAMES).index("fick")
    musical_fick_agree_lost = 0
    for d in corpus_data:
        if not d["correct"]:
            votes = d["carver_votes"]
            m_top = max(votes[MUSICAL_IDX], key=votes[MUSICAL_IDX].get)
            f_top = max(votes[fick_idx], key=votes[fick_idx].get)
            if m_top == f_top:
                musical_fick_agree_lost += 1
    musical_fick_pct = 100 * musical_fick_agree_lost / n_lost if n_lost > 0 else 0
    p3 = musical_fick_pct < 25.0
    log(f"\n  P3: Musical agrees with fick < 25% on LOSTs")
    log(f"      Agreement on LOSTs = {musical_fick_agree_lost}/{n_lost} "
        f"({musical_fick_pct:.1f}%) → {'CONFIRMED' if p3 else 'REFUTED'}")

    # P4: Removing musical improves accuracy ≥ 31/52
    p4 = musical_abl_acc >= n_correct
    log(f"\n  P4: Removing musical: accuracy ≥ {n_correct}/52")
    log(f"      Ablated accuracy = {musical_abl_acc}/52 (net {musical_abl_net:+d}) "
        f"→ {'CONFIRMED' if p4 else 'REFUTED'}")

    # P5: Musical correct when all 3 Fano-line partners agree
    # For each of musical's 3 lines: when BOTH partners agree with each other,
    # is musical's vote correct?
    p5_total = 0
    p5_musical_correct = 0
    for line_idx, line in MUSICAL_LINES:
        partners = [i for i in line if i != MUSICAL_IDX]
        for d in corpus_data:
            votes = d["carver_votes"]
            truth = d["truth"]
            p_tops = [max(votes[p], key=votes[p].get) for p in partners]
            m_top = max(votes[MUSICAL_IDX], key=votes[MUSICAL_IDX].get)
            # Both partners agree with each other
            if p_tops[0] == p_tops[1]:
                p5_total += 1
                if m_top == truth:
                    p5_musical_correct += 1

    p5_pct = 100 * p5_musical_correct / p5_total if p5_total > 0 else 0
    # "correct" = higher accuracy than baseline
    p5 = p5_pct > musical_acc_all
    log(f"\n  P5: Musical correct when Fano-line partners agree")
    log(f"      {p5_musical_correct}/{p5_total} ({p5_pct:.1f}%) vs baseline {musical_acc_all:.1f}% "
        f"→ {'CONFIRMED' if p5 else 'REFUTED'}")

    n_confirmed = sum([p1, p2, p3, p4, p5])
    log(f"\n  Score: {n_confirmed}/5 predictions confirmed")

    # ──────────────────────────────────────────────────────────────
    # VERDICT
    # ──────────────────────────────────────────────────────────────
    log_section("Verdict: Is Musical INFORMATIVE or NOISE?")

    # Summarise findings
    log(f"\n  Musical overall accuracy:   {musical_acc_all:.1f}%")
    log(f"  Musical LOST accuracy:      {musical_acc_lost_pct:.1f}%")
    log(f"  Ablation accuracy:          {musical_abl_acc}/52 (net {musical_abl_net:+d})")
    log(f"  Unique correct (musical alone right): {len(musical_unique)}")
    log(f"  Sole dissenter (musical alone wrong):  {len(sole_dissenter)}")
    log(f"  Mean pairwise agreement:    {np.mean([agree_matrix[MUSICAL_IDX, j] for j in range(7) if j != MUSICAL_IDX]):.3f}")

    if musical_abl_net < 0:
        log(f"\n  VERDICT: Musical is NET INFORMATIVE — removing it costs "
            f"{abs(musical_abl_net)} classification(s).")
        log(f"  Despite low agreement, musical provides unique signal "
            f"(particularly for {len(musical_unique)} uniquely-correct proteins).")
    elif musical_abl_net > 0:
        log(f"\n  VERDICT: Musical is NET NOISE — removing it gains "
            f"{musical_abl_net} classification(s).")
        log(f"  Musical's disagreements are more harmful than informative.")
    else:
        log(f"\n  VERDICT: Musical is NET NEUTRAL — removing it changes "
            f"classifications but net accuracy is unchanged.")
        log(f"  Musical's unique correct cases ({len(musical_unique)}) are balanced "
            f"by its harmful disagreements.")

    # ──────────────────────────────────────────────────────────────
    # SAVE RESULTS
    # ──────────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = {
        "experiment": "D166_musical_diagnostic",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "corpus_size": n,
        "production_accuracy": n_correct,
        "n_lost": n_lost,

        "phase1_per_instrument_accuracy": {
            name: round(inst_acc_pcts[name], 1) for name in INSTRUMENT_NAMES
        },
        "musical_accuracy_all": round(musical_acc_all, 1),
        "mean_instrument_accuracy_all": round(mean_acc_all, 1),

        "phase2_musical_accuracy_lost": round(musical_acc_lost_pct, 1),

        "phase3_musical_per_archetype": {
            arch: {
                "correct": arch_stats[arch]["correct"],
                "total": arch_stats[arch]["total"],
                "accuracy": round(100 * arch_stats[arch]["correct"] / arch_stats[arch]["total"], 1)
                if arch_stats[arch]["total"] > 0 else 0,
                "confusion": dict(arch_stats[arch]["predicted_as"]),
            }
            for arch in ALL_ARCHS
        },

        "phase5_ablation": {
            name: {
                "accuracy": ablation_results[name]["accuracy"],
                "net": ablation_results[name]["net"],
                "gains": ablation_results[name]["gains"],
                "losses": ablation_results[name]["losses"],
                "changes": ablation_results[name]["changes"],
            }
            for name in INSTRUMENT_NAMES
        },

        "phase7_musical_uniquely_correct": musical_unique,
        "phase7_sole_dissenter": sole_dissenter,

        "phase9_musical_agreement_row": {
            INSTRUMENT_NAMES[j]: round(float(agree_matrix[MUSICAL_IDX, j]), 3)
            for j in range(7)
        },

        "predictions": {
            "P1_musical_acc_lt_35_lost": {
                "value": round(musical_acc_lost_pct, 1),
                "confirmed": p1,
            },
            "P2_barrel_enzyme_cluster": {
                "value": round(100 * p2_frac, 1),
                "confirmed": p2,
            },
            "P3_fick_agree_lt_25_lost": {
                "value": round(musical_fick_pct, 1),
                "confirmed": p3,
            },
            "P4_ablation_ge_31": {
                "value": musical_abl_acc,
                "net": musical_abl_net,
                "confirmed": p4,
            },
            "P5_correct_when_partners_agree": {
                "value": round(p5_pct, 1),
                "baseline": round(musical_acc_all, 1),
                "confirmed": p5,
            },
        },
        "n_confirmed": n_confirmed,

        "verdict": "informative" if musical_abl_net < 0 else
                   "noise" if musical_abl_net > 0 else "neutral",
    }

    out_path = RESULTS_DIR / "d166_musical_diagnostic.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"\n  Results saved to {out_path}")

    elapsed = time.time() - t0
    log(f"\n  Total time: {elapsed:.1f}s")

    return results


if __name__ == "__main__":
    main()
