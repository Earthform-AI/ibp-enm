# Discovery Sprint Log

## Current State — D169 (Sprint 10: Confusion-Pair Resolution)

**Accuracy: 32/52 (61.5%)** on the expanded corpus, 0 free parameters.

### Production changes
- `AlgebraicFickBalancer` integrated into `LensStackSynthesizer` (was `MetaFickBalancer`)
- Route-score gated context_boost in bridge pathway (D160)
- Formula: `bridge = route_score[arch] × context_boost + α₈ × fano_bridge`
- **D168: √α₈ gate** — `bridge_weight = 0.5 × (1−α₀) × √α₈` (was linear α₈).
  Compressive gate lifts bridge-blind proteins (0.14→0.37) without over-boosting
  high-α₈ proteins (1.0→1.0). Gains ATCase_cat with full structural data pipeline,
  0 regressions. Invisible in cache-only _rescore (no structural data for lenses).
- **ProfileCache.repair()**: fills missing `per_instrument` metadata from stored profiles
- **ProfileCache.is_complete()**: detects stale/incomplete cache entries
- **BenchmarkRunner._run_protein**: now saves actual ThermoReactionProfile objects (was `[]`)
- **BenchmarkRunner._rescore**: auto-rebuilds stale entries on access
- 54/54 tests pass (cache: 25, benchmark: 29)

### D169: Rank-2 Correction Lens — Confusion-Pair Discriminants (NEGATIVE RESULT)

**Date**: 2026-02
**Script**: `experiments/discovery_169_rank2_correction.py`
**Results**: `experiments/results/d169_rank2_correction.json`

**Question**: Can D167's per-axis discriminant features be used as a post-hoc
lens to flip near-miss rank-2 predictions correct?

**Method**: Built a `ConfusionPairLens` implementing the Lens protocol.
Gate: top-2 margin < threshold. Apply: extract D167 discriminant feature
from all 7 instrument profiles, compare against population-level medians
(corpus central tendency), boost rank-2 / demote rank-1 if majority of
instruments favour rank-2. 5 variants tested on 32 expanded-corpus proteins.

| Variant | Strategy | Accuracy | Δ |
|---------|----------|----------|---|
| A (baseline) | Production pipeline | 16/32 | — |
| B (CPL only) | ConfusionPairLens, no default lenses | 15/32 | -1 |
| C (full+CPL) | Default stack + CPL appended | 14/32 | -2 |
| D (aggressive) | Wider margin (0.15), larger boost | 14/32 | -2 |
| E (conservative) | Higher quorum (5/7), smaller boost | 16/32 | 0 |

**Key findings — NEGATIVE RESULT, no production change**:

1. **Most near-miss proteins have truth at rank-3+, not rank-2.**
   The lens can only flip rank-1↔rank-2, making it structurally unable
   to help Neuroglobin (truth rank-3), Erythrocruorin (rank-3),
   Transferrin (rank-5).

2. **The enzyme lens is the primary error source, not confusion margins.**
   KDPG_aldolase is correct PRE-lens (barrel rank-1) but the enzyme lens
   pushes enzyme_active above barrel. GroEL_subunit is rank-2 pre-lens
   but enzyme lens pushes it to rank-4+. Future work should target
   enzyme lens false-positive rate.

3. **Discriminant features create false positives on correct proteins.**
   Leghemoglobin (correct) and Phosphoglycerate_k (correct) have small
   margins and D167 features that point the WRONG way, causing regressions.

4. **Only Truncated_Hb is recoverable** via aggressive variant D (+1),
   but at cost of -3 regressions (net -2).

**Predictions**: 2/5 confirmed (P4✓ Transferrin not recovered, P5✓ E has
0 regressions). P1✗ P2✗ P3✗ — lens doesn't work as designed.

**Key redirect**: Next experiment should investigate **enzyme lens
over-activation** — the enzyme lens's false-positive rate on barrel and
allosteric proteins is the dominant error mode for near-miss proteins.

### D168: Conditional Bridge — Surgical α₈ De-gating

**Date**: 2026-02
**Script**: `experiments/discovery_168_conditional_bridge.py`
**Results**: `experiments/results/d168_conditional_bridge.json`

**Question**: Can we recover any of the 4 bridge-blind proteins (α₈ ≤ 1/7)
by surgically de-gating α₈ from `bridge_weight` without the 5 regressions
that D160 Variant C caused?

**Method**: Test 6 variants on 32 expanded-corpus proteins (cached profiles +
fresh structural data for lens stack):

| Variant | Strategy | Formula |
|---------|----------|---------|
| A (baseline) | Production (route-gated) | bw = 0.5 × (1−α₀) × α₈ |
| B (floor) | Clamp α₈ ≥ 1/7 in bw only | bw = 0.5 × (1−α₀) × max(α₈, 1/7) |
| C (soft floor) | Clamp α₈ ≥ 1/7 everywhere | Uses max(α₈, 1/7) in bw + bridge |
| D (margin-cond) | Floor only when α₀ < 0.05 | Selective floor for confused proteins |
| E (√ gate) | √α₈ in bridge_weight | bw = 0.5 × (1−α₀) × √α₈ |
| F (√+floor) | √max(α₈, 1/7) everywhere | Combined compressive + floor |

**Results:**

| Variant | Correct/32 | Δ | Gains | Regressions |
|---------|-----------|---|-------|-------------|
| A baseline | 15 | — | — | — |
| B floor | 15 | 0 | — | — |
| C soft floor | 15 | 0 | — | — |
| D margin-cond | 15 | 0 | — | — |
| **E √ gate** | **16** | **+1** | ATCase_cat | **0** |
| **F √+floor** | **16** | **+1** | ATCase_cat | **0** |

**Bridge-blind proteins (all 4 loaded, 0/4 recovered by any variant):**
- **Neuroglobin** (α₈=0.143): truth=globin at rank 3, barrel dominates (+0.047 gap)
- **Cytochrome_b5** (α₈=0.000): truth=globin at rank 3, enzyme_active dominates (+0.226 gap)
- **Erythrocruorin** (α₈=0.143): context_boost[globin]=0.0 — bridge pathway CAN'T help
- **GroEL_subunit** (α₈=0.143): truth=allosteric at rank 5, enzyme_active dominates (+0.365 gap)

**Key findings:**
1. **Floor gates (B,C,D) are inert**: 3/4 bridge-blind proteins have α₈=1/7=floor;
   Cytochrome_b5 has α₈=0 where floor helps minimally (bw 0→0.063). The bridge
   content for these proteins is misaligned — context_boost doesn't favour truth.
2. **√ gate (E) gains ATCase_cat** (α₈=0.429, α₀=0.049): compressive gate lifts
   bridge_weight from 0.204→0.312, letting allosteric context_boost overcome
   enzyme_active. This is NOT a bridge-blind protein — it's a confused protein
   where more bridge weight helps.
3. **0 regressions for E/F**: √ is naturally compressive at high α₈ (0.86→0.93,
   1.0→1.0), so the change is conservative for well-classified proteins.
4. **Bridge-blind are rule-level problems**: instruments genuinely vote wrong for
   these proteins. No amount of bridge reweighting can fix incorrect context_boost
   values or consensually wrong instrument votes.
5. **Pipeline discrepancy**: _rescore (evals=None) gives 32/52 including ATCase_cat
   correct. D168 (full structural data) shows ATCase_cat LOST in baseline — the
   lens stack with real eigenvalues hurts ATCase_cat, but √ gate overcomes this.

**Predictions**: 3/5 confirmed (P1 ✗ 0/4 bridge-blind recovered, P2 ✗ both have
0 regressions, P3 ✓ margin-cond has 0 regressions, P4 ✓ best=16/52 < 33,
P5 ✓ Cytochrome_b5 not recovered by floor variants)

### D167: Confusion-Pair Discriminants

**Date**: 2026-02
**Script**: `experiments/discovery_167_confusion_discriminants.py`
**Results**: `experiments/results/d167_confusion_discriminants.json`

**Question**: For each confusion axis, which per-instrument features best
discriminate the confused archetypes?  Are there novel features (not in
current rules) that could power new lens rules?

**Method**: Build 52×187 feature matrix (17 features × 7 instruments +
4 cross-instrument aggregates × 17).  Identify 6 confusion axes with ≥2
crossings.  Compute AUC via Mann–Whitney U for every feature on each axis.
Near-miss analysis checks if top discriminants favour the correct class.

**Top discriminant per axis:**

| Axis | Best Feature | AUC | Cohen's d |
|------|-------------|-----|-----------|
| allosteric↔barrel | fick_mean_delta_beta | **1.000** | 0.84 |
| allosteric↔globin | propagative_mean_scatter | **0.983** | 2.58 |
| enzyme_active↔globin | thermal_gap_trend | **0.969** | 2.18 |
| barrel↔dumbbell | algebraic_mean_delta_beta | **0.960** | 0.65 |
| barrel↔enzyme_active | agg_min_entropy_change | **0.946** | 1.37 |
| allosteric↔dumbbell | fragile_mean_delta_entropy | **0.800** | 1.05 |

**Cross-axis champions (base features in top-5 across most axes):**
1. `gap_retained` — 4 axes — **NOT in current rules**
2. `mean_delta_beta` — 3 axes — in rules
3. `gap_flatness` — 3 axes — in rules
4. `mean_scatter` — 2 axes — in rules

**Near-miss analysis**: 6/7 near-miss proteins rescuable (top discriminant
features favour correct class on 2+/3 features).  Only Transferrin is stuck.

**Novel features discovered**: `gap_retained`, `mean_delta_entropy`,
`free_energy_cost`, `entropy_change`, `heat_cap_change` — all absent from
current archetype rules.  `gap_retained` is the strongest lens candidate.

**Predictions**: 4/5 confirmed (P1 ✓ AUC>0.85, P2 ✗ agg not #1 on ≥2 axes,
P3 ✓ 6 rescuable, P4 ✓ gap_flatness on 3 axes, P5 ✓ 11 novel features)

### D163: 45° Plane Bridge Signals

**Date**: 2025-06
**Script**: `experiments/discovery_163_fortyfive_plane_bridge.py`
**Results**: `experiments/results/d163_fortyfive_plane_bridge.json`

**Question**: Do D156's 210 unique 45° planes (from 2016 half-diagonal
kernel–subalgebra pairs at {π/4,π/4,π/2,π/2}) provide supplementary
correction signals for the 21 LOST proteins where contained channels alone
are insufficient?

**Method**: Enumerate 210 unique 45° planes from sedenion geometry.  Each
plane loads 0.5 on exactly 4 basis elements (2 low-half e₁-e₇, 2 high-half
e₈-e₁₅).  Map low-half indices to instruments via OCTO_TO_CARVING.  Compute
per-archetype "45° plane score" and compare against contained route_score on
all 52 proteins.

**Geometric results:**

| Property | Contained Channels | 45° Planes |
|----------|--------------------|------------|
| Total unique objects | 168 edges | 210 planes |
| Per kernel | 4 subs | 24 subs |
| Per Fano line | 72 routes | 30 planes |
| Instrument pairs | 21 (via lines) | **ALL 21 = C(7,2)** |
| Planes per pair | — | 10 (perfectly uniform) |
| Signal strength | 1/√2 ≈ 0.707 | cos(π/4) = 1/√2 (**same**) |
| Subalgebra coverage | 21 cross-half only | All 35 (incl. pure-low/high) |

**Key structural finding**: 45° planes are PERFECTLY UNIFORM — every Fano
line has exactly 30 planes, every instrument pair has exactly 10 planes.
Each plane maps to exactly 1 Fano line.  This is a second-tier routing
structure covering all C(7,2)=21 pairwise instrument interactions.

**Signal comparison (CORRECT vs LOST):**

| Signal | CORRECT mean | LOST mean | Separation |
|--------|-------------|-----------|------------|
| Contained route_score | 0.727 | 0.276 | **0.451** |
| 45° plane score | 0.614 | 0.198 | 0.417 |

| Metric | Contained | 45° Plane |
|--------|-----------|-----------|
| Truth rank (CORRECT) | 1.74 | 1.71 |
| Truth rank (LOST) | 3.24 | **3.14** |

**Correction potential for 21 LOSTs:**
- 45° favours truth (vs contained): **11/21** (52%)
- 45° favours pred (vs contained): 6/21 (29%)
- Neutral: 4/21 (19%)
- **Net correction tendency: +5**

**Conclusion**: SPRINT prediction CONFIRMED — 45° planes provide a
*weaker but broader* correction signal.  They are weaker (separation
0.417 vs 0.451) but have a net favourable correction tendency (11 vs 6
LOSTs).  The perfect uniformity (30 planes/line, 10 planes/pair)
suggests a structured second-tier channel that could complement
contained routing if the bridge pathway weight were increased.
Currently, the bridge_weight damping that makes Hamming=Sedenon
(D162) also suppresses 45° plane corrections.

### D162: Bridge Benchmark (HammingBridge vs SedenonBridge)

**Date**: 2025-06
**Script**: `experiments/discovery_162_bridge_benchmark.py`
**Results**: `experiments/results/d162_bridge_benchmark.json`

**Question**: Does the D158 SedenonBridge (rank-based dual-threshold, 40.8%
valid syndrome rate) improve accuracy over the D153 HammingBridge
(mean-threshold Hamming(7,4), 13.1% valid rate)?

**Method**: Run both bridges through the identical production pipeline
(AlgebraicFickBalancer + D160 route-gating + lens stack) on all 52 proteins.

**Key results:**

| Bridge | Accuracy | Valid Rate |
|--------|----------|-----------|
| HammingBridge (D153) | 31/52 (59.6%) | 13.1% |
| SedenonBridge (D158) | 31/52 (59.6%) | 40.8% |
| Delta | **0** | +27.7pp |

- **Identical classifications on all 52 proteins** — 0 gains, 0 losses, 0 diffs
- **Bridge score divergence exists** (max 0.087 on Chymotrypsin) but
  `bridge_weight = 0.5 × (1-α₀) × α₈` damps the pathway enough that
  the ~0.05–0.09 fano_bridge differences can't flip any classification
- **Implication**: Accuracy comes from route-score gating and spectral
  weights, not from the specific syndrome decoding method. SedenonBridge
  is geometrically better-founded (matches contained-channel theory) but
  functionally equivalent at current bridge_weight levels

**SPRINT P5 CONFIRMED**: SedenonBridge ≥ 30/52 → 31/52 ✓

### D161: Pivot Instrument Validation — Cooperative (BECOMING)

**Date**: 2025-06
**Script**: `experiments/discovery_161_pivot_instrument_validation.py`
**Results**: `experiments/results/d161_pivot_validation.json`

**Question**: Does cooperative (instrument 4 / BECOMING / e₅) actually pivot
classification on proteins where its Fano lines carry conflicting signals?

**Method**: Ablation test — classify all 52 proteins with and without
cooperative's vote (zeroed). Analyse Fano line conflict patterns.

**Key results:**

| Metric | Value |
|--------|-------|
| A (production) | 31/52 (59.6%) |
| B (coop ablated) | 29/52 (55.8%) |
| Delta (A−B) | **+2** |
| Pivots (classification changes) | 2 |
| Pivot saves | 2 (Aldolase_A, Rubisco_large) |
| Pivot hurts | **0** |
| Cooperative agrees with truth | 21/52 (40.4%) |
| Line conflicts | 20/52 (38.5%) |

**Surprising findings:**
1. **Conflict group has NO pivots** — all 20 proteins with conflicting Fano
   lines classify identically with/without cooperative. Accuracy 75.0% both ways.
2. **No-conflict group has ALL pivots** — cooperative's value comes from barrel
   proteins specifically (Aldolase_A, Rubisco_large), both non-conflicting.
3. **Low agreement rate (40.4%)** — cooperative is "wrong" 60% of the time, yet
   still helps. Route-score gating (D160) makes the architecture robust to
   cooperative's frequent disagreements by damping archetypes with weak support.
4. **Route score is the real protection** — cooperative doesn't need to be right;
   it needs to shift the Fano support pattern so route-score gating can damp
   incorrect archetypes.

**Conclusion**: Cooperative is validated as a net positive pivot (+2, 0 hurts).
Its value is indirect: adding cooperative to the support vector changes
route_scores enough to damp wrong-archetype context_boost. No production changes
needed — the current architecture correctly leverages cooperative.

### D160: Route-Score Gated Context-Boost Corrections

**Date**: 2025-02 (continued 2025-06)
**Script**: `experiments/discovery_160_route_gated_corrections.py`
**Results**: `experiments/results/d160_route_gated.json`

**Key finding**: Context_boost (structural corrections from D109–D113) was being
applied equally to all archetypes regardless of instrument-level support. By
gating context_boost with the per-archetype Fano route_score, structurally
unmotivated boosts are damped. 0 new free parameters.

**Variants tested (all 0 new free params):**

| Variant | Formula change | Accuracy | Δ |
|---------|---------------|----------|---|
| A: Baseline | Current AlgebraicFickBalancer | 30/52 (57.7%) | — |
| **B: Route-gated** | `bridge = rs × ctx_boost + α₈ × fano` | **31/52 (59.6%)** | **+1** |
| C: De-doubled α₈ | `bw = 0.5 × (1-α₀)` (remove α₈ from bw) | 26/52 (50.0%) | -4 |
| D: Combined (B+C) | Both changes | 31/52 (59.6%) | +1 |

**Variant B applied to production.** Gains: Aldolase_A, Rubisco_large (+2).
Loss: ATCase_cat (-1). Net: +1.

**Why Variant C fails**: Removing α₈ from bridge_weight gives too much bridge
weight to proteins that previously had low α₈, overwhelming hinge_lens
corrections for T4_lysozyme, Thermolysin, Papain (5 regressions).

**Why Variant B wins**: Route-score gating focuses context_boost on archetypes
that have Fano-structure support from instrument votes. For Rubisco_large
(margin -0.005), allosteric context_boost (+0.3) was damped by route_score
0.303 → barrel wins. For Aldolase_A, enzyme context_boost damped similarly.

### D159: Error Analysis (preceding D160)

**Accuracy**: 30/52 (57.7%), 0 free parameters
**Predictions**: 3/5 confirmed

Identified 22 LOSTs organized into 3 failure categories:
- **3 high-confidence wrong** (Chymotrypsin, Subtilisin, Glycogen_phosph):
  6-7/7 instruments vote barrel, margins -0.62 to -0.69. Rule-level problem —
  low scatter + low Δβ trigger barrel rules for enzyme_active/allosteric proteins.
- **2 easy flips** (Rubisco_large margin -0.005, Protein_kinase_A margin -0.026):
  Context_boost flipping correct instrument votes.
- **4 bridge-blind** (Neuroglobin α₈=0.14, Cytochrome_b5 α₈=0.00,
  Erythrocruorin α₈=0.14, GroEL_subunit α₈=0.14): No bridge pathway
  because α₈ doubles as both bridge weight and fano_bridge gate.

### Key insight: α₈ double-gating

```
bridge_weight = 0.5 × (1-α₀) × α₈     ← α₈ here
bridge = context_boost + α₈ × fano_bridge  ← AND here
```

So fano_bridge is effectively scaled by α₈². Context_boost is completely
zeroed when α₈=0, even though it's independent structural signal.
Variant C tried to fix this but caused 5 regressions. Variant B addresses
the easy flips without touching bridge_weight.

### Remaining LOSTs (21 after D160)

**Still wrong in best variant (B_route_gated):**
- enzyme_active: Chymotrypsin, Subtilisin (→barrel), Carbonic_anhyd_II (→globin)
- barrel: Xylose_isomerase (→allosteric), KDPG_aldolase (→enzyme_active)
- globin: Neuroglobin (→barrel), Cytochrome_b5 (→enzyme_active),
  Cytochrome_c (→dumbbell), Erythrocruorin (→allosteric), Truncated_Hb (→enzyme_active)
- dumbbell: Transferrin (→enzyme_active), Pyruvate_kinase (→barrel),
  MBP (→allosteric), HSP70_NBD (→barrel)
- allosteric: ATCase_cat (→enzyme_active, **new regression**),
  Glycogen_phosph (→barrel), Protein_kinase_A (→globin),
  ABP_open (→enzyme_active), CheY (→enzyme_active), GroEL_subunit (→enzyme_active)

### Next targets
1. **ATCase_cat regression** — investigate why route-gating moved it from
   allosteric to enzyme_active (new D160 regression)
2. **3 high-confidence wrong** — need rule-level fix (new instrument features
   or rule threshold adjustments)
3. **4 bridge-blind** — Variant C shows Cytochrome_b5 recoverable via
   de-doubled α₈, but need to protect hinge-dependent proteins

---

## Discovery Trail (D148–D164)

| ID | Title | Accuracy | Free Params | Key Result |
|----|-------|----------|-------------|------------|
| D148 | Sedenion spectral structure | — | — | √2:1 weight ratio from SVD |
| D149 | Algebraic Fick experiment | — | 0 | AlgebraicFickBalancer design |
| D152 | Implementation + α₀ fix | 30/52 | 0 | Vote-margin α₀ replaces entropy |
| D153 | Hamming bridge (syndrome) | 30/52 | 0 | Hamming(7,4) syndrome correction |
| D154 | Predictions | — | — | 0/5 confirmed |
| D155 | Predictions | — | — | 5/5 confirmed |
| D156 | Predictions | — | — | 2/4 confirmed |
| D157 | ZD pair routing | — | — | 168 contained edges through 21 subalgebras |
| D158 | SedenonBridge | 30/52 | 0 | Rank-based dual-threshold (40% valid rate) |
| D159 | Error analysis | 30/52 | 0 | 22 LOSTs in 3 failure categories |
| **D160** | **Route-gated context boost** | **31/52** | **0** | **+1 net gain, Variant B** |
| **D161** | **Pivot instrument validation** | **31/52** | **0** | **Cooperative is net +2 pivot, 0 hurts** |
| **D162** | **Bridge benchmark** | **31/52** | **0** | **Hamming=Sedenon (identical classifications), P5 confirmed** |
| **D163** | **45° plane bridge signals** | **31/52** | **0** | **210 planes, perfect uniformity (30/line, 10/pair), weaker but broader correction (+5 net)** |
| **D164** | **CD tower n=5 (64-nions)** | **—** | **—** | **53 components = 31 new + 22 inherited. C(n) = 2ⁿ−n−5 closed form. Cumulative recurrence confirmed** |
| **D165** | **Transitivity decorrelation** | **31/52** | **0** | **Sharing uniform in PG(2,2). LOO correction 42% reduction (0.077→0.045). Root cause: musical instrument low agreement on lines 1,2** |
| **D166** | **Musical instrument diagnostic** | **31/52** | **0** | **Musical worst instrument (23.1% vs mean 38.2%). 0% globin accuracy (all→dumbbell). NET NEUTRAL on ablation. 5 uniquely-correct allosteric calls. Rule-level fix needed: globin rule + scatter thresholds** |
| **D166b** | **Musical rule repair** | **31/52** | **0** | **7 variants tested. Pipeline robust (all 31/52). F_combined fixes musical globin 0→4/10 (40%). G_reduce_dumbbell gives +5 musical votes. No single change overcomes dumbbell dominance — combination needed** |
