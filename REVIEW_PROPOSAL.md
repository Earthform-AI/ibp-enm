# IBP-ENM Review & Cleanup Proposal

**Date:** 2026-04-14  
**Status:** Draft for review  
**Goal:** Prepare IBP-ENM for public-facing credibility (grants, preprints, service packaging)

---

## 1. The Name Problem

### Current State

"IBP" is **never defined anywhere** in the codebase — not in the README, docstrings, `__init__.py`, commit messages, or any documentation. The only contextual clue is in `archetypes.py` which references "the meta-IBP protocol" in `SurgeonsHandbook`. The full package title is "IBP-ENM: Spectral Elastic Network Model for Protein Structural Analysis."

### What does the codebase actually do?

The core loop is: build elastic network → compute spectral decomposition → partition via Fiedler vector → probe via 7 thermodynamic instruments → classify via lens fusion. The distinctive contribution is the **instrument-based probing** paradigm — not iterating, not Bayesian, not inverse. The library is fundamentally a *spectral instrument suite for protein structure*.

### Naming Options

| Option | Name | Package | Rationale |
|--------|------|---------|-----------|
| A | **Spectral Protein Instruments** (`spi-enm`) | `spi-enm` | Describes exactly what it does. "Instruments" is the core abstraction. |
| B | **Thermodynamic Band ENM** (`tband`) | `tband-enm` | The ThermodynamicBand is the distinctive contribution. Short, memorable. |
| C | **Fano Spectral ENM** (`fano-enm`) | `fano-enm` | Highlights the Fano plane geometry underlying the 7 instruments. |
| D | **Keep IBP-ENM** | `ibp-enm` | Already on PyPI, has CI, avoid breakage. Backronym it. |

### Recommendation

**Option D (keep + backronym) for now.** Renaming a published PyPI package mid-grant-application is risky — broken imports, stale references, SEO loss. Instead:

1. Define IBP as **"Instrument-Based Probing"** — this accurately describes the methodology and retroactively makes the name meaningful.
2. Add a one-line expansion in the README: *"IBP-ENM (**Instrument-Based Probing** Elastic Network Model)"*
3. Add the same expansion to the `__init__.py` module docstring.
4. Revisit renaming only if/when you do a v1.0 release with a publication.

The backronym is **honest** — the library really does probe proteins with instruments. It just wasn't named deliberately.

---

## 2. Architecture Map — All Moving Parts

### Source Modules (24 files, 17,141 lines)

#### Tier 1: Core Analysis (a user must understand these)

| Module | Lines | Role |
|--------|------:|------|
| `analyzer.py` | 2,756 | **Core engine.** 4 entry points: `analyze()`, `compare()`, `probe()`, `listen()`. Builds ENM, computes eigensystem, Fiedler partitioning, hinge detection, B-factor prediction from 3 shadow perspectives. |
| `band.py` | 585 | **Orchestrator.** Creates 7 instrument carvers, runs them, feeds results to synthesis. `run_single_protein()` lives here — the main user entry point. |
| `instruments.py` | 624 | **7 thermodynamic probes.** Each carves a protein and measures a different physical signal. This is the conceptual heart of the method. |
| `synthesis.py` | 1,085 | **Vote fusion.** `MetaFickBalancer` (5 params), `AlgebraicFickBalancer` (0 params). Combines 7 instrument votes into archetype classification. |
| `archetypes.py` | 413 | **Ground truth.** 5 archetype definitions, 12-protein benchmark corpus, `GROUND_TRUTH` mapping. `SurgeonsHandbook` for quick diagnosis. |
| `fetch.py` | 248 | **Data access.** Downloads PDB structures from RCSB. |

#### Tier 2: Classification Refinement (for power users / researchers)

| Module | Lines | Role |
|--------|------:|------|
| `lens_stack.py` | 1,750 | **Post-hoc scoring lenses.** EnzymeLens, HingeLens, BarrelPenaltyLens, AllostericLens, FlowGrammarLens. Composable stack that refines classification. |
| `rules.py` | 655 | **~90 archetype rules.** Decomposed, testable, sweepable. Each rule is a boolean condition on instrument outputs. |
| `thresholds.py` | 441 | **~90 named thresholds.** Immutable registry consumed by rules. Enables programmatic threshold sweeping. |
| `carving.py` | 748 | **Carving primitives.** `CarvingIntent`, `FickBalancer`, `ShadowInspector`. Low-level edge-removal logic. |
| `thermodynamics.py` | 507 | **Physical observables.** Entropy, heat capacity, free energy, IPR, per-residue decomposition. |

#### Tier 3: Analysis & Validation (for research / experimentation)

| Module | Lines | Role |
|--------|------:|------|
| `benchmark.py` | 1,887 | **Benchmark harness.** 3 corpora (12/52/200+ proteins), `BenchmarkRunner`, `CrossValidator`. |
| `analysis.py` | 715 | **Result analysis.** Archetype profiles, confusion clusters, co-firing matrices, cross-experiment comparison. |
| `algebra.py` | 764 | **Algebraic fingerprinting.** Firing lattice, collinearity, threshold sensitivity (Fano plane connections). |
| `belief_algebra.py` | 896 | **Error-corrected fusion.** Hamming(7,4) bridge, sedenion bridge, zero-divisor pair selection. |
| `trace.py` | 323 | **Classification audit trail.** Per-protein trace of every step. |
| `cache.py` | 325 | **Profile caching.** Skip re-carving for fast re-scoring experiments. |

#### Tier 4: Extensions

| Module | Lines | Role |
|--------|------:|------|
| `graph_data.py` | 895 | **PyG pipeline.** Converts proteins → PyTorch Geometric `Data` objects. Requires `[gnca]` extra. |
| `gnca.py` | 296 | **Graph NCA model.** 6,576-parameter classifier. Requires `[gnca]` extra. |
| `gnca_trainer.py` | 546 | **Training loop.** Early stopping, k-fold CV. Requires `[gnca]` extra. |
| `corpus_builder.py` | 1,023 | **Corpus expansion.** Automated RCSB/CATH queries to build protein corpora. |
| `functional_sites.py` | 651 | **UniProt/PDBe integration.** Functional site resolution for validation. |
| `baselines.py` | 60 | **GNM baseline.** Simple B-factor prediction for comparison. |

### The 7 Instruments

| # | Name | Signal | Physical Meaning |
|---|------|--------|-----------------|
| 1 | algebraic | max \|Δgap\| | Which edge removal most disrupts spectral gap → symmetry breaking |
| 2 | musical | max mode_scatter | Which removal scatters mode frequencies → resonance sensitivity |
| 3 | fick | FickBalancer score | Diffusion-optimal cut location |
| 4 | thermal | max \|Δτ\| | Which removal most disrupts entropy → thermal fingerprint |
| 5 | cooperative | max \|Δβ\| | Which removal breaks cooperative coupling |
| 6 | propagative | max spatial_radius | Which removal disconnects allosteric reach |
| 7 | fragile | max bus_mass | Thermally soft spots → structural fragility |

### Classification Pipeline Flow

```
PDB ID + Chain
    ↓
fetch_pdb_ca_data()              → Cα coords + B-factors
    ↓
IBPProteinAnalyzer.analyze()     → IBPResult (domains, hinges, B-factors)
    ↓
ThermodynamicBand.classify()     → 7 instrument reaction profiles
    ↓
AlgebraicFickBalancer.classify() → raw archetype vote
    ↓
LensStack.apply()                → refined classification
    ├─ EnzymeLens                  (92% accuracy)
    ├─ HingeLens                   (100% accuracy on 12-protein corpus)
    ├─ BarrelPenaltyLens
    ├─ AllostericLens
    └─ FlowGrammarLens
    ↓
Final archetype: enzyme_active | barrel | allosteric | dumbbell | globin
```

---

## 3. Claims Audit

### Verified TRUE ✓

| Claim | Evidence |
|-------|----------|
| Zero-parameter domain detection | `analyzer.py` Fiedler partitioning: `labels = (fiedler >= 0).astype(int)`. No thresholds. |
| Exactly 7 instruments | `instruments.py` defines exactly 7, named and documented. |
| AlgebraicFickBalancer has 0 tunable parameters | Weights derived from sedenion spectral constants (√2 ratios). Not fitted. |
| Only numpy/scipy/requests required | `pyproject.toml` confirms. GNCA is optional extra. |
| MIT licensed, on PyPI | Confirmed. PyPI page exists, badge works. |

### Unverified — needs automated tests ⚠

| Claim | Current Evidence | Risk |
|-------|-----------------|------|
| HingeLens 100% on 12 proteins | Experiment D111 log only | **HIGH** — headline claim, no CI test |
| EnzymeLens 92% on 12 proteins | Experiment D110 log only | HIGH — stated in README |
| Band 83% on 12 proteins | README table | MEDIUM |
| 57.7% on 52 proteins (0 params) | SPRINT.md line | MEDIUM |
| GNCA 64.1% on 865 proteins | README only | LOW (less central claim) |

### Inaccurate — needs correction ✗

| Claim | Reality | Fix |
|-------|---------|-----|
| "475+ tests" (README) | **465 test functions** across 12 files | Update count |
| README elsewhere says "402 tests" | Also stale | Update count |

### Needs Clarification

| Claim | Issue |
|-------|-------|
| "Zero free parameters" | True for `AlgebraicFickBalancer` itself, but the full pipeline uses ~90 thresholds in `ThresholdRegistry`. Should clarify scope. |

---

## 4. Proposed Cleanup Plan

### Phase 1: Credibility (do before preprint/grants) — ~2-3 days

| # | Task | Priority | Effort |
|---|------|----------|--------|
| 1.1 | **Add integration test: 12-protein HingeLens 100%** | CRITICAL | 2h |
| | Create `tests/test_integration.py` that fetches 12 PDBs and asserts HingeLens correctness. Mark with `@pytest.mark.network` so offline tests still pass. | | |
| 1.2 | **Add integration test: EnzymeLens 92%** | CRITICAL | 1h |
| | Same file, assert ≥11/12 correct. | | |
| 1.3 | **Define "IBP" in README + __init__.py** | HIGH | 15min |
| | "Instrument-Based Probing" — add to first line of README and module docstring. | | |
| 1.4 | **Fix test count in README** | HIGH | 10min |
| | Update to actual count (465), or just say "460+" to allow for minor fluctuation. | | |
| 1.5 | **Clarify "zero parameters" scope** | HIGH | 15min |
| | README should say "zero-parameter vote fusion" not imply the whole pipeline is parameter-free. | | |
| 1.6 | **Add CI job for integration tests** | HIGH | 1h |
| | GitHub Actions workflow that runs network tests weekly (not on every push). | | |

### Phase 2: Test Coverage — ~3-4 days

| # | Task | Priority | Effort |
|---|------|----------|--------|
| 2.1 | **Add `test_analyzer.py`** | HIGH | 4h |
| | The 2,756-line core module has NO dedicated tests. Test analyze/compare/probe/listen with synthetic matrices. | | |
| 2.2 | **Add `test_synthesis.py`** | HIGH | 2h |
| | Test MetaFickBalancer and AlgebraicFickBalancer vote fusion directly. | | |
| 2.3 | **Add `test_carving.py`** | MEDIUM | 2h |
| 2.4 | **Add `test_archetypes.py`** | MEDIUM | 1h |
| 2.5 | **Add `test_functional_sites.py`** | LOW | 2h |
| 2.6 | **Add `test_graph_data.py`** | LOW | 2h |

12 of 24 source modules currently lack dedicated test files.

### Phase 3: API Surface Cleanup — ~2 days

| # | Task | Priority | Effort |
|---|------|----------|--------|
| 3.1 | **Trim `__init__.py` exports** | MEDIUM | 1h |
| | Currently exports ~80 symbols. A typical user needs ~10. Move internal abstractions to submodule imports. Keep public API to: `IBPProteinAnalyzer`, `IBPResult`, `ThermodynamicBand`, `run_single_protein`, `fetch_pdb_ca_data`, `PROTEINS`, `GROUND_TRUTH`, `ProteinArchetype`, `BenchmarkRunner`, and the thermodynamic observables. | | |
| 3.2 | **Reorganize into subpackages** | LOW | 4h |
| | Consider: `ibp_enm.core` (analyzer, band, instruments, synthesis), `ibp_enm.classify` (rules, thresholds, lenses, archetypes), `ibp_enm.research` (algebra, belief, analysis, trace, cache), `ibp_enm.gnca` (graph_data, gnca, trainer). This improves discoverability. | | |
| 3.3 | **Write proper API docs page** | MEDIUM | 3h |
| | The README's API reference is good but a dedicated docs site (even MkDocs) would serve grant reviewers. | | |

### Phase 4: Documentation for Publication — ~2-3 days

| # | Task | Priority | Effort |
|---|------|----------|--------|
| 4.1 | **Write METHODS.md** | HIGH | 4h |
| | Formal methods description suitable for a preprint. Cover: ENM construction, Fiedler partitioning, 7 instruments (with equations), AlgebraicFickBalancer derivation, lens stack architecture. | | |
| 4.2 | **Write BENCHMARKS.md** | HIGH | 2h |
| | Full reproducible benchmark results: 12-protein, 52-protein, 200-protein, 865-protein (GNCA). Comparison vs DynDom, HingeProt if feasible. | | |
| 4.3 | **BioRxiv preprint draft** | HIGH | 8h |
| | Standalone document. See funding roadmap for structure. | | |

---

## 5. Priority Ordering

If time is extremely limited (3-month runway), do **only Phase 1** before applying for grants. The integration tests (1.1, 1.2) are the single most important items — they turn "we claim 100%" into "CI proves 100% on every commit."

**Critical path:** 1.1 → 1.2 → 1.3 → 1.4 → 1.5 → 4.3 (preprint draft)

Everything else increases quality but isn't blocking for grant applications.
