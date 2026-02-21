# IBP-ENM

**Spectral Elastic Network Model for Protein Structural Analysis**

[![CI](https://github.com/Earthform-AI/ibp-enm/actions/workflows/ci.yml/badge.svg)](https://github.com/Earthform-AI/ibp-enm/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/ibp-enm)](https://pypi.org/project/ibp-enm/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

IBP-ENM extracts per-residue structural role profiles, unsupervised domain boundaries, hinge sites, and multi-perspective flexibility predictions from a single static protein structure — no training data, no homology templates, no MD simulations.

The **Thermodynamic Band** extends the core elastic network analyzer with a 7-instrument carving protocol that classifies proteins into structural archetypes (enzyme, barrel, allosteric, dumbbell, globin) achieving **100% accuracy** on the 12-protein benchmark corpus.

## Key Features

- **Zero-parameter domain detection** — Fiedler vector sign changes partition the contact graph into structural domains without any tunable thresholds.
- **7 thermodynamic instruments** — Each probes a distinct physical signal (symmetry breaking, resonance sensitivity, diffusion-optimal cuts, entropy disruption, cooperativity, allosteric reach, thermal fragility).
- **Two synthesis lenses** — EnzymeLens (92% accuracy) and HingeLens (100% accuracy) fuse instrument votes via Meta-Fick diffusion consensus.
- **Self-contained** — Only requires `numpy`, `scipy`, and `requests`. Fetches PDB structures directly from RCSB.

## Installation

```bash
pip install ibp-enm
```

Or from source:

```bash
git clone https://github.com/Earthform-AI/ibp-enm.git
cd ibp-enm
pip install -e .
```

## Quick Start

### Classify a protein

```python
from ibp_enm import run_single_protein

result = run_single_protein("2LZM", chain="A", verbose=True)
print(result["band_identity"])   # → "enzyme_active"
print(result["hinge_identity"])  # → "enzyme_active"
```

### Core structural analysis

```python
from ibp_enm import IBPProteinAnalyzer, fetch_pdb_ca_data

coords, bfactors = fetch_pdb_ca_data("1MBO", "A")
analyzer = IBPProteinAnalyzer()
result = analyzer.analyze(coords, bfactors)

print(f"Domains: {result.n_domains}")
print(f"Hinge residues: {result.hinge_indices}")
print(f"B-factor correlation: {result.consensus_correlation:.3f}")
```

### Run the full 12-protein benchmark

```python
from ibp_enm import PROTEINS, GROUND_TRUTH, run_single_protein

for pdb_id, chain, name in PROTEINS:
    r = run_single_protein(pdb_id, chain, name)
    expected = GROUND_TRUTH.get(pdb_id, "?")
    got = r["hinge_identity"]
    match = "✓" if got == expected else "✗"
    print(f"  {match} {name:20s}  expected={expected:20s}  got={got}")
```

## Architecture

```
IBPProteinAnalyzer          ← Core ENM: analyze / compare / probe / listen
    │
ThermodynamicBand           ← 7-instrument orchestrator
├─ 7 × ThermoInstrumentCarver
│      algebraic   — max |Δgap|         (symmetry breaking)
│      musical     — max mode_scatter   (resonance sensitivity)
│      fick        — FickBalancer       (diffusion-optimal cut)
│      thermal     — max |Δτ|           (entropy disruption)
│      cooperative — max |Δβ|           (cooperativity probe)
│      propagative — max spatial_r      (allosteric reach)
│      fragile     — max bus_mass       (thermal soft spots)
│
├─ MetaFickBalancer         ← Consensus / disagreement fusion
│   ├─ EnzymeLensSynthesis  ← 92% accuracy (D110)
│   └─ HingeLensSynthesis   ← 100% accuracy (D111)
│
└─ SurgeonsHandbook         ← Initial snapshot diagnosis
```

## The 12-Protein Benchmark

| PDB  | Protein              | Archetype     | Band | Enzyme Lens | Hinge Lens |
|------|----------------------|---------------|------|-------------|------------|
| 2LZM | T4 Lysozyme         | enzyme_active | ✓    | ✓           | ✓          |
| 1MBO | Myoglobin            | globin        | ✓    | ✓           | ✓          |
| 2DHB | Deoxyhemoglobin      | globin        | ✓    | ✓           | ✓          |
| 1GGG | Galactose Oxidase    | barrel        | ✓    | ✓           | ✓          |
| 2POR | Porin                | barrel        | ✓    | ✓           | ✓          |
| 4AKE | Adenylate Kinase     | enzyme_active | ✗    | ✗           | ✓          |
| 1ANF | ABP (open)           | allosteric    | ✓    | ✓           | ✓          |
| 3CLN | Calmodulin           | dumbbell      | ✓    | ✓           | ✓          |
| 1LFG | Lactoferrin          | allosteric    | ✗    | ✓           | ✓          |
| 1OMP | OmpA                 | barrel        | ✓    | ✓           | ✓          |
| 1HNF | Inorganic PPase      | enzyme_active | ✓    | ✓           | ✓          |
| 5CYT | Cytochrome c         | globin        | ✓    | ✓           | ✓          |

**Accuracy**: Band 83% → EnzymeLens 92% → HingeLens **100%**

## API Reference

### Core Analysis

| Symbol | Description |
|--------|-------------|
| `IBPProteinAnalyzer` | Core ENM analyzer with `analyze()`, `compare()`, `probe()`, `listen()` |
| `IBPResult` | Dataclass holding analysis results (domains, hinges, B-factors, etc.) |
| `fetch_pdb_ca_data(pdb_id, chain)` | Fetch Cα coordinates + B-factors from RCSB PDB |
| `search_rcsb(query)` | Search RCSB for PDB entries |
| `gnm_predict(coords)` | GNM baseline B-factor prediction |

### Thermodynamic Band

| Symbol | Description |
|--------|-------------|
| `ThermodynamicBand` | 7-instrument orchestrator class |
| `run_single_protein(pdb_id, chain)` | Run full band pipeline on one protein |
| `ThermoInstrumentCarver` | Individual instrument carver |
| `ThermoReactionProfile` | Per-instrument reaction data |
| `MetaFickBalancer` | Vote fusion engine |
| `EnzymeLensSynthesis` | Enzyme-calibrated lens (92% accuracy) |
| `HingeLensSynthesis` | Hinge-calibrated lens (100% accuracy) |

### Archetypes & Data

| Symbol | Description |
|--------|-------------|
| `ProteinArchetype` | Enum: `enzyme_active`, `barrel`, `allosteric`, `dumbbell`, `globin` |
| `PROTEINS` | The 12-protein benchmark corpus |
| `GROUND_TRUTH` | Expected archetype for each benchmark protein |
| `SurgeonsHandbook` | Initial snapshot diagnosis generator |

### Thermodynamic Observables

| Symbol | Description |
|--------|-------------|
| `vibrational_entropy(evals)` | Vibrational entropy from eigenvalue spectrum |
| `heat_capacity(evals, T)` | Heat capacity from mode occupation |
| `helmholtz_free_energy(evals, T)` | Helmholtz free energy |
| `spectral_entropy_shannon(evals)` | Shannon entropy of the eigenvalue distribution |
| `per_residue_entropy_contribution(evecs, evals)` | Per-residue entropy decomposition |
| `hinge_occupation_ratio(evecs, hinge_indices)` | Mode participation at hinge sites |
| `domain_stiffness_asymmetry(evals_d1, evals_d2)` | Stiffness asymmetry between domains |

### Graph Data Pipeline (v0.9.0)

Converts proteins into PyTorch Geometric `Data` objects for
Graph Neural Cellular Automata (GNCA) training.

```python
# Single protein → PyG graph
from ibp_enm.graph_data import protein_to_pyg_data
data = protein_to_pyg_data("2LZM", "A", archetype="globin")
# data.x: (N, 15)  node features
# data.edge_index: (2, 2E)  bidirectional edges
# data.edge_attr: (2, E)  distance + cross-domain flag
# data.pos: (N, 3)  Cα coordinates

# Full corpus → dataset + DataLoader
from ibp_enm.graph_data import corpus_to_dataset, dataset_to_loader, stratified_split
from ibp_enm.benchmark import LARGE_CORPUS
dataset = corpus_to_dataset(LARGE_CORPUS)
train, val, test = stratified_split(dataset)
loader = dataset_to_loader(train, batch_size=16)
```

| Symbol | Description |
|--------|-------------|
| `protein_to_pyg_data(pdb_id, chain, archetype)` | Convert one protein to PyG `Data` |
| `corpus_to_dataset(corpus)` | Batch-convert a `ProteinEntry` list |
| `dataset_to_loader(dataset, batch_size)` | Wrap in PyG `DataLoader` |
| `stratified_split(dataset)` | Archetype-stratified train/val/test split |
| `compute_dataset_stats(dataset)` | Summary statistics (`DatasetStats`) |
| `ARCHETYPE_NAMES`, `ARCHETYPE_TO_IDX` | Label mapping (5 classes) |
| `NODE_FEATURE_DIM` (15), `EDGE_FEATURE_DIM` (2) | Feature dimensions |

**Node features (15):** B-factor (exp + 3 predicted), Fiedler vector,
domain label, hinge score, stabiliser profiles (2), degree, spectral gap,
per-residue entropy, Cα coordinates (x, y, z) *or* rotationally-invariant
spatial features (centroid distance, local density, contact density) when
`include_coords=False`.

**Edge features (2):** Contact distance (Å), cross-domain flag.

Requires: `pip install ibp-enm[gnca]` (adds `torch`, `torch-geometric`).

### Graph NCA Classifier (v0.9.0)

Graph Neural Cellular Automata for protein archetype classification,
following Grattarola et al. 2021 (Graph NCA architecture) and Walker et al.
2022 (classification readout).  Operates directly on the protein contact
graph without hand-crafted global features.

```python
from ibp_enm.gnca import GNCAConfig, GNCAClassifier
from ibp_enm.gnca_trainer import cross_validate_gnca
from ibp_enm.graph_data import corpus_to_dataset
from ibp_enm.benchmark import LARGE_CORPUS

import torch

dataset = corpus_to_dataset(LARGE_CORPUS, normalize_features=False, include_coords=False)
# Dataset-level z-normalization (critical — per-protein norm destroys global features)
all_x = torch.cat([d.x for d in dataset], dim=0)
mean, std = all_x.mean(0, keepdim=True), all_x.std(0, keepdim=True)
std[std < 1e-6] = 1.0
for d in dataset:
    d.x = (d.x - mean) / std

config = GNCAConfig(state_dim=48, hidden_dim=32, t_min=1, t_max=8)
result = cross_validate_gnca(dataset, n_folds=10, config=config)
print(result.summary())  # ~55% accuracy (10-fold CV)
```

| Symbol | Description |
|--------|-------------|
| `GNCAConfig` | Hyperparameter dataclass (state/hidden dims, NCA steps, LR, etc.) |
| `GNCAClassifier` | Encoder → T×GNCACell → mean-pool readout classifier |
| `GNCACell` | Single NCA step (message-passing + additive residual update) |
| `GNCATrainer` | Training loop with early stopping, class weighting, gradient clipping |
| `cross_validate_gnca(dataset, n_folds)` | Stratified k-fold cross-validation |
| `TrainResult` / `CVResult` | Structured result objects with confusion matrices |

**Key findings (D137c–h):**
- **64.1% ± 4.5%** (10-fold CV, 865 proteins) — current high-water mark (D137h)
- **55.5% ± 10.1%** (10-fold CV, 200 proteins) — baseline with proper normalization
- **Dataset-level z-normalization** is critical: per-protein normalization
  destroys cross-protein discriminative signal in global features (spectral gap,
  per-residue entropy scale), dropping accuracy from 55% → 50.5%
- Expanding corpus from 200 → 865 proteins gave +9pp accuracy and
  halved the variance (±10.7% → ±4.5%)
- Rotationally-invariant spatial features (`include_coords=False`) were
  critical: 22.5% → 46–50%
- NCA dynamics genuinely help over static GNN: 39.5% (T=1) → 50.5% (T=[1,8])
- Virtual node and thermodynamic band features did not improve over the
  plain 15-feature baseline with proper normalization
- Barrel accuracy nearly doubled with expanded corpus: 25% → 53%
- Model: 6,576 parameters (deliberately tiny for the data regime)

## Testing

```bash
pip install pytest
pytest tests/ -v
```

All 402 tests pass on Python 3.10–3.12.

## Citation

If you use IBP-ENM in your research, please cite:

```bibtex
@article{byrom2025ibpenm,
  title   = {IBP-ENM: Spectral Elastic Network Fingerprints for
             Protein Conformational Analysis},
  author  = {Joshua Byrom},
  year    = {2025},
  url     = {https://earthform.ai/papers/ibp_enm_spectral_protein.pdf}
}

@article{byrom2025thermoband,
  title   = {Thermodynamic Band Classification of Protein
             Structural Archetypes},
  author  = {Joshua Byrom},
  year    = {2025},
  url     = {https://earthform.ai/papers/thermo_band_archetypes.pdf}
}
```

## License

MIT — see [LICENSE](LICENSE).
