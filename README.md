# IBP-ENM

**Spectral Elastic Network Model for Protein Structural Analysis**

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

## Testing

```bash
pip install pytest
pytest tests/ -v
```

All 50 tests pass on Python 3.10–3.12.

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
