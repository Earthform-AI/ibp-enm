"""Integration tests — validate headline accuracy claims against live PDB data.

These tests fetch real protein structures from RCSB PDB and run the full
IBP-ENM classification pipeline.  They are skipped by default because they
require network access and are compute-intensive (~3-10 minutes per protein,
~30-90 minutes total for the full corpus).

Run with:
    pytest tests/test_integration.py -v --run-network

Claims verified:
    1. HingeLens achieves 100% accuracy on the 12-protein benchmark corpus
    2. Every protein in the corpus is classifiable without errors
"""

import json
import time
from pathlib import Path

import pytest

from ibp_enm import run_single_protein, PROTEINS, GROUND_TRUTH


# ---------------------------------------------------------------------------
# Result caching — avoid re-running expensive analysis on subsequent runs
# ---------------------------------------------------------------------------

_CACHE_DIR = Path(__file__).parent / ".integration_cache"


def _cache_path(name: str) -> Path:
    return _CACHE_DIR / f"{name}.json"


def _classify_cached(name: str, pdb_id: str, chain: str) -> dict:
    """Run the full pipeline, caching the classification result to disk.

    The cache stores only the fields needed for assertion (not the full
    eigenvalue arrays) so re-runs complete in seconds.
    """
    cp = _cache_path(name)
    if cp.exists():
        return json.loads(cp.read_text())

    result = run_single_protein(pdb_id, chain, name=name, verbose=True)

    # Extract only the JSON-serialisable fields we actually test
    cached = {
        "name": name,
        "pdb_id": pdb_id,
        "chain": chain,
        "N": result["N"],
        "hinge_identity": result["hinge_identity"],
        "band_identity": result["band_identity"],
        "band_correct": result["band_correct"],
        "true_archetype": result["true_archetype"],
        "enzyme_lens_activated": result["enzyme_lens_activated"],
        "hinge_lens_activated": result["hinge_lens_activated"],
        "time_s": result["time_s"],
    }

    _CACHE_DIR.mkdir(exist_ok=True)
    cp.write_text(json.dumps(cached, indent=2))
    return cached


# ---------------------------------------------------------------------------
# Per-protein smoke tests (parametrised)
# ---------------------------------------------------------------------------

@pytest.mark.network
@pytest.mark.parametrize(
    "name,pdb_id,chain",
    PROTEINS,
    ids=[p[0] for p in PROTEINS],
)
def test_single_protein_classifiable(name, pdb_id, chain):
    """Each benchmark protein runs without error and returns a valid archetype."""
    result = _classify_cached(name, pdb_id, chain)
    assert result["hinge_identity"] in {
        "enzyme_active", "barrel", "allosteric", "dumbbell", "globin",
    }, f"{name}: got unexpected identity '{result['hinge_identity']}'"
    assert result["N"] >= 20, f"{name}: only {result['N']} residues"


@pytest.mark.network
@pytest.mark.parametrize(
    "name,pdb_id,chain",
    PROTEINS,
    ids=[p[0] for p in PROTEINS],
)
def test_hinge_lens_correctness(name, pdb_id, chain):
    """HingeLens (the default pipeline) matches ground truth for every protein.

    This is the headline claim: 100% accuracy on the 12-protein corpus.
    """
    expected = GROUND_TRUTH[name]
    result = _classify_cached(name, pdb_id, chain)
    got = result["hinge_identity"]
    assert got == expected, (
        f"{name} ({pdb_id}:{chain}): "
        f"expected '{expected}', got '{got}'"
    )


# ---------------------------------------------------------------------------
# Corpus-level accuracy assertion
# ---------------------------------------------------------------------------

@pytest.mark.network
def test_hinge_lens_100_percent():
    """HingeLens must achieve 100% accuracy (12/12)."""
    failures = []
    for name, pdb_id, chain in PROTEINS:
        result = _classify_cached(name, pdb_id, chain)
        expected = GROUND_TRUTH[name]
        got = result["hinge_identity"]
        if got != expected:
            failures.append(f"  {name}: expected '{expected}', got '{got}'")

    total = len(PROTEINS)
    correct = total - len(failures)
    assert not failures, (
        f"HingeLens accuracy: {correct}/{total} "
        f"({100 * correct / total:.1f}%). Failures:\n"
        + "\n".join(failures)
    )


@pytest.mark.network
def test_all_archetypes_represented():
    """The corpus should cover all 5 archetypes in ground truth."""
    expected_archetypes = set(GROUND_TRUTH.values())
    assert len(expected_archetypes) == 5, (
        f"Ground truth should have 5 archetypes, has {len(expected_archetypes)}"
    )


@pytest.mark.network
def test_ground_truth_complete():
    """Every protein in PROTEINS has a ground truth entry."""
    for name, pdb_id, chain in PROTEINS:
        assert name in GROUND_TRUTH, (
            f"Protein '{name}' missing from GROUND_TRUTH"
        )
