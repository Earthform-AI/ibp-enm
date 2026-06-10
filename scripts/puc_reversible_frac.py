"""PUC Observation Metric: Reversible Fraction via Fano Spectral Analysis.

Bridges ibp_enm's reversible_frac concept to PUC v6 hardware state trajectories.

The PUC evolves 7-component state vectors on the Fano plane PG(2,2).
This script computes what fraction of evolution steps preserve the global
spectral structure of the state-weighted Fano graph — i.e., whether the
perturbation chain satisfies a structural reversibility criterion.

High reversible_frac (>0.7): system is making coherent, globally-consistent
    perturbations (allosteric-like behavior). Evolution respects topology.
Low reversible_frac (<0.3): system is making destructive, topology-breaking
    changes (globin-like). May indicate either divergence or deep restructuring.

Usage:
    python scripts/puc_reversible_frac.py trajectory.json
    python scripts/puc_reversible_frac.py --from-uart /dev/ttyUSB0

The trajectory JSON format:
    {"states": [[v0, v1, v2, v3, v4, v5, v6], ...], "epochs": N}

Output:
    Reversible fraction, per-epoch gap trajectory, and classification.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

# ─────────────────────── Fano Plane PG(2,2) ───────────────────────

# The 7 lines of the Fano plane (each containing 3 points)
FANO_LINES = [
    (0, 1, 3),  # Line 0
    (1, 2, 4),  # Line 1
    (2, 3, 5),  # Line 2
    (3, 4, 6),  # Line 3
    (4, 5, 0),  # Line 4
    (5, 6, 1),  # Line 5
    (6, 0, 2),  # Line 6
]

# Point-to-line incidence: which lines does each point belong to?
POINT_LINES = [
    [0, 4, 6],  # Point 0: lines 0, 4, 6
    [0, 1, 5],  # Point 1: lines 0, 1, 5
    [1, 2, 6],  # Point 2: lines 1, 2, 6
    [0, 2, 3],  # Point 3: lines 0, 2, 3
    [1, 3, 4],  # Point 4: lines 1, 3, 4
    [2, 4, 5],  # Point 5: lines 2, 4, 5
    [3, 5, 6],  # Point 6: lines 3, 5, 6
]


def fano_weighted_laplacian(state: np.ndarray) -> np.ndarray:
    """Build a state-weighted Fano graph Laplacian.

    Each Fano line connects 3 points. The edge weight between two points
    on a line is the product of their absolute state values (Q4.12 → float).
    This captures "interaction strength" — two active axes on the same line
    create a strong topological coupling.

    Parameters
    ----------
    state : (7,) array
        State vector (Q4.12 converted to float, or raw float).

    Returns
    -------
    L : (7, 7) ndarray
        Weighted graph Laplacian.
    """
    L = np.zeros((7, 7))
    abs_state = np.abs(state) + 1e-12  # avoid zero weights

    for line in FANO_LINES:
        p0, p1, p2 = line
        # Weight edges by geometric mean of connected point values
        w01 = np.sqrt(abs_state[p0] * abs_state[p1])
        w12 = np.sqrt(abs_state[p1] * abs_state[p2])
        w02 = np.sqrt(abs_state[p0] * abs_state[p2])

        L[p0, p1] -= w01
        L[p1, p0] -= w01
        L[p1, p2] -= w12
        L[p2, p1] -= w12
        L[p0, p2] -= w02
        L[p2, p0] -= w02

        L[p0, p0] += w01 + w02
        L[p1, p1] += w01 + w12
        L[p2, p2] += w12 + w02

    return L


def spectral_features(L: np.ndarray) -> Tuple[float, np.ndarray]:
    """Extract spectral gap and Fiedler vector from Laplacian.

    Returns
    -------
    gap : float
        Second-smallest eigenvalue (algebraic connectivity).
    fiedler : (7,) array
        Eigenvector corresponding to λ₂ (normalized).
    """
    evals, evecs = np.linalg.eigh(L)
    # Sort by eigenvalue (should already be sorted, but be safe)
    idx = np.argsort(evals)
    evals = evals[idx]
    evecs = evecs[:, idx]

    gap = float(evals[1]) if len(evals) > 1 else 0.0
    fiedler = evecs[:, 1] if evecs.shape[1] > 1 else np.zeros(7)
    # Normalize Fiedler vector
    norm = np.linalg.norm(fiedler)
    if norm > 1e-10:
        fiedler = fiedler / norm

    return gap, fiedler


def is_reversible(
    gap_before: float,
    gap_after: float,
    fiedler_before: np.ndarray,
    fiedler_after: np.ndarray,
    gap_tol: float = 0.10,
    fiedler_tol: float = 0.90,
) -> bool:
    """Test if a state transition preserves global spectral structure.

    Adapted from ibp_enm's carving reversibility test:
    - Gap recovery: |gap_after/gap_before - 1| < gap_tol
    - Fiedler recovery: |dot(f_before, f_after)| > fiedler_tol

    Parameters
    ----------
    gap_before, gap_after : float
        Spectral gap (λ₂) before and after the perturbation.
    fiedler_before, fiedler_after : (7,) arrays
        Fiedler vectors before and after.
    gap_tol : float
        Maximum relative gap change (default 10%).
    fiedler_tol : float
        Minimum Fiedler vector similarity (default 0.90).

    Returns
    -------
    bool
        True if the perturbation is spectrally reversible.
    """
    if gap_before < 1e-10:
        # Degenerate base: any non-zero gap after is irreversible
        return gap_after < 1e-10

    gap_recovery = gap_after / gap_before
    fiedler_similarity = abs(float(np.dot(fiedler_before, fiedler_after)))

    return (abs(gap_recovery - 1.0) < gap_tol and
            fiedler_similarity > fiedler_tol)


def compute_reversible_frac(
    trajectory: List[np.ndarray],
    gap_tol: float = 0.10,
    fiedler_tol: float = 0.90,
) -> dict:
    """Compute the reversible fraction for a PUC state trajectory.

    Parameters
    ----------
    trajectory : list of (7,) arrays
        Sequence of state vectors from evolution epochs.
    gap_tol : float
        Gap tolerance (default 10%).
    fiedler_tol : float
        Fiedler similarity threshold (default 0.90).

    Returns
    -------
    dict with keys:
        reversible_frac : float (0.0 to 1.0)
        gap_trajectory : list of float (spectral gap per epoch)
        reversibility : list of bool (per-transition)
        classification : str ('coherent', 'active', 'destructive')
    """
    if len(trajectory) < 2:
        return {
            "reversible_frac": 1.0,
            "gap_trajectory": [],
            "reversibility": [],
            "classification": "insufficient_data",
        }

    gaps = []
    fiedlers = []

    for state in trajectory:
        L = fano_weighted_laplacian(state)
        gap, fiedler = spectral_features(L)
        gaps.append(gap)
        fiedlers.append(fiedler)

    reversibility = []
    for i in range(len(trajectory) - 1):
        rev = is_reversible(
            gaps[i], gaps[i + 1],
            fiedlers[i], fiedlers[i + 1],
            gap_tol=gap_tol,
            fiedler_tol=fiedler_tol,
        )
        reversibility.append(rev)

    rev_frac = float(np.mean(reversibility)) if reversibility else 1.0

    # Classification (matching ibp_enm archetype thresholds)
    if rev_frac > 0.7:
        classification = "coherent"      # allosteric-like
    elif rev_frac > 0.3:
        classification = "active"        # enzyme-like
    else:
        classification = "destructive"   # globin-like

    return {
        "reversible_frac": rev_frac,
        "gap_trajectory": gaps,
        "reversibility": reversibility,
        "classification": classification,
    }


# ─────────────────────── CLI Entry Point ───────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/puc_reversible_frac.py <trajectory.json>")
        print("       python scripts/puc_reversible_frac.py --demo")
        sys.exit(1)

    if sys.argv[1] == "--demo":
        # Demo: generate a synthetic trajectory (convergence scenario)
        np.random.seed(42)
        states = []
        state = np.random.randn(7) * 2.0  # random start
        target = np.array([0.5] * 7)
        for epoch in range(50):
            states.append(state.copy())
            # Move 10% toward target + small noise
            state = state + 0.1 * (target - state) + np.random.randn(7) * 0.05
        result = compute_reversible_frac(states)
        print(f"Demo trajectory ({len(states)} epochs):")
        print(f"  Reversible fraction: {result['reversible_frac']:.3f}")
        print(f"  Classification: {result['classification']}")
        print(f"  Gap range: [{min(result['gap_trajectory']):.4f}, "
              f"{max(result['gap_trajectory']):.4f}]")
        return

    # Load trajectory from JSON
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"Error: file not found: {path}")
        sys.exit(1)

    with open(path) as f:
        data = json.load(f)

    if "states" not in data:
        print("Error: JSON must have a 'states' key with list of 7-element arrays")
        sys.exit(1)

    trajectory = [np.array(s, dtype=float) for s in data["states"]]
    result = compute_reversible_frac(trajectory)

    print(f"Trajectory: {len(trajectory)} epochs")
    print(f"  Reversible fraction: {result['reversible_frac']:.3f}")
    print(f"  Classification: {result['classification']}")
    print(f"  Gap range: [{min(result['gap_trajectory']):.4f}, "
          f"{max(result['gap_trajectory']):.4f}]")
    print(f"  Transitions: {sum(result['reversibility'])} reversible / "
          f"{len(result['reversibility'])} total")

    # Write full result
    output_path = path.with_suffix(".reversibility.json")
    with open(output_path, "w") as f:
        json.dump({
            "reversible_frac": result["reversible_frac"],
            "classification": result["classification"],
            "gap_trajectory": result["gap_trajectory"],
            "reversibility": result["reversibility"],
        }, f, indent=2)
    print(f"  Full results: {output_path}")


if __name__ == "__main__":
    main()
