"""Baseline methods for comparison.

GNM (Gaussian Network Model) — the standard reference method for
B-factor prediction from ENM.
"""

import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform
from typing import Tuple, Optional


def gnm_predict(coords: np.ndarray, b_factors: Optional[np.ndarray] = None,
                cutoff: float = 7.3) -> Tuple[np.ndarray, Optional[float]]:
    """Predict B-factors using standard GNM.

    Parameters
    ----------
    coords : (N, 3) array
        Cα positions.
    b_factors : (N,) array, optional
        Experimental B-factors (for correlation).
    cutoff : float
        Contact cutoff (default 7.3 Å, standard for GNM).

    Returns
    -------
    b_pred : (N,) array
        Predicted B-factors.
    rho : float or None
        Spearman correlation with experiment, if b_factors given.
    """
    N = len(coords)
    dmat = squareform(pdist(coords))

    # Build Kirchhoff matrix (unweighted graph Laplacian)
    G = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            if dmat[i, j] <= cutoff:
                G[i, j] = -1
                G[j, i] = -1
                G[i, i] += 1
                G[j, j] += 1

    evals, evecs = np.linalg.eigh(G)
    b_pred = np.zeros(N)
    for k in range(1, len(evals)):
        if evals[k] > 1e-10:
            b_pred += evecs[:, k]**2 / evals[k]

    b_scale = b_factors.max() if b_factors is not None else 1.0
    if b_pred.max() > 0:
        b_pred = b_pred / b_pred.max() * b_scale

    rho = None
    if b_factors is not None:
        rho, _ = spearmanr(b_pred, b_factors)

    return b_pred, rho
