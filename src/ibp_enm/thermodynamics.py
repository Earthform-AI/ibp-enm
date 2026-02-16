"""Thermodynamic observables from the elastic network eigenvalue spectrum.

All quantities computed in reduced units (kB = ℏ = 1).  The ENM
eigenvalues λ_k carry units of spring-constant / mass; the normal-mode
frequencies are ω_k = √λ_k.  Temperature *T* sets the energy scale and
defaults to 1.0 (reduced).

Observables
-----------
vibrational_entropy    S_vib = Σ [x/(eˣ-1) − ln(1−e⁻ˣ)]
heat_capacity          C_v   = Σ x²eˣ/(eˣ−1)²
helmholtz_free_energy  F     = T Σ ln(1−e⁻ˣ)
heat_kernel_trace      Z(t)  = Σ e⁻λt
inverse_participation_ratio  IPR_k = Σ v_{k,i}⁴
spectral_entropy       H(λ)  = −Σ p_k ln p_k    (p_k = λ_k / Σλ)

where x_k = ω_k / T = √λ_k / T.

Historical notes
----------------
These functions were introduced in D109 (The Thermodynamic Band) to
provide the missing thermodynamic fingerprints that separate protein
archetypes.  The key discovery: each archetype has a unique
thermodynamic response to edge-removal ("carving"), and recording
ΔS, ΔC_v, ΔF per cut gives the discriminating signal that gap-only
carving missed.

Reference accuracy: D109 Run 2 = 83% (10/12), up from D106 = 58%.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "DEFAULT_TEMPERATURE",
    "vibrational_entropy",
    "heat_capacity",
    "helmholtz_free_energy",
    "heat_kernel_trace",
    "inverse_participation_ratio",
    "mean_ipr_low_modes",
    "spectral_entropy_shannon",
    "per_residue_entropy_contribution",
    "entropy_asymmetry_score",
    "multimode_ipr",
    "hinge_occupation_ratio",
    "domain_stiffness_asymmetry",
]

DEFAULT_TEMPERATURE: float = 1.0


# ── helpers ──────────────────────────────────────────────────────

def _positive_eigenvalues(eigenvalues: np.ndarray) -> np.ndarray:
    """Return eigenvalues > 1e-10 (skip zero modes)."""
    return eigenvalues[eigenvalues > 1e-10]


def _reduced_frequencies(eigenvalues: np.ndarray,
                         T: float = DEFAULT_TEMPERATURE) -> np.ndarray:
    """Convert λ → x = ω/T = √λ / T, clamped to [1e-6, 30]."""
    omega = np.sqrt(_positive_eigenvalues(eigenvalues))
    return np.clip(omega / T, 1e-6, 30.0)


# ── thermodynamic functions ──────────────────────────────────────

def vibrational_entropy(eigenvalues: np.ndarray,
                        T: float = DEFAULT_TEMPERATURE) -> float:
    """Vibrational entropy from the normal-mode spectrum.

    .. math::

        S_{\\text{vib}} = k_B \\sum_{k \\ge 1}
            \\left[\\frac{x_k}{e^{x_k}-1}
                   - \\ln\\bigl(1-e^{-x_k}\\bigr)\\right]

    High S_vib → floppy protein, many accessible conformations.
    Low  S_vib → rigid scaffold, few accessible conformations.
    """
    pos = _positive_eigenvalues(eigenvalues)
    if len(pos) == 0:
        return 0.0
    x = _reduced_frequencies(eigenvalues, T)
    return float(np.sum(x / (np.exp(x) - 1) - np.log(1 - np.exp(-x))))


def heat_capacity(eigenvalues: np.ndarray,
                  T: float = DEFAULT_TEMPERATURE) -> float:
    """Vibrational heat capacity.

    .. math::

        C_v = k_B \\sum_{k \\ge 1}
            \\frac{x_k^2 \\, e^{x_k}}{\\bigl(e^{x_k}-1\\bigr)^2}

    High C_v → many modes near kBT (rich dynamics).
    Low  C_v → modes too stiff or too soft for the current T.
    """
    pos = _positive_eigenvalues(eigenvalues)
    if len(pos) == 0:
        return 0.0
    x = _reduced_frequencies(eigenvalues, T)
    return float(np.sum(x**2 * np.exp(x) / (np.exp(x) - 1)**2))


def helmholtz_free_energy(eigenvalues: np.ndarray,
                          T: float = DEFAULT_TEMPERATURE) -> float:
    """Helmholtz free energy of the vibrational mode bath.

    .. math::

        F = k_B T \\sum_{k \\ge 1} \\ln\\bigl(1 - e^{-x_k}\\bigr)

    ΔF per carving cut measures the thermodynamic cost of removing
    a contact from the network.
    """
    pos = _positive_eigenvalues(eigenvalues)
    if len(pos) == 0:
        return 0.0
    x = _reduced_frequencies(eigenvalues, T)
    return float(np.sum(T * np.log(1 - np.exp(-x))))


def heat_kernel_trace(eigenvalues: np.ndarray, t: float = 1.0) -> float:
    """Heat kernel trace (spectral partition function).

    .. math::

        Z(t) = \\sum_k e^{-\\lambda_k t}

    At *t* → 0 it counts modes; at *t* → ∞ it selects the lowest
    mode.  At intermediate *t* it probes the effective spectral
    dimension.
    """
    pos = _positive_eigenvalues(eigenvalues)
    return float(np.sum(np.exp(-pos * t)))


def inverse_participation_ratio(eigenvectors: np.ndarray,
                                mode_idx: int) -> float:
    """Inverse participation ratio for a single mode.

    .. math::

        \\text{IPR}_k = \\sum_i v_{k,i}^4

    Low  IPR → delocalised mode (spread over many residues).
    High IPR → localised mode (concentrated on few residues).
    """
    v = eigenvectors[:, mode_idx]
    return float(np.sum(v**4))


def mean_ipr_low_modes(eigenvectors: np.ndarray,
                       n_modes: int = 5) -> float:
    """Mean IPR of the lowest non-trivial normal modes.

    Enzyme active sites tend to have *localised* low modes (high IPR).
    Barrels have *delocalised* modes (low IPR).
    """
    N = eigenvectors.shape[1]
    iprs = [inverse_participation_ratio(eigenvectors, k)
            for k in range(1, min(1 + n_modes, N))]
    return float(np.mean(iprs)) if iprs else 0.0


def spectral_entropy_shannon(eigenvalues: np.ndarray) -> float:
    """Shannon entropy of the normalised eigenvalue distribution.

    .. math::

        H(\\lambda) = -\\sum_k p_k \\ln p_k,
        \\qquad p_k = \\lambda_k / \\sum \\lambda

    Measures how uniformly vibrational energy is spread across modes.
    """
    pos = _positive_eigenvalues(eigenvalues)
    if len(pos) == 0:
        return 0.0
    p = pos / np.sum(pos)
    return float(-np.sum(p * np.log(p + 1e-15)))


# ── per-residue entropy decomposition (D110) ────────────────────

def per_residue_entropy_contribution(
        eigenvalues: np.ndarray,
        eigenvectors: np.ndarray,
        T: float = DEFAULT_TEMPERATURE) -> np.ndarray:
    """Per-residue contribution to vibrational entropy.

    For each residue *i*, compute its contribution to S_vib:

    .. math::

        s_i = \\sum_k |v_{k,i}|^2 \\cdot s_k

    where *s_k* is mode *k*'s entropy contribution and
    |v_{k,i}|² gives residue *i*'s participation in mode *k*.

    Returns an (N,) array of per-residue entropy contributions.

    Enzymes have **asymmetric** s_i distributions (active site is
    a thermodynamic hot spot).  Allosteric proteins have more
    **symmetric** distributions.  This observable was discovered in
    D110 and is the key input to the enzyme lens.
    """
    pos_mask = eigenvalues > 1e-10
    pos_evals = eigenvalues[pos_mask]
    pos_evecs = eigenvectors[:, pos_mask]

    if len(pos_evals) == 0:
        return np.zeros(eigenvectors.shape[0])

    omega = np.sqrt(pos_evals)
    x = np.clip(omega / T, 1e-6, 30.0)

    # Per-mode entropy contribution
    s_k = x / (np.exp(x) - 1) - np.log(1 - np.exp(-x))  # (n_modes,)

    # Per-residue: weight by squared eigenvector component
    participation = pos_evecs ** 2   # (N, n_modes)
    s_per_residue = participation @ s_k  # (N,)

    return s_per_residue


def entropy_asymmetry_score(
        s_per_residue: np.ndarray) -> dict[str, float]:
    """Measure how asymmetric the per-residue entropy distribution is.

    Returns a dict with four asymmetry metrics:

    - **gini**: Gini coefficient (0 = uniform, 1 = one residue has all)
    - **cv**: coefficient of variation (std / mean)
    - **top5_frac**: fraction of total entropy in the top 5% of residues
    - **kurtosis**: excess kurtosis (high = peaked / heavy-tailed)

    Empirical ranges from D110 (12-protein benchmark):

    =============  =====  =====  ======  =====
    Archetype      gini    cv    top5%   kurt
    =============  =====  =====  ======  =====
    enzyme_active  0.202  0.379  0.094   1.43
    barrel         0.205  0.379  0.098   1.50
    allosteric     0.154  0.280  0.079   0.89
    dumbbell       0.169  0.314  0.088   3.24
    globin         0.163  0.302  0.086   2.17
    =============  =====  =====  ======  =====

    The key discriminator: enzymes have **high gini** (>0.19) and
    **high cv** (>0.35) because their active sites concentrate
    vibrational entropy.  Allosteric proteins have **low gini** (<0.16)
    and **low cv** (<0.30).
    """
    s = np.sort(s_per_residue)
    N = len(s)
    total = np.sum(s)

    if total < 1e-10 or N < 2:
        return {"gini": 0.0, "cv": 0.0, "top5_frac": 0.0, "kurtosis": 0.0}

    # Gini coefficient
    index = np.arange(1, N + 1)
    gini = float((2 * np.sum(index * s) / (N * total)) - (N + 1) / N)

    # Coefficient of variation
    cv = float(np.std(s) / (np.mean(s) + 1e-10))

    # Top-5% entropy fraction
    k = max(1, N // 20)
    top5_frac = float(np.sum(s[-k:]) / total)

    # Excess kurtosis
    mu = np.mean(s)
    sig = np.std(s)
    if sig > 1e-10:
        kurtosis = float(np.mean(((s - mu) / sig) ** 4) - 3)
    else:
        kurtosis = 0.0

    return {"gini": gini, "cv": cv, "top5_frac": top5_frac,
            "kurtosis": kurtosis}


# ── multi-mode hinge observables (D111) ─────────────────────────

def multimode_ipr(
        eigenvectors: np.ndarray,
        modes: range = range(2, 6),
) -> float:
    """Mean IPR of higher normal modes (default: modes 2–5).

    Mode 1 (Fiedler) captures the global domain hinge motion.
    Modes 2–5 capture finer-grained dynamics.  Enzymes tend to
    have *more localised* higher modes than allosteric proteins
    because the active site imposes constraints that shape the
    local mode landscape.

    Empirical ranges from D111 (12-protein benchmark):

    =============  ========
    Archetype      IPR₂₋₅
    =============  ========
    enzyme_active  0.018–0.057
    barrel         0.014–0.016
    allosteric     0.013
    dumbbell       0.017–0.022
    globin         0.017
    =============  ========

    Parameters
    ----------
    eigenvectors : (N, N) ndarray
        Eigenvectors of the graph Laplacian.
    modes : range
        Column indices to average (default ``range(2, 6)``,
        i.e. the 3rd through 6th eigenvectors).

    Returns
    -------
    float
        Mean IPR across the specified modes.

    Historical notes
    ----------------
    Discovered in D111 (Multi-Mode Hinge Analysis).
    """
    N = eigenvectors.shape[1]
    iprs = []
    for k in modes:
        if k < N:
            v = eigenvectors[:, k]
            iprs.append(float(np.sum(v ** 4)))
    return float(np.mean(iprs)) if iprs else 0.0


def hinge_occupation_ratio(
        eigenvectors: np.ndarray,
        domain_labels: np.ndarray,
        modes: range = range(2, 6),
        hinge_radius_frac: float = 0.10,
        min_hinge_radius: int = 3,
) -> float:
    """How much modes 2–5 amplitude concentrates near the domain hinge.

    The domain hinge is defined as residues within *hinge_radius*
    of a Fiedler domain boundary.  For each mode *k*, compute:

    .. math::

        h_k = \\frac{\\langle v_{k,i}^2 \\rangle_{\\text{near}}}
                    {\\langle v_{k,i}^2 \\rangle_{\\text{far}}}

    and return the average across modes.

    Physical interpretation:

    - **hinge_R > 1.0** — higher modes still "feel" the hinge.
      The inter-domain cleft is functionally important; enzymes
      use it for catalysis.
    - **hinge_R ≤ 1.0** — mode 1 exhausts the hinge's contribution.
      Typical of allosteric proteins where the hinge is purely
      structural.

    Empirical ranges from D111 (12-protein benchmark):

    =============  ===========
    Archetype      hinge_R₂₋₅
    =============  ===========
    enzyme_active  0.5–1.8
    barrel         3.2–10.6
    allosteric     0.95
    dumbbell       0.7–1.3
    globin         0.9–1.2
    =============  ===========

    The key discriminator: **enzymes** show hinge_R > 1.0
    consistently (modes 2–5 concentrate at the catalytic cleft),
    while **allosteric** proteins show hinge_R ≤ 1.0.

    Parameters
    ----------
    eigenvectors : (N, N) ndarray
        Eigenvectors of the graph Laplacian.
    domain_labels : (N,) ndarray
        Binary domain assignment (from Fiedler vector sign).
    modes : range
        Which modes to analyse (default ``range(2, 6)``).
    hinge_radius_frac : float
        Fraction of sequence length defining hinge neighbourhood.
    min_hinge_radius : int
        Minimum hinge radius in residues.

    Returns
    -------
    float
        Mean hinge occupation ratio across modes.

    Historical notes
    ----------------
    Discovered in D111 (Multi-Mode Hinge Analysis).  This is the
    observable that cracked the T4 lysozyme problem: T4 has
    hinge_R = 1.091 (enzyme — modes concentrate at the catalytic
    cleft) vs AdK hinge_R = 0.952 (allosteric — mode 1 exhausts
    the hinge).
    """
    N = len(domain_labels)
    hinge_radius = max(min_hinge_radius, int(N * hinge_radius_frac))

    # Find domain boundaries
    boundaries = [i for i in range(N - 1)
                  if domain_labels[i] != domain_labels[i + 1]]
    if not boundaries:
        return 1.0  # no boundary → neutral

    # Near-hinge residues
    near_hinge = set()
    for b in boundaries:
        for r in range(max(0, b - hinge_radius),
                       min(N, b + hinge_radius + 1)):
            near_hinge.add(r)
    far_hinge = set(range(N)) - near_hinge

    if not near_hinge or not far_hinge:
        return 1.0

    near_idx = sorted(near_hinge)
    far_idx = sorted(far_hinge)

    ratios = []
    for k in modes:
        if k >= eigenvectors.shape[1]:
            continue
        v2 = eigenvectors[:, k] ** 2
        h_near = np.mean(v2[near_idx])
        h_far = np.mean(v2[far_idx])
        ratios.append(h_near / (h_far + 1e-12))

    return float(np.mean(ratios)) if ratios else 1.0


def domain_stiffness_asymmetry(
        contacts: dict,
        domain_labels: np.ndarray,
) -> float:
    """Asymmetry in contact density between the two Fiedler domains.

    For each domain *d*, compute:

    .. math::

        \\rho_d = \\frac{\\text{intra-domain contacts}}
                       {\\text{domain size}}

    Return the normalised difference:

    .. math::

        \\text{asym} = \\frac{|\\rho_0 - \\rho_1|}
                            {\\max(\\rho_0, \\rho_1)}

    Physical interpretation:

    - **High asymmetry** — one domain is much stiffer than the
      other (typical of some enzymes with a rigid core + flexible
      lid).
    - **Low asymmetry** — both domains are equally packed.
      T4 lysozyme has the lowest asymmetry (0.002) of any
      benchmark protein.

    Parameters
    ----------
    contacts : dict
        ``{(i, j): distance}`` contact map.
    domain_labels : (N,) ndarray
        Binary domain labels.

    Returns
    -------
    float
        Normalised stiffness asymmetry in [0, 1].

    Historical notes
    ----------------
    Discovered in D111 (Multi-Mode Hinge Analysis).
    """
    dom0 = int(np.sum(domain_labels == 0))
    dom1 = int(np.sum(domain_labels == 1))

    contacts_dom0 = sum(
        1 for (i, j) in contacts
        if domain_labels[i] == 0 and domain_labels[j] == 0)
    contacts_dom1 = sum(
        1 for (i, j) in contacts
        if domain_labels[i] == 1 and domain_labels[j] == 1)

    density0 = contacts_dom0 / max(dom0, 1)
    density1 = contacts_dom1 / max(dom1, 1)

    dmax = max(density0, density1)
    if dmax < 1e-10:
        return 0.0
    return abs(density0 - density1) / dmax
