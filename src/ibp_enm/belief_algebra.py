"""Hamming(7,4) bridge protocol for syndrome-corrected vote fusion.

Implements the 4-step bridge protocol from D148:

  Step 1 (DIFF):      For each archetype, compare each instrument's vote
                       against the mean to produce a binary support vector.
  Step 2 (SYNDROME):   Hamming parity check identifies if the support
                       pattern is structurally consistent with Fano geometry.
  Step 3 (LOCATE):     Non-zero syndrome → single-error correction
                       identifies the inconsistent instrument position.
  Step 4 (SHIFT):      Dampen the erring instrument's vote by the spectral
                       factor 1/√2 (strong→weak demotion) before computing
                       Fano-line coherence scores.

Free parameters: **0** (damping = 1/√2 from D148 spectral structure).

Valid Hamming(7,4) codewords have weights {0, 3, 4, 7}:
  - Weight 3: the 7 Fano lines (3 instruments support an archetype)
  - Weight 4: the 7 Fano line complements
  - Weight 0/7: unanimous rejection/support

Any other support pattern has exactly one instrument in error,
which the syndrome uniquely identifies.

See Also
--------
D148 : Hamming bridge specification (spectral split, syndrome tables).
D149 : AlgebraicFickBalancer using the bridge pathway.
D152b : First real-data validation (bridge_mean ≈ 0.25).
D153 : This implementation.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple

from .algebra import FANO_LINES

__all__ = [
    "HAMMING_H",
    "SYNDROME_RETENTION",
    "compute_syndrome",
    "decode_error_position",
    "HammingBridge",
]

# ── Hamming(7,4) parity check matrix ──────────────────────────────
#
# Adapted from the standard Hamming(7,4) H matrix by applying the
# permutation σ = [0,1,3,4,5,2,6] that aligns the code's Fano plane
# with FANO_LINES from algebra.py.
#
# With this adapted H, the 7 Fano lines are exactly the weight-3
# codewords, and the 7 Fano-line complements are the weight-4 ones.
#
# Column syndrome values:  col 0→1, 1→2, 2→6, 3→3, 4→4, 5→5, 6→7

HAMMING_H = np.array([
    [1, 0, 0, 1, 0, 1, 1],   # parity check p₀
    [0, 1, 1, 1, 0, 0, 1],   # parity check p₁
    [0, 0, 1, 0, 1, 1, 1],   # parity check p₂
], dtype=int)

# ── Spectral damping factor ──────────────────────────────────────
#
# From D148 spectral split {2√2⁴, 2⁴, 0⁸}:
#   strong_coupling = 2√2,  weak_coupling = 2
#
# An instrument flagged by the syndrome is demoted from strong
# to weak coupling:
#   retention = weak / strong = 2 / (2√2) = 1/√2 ≈ 0.7071
#
# This is the algebraically determined damping — 0 free parameters.

_SQRT2 = np.sqrt(2)
SYNDROME_RETENTION: float = 1.0 / _SQRT2  # ≈ 0.7071


# ── Core syndrome functions ──────────────────────────────────────

def compute_syndrome(support: np.ndarray) -> np.ndarray:
    """Hamming(7,4) syndrome for a 7-bit instrument support vector.

    Parameters
    ----------
    support : ndarray of shape (7,)
        Binary vector: 1 if instrument supports the archetype, 0 otherwise.

    Returns
    -------
    ndarray of shape (3,)
        Syndrome bits.  All zeros iff ``support`` is a valid codeword.
    """
    return (HAMMING_H @ support.astype(int)) % 2


def decode_error_position(syndrome: np.ndarray) -> Optional[int]:
    """Decode 3-bit syndrome to 0-indexed error position, or None.

    Maps the syndrome to the column of :data:`HAMMING_H` it matches.
    Returns ``None`` when the syndrome is all zeros (valid codeword).

    Parameters
    ----------
    syndrome : ndarray of shape (3,)

    Returns
    -------
    int or None
        0-indexed instrument position if error detected, else ``None``.
    """
    s = int(syndrome[0]) + 2 * int(syndrome[1]) + 4 * int(syndrome[2])
    if s == 0:
        return None
    # Syndrome value s ∈ {1..7} → find the column whose binary value == s
    for col_idx in range(7):
        col_val = (int(HAMMING_H[0, col_idx])
                   + 2 * int(HAMMING_H[1, col_idx])
                   + 4 * int(HAMMING_H[2, col_idx]))
        if col_val == s:
            return col_idx
    return None  # unreachable for s in 1..7


# ── HammingBridge class ──────────────────────────────────────────

class HammingBridge:
    """Syndrome-corrected Fano bridge for instrument vote fusion.

    Wraps the D148 4-step bridge protocol into a callable that
    takes raw instrument votes and returns Fano-coherent bridge
    scores with syndrome-based threshold shifts applied.

    The bridge has **0 free parameters**:

    - Binarisation threshold = per-archetype vote mean (data-determined)
    - Damping factor = 1/√2 (spectral structure, see D148 Exp 3)
    - Fano-line bonus = 0.3 (inherited from AlgebraicFickBalancer)

    Usage
    -----
    >>> bridge = HammingBridge()
    >>> scores = bridge.bridge_scores(carver_votes, all_archs)
    >>> diag = bridge.diagnose(carver_votes, all_archs)
    """

    # Fano-line coherence bonus per line (same as _compute_fano_bridge)
    FANO_BONUS = 0.3

    def __init__(self) -> None:
        # Pre-compute syndrome → position lookup table
        self._syn_to_pos: Dict[Tuple[int, int, int], Optional[int]] = {
            (0, 0, 0): None,  # valid codeword — no correction
        }
        for col_idx in range(7):
            key = tuple(int(x) for x in HAMMING_H[:, col_idx])
            self._syn_to_pos[key] = col_idx

    # ── Step 1–3: threshold_shift ────────────────────────────────

    def threshold_shift(
        self,
        carver_votes: List[Dict[str, float]],
        all_archs: List[str],
    ) -> Dict[str, np.ndarray]:
        """Per-archetype retention factors for each instrument.

        Performs steps 1–3 of the bridge protocol for every archetype.

        Parameters
        ----------
        carver_votes : list of 7 dicts
            Each dict maps archetype name → vote probability.
        all_archs : list of str
            Archetype names to evaluate.

        Returns
        -------
        dict mapping archetype → ndarray of shape (7,)
            Multiplicative retention factors in (0, 1].
            **1.0** = instrument is Hamming-consistent (no correction).
            **1/√2 ≈ 0.707** = instrument flagged by syndrome (dampened).
        """
        n = min(len(carver_votes), 7)
        shifts: Dict[str, np.ndarray] = {}

        for arch in all_archs:
            retention = np.ones(7)
            if n < 7:
                shifts[arch] = retention
                continue

            # Step 1: binary support vector (above-mean = 1)
            raw = np.array(
                [carver_votes[i].get(arch, 0.0) for i in range(7)])
            mean_v = float(np.mean(raw))
            support = (raw > mean_v).astype(int)

            # Step 2: syndrome
            syn = tuple(
                int(x) for x in (HAMMING_H @ support) % 2)

            # Step 3: decode error position
            err_pos = self._syn_to_pos.get(syn)
            if err_pos is not None and support[err_pos] == 1:
                # Spurious supporter: instrument votes above mean
                # but Fano structure says it shouldn't.
                # Dampen from strong→weak coupling (× 1/√2).
                retention[err_pos] = SYNDROME_RETENTION
            # If support[err_pos] == 0, the syndrome suggests a
            # *missing* supporter — we do NOT boost (conservative).

            shifts[arch] = retention

        return shifts

    # ── Step 4: bridge_scores ────────────────────────────────────

    def bridge_scores(
        self,
        carver_votes: List[Dict[str, float]],
        all_archs: List[str],
    ) -> Dict[str, float]:
        """Syndrome-corrected Fano bridge scores per archetype.

        Full 4-step protocol:

        1. Binarise each archetype's support pattern.
        2. Compute Hamming syndrome.
        3. If nonzero, identify the erring instrument.
        4. Dampen erring instrument by 1/√2, then measure
           Fano-line coherence.

        Returns normalised scores summing to ~1.
        """
        n = min(len(carver_votes), 7)
        shifts = self.threshold_shift(carver_votes, all_archs)

        bridge: Dict[str, float] = {}
        for arch in all_archs:
            retention = shifts[arch]

            # Step 4: apply retention and compute bridge score
            corrected = [
                carver_votes[i].get(arch, 0.0) * retention[i]
                for i in range(n)
            ]

            # Top 3 corrected voters
            indexed = sorted(
                [(corrected[i], i) for i in range(n)],
                reverse=True,
            )
            top_voters = [idx for val, idx in indexed[:3] if val > 0.05]

            # Fano-line coherence
            fano_links = 0
            for line in FANO_LINES:
                members_in_top = sum(1 for p in line if p in top_voters)
                if members_in_top >= 2:
                    fano_links += 1

            top_mean = float(np.mean([v for v, _ in indexed[:3]]))
            bridge[arch] = top_mean * (1.0 + self.FANO_BONUS * fano_links)

        # Normalise
        total = sum(bridge.values())
        if total > 1e-10:
            bridge = {k: v / total for k, v in bridge.items()}
        return bridge

    # ── Diagnostics ──────────────────────────────────────────────

    def diagnose(
        self,
        carver_votes: List[Dict[str, float]],
        all_archs: List[str],
    ) -> Dict:
        """Full diagnostic report for the bridge protocol.

        Returns per-archetype syndrome analysis including which
        instrument (if any) was flagged, support weights, and
        aggregate statistics.

        Parameters
        ----------
        carver_votes : list of 7 dicts
        all_archs : list of str

        Returns
        -------
        dict with keys:
            - ``per_archetype``: per-arch syndrome details
            - ``flagged_counts``: how often each instrument was flagged
            - ``valid_fraction``: fraction of archetypes with zero syndrome
            - ``most_flagged_instrument``: the most frequently flagged index
        """
        n = min(len(carver_votes), 7)
        per_arch: Dict[str, Dict] = {}
        flagged_counts = np.zeros(7, dtype=int)
        total_syndromes = 0
        zero_syndromes = 0

        for arch in all_archs:
            total_syndromes += 1
            if n < 7:
                per_arch[arch] = {
                    "support": [0] * 7,
                    "syndrome": (0, 0, 0),
                    "error_position": None,
                    "support_weight": 0,
                }
                zero_syndromes += 1
                continue

            raw = np.array(
                [carver_votes[i].get(arch, 0.0) for i in range(7)])
            mean_v = float(np.mean(raw))
            support = (raw > mean_v).astype(int)
            syn = tuple(
                int(x) for x in (HAMMING_H @ support) % 2)
            err_pos = self._syn_to_pos.get(syn)

            if err_pos is None:
                zero_syndromes += 1
            else:
                flagged_counts[err_pos] += 1

            per_arch[arch] = {
                "support": support.tolist(),
                "syndrome": syn,
                "error_position": err_pos,
                "support_weight": int(np.sum(support)),
            }

        return {
            "per_archetype": per_arch,
            "flagged_counts": flagged_counts.tolist(),
            "valid_fraction": zero_syndromes / max(total_syndromes, 1),
            "most_flagged_instrument": int(np.argmax(flagged_counts)),
        }
