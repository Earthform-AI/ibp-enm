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
    "ZDPairSelector",
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
        # D157: ZD pair routing selector
        self._zd_selector = ZDPairSelector()

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

            # D157: per-archetype support and routing activation
            raw = np.array(
                [carver_votes[i].get(arch, 0.0) for i in range(min(n, 7))])
            if n >= 7:
                mean_v = float(np.mean(raw))
                support = (raw > mean_v).astype(int)
            else:
                support = np.zeros(7, dtype=int)
            line_act = self._zd_selector.fano_activation(support)

            # Routing-weighted Fano-line coherence (D157)
            # Each Fano line's contribution is weighted by its routing
            # activation.  When the support pattern fully activates a
            # line (≥2 supporters), activation = 1.0 and the weight
            # is identical to the original fano_links count.  When
            # syndrome correction moves a supporter out of the top
            # voters, the activation provides a geometrically-grounded
            # discount factor (1/√2 = contained-channel purity).
            fano_score = 0.0
            for i, line in enumerate(FANO_LINES):
                members_in_top = sum(1 for p in line if p in top_voters)
                if members_in_top >= 2:
                    fano_score += line_act[i]

            top_mean = float(np.mean([v for v, _ in indexed[:3]]))
            bridge[arch] = top_mean * (1.0 + self.FANO_BONUS * fano_score)

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

        # D157: aggregate routing diagnostics
        all_route_scores: List[float] = []
        routing_per_arch: Dict[str, Dict] = {}
        n_diag = min(len(carver_votes), 7)
        for arch in all_archs:
            if n_diag < 7:
                routing_per_arch[arch] = self._zd_selector.diagnose_routing(
                    np.zeros(7, dtype=int))
                all_route_scores.append(0.0)
                continue
            raw = np.array(
                [carver_votes[i].get(arch, 0.0) for i in range(7)])
            mean_v = float(np.mean(raw))
            support = (raw > mean_v).astype(int)
            rd = self._zd_selector.diagnose_routing(support)
            routing_per_arch[arch] = rd
            all_route_scores.append(rd['route_score'])

        return {
            "per_archetype": per_arch,
            "flagged_counts": flagged_counts.tolist(),
            "valid_fraction": zero_syndromes / max(total_syndromes, 1),
            "most_flagged_instrument": int(np.argmax(flagged_counts)),
            "routing": routing_per_arch,
            "mean_route_score": float(np.mean(all_route_scores))
                if all_route_scores else 0.0,
        }


# ── ZD Pair Selection (D157) ────────────────────────────────────


class ZDPairSelector:
    """ZD pair → Fano line routing via contained channels (D157).

    D157 discovered that each ZD-pair kernel has exactly 4
    "contained" quaternionic subalgebras — subalgebras with one
    dimension lying inside the kernel.  These contained channels
    route through 21 cross-half subalgebras (spanning both e₁-e₇
    and e₈-e₁₅), providing **72 routes** to each of the 7 Fano
    lines, perfectly uniform across all 42 kernel equivalence
    classes.

    The contained direction purity is exactly 1/√2, confirming the
    geometric origin of :data:`SYNDROME_RETENTION`.

    This class provides per-archetype routing metrics based on how
    well a vote pattern's instrument support activates the Fano-line
    routing structure.

    **0 free parameters** — all constants from sedenion algebra.

    D157 Structural Constants
    -------------------------
    - 42 kernel classes × 4 contained subs = 168 edges
    - 21 cross-half subs (8 kernels each) carry all edges
    - 14 pure-half subs (7 low + 7 high) have 0 contained edges
    - 72 routes per Fano line (uniform across all kernel classes)
    - Contained purity = 1/√2 ≈ 0.7071

    Key result: **every kernel can route to every Fano line**.
    ZD pair selection is therefore unconstrained — the algebra is
    perfectly symmetric.  The routing score quantifies how well a
    *vote pattern* engages the Fano backbone, not which ZD pair to
    pick (they are all equivalent).

    See Also
    --------
    D157 : Contained-channel routing experiment (5/5 confirmed).
    D156 : Three angle spectra ({0,π/2,π/2,π/2} = contained).
    HammingBridge : Syndrome-corrected Fano bridge.
    """

    # ── D157 structural constants (0 free parameters) ────────────
    KERNEL_CLASSES: int = 42
    CONTAINED_PER_KERNEL: int = 4
    CROSS_HALF_SUBS: int = 21
    PURE_HALF_SUBS: int = 14       # 7 pure-low + 7 pure-high
    ROUTES_PER_LINE: int = 72
    TOTAL_CONTAINED: int = 168
    # Contained direction purity = 1/√2 = SYNDROME_RETENTION
    CONTAINED_PURITY: float = SYNDROME_RETENTION

    def __init__(self) -> None:
        self._line_sets = [set(line) for line in FANO_LINES]

    # ── Per-line activation ──────────────────────────────────────

    def fano_activation(self, support: np.ndarray) -> np.ndarray:
        """Per-Fano-line activation from instrument support vector.

        D157 guarantees every activated line has contained routes from
        all 42 kernel classes (72 routes).  Activation levels:

        - **1.0** : ≥2 instruments on this line support the archetype
          (strong — full Fano triple engagement).
        - **1/√2** : exactly 1 instrument supports (weak — partial
          routing, damped to contained-channel purity).
        - **0.0** : no supporting instrument (no routing).

        Parameters
        ----------
        support : ndarray of shape (7,)
            Binary instrument support vector (1 = above-mean vote).

        Returns
        -------
        ndarray of shape (7,)
            Per-line activation in [0, 1].
        """
        active = set(int(i) for i in np.where(support > 0)[0])
        out = np.zeros(7)
        for i, ls in enumerate(self._line_sets):
            k = len(active & ls)
            if k >= 2:
                out[i] = 1.0
            elif k == 1:
                out[i] = self.CONTAINED_PURITY
        return out

    # ── Scalar summaries ─────────────────────────────────────────

    def route_score(self, support: np.ndarray) -> float:
        """Mean Fano-line activation ∈ [0, 1].

        Captures how much of the contained-channel backbone the
        support pattern engages.

        Parameters
        ----------
        support : ndarray of shape (7,)
            Binary instrument support vector.
        """
        return float(np.mean(self.fano_activation(support)))

    def select_lines(self, support: np.ndarray) -> List[int]:
        """Fano lines with strong routing (≥2 supporters).

        Each returned line has 72 contained routes from all 42
        kernel classes (D157 uniformity guarantee).

        Parameters
        ----------
        support : ndarray of shape (7,)

        Returns
        -------
        list of int
            Indices of strongly-activated Fano lines.
        """
        act = self.fano_activation(support)
        return [i for i in range(7) if act[i] >= 1.0]

    # ── Diagnostics ──────────────────────────────────────────────

    def diagnose_routing(self, support: np.ndarray) -> Dict:
        """Full routing diagnostics for a support pattern.

        Returns
        -------
        dict with keys:
            fano_activation : list of 7 floats
            strong_lines : list of int (activation ≥ 1.0)
            weak_lines : list of int (0 < activation < 1.0)
            n_strong : int
            n_weak : int
            route_score : float  (mean activation)
            total_strong_routes : int (n_strong × 72)
            support_weight : int (sum of support bits)
        """
        act = self.fano_activation(support)
        strong = [i for i in range(7) if act[i] >= 1.0]
        weak = [i for i in range(7) if 0 < act[i] < 1.0]
        return {
            'fano_activation': act.tolist(),
            'strong_lines': strong,
            'weak_lines': weak,
            'n_strong': len(strong),
            'n_weak': len(weak),
            'route_score': float(np.mean(act)),
            'total_strong_routes': len(strong) * self.ROUTES_PER_LINE,
            'support_weight': int(np.sum(support)),
        }
