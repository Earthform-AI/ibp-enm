"""IBP-ENM core analyzer.

Self-contained (numpy + scipy only). Three entry points:

  analyze(coords, b_factors) → IBPResult
    Single-state structural snapshot:
    - B-factor predictions from 3 shadow perspectives + consensus
    - Fiedler vector → unsupervised domain decomposition
    - Hinge scores from Fiedler sign-change boundaries
    - Stabilizer profiles (per-residue structural role scores)
    - Debris recycling: domain-localised higher-mode analysis (D84)

  compare(coords_a, coords_b) → ConformationalComparison
    Two-state conformational pipeline (D84/D86):
    - Spectral time: eigenvalue spectrum distances
      (log-spectral ρ=0.71 p=0.0003, Fiedler gap Δ, Wasserstein, JS)
    - Domain clocks: per-domain spectral gap changes
    - Debris flow, hinge overlap between states

  probe(coords) → DisturbanceResponse
    Perturbation-response fingerprint (D84/D87 Disturber Language):
    - Sweeps an alphabet of perturbations {mode(k), cutoff(r), perspective(p)}
    - Per-residue cutoff entropy = structural ambiguity score
    - Mode-cutoff interaction matrix = scale robustness fingerprint
    - Mode robustness vs cutoff robustness statistics
    - Perspective disagreement = multi-parser information

  listen(response_a, response_b) → DisturbanceMessage
    Cross-state disturber message:
    - Difference in perturbation responses between two conformational states
    - Conformational hotspots: residues whose ambiguity changed
    - Interaction delta: which mode/cutoff features are state-dependent
    - The disturber reads the system's response and identifies
      what the conformational change "said"

  probe_distribution(coords) → DistributionResponse
    Multi-perturbation distribution network (D89):
    - Fires 4 perturbation types through the protein simultaneously:
      stiffness (spring constants), topology (contact deletion),
      geometry (coordinate noise), mass (local stabilizer doubling)
    - Per-residue response profile: which channels each residue
      responds to most → structural typing
    - Channel independence: do different perturbation types see
      different structure? (high = rich information content)

  shadow_probe(coords) → ShadowResponse
    Shadow perturbation — perturbing the disturber itself (D89):
    - Adds controlled noise to each stabilizer profile
    - Measures per-residue parser fragility: does the partition
      change when the parser is slightly wrong?
    - Fragile residues sit at parsing boundaries — their structural
      role depends on which mathematical lens you use
    - Tests time-independence: shadow fragility should be orthogonal
      to spectral time (conformational magnitude)

  shadow_probe_sweep(coords) → ShadowSweepResult
    Shadow ε-sweep — fragility landscape and phase transition (D90):
    - Sweeps perturbation magnitude ε from ~0 to ~1.0
    - Builds per-residue fragility landscape f_i(ε)
    - Finds critical ε where partition shatters (≥50% fragile)
    - Tracks spectral gap λ₂/λ₃ response to parser noise
    - Gap sensitivity d(gap)/dε → structural rigidity scalar
    - Critical ε is time-independent: same value regardless of
      which conformational state the protein is in

  meta_predict(coords) → MetaPrediction
    Multi-channel fusion — shadow casts + consensus voting (D91):
    - Runs all five channels: shadow, debris, language, distribution,
      spectral (boundary proximity)
    - Normalises each to [0,1], then per-residue voting:
      how many channels flag each residue as ambiguous?
    - Shadow cast: projects shadow's prediction into every other
      channel (debris, language, distribution, spectral) and
      measures cross-channel agreement
    - Consensus classes: stable (≤1 vote), mixed (2-3), ambiguous (≥4)
    - Channel agreement matrix: pairwise Spearman between all 5 signals
"""

import time
import numpy as np
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform
from scipy.stats import wasserstein_distance as _wasserstein


@dataclass
class DomainDetail:
    """Per-domain structural analysis from debris recycling.

    Each Fiedler-identified domain gets its own sub-Laplacian analysis,
    yielding internal spectral structure invisible to the global analysis.
    """
    domain_id: int = 0
    residue_indices: Optional[np.ndarray] = field(default=None, repr=False)
    n_residues: int = 0
    n_contacts: int = 0

    # Sub-Laplacian spectral data
    eigenvalues: Optional[np.ndarray] = field(default=None, repr=False)
    spectral_gap: float = 0.0          # λ₂/λ₃ of the sub-Laplacian
    fiedler: Optional[np.ndarray] = field(default=None, repr=False)

    # Sub-domain structure within this domain
    subdomain_labels: Optional[np.ndarray] = field(default=None, repr=False)
    n_subdomains: int = 1

    # Recycled hinges: within-domain sign changes from higher modes
    recycled_hinges: Optional[List[int]] = field(default=None, repr=False)
    recycled_hinge_scores: Optional[np.ndarray] = field(default=None, repr=False)
    n_recycled_sign_changes: int = 0      # total within-domain SCs
    n_discarded_cross_domain: int = 0     # cross-domain SCs (true debris)


@dataclass
class IBPResult:
    """Results from IBP-ENM analysis of a single protein."""
    n_residues: int = 0
    n_contacts: int = 0

    # B-factor correlations (vs experimental, if provided)
    rho_laplacian: float = 0.0
    rho_continuous: float = 0.0
    rho_uniform: float = 0.0
    rho_consensus: float = 0.0
    rho_best: float = 0.0
    best_method: str = ""

    # Predicted B-factors (raw arrays)
    b_laplacian: Optional[np.ndarray] = field(default=None, repr=False)
    b_continuous: Optional[np.ndarray] = field(default=None, repr=False)
    b_uniform: Optional[np.ndarray] = field(default=None, repr=False)

    # Stabilizer profiles (per-residue structural role)
    stabilizer_laplacian: Optional[np.ndarray] = field(default=None, repr=False)
    stabilizer_continuous: Optional[np.ndarray] = field(default=None, repr=False)

    # Domain decomposition
    fiedler_vector: Optional[np.ndarray] = field(default=None, repr=False)
    domain_labels: Optional[np.ndarray] = field(default=None, repr=False)
    n_domains: int = 0
    spectral_gap: Optional[float] = None  # λ₂/λ₃ — confidence for 2-domain split

    # k-way spectral clustering (uses top-k Laplacian eigenvectors)
    spectral_labels: Optional[np.ndarray] = field(default=None, repr=False)
    spectral_k: int = 0  # number of clusters chosen
    spectral_eigengaps: Optional[np.ndarray] = field(default=None, repr=False)

    # Hinge detection
    hinge_scores: Optional[np.ndarray] = field(default=None, repr=False)
    hinge_residues: Optional[List[int]] = field(default=None, repr=False)

    # Debris recycling (D84): domain-localised higher-mode analysis
    domain_details: Optional[List[DomainDetail]] = field(default=None, repr=False)
    subdomain_labels: Optional[np.ndarray] = field(default=None, repr=False)
    n_subdomains_total: int = 0
    recycled_hinges: Optional[List[int]] = field(default=None, repr=False)
    recycled_hinge_scores: Optional[np.ndarray] = field(default=None, repr=False)
    debris_recycled_fraction: float = 0.0  # within-domain / total SCs

    # Spectral data
    laplacian_eigenvalues: Optional[np.ndarray] = field(default=None, repr=False)
    laplacian_eigenvectors: Optional[np.ndarray] = field(default=None, repr=False)

    time_total: float = 0.0


@dataclass
class DomainClock:
    """Per-domain spectral gap change between two conformational states."""
    domain_id: int = 0
    n_residues: int = 0
    gap_a: float = 0.0        # spectral gap in state A
    gap_b: float = 0.0        # spectral gap in state B
    delta_gap: float = 0.0    # gap_b - gap_a (signed change)
    eigenvalues_a: Optional[np.ndarray] = field(default=None, repr=False)
    eigenvalues_b: Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class ConformationalComparison:
    """Results from comparing two conformational states (D84/D86).

    The compare() pipeline aligns domain partitions from state A onto
    state B, then extracts spectral time metrics, domain clocks, and
    debris flow between the two states.
    """
    # Global spectral time metrics
    rmsd: float = 0.0
    fiedler_gap_delta: float = 0.0     # |λ₂/λ₃ change| — best single metric (ρ≈0.63)
    spectral_dist_log: float = 0.0     # log-spectral distance (ρ≈0.55)
    spectral_dist_norm: float = 0.0    # normalised L2 spectral distance
    spectral_dist_weighted: float = 0.0  # 1/k-weighted spectral distance
    wasserstein: float = 0.0           # Wasserstein between eigenvalue distributions
    spectral_JS: float = 0.0           # symmetrised KL divergence
    entropy_delta: float = 0.0         # |ΔH| of eigenvalue distributions

    # Domain clocks
    domain_clocks: Optional[List[DomainClock]] = field(default=None, repr=False)
    clock_anti_correlated: bool = False  # True if domain gaps move inversely
    clock_spearman: float = 0.0          # correlation between domain gap deltas

    # Debris flow between states
    debris_fraction_a: float = 0.0
    debris_fraction_b: float = 0.0
    debris_delta: float = 0.0  # change in within-domain fraction

    # Per-residue change maps
    domain_label_changes: Optional[np.ndarray] = field(default=None, repr=False)
    hinge_overlap: float = 0.0  # Jaccard of hinge residues between states

    # Disturber language (D88): probe/listen integration
    disturber_message: Optional['DisturbanceMessage'] = field(default=None, repr=False)
    response_a: Optional['DisturbanceResponse'] = field(default=None, repr=False)
    response_b: Optional['DisturbanceResponse'] = field(default=None, repr=False)
    hotspot_weighted_log_spectral: float = 0.0  # log-spectral re-weighted by hotspot sensitivity
    functional_overlap: Optional['FunctionalSiteOverlap'] = field(default=None, repr=False)

    # Reference to source results
    result_a: Optional[IBPResult] = field(default=None, repr=False)
    result_b: Optional[IBPResult] = field(default=None, repr=False)


@dataclass
class DisturbanceResponse:
    """A protein's full response to a perturbation sweep.

    The "sentence" the protein speaks when the disturber probes it
    across modes, cutoffs, and stabilizer perspectives.  Every field
    is a parsed aspect of that sentence.

    Alphabet:
      mode(k)        — which eigenvector drives the partition
      cutoff(r)      — spatial scale of contacts (Å)
      perspective(p)  — which stabilizer parses the signal

    Semantics:
      Modes   = semantic content ("what structural question is asked")
      Cutoffs = inflection       ("at what spatial scale")
      Perspectives = parsers     ("who is listening")
    """
    n_residues: int = 0

    # Per-residue structural ambiguity: entropy of partition labels
    # across the cutoff sweep.  High entropy = the residue sits at a
    # structural boundary that is scale-dependent (hinge/loop).
    cutoff_entropy: Optional[np.ndarray] = field(default=None, repr=False)
    ambiguous_residues: Optional[List[int]] = field(default=None, repr=False)
    fraction_ambiguous: float = 0.0

    # Scale-dependent domain counts: cutoff → spectral k
    domain_counts: Optional[Dict[float, int]] = field(default=None, repr=False)

    # Mode-cutoff interaction: (mode_k, cutoff_r) → partition labels
    # Stored as agreement scores between all pairs of (mode, cutoff)
    interaction_agreement: Optional[Dict[str, float]] = field(default=None, repr=False)

    # Robustness statistics (D84 findings: mode >> cutoff)
    mode_robustness: float = 0.0     # same-mode / different-cutoff agreement
    cutoff_robustness: float = 0.0   # same-cutoff / different-mode agreement

    # Per-perspective partition labels (how each stabilizer "hears")
    perspective_labels: Optional[Dict[str, np.ndarray]] = field(default=None, repr=False)
    perspective_agreement: Optional[Dict[str, float]] = field(default=None, repr=False)

    # The canonical analyze() result at default settings
    canonical_result: Optional[IBPResult] = field(default=None, repr=False)


@dataclass
class DisturbanceMessage:
    """The conformational message: what the system 'said' changed.

    Compares two DisturbanceResponses (typically open vs closed states)
    and extracts the difference.  The disturber listens to both
    sentences and reports what changed — which residues shifted
    ambiguity, which features lost/gained robustness, which
    perspectives now disagree.

    This IS the conformational signal, expressed in the disturber's
    own language.
    """
    # Per-residue change in structural ambiguity
    entropy_delta: Optional[np.ndarray] = field(default=None, repr=False)
    hotspot_residues: Optional[List[int]] = field(default=None, repr=False)
    n_hotspots: int = 0

    # Change in robustness
    mode_robustness_delta: float = 0.0
    cutoff_robustness_delta: float = 0.0

    # Change in scale-dependent domain counts
    domain_count_deltas: Optional[Dict[float, int]] = field(default=None, repr=False)

    # Interaction delta: which (mode, cutoff) pairs changed partition
    interaction_delta: Optional[Dict[str, float]] = field(default=None, repr=False)

    # Perspective shift: which parsers now disagree more/less
    perspective_shift: Optional[Dict[str, float]] = field(default=None, repr=False)

    # Summary: how different are the two sentences?
    total_information: float = 0.0  # sum of |entropy_delta|
    message_strength: float = 0.0   # fraction of residues that are hotspots


@dataclass
class FunctionalSiteOverlap:
    """Enrichment of disturber hotspots at known functional residues.

    Measures whether the residues whose cutoff entropy changes most
    between conformational states (hotspots) are disproportionately
    located at known functional sites (catalytic, binding, allosteric).

    The enrichment ratio (hotspot_rate_at_func / hotspot_rate_overall)
    tells us whether the conformational message is "about" function:
      enrichment > 1 → hotspots cluster at functional sites
      enrichment ≈ 1 → hotspots are uniformly distributed
      enrichment < 1 → hotspots avoid functional sites
    """
    n_residues: int = 0
    n_functional: int = 0            # known functional residues
    n_hotspots: int = 0              # from DisturbanceMessage
    n_overlap: int = 0               # hotspots ∩ functional
    overlap_residues: Optional[List[int]] = field(default=None, repr=False)

    # Rates
    hotspot_rate: float = 0.0        # n_hotspots / n_residues
    functional_rate: float = 0.0     # n_functional / n_residues
    overlap_rate: float = 0.0        # n_overlap / n_functional (sensitivity)
    precision: float = 0.0           # n_overlap / n_hotspots

    # Enrichment
    enrichment: float = 0.0          # overlap_rate / hotspot_rate
    p_value: float = 1.0             # hypergeometric test p-value

    # Per-residue entropy delta at functional sites vs elsewhere
    mean_abs_delta_func: float = 0.0
    mean_abs_delta_other: float = 0.0
    delta_ratio: float = 0.0         # func / other — do functional sites change more?


@dataclass
class DistributionResponse:
    """Per-residue response profile across multiple perturbation types.

    The disturber channel as a DISTRIBUTION NETWORK: fire different
    types of perturbation into the system and record how each residue
    responds to each type.  The per-residue response vector across
    perturbation types is a richer fingerprint than cutoff entropy alone.

    Perturbation types:
      stiffness  — scale spring constants (s_min, s_max) by factors
      topology   — randomly delete fraction of contacts
      geometry   — add Gaussian noise to coordinates
      mass       — weight specific residues (simulated ligand/binding)
    """
    n_residues: int = 0

    # Per-residue sensitivity to each perturbation type:
    # How much does residue i's domain assignment change when
    # perturbation type t is applied? (0 = robust, 1 = maximally sensitive)
    stiffness_sensitivity: Optional[np.ndarray] = field(default=None, repr=False)
    topology_sensitivity: Optional[np.ndarray] = field(default=None, repr=False)
    geometry_sensitivity: Optional[np.ndarray] = field(default=None, repr=False)
    mass_sensitivity: Optional[np.ndarray] = field(default=None, repr=False)

    # Per-residue response profile: (N, n_perturbation_types) matrix
    response_matrix: Optional[np.ndarray] = field(default=None, repr=False)

    # Which perturbation types each residue is most sensitive to
    dominant_channel: Optional[np.ndarray] = field(default=None, repr=False)  # (N,) int

    # Channel agreement: do different perturbation types see the same structure?
    channel_agreement: Optional[Dict[str, float]] = field(default=None, repr=False)

    # Summary statistics
    mean_sensitivity: Optional[Dict[str, float]] = field(default=None, repr=False)
    channel_independence: float = 0.0  # mean pairwise disagreement across channels

    # Reference partition
    canonical_result: Optional['IBPResult'] = field(default=None, repr=False)


@dataclass
class ShadowResponse:
    """The parser's own fragility — what happens when you perturb the disturber.

    Instead of perturbing the PROTEIN and measuring through a fixed
    stabilizer, perturb the STABILIZER ITSELF and measure how the
    partition changes.  This is meta-perturbation: the disturber
    disturbing itself.

    For each stabilizer (Laplacian, continuous, uniform), we add
    controlled noise and measure partition sensitivity.  A residue
    where the partition changes when the parser is perturbed is at a
    PARSING BOUNDARY — the structural assignment there depends on
    which parser you use AND how precisely you specify it.

    Independence from spectral time: shadow sensitivity should capture
    parser-fragility, not conformational magnitude.
    """
    n_residues: int = 0

    # Per-residue shadow sensitivity per stabilizer type:
    # How much does the partition at residue i change when
    # stabilizer s is perturbed? (averaged over perturbation trials)
    shadow_laplacian: Optional[np.ndarray] = field(default=None, repr=False)
    shadow_continuous: Optional[np.ndarray] = field(default=None, repr=False)
    shadow_uniform: Optional[np.ndarray] = field(default=None, repr=False)

    # Composite shadow sensitivity (mean across stabilizers)
    shadow_composite: Optional[np.ndarray] = field(default=None, repr=False)

    # Parser-fragile residues: high shadow sensitivity across stabilizers
    fragile_residues: Optional[List[int]] = field(default=None, repr=False)
    n_fragile: int = 0
    fraction_fragile: float = 0.0

    # Cross-stabilizer shadow agreement: do all parsers become fragile
    # at the same residues?
    shadow_agreement: Optional[Dict[str, float]] = field(default=None, repr=False)

    # Summary
    mean_fragility: float = 0.0  # global mean shadow sensitivity

    # The unperturbed partition for reference
    canonical_result: Optional['IBPResult'] = field(default=None, repr=False)


@dataclass
class ShadowSweepResult:
    """Shadow perturbation sweep across ε values — the fragility landscape.

    Sweeps the parser perturbation magnitude ε from near-zero to large
    and records how partition stability and spectral gap respond.
    The critical ε where the partition first breaks is a structural
    invariant: it characterises how much parser ambiguity the protein's
    topology can absorb before the algebraic identity cracks.

    Key quantities:
      critical_epsilon  — ε at which ≥50% of residues become fragile
      gap_sensitivity   — d(λ₂/λ₃)/dε at ε→0: spectral gap fragility
      profile_at_eps    — per-residue fragility curve f_i(ε)
      cross_state_corr  — Spearman of per-residue fragility profile
                          between states A and B (continuous, not Jaccard)
    """
    n_residues: int = 0

    # Sweep parameters
    epsilon_values: Optional[np.ndarray] = field(default=None, repr=False)
    n_epsilon: int = 0

    # Per-residue fragility at each ε: shape (N, n_epsilon)
    fragility_landscape: Optional[np.ndarray] = field(default=None, repr=False)

    # Per-ε summary: fraction of residues that are fragile (>0.3)
    fraction_fragile_curve: Optional[np.ndarray] = field(default=None, repr=False)

    # Mean fragility at each ε
    mean_fragility_curve: Optional[np.ndarray] = field(default=None, repr=False)

    # Spectral gap (λ₂/λ₃) at each ε: mean across perturbation trials
    gap_curve: Optional[np.ndarray] = field(default=None, repr=False)
    gap_std_curve: Optional[np.ndarray] = field(default=None, repr=False)

    # Critical epsilon: smallest ε where fraction_fragile ≥ threshold
    critical_epsilon: float = float('inf')  # inf = never breaks
    critical_threshold: float = 0.5  # what fraction = "broken"

    # Spectral gap sensitivity: slope of gap_curve at small ε
    gap_sensitivity: float = 0.0  # d(gap)/d(ε) estimated by finite diff
    gap_at_zero: float = 0.0     # unperturbed spectral gap

    # Cross-state fragility correlation (filled by experiment, not method)
    cross_state_rho: float = 0.0  # Spearman of fragility profiles A vs B
    cross_state_p: float = 1.0

    canonical_result: Optional['IBPResult'] = field(default=None, repr=False)


@dataclass
class ShadowCast:
    """Shadow prediction projected into every other channel.

    The shadow predictor says: 'these residues are parser-fragile.'
    Casting checks whether the other channels AGREE:

      shadow → debris:  fragile residues ∩ recycled hinges?
      shadow → language: fragile residues ∩ cutoff-ambiguous?
      shadow → distribution: fragile residues' dominant perturbation type?
      shadow → spectral: fragile residues near domain boundaries?

    High agreement across casts = real structural ambiguity.
    Low agreement = the shadow sees something the others don't.
    """
    n_residues: int = 0

    # Shadow → debris cast
    shadow_debris_rho: float = 0.0     # Spearman(shadow_composite, recycled_hinge_scores)
    shadow_at_hinges: float = 0.0      # mean shadow fragility at recycled hinges
    shadow_at_non_hinges: float = 0.0  # mean shadow fragility elsewhere
    debris_enrichment: float = 0.0     # at_hinges / at_non_hinges

    # Shadow → language cast
    shadow_language_rho: float = 0.0   # Spearman(shadow_composite, cutoff_entropy)
    shadow_at_ambiguous: float = 0.0   # mean shadow fragility at ambiguous residues
    shadow_at_unambiguous: float = 0.0
    language_enrichment: float = 0.0

    # Shadow → distribution cast
    shadow_distribution_rho: float = 0.0  # Spearman(shadow_composite, max_sensitivity)
    fragile_dominant_channel: str = ""  # most common dominant channel among fragile residues
    stable_dominant_channel: str = ""   # most common dominant channel among stable residues
    channel_divergence: float = 0.0     # are fragile/stable residues in different channels?

    # Shadow → spectral cast
    shadow_spectral_rho: float = 0.0   # Spearman(shadow_composite, |fiedler|)
    shadow_at_boundary: float = 0.0    # mean fragility near domain boundary (|fiedler| < median)
    shadow_at_interior: float = 0.0    # mean fragility in domain interior
    boundary_enrichment: float = 0.0

    # Cross-cast agreement: how many channels agree with shadow?
    n_agreeing_channels: int = 0       # channels with |rho| > 0.2
    mean_cast_rho: float = 0.0        # mean |rho| across all casts


@dataclass
class MetaPrediction:
    """Multi-channel fusion: every channel votes on each residue.

    Five channels each produce a per-residue 'ambiguity' signal.
    The meta-predictor normalises each to [0, 1], then counts how
    many channels flag each residue as ambiguous (above median).

    Per-residue consensus_score ∈ {0, 1, 2, 3, 4, 5}:
      0-1 = all channels agree: structurally stable
      2-3 = mixed signal: some channels see ambiguity
      4-5 = all channels agree: structurally ambiguous

    The consensus profile is richer than any single channel because
    it distinguishes GENUINE ambiguity (multi-channel agreement)
    from CHANNEL-SPECIFIC noise (one channel sees it, others don't).
    """
    n_residues: int = 0

    # Per-residue normalised signals from each channel (all [0,1])
    signal_shadow: Optional[np.ndarray] = field(default=None, repr=False)
    signal_debris: Optional[np.ndarray] = field(default=None, repr=False)
    signal_language: Optional[np.ndarray] = field(default=None, repr=False)
    signal_distribution: Optional[np.ndarray] = field(default=None, repr=False)
    signal_spectral: Optional[np.ndarray] = field(default=None, repr=False)

    # Per-residue vote count: how many channels flag as ambiguous
    consensus_votes: Optional[np.ndarray] = field(default=None, repr=False)
    # Mean normalised signal across channels
    consensus_score: Optional[np.ndarray] = field(default=None, repr=False)

    # Classification
    n_stable: int = 0       # consensus_votes <= 1
    n_mixed: int = 0        # consensus_votes in {2, 3}
    n_ambiguous: int = 0    # consensus_votes >= 4
    fraction_ambiguous: float = 0.0

    # Pairwise channel agreement matrix (5x5 Spearman)
    channel_agreement: Optional[Dict[str, float]] = field(default=None, repr=False)
    mean_pairwise_rho: float = 0.0

    # Shadow cast results
    shadow_cast: Optional[ShadowCast] = field(default=None, repr=False)

    # Source data
    ibp_result: Optional[IBPResult] = field(default=None, repr=False)
    shadow_response: Optional[ShadowResponse] = field(default=None, repr=False)
    disturbance_response: Optional[DisturbanceResponse] = field(default=None, repr=False)
    distribution_response: Optional[DistributionResponse] = field(default=None, repr=False)


@dataclass
class SurgeryResult:
    """Results from spectral surgery analysis (D94/D96/D97).

    Spectral surgery removes contacts one at a time and measures the
    spectral gap (λ₂/λ₃) response.  Contacts whose removal maximally
    DROPS the gap are 'locks' — the bonds holding the protein rigid.

    D96 validated: locks concentrate at known functional sites with
    2.6× mean enrichment (7/8 proteins significant).  The category
    hierarchy (allosteric 4.5× > hinge 4.0× > active 1.5× > mutation 1.4×)
    reveals that surgery finds mechanical control points, not just
    chemical hotspots.

    Mechanical vs Chemical classification (D97):
      mechanical — locks at domain boundaries / hinges (control motion)
      chemical   — locks at active / binding sites (control chemistry)
      signalling — locks at allosteric / mutation sites (control communication)
    """
    n_residues: int = 0
    n_contacts: int = 0
    spectral_gap: float = 0.0       # baseline λ₂/λ₃

    # Lock contacts (sorted by Δgap, most negative first)
    lock_edges: Optional[List[Tuple[int, int]]] = field(default=None, repr=False)
    lock_delta_gaps: Optional[List[float]] = field(default=None, repr=False)
    lock_residues: Optional[List[int]] = field(default=None, repr=False)
    n_locks: int = 0                 # top_k used

    # Per-residue surgical importance (max |Δgap| per residue)
    residue_importance: Optional[np.ndarray] = field(default=None, repr=False)

    # Full surgery results (all edges)
    all_delta_gaps: Optional[np.ndarray] = field(default=None, repr=False)
    all_edges: Optional[List[Tuple[int, int]]] = field(default=None, repr=False)

    # Lock boundary classification
    n_cross_domain: int = 0          # locks at domain boundaries
    n_within_domain: int = 0         # locks within domains
    boundary_enrichment: float = 0.0  # cross_fraction / baseline_fraction

    # Lock topology
    lock_topology: str = ""          # STAR, PATH, CYCLE, COMPLEX
    hub_residue: Optional[int] = None
    lock_degree_hist: Optional[Dict[int, int]] = field(default=None, repr=False)

    # Functional site enrichment (populated if annotations provided)
    enrichment_all: float = 0.0      # all_functional enrichment ratio
    enrichment_p: float = 1.0        # Fisher p-value
    enrichment_by_category: Optional[Dict[str, Dict]] = field(default=None, repr=False)
    importance_ratio: float = 0.0    # functional / non-functional importance
    importance_rho: float = 0.0      # Spearman(importance, is_functional)
    importance_p: float = 1.0

    # Mechanical vs chemical classification
    mechanical_locks: Optional[List[Tuple[int, int]]] = field(default=None, repr=False)
    chemical_locks: Optional[List[Tuple[int, int]]] = field(default=None, repr=False)
    signalling_locks: Optional[List[Tuple[int, int]]] = field(default=None, repr=False)
    unclassified_locks: Optional[List[Tuple[int, int]]] = field(default=None, repr=False)
    mechanical_fraction: float = 0.0
    chemical_fraction: float = 0.0
    signalling_fraction: float = 0.0

    # Domain interaction data (for peptide-module analysis)
    # Keys: (domain_i, domain_j) → number of lock contacts between them
    domain_lock_matrix: Optional[Dict[str, int]] = field(default=None, repr=False)
    # Fraction of total |Δgap| in inter-domain vs intra-domain contacts
    inter_domain_gap_fraction: float = 0.0
    modularity_score: float = 0.0    # 1 = fully modular, 0 = fully coupled

    time_total: float = 0.0


class IBPProteinAnalyzer:
    """Spectral ENM analyzer producing multi-perspective structural profiles.

    Parameters
    ----------
    cutoff : float
        Contact distance cutoff in Angstroms (default 8.0).
    s_min, s_max : float
        Stabilizer range for ENM spring constants.
    g_size : float
        Group size divisor (historical; acts as a global spring-constant scale).
    """

    def __init__(self, cutoff: float = 8.0, s_min: float = 12.0,
                 s_max: float = 18.0, g_size: float = 72.0):
        self.cutoff = cutoff
        self.s_min = s_min
        self.s_max = s_max
        self.g_size = g_size

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def analyze(self, coords: np.ndarray,
                b_factors: Optional[np.ndarray] = None) -> IBPResult:
        """Run the full IBP-ENM pipeline on a set of Cα coordinates.

        Parameters
        ----------
        coords : (N, 3) array
            Cα atom positions.
        b_factors : (N,) array, optional
            Experimental B-factors for correlation evaluation.

        Returns
        -------
        IBPResult
            Comprehensive structural analysis results.
        """
        t0 = time.perf_counter()
        result = IBPResult()
        N = len(coords)
        result.n_residues = N

        # --- contact map ---
        contacts, degrees = self._build_contacts(coords, N)
        result.n_contacts = len(contacts)

        if len(contacts) < N // 2:
            result.time_total = time.perf_counter() - t0
            return result  # Too sparse — bail

        # --- graph Laplacian + spectral decomposition ---
        L = self._build_laplacian(N, contacts)
        evals, evecs = np.linalg.eigh(L)
        result.laplacian_eigenvalues = evals
        result.laplacian_eigenvectors = evecs

        # --- stabilizer profiles ---
        s_cont = self._continuous_stabilizer(coords, degrees, N)
        s_lap = self._laplacian_stabilizer(evecs, N)
        s_uniform = np.full(N, (self.s_min + self.s_max) / 2)
        result.stabilizer_continuous = s_cont
        result.stabilizer_laplacian = s_lap

        # --- Fiedler vector → domain decomposition ---
        fiedler = evecs[:, 1]  # second-smallest eigenvector
        result.fiedler_vector = fiedler
        domain_labels, n_domains, spectral_gap = self._decompose_domains(
            fiedler, N, eigenvalues=evals)
        result.domain_labels = domain_labels
        result.n_domains = n_domains
        result.spectral_gap = spectral_gap

        # --- k-way spectral clustering ---
        spec_labels, spec_k, eigengaps = self._spectral_cluster(
            evals, evecs, N, max_k=8)
        result.spectral_labels = spec_labels
        result.spectral_k = spec_k
        result.spectral_eigengaps = eigengaps

        # --- Hinge detection ---
        hinge_scores, hinge_residues = self._detect_hinges(fiedler, N)
        result.hinge_scores = hinge_scores
        result.hinge_residues = hinge_residues

        # --- Debris recycling: domain-localised higher-mode analysis ---
        self._recycle_debris(result, coords, evals, evecs, domain_labels, N)

        # --- B-factor predictions ---
        b_scale = b_factors.max() if b_factors is not None else 1.0
        b_lap = self._solve_enm(N, contacts, s_lap, b_scale)
        b_cont = self._solve_enm(N, contacts, s_cont, b_scale)
        b_uni = self._solve_enm(N, contacts, s_uniform, b_scale)
        result.b_laplacian = b_lap
        result.b_continuous = b_cont
        result.b_uniform = b_uni

        # --- Correlation with experiment ---
        if b_factors is not None:
            rho_lap, _ = spearmanr(b_lap, b_factors)
            rho_cont, _ = spearmanr(b_cont, b_factors)
            rho_uni, _ = spearmanr(b_uni, b_factors)
            result.rho_laplacian = rho_lap
            result.rho_continuous = rho_cont
            result.rho_uniform = rho_uni

            # Consensus (weighted blend)
            weights = np.array([max(0, rho_lap), max(0, rho_cont), max(0, rho_uni)])
            if weights.sum() > 0:
                weights /= weights.sum()
            else:
                weights = np.ones(3) / 3
            b_cons = weights[0]*b_lap + weights[1]*b_cont + weights[2]*b_uni
            result.rho_consensus, _ = spearmanr(b_cons, b_factors)

            methods = [("LAPLACIAN", rho_lap), ("CONTINUOUS", rho_cont),
                       ("UNIFORM", rho_uni), ("CONSENSUS", result.rho_consensus)]
            result.best_method, result.rho_best = max(methods, key=lambda x: x[1])

        result.time_total = time.perf_counter() - t0
        return result

    # ------------------------------------------------------------------
    # Two-state comparison pipeline (D84/D86)
    # ------------------------------------------------------------------
    def compare(self, coords_a: np.ndarray, coords_b: np.ndarray,
                b_factors_a: Optional[np.ndarray] = None,
                b_factors_b: Optional[np.ndarray] = None,
                use_disturber: bool = False,
                functional_residues: Optional[List[int]] = None,
                ) -> 'ConformationalComparison':
        """Compare two conformational states of the same protein.

        Runs analyze() on both states, then extracts spectral time
        metrics, domain clocks, and debris flow between the two.

        Parameters
        ----------
        coords_a, coords_b : (N, 3) arrays
            Cα coordinates of the two states (must be same chain length).
        b_factors_a, b_factors_b : (N,) arrays, optional
            Experimental B-factors for each state.
        use_disturber : bool
            If True, also runs probe()+listen() and attaches the
            DisturbanceMessage + hotspot-weighted spectral time.
        functional_residues : list of int, optional
            Known functional residue indices (catalytic, binding, etc.).
            If provided with use_disturber, computes FunctionalSiteOverlap.

        Returns
        -------
        ConformationalComparison with full two-state analysis.
        """
        N = len(coords_a)
        assert len(coords_b) == N, f"Chain length mismatch: {N} vs {len(coords_b)}"

        # Run single-state analysis on both
        result_a = self.analyze(coords_a, b_factors_a)
        result_b = self.analyze(coords_b, b_factors_b)

        comp = ConformationalComparison()
        comp.result_a = result_a
        comp.result_b = result_b

        # --- RMSD ---
        comp.rmsd = self._kabsch_rmsd(coords_a, coords_b)

        # --- Spectral time metrics ---
        evals_a = result_a.laplacian_eigenvalues
        evals_b = result_b.laplacian_eigenvalues
        if evals_a is not None and evals_b is not None:
            st = self._compute_spectral_time(evals_a, evals_b, N)
            comp.fiedler_gap_delta = st['fiedler_gap_delta']
            comp.spectral_dist_log = st['spectral_dist_log']
            comp.spectral_dist_norm = st['spectral_dist_norm']
            comp.spectral_dist_weighted = st['spectral_dist_weighted']
            comp.wasserstein = st['wasserstein']
            comp.spectral_JS = st['spectral_JS']
            comp.entropy_delta = st['entropy_delta']

        # --- Domain clocks ---
        # Use state A's Fiedler partition for BOTH states (apples-to-apples)
        if result_a.domain_details and result_b.domain_labels is not None:
            clocks = []
            domain_labels_a = result_a.domain_labels
            unique_doms = sorted(set(domain_labels_a.tolist()))

            for dom_id in unique_doms:
                idx = np.where(domain_labels_a == dom_id)[0]
                if len(idx) < 5:
                    continue
                # Sub-analysis on state B using state A's partition
                dd_b = self._domain_sub_analysis(coords_b, idx, dom_id)
                # Find matching domain detail from state A
                dd_a = None
                for d in result_a.domain_details:
                    if d.domain_id == dom_id:
                        dd_a = d
                        break
                if dd_a is None:
                    continue

                clock = DomainClock(
                    domain_id=dom_id,
                    n_residues=len(idx),
                    gap_a=dd_a.spectral_gap,
                    gap_b=dd_b.spectral_gap,
                    delta_gap=dd_b.spectral_gap - dd_a.spectral_gap,
                    eigenvalues_a=dd_a.eigenvalues,
                    eigenvalues_b=dd_b.eigenvalues,
                )
                clocks.append(clock)

            comp.domain_clocks = clocks

            # Anti-correlation test
            if len(clocks) >= 2:
                deltas = [c.delta_gap for c in clocks]
                # Two domains: simple sign check
                if len(deltas) == 2:
                    comp.clock_anti_correlated = (deltas[0] * deltas[1] < 0)
                    comp.clock_spearman = -1.0 if comp.clock_anti_correlated else 1.0
                else:
                    # Multiple domains: Spearman on gap deltas
                    rho, _ = spearmanr(range(len(deltas)), deltas)
                    comp.clock_spearman = float(rho) if not np.isnan(rho) else 0.0
                    comp.clock_anti_correlated = comp.clock_spearman < -0.3

        # --- Debris flow ---
        comp.debris_fraction_a = result_a.debris_recycled_fraction
        comp.debris_fraction_b = result_b.debris_recycled_fraction
        comp.debris_delta = result_b.debris_recycled_fraction - result_a.debris_recycled_fraction

        # --- Domain label changes ---
        if result_a.domain_labels is not None and result_b.domain_labels is not None:
            comp.domain_label_changes = (
                result_a.domain_labels != result_b.domain_labels).astype(int)

        # --- Hinge overlap (Jaccard) ---
        h_a = set(result_a.hinge_residues or [])
        h_b = set(result_b.hinge_residues or [])
        if h_a or h_b:
            comp.hinge_overlap = len(h_a & h_b) / len(h_a | h_b)

        # --- Disturber language integration (D88) ---
        if use_disturber:
            resp_a = self.probe(coords_a, b_factors_a)
            resp_b = self.probe(coords_b, b_factors_b)
            msg = self.listen(resp_a, resp_b)

            comp.response_a = resp_a
            comp.response_b = resp_b
            comp.disturber_message = msg

            # Hotspot-weighted log-spectral distance:
            # Weight each eigenmode's contribution by how many of its
            # sign-change residues are disturber hotspots.
            if (evals_a is not None and evals_b is not None and
                    msg.hotspot_residues is not None):
                comp.hotspot_weighted_log_spectral = (
                    self._hotspot_weighted_spectral(
                        evals_a, evals_b, result_a, result_b,
                        set(msg.hotspot_residues), N))

            # Functional site overlap
            if functional_residues is not None and msg.hotspot_residues:
                comp.functional_overlap = self._compute_functional_overlap(
                    msg, functional_residues, N)

        return comp

    # ------------------------------------------------------------------
    # Disturber Language (D84/D87)
    # ------------------------------------------------------------------
    def probe(self, coords: np.ndarray,
              b_factors: Optional[np.ndarray] = None,
              cutoffs: Optional[List[float]] = None,
              modes: Optional[List[int]] = None,
              ambiguity_threshold: float = 0.1,
              ) -> 'DisturbanceResponse':
        """Generate the perturbation-response fingerprint for a protein.

        Sweeps the perturbation alphabet and records how the protein's
        structural partition changes.  The response IS the protein's
        "sentence" in the disturber's language.

        Parameters
        ----------
        coords : (N, 3) array
            Cα coordinates.
        b_factors : (N,) array, optional
            Experimental B-factors.
        cutoffs : list of float, optional
            Cutoff radii to sweep (default [6, 7, 8, 9, 10, 12]).
        modes : list of int, optional
            Mode indices to sweep (default [1, 2, 3]).
        ambiguity_threshold : float
            Cutoff entropy above which a residue is "ambiguous".

        Returns
        -------
        DisturbanceResponse
        """
        if cutoffs is None:
            cutoffs = [6.0, 7.0, 8.0, 9.0, 10.0, 12.0]
        if modes is None:
            modes = [1, 2, 3]

        N = len(coords)
        resp = DisturbanceResponse(n_residues=N)

        # --- Canonical analysis at default cutoff ---
        resp.canonical_result = self.analyze(coords, b_factors)

        # --- Cutoff sweep: collect partition labels per cutoff ---
        cutoff_partitions = {}   # cutoff -> (labels, k)
        for c in cutoffs:
            prober = IBPProteinAnalyzer(cutoff=c, s_min=self.s_min,
                                        s_max=self.s_max, g_size=self.g_size)
            r = prober.analyze(coords)
            cutoff_partitions[c] = (r.spectral_labels.copy()
                                    if r.spectral_labels is not None
                                    else np.zeros(N, dtype=int),
                                    r.spectral_k)

        # Domain counts per cutoff
        resp.domain_counts = {c: v[1] for c, v in cutoff_partitions.items()}

        # --- Per-residue cutoff entropy ---
        # For each residue, compute Shannon entropy of its label
        # distribution across cutoffs.
        label_matrix = np.column_stack([cutoff_partitions[c][0] for c in cutoffs])
        entropy = np.zeros(N)
        n_cuts = len(cutoffs)
        for i in range(N):
            counts = np.bincount(label_matrix[i])
            probs = counts / n_cuts
            probs = probs[probs > 0]
            entropy[i] = -np.sum(probs * np.log2(probs))

        resp.cutoff_entropy = entropy
        resp.ambiguous_residues = [int(i) for i in range(N)
                                    if entropy[i] > ambiguity_threshold]
        resp.fraction_ambiguous = len(resp.ambiguous_residues) / N

        # --- Mode-cutoff interaction matrix ---
        # For each (mode, cutoff), compute the partition from that mode's
        # eigenvector sign at that cutoff.
        interaction = {}  # (mode_k, cutoff_r) -> labels
        for c in cutoffs:
            prober = IBPProteinAnalyzer(cutoff=c, s_min=self.s_min,
                                        s_max=self.s_max, g_size=self.g_size)
            contacts, degrees = prober._build_contacts(coords, N)
            if len(contacts) < N // 2:
                continue
            L = prober._build_laplacian(N, contacts)
            evals, evecs = np.linalg.eigh(L)
            for mk in modes:
                if mk < len(evals):
                    labels = (evecs[:, mk] >= 0).astype(int)
                    interaction[(mk, c)] = labels

        # Pairwise agreement (Adjusted Rand Index via simple overlap)
        keys = sorted(interaction.keys())
        agreement = {}
        for i, k1 in enumerate(keys):
            for j, k2 in enumerate(keys):
                if j <= i:
                    continue
                l1, l2 = interaction[k1], interaction[k2]
                # Simple agreement: fraction of residues with same label
                ag = float(np.mean(l1 == l2))
                key_str = f"m{k1[0]}c{k1[1]}_m{k2[0]}c{k2[1]}"
                agreement[key_str] = round(ag, 4)

        resp.interaction_agreement = agreement

        # --- Robustness statistics ---
        # Mode robustness: same mode, different cutoffs
        mode_agreements = []
        for mk in modes:
            mode_keys = [(m, c) for (m, c) in keys if m == mk]
            for i in range(len(mode_keys)):
                for j in range(i + 1, len(mode_keys)):
                    l1 = interaction[mode_keys[i]]
                    l2 = interaction[mode_keys[j]]
                    mode_agreements.append(float(np.mean(l1 == l2)))

        # Cutoff robustness: same cutoff, different modes
        cutoff_agreements = []
        for c in cutoffs:
            cut_keys = [(m, cc) for (m, cc) in keys if cc == c]
            for i in range(len(cut_keys)):
                for j in range(i + 1, len(cut_keys)):
                    l1 = interaction[cut_keys[i]]
                    l2 = interaction[cut_keys[j]]
                    cutoff_agreements.append(float(np.mean(l1 == l2)))

        resp.mode_robustness = (float(np.mean(mode_agreements))
                                if mode_agreements else 0.0)
        resp.cutoff_robustness = (float(np.mean(cutoff_agreements))
                                  if cutoff_agreements else 0.0)

        # --- Perspective analysis ---
        # Three stabilizers "parse" the same Laplacian differently.
        # Their agreement measures how convergent the structural reading is.
        canonical = resp.canonical_result
        if (canonical.stabilizer_laplacian is not None and
                canonical.stabilizer_continuous is not None and
                canonical.laplacian_eigenvalues is not None):
            perspectives = {}
            for name, s_eff in [('laplacian', canonical.stabilizer_laplacian),
                                ('continuous', canonical.stabilizer_continuous),
                                ('uniform', np.full(N, (self.s_min + self.s_max) / 2))]:
                # Use the perspective's stabilizer to weight the ENM,
                # then do spectral clustering on that weighted graph.
                contacts, _ = self._build_contacts(coords, N)
                G = np.zeros((N, N))
                for (ii, jj), d in contacts.items():
                    si, sj = s_eff[ii], s_eff[jj]
                    w = 2 * si * sj / (si + sj) / self.g_size * (self.cutoff / d) ** 2
                    G[ii, jj] = -w
                    G[jj, ii] = -w
                    G[ii, ii] += w
                    G[jj, jj] += w
                ev, ec = np.linalg.eigh(G)
                labels, k, _ = self._spectral_cluster(ev, ec, N, max_k=8)
                perspectives[name] = labels

            resp.perspective_labels = perspectives

            # Pairwise perspective agreement
            persp_names = sorted(perspectives.keys())
            persp_agreement = {}
            for i in range(len(persp_names)):
                for j in range(i + 1, len(persp_names)):
                    n1, n2 = persp_names[i], persp_names[j]
                    ag = float(np.mean(perspectives[n1] == perspectives[n2]))
                    persp_agreement[f"{n1}_vs_{n2}"] = round(ag, 4)
            resp.perspective_agreement = persp_agreement

        return resp

    @staticmethod
    def listen(response_a: 'DisturbanceResponse',
              response_b: 'DisturbanceResponse') -> 'DisturbanceMessage':
        """Compare two perturbation responses to extract the conformational message.

        The disturber "listens" to how the protein's sentence changed
        between two states.  The difference IS the conformational signal,
        expressed in the disturber's own language.

        Parameters
        ----------
        response_a, response_b : DisturbanceResponse
            Perturbation fingerprints from two conformational states.

        Returns
        -------
        DisturbanceMessage describing what changed.
        """
        msg = DisturbanceMessage()
        N = response_a.n_residues

        # --- Per-residue entropy delta ---
        if (response_a.cutoff_entropy is not None and
                response_b.cutoff_entropy is not None):
            delta = response_b.cutoff_entropy - response_a.cutoff_entropy
            msg.entropy_delta = delta
            # Hotspots: residues that changed ambiguity substantially
            threshold = 0.3  # bits
            msg.hotspot_residues = [int(i) for i in range(N)
                                    if abs(delta[i]) > threshold]
            msg.n_hotspots = len(msg.hotspot_residues)
            msg.total_information = float(np.sum(np.abs(delta)))
            msg.message_strength = msg.n_hotspots / N if N > 0 else 0.0

        # --- Robustness delta ---
        msg.mode_robustness_delta = (response_b.mode_robustness -
                                     response_a.mode_robustness)
        msg.cutoff_robustness_delta = (response_b.cutoff_robustness -
                                       response_a.cutoff_robustness)

        # --- Domain count deltas ---
        if (response_a.domain_counts is not None and
                response_b.domain_counts is not None):
            common_cuts = set(response_a.domain_counts) & set(response_b.domain_counts)
            msg.domain_count_deltas = {
                c: response_b.domain_counts[c] - response_a.domain_counts[c]
                for c in sorted(common_cuts)
            }

        # --- Interaction delta ---
        if (response_a.interaction_agreement is not None and
                response_b.interaction_agreement is not None):
            common_keys = (set(response_a.interaction_agreement) &
                           set(response_b.interaction_agreement))
            msg.interaction_delta = {
                k: round(response_b.interaction_agreement[k] -
                         response_a.interaction_agreement[k], 4)
                for k in sorted(common_keys)
            }

        # --- Perspective shift ---
        if (response_a.perspective_agreement is not None and
                response_b.perspective_agreement is not None):
            common = (set(response_a.perspective_agreement) &
                      set(response_b.perspective_agreement))
            msg.perspective_shift = {
                k: round(response_b.perspective_agreement[k] -
                         response_a.perspective_agreement[k], 4)
                for k in sorted(common)
            }

        return msg

    # ------------------------------------------------------------------
    # Distribution Network (D89)
    # ------------------------------------------------------------------
    def probe_distribution(self, coords: np.ndarray,
                           b_factors: Optional[np.ndarray] = None,
                           n_trials: int = 5,
                           rng_seed: int = 42,
                           ) -> 'DistributionResponse':
        """Fire multiple perturbation types through the protein.

        The disturber channel as a distribution network: each perturbation
        type is a different signal routed through the same structure.
        Per-residue sensitivity to each type creates a response profile
        that characterises HOW the protein responds, not just WHETHER.

        Perturbation types
        ------------------
        stiffness : scale s_min/s_max by [0.5, 0.75, 1.25, 1.5, 2.0]
            Tests: does the partition depend on spring constant magnitude?
        topology  : delete 10%, 20%, 30% of contacts at random (n_trials each)
            Tests: which residues depend on specific contacts?
        geometry  : add Gaussian noise σ=[0.5, 1.0, 2.0] Å to coords
            Tests: which residues are geometrically fragile?
        mass      : double the stabilizer at 10 random residues (n_trials each)
            Tests: which residues propagate local perturbations?

        Returns
        -------
        DistributionResponse with per-residue sensitivity per channel.
        """
        rng = np.random.RandomState(rng_seed)
        N = len(coords)
        resp = DistributionResponse(n_residues=N)

        # Canonical partition (reference)
        canonical = self.analyze(coords, b_factors)
        resp.canonical_result = canonical
        ref_labels = (canonical.spectral_labels.copy()
                      if canonical.spectral_labels is not None
                      else np.zeros(N, dtype=int))

        def _label_distance(labels_a, labels_b):
            """Per-residue binary: did label change? Averaged over trials."""
            return (labels_a != labels_b).astype(float)

        # --- Stiffness perturbation ---
        stiffness_factors = [0.5, 0.75, 1.25, 1.5, 2.0]
        stiffness_sens = np.zeros(N)
        for factor in stiffness_factors:
            a = IBPProteinAnalyzer(cutoff=self.cutoff,
                                   s_min=self.s_min * factor,
                                   s_max=self.s_max * factor,
                                   g_size=self.g_size)
            r = a.analyze(coords)
            labels = (r.spectral_labels if r.spectral_labels is not None
                      else np.zeros(N, dtype=int))
            stiffness_sens += _label_distance(ref_labels, labels)
        stiffness_sens /= len(stiffness_factors)
        resp.stiffness_sensitivity = stiffness_sens

        # --- Topology perturbation ---
        contacts, degrees = self._build_contacts(coords, N)
        contact_list = list(contacts.keys())
        delete_fracs = [0.1, 0.2, 0.3]
        topology_sens = np.zeros(N)
        n_topo_trials = 0
        for frac in delete_fracs:
            n_delete = max(1, int(len(contact_list) * frac))
            for trial in range(n_trials):
                # Random contact deletion
                keep_mask = np.ones(len(contact_list), dtype=bool)
                delete_idx = rng.choice(len(contact_list), n_delete, replace=False)
                keep_mask[delete_idx] = False
                # Build perturbed Laplacian
                L_pert = np.zeros((N, N))
                for ci, (i, j) in enumerate(contact_list):
                    if keep_mask[ci]:
                        L_pert[i, j] = -1
                        L_pert[j, i] = -1
                        L_pert[i, i] += 1
                        L_pert[j, j] += 1
                try:
                    ev, ec = np.linalg.eigh(L_pert)
                    labels, k, _ = self._spectral_cluster(ev, ec, N, max_k=8)
                    topology_sens += _label_distance(ref_labels, labels)
                    n_topo_trials += 1
                except Exception:
                    pass
        if n_topo_trials > 0:
            topology_sens /= n_topo_trials
        resp.topology_sensitivity = topology_sens

        # --- Geometry perturbation ---
        noise_sigmas = [0.5, 1.0, 2.0]
        geometry_sens = np.zeros(N)
        n_geo_trials = 0
        for sigma in noise_sigmas:
            for trial in range(n_trials):
                noisy_coords = coords + rng.randn(N, 3) * sigma
                r = self.analyze(noisy_coords)
                labels = (r.spectral_labels if r.spectral_labels is not None
                          else np.zeros(N, dtype=int))
                geometry_sens += _label_distance(ref_labels, labels)
                n_geo_trials += 1
        if n_geo_trials > 0:
            geometry_sens /= n_geo_trials
        resp.geometry_sensitivity = geometry_sens

        # --- Mass perturbation ---
        # "Mass" = doubling the stabilizer at random residue subsets
        # simulates local binding/ligand effects
        mass_sens = np.zeros(N)
        n_mass_trials = 0
        n_mass_residues = max(3, N // 20)  # ~5% of residues
        for trial in range(n_trials * 3):
            mass_idx = rng.choice(N, n_mass_residues, replace=False)
            # Perturb the continuous stabilizer
            s_eff = canonical.stabilizer_continuous.copy()
            if s_eff is None:
                break
            s_eff[mass_idx] *= 2.0  # double stiffness at these sites
            # Solve perturbed ENM and cluster
            G_pert = np.zeros((N, N))
            for (i, j), d in contacts.items():
                si, sj = s_eff[i], s_eff[j]
                w = 2 * si * sj / (si + sj) / self.g_size * (self.cutoff / d) ** 2
                G_pert[i, j] = -w
                G_pert[j, i] = -w
                G_pert[i, i] += w
                G_pert[j, j] += w
            try:
                ev, ec = np.linalg.eigh(G_pert)
                labels, k, _ = self._spectral_cluster(ev, ec, N, max_k=8)
                mass_sens += _label_distance(ref_labels, labels)
                n_mass_trials += 1
            except Exception:
                pass
        if n_mass_trials > 0:
            mass_sens /= n_mass_trials
        resp.mass_sensitivity = mass_sens

        # --- Build response matrix and derived quantities ---
        resp.response_matrix = np.column_stack([
            stiffness_sens, topology_sens, geometry_sens, mass_sens])
        channel_names = ['stiffness', 'topology', 'geometry', 'mass']
        resp.dominant_channel = np.argmax(resp.response_matrix, axis=1)

        resp.mean_sensitivity = {
            name: float(np.mean(resp.response_matrix[:, i]))
            for i, name in enumerate(channel_names)
        }

        # Channel agreement: pairwise correlation between sensitivity profiles
        ch_agree = {}
        for i in range(4):
            for j in range(i + 1, 4):
                rho, _ = spearmanr(resp.response_matrix[:, i],
                                   resp.response_matrix[:, j])
                key = f"{channel_names[i]}_vs_{channel_names[j]}"
                ch_agree[key] = round(float(rho) if not np.isnan(rho) else 0.0, 4)
        resp.channel_agreement = ch_agree

        # Channel independence: mean pairwise disagreement
        agreements = list(ch_agree.values())
        resp.channel_independence = round(
            1.0 - float(np.mean(np.abs(agreements))) if agreements else 0.0, 4)

        return resp

    # ------------------------------------------------------------------
    # Shadow Perturbation (D89)
    # ------------------------------------------------------------------
    def shadow_probe(self, coords: np.ndarray,
                     b_factors: Optional[np.ndarray] = None,
                     n_trials: int = 10,
                     epsilon: float = 0.15,
                     rng_seed: int = 42,
                     ) -> 'ShadowResponse':
        """Perturb the disturber: measure parser fragility.

        Instead of perturbing the protein and measuring through fixed
        stabilizers, this perturbs each stabilizer profile directly:

            s_perturbed(i) = s(i) * (1 + ε·δ_i),  δ_i ~ N(0,1)

        then re-solves the ENM and checks if the partition changed.
        A residue where the partition consistently changes under
        stabilizer perturbation is at a PARSING BOUNDARY — its
        structural assignment depends on the precise parser weights.

        This tests time-independent sensitivity: the shadow's own
        fragility, not the protein's conformational state.

        Parameters
        ----------
        coords : (N, 3) array
            Cα coordinates.
        b_factors : (N,) array, optional
            Experimental B-factors.
        n_trials : int
            Number of random perturbation trials per stabilizer.
        epsilon : float
            Perturbation magnitude (fraction of stabilizer value).
        rng_seed : int
            Random seed for reproducibility.

        Returns
        -------
        ShadowResponse with per-residue parser fragility.
        """
        rng = np.random.RandomState(rng_seed)
        N = len(coords)
        sr = ShadowResponse(n_residues=N)

        # Canonical analysis
        canonical = self.analyze(coords, b_factors)
        sr.canonical_result = canonical
        ref_labels = (canonical.spectral_labels.copy()
                      if canonical.spectral_labels is not None
                      else np.zeros(N, dtype=int))

        contacts, degrees = self._build_contacts(coords, N)
        if len(contacts) < N // 2:
            return sr

        # Get the three stabilizer profiles
        s_lap = canonical.stabilizer_laplacian
        s_cont = canonical.stabilizer_continuous
        s_uni = np.full(N, (self.s_min + self.s_max) / 2)

        if s_lap is None or s_cont is None:
            return sr

        stabilizers = [
            ('laplacian', s_lap),
            ('continuous', s_cont),
            ('uniform', s_uni),
        ]

        shadow_maps = {}  # name -> (N,) sensitivity

        for stab_name, s_base in stabilizers:
            sens = np.zeros(N)
            valid_trials = 0
            for trial in range(n_trials):
                # Perturb the stabilizer: multiplicative noise
                delta = rng.randn(N)
                s_pert = s_base * (1.0 + epsilon * delta)
                # Clamp to valid range
                s_pert = np.clip(s_pert, self.s_min * 0.3, self.s_max * 3.0)

                # Build perturbed weighted ENM
                G = np.zeros((N, N))
                for (i, j), d in contacts.items():
                    si, sj = s_pert[i], s_pert[j]
                    w = 2 * si * sj / (si + sj) / self.g_size * (self.cutoff / d) ** 2
                    G[i, j] = -w
                    G[j, i] = -w
                    G[i, i] += w
                    G[j, j] += w

                try:
                    ev, ec = np.linalg.eigh(G)
                    labels, k, _ = self._spectral_cluster(ev, ec, N, max_k=8)
                    sens += (labels != ref_labels).astype(float)
                    valid_trials += 1
                except Exception:
                    pass

            if valid_trials > 0:
                sens /= valid_trials
            shadow_maps[stab_name] = sens

        sr.shadow_laplacian = shadow_maps['laplacian']
        sr.shadow_continuous = shadow_maps['continuous']
        sr.shadow_uniform = shadow_maps['uniform']

        # Composite: mean across stabilizers
        sr.shadow_composite = (
            shadow_maps['laplacian'] +
            shadow_maps['continuous'] +
            shadow_maps['uniform']
        ) / 3.0

        # Fragile residues: high composite sensitivity
        fragile_threshold = 0.3
        sr.fragile_residues = [int(i) for i in range(N)
                               if sr.shadow_composite[i] > fragile_threshold]
        sr.n_fragile = len(sr.fragile_residues)
        sr.fraction_fragile = sr.n_fragile / N if N > 0 else 0.0

        # Cross-stabilizer agreement on fragility
        stab_names = sorted(shadow_maps.keys())
        sh_agree = {}
        for i in range(len(stab_names)):
            for j in range(i + 1, len(stab_names)):
                rho, _ = spearmanr(shadow_maps[stab_names[i]],
                                   shadow_maps[stab_names[j]])
                key = f"{stab_names[i]}_vs_{stab_names[j]}"
                sh_agree[key] = round(float(rho) if not np.isnan(rho) else 0.0, 4)
        sr.shadow_agreement = sh_agree

        sr.mean_fragility = float(np.mean(sr.shadow_composite))

        return sr

    # ------------------------------------------------------------------
    # Shadow ε-sweep (D90)
    # ------------------------------------------------------------------
    def shadow_probe_sweep(self, coords: np.ndarray,
                           b_factors: Optional[np.ndarray] = None,
                           epsilon_values: Optional[np.ndarray] = None,
                           n_trials: int = 8,
                           critical_threshold: float = 0.5,
                           rng_seed: int = 42,
                           ) -> 'ShadowSweepResult':
        """Sweep ε and measure partition fragility + spectral gap.

        For each ε in epsilon_values:
          1. For each trial, perturb ALL three stabilizers by
             s_perturbed = s * (1 + ε·δ), δ ~ N(0,1)
          2. Build perturbed weighted ENM, spectral-cluster
          3. Record per-residue label-change rate = fragility(ε)
          4. Record perturbed spectral gap λ₂/λ₃

        This builds the "fragility landscape" f_i(ε): how each
        residue's structural assignment responds as parser noise
        increases.  The critical ε where the partition shatters
        is a protein-level structural invariant.

        Parameters
        ----------
        coords : (N, 3)
        b_factors : optional
        epsilon_values : 1D array of ε to sweep (default: logspace)
        n_trials : trials per ε per stabilizer
        critical_threshold : fraction_fragile defining "broken"
        rng_seed : seed

        Returns
        -------
        ShadowSweepResult
        """
        rng = np.random.RandomState(rng_seed)
        N = len(coords)
        sr = ShadowSweepResult(n_residues=N)

        if epsilon_values is None:
            epsilon_values = np.array([
                0.01, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20,
                0.30, 0.40, 0.50, 0.75, 1.00])
        sr.epsilon_values = epsilon_values
        sr.n_epsilon = len(epsilon_values)
        sr.critical_threshold = critical_threshold

        # Canonical analysis
        canonical = self.analyze(coords, b_factors)
        sr.canonical_result = canonical
        ref_labels = (canonical.spectral_labels.copy()
                      if canonical.spectral_labels is not None
                      else np.zeros(N, dtype=int))

        # Unperturbed spectral gap
        if (canonical.laplacian_eigenvalues is not None and
                len(canonical.laplacian_eigenvalues) > 2):
            lam2 = canonical.laplacian_eigenvalues[1]
            lam3 = canonical.laplacian_eigenvalues[2]
            sr.gap_at_zero = float(lam2 / lam3) if lam3 > 1e-10 else 0.0
        else:
            sr.gap_at_zero = 0.0

        contacts, degrees = self._build_contacts(coords, N)
        if len(contacts) < N // 2:
            return sr

        # Get stabilizer profiles
        s_lap = canonical.stabilizer_laplacian
        s_cont = canonical.stabilizer_continuous
        s_uni = np.full(N, (self.s_min + self.s_max) / 2)
        if s_lap is None or s_cont is None:
            return sr

        stabilizers = [s_lap, s_cont, s_uni]

        # Pre-allocate
        landscape = np.zeros((N, len(epsilon_values)))
        frac_fragile = np.zeros(len(epsilon_values))
        mean_frag = np.zeros(len(epsilon_values))
        gap_mean = np.zeros(len(epsilon_values))
        gap_std = np.zeros(len(epsilon_values))

        for ei, eps in enumerate(epsilon_values):
            # Accumulate across all stabilizers and trials
            total_sens = np.zeros(N)
            gap_samples = []
            n_valid = 0

            for s_base in stabilizers:
                for trial in range(n_trials):
                    delta = rng.randn(N)
                    s_pert = s_base * (1.0 + eps * delta)
                    s_pert = np.clip(s_pert, self.s_min * 0.3,
                                     self.s_max * 3.0)

                    G = np.zeros((N, N))
                    for (i, j), d in contacts.items():
                        si, sj = s_pert[i], s_pert[j]
                        w = (2 * si * sj / (si + sj) / self.g_size
                             * (self.cutoff / d) ** 2)
                        G[i, j] = -w
                        G[j, i] = -w
                        G[i, i] += w
                        G[j, j] += w

                    try:
                        ev, ec = np.linalg.eigh(G)
                        labels, k, _ = self._spectral_cluster(
                            ev, ec, N, max_k=8)
                        total_sens += (labels != ref_labels).astype(float)
                        n_valid += 1
                        # Spectral gap
                        if len(ev) > 2 and ev[2] > 1e-10:
                            gap_samples.append(float(ev[1] / ev[2]))
                    except Exception:
                        pass

            if n_valid > 0:
                total_sens /= n_valid
            landscape[:, ei] = total_sens
            frac_fragile[ei] = float(np.mean(total_sens > 0.3))
            mean_frag[ei] = float(np.mean(total_sens))
            gap_mean[ei] = float(np.mean(gap_samples)) if gap_samples else 0.0
            gap_std[ei] = float(np.std(gap_samples)) if gap_samples else 0.0

        sr.fragility_landscape = landscape
        sr.fraction_fragile_curve = frac_fragile
        sr.mean_fragility_curve = mean_frag
        sr.gap_curve = gap_mean
        sr.gap_std_curve = gap_std

        # Critical epsilon
        crit_mask = np.where(frac_fragile >= critical_threshold)[0]
        if len(crit_mask) > 0:
            sr.critical_epsilon = float(epsilon_values[crit_mask[0]])
        else:
            sr.critical_epsilon = float('inf')

        # Gap sensitivity: finite-difference slope at smallest ε
        if len(epsilon_values) >= 2 and gap_mean[0] > 0:
            sr.gap_sensitivity = float(
                (gap_mean[1] - gap_mean[0]) /
                (epsilon_values[1] - epsilon_values[0]))
        else:
            sr.gap_sensitivity = 0.0

        return sr

    # ------------------------------------------------------------------
    # Meta-predictor: multi-channel fusion (D91)
    # ------------------------------------------------------------------
    def meta_predict(self, coords: np.ndarray,
                     b_factors: Optional[np.ndarray] = None,
                     shadow_trials: int = 8,
                     distribution_trials: int = 3,
                     ) -> 'MetaPrediction':
        """Fuse all channels into a per-residue consensus prediction.

        Runs all five measurement channels, normalises each to [0,1],
        then builds a per-residue vote count and continuous consensus
        score.  Also casts the shadow prediction into every other
        channel to check cross-channel agreement.

        Channels
        --------
        1. shadow      — parser fragility (shadow_probe)
        2. debris       — recycled hinge scores (analyze)
        3. language     — cutoff entropy (probe)
        4. distribution — max perturbation sensitivity (probe_distribution)
        5. spectral     — 1 - |fiedler| (boundary proximity)

        Parameters
        ----------
        coords : (N, 3)
        b_factors : optional
        shadow_trials : trials for shadow_probe
        distribution_trials : trials for probe_distribution

        Returns
        -------
        MetaPrediction with per-residue consensus.
        """
        N = len(coords)
        mp = MetaPrediction(n_residues=N)

        # --- Run all channels ---
        ibp = self.analyze(coords, b_factors)
        mp.ibp_result = ibp

        shadow = self.shadow_probe(coords, b_factors,
                                   n_trials=shadow_trials)
        mp.shadow_response = shadow

        language = self.probe(coords, b_factors)
        mp.disturbance_response = language

        distribution = self.probe_distribution(coords, b_factors,
                                               n_trials=distribution_trials)
        mp.distribution_response = distribution

        # --- Extract per-residue signals ---
        norm = self._norm01

        # Shadow: parser fragility
        sig_shadow = np.zeros(N)
        if shadow.shadow_composite is not None:
            sig_shadow = norm(shadow.shadow_composite)
        mp.signal_shadow = sig_shadow

        # Debris: recycled hinge scores
        sig_debris = np.zeros(N)
        if ibp.recycled_hinge_scores is not None:
            sig_debris = norm(ibp.recycled_hinge_scores)
        mp.signal_debris = sig_debris

        # Language: cutoff entropy
        sig_lang = np.zeros(N)
        if language.cutoff_entropy is not None:
            sig_lang = norm(language.cutoff_entropy)
        mp.signal_language = sig_lang

        # Distribution: max sensitivity across perturbation types
        sig_dist = np.zeros(N)
        if distribution.response_matrix is not None:
            sig_dist = norm(np.max(distribution.response_matrix, axis=1))
        mp.signal_distribution = sig_dist

        # Spectral: boundary proximity = 1 - normalised |fiedler|
        sig_spec = np.zeros(N)
        if ibp.fiedler_vector is not None:
            sig_spec = 1.0 - norm(np.abs(ibp.fiedler_vector))
        mp.signal_spectral = sig_spec

        # --- Per-residue voting ---
        signals = np.column_stack([
            sig_shadow, sig_debris, sig_lang, sig_dist, sig_spec])
        channel_names = ['shadow', 'debris', 'language',
                         'distribution', 'spectral']

        # Each channel votes "ambiguous" if residue is above its median
        medians = np.median(signals, axis=0)
        votes = (signals > medians).astype(int)
        mp.consensus_votes = votes.sum(axis=1)  # 0..5

        # Continuous consensus: mean normalised signal
        mp.consensus_score = signals.mean(axis=1)

        # Classification
        mp.n_stable = int(np.sum(mp.consensus_votes <= 1))
        mp.n_mixed = int(np.sum((mp.consensus_votes >= 2) &
                                (mp.consensus_votes <= 3)))
        mp.n_ambiguous = int(np.sum(mp.consensus_votes >= 4))
        mp.fraction_ambiguous = mp.n_ambiguous / N if N > 0 else 0.0

        # --- Pairwise channel agreement ---
        ch_agree = {}
        rho_vals = []
        for i in range(5):
            for j in range(i + 1, 5):
                r, _ = spearmanr(signals[:, i], signals[:, j])
                r = float(r) if not np.isnan(r) else 0.0
                key = f"{channel_names[i]}_vs_{channel_names[j]}"
                ch_agree[key] = round(r, 4)
                rho_vals.append(abs(r))
        mp.channel_agreement = ch_agree
        mp.mean_pairwise_rho = round(float(np.mean(rho_vals)), 4)

        # --- Shadow cast ---
        mp.shadow_cast = self._cast_shadow(
            shadow, ibp, language, distribution, N)

        return mp

    @staticmethod
    def _cast_shadow(shadow: 'ShadowResponse',
                     ibp: 'IBPResult',
                     language: 'DisturbanceResponse',
                     distribution: 'DistributionResponse',
                     N: int) -> 'ShadowCast':
        """Project shadow predictions into every other channel.

        The shadow says 'these residues are fragile.'  Cast that
        prediction into debris, language, distribution, and spectral
        spaces to see if the other microphones hear the same thing.
        """
        sc = ShadowCast(n_residues=N)
        s_comp = shadow.shadow_composite
        if s_comp is None:
            return sc

        fragile_set = set(shadow.fragile_residues or [])
        stable_set = set(range(N)) - fragile_set
        channel_names_dist = ['stiffness', 'topology', 'geometry', 'mass']

        # --- Shadow → debris ---
        rhs = ibp.recycled_hinge_scores
        if rhs is not None and np.std(s_comp) > 1e-10:
            r, _ = spearmanr(s_comp, rhs)
            sc.shadow_debris_rho = float(r) if not np.isnan(r) else 0.0

            hinge_set = set(ibp.recycled_hinges or [])
            hinge_idx = np.array(sorted(hinge_set), dtype=int)
            non_hinge_idx = np.array([i for i in range(N)
                                      if i not in hinge_set], dtype=int)
            if len(hinge_idx) > 0:
                sc.shadow_at_hinges = float(np.mean(s_comp[hinge_idx]))
            if len(non_hinge_idx) > 0:
                sc.shadow_at_non_hinges = float(np.mean(s_comp[non_hinge_idx]))
            if sc.shadow_at_non_hinges > 1e-10:
                sc.debris_enrichment = sc.shadow_at_hinges / sc.shadow_at_non_hinges

        # --- Shadow → language ---
        ce = language.cutoff_entropy
        if ce is not None and np.std(s_comp) > 1e-10:
            r, _ = spearmanr(s_comp, ce)
            sc.shadow_language_rho = float(r) if not np.isnan(r) else 0.0

            ambig = set(language.ambiguous_residues or [])
            ambig_idx = np.array(sorted(ambig), dtype=int)
            unambig_idx = np.array([i for i in range(N)
                                    if i not in ambig], dtype=int)
            if len(ambig_idx) > 0:
                sc.shadow_at_ambiguous = float(np.mean(s_comp[ambig_idx]))
            if len(unambig_idx) > 0:
                sc.shadow_at_unambiguous = float(np.mean(s_comp[unambig_idx]))
            if sc.shadow_at_unambiguous > 1e-10:
                sc.language_enrichment = (sc.shadow_at_ambiguous /
                                          sc.shadow_at_unambiguous)

        # --- Shadow → distribution ---
        rm = distribution.response_matrix
        if rm is not None and np.std(s_comp) > 1e-10:
            max_sens = np.max(rm, axis=1)
            r, _ = spearmanr(s_comp, max_sens)
            sc.shadow_distribution_rho = float(r) if not np.isnan(r) else 0.0

            dom_ch = distribution.dominant_channel
            if dom_ch is not None and len(fragile_set) > 0:
                from collections import Counter as _Ctr
                frag_idx = np.array(sorted(fragile_set), dtype=int)
                stab_idx = np.array(sorted(stable_set), dtype=int)
                if len(frag_idx) > 0:
                    fc = _Ctr(dom_ch[frag_idx].tolist())
                    sc.fragile_dominant_channel = channel_names_dist[
                        fc.most_common(1)[0][0]]
                if len(stab_idx) > 0:
                    stc = _Ctr(dom_ch[stab_idx].tolist())
                    sc.stable_dominant_channel = channel_names_dist[
                        stc.most_common(1)[0][0]]
                # Channel divergence: do fragile and stable residues
                # prefer different perturbation channels?
                if len(frag_idx) > 0 and len(stab_idx) > 0:
                    frag_dist = np.bincount(dom_ch[frag_idx], minlength=4)
                    stab_dist = np.bincount(dom_ch[stab_idx], minlength=4)
                    frag_p = frag_dist / (frag_dist.sum() + 1e-10)
                    stab_p = stab_dist / (stab_dist.sum() + 1e-10)
                    # Jensen-Shannon divergence
                    m = (frag_p + stab_p) / 2
                    eps = 1e-12
                    kl1 = np.sum(frag_p * np.log((frag_p + eps) / (m + eps)))
                    kl2 = np.sum(stab_p * np.log((stab_p + eps) / (m + eps)))
                    sc.channel_divergence = round(0.5 * (kl1 + kl2), 4)

        # --- Shadow → spectral ---
        fv = ibp.fiedler_vector
        if fv is not None and np.std(s_comp) > 1e-10:
            abs_fiedler = np.abs(fv)
            r, _ = spearmanr(s_comp, abs_fiedler)
            sc.shadow_spectral_rho = float(r) if not np.isnan(r) else 0.0

            med_f = np.median(abs_fiedler)
            boundary_idx = np.where(abs_fiedler < med_f)[0]
            interior_idx = np.where(abs_fiedler >= med_f)[0]
            if len(boundary_idx) > 0:
                sc.shadow_at_boundary = float(np.mean(s_comp[boundary_idx]))
            if len(interior_idx) > 0:
                sc.shadow_at_interior = float(np.mean(s_comp[interior_idx]))
            if sc.shadow_at_interior > 1e-10:
                sc.boundary_enrichment = (sc.shadow_at_boundary /
                                          sc.shadow_at_interior)

        # --- Cross-cast summary ---
        cast_rhos = [
            abs(sc.shadow_debris_rho),
            abs(sc.shadow_language_rho),
            abs(sc.shadow_distribution_rho),
            abs(sc.shadow_spectral_rho),
        ]
        sc.n_agreeing_channels = sum(1 for r in cast_rhos if r > 0.2)
        sc.mean_cast_rho = round(float(np.mean(cast_rhos)), 4)

        return sc

    # ------------------------------------------------------------------
    # Spectral Surgery (D94/D96/D97)
    # ------------------------------------------------------------------
    def surgery(self, coords: np.ndarray,
                b_factors: Optional[np.ndarray] = None,
                top_k: int = 20,
                functional_annotation=None,
                ) -> 'SurgeryResult':
        """Spectral surgery: identify structural locks and classify them.

        Removes each contact one at a time from the graph Laplacian and
        measures the spectral gap (λ₂/λ₃) response.  Contacts whose
        removal maximally DROPS the gap are 'locks' — the bonds holding
        the protein rigid against conformational change.

        D94 established: locks concentrate at domain boundaries (11×).
        D96 validated:   locks hit known functional sites (2.6×).
        D97 classifies:  mechanical (hinge) vs chemical (active) vs
                         signalling (allosteric) locks.

        Parameters
        ----------
        coords : (N, 3) array
            Cα atom positions.
        b_factors : (N,) array, optional
            Experimental B-factors (passed through to analyze()).
        top_k : int
            Number of top lock contacts to report (default 20).
        functional_annotation : FunctionalAnnotation, optional
            From FunctionalSiteResolver.resolve(). If provided,
            computes enrichment and mechanical/chemical classification.
            Also accepts a plain dict with {category: [indices]}.

        Returns
        -------
        SurgeryResult with lock contacts, importance profile,
        topology, enrichment, and mechanical/chemical classification.
        """
        from scipy.stats import fisher_exact

        t0 = time.perf_counter()
        N = len(coords)
        sr = SurgeryResult(n_residues=N)

        # --- Single-state structural analysis ---
        result = self.analyze(coords, b_factors)
        if result.n_contacts < N // 2:
            sr.time_total = time.perf_counter() - t0
            return sr

        contacts, _ = self._build_contacts(coords, N)
        sr.n_contacts = len(contacts)
        fiedler = result.fiedler_vector
        domain_labels = result.domain_labels

        # --- Baseline spectrum ---
        L_base = self._build_laplacian(N, contacts)
        evals_base = np.linalg.eigh(L_base)[0]
        if len(evals_base) > 2 and evals_base[2] > 1e-10:
            sr.spectral_gap = float(evals_base[1] / evals_base[2])

        # --- Single-edge surgery: remove each contact ---
        edge_list = list(contacts.keys())
        n_edges = len(edge_list)
        delta_gaps = np.zeros(n_edges)

        for ei, edge in enumerate(edge_list):
            reduced = {k: v for k, v in contacts.items() if k != edge}
            L_red = self._build_laplacian(N, reduced)
            evals_red = np.linalg.eigh(L_red)[0]
            if len(evals_red) > 2 and evals_red[2] > 1e-10:
                gap_red = float(evals_red[1] / evals_red[2])
            else:
                gap_red = 0.0
            delta_gaps[ei] = gap_red - sr.spectral_gap

        # Sort by Δgap (most negative first = biggest gap drop)
        order = np.argsort(delta_gaps)
        sr.all_edges = [edge_list[i] for i in order]
        sr.all_delta_gaps = delta_gaps[order]

        # --- Top-k locks ---
        top_k = min(top_k, n_edges)
        sr.n_locks = top_k
        sr.lock_edges = sr.all_edges[:top_k]
        sr.lock_delta_gaps = sr.all_delta_gaps[:top_k].tolist()

        lock_residue_set = set()
        for i, j in sr.lock_edges:
            lock_residue_set.add(i)
            lock_residue_set.add(j)
        sr.lock_residues = sorted(lock_residue_set)

        # --- Per-residue surgical importance ---
        residue_importance = np.zeros(N)
        for ei, edge in enumerate(edge_list):
            imp = abs(delta_gaps[ei])
            i, j = edge
            if imp > residue_importance[i]:
                residue_importance[i] = imp
            if imp > residue_importance[j]:
                residue_importance[j] = imp
        sr.residue_importance = residue_importance

        # --- Boundary classification ---
        if fiedler is not None:
            n_cross = 0
            for i, j in sr.lock_edges:
                if fiedler[i] * fiedler[j] < 0:
                    n_cross += 1
            sr.n_cross_domain = n_cross
            sr.n_within_domain = top_k - n_cross

            # Baseline cross-domain fraction
            all_cross = sum(1 for (i, j) in edge_list
                            if fiedler[i] * fiedler[j] < 0)
            baseline_frac = all_cross / n_edges if n_edges > 0 else 0
            lock_frac = n_cross / top_k if top_k > 0 else 0
            sr.boundary_enrichment = (lock_frac / baseline_frac
                                      if baseline_frac > 1e-10 else 0.0)

        # --- Lock topology ---
        lock_degrees = Counter()
        for i, j in sr.lock_edges:
            lock_degrees[i] += 1
            lock_degrees[j] += 1

        if lock_degrees:
            max_deg = max(lock_degrees.values())
            sr.lock_degree_hist = dict(Counter(lock_degrees.values()))

            if max_deg >= top_k // 2:
                sr.lock_topology = "STAR"
                sr.hub_residue = max(lock_degrees, key=lock_degrees.get)
            elif all(d <= 2 for d in lock_degrees.values()):
                if len(lock_residue_set) == len(sr.lock_edges):
                    sr.lock_topology = "CYCLE"
                else:
                    sr.lock_topology = "PATH"
            else:
                sr.lock_topology = "COMPLEX"

        # --- Domain interaction analysis ---
        if domain_labels is not None:
            inter_gap_sum = 0.0
            intra_gap_sum = 0.0
            domain_lock_counts = Counter()

            for ei, edge in enumerate(edge_list):
                i, j = edge
                d_i, d_j = int(domain_labels[i]), int(domain_labels[j])
                imp = abs(delta_gaps[ei])
                if d_i != d_j:
                    inter_gap_sum += imp
                    key = f"{min(d_i,d_j)}_{max(d_i,d_j)}"
                    if ei < top_k:  # Only count top-k as lock interactions
                        domain_lock_counts[key] += 1
                else:
                    intra_gap_sum += imp

            total_gap_sum = inter_gap_sum + intra_gap_sum
            sr.inter_domain_gap_fraction = (
                inter_gap_sum / total_gap_sum if total_gap_sum > 0 else 0.0)
            # Modularity: low inter-domain fraction = modular
            sr.modularity_score = 1.0 - sr.inter_domain_gap_fraction
            sr.domain_lock_matrix = dict(domain_lock_counts)

        # --- Functional annotation enrichment ---
        func_all = set()
        func_categories = {}

        if functional_annotation is not None:
            # Accept FunctionalAnnotation object or plain dict
            if hasattr(functional_annotation, 'all_functional'):
                func_all = functional_annotation.all_functional
                func_categories = {
                    "mechanical": functional_annotation.mechanical,
                    "chemical": functional_annotation.chemical,
                    "signalling": functional_annotation.signalling,
                    "active_site": functional_annotation.active_site,
                    "binding": functional_annotation.binding,
                    "domain_boundary": functional_annotation.domain_boundary,
                    "mutagenesis": functional_annotation.mutagenesis,
                    "regulatory": functional_annotation.regulatory,
                }
            elif isinstance(functional_annotation, dict):
                # D96-format dict
                for cat, residues in functional_annotation.items():
                    cat_set = set(r for r in residues if r < N)
                    func_categories[cat] = cat_set
                    func_all.update(cat_set)

            func_all = set(r for r in func_all if r < N)

            if func_all:
                # Overall enrichment (Fisher's exact test)
                lock_set = set(sr.lock_residues)
                a = len(lock_set & func_all)
                b = len(lock_set - func_all)
                c = len(func_all - lock_set)
                d = N - len(lock_set | func_all)
                _, p_val = fisher_exact([[a, b], [c, d]],
                                        alternative='greater')
                expected = len(lock_set) * len(func_all) / N
                sr.enrichment_all = (a / expected
                                     if expected > 0 else float('inf'))
                sr.enrichment_p = float(p_val)

                # Per-category enrichment
                cat_results = {}
                for cat, cat_set in func_categories.items():
                    if not cat_set:
                        continue
                    ca = len(lock_set & cat_set)
                    cb = len(lock_set - cat_set)
                    cc = len(cat_set - lock_set)
                    cd = N - len(lock_set | cat_set)
                    _, cp = fisher_exact([[ca, cb], [cc, cd]],
                                         alternative='greater')
                    c_exp = len(lock_set) * len(cat_set) / N
                    cat_results[cat] = {
                        "observed": ca,
                        "expected": round(c_exp, 2),
                        "enrichment": round(ca / c_exp if c_exp > 0 else 0, 2),
                        "p_value": round(float(cp), 4),
                        "n_category": len(cat_set),
                    }
                sr.enrichment_by_category = cat_results

                # Importance ratio (functional vs non-functional)
                func_list = sorted(func_all)
                nonfunc_list = [i for i in range(N) if i not in func_all]
                if func_list and nonfunc_list:
                    mean_f = float(np.mean(residue_importance[func_list]))
                    mean_nf = float(np.mean(residue_importance[nonfunc_list]))
                    sr.importance_ratio = mean_f / mean_nf if mean_nf > 0 else 0
                    func_mask = np.zeros(N)
                    for r in func_list:
                        func_mask[r] = 1.0
                    rho, p_rho = spearmanr(residue_importance, func_mask)
                    sr.importance_rho = float(rho) if not np.isnan(rho) else 0.0
                    sr.importance_p = float(p_rho) if not np.isnan(p_rho) else 1.0

                # --- Mechanical vs chemical vs signalling ---
                mech_set = func_categories.get("mechanical",
                           func_categories.get("hinge",
                           func_categories.get("domain_boundary", set())))
                chem_set = func_categories.get("chemical",
                           func_categories.get("active_site", set()))
                sig_set = func_categories.get("signalling",
                          func_categories.get("allosteric",
                          func_categories.get("regulatory", set())))

                sr.mechanical_locks = []
                sr.chemical_locks = []
                sr.signalling_locks = []
                sr.unclassified_locks = []

                for edge in sr.lock_edges:
                    i, j = edge
                    edge_residues = {i, j}
                    if edge_residues & mech_set:
                        sr.mechanical_locks.append(edge)
                    elif edge_residues & chem_set:
                        sr.chemical_locks.append(edge)
                    elif edge_residues & sig_set:
                        sr.signalling_locks.append(edge)
                    else:
                        sr.unclassified_locks.append(edge)

                n_classified = (len(sr.mechanical_locks) +
                                len(sr.chemical_locks) +
                                len(sr.signalling_locks) +
                                len(sr.unclassified_locks))
                if n_classified > 0:
                    sr.mechanical_fraction = len(sr.mechanical_locks) / n_classified
                    sr.chemical_fraction = len(sr.chemical_locks) / n_classified
                    sr.signalling_fraction = len(sr.signalling_locks) / n_classified

        sr.time_total = time.perf_counter() - t0
        return sr

    @staticmethod
    def _compute_spectral_time(evals_a: np.ndarray,
                                evals_b: np.ndarray,
                                N: int) -> Dict[str, float]:
        """Compute spectral distance metrics between two eigenvalue spectra.

        From D84: Fiedler gap Δ (ρ=0.633) and log-spectral (ρ=0.55) are
        the best predictors of conformational change magnitude (RMSD).
        """
        K = min(len(evals_a) - 1, len(evals_b) - 1, N // 3)
        la = evals_a[1:K + 1]
        lb = evals_b[1:K + 1]

        sa, sb = la.sum(), lb.sum()
        la_n = la / sa if sa > 0 else la
        lb_n = lb / sb if sb > 0 else lb

        results = {}

        # Fiedler gap delta
        gap_a = float(evals_a[1] / evals_a[2]) if len(evals_a) > 2 and evals_a[2] > 1e-10 else 0.0
        gap_b = float(evals_b[1] / evals_b[2]) if len(evals_b) > 2 and evals_b[2] > 1e-10 else 0.0
        results['fiedler_gap_delta'] = abs(gap_a - gap_b)

        # Normalised spectral distance
        results['spectral_dist_norm'] = float(np.sqrt(np.sum((la_n - lb_n) ** 2)))

        # Weighted spectral distance (1/k)
        weights = 1.0 / np.arange(1, K + 1)
        results['spectral_dist_weighted'] = float(np.sqrt(np.sum(weights * (la_n - lb_n) ** 2)))

        # Log-spectral distance
        mask = (la > 1e-10) & (lb > 1e-10)
        if mask.sum() > 0:
            results['spectral_dist_log'] = float(np.sqrt(np.sum(
                (np.log(la[mask]) - np.log(lb[mask])) ** 2)))
        else:
            results['spectral_dist_log'] = 0.0

        # Wasserstein
        results['wasserstein'] = float(_wasserstein(la_n, lb_n))

        # Spectral entropy delta
        def _entropy(vals):
            p = vals / vals.sum() if vals.sum() > 0 else vals
            p = p[p > 0]
            return -float(np.sum(p * np.log(p)))
        results['entropy_delta'] = abs(_entropy(la) - _entropy(lb))

        # JS divergence (symmetrised KL)
        eps = 1e-12
        pa = la_n + eps
        pb = lb_n + eps
        pa /= pa.sum()
        pb /= pb.sum()
        kl_ab = float(np.sum(pa * np.log(pa / pb)))
        kl_ba = float(np.sum(pb * np.log(pb / pa)))
        results['spectral_JS'] = 0.5 * (kl_ab + kl_ba)

        return results

    @staticmethod
    def _kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
        """Kabsch-aligned RMSD between two coordinate sets."""
        p = P - P.mean(axis=0)
        q = Q - Q.mean(axis=0)
        H = p.T @ q
        U, S, Vt = np.linalg.svd(H)
        d = np.linalg.det(Vt.T @ U.T)
        R = Vt.T @ np.diag([1, 1, np.sign(d)]) @ U.T
        return float(np.sqrt(np.mean(np.sum((p - q @ R.T) ** 2, axis=1))))

    # ------------------------------------------------------------------
    # Disturber-integrated metrics (D88)
    # ------------------------------------------------------------------
    @staticmethod
    def _hotspot_weighted_spectral(
            evals_a: np.ndarray, evals_b: np.ndarray,
            result_a: 'IBPResult', result_b: 'IBPResult',
            hotspot_set: set, N: int) -> float:
        """Log-spectral distance weighted by hotspot-mode coupling.

        For each eigenmode k, compute how much its eigenvector
        concentrates on hotspot residues (sum of squared components
        at hotspot indices).  Use this as a weight on the log-spectral
        distance, so modes that "talk through" hotspot residues
        contribute more to the conformational distance.

        This bridges probe/listen (spatial per-residue channel) with
        compare (global spectral channel).
        """
        K = min(len(evals_a) - 1, len(evals_b) - 1, N // 3)
        la = evals_a[1:K + 1]
        lb = evals_b[1:K + 1]

        mask = (la > 1e-10) & (lb > 1e-10)
        if mask.sum() == 0:
            return 0.0

        # Compute hotspot coupling per mode
        evecs_a = result_a.laplacian_eigenvectors
        evecs_b = result_b.laplacian_eigenvectors
        if evecs_a is None or evecs_b is None:
            # Fall back to unweighted
            return float(np.sqrt(np.sum(
                (np.log(la[mask]) - np.log(lb[mask])) ** 2)))

        hotspot_idx = np.array(sorted(hotspot_set), dtype=int)
        weights = np.ones(K)
        for k in range(K):
            mode_idx = k + 1  # skip zero mode
            if mode_idx < evecs_a.shape[1] and mode_idx < evecs_b.shape[1]:
                # Average hotspot participation from both states
                ha = np.sum(evecs_a[hotspot_idx, mode_idx] ** 2)
                hb = np.sum(evecs_b[hotspot_idx, mode_idx] ** 2)
                hotspot_participation = (ha + hb) / 2.0
                # Normalise by total participation (should sum to 1)
                total = (np.sum(evecs_a[:, mode_idx] ** 2) +
                         np.sum(evecs_b[:, mode_idx] ** 2)) / 2.0
                if total > 1e-10:
                    weights[k] = 1.0 + hotspot_participation / total
                    # Range: [1, 2] — modes through hotspots get up to 2x weight

        log_diff = np.zeros(K)
        log_diff[mask] = (np.log(la[mask]) - np.log(lb[mask])) ** 2
        return float(np.sqrt(np.sum(weights * log_diff)))

    @staticmethod
    def _compute_functional_overlap(
            message: 'DisturbanceMessage',
            functional_residues: List[int],
            N: int) -> 'FunctionalSiteOverlap':
        """Compute enrichment of disturber hotspots at functional sites.

        Uses the hypergeometric distribution to test whether hotspot
        residues overlap with known functional residues more than
        expected by chance.
        """
        from scipy.stats import hypergeom

        func_set = set(functional_residues)
        hot_set = set(message.hotspot_residues or [])
        overlap = func_set & hot_set

        ov = FunctionalSiteOverlap()
        ov.n_residues = N
        ov.n_functional = len(func_set)
        ov.n_hotspots = len(hot_set)
        ov.n_overlap = len(overlap)
        ov.overlap_residues = sorted(overlap)

        ov.hotspot_rate = len(hot_set) / N if N > 0 else 0.0
        ov.functional_rate = len(func_set) / N if N > 0 else 0.0
        ov.overlap_rate = (len(overlap) / len(func_set)
                           if len(func_set) > 0 else 0.0)
        ov.precision = (len(overlap) / len(hot_set)
                        if len(hot_set) > 0 else 0.0)

        # Enrichment ratio
        expected_rate = ov.hotspot_rate  # background rate of being a hotspot
        ov.enrichment = (ov.overlap_rate / expected_rate
                         if expected_rate > 0 else 0.0)

        # Hypergeometric test: P(X >= n_overlap)
        # Population = N, success states = n_functional, draws = n_hotspots
        ov.p_value = float(hypergeom.sf(
            len(overlap) - 1, N, len(func_set), len(hot_set)))

        # Mean |entropy_delta| at functional vs other sites
        if message.entropy_delta is not None:
            abs_delta = np.abs(message.entropy_delta)
            func_idx = np.array(sorted(func_set), dtype=int)
            other_idx = np.array([i for i in range(N) if i not in func_set],
                                 dtype=int)
            if len(func_idx) > 0:
                ov.mean_abs_delta_func = float(np.mean(abs_delta[func_idx]))
            if len(other_idx) > 0:
                ov.mean_abs_delta_other = float(np.mean(abs_delta[other_idx]))
            if ov.mean_abs_delta_other > 0:
                ov.delta_ratio = ov.mean_abs_delta_func / ov.mean_abs_delta_other

        return ov

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_contacts(self, coords, N):
        contacts = {}
        degrees = Counter()
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(coords[i] - coords[j])
                if d <= self.cutoff:
                    contacts[(i, j)] = d
                    degrees[i] += 1
                    degrees[j] += 1
        for i in range(N):
            if i not in degrees:
                degrees[i] = 0
        return contacts, degrees

    @staticmethod
    def _build_laplacian(N, contacts):
        L = np.zeros((N, N))
        for (i, j) in contacts:
            L[i, j] = -1
            L[j, i] = -1
            L[i, i] += 1
            L[j, j] += 1
        return L

    @staticmethod
    def _norm01(x):
        rng = x.max() - x.min()
        return (x - x.min()) / (rng + 1e-10) if rng > 0 else np.zeros_like(x)

    def _continuous_stabilizer(self, coords, degrees, N):
        deg_arr = np.array([degrees[i] for i in range(N)])
        com = coords.mean(axis=0)
        d_com = np.linalg.norm(coords - com, axis=1)
        dmat = squareform(pdist(coords))
        local_dens = (dmat < 6.0).sum(axis=1) - 1
        deg_n = self._norm01(deg_arr)
        com_n = self._norm01(d_com)
        dens_n = self._norm01(local_dens.astype(float))
        composite = 0.5 * deg_n + 0.3 * dens_n + 0.2 * (1 - com_n)
        return self.s_min + (self.s_max - self.s_min) * (1 - composite)

    def _laplacian_stabilizer(self, evecs_L, N):
        k = min(4, N)
        spec_coords = evecs_L[:, 1:k]
        spec_com = spec_coords.mean(axis=0)
        spec_dist = np.linalg.norm(spec_coords - spec_com, axis=1)
        spec_norm = self._norm01(spec_dist)
        return self.s_min + (self.s_max - self.s_min) * spec_norm

    def _solve_enm(self, N, contacts, s_eff, b_scale):
        G_k = np.zeros((N, N))
        for (i, j), d in contacts.items():
            si, sj = s_eff[i], s_eff[j]
            w = 2 * si * sj / (si + sj) / self.g_size * (self.cutoff / d) ** 2
            G_k[i, j] = -w
            G_k[j, i] = -w
            G_k[i, i] += w
            G_k[j, j] += w
        evals, evecs = np.linalg.eigh(G_k)
        b_pred = np.zeros(N)
        for k in range(1, len(evals)):
            if evals[k] > 1e-10:
                b_pred += evecs[:, k]**2 / evals[k]
        if b_pred.max() > 0:
            b_pred = b_pred / b_pred.max() * b_scale
        return b_pred

    @staticmethod
    def _decompose_domains(fiedler, N, eigenvalues=None):
        """Split residues into domains using Fiedler sign.

        Also reports the spectral gap (λ₂/λ₃) as a confidence measure:
        large gap → clear 2-domain split; small gap → likely single domain.
        """
        labels = (fiedler >= 0).astype(int)
        n_domains = len(set(labels))

        # Spectral gap confidence: λ₂/λ₃
        # If λ₂ << λ₃, the 2-way split is well-separated
        spectral_gap = None
        if eigenvalues is not None and len(eigenvalues) > 2:
            lam2 = eigenvalues[1]
            lam3 = eigenvalues[2]
            if lam3 > 1e-10:
                spectral_gap = float(lam2 / lam3)

        return labels, n_domains, spectral_gap

    @staticmethod
    def _spectral_cluster(eigenvalues, eigenvectors, N, max_k=8,
                          silhouette_threshold=0.15):
        """k-way spectral clustering using silhouette-based k selection.

        Replaced eigengap heuristic (D79: 20% k-accuracy) with silhouette
        scoring on spectral embeddings (D81: 87% k-accuracy on CATH benchmark).

        Algorithm:
        1. Compute eigengaps for diagnostic purposes
        2. For each candidate k in 2..max_k:
           a. Embed residues into k-dimensional NJW-normalised spectral space
           b. Run k-means, compute silhouette score on the embedding
        3. Pick k with highest silhouette score
        4. Fall back to k=1 if best silhouette < threshold

        Returns (labels, k, eigengaps).
        """
        # Eigengaps (diagnostic, kept for IBPResult compatibility)
        n_eig = min(max_k + 1, len(eigenvalues) - 1)
        eigengaps = np.diff(eigenvalues[1:n_eig + 1])

        if len(eigengaps) == 0 or N < 10:
            return np.zeros(N, dtype=int), 1, np.array([])

        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
        except ImportError:
            # Fallback to eigengap if sklearn unavailable
            if len(eigengaps) > 0:
                k = int(np.argmax(eigengaps)) + 1
                k = max(1, min(k, max_k, N // 5))
            else:
                k = 1
            if k <= 1:
                return np.zeros(N, dtype=int), 1, eigengaps
            X = eigenvectors[:, 1:k + 1].copy()
            labels = np.argmax(np.abs(X), axis=1)
            return labels.astype(int), int(k), eigengaps

        best_k = 1
        best_sil = -1.0
        best_labels = np.zeros(N, dtype=int)
        all_candidates = {}  # k -> (labels, sil)

        upper = min(max_k + 1, N // 5)
        for k in range(2, upper):
            # NJW-normalised spectral embedding
            X = eigenvectors[:, 1:k + 1].copy()
            row_norms = np.linalg.norm(X, axis=1, keepdims=True)
            row_norms[row_norms < 1e-10] = 1.0
            X = X / row_norms

            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = km.fit_predict(X)

            try:
                sil = silhouette_score(X, labels)
            except Exception:
                sil = -1.0

            all_candidates[k] = (labels.astype(int), sil)
            if sil > best_sil:
                best_sil = sil
                best_k = k
                best_labels = labels.astype(int)

        # Only accept clustering if silhouette exceeds threshold
        if best_sil < silhouette_threshold:
            return np.zeros(N, dtype=int), 1, eigengaps

        return best_labels, int(best_k), eigengaps

    @staticmethod
    def _detect_hinges(fiedler, N, window: int = 3):
        """Detect hinge residues from Fiedler sign-change boundaries.

        Hinge score = |Δ fiedler| at sign-change boundaries, smoothed.
        """
        # Raw gradient magnitude
        grad = np.abs(np.gradient(fiedler))
        # Smooth
        kernel = np.ones(2 * window + 1) / (2 * window + 1)
        hinge_scores = np.convolve(grad, kernel, mode='same')
        hinge_scores /= (hinge_scores.max() + 1e-10)

        # Find sign-change boundaries
        signs = np.sign(fiedler)
        sign_changes = []
        for i in range(1, N):
            if signs[i] != signs[i - 1] and signs[i] != 0 and signs[i - 1] != 0:
                sign_changes.append(i)

        # Hinge residues = peaks near sign changes
        hinge_residues = []
        for sc in sign_changes:
            lo = max(0, sc - window)
            hi = min(N, sc + window + 1)
            best = lo + np.argmax(hinge_scores[lo:hi])
            hinge_residues.append(int(best))

        return hinge_scores, sorted(set(hinge_residues))

    # ------------------------------------------------------------------
    # Debris recycling (D84)
    # ------------------------------------------------------------------
    def _recycle_debris(self, result: 'IBPResult', coords: np.ndarray,
                        evals: np.ndarray, evecs: np.ndarray,
                        domain_labels: np.ndarray, N: int,
                        max_recycle_modes: int = 5) -> None:
        """Domain-localised analysis of higher eigenmodes.

        D84 demonstrated that ~96% of higher-mode sign changes fall
        within Fiedler-identified domains. This method recycles that
        "debris" as sub-domain structural signal:

        1. For each Fiedler domain, build a sub-Laplacian and extract
           internal spectral structure (sub-domains, internal hinges).
        2. Classify higher-mode (global) sign changes as within-domain
           or cross-domain.
        3. Within-domain sign changes become refined hinge candidates;
           cross-domain sign changes are discarded as true debris.

        Results are written directly into the IBPResult.
        """
        unique_domains = sorted(set(domain_labels.tolist()))
        if len(unique_domains) < 2:
            # Single-domain protein — sub-Laplacian IS the Laplacian;
            # still recycle higher modes for sub-domain structure.
            unique_domains = [0]

        # --- Per-domain sub-analysis ---
        domain_details = []
        global_subdomain_labels = np.zeros(N, dtype=int)
        subdomain_counter = 0

        for dom_id in unique_domains:
            idx = np.where(domain_labels == dom_id)[0]
            dd = self._domain_sub_analysis(coords, idx, dom_id)

            # Map sub-domain labels back to global residue indices
            if dd.subdomain_labels is not None:
                for i_local, i_global in enumerate(idx):
                    global_subdomain_labels[i_global] = (
                        subdomain_counter + dd.subdomain_labels[i_local]
                    )
                subdomain_counter += dd.n_subdomains
            else:
                for i_global in idx:
                    global_subdomain_labels[i_global] = subdomain_counter
                subdomain_counter += 1

            domain_details.append(dd)

        # --- Classify global higher-mode sign changes ---
        all_recycled_hinges = []
        recycled_scores = np.zeros(N)
        total_within = 0
        total_cross = 0

        upper_mode = min(max_recycle_modes + 1, N // 5, len(evals))
        for mode_k in range(2, upper_mode + 1):
            within, cross = self._classify_sign_changes(
                evecs[:, mode_k], domain_labels, N)
            total_within += len(within)
            total_cross += len(cross)

            # Within-domain sign changes → hinge candidates
            # Weight by eigenvalue (lower modes = stronger signal)
            weight = 1.0 / mode_k
            for pos in within:
                recycled_scores[pos] += weight

        # Normalise recycled scores
        if recycled_scores.max() > 0:
            recycled_scores /= recycled_scores.max()

        # Merge with Fiedler hinges: recycled positions that are NOT
        # already Fiedler hinges are new sub-domain hinge candidates.
        # Use a high threshold + top-N cap to keep precision.
        fiedler_hinge_set = set(result.hinge_residues or [])
        recycled_threshold = 0.6  # high bar: only strong sign-change accumulators
        max_recycled = max(2, len(fiedler_hinge_set))  # at most as many as Fiedler found

        # Rank all candidates by score, then apply threshold + cap
        candidates = []
        for i in range(N):
            if recycled_scores[i] > recycled_threshold and i not in fiedler_hinge_set:
                candidates.append((recycled_scores[i], i))
        # Also add sub-analysis hinges, but only if they pass the score gate
        for dd in domain_details:
            if dd.recycled_hinges:
                for local_pos in dd.recycled_hinges:
                    global_pos = int(dd.residue_indices[local_pos])
                    if (global_pos not in fiedler_hinge_set and
                            recycled_scores[global_pos] > recycled_threshold * 0.5):
                        candidates.append((recycled_scores[global_pos], global_pos))

        # De-duplicate by position, keep best score
        seen = {}
        for score, pos in candidates:
            if pos not in seen or score > seen[pos]:
                seen[pos] = score
        ranked = sorted(seen.items(), key=lambda x: -x[1])
        all_recycled_hinges = [pos for pos, _ in ranked[:max_recycled]]

        all_recycled_hinges = sorted(set(all_recycled_hinges))

        total_sc = total_within + total_cross
        recycled_fraction = total_within / total_sc if total_sc > 0 else 0.0

        # --- Write results ---
        result.domain_details = domain_details
        result.subdomain_labels = global_subdomain_labels
        result.n_subdomains_total = subdomain_counter
        result.recycled_hinges = all_recycled_hinges
        result.recycled_hinge_scores = recycled_scores
        result.debris_recycled_fraction = round(recycled_fraction, 4)

    def _domain_sub_analysis(self, coords: np.ndarray,
                             indices: np.ndarray,
                             domain_id: int) -> 'DomainDetail':
        """Build sub-Laplacian for a single domain and extract internals.

        Parameters
        ----------
        coords : (N, 3) full coordinate array
        indices : residue indices belonging to this domain
        domain_id : integer label for this domain

        Returns
        -------
        DomainDetail with internal spectral structure.
        """
        dd = DomainDetail(domain_id=domain_id)
        idx = np.asarray(indices)
        dd.residue_indices = idx
        n_sub = len(idx)
        dd.n_residues = n_sub

        if n_sub < 5:
            dd.subdomain_labels = np.zeros(n_sub, dtype=int)
            dd.n_subdomains = 1
            return dd

        # Build sub-contact map and sub-Laplacian
        sub_coords = coords[idx]
        sub_contacts = {}
        for ii in range(n_sub):
            for jj in range(ii + 1, n_sub):
                d = np.linalg.norm(sub_coords[ii] - sub_coords[jj])
                if d <= self.cutoff:
                    sub_contacts[(ii, jj)] = d

        dd.n_contacts = len(sub_contacts)
        if dd.n_contacts < n_sub // 3:
            dd.subdomain_labels = np.zeros(n_sub, dtype=int)
            dd.n_subdomains = 1
            return dd

        L_sub = self._build_laplacian(n_sub, sub_contacts)
        evals_sub, evecs_sub = np.linalg.eigh(L_sub)
        dd.eigenvalues = evals_sub[1:min(11, len(evals_sub))]  # first 10 non-zero

        # Sub-domain spectral gap
        if len(evals_sub) > 2 and evals_sub[2] > 1e-10:
            dd.spectral_gap = float(evals_sub[1] / evals_sub[2])

        # Sub-Fiedler
        if len(evecs_sub[0]) > 1:
            dd.fiedler = evecs_sub[:, 1]

        # Sub-domain clustering (silhouette on sub-Laplacian)
        sub_labels, sub_k, _ = self._spectral_cluster(
            evals_sub, evecs_sub, n_sub, max_k=4,
            silhouette_threshold=0.20)  # slightly higher bar for sub-domains
        dd.subdomain_labels = sub_labels
        dd.n_subdomains = sub_k

        # Internal hinge detection from sub-Fiedler
        if dd.fiedler is not None and n_sub > 6:
            _, hinges = self._detect_hinges(dd.fiedler, n_sub, window=2)
            dd.recycled_hinges = hinges

        # Classify sub-Fiedler higher modes within this domain
        n_within = 0
        n_cross = 0
        if dd.fiedler is not None and len(evals_sub) > 2:
            sub_domain_labels = (dd.fiedler >= 0).astype(int)
            for mk in range(2, min(4, n_sub // 5)):
                if mk >= len(evals_sub):
                    break
                w, c = self._classify_sign_changes(
                    evecs_sub[:, mk], sub_domain_labels, n_sub)
                n_within += len(w)
                n_cross += len(c)
        dd.n_recycled_sign_changes = n_within
        dd.n_discarded_cross_domain = n_cross

        return dd

    @staticmethod
    def _classify_sign_changes(eigenvector: np.ndarray,
                               domain_labels: np.ndarray,
                               N: int) -> Tuple[List[int], List[int]]:
        """Classify sign changes as within-domain or cross-domain.

        Parameters
        ----------
        eigenvector : eigenvector to analyse
        domain_labels : per-residue domain assignments
        N : number of residues

        Returns
        -------
        (within_domain_positions, cross_domain_positions)
        """
        signs = np.sign(eigenvector)
        within = []
        cross = []
        for i in range(1, N):
            if signs[i] != signs[i - 1] and signs[i] != 0 and signs[i - 1] != 0:
                if domain_labels[i] == domain_labels[i - 1]:
                    within.append(i)
                else:
                    cross.append(i)
        return within, cross
