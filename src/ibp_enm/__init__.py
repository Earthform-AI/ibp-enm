"""IBP-ENM: Spectral Elastic Network Model for Protein Structural Analysis.

Provides per-residue structural role profiles, unsupervised domain detection,
hinge identification, and multi-perspective flexibility prediction — all from
a single static structure with no training data.

The **Thermodynamic Band** (D109) extends the core analyser with a 7-instrument
carving protocol that achieves 83% archetype accuracy on the 12-protein benchmark,
with 100% barrel detection and zero false reclassifications.
"""
from .analyzer import IBPProteinAnalyzer, IBPResult
from .fetch import fetch_pdb_ca_data, search_rcsb
from .baselines import gnm_predict

# D109 Thermodynamic Band — public API
from .band import ThermodynamicBand, run_single_protein
from .instruments import ThermoReactionProfile, ThermoInstrumentCarver, steps_for_protein
from .synthesis import MetaFickBalancer, EnzymeLensSynthesis, HingeLensSynthesis, SizeAwareHingeLens
from .archetypes import (
    ProteinArchetype, ARCHETYPES, ARCHETYPE_EXPECTATIONS,
    SurgeonsHandbook, PROTEINS, GROUND_TRUTH,
)
from .carving import CarvingIntent, ReactionSignature, FickBalancer
from .thermodynamics import (
    vibrational_entropy, heat_capacity, helmholtz_free_energy,
    mean_ipr_low_modes, spectral_entropy_shannon,
    per_residue_entropy_contribution, entropy_asymmetry_score,
    multimode_ipr, hinge_occupation_ratio, domain_stiffness_asymmetry,
)

# Profile caching & benchmark (v0.3.0)
from .cache import ProfileCache, profile_to_dict, profile_from_dict
from .benchmark import (
    ProteinEntry, ProteinResult, BenchmarkReport, BenchmarkRunner,
    ORIGINAL_CORPUS, EXPANDED_CORPUS,
)

__all__ = [
    # Core analyser
    "IBPProteinAnalyzer", "IBPResult",
    "fetch_pdb_ca_data", "search_rcsb",
    "gnm_predict",
    # Thermodynamic Band (D109)
    "ThermodynamicBand", "run_single_protein",
    "ThermoReactionProfile", "ThermoInstrumentCarver",
    "MetaFickBalancer", "EnzymeLensSynthesis", "HingeLensSynthesis", "SizeAwareHingeLens",
    "ProteinArchetype", "ARCHETYPES", "ARCHETYPE_EXPECTATIONS",
    "SurgeonsHandbook", "PROTEINS", "GROUND_TRUTH",
    "CarvingIntent", "ReactionSignature", "FickBalancer",
    "vibrational_entropy", "heat_capacity", "helmholtz_free_energy",
    "mean_ipr_low_modes", "spectral_entropy_shannon",
    "per_residue_entropy_contribution", "entropy_asymmetry_score",
    "multimode_ipr", "hinge_occupation_ratio", "domain_stiffness_asymmetry",
    # Profile caching & benchmark (v0.3.0)
    "ProfileCache", "profile_to_dict", "profile_from_dict",
    "ProteinEntry", "ProteinResult", "BenchmarkReport", "BenchmarkRunner",
    "ORIGINAL_CORPUS", "EXPANDED_CORPUS",
]
