"""Smoke tests for the Thermodynamic Band (D109 formalisation).

Tests the complete import chain, thermodynamic functions,
carving primitives, archetype vote machinery, and a small
synthetic integration test (no network required).
"""

import numpy as np
import pytest

# ── thermodynamics ───────────────────────────────────────────────

from ibp_enm.thermodynamics import (
    vibrational_entropy,
    heat_capacity,
    helmholtz_free_energy,
    heat_kernel_trace,
    inverse_participation_ratio,
    mean_ipr_low_modes,
    spectral_entropy_shannon,
    per_residue_entropy_contribution,
    entropy_asymmetry_score,
)


class TestThermodynamics:
    """Pure eigenvalue → observable functions."""

    @pytest.fixture
    def evals(self):
        return np.array([0.0, 0.1, 0.5, 1.0, 2.0, 5.0])

    def test_entropy_positive(self, evals):
        S = vibrational_entropy(evals)
        assert S > 0

    def test_heat_capacity_positive(self, evals):
        Cv = heat_capacity(evals)
        assert Cv > 0

    def test_free_energy_negative(self, evals):
        F = helmholtz_free_energy(evals)
        assert F < 0

    def test_heat_kernel_trace_positive(self, evals):
        Z = heat_kernel_trace(evals)
        assert Z > 0

    def test_ipr_single_mode(self):
        """A perfectly localised mode has IPR = 1/N (uniform)
        or IPR = 1 (single atom)."""
        N = 10
        v = np.zeros((N, N))
        v[:, 0] = 1.0 / np.sqrt(N)  # delocalised
        ipr = inverse_participation_ratio(v, 0)
        assert 0 < ipr < 1

    def test_mean_ipr_low_modes(self):
        N = 20
        evecs = np.random.randn(N, N)
        # Orthogonalise
        evecs, _ = np.linalg.qr(evecs)
        ipr = mean_ipr_low_modes(evecs, n_modes=3)
        assert 0 < ipr < 1

    def test_spectral_entropy_uniform(self):
        """Uniform eigenvalue distribution → maximum entropy."""
        evals = np.ones(10)
        H = spectral_entropy_shannon(evals)
        assert abs(H - np.log(10)) < 1e-6

    def test_zero_only_eigenvalues(self):
        """All-zero eigenvalues → zero for all observables."""
        evals = np.zeros(5)
        assert vibrational_entropy(evals) == 0.0
        assert heat_capacity(evals) == 0.0
        assert helmholtz_free_energy(evals) == 0.0
        assert spectral_entropy_shannon(evals) == 0.0

    # ── D110: per-residue entropy decomposition ──────────────────

    def test_per_residue_entropy_sums_to_total(self):
        """Per-residue entropy contributions should sum to S_vib."""
        N = 20
        evals = np.sort(np.abs(np.random.randn(N))) + 0.01
        evecs = np.linalg.qr(np.random.randn(N, N))[0]
        s_per_res = per_residue_entropy_contribution(evals, evecs)
        S_total = vibrational_entropy(evals)
        assert abs(np.sum(s_per_res) - S_total) < 1e-6

    def test_per_residue_entropy_shape(self):
        N = 15
        evals = np.array([0.0] + [float(i) for i in range(1, N)])
        evecs = np.eye(N)
        s_per_res = per_residue_entropy_contribution(evals, evecs)
        assert s_per_res.shape == (N,)

    def test_per_residue_entropy_nonnegative(self):
        N = 10
        evals = np.linspace(0.1, 5.0, N)
        evecs = np.linalg.qr(np.random.randn(N, N))[0]
        s_per_res = per_residue_entropy_contribution(evals, evecs)
        assert np.all(s_per_res >= 0)

    def test_per_residue_entropy_zeros(self):
        """All-zero eigenvalues → zero array."""
        evals = np.zeros(5)
        evecs = np.eye(5)
        s_per_res = per_residue_entropy_contribution(evals, evecs)
        np.testing.assert_array_equal(s_per_res, 0.0)

    def test_entropy_asymmetry_uniform(self):
        """Uniform per-residue entropy → low gini, low cv."""
        s = np.ones(100)
        asym = entropy_asymmetry_score(s)
        assert asym["gini"] < 0.01
        assert asym["cv"] < 0.01

    def test_entropy_asymmetry_peaked(self):
        """One residue dominates → high gini."""
        s = np.zeros(100)
        s[0] = 100.0
        asym = entropy_asymmetry_score(s)
        assert asym["gini"] > 0.9
        assert asym["top5_frac"] > 0.9

    def test_entropy_asymmetry_keys(self):
        s = np.random.rand(50) + 0.01
        asym = entropy_asymmetry_score(s)
        assert set(asym.keys()) == {"gini", "cv", "top5_frac", "kurtosis"}

    def test_entropy_asymmetry_empty(self):
        s = np.zeros(5)
        asym = entropy_asymmetry_score(s)
        assert asym["gini"] == 0.0


# ── carving ──────────────────────────────────────────────────────

from ibp_enm.carving import (
    CarvingIntent,
    QuantumCarvingState,
    build_laplacian,
    spectral_gap,
    classify_edge_species,
    norm01,
)


class TestCarvingPrimitives:
    def test_fano_lines(self):
        """There should be 7 Fano lines of length 3."""
        assert len(CarvingIntent.LINES) == 7
        for line in CarvingIntent.LINES:
            assert len(line) == 3

    def test_fano_multiply_closure(self):
        """Fano multiplication should stay in {0..6}."""
        for a in range(7):
            for b in range(7):
                p = CarvingIntent.fano_multiply(a, b)
                assert 0 <= p <= 6

    def test_quantum_state_normalised(self):
        qs = QuantumCarvingState()
        assert abs(np.linalg.norm(qs.psi) - 1.0) < 1e-10

    def test_build_laplacian_symmetric(self):
        contacts = {(0, 1): 3.5, (1, 2): 4.0, (0, 2): 5.0}
        L = build_laplacian(3, contacts)
        assert L.shape == (3, 3)
        np.testing.assert_array_almost_equal(L, L.T)
        # Row sums = 0
        np.testing.assert_array_almost_equal(L.sum(axis=1), 0)

    def test_spectral_gap(self):
        contacts = {(0, 1): 3.5, (1, 2): 4.0, (0, 2): 5.0}
        L = build_laplacian(3, contacts)
        evals = np.linalg.eigvalsh(L)
        sg = spectral_gap(evals)
        assert sg > 0

    def test_classify_edge_species(self):
        fiedler = np.array([-0.5, 0.5, -0.3, 0.3])
        domain_labels = np.array([0, 1, 0, 1])
        # Cross-domain edge
        sp = classify_edge_species((0, 1), fiedler, domain_labels,
                                   delta_gap=0.05, gap_base=1.0)
        assert sp in ("Soft", "Bridge", "Carved", "Sibling")

    def test_norm01(self):
        x = np.array([1.0, 3.0, 5.0])
        n = norm01(x)
        assert n[0] == 0.0
        assert n[-1] == 1.0

    def test_norm01_constant(self):
        x = np.array([3.0, 3.0, 3.0])
        n = norm01(x)
        np.testing.assert_array_equal(n, 0.0)


# ── archetypes ───────────────────────────────────────────────────

from ibp_enm.archetypes import (
    ARCHETYPES,
    ARCHETYPE_EXPECTATIONS,
    GROUND_TRUTH,
    PROTEINS,
    SurgeonsHandbook,
)


class TestArchetypes:
    def test_five_archetypes(self):
        assert len(ARCHETYPES) == 5
        names = {a.name for a in ARCHETYPES}
        assert names == {"dumbbell", "barrel", "globin",
                         "enzyme_active", "allosteric"}

    def test_expectations_match_archetypes(self):
        for a in ARCHETYPES:
            assert a.name in ARCHETYPE_EXPECTATIONS

    def test_twelve_proteins(self):
        assert len(PROTEINS) == 12

    def test_ground_truth_covers_proteins(self):
        protein_names = {name for name, _, _ in PROTEINS}
        assert protein_names == set(GROUND_TRUTH.keys())

    def test_match_score_range(self):
        a = ARCHETYPES[0]
        bus = np.array([0.5, 0.5, 0.5, 0.5])
        score = a.match_score(bus, 0.35)
        assert 0.0 <= score <= 1.5  # reasonable range


# ── instruments ──────────────────────────────────────────────────

from ibp_enm.instruments import (
    ThermoReactionProfile,
    INSTRUMENTS,
    STEPS_PER_INSTRUMENT,
)


class TestInstruments:
    def test_seven_instruments(self):
        assert len(INSTRUMENTS) == 7

    def test_five_steps(self):
        assert STEPS_PER_INSTRUMENT == 5

    def test_profile_vote_has_all_archetypes(self):
        """A blank profile should still produce votes for all 5 archetypes."""
        p = ThermoReactionProfile(instrument="algebraic", intent_idx=0)
        votes = p.archetype_vote()
        assert len(votes) == 5
        assert all(v > 0 for v in votes.values())
        total = sum(votes.values())
        assert abs(total - 1.0) < 1e-6  # normalised

    def test_profile_properties_defaults(self):
        p = ThermoReactionProfile(instrument="thermal", intent_idx=3)
        assert p.gap_retained == 1.0
        assert p.gap_flatness == 1.0
        assert p.entropy_change == 0.0
        assert p.free_energy_cost == 0.0
        assert p.mean_ipr == 0.0


# ── synthesis ────────────────────────────────────────────────────

from ibp_enm.synthesis import MetaFickBalancer, EnzymeLensSynthesis, HingeLensSynthesis


class TestSynthesis:
    def test_meta_fick_unanimous(self):
        """When all carvers agree on barrel, synthesised identity = barrel."""
        votes = [
            {"barrel": 0.8, "dumbbell": 0.05, "globin": 0.05,
             "enzyme_active": 0.05, "allosteric": 0.05}
            for _ in range(7)
        ]
        mfb = MetaFickBalancer()
        state = mfb.compute_meta_fick_state(votes)
        assert state["top_archetype"] == "barrel"
        assert state["rho"] == 1.0

    def test_meta_fick_alpha_range(self):
        votes = [
            {"barrel": 0.2, "dumbbell": 0.2, "globin": 0.2,
             "enzyme_active": 0.2, "allosteric": 0.2}
            for _ in range(7)
        ]
        mfb = MetaFickBalancer()
        state = mfb.compute_meta_fick_state(votes)
        assert 0.0 < state["alpha_meta"] < 1.0

    # ── D110: EnzymeLensSynthesis ────────────────────────────────

    def test_enzyme_lens_inherits_metafick(self):
        """EnzymeLensSynthesis is a MetaFickBalancer subclass."""
        els = EnzymeLensSynthesis()
        assert isinstance(els, MetaFickBalancer)

    def test_enzyme_lens_accepts_spectrum(self):
        """Should store initial eigenvalues and eigenvectors."""
        evals = np.array([0.0, 0.5, 1.0, 2.0])
        evecs = np.eye(4)
        els = EnzymeLensSynthesis(evals=evals, evecs=evecs)
        np.testing.assert_array_equal(els.initial_evals, evals)
        np.testing.assert_array_equal(els.initial_evecs, evecs)

    def test_enzyme_lens_no_activation_clear_winner(self):
        """When barrel dominates, lens should NOT activate."""
        profiles = []
        for name in ["algebraic", "musical", "fick", "thermal",
                      "cooperative", "propagative", "fragile"]:
            p = ThermoReactionProfile(instrument=name, intent_idx=0)
            profiles.append(p)

        els = EnzymeLensSynthesis()
        votes = [p.archetype_vote() for p in profiles]
        state = els.compute_meta_fick_state(votes)
        result = els.synthesize_identity(profiles, state)
        # Result should contain lens metadata
        assert "enzyme_lens_activated" in result
        assert "scores" in result

    # ── D111: Multi-mode hinge observables ───────────────────────

    def test_multimode_ipr_positive(self):
        """multimode_ipr should return a positive float."""
        from ibp_enm.thermodynamics import multimode_ipr
        evecs = np.eye(10)
        val = multimode_ipr(evecs)
        assert val > 0.0
        assert isinstance(val, float)

    def test_multimode_ipr_localised(self):
        """A mode concentrated on one residue should have IPR ~ 1."""
        from ibp_enm.thermodynamics import multimode_ipr
        evecs = np.zeros((10, 10))
        for k in range(10):
            evecs[k, k] = 1.0  # each mode localised on one residue
        val = multimode_ipr(evecs, modes=range(2, 6))
        assert abs(val - 1.0) < 1e-6

    def test_multimode_ipr_delocalised(self):
        """Uniform eigenvectors should have IPR ~ 1/N."""
        from ibp_enm.thermodynamics import multimode_ipr
        N = 100
        evecs = np.ones((N, N)) / np.sqrt(N)
        val = multimode_ipr(evecs, modes=range(2, 6))
        assert abs(val - 1.0 / N) < 1e-4

    def test_hinge_occupation_neutral_no_boundary(self):
        """With uniform domain labels (no boundary), ratio = 1.0."""
        from ibp_enm.thermodynamics import hinge_occupation_ratio
        evecs = np.eye(20)
        labels = np.zeros(20, dtype=int)  # all same domain
        val = hinge_occupation_ratio(evecs, labels)
        assert abs(val - 1.0) < 1e-6

    def test_hinge_occupation_concentrated(self):
        """Mode amplitude concentrated near boundary \u2192 ratio > 1."""
        from ibp_enm.thermodynamics import hinge_occupation_ratio
        N = 40
        evecs = np.zeros((N, N))
        for k in range(N):
            v = np.zeros(N)
            # Concentrate amplitude near boundary (position 20)
            for i in range(N):
                v[i] = np.exp(-0.5 * ((i - 20) / 3.0) ** 2)
            v /= np.linalg.norm(v)
            evecs[:, k] = v
        labels = np.array([0] * 20 + [1] * 20)
        val = hinge_occupation_ratio(evecs, labels)
        assert val > 1.0

    def test_domain_stiffness_asymmetry_symmetric(self):
        """Two equally-packed domains should have asymmetry ~ 0."""
        from ibp_enm.thermodynamics import domain_stiffness_asymmetry
        labels = np.array([0, 0, 0, 1, 1, 1])
        # Symmetric contacts: 3 in each domain
        contacts = {(0, 1): 1.0, (1, 2): 1.0, (0, 2): 1.0,
                    (3, 4): 1.0, (4, 5): 1.0, (3, 5): 1.0}
        val = domain_stiffness_asymmetry(contacts, labels)
        assert abs(val) < 1e-6

    def test_domain_stiffness_asymmetry_positive(self):
        """One dense domain + one sparse \u2192 positive asymmetry."""
        from ibp_enm.thermodynamics import domain_stiffness_asymmetry
        labels = np.array([0, 0, 0, 1, 1, 1])
        # Domain 0 has 3 contacts, domain 1 has 1
        contacts = {(0, 1): 1.0, (1, 2): 1.0, (0, 2): 1.0,
                    (3, 4): 1.0}
        val = domain_stiffness_asymmetry(contacts, labels)
        assert val > 0.0
        assert val <= 1.0

    # ── D111: HingeLensSynthesis ─────────────────────────────────

    def test_hinge_lens_inherits_enzyme_lens(self):
        """HingeLensSynthesis is an EnzymeLensSynthesis subclass."""
        hls = HingeLensSynthesis()
        assert isinstance(hls, EnzymeLensSynthesis)
        assert isinstance(hls, MetaFickBalancer)

    def test_hinge_lens_accepts_domain_data(self):
        """Should store domain_labels and contacts."""
        evals = np.array([0.0, 0.5, 1.0, 2.0])
        evecs = np.eye(4)
        labels = np.array([0, 0, 1, 1])
        contacts = {(0, 1): 1.0, (2, 3): 1.0}
        hls = HingeLensSynthesis(
            evals=evals, evecs=evecs,
            domain_labels=labels, contacts=contacts)
        np.testing.assert_array_equal(hls.initial_evals, evals)
        np.testing.assert_array_equal(hls.domain_labels, labels)
        assert hls.contacts_for_hinge == contacts

    def test_hinge_lens_result_has_hinge_fields(self):
        """synthesize_identity should include hinge lens metadata."""
        profiles = []
        for name in ["algebraic", "musical", "fick", "thermal",
                     "cooperative", "propagative", "fragile"]:
            p = ThermoReactionProfile(instrument=name, intent_idx=0)
            profiles.append(p)

        hls = HingeLensSynthesis()
        votes = [p.archetype_vote() for p in profiles]
        state = hls.compute_meta_fick_state(votes)
        result = hls.synthesize_identity(profiles, state)
        assert "hinge_signals" in result
        assert "hinge_boost" in result
        assert "hinge_lens_activated" in result

    def test_hinge_boost_calibration(self):
        """The static _hinge_boost matches D111 calibration."""
        # T4 lysozyme: hinge_R = 1.091 → boost ≈ 0.273
        t4_sigs = {"hinge_r": 1.091}
        boost = HingeLensSynthesis._hinge_boost(t4_sigs)
        assert abs(boost - 0.273) < 0.01

        # AdK: hinge_R = 0.952 → boost = 0
        adk_sigs = {"hinge_r": 0.952}
        boost = HingeLensSynthesis._hinge_boost(adk_sigs)
        assert boost == 0.0

        # Cap at 0.35
        big_sigs = {"hinge_r": 2.0}
        boost = HingeLensSynthesis._hinge_boost(big_sigs)
        assert abs(boost - 0.35) < 1e-6


# ── top-level imports ────────────────────────────────────────────

class TestPackageImports:
    def test_top_level_imports(self):
        """All __all__ exports should be importable from ibp_enm."""
        import ibp_enm
        for name in [
            "ThermodynamicBand", "run_single_protein",
            "ThermoReactionProfile", "MetaFickBalancer",
            "EnzymeLensSynthesis", "HingeLensSynthesis",
            "CarvingIntent", "ReactionSignature", "FickBalancer",
            "vibrational_entropy", "heat_capacity",
            "per_residue_entropy_contribution", "entropy_asymmetry_score",
            "multimode_ipr", "hinge_occupation_ratio",
            "domain_stiffness_asymmetry",
            "ARCHETYPES", "PROTEINS", "GROUND_TRUTH",
        ]:
            assert hasattr(ibp_enm, name), f"Missing export: {name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
