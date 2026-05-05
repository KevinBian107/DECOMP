"""Synthetic ground-truth tests for the methods that drive fig02 / fig03 / fig04.

These tests construct synthetic neural-population-style data with **known shared structure**
and verify that our SVCA, CCA, pCCA, and phase-shuffle null implementations recover it.
This is the right form of correctness check for math-heavy modules where ordinary unit
tests would just re-compute the same arithmetic.

Design:
    * Each test targets ONE mathematical claim.
    * Each test prints diagnostics on failure (ρ values, expected vs observed) so a
      regression is interpretable.
    * Tolerances are conservative: tests should pass on every seed with high probability.

Run with:
    PYTHONPATH=src pytest tests/test_synthetic.py -v
"""

from __future__ import annotations

import numpy as np

from decomp.cca.nulls import phase_shuffle
from decomp.cca.pcca import (
    cca_svd,
    cv_canonical_correlations,
    fit_pcca,
    residualize,
)
from decomp.glm.design import DesignSpec
from decomp.glm.fit import fit_region
from decomp.svca.run_svca import run_region_svca


# ----------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------

def make_shared_latent(T: int, dim_shared: int, dim_priv_x: int, dim_priv_y: int,
                       p_x: int = 8, p_y: int = 8, noise: float = 0.2,
                       seed: int = 0):
    """Construct (X, Y) with a known number of shared latent dimensions.

    X = s_shared @ A + s_priv_x @ A_x + ε_x          (shape T × p_x)
    Y = s_shared @ B + s_priv_y @ B_y + ε_y          (shape T × p_y)

    s_shared is identical in both views, so CCA should recover dim_shared canonical
    correlations near 1 and the rest near zero.
    """
    rng = np.random.default_rng(seed)
    s_shared = rng.standard_normal((T, dim_shared))
    s_priv_x = rng.standard_normal((T, dim_priv_x))
    s_priv_y = rng.standard_normal((T, dim_priv_y))
    A_shared = rng.standard_normal((dim_shared, p_x))
    A_priv = rng.standard_normal((dim_priv_x, p_x)) if dim_priv_x else np.zeros((0, p_x))
    B_shared = rng.standard_normal((dim_shared, p_y))
    B_priv = rng.standard_normal((dim_priv_y, p_y)) if dim_priv_y else np.zeros((0, p_y))
    eps_x = noise * rng.standard_normal((T, p_x))
    eps_y = noise * rng.standard_normal((T, p_y))
    X = s_shared @ A_shared + s_priv_x @ A_priv + eps_x
    Y = s_shared @ B_shared + s_priv_y @ B_priv + eps_y
    return X, Y, s_shared


# ----------------------------------------------------------------------------------------
# CCA — protects fig03
# ----------------------------------------------------------------------------------------

class TestCCA:
    """Verify the SVD-based CCA implementation in src/decomp/cca/pcca.py."""

    def test_self_canonical_gives_unit_correlation(self):
        """CCA(X, X) must give ρ_k = 1 for all k up to the rank of X."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((1000, 5))
        rho, _, _ = cca_svd(X, X, n_components=5)
        assert np.allclose(rho, 1.0, atol=1e-3), \
            f"CCA(X, X) should give ρ=1, got {rho.tolist()}"

    def test_independent_data_gives_low_correlation(self):
        """CCA between independent X and Y on a long time series should give ρ_max
        close to zero. Finite-sample bias gives a small floor; we assert ρ_max < 0.15."""
        rng = np.random.default_rng(0)
        T = 5000
        X = rng.standard_normal((T, 5))
        Y = rng.standard_normal((T, 5))
        rho, _, _ = cca_svd(X, Y, n_components=5)
        assert np.max(rho) < 0.15, \
            f"CCA on independent T={T} data should give ρ_max<0.15, got {rho.tolist()}"

    def test_recovers_shared_dimensionality(self):
        """If X and Y share exactly r=2 latent dimensions, ρ_1 and ρ_2 should be near
        1 and ρ_3 should drop sharply."""
        T = 8000
        X, Y, _ = make_shared_latent(T, dim_shared=2, dim_priv_x=2, dim_priv_y=2,
                                      noise=0.1, seed=0)
        rho, _, _ = cca_svd(X, Y, n_components=5)
        assert rho[0] > 0.95, f"ρ_1 should be near 1, got {rho[0]:.3f}"
        assert rho[1] > 0.90, f"ρ_2 should be near 1, got {rho[1]:.3f}"
        assert rho[2] < 0.50, \
            f"ρ_3 should drop sharply (no 3rd shared dim), got {rho[2]:.3f}"

    def test_scale_invariance(self):
        """CCA is invariant to per-side scaling under unit-scale data. We allow a small
        tolerance because our SVD-based CCA includes a Tikhonov regularizer (reg=1e-6)
        that breaks exact scale invariance at extreme magnitudes; a modest 2× / 0.5×
        scaling stays well inside the regime where the regularizer is negligible."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((1000, 3))
        Y = X + 0.5 * rng.standard_normal((1000, 3))
        rho_a, _, _ = cca_svd(X, Y, n_components=3)
        rho_b, _, _ = cca_svd(X * 2.0, Y * 0.5, n_components=3)
        assert np.allclose(rho_a, rho_b, atol=1e-3), \
            f"CCA should be scale-invariant: rho_a={rho_a.tolist()} vs rho_b={rho_b.tolist()}"

    def test_orthogonal_views_give_zero(self):
        """If X and Y are constructed to be orthogonal in time (no shared latent),
        ρ_k should be at the chance floor."""
        rng = np.random.default_rng(0)
        T = 5000
        # Orthogonal time series: Y is X "rotated" 90° in time via a random latent
        # that is independent of X.
        X = rng.standard_normal((T, 4))
        Y = rng.standard_normal((T, 4))  # truly independent
        rho, _, _ = cca_svd(X, Y, n_components=4)
        assert np.max(rho) < 0.15, \
            f"orthogonal views should give ρ near 0, got {rho.tolist()}"


# ----------------------------------------------------------------------------------------
# Partial CCA — protects fig04 (the headline answer figure)
# ----------------------------------------------------------------------------------------

class TestPartialCCA:
    """Verify the residualize-then-CCA construction in src/decomp/cca/pcca.py.

    These tests are the most important in the suite because the headline result of the
    project (V1↔CB shared subspace survives partialling out wheel + pupil) hinges on
    pCCA giving the correct answer.
    """

    def test_confound_equals_shared_collapses_pcca(self):
        """If Z is *exactly* the shared latent that drives both X and Y, partialling Z
        out should collapse pCCA to ~0. Survival ratio should drop near zero."""
        rng = np.random.default_rng(0)
        T = 5000
        Z = rng.standard_normal((T, 3))
        # X and Y both observe Z plus a small private noise term
        X = Z + 0.1 * rng.standard_normal((T, 3))
        Y = Z + 0.1 * rng.standard_normal((T, 3))
        rho_cca, _, _ = cca_svd(X, Y, n_components=3)
        rho_pcca, _, _ = fit_pcca(X, Y, Z, n_components=3)
        assert rho_cca[0] > 0.95, \
            f"CCA(X, Y) should detect the strong shared Z, got ρ_1={rho_cca[0]:.3f}"
        assert rho_pcca[0] < 0.20, \
            f"pCCA(X, Y | Z=shared) should collapse to near 0, got ρ_1={rho_pcca[0]:.3f}"
        survival = rho_pcca.sum() / max(rho_cca.sum(), 1e-9)
        assert survival < 0.20, \
            f"Survival ratio should collapse, got {survival:.3f}"

    def test_independent_confound_preserves_correlation(self):
        """If Z is independent of X and Y, partialling Z must NOT change canonical
        correlations. Survival ≈ 1."""
        rng = np.random.default_rng(0)
        T = 5000
        X, Y, _ = make_shared_latent(T, dim_shared=2, dim_priv_x=2, dim_priv_y=2,
                                      noise=0.1, seed=42)
        Z = rng.standard_normal((T, 2))  # independent of X and Y
        rho_cca, _, _ = cca_svd(X, Y, n_components=3)
        rho_pcca, _, _ = fit_pcca(X, Y, Z, n_components=3)
        assert np.allclose(rho_cca[:2], rho_pcca[:2], atol=0.05), \
            f"Independent Z should preserve ρ: cca={rho_cca.tolist()} pcca={rho_pcca.tolist()}"

    def test_partial_collapse_when_z_is_one_of_two_shared_latents(self):
        """X and Y share TWO independent latents Z and S. Partialling Z should remove
        ONE shared canonical correlation (the Z one) and leave the S one intact."""
        rng = np.random.default_rng(0)
        T = 8000
        Z = rng.standard_normal((T, 1))
        S = rng.standard_normal((T, 1))
        # Each view observes [Z, S] linearly mixed into 6 features + small noise
        latents = np.hstack([Z, S])
        A = rng.standard_normal((2, 6))
        B = rng.standard_normal((2, 6))
        X = latents @ A + 0.05 * rng.standard_normal((T, 6))
        Y = latents @ B + 0.05 * rng.standard_normal((T, 6))
        rho_cca, _, _ = cca_svd(X, Y, n_components=3)
        rho_pcca, _, _ = fit_pcca(X, Y, Z, n_components=3)
        # CCA: 2 shared dims -> ρ_1 and ρ_2 both near 1
        assert rho_cca[0] > 0.95, f"CCA ρ_1 should be high, got {rho_cca[0]:.3f}"
        assert rho_cca[1] > 0.85, f"CCA ρ_2 should be high, got {rho_cca[1]:.3f}"
        # pCCA: only 1 shared dim survives (S) -> ρ_1 high, ρ_2 collapses
        assert rho_pcca[0] > 0.85, \
            f"pCCA ρ_1 (S, not in Z) should still be high, got {rho_pcca[0]:.3f}"
        assert rho_pcca[1] < 0.30, \
            f"pCCA ρ_2 (Z removed) should collapse, got {rho_pcca[1]:.3f}"

    def test_residualize_when_z_none_returns_input(self):
        """residualize(Y, None) is the identity. Sanity for the pCCA = CCA fallback."""
        rng = np.random.default_rng(0)
        Y = rng.standard_normal((100, 3))
        Y_res = residualize(Y, None)
        assert np.array_equal(Y_res, Y), "residualize(Y, None) must be identity"

    def test_cv_train_test_discipline_no_leakage(self):
        """The CV pCCA function fits the residualization on the training fold and
        applies it to the held-out test fold. If we accidentally leaked Z fitting across
        folds, even a Z that is independent of X, Y would systematically shrink the
        test-set ρ. Verify that on truly-independent data, pCCA and CCA give similar
        held-out ρ."""
        rng = np.random.default_rng(0)
        T = 3000
        X = rng.standard_normal((T, 3))
        Y = rng.standard_normal((T, 3))
        Z = rng.standard_normal((T, 2))  # independent of X and Y
        out_cca = cv_canonical_correlations(X, Y, Z=None, n_components=3, n_splits=5)
        out_pcca = cv_canonical_correlations(X, Y, Z=Z, n_components=3, n_splits=5)
        diff = out_pcca["rho_mean"] - out_cca["rho_mean"]
        assert np.abs(diff).max() < 0.10, \
            (f"pCCA should not systematically shrink ρ when Z is independent: "
             f"diff={diff.tolist()}")


# ----------------------------------------------------------------------------------------
# SVCA — protects fig02
# ----------------------------------------------------------------------------------------

class TestSVCA:
    """Verify the cross-half reliability machinery in src/decomp/svca/run_svca.py."""

    def test_pure_noise_low_reliability(self):
        """SVCA on pure independent Gaussian noise must give reliability near zero on
        every component. If SVCA returns reliability near 1 on noise, it's overfitting
        to spurious cross-half coincidences and the entire fig02 interpretation is wrong."""
        rng = np.random.default_rng(0)
        n_units, T = 80, 6000
        X = rng.standard_normal((n_units, T))
        result = run_region_svca(X, region="noise", random_state=0, n_components=4)
        max_abs = float(np.max(np.abs(result.reliability)))
        assert max_abs < 0.30, \
            f"SVCA on noise should give |ρ| < 0.3 everywhere, got max={max_abs:.3f}"

    def test_rank1_latent_high_reliability(self):
        """SVCA on a population driven by a single shared latent should yield ρ_1
        clearly above 0.5."""
        rng = np.random.default_rng(0)
        n_units, T = 80, 6000
        s = rng.standard_normal(T)
        loadings = rng.standard_normal(n_units)
        X = np.outer(loadings, s) + 0.3 * rng.standard_normal((n_units, T))
        result = run_region_svca(X, region="rank1", random_state=0, n_components=4)
        assert result.reliability[0] > 0.7, \
            f"SVCA ρ_1 on rank-1 latent should be > 0.7, got {result.reliability[0]:.3f}"

    def test_rank3_drops_after_third_component(self):
        """SVCA on a rank-3 driven population should show high reliability at k = 1, 2, 3
        and a clear drop at k = 4+."""
        rng = np.random.default_rng(0)
        n_units, T = 120, 8000
        rank = 3
        S = rng.standard_normal((rank, T))
        L = rng.standard_normal((n_units, rank))
        X = L @ S + 0.3 * rng.standard_normal((n_units, T))
        result = run_region_svca(X, region="rank3", random_state=0, n_components=8)
        rel = result.reliability
        assert all(r > 0.5 for r in rel[:rank]), \
            f"top {rank} components should have ρ > 0.5, got {rel[:rank].tolist()}"
        # Components beyond rank should drop sharply (allow some slack for finite sample noise)
        assert rel[rank] < rel[rank - 1] - 0.2, \
            (f"reliability should drop at k={rank+1} (1-indexed): "
             f"rel[{rank-1}]={rel[rank-1]:.3f} vs rel[{rank}]={rel[rank]:.3f}")


# ----------------------------------------------------------------------------------------
# Phase-shuffle null — protects fig03 (the null bound on canonical correlations)
# ----------------------------------------------------------------------------------------

class TestPhaseShuffle:
    """Verify the phase-shuffle null in src/decomp/cca/nulls.py."""

    def test_preserves_power_spectrum(self):
        """Phase shuffling must preserve the per-frequency magnitude. If it changes the
        spectrum, the null is wrong (it tests a different null hypothesis than claimed)."""
        rng = np.random.default_rng(0)
        x = rng.standard_normal((1024, 4))
        x_shuf = phase_shuffle(x, rng)
        m_orig = np.abs(np.fft.rfft(x, axis=0))
        m_shuf = np.abs(np.fft.rfft(x_shuf, axis=0))
        max_dev = float(np.abs(m_orig - m_shuf).max())
        assert max_dev < 1e-8, \
            f"phase_shuffle must preserve power spectrum, max deviation = {max_dev:.2e}"

    def test_breaks_cross_correlation_with_other_signal(self):
        """Phase-shuffling X destroys its temporal alignment with Y. If we set X = Y
        and shuffle X, CCA(shuffled_X, Y) must drop dramatically from ρ ≈ 1 to near 0."""
        rng = np.random.default_rng(0)
        T = 2000
        Y = rng.standard_normal((T, 3))
        X = Y + 0.05 * rng.standard_normal((T, 3))
        rho_real, _, _ = cca_svd(X, Y, n_components=3)
        X_shuf = phase_shuffle(X, rng)
        rho_shuf, _, _ = cca_svd(X_shuf, Y, n_components=3)
        assert rho_real[0] > 0.95, \
            f"CCA on aligned X≈Y should give ρ_1 near 1, got {rho_real[0]:.3f}"
        assert rho_shuf[0] < 0.30, \
            f"CCA after phase-shuffle should drop to near 0, got {rho_shuf[0]:.3f}"


# ----------------------------------------------------------------------------------------
# GLM ΔR² — protects fig01
# ----------------------------------------------------------------------------------------

class TestGLM:
    """Verify the vectorized Ridge + leave-one-group-out ΔR² in src/decomp/glm/fit.py."""

    def test_irrelevant_group_has_zero_delta_r2(self):
        """A kernel group that is pure noise (independent of y) should have ΔR² ≈ 0.
        A nonzero systematic ΔR² for noise regressors would indicate leakage in the CV."""
        rng = np.random.default_rng(0)
        T, n_neurons = 4000, 1
        X_real = rng.standard_normal((T, 2))
        X_noise = rng.standard_normal((T, 3))
        beta = np.array([1.0, -0.5])
        Y = (X_real @ beta + 0.5 * rng.standard_normal(T))[:, None]
        X = np.hstack([X_real, X_noise])
        design = DesignSpec(
            columns=["real_0", "real_1", "noise_0", "noise_1", "noise_2"],
            groups={"real": [0, 1], "noise": [2, 3, 4]},
        )
        result = fit_region(X, Y, design, n_splits=5, alpha=0.1, random_state=0)
        assert result.deltas["real"][0] > 0.5, \
            f"Real group ΔR² should be substantial, got {result.deltas['real'][0]:.3f}"
        assert abs(result.deltas["noise"][0]) < 0.05, \
            f"Noise group ΔR² should be near 0, got {result.deltas['noise'][0]:.3f}"

    def test_full_dependency_group_dominates(self):
        """If y depends entirely on group A and not at all on group B, then ΔR²(A)
        should equal full_R² and ΔR²(B) should be ~0."""
        rng = np.random.default_rng(0)
        T = 4000
        X_a = rng.standard_normal((T, 2))
        X_b = rng.standard_normal((T, 2))
        beta_a = np.array([1.0, -0.5])
        Y = (X_a @ beta_a + 0.3 * rng.standard_normal(T))[:, None]
        X = np.hstack([X_a, X_b])
        design = DesignSpec(
            columns=["a0", "a1", "b0", "b1"],
            groups={"a": [0, 1], "b": [2, 3]},
        )
        result = fit_region(X, Y, design, n_splits=5, alpha=0.1, random_state=0)
        # ΔR²(a) should be large (close to the full R²) since dropping a destroys the model.
        assert result.deltas["a"][0] > 0.5, \
            f"Causal group ΔR² should be high, got {result.deltas['a'][0]:.3f}"
        # ΔR²(b) should be near zero since b doesn't drive y.
        assert abs(result.deltas["b"][0]) < 0.05, \
            f"Non-causal group ΔR² should be near 0, got {result.deltas['b'][0]:.3f}"

    def test_vectorized_matches_per_neuron(self):
        """fit_region(X, Y) with multi-neuron Y must produce the same per-neuron results
        as feeding columns of Y one at a time. Catches accidental cross-neuron leakage
        in the vectorized solver."""
        rng = np.random.default_rng(0)
        T, n_neurons = 2000, 3
        X = rng.standard_normal((T, 4))
        beta = rng.standard_normal((4, n_neurons))
        Y = X @ beta + 0.5 * rng.standard_normal((T, n_neurons))
        design = DesignSpec(
            columns=["c0", "c1", "c2", "c3"],
            groups={"first": [0, 1], "second": [2, 3]},
        )
        # Vectorized: all 3 neurons at once
        joint = fit_region(X, Y, design, n_splits=5, alpha=1.0, random_state=0)
        # Per-neuron loop
        for i in range(n_neurons):
            single = fit_region(X, Y[:, i:i+1], design, n_splits=5, alpha=1.0,
                                random_state=0)
            assert np.allclose(joint.full_R2[i], single.full_R2[0], atol=1e-9), \
                f"neuron {i} full_R2 differs: joint={joint.full_R2[i]} single={single.full_R2[0]}"
            for g in design.groups:
                assert np.allclose(joint.deltas[g][i], single.deltas[g][0], atol=1e-9), \
                    (f"neuron {i} group {g} ΔR² differs: "
                     f"joint={joint.deltas[g][i]} single={single.deltas[g][0]}")
