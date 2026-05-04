"""Phase-shuffle null distribution for canonical correlations.

Per-channel Fourier phase randomization preserves the power spectrum of each column while
breaking temporal correlation across X and Y. Stronger than simple time-circular shifts at
short recording lengths.
"""

from __future__ import annotations

import numpy as np

from .pcca import cv_canonical_correlations


def phase_shuffle(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Phase-randomize each column of X independently."""
    n = X.shape[0]
    Xf = np.fft.rfft(X, axis=0)
    # randomize phase but keep amplitude; preserve DC and Nyquist bins
    phases = np.exp(1j * rng.uniform(0, 2 * np.pi, size=Xf.shape))
    phases[0] = 1.0
    if n % 2 == 0:
        phases[-1] = 1.0
    return np.fft.irfft(Xf * phases, n=n, axis=0)


def shuffle_null(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray | None = None,
    n_components: int = 10,
    n_surrogates: int = 200,
    n_splits: int = 5,
    random_state: int = 0,
) -> dict:
    """Build a phase-shuffle null for cv_canonical_correlations(X, Y, Z).

    Returns the matrix of (n_surrogates, n_components) null mean-rho values plus the 99th
    percentile cutoff per component.
    """
    rng = np.random.default_rng(random_state)
    nulls = np.zeros((n_surrogates, n_components))
    for s in range(n_surrogates):
        Xs = phase_shuffle(X, rng)
        out = cv_canonical_correlations(Xs, Y, Z=Z, n_components=n_components, n_splits=n_splits,
                                        random_state=random_state + s)
        nulls[s] = out["rho_mean"]
    return {
        "null_rhos": nulls,
        "null_99": np.quantile(np.abs(nulls), 0.99, axis=0),
    }
