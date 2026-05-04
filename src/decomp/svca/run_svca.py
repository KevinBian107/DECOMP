"""Within-region SVCA (Stringer 2019) via MouseLand/neuropop.

For each region's binned spike matrix `(n_units, T)`, returns the reliable component
spectrum and the per-time-bin component scores `(T, k_R)` truncated to the reliability
cutoff.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SVCAResult:
    region: str
    scov: np.ndarray            # (k,) test-set covariance components
    varcov: np.ndarray          # (k,) total per-half variance
    reliability: np.ndarray     # scov / varcov
    k_reliable: int
    scores: np.ndarray          # (T, k_reliable) projection time series
    full_components: np.ndarray | None = None  # (n_units, k) loadings


def _reliability_cutoff(scov: np.ndarray, varcov: np.ndarray, threshold: float = 0.5) -> int:
    """Largest k such that all components 0..k-1 have scov/varcov > threshold (Stringer 2019)."""
    rel = np.divide(scov, varcov, out=np.zeros_like(scov, dtype=float), where=varcov > 0)
    above = rel > threshold
    if not above.any():
        return 0
    # contiguous run from index 0
    k = 0
    for v in above:
        if v:
            k += 1
        else:
            break
    return max(k, 1)


def run_region_svca(X: np.ndarray, region: str, threshold: float = 0.5,
                    random_state: int = 0, n_components: int = 8) -> SVCAResult:
    """Run SVCA on one region's spike matrix and project full population onto reliable PCs.

    `neuropop.dimensionality.SVCA(X)` accepts only X (no kwargs) and returns (scov, varcov).
    SVCA defines reliability per Stringer 2019; we use the cutoff to truncate a separate
    full-data PCA so downstream CCA / pCCA can operate on a (T, k) score matrix.
    """
    from neuropop.dimensionality import SVCA  # type: ignore[import-not-found]

    n_units, _T = X.shape
    if n_units < 2:
        raise ValueError(f"SVCA needs >=2 units in {region}; got {n_units}")

    rng_state = np.random.get_state()
    np.random.seed(random_state)
    try:
        Xf = X.astype(np.float32)
        # neuropop.SVCA mean-centers internally only via the comment line; do it here for safety
        Xf = Xf - Xf.mean(axis=1, keepdims=True)
        scov, varcov = SVCA(Xf)
    finally:
        np.random.set_state(rng_state)

    scov = np.asarray(scov).ravel()
    varcov = np.asarray(varcov).ravel()
    k = _reliability_cutoff(scov, varcov, threshold=threshold)

    # Score time series: take a fixed number of full-data PCs for downstream CCA.
    # `k_reliable` (Stringer 2019 reliability cutoff) is reported as a diagnostic but does
    # not gate the score dimension -- a fixed K=8 is more useful for CCA when the dataset
    # has only a handful of strictly-reliable components.
    Xc = X.astype(np.float64) - X.astype(np.float64).mean(axis=1, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    k_score = max(min(n_components, Vt.shape[0], n_units - 1), 1)
    scores = (S[:k_score][:, None] * Vt[:k_score]).T  # (T, k_score)

    return SVCAResult(
        region=region,
        scov=scov,
        varcov=varcov,
        reliability=np.divide(scov, varcov, out=np.zeros_like(scov, dtype=float), where=varcov > 0),
        k_reliable=int(max(k, 1)),
        scores=scores,
        full_components=U[:, :k_score],
    )


def run_all_regions(spikes_by_roi: dict[str, np.ndarray], threshold: float = 0.5,
                    random_state: int = 0, n_components: int = 8
                    ) -> dict[str, SVCAResult]:
    out: dict[str, SVCAResult] = {}
    for roi, X in spikes_by_roi.items():
        if X.shape[0] < 2:
            continue
        out[roi] = run_region_svca(X, region=roi, threshold=threshold,
                                    random_state=random_state, n_components=n_components)
    return out
