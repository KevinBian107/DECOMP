"""Representation-geometry visualizations of cross-region shared subspaces.

This module supports two geometric pictures that the scalar canonical correlations
and survival ratios miss:

1. Joint state-space density (`joint_state_space_2d`):
     2-D scatter of (U_A(t), U_B(t)) on the leading canonical pair, binned along a
     behavioural covariate (wheel-velocity quantile, pupil quantile). Reveals what
     the shared mode IS — and how that gradient washes out under pCCA.

2. Cross-region representational similarity analysis (`cross_region_rsa`):
     Mean SVCA-score vector per trial condition per region → per-region pairwise
     dissimilarity matrix → Spearman correlation between the two RDMs. Basis-free
     check that complements the canonical-correlation findings.

Both work from the SVCA scores already cached by the main pipeline; no new
spike-binning is required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold


# ---------------------------------------------------------------------------
# 1.  Joint state-space density coloured by behaviour
# ---------------------------------------------------------------------------

def joint_state_space_2d(
    U_A: np.ndarray, U_B: np.ndarray, behaviour: np.ndarray,
    n_quantile: int = 4,
) -> dict:
    """Returns coupled arrays describing the joint state-space cloud:

        a, b, z      (n,)        per-time-bin canonical variates + behaviour value
        z_std        (n,)        standardised behaviour (z-score)
        clouds       list[(2,k)] points partitioned into n_quantile bins of z
        edges        (n_q+1,)    quantile boundaries on z
        rho          float       overall Pearson(U_A, U_B)
    """
    a = np.asarray(U_A, dtype=float).ravel()
    b = np.asarray(U_B, dtype=float).ravel()
    z = np.asarray(behaviour, dtype=float).ravel()
    keep = np.isfinite(a) & np.isfinite(b) & np.isfinite(z)
    a, b, z = a[keep], b[keep], z[keep]
    if len(a) == 0:
        return {"a": a, "b": b, "z": z, "z_std": z,
                "edges": np.zeros(n_quantile + 1), "clouds": [], "rho": 0.0}

    z_std = (z - z.mean()) / (z.std() + 1e-9)
    edges = np.quantile(z, np.linspace(0, 1, n_quantile + 1))
    edges[-1] = edges[-1] + 1e-9
    clouds = []
    for i in range(n_quantile):
        mask = (z >= edges[i]) & (z < edges[i + 1])
        clouds.append(np.stack([a[mask], b[mask]], axis=0))
    rho = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0
    return {"a": a, "b": b, "z": z, "z_std": z_std,
            "edges": edges, "clouds": clouds, "rho": rho}


# ---------------------------------------------------------------------------
# 2.  Cross-region representational similarity (RSA)
# ---------------------------------------------------------------------------

def _residualize_train_test(X: np.ndarray, Y: np.ndarray, Z: np.ndarray | None,
                              n_splits: int = 5, random_state: int = 0,
                              ) -> tuple[np.ndarray, np.ndarray]:
    """Fit Z->X and Z->Y on train, apply to whole-array (concatenate held-out preds).

    Mirrors the no-leakage residualisation used elsewhere in this codebase; returns
    fully-residualised arrays the same length as the inputs.
    """
    if Z is None:
        return X.copy(), Y.copy()
    Xr = np.empty_like(X)
    Yr = np.empty_like(Y)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for tr_idx, te_idx in kf.split(X):
        rx = LinearRegression(fit_intercept=True).fit(Z[tr_idx], X[tr_idx])
        ry = LinearRegression(fit_intercept=True).fit(Z[tr_idx], Y[tr_idx])
        Xr[te_idx] = X[te_idx] - rx.predict(Z[te_idx])
        Yr[te_idx] = Y[te_idx] - ry.predict(Z[te_idx])
    return Xr, Yr

def _trial_condition_means(
    scores: np.ndarray, bin_centers: np.ndarray,
    trial_times: np.ndarray, trial_labels: np.ndarray,
    pre_s: float = 0.0, post_s: float = 0.4,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean SVCA-score vector in [pre_s, post_s] around each trial start, grouped by label.

    Returns:
        conds      (n_cond,)         unique labels, sorted
        cond_mean  (n_cond, K)       mean score vector per condition
    """
    bc = np.asarray(bin_centers, dtype=float)
    dt = float(np.median(np.diff(bc[:200])))
    n_pre = int(round(pre_s / dt))
    n_post = int(round(post_s / dt))
    K = scores.shape[1]

    per_trial = []
    per_label = []
    for t, lab in zip(trial_times, trial_labels):
        if not np.isfinite(t):
            continue
        idx = int(round((t - bc[0]) / dt))
        lo, hi = idx - n_pre, idx + n_post + 1
        if lo < 0 or hi > len(scores):
            continue
        seg = scores[lo:hi]
        if not np.all(np.isfinite(seg)):
            continue
        per_trial.append(seg.mean(axis=0))
        per_label.append(lab)
    if not per_trial:
        return np.array([]), np.zeros((0, K))

    df = pd.DataFrame(per_trial)
    df["__lab"] = per_label
    grouped = df.groupby("__lab").mean()
    conds = grouped.index.to_numpy()
    cond_mean = grouped.to_numpy()
    return conds, cond_mean


def cross_region_rsa(
    scores_A: np.ndarray, scores_B: np.ndarray,
    bin_centers: np.ndarray,
    trial_times: np.ndarray, trial_labels: np.ndarray,
    Z: np.ndarray | None = None,
    pre_s: float = 0.0, post_s: float = 0.4,
) -> dict:
    """Per-region RDMs over trial conditions + Spearman correlation between them.

    When Z is given, each region's scores are residualised against Z (train/test
    KFold no-leakage) before condition averaging — this measures geometry that is
    not linearly explainable by the behavioural covariates.
    """
    if Z is not None:
        SA, SB = _residualize_train_test(scores_A, scores_B, Z)
    else:
        SA, SB = scores_A, scores_B

    conds_a, mean_a = _trial_condition_means(SA, bin_centers, trial_times, trial_labels,
                                              pre_s=pre_s, post_s=post_s)
    conds_b, mean_b = _trial_condition_means(SB, bin_centers, trial_times, trial_labels,
                                              pre_s=pre_s, post_s=post_s)
    # Both regions are aligned by trial — same conditions, same order
    n = min(len(conds_a), len(conds_b))
    if n < 3:
        return {"conds": np.array([]), "rdm_a": np.zeros((0, 0)), "rdm_b": np.zeros((0, 0)),
                "spearman_r": np.nan, "spearman_p": np.nan}

    rdm_a = _pairwise_correlation_distance(mean_a[:n])
    rdm_b = _pairwise_correlation_distance(mean_b[:n])
    # Upper triangle without diagonal
    iu = np.triu_indices(n, k=1)
    rho, p = spearmanr(rdm_a[iu], rdm_b[iu])
    return {
        "conds": conds_a[:n], "rdm_a": rdm_a, "rdm_b": rdm_b,
        "spearman_r": float(rho), "spearman_p": float(p),
    }


def _pairwise_correlation_distance(X: np.ndarray) -> np.ndarray:
    """1 - Pearson(row_i, row_j). Zero diagonal, symmetric."""
    n = X.shape[0]
    Xn = X - X.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(Xn, axis=1, keepdims=True) + 1e-12
    Xz = Xn / norms
    R = Xz @ Xz.T
    R = np.clip(R, -1.0, 1.0)
    return 1.0 - R
