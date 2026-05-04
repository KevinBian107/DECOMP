"""Pairwise CCA + pCCA across the 6 region pairs, with phase-shuffle nulls.

Inputs are SVCA score time series per region (T, k_R). Confounds Z = [wheel_velocity, pupil]
on the same 20 ms bin grid.
"""

from __future__ import annotations

from itertools import combinations
from typing import Iterable

import numpy as np
import pandas as pd

from .nulls import shuffle_null
from .pcca import cv_canonical_correlations


def all_pairs(rois: list[str]) -> list[tuple[str, str]]:
    return list(combinations(rois, 2))


def run_pair(
    S_A: np.ndarray,
    S_B: np.ndarray,
    Z: np.ndarray | None,
    n_components: int = 10,
    n_splits: int = 5,
    n_surrogates: int = 200,
    random_state: int = 0,
) -> dict:
    """Run CCA, pCCA, and phase-shuffle nulls for one region pair.

    Returns a dict with keys:
      'rho_cca', 'rho_pcca', 'null_cca', 'null_pcca', 'n_components_used'
    """
    n_comp = min(n_components, S_A.shape[1], S_B.shape[1])
    cca = cv_canonical_correlations(S_A, S_B, Z=None, n_components=n_comp, n_splits=n_splits,
                                    random_state=random_state)
    pcca = cv_canonical_correlations(S_A, S_B, Z=Z, n_components=n_comp, n_splits=n_splits,
                                     random_state=random_state)
    null_cca = shuffle_null(S_A, S_B, Z=None, n_components=n_comp, n_surrogates=n_surrogates,
                            n_splits=n_splits, random_state=random_state)
    null_pcca = shuffle_null(S_A, S_B, Z=Z, n_components=n_comp, n_surrogates=n_surrogates,
                             n_splits=n_splits, random_state=random_state)
    return {
        "rho_cca": cca["rho_mean"],
        "rho_pcca": pcca["rho_mean"],
        "null_cca_99": null_cca["null_99"],
        "null_pcca_99": null_pcca["null_99"],
        "n_components_used": n_comp,
    }


def run_all_pairs(
    svca_scores: dict[str, np.ndarray],
    confounds: np.ndarray | None,
    rois: Iterable[str],
    **kwargs,
) -> pd.DataFrame:
    """Run all unique region-pair CCA + pCCA. Returns long-format DataFrame.

    Each row is one (pair, component) entry with columns
      pair_a, pair_b, component, rho_cca, rho_pcca, null_cca_99, null_pcca_99.
    """
    rows: list[dict] = []
    for a, b in all_pairs(list(rois)):
        if a not in svca_scores or b not in svca_scores:
            continue
        out = run_pair(svca_scores[a], svca_scores[b], confounds, **kwargs)
        for k in range(out["n_components_used"]):
            rows.append({
                "pair_a": a,
                "pair_b": b,
                "component": k,
                "rho_cca": float(out["rho_cca"][k]),
                "rho_pcca": float(out["rho_pcca"][k]),
                "null_cca_99": float(out["null_cca_99"][k]),
                "null_pcca_99": float(out["null_pcca_99"][k]),
            })
    return pd.DataFrame(rows)
