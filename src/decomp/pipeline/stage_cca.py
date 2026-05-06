"""Stage 5: pairwise CCA + pCCA across regions.

Accepts an explicit list of region pairs to run, so the caller can include only the pairs
that actually have units in this session (avoids running CCA on a single-region session).
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from ..cca.run_cca import run_pair
from ..data.binning import BinnedSession
from ..svca.run_svca import SVCAResult


def _confound_matrix(binned: BinnedSession) -> np.ndarray | None:
    cols = [c for c in ("wheel_velocity", "pupil") if c in binned.covariates.columns]
    if not cols:
        return None
    Z = binned.covariates[cols].to_numpy()
    # standardize so residualization is well-conditioned
    Z = (Z - Z.mean(axis=0, keepdims=True)) / (Z.std(axis=0, keepdims=True) + 1e-9)
    return Z


def run_session_cca(binned: BinnedSession, svca: dict[str, SVCAResult],
                    cache_dir: Path = Path("data/cache"),
                    pair_rois: list[tuple[str, str]] | None = None,
                    n_components: int = 8, n_surrogates: int = 200,
                    n_splits: int = 5, random_state: int = 0) -> pd.DataFrame:
    """Run CCA + pCCA on the requested region pairs that exist in `svca`.

    If `pair_rois` is None, falls back to all unique pairs of regions present in `svca`
    (MVP behavior). Otherwise, only the listed pairs are run, and pairs whose regions are
    missing from `svca` are silently skipped.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    scores = {roi: res.scores for roi, res in svca.items() if res.scores.size > 0}
    Z = _confound_matrix(binned)

    if pair_rois is None:
        pair_rois = list(combinations(sorted(scores.keys()), 2))

    rows: list[dict] = []
    for a, b in pair_rois:
        if a not in scores or b not in scores:
            continue
        out = run_pair(scores[a], scores[b], Z,
                       n_components=n_components, n_splits=n_splits,
                       n_surrogates=n_surrogates, random_state=random_state)
        for k in range(out["n_components_used"]):
            rows.append({
                "pair_a": a,
                "pair_b": b,
                "component": k,
                "rho_cca":      float(out["rho_cca"][k]),
                "rho_pcca":     float(out["rho_pcca"][k]),
                "null_cca_99":  float(out["null_cca_99"][k]),
                "null_pcca_99": float(out["null_pcca_99"][k]),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df.insert(0, "eid", binned.eid)
        df.to_csv(cache_dir / f"{binned.eid}_cca_results.csv", index=False)
    return df
