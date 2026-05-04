"""Stage 5: pairwise CCA + pCCA across regions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..cca.run_cca import run_all_pairs
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
                    n_components: int = 8, n_surrogates: int = 200,
                    n_splits: int = 5, random_state: int = 0) -> pd.DataFrame:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    scores = {roi: res.scores for roi, res in svca.items() if res.scores.size > 0}
    Z = _confound_matrix(binned)

    df = run_all_pairs(
        svca_scores=scores,
        confounds=Z,
        rois=list(scores.keys()),
        n_components=n_components,
        n_splits=n_splits,
        n_surrogates=n_surrogates,
        random_state=random_state,
    )
    df.insert(0, "eid", binned.eid)
    df.to_csv(cache_dir / f"{binned.eid}_cca_results.csv", index=False)
    return df
