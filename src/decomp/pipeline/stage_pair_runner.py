"""Per-session multi-pair workhorse.

Given a list of region pairs, runs SVCA on every region that appears in any pair (once),
then CCA + pCCA on each pair where the session has both regions present. Returns the
SVCA results bundle (per-region) and a long-form CCA dataframe (one row per
(pair, component)).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..data.binning import BinnedSession
from ..svca.run_svca import SVCAResult, run_region_svca
from .stage_cca import run_session_cca


def run_pair_session(eid: str, binned: BinnedSession,
                     pair_rois: list[tuple[str, str]],
                     cache_dir: Path = Path("data/cache"),
                     n_components: int = 8, n_surrogates: int = 200,
                     n_splits: int = 5, random_state: int = 0,
                     ) -> tuple[dict[str, SVCAResult], pd.DataFrame]:
    """Run SVCA + multi-pair CCA/pCCA for one session.

    Args:
        eid: session id (used for cache keying).
        binned: BinnedSession with `spikes_by_roi` for whichever ROIs are present.
        pair_rois: list of (roi_a, roi_b) pairs to consider. Only pairs where BOTH
            regions are present in `binned.spikes_by_roi` are actually run.
        cache_dir: where intermediate parquet/npy/csv land.

    Returns:
        (svca_results, cca_df).
        svca_results: dict[roi -> SVCAResult] for every region that appeared in any
            pair AND had >= 2 units in this session.
        cca_df: long-form DataFrame with columns
            [eid, pair_a, pair_b, component, rho_cca, rho_pcca, null_cca_99, null_pcca_99].
            May be empty if no pair was runnable.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Which regions does this session actually have? Union over all requested pairs.
    needed = {r for pair in pair_rois for r in pair}
    available = {r for r in needed if r in binned.spikes_by_roi
                 and binned.spikes_by_roi[r].shape[0] >= 2}

    # SVCA per region (cached on disk per eid+roi via run_session_svca's pattern; we mirror
    # that by saving per-region npys here so the rendering script can pick them up later)
    svca_results: dict[str, SVCAResult] = {}
    for roi in sorted(available):
        X = binned.spikes_by_roi[roi]
        res = run_region_svca(X, region=roi, random_state=random_state,
                              n_components=n_components)
        svca_results[roi] = res
        # Persist scov / varcov / scores so render_figures.py can rebuild figures from cache
        np.save(cache_dir / f"{eid}_svca_scores_{roi}.npy", res.scores)
        np.save(cache_dir / f"{eid}_svca_scov_{roi}.npy",   res.scov)
        np.save(cache_dir / f"{eid}_svca_varcov_{roi}.npy", res.varcov)

    # Filter pair_rois to only those where both regions were SVCA-able this session
    runnable_pairs = [(a, b) for (a, b) in pair_rois
                       if a in svca_results and b in svca_results]

    cca_df = pd.DataFrame()
    if runnable_pairs:
        cca_df = run_session_cca(
            binned, svca_results, cache_dir=cache_dir,
            pair_rois=runnable_pairs, n_components=n_components,
            n_surrogates=n_surrogates, n_splits=n_splits,
            random_state=random_state,
        )

    # Per-session SVCA summary (one row per region) for downstream reporting
    summary_rows = [
        {
            "eid": eid, "region": roi,
            "n_units": int(binned.spikes_by_roi[roi].shape[0]),
            "k_reliable": int(res.k_reliable),
            "rho_top1": float(res.reliability[0]) if len(res.reliability) else float("nan"),
        }
        for roi, res in svca_results.items()
    ]
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            cache_dir / f"{eid}_svca_summary.csv", index=False
        )

    return svca_results, cca_df
