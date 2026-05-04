"""Stage 4: within-region SVCA across sessions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.binning import BinnedSession
from ..svca.run_svca import SVCAResult, run_all_regions


def run_session_svca(binned: BinnedSession, cache_dir: Path = Path("data/cache"),
                     threshold: float = 0.5, random_state: int = 0,
                     ) -> dict[str, SVCAResult]:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    results = run_all_regions(binned.spikes_by_roi, threshold=threshold,
                              random_state=random_state)
    summary = []
    for roi, res in results.items():
        np.save(cache_dir / f"{binned.eid}_svca_scores_{roi}.npy", res.scores)
        np.save(cache_dir / f"{binned.eid}_svca_scov_{roi}.npy", res.scov)
        np.save(cache_dir / f"{binned.eid}_svca_varcov_{roi}.npy", res.varcov)
        summary.append({
            "eid": binned.eid,
            "region": roi,
            "k_reliable": res.k_reliable,
            "n_units": int(binned.spikes_by_roi[roi].shape[0]),
        })
    pd.DataFrame(summary).to_csv(cache_dir / f"{binned.eid}_svca_summary.csv", index=False)
    return results
