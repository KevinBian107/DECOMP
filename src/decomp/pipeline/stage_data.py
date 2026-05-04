"""Stage 1+2: session selection, data loading, and binning."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.binning import BIN_S, BinnedSession, bin_session
from ..data.load import load_session
from ..data.sessions import SessionPlan, adapt_strategy, query_unit_table


def select_sessions(one, target_n: int = 3, freeze: str = "2023_12_bwm_release",
                    cache_dir: Path = Path("data/cache")) -> SessionPlan:
    """Run the Gate-1 fallback ladder and cache the resulting plan."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "session_plan.json"

    if cache_path.exists():
        payload = json.loads(cache_path.read_text())
        cov_df = pd.DataFrame(payload["coverage"])
        return SessionPlan(
            eids=payload["eids"],
            coverage=cov_df,
            min_units_used=payload["min_units_used"],
            rois_used=payload["rois_used"],
            strategy_note=payload["strategy_note"],
            pair_eids=payload.get("pair_eids", []),
            pool_eids=payload.get("pool_eids", {}),
        )

    unit_df = query_unit_table(one=one, freeze=freeze)
    plan = adapt_strategy(unit_df, target_n=target_n)
    payload = {
        "eids": plan.eids,
        "coverage": plan.coverage.to_dict(orient="list"),
        "min_units_used": plan.min_units_used,
        "rois_used": plan.rois_used,
        "strategy_note": plan.strategy_note,
        "pair_eids": plan.pair_eids,
        "pool_eids": plan.pool_eids,
    }
    cache_path.write_text(json.dumps(payload, indent=2))
    return plan


def load_and_bin(one, eid: str, pids: list[str], rois: list[str],
                 cache_dir: Path = Path("data/cache")):
    """Load + bin one session. Returns (SessionData, BinnedSession) and writes a cache."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    sd = load_session(eid=eid, pids=pids, one=one, rois=rois)
    binned = bin_session(sd, bin_s=BIN_S)

    binned.covariates.to_parquet(cache_dir / f"{eid}_covariates.parquet")
    for roi, mat in binned.spikes_by_roi.items():
        np.save(cache_dir / f"{eid}_spikes_{roi}.npy", mat)
    np.save(cache_dir / f"{eid}_bin_centers.npy", binned.bin_centers)
    return sd, binned
