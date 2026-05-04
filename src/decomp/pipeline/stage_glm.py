"""Stage 3: per-session GLM ΔR² across all four regions."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..data.binning import BinnedSession
from ..data.load import SessionData
from ..data.sessions import ROI_ORDER
from ..glm.run_session import fit_session


def run_session_glm(sd: SessionData, binned: BinnedSession,
                    rois: list[str] = ROI_ORDER,
                    cache_dir: Path = Path("data/cache"),
                    random_state: int = 0) -> pd.DataFrame:
    cache_dir = Path(cache_dir)
    cache_path = cache_dir / f"{sd.eid}_glm_results.csv"
    if cache_path.exists():
        return pd.read_csv(cache_path)
    return fit_session(sd, binned, rois=rois, cache_dir=cache_dir, random_state=random_state)
