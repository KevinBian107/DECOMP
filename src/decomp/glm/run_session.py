"""Per-session GLM driver: fit every neuron in {V1, CB, M1, CA1} and compile a tidy DataFrame."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..data.binning import BinnedSession
from ..data.load import SessionData
from ..data.sessions import ROI_ORDER
from .design import (
    CHOICE_GROUPS,
    MOVEMENT_GROUPS,
    STIMULUS_GROUPS,
    build_design_matrix,
)
from .fit import fit_region


def _aggregate_groups(deltas: dict[str, float], groups: tuple[str, ...]) -> float:
    """Sum ΔR² across a set of related groups (e.g. all movement-related kernels)."""
    return float(sum(deltas.get(g, 0.0) for g in groups))


def fit_session(sd: SessionData, binned: BinnedSession,
                rois: list[str] = ROI_ORDER,
                cache_dir: Path = Path("data/cache"),
                random_state: int = 0) -> pd.DataFrame:
    """Loop over all neurons in the requested ROIs and return one DataFrame."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    X, design = build_design_matrix(binned, sd.trials)

    rows: list[dict] = []
    for roi in rois:
        if roi not in binned.spikes_by_roi:
            continue
        Y_int = binned.spikes_by_roi[roi]                # (n_units, T) int16
        Y = Y_int.astype(np.float64).T                    # (T, n_units)
        meta = sd.unit_meta_by_roi.get(roi, pd.DataFrame())

        # Drop silent neurons (no variance)
        active = Y.std(axis=0) > 1e-9
        if not active.any():
            continue

        Y_active = Y[:, active]
        active_idx = np.where(active)[0]

        for label in tqdm([roi], desc=f"GLM {sd.eid[:8]} {roi} ({Y_active.shape[1]} units)"):
            res = fit_region(X, Y_active, design, n_splits=5, random_state=random_state)

        for j, u_idx in enumerate(active_idx):
            deltas = {g: float(res.deltas[g][j]) for g in res.deltas}
            row = {
                "eid": sd.eid,
                "region": roi,
                "unit_idx": int(u_idx),
                "cluster_id": (meta.iloc[u_idx]["cluster_id"]
                               if u_idx < len(meta) and "cluster_id" in meta.columns else u_idx),
                "full_R2": float(res.full_R2[j]),
                "alpha": float(res.alpha),
                "dR2_movement": _aggregate_groups(deltas, MOVEMENT_GROUPS),
                "dR2_stim": _aggregate_groups(deltas, STIMULUS_GROUPS),
                "dR2_choice": _aggregate_groups(deltas, CHOICE_GROUPS),
            }
            for g, v in deltas.items():
                row[f"dR2_{g}"] = v
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(cache_dir / f"{sd.eid}_glm_results.csv", index=False)
    return df
