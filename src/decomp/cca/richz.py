"""Richer-Z stress test for the pCCA survival result.

Re-runs pCCA only (CCA itself is deterministic from cached SVCA scores) under a richer
confound set Z_rich = [wheel_velocity, wheel_acceleration, me_whisker, pupil, lick_rate]
and compares against the baseline Z_min = [wheel_velocity, pupil].

The point is to ask whether the surviving-shared-subspace result (median survival ≈ 0.97
across all three pairs in the expansion run) is robust to including more behavioral
covariates that operationalize "global movement state," or whether wheel + pupil were too
narrow a Z and the surviving structure would shrink under a fuller proxy.

Reads from data/cache/:
    {eid}_svca_scores_{roi}.npy
    {eid}_covariates.parquet
    {eid}_cca_results.csv      (for the rho_cca column we re-attach)

Writes:
    {eid}_cca_richz.csv        long-form, columns:
        eid, pair_a, pair_b, component,
        rho_cca, rho_pcca_min, rho_pcca_rich,
        null_pcca_min_99, null_pcca_rich_99,
        n_z_min, n_z_rich
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .nulls import shuffle_null
from .pcca import cv_canonical_correlations

# Z columns that exist on cached covariates. Robust to columns being absent on some sessions.
Z_MIN_COLS = ("wheel_velocity", "pupil")
Z_RICH_COLS = ("wheel_velocity", "wheel_acceleration", "me_whisker", "me_body",
                "pupil", "lick_rate")


def _z_matrix(cov: pd.DataFrame, cols: Iterable[str]) -> tuple[np.ndarray | None, list[str]]:
    """Standardize the requested columns, drop any that are missing or constant."""
    use = [c for c in cols if c in cov.columns and cov[c].std() > 0]
    if not use:
        return None, []
    Z = cov[use].to_numpy(dtype=float)
    Z = (Z - Z.mean(axis=0, keepdims=True)) / (Z.std(axis=0, keepdims=True) + 1e-9)
    return Z, use


def _list_pair_sessions(cache_dir: Path) -> list[str]:
    """All eids that have a *_cca_results.csv on disk (i.e. were part of the expansion)."""
    return sorted({p.name.split("_cca_results.csv")[0]
                   for p in cache_dir.glob("*_cca_results.csv")})


def _scores_for(eid: str, roi: str, cache_dir: Path) -> np.ndarray | None:
    p = cache_dir / f"{eid}_svca_scores_{roi}.npy"
    if not p.exists():
        return None
    arr = np.load(p)
    return arr if arr.size else None


def rerun_session(eid: str, cache_dir: Path = Path("data/cache"),
                  n_components: int = 8, n_splits: int = 5,
                  n_surrogates: int = 200, random_state: int = 0) -> pd.DataFrame:
    """Re-run pCCA under both Z specs on one session, return the long-form richz frame.

    CCA itself is recomputed too (cheap on K=8 scores) so the output is fully self-contained.
    """
    cache_dir = Path(cache_dir)
    cca_path = cache_dir / f"{eid}_cca_results.csv"
    cov_path = cache_dir / f"{eid}_covariates.parquet"
    if not cca_path.exists() or not cov_path.exists():
        return pd.DataFrame()

    base = pd.read_csv(cca_path)
    cov = pd.read_parquet(cov_path)
    Z_min, used_min = _z_matrix(cov, Z_MIN_COLS)
    Z_rich, used_rich = _z_matrix(cov, Z_RICH_COLS)

    # T from any score array; check covariates align (caller guarantees they do)
    T_cov = len(cov)
    rows: list[dict] = []
    for (a, b), pair_df in base.groupby(["pair_a", "pair_b"]):
        Sa = _scores_for(eid, a, cache_dir)
        Sb = _scores_for(eid, b, cache_dir)
        if Sa is None or Sb is None:
            continue
        # Trim to covariate length (caches are aligned by construction; defensive trim)
        T = min(Sa.shape[0], Sb.shape[0], T_cov)
        Sa, Sb = Sa[:T], Sb[:T]
        Z_m = Z_min[:T] if Z_min is not None else None
        Z_r = Z_rich[:T] if Z_rich is not None else None

        n_comp = min(n_components, Sa.shape[1], Sb.shape[1])
        cca = cv_canonical_correlations(Sa, Sb, Z=None, n_components=n_comp,
                                          n_splits=n_splits, random_state=random_state)
        pcca_min = cv_canonical_correlations(Sa, Sb, Z=Z_m, n_components=n_comp,
                                               n_splits=n_splits, random_state=random_state)
        pcca_rich = cv_canonical_correlations(Sa, Sb, Z=Z_r, n_components=n_comp,
                                                n_splits=n_splits, random_state=random_state)
        null_min = shuffle_null(Sa, Sb, Z=Z_m, n_components=n_comp,
                                  n_surrogates=n_surrogates, n_splits=n_splits,
                                  random_state=random_state)
        null_rich = shuffle_null(Sa, Sb, Z=Z_r, n_components=n_comp,
                                   n_surrogates=n_surrogates, n_splits=n_splits,
                                   random_state=random_state)
        for k in range(n_comp):
            rows.append({
                "eid": eid, "pair_a": a, "pair_b": b, "component": k,
                "rho_cca":           float(cca["rho_mean"][k]),
                "rho_pcca_min":      float(pcca_min["rho_mean"][k]),
                "rho_pcca_rich":     float(pcca_rich["rho_mean"][k]),
                "null_pcca_min_99":  float(null_min["null_99"][k]),
                "null_pcca_rich_99": float(null_rich["null_99"][k]),
                "n_z_min": len(used_min),
                "n_z_rich": len(used_rich),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df.attrs["z_min_cols"] = used_min
        df.attrs["z_rich_cols"] = used_rich
        df.to_csv(cache_dir / f"{eid}_cca_richz.csv", index=False)
    return df


def rerun_all(cache_dir: Path = Path("data/cache"), **kwargs) -> pd.DataFrame:
    """Walk every cached session and re-run pCCA under both Z specs."""
    cache_dir = Path(cache_dir)
    eids = _list_pair_sessions(cache_dir)
    frames = []
    for i, eid in enumerate(eids):
        print(f"[{i+1}/{len(eids)}] {eid[:8]} ", end="", flush=True)
        df = rerun_session(eid, cache_dir=cache_dir, **kwargs)
        if df.empty:
            print("(skipped — missing cache)")
            continue
        n_pairs = df[["pair_a", "pair_b"]].drop_duplicates().shape[0]
        print(f"  {n_pairs} pair(s)  Z_min={int(df['n_z_min'].iloc[0])}D  "
              f"Z_rich={int(df['n_z_rich'].iloc[0])}D")
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
