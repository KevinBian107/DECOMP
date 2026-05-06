"""Lag + peri-event analysis on residual canonical variates after Z_rich partialling.

For each cached pair-session:
  - Load SVCA scores per region, binned covariates, bin centers.
  - Build Z_rich and extract the leading residual canonical variates U_A(t), U_B(t).
  - Compute the cross-correlation function with phase-shuffle null.
  - Load trials via ONE (uses local cache) and average U(t) around stim-onset / first-
    movement / feedback events.
  - Save per-session npz.

After the per-session loop, aggregate across sessions per pair (median CCF, mean PETH)
and render the cross-pair dynamics figure.

Usage:
    PYTHONPATH=src python scripts/run_dynamics.py
"""

from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

from decomp.cca.dynamics import (
    cross_correlation_function,
    ccf_phase_null,
    peri_event_average,
)
from decomp.cca.pcca import cv_residual_variates
from decomp.cca.richz import Z_RICH_COLS
from decomp.viz.figures import fig_dynamics

CACHE = Path("data/cache")
OUT = Path("outputs")
MAX_LAG_BINS = 25       # ±25 bins × 20 ms = ±500 ms
BIN_S = 0.020
WINDOW_S = 1.5          # ±1.5 s peri-event window


def _list_pair_sessions() -> list[str]:
    return sorted({Path(p).name.split("_cca_results.csv")[0]
                   for p in glob.glob(str(CACHE / "*_cca_results.csv"))})


def _load_z_rich(eid: str) -> tuple[np.ndarray | None, list[str]]:
    cov_path = CACHE / f"{eid}_covariates.parquet"
    if not cov_path.exists():
        return None, []
    cov = pd.read_parquet(cov_path)
    use = [c for c in Z_RICH_COLS if c in cov.columns and cov[c].std() > 0]
    if not use:
        return None, []
    Z = cov[use].to_numpy(dtype=float)
    Z = (Z - Z.mean(axis=0, keepdims=True)) / (Z.std(axis=0, keepdims=True) + 1e-9)
    return Z, use


def _load_trials(eid: str) -> dict[str, np.ndarray]:
    """Load trial event times via ONE (local cache hit if data already downloaded).

    Returns dict mapping event-type -> array of times.
    """
    cache_path = CACHE / f"{eid}_trials.parquet"
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
    else:
        from one.api import ONE  # local import to keep startup quick when cache hits
        one = ONE(base_url="https://openalyx.internationalbrainlab.org",
                   password="international", silent=True)
        from brainbox.io.one import SessionLoader
        sl = SessionLoader(one=one, eid=eid)
        sl.load_trials()
        df = sl.trials
        if df is not None:
            df.to_parquet(cache_path)
    if df is None:
        return {}
    out = {}
    for k_in, k_out in [("stimOn_times",        "stim_on"),
                          ("firstMovement_times", "first_move"),
                          ("feedback_times",      "feedback")]:
        if k_in in df.columns:
            out[k_out] = df[k_in].to_numpy()
    return out


def _process_session(eid: str, n_surrogates: int = 50) -> dict | None:
    print(f"  {eid[:8]}", end="", flush=True)
    cca_path = CACHE / f"{eid}_cca_results.csv"
    bin_centers_path = CACHE / f"{eid}_bin_centers.npy"
    if not cca_path.exists() or not bin_centers_path.exists():
        print("  (skipped, missing cache)")
        return None
    cca = pd.read_csv(cca_path)
    bin_centers = np.load(bin_centers_path)
    Z_rich, _ = _load_z_rich(eid)
    trials = _load_trials(eid)
    if Z_rich is None or not trials:
        print("  (skipped, missing Z or trials)")
        return None

    out = {"eid": eid, "bin_s": BIN_S, "pairs": []}
    for (a, b), _ in cca.groupby(["pair_a", "pair_b"]):
        sa_p = CACHE / f"{eid}_svca_scores_{a}.npy"
        sb_p = CACHE / f"{eid}_svca_scores_{b}.npy"
        if not sa_p.exists() or not sb_p.exists():
            continue
        Sa, Sb = np.load(sa_p), np.load(sb_p)
        T = min(Sa.shape[0], Sb.shape[0], Z_rich.shape[0], len(bin_centers))
        Sa, Sb, Z, bc = Sa[:T], Sb[:T], Z_rich[:T], bin_centers[:T]

        # Residual variates (leading canonical component only)
        U_A_resid, U_B_resid = cv_residual_variates(Sa, Sb, Z=Z, n_components=8)
        U_A_raw,   U_B_raw   = cv_residual_variates(Sa, Sb, Z=None, n_components=8)
        u_a_r, u_b_r = U_A_resid[:, 0], U_B_resid[:, 0]
        u_a_x, u_b_x = U_A_raw[:, 0],   U_B_raw[:, 0]

        # Lag analysis
        lags, ccf_resid = cross_correlation_function(u_a_r, u_b_r, max_lag_bins=MAX_LAG_BINS)
        _,    ccf_raw   = cross_correlation_function(u_a_x, u_b_x, max_lag_bins=MAX_LAG_BINS)
        nulls_resid = ccf_phase_null(u_a_r, u_b_r, max_lag_bins=MAX_LAG_BINS,
                                       n_surrogates=n_surrogates)
        null_99 = np.quantile(np.abs(nulls_resid), 0.99, axis=0)

        # PETH per event type, residual variate (averaged across regions)
        u_avg_resid = 0.5 * (u_a_r + u_b_r)
        u_avg_raw   = 0.5 * (u_a_x + u_b_x)
        peth = {}
        for ev, et in trials.items():
            t_ax, m_r, s_r, n_r = peri_event_average(u_avg_resid, et, bc, window_s=WINDOW_S)
            _,    m_x, s_x, _   = peri_event_average(u_avg_raw,   et, bc, window_s=WINDOW_S)
            peth[ev] = {"t_axis": t_ax, "mean_resid": m_r, "sem_resid": s_r,
                          "mean_raw": m_x, "sem_raw": s_x, "n_events": n_r}

        out["pairs"].append({
            "pair_a": a, "pair_b": b,
            "lags": lags, "ccf_resid": ccf_resid, "ccf_raw": ccf_raw,
            "ccf_null_99": null_99, "peth": peth,
        })
    print(f"  {len(out['pairs'])} pair(s)")
    return out


def _aggregate(per_session: list[dict]) -> dict:
    """Per pair, stack CCFs and PETHs across sessions."""
    pairs: dict[tuple[str, str], dict] = {}
    for sess in per_session:
        for p in sess["pairs"]:
            key = (p["pair_a"], p["pair_b"])
            pairs.setdefault(key, {"ccf_resid": [], "ccf_raw": [], "ccf_null_99": [],
                                      "peth_resid": {}, "peth_raw": {}, "lags": p["lags"]})
            pairs[key]["ccf_resid"].append(p["ccf_resid"])
            pairs[key]["ccf_raw"].append(p["ccf_raw"])
            pairs[key]["ccf_null_99"].append(p["ccf_null_99"])
            for ev, d in p["peth"].items():
                pairs[key]["peth_resid"].setdefault(ev,
                    {"t_axis": d["t_axis"], "vals": []})["vals"].append(d["mean_resid"])
                pairs[key]["peth_raw"].setdefault(ev,
                    {"t_axis": d["t_axis"], "vals": []})["vals"].append(d["mean_raw"])
    # collapse to medians / means
    for key, d in pairs.items():
        d["ccf_resid"]   = np.nanmedian(np.array(d["ccf_resid"]),   axis=0)
        d["ccf_raw"]     = np.nanmedian(np.array(d["ccf_raw"]),     axis=0)
        d["ccf_null_99"] = np.nanmedian(np.array(d["ccf_null_99"]), axis=0)
        for ev in d["peth_resid"]:
            arr = np.array(d["peth_resid"][ev]["vals"])
            d["peth_resid"][ev]["mean"] = np.nanmean(arr, axis=0)
            d["peth_resid"][ev]["sem"]  = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(len(arr))
            arr2 = np.array(d["peth_raw"][ev]["vals"])
            d["peth_raw"][ev]["mean"] = np.nanmean(arr2, axis=0)
            d["peth_raw"][ev]["sem"]  = np.nanstd(arr2, axis=0, ddof=1) / np.sqrt(len(arr2))
    return pairs


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    eids = _list_pair_sessions()
    print(f"Processing {len(eids)} pair-sessions:")
    per_session = []
    for eid in eids:
        try:
            r = _process_session(eid, n_surrogates=50)
        except Exception as exc:
            print(f"  {eid[:8]}  ERROR: {exc}")
            continue
        if r is not None:
            per_session.append(r)
    if not per_session:
        print("No sessions processed.")
        return
    agg = _aggregate(per_session)
    print(f"\nAggregated {sum(1 for s in per_session)} sessions across "
          f"{len(agg)} pair types.")
    fig_dynamics(agg, OUT, bin_s=BIN_S)
    print(f"Wrote {OUT / 'dynamics.png'}")


if __name__ == "__main__":
    main()
