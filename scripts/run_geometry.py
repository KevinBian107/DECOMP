"""Representation-geometry visualizations on cached pair-sessions.

Runs two geometry analyses on already-cached SVCA scores + covariates + trials,
aggregates per pair (V1↔CB, V1↔M1, CB↔M1), and renders three figures:

    outputs/geometry_state_space.png   joint state-space density vs wheel quantile
    outputs/geometry_rsa.png           cross-region RSA per pair
    outputs/geometry_schematic.png     pure illustration (no data needed)

Usage:
    PYTHONPATH=src python scripts/run_geometry.py
"""

from __future__ import annotations

import glob
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from decomp.cca.geometry import joint_state_space_2d, cross_region_rsa
from decomp.cca.pcca import cv_residual_variates
from decomp.cca.richz import Z_RICH_COLS
from decomp.viz.geometry_figures import (
    fig_state_space_density, fig_cross_region_rsa, fig_manifold_schematic,
    PAIR_ORDER,
)

CACHE = Path("data/cache")
OUT = Path("outputs")
BIN_S = 0.02
RSA_POST_S = 0.4


def _list_pair_sessions() -> list[str]:
    return sorted({Path(p).name.split("_cca_results.csv")[0]
                   for p in glob.glob(str(CACHE / "*_cca_results.csv"))})


def _load_z(cov: pd.DataFrame, cols) -> tuple[np.ndarray | None, list[str]]:
    use = [c for c in cols if c in cov.columns and cov[c].std() > 0]
    if not use:
        return None, []
    Z = cov[use].to_numpy(dtype=float)
    Z = (Z - Z.mean(axis=0, keepdims=True)) / (Z.std(axis=0, keepdims=True) + 1e-9)
    return Z, use


def _signed_contrast_bin(cL: float, cR: float) -> int:
    """Six-bin signed contrast (+:right, -:left)."""
    sc = (cR if np.isfinite(cR) else 0.0) - (cL if np.isfinite(cL) else 0.0)
    edges = np.array([-1.0, -0.25, -0.0625, 0.0, 0.0625, 0.25, 1.0])
    return int(np.searchsorted(edges, sc, side="right") - 1)


def _trial_labels_from_table(trials: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Per-trial stim-onset time and a discrete condition label combining feedback + side."""
    need = {"stimOn_times", "feedbackType", "contrastLeft", "contrastRight"}
    if not need.issubset(trials.columns):
        return np.array([]), np.array([])
    t = trials["stimOn_times"].to_numpy()
    fb = trials["feedbackType"].to_numpy()
    cL = trials["contrastLeft"].to_numpy()
    cR = trials["contrastRight"].to_numpy()
    side = np.where(cR > cL, "R", "L")
    res  = np.where(fb > 0, "+", "-")
    abs_c = np.maximum(np.where(np.isfinite(cL), cL, 0),
                          np.where(np.isfinite(cR), cR, 0))
    bin_c = np.where(abs_c <= 0.0,  "0",
              np.where(abs_c <= 0.0625, "lo",
              np.where(abs_c <= 0.25,   "mi", "hi")))
    label = np.array([f"{s}{c}{r}" for s, c, r in zip(side, bin_c, res)])
    valid = np.isfinite(t)
    return t[valid], label[valid]


def _process_session(eid: str) -> dict | None:
    cca_path = CACHE / f"{eid}_cca_results.csv"
    cov_path = CACHE / f"{eid}_covariates.parquet"
    bc_path  = CACHE / f"{eid}_bin_centers.npy"
    trial_path = CACHE / f"{eid}_trials.parquet"
    if not (cca_path.exists() and cov_path.exists() and bc_path.exists()
            and trial_path.exists()):
        print(f"  {eid[:8]}  (skip — missing cache)")
        return None
    cca = pd.read_csv(cca_path)
    cov = pd.read_parquet(cov_path)
    bc  = np.load(bc_path)
    trials = pd.read_parquet(trial_path)
    Z_rich, _ = _load_z(cov, Z_RICH_COLS)
    wheel = cov["wheel_velocity"].to_numpy() if "wheel_velocity" in cov.columns else None
    t_trial, lab_trial = _trial_labels_from_table(trials)

    out = {"eid": eid, "pairs": []}
    for (a, b), _ in cca.groupby(["pair_a", "pair_b"]):
        sa_p = CACHE / f"{eid}_svca_scores_{a}.npy"
        sb_p = CACHE / f"{eid}_svca_scores_{b}.npy"
        if not (sa_p.exists() and sb_p.exists()):
            continue
        Sa, Sb = np.load(sa_p), np.load(sb_p)
        T = min(Sa.shape[0], Sb.shape[0], len(bc),
                  Z_rich.shape[0] if Z_rich is not None else len(bc))
        Sa, Sb, bc_ = Sa[:T], Sb[:T], bc[:T]
        Zr = Z_rich[:T] if Z_rich is not None else None
        wh = wheel[:T]  if wheel  is not None else None

        # Residual canonical variates (leading component) — both raw and rich-Z
        U_raw_a, U_raw_b   = cv_residual_variates(Sa, Sb, Z=None, n_components=8)
        U_rich_a, U_rich_b = cv_residual_variates(Sa, Sb, Z=Zr,   n_components=8)
        u_raw_a, u_raw_b   = U_raw_a[:, 0],   U_raw_b[:, 0]
        u_rich_a, u_rich_b = U_rich_a[:, 0],  U_rich_b[:, 0]

        # 1. State-space density (raw and rich) coloured by wheel quantile
        density_raw = (joint_state_space_2d(u_raw_a,  u_raw_b,  wh, n_quantile=4)
                       if wh is not None else None)
        density_rich = (joint_state_space_2d(u_rich_a, u_rich_b, wh, n_quantile=4)
                        if wh is not None else None)

        # 2. Cross-region RSA over trial conditions, raw + rich-Z partialled
        rsa_raw = (cross_region_rsa(Sa, Sb, bc_, t_trial, lab_trial,
                                      Z=None, post_s=RSA_POST_S)
                   if len(t_trial) > 10 else None)
        rsa_rich = (cross_region_rsa(Sa, Sb, bc_, t_trial, lab_trial,
                                       Z=Zr, post_s=RSA_POST_S)
                    if len(t_trial) > 10 and Zr is not None else None)

        out["pairs"].append({
            "pair_a": a, "pair_b": b,
            "density_raw":  density_raw,
            "density_rich": density_rich,
            "rsa_raw":  rsa_raw,
            "rsa_rich": rsa_rich,
        })
    print(f"  {eid[:8]}  {len(out['pairs'])} pair(s)")
    return out


def _align_and_mean_rdm(rdms: list[np.ndarray], per_sess_conds: list) -> np.ndarray:
    """Reindex every session's RDM into a global label×label grid (NaN for missing
    condition pairs) and take pixel-wise nanmean."""
    if not rdms:
        return np.zeros((0, 0))
    all_conds = sorted({c for cl in per_sess_conds for c in cl})
    idx = {c: i for i, c in enumerate(all_conds)}
    n = len(all_conds)
    stack = np.full((len(rdms), n, n), np.nan)
    for s, (rdm, conds) in enumerate(zip(rdms, per_sess_conds)):
        if rdm.size == 0 or len(conds) == 0:
            continue
        rows = [idx[c] for c in conds]
        for i, ri in enumerate(rows):
            for j, rj in enumerate(rows):
                stack[s, ri, rj] = rdm[i, j]
    return np.nanmean(stack, axis=0)


def _aggregate(per_session: list[dict]) -> dict:
    """For each pair, aggregate across sessions:
       Density → concatenate clouds + (a, b, z_std) arrays across sessions
       RSA → mean across sessions (Fisher-z for the correlation, label-aligned RDMs)
    """
    per_pair: dict = {}
    for sess in per_session:
        for p in sess["pairs"]:
            key = (p["pair_a"], p["pair_b"])
            acc = per_pair.setdefault(key, {
                "density_raw_clouds": [],
                "density_rich_clouds": [],
                "density_raw_rho":  [],
                "density_rich_rho": [],
                "density_raw_abz": [],
                "density_rich_abz": [],
                "rsa_raw_r": [], "rsa_rich_r": [],
                "rsa_raw_rdm_a": [], "rsa_raw_rdm_b": [], "rsa_raw_conds": None,
                "rsa_rich_rdm_a": [], "rsa_rich_rdm_b": [],
            })
            if p["density_raw"] is not None:
                acc["density_raw_clouds"].append(p["density_raw"]["clouds"])
                acc["density_raw_rho"].append(p["density_raw"]["rho"])
                acc["density_raw_abz"].append(
                    (p["density_raw"]["a"], p["density_raw"]["b"],
                     p["density_raw"]["z_std"]))
            if p["density_rich"] is not None:
                acc["density_rich_clouds"].append(p["density_rich"]["clouds"])
                acc["density_rich_rho"].append(p["density_rich"]["rho"])
                acc["density_rich_abz"].append(
                    (p["density_rich"]["a"], p["density_rich"]["b"],
                     p["density_rich"]["z_std"]))
            if p["rsa_raw"] is not None:
                acc["rsa_raw_r"].append(p["rsa_raw"]["spearman_r"])
                acc["rsa_raw_rdm_a"].append(p["rsa_raw"]["rdm_a"])
                acc["rsa_raw_rdm_b"].append(p["rsa_raw"]["rdm_b"])
                acc["rsa_raw_conds"] = p["rsa_raw"]["conds"]
            if p["rsa_rich"] is not None:
                acc["rsa_rich_r"].append(p["rsa_rich"]["spearman_r"])
                acc["rsa_rich_rdm_a"].append(p["rsa_rich"]["rdm_a"])
                acc["rsa_rich_rdm_b"].append(p["rsa_rich"]["rdm_b"])

    # collapse each pair into the form the figure functions expect
    collapsed: dict = {}
    for pair, acc in per_pair.items():
        d: dict = {}
        # Density: concatenate clouds across sessions per quantile bin AND
        # collect raw (a, b, z_std) arrays for the hexbin colouring
        for label, src_clouds, src_rho, src_abz in [
            ("density_raw",  acc["density_raw_clouds"],  acc["density_raw_rho"],
                acc["density_raw_abz"]),
            ("density_rich", acc["density_rich_clouds"], acc["density_rich_rho"],
                acc["density_rich_abz"]),
        ]:
            if not src_clouds:
                d[label] = None
                continue
            n_q = len(src_clouds[0])
            merged = [np.concatenate([sess[i] for sess in src_clouds if len(sess) > i],
                                       axis=1)
                      for i in range(n_q)]
            a_all = np.concatenate([abz[0] for abz in src_abz])
            b_all = np.concatenate([abz[1] for abz in src_abz])
            z_all = np.concatenate([abz[2] for abz in src_abz])
            d[label] = {"clouds": merged,
                          "rho": float(np.nanmean(src_rho)) if src_rho else 0.0,
                          "a": a_all, "b": b_all, "z_std": z_all}
        # RSA: mean of Fisher-z-transformed correlations + label-aligned RDMs
        def _agg_rsa(rs, rdms_a, rdms_b, per_sess_conds) -> dict:
            if not rs:
                return {"spearman_r": np.nan, "rdm_a": np.zeros((0, 0)),
                        "rdm_b": np.zeros((0, 0)), "conds": np.array([])}
            z = np.arctanh(np.clip(rs, -0.999, 0.999))
            r = float(np.tanh(np.nanmean(z)))
            rdm_a = _align_and_mean_rdm(rdms_a, per_sess_conds)
            rdm_b = _align_and_mean_rdm(rdms_b, per_sess_conds)
            # Combined condition list = union, ordered
            all_conds = sorted({c for cl in per_sess_conds for c in cl})
            return {"spearman_r": r, "rdm_a": rdm_a, "rdm_b": rdm_b,
                    "conds": np.array(all_conds)}

        # Per-session condition labels for RSA
        raw_conds_per_sess = [p.get("rsa_raw", {}).get("conds", np.array([]))
                                for sess in per_session for p in sess["pairs"]
                                if (p["pair_a"], p["pair_b"]) == pair
                                and p.get("rsa_raw")]
        rich_conds_per_sess = [p.get("rsa_rich", {}).get("conds", np.array([]))
                                 for sess in per_session for p in sess["pairs"]
                                 if (p["pair_a"], p["pair_b"]) == pair
                                 and p.get("rsa_rich")]
        d["rsa_raw"]  = _agg_rsa(acc["rsa_raw_r"],  acc["rsa_raw_rdm_a"],
                                   acc["rsa_raw_rdm_b"],  raw_conds_per_sess)
        d["rsa_rich"] = _agg_rsa(acc["rsa_rich_r"], acc["rsa_rich_rdm_a"],
                                   acc["rsa_rich_rdm_b"], rich_conds_per_sess)
        collapsed[pair] = d
    return collapsed


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    OUT.mkdir(parents=True, exist_ok=True)
    eids = _list_pair_sessions()
    print(f"Processing {len(eids)} pair-sessions:")
    per_session = []
    for eid in eids:
        try:
            r = _process_session(eid)
        except Exception as exc:  # noqa: BLE001
            print(f"  {eid[:8]}  ERROR: {exc!r}")
            continue
        if r is not None:
            per_session.append(r)
    if not per_session:
        print("No sessions processed.")
        return
    per_pair = _aggregate(per_session)
    print(f"\nAggregated {len(per_session)} sessions across {len(per_pair)} pair types.")

    print("Writing figures...")
    fig_state_space_density(per_pair, OUT)
    fig_cross_region_rsa(per_pair, OUT)
    fig_manifold_schematic(OUT)
    print(f"Wrote 3 figures to {OUT.resolve()}")


if __name__ == "__main__":
    main()
