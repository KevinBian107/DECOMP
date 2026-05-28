"""Animate all three pair joint state-space heatmaps fading from raw CCA to rich-Z pCCA.

3 panels side-by-side, one per pair (V1↔CB, V1↔M1, CB↔M1). At each frame the partial
coefficient alpha controls how much of Z is partialled out of each region's SVCA
scores: X_alpha = X - alpha * Z @ pinv(Z) @ X (per-fold, no leakage). alpha = 0
reduces to raw CCA, alpha = 1 to the full rich-Z partial CCA. The cloud shapes and
tight diagonal alignments barely shift across alpha; the wheel-velocity colour
gradient flattens visibly in every panel.

Output: outputs/geometry_morph.gif (ping-pong loop, alpha sweeps 0 -> 1 -> 0).

Usage:
    PYTHONPATH=src python scripts/run_geometry_morph.py
"""

from __future__ import annotations

import glob
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from decomp.cca.pcca import cca_svd
from decomp.cca.richz import Z_RICH_COLS
from decomp.viz.figures import _BASE_STYLE

mpl.rcParams.update(_BASE_STYLE)

CACHE = Path("data/cache")
OUT = Path("outputs")
PAIRS = [("VIS", "CB"), ("VIS", "MO"), ("CB", "MO")]
PAIR_LABEL = {("VIS", "CB"): r"V1 $\leftrightarrow$ CB",
               ("VIS", "MO"): r"V1 $\leftrightarrow$ M1",
               ("CB",  "MO"): r"CB $\leftrightarrow$ M1"}
PAIR_COLOR = {("VIS", "CB"): "#c44536",
               ("VIS", "MO"): "#3a8a4d",
               ("CB",  "MO"): "#c44536"}
PAIR_LABEL_AB = {("VIS", "CB"): ("V1", "CB"),
                  ("VIS", "MO"): ("V1", "M1"),
                  ("CB",  "MO"): ("CB", "M1")}
N_FRAMES = 24
PING_PONG = True


def _shrink_residualize(X: np.ndarray, Z: np.ndarray | None, alpha: float,
                          n_splits: int = 5, random_state: int = 0) -> np.ndarray:
    """X - alpha * Ẑ where Ẑ is OLS prediction of X from Z. Per-fold no leakage."""
    if Z is None or alpha == 0.0:
        return X.copy()
    Xr = np.empty_like(X)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for tr_idx, te_idx in kf.split(X):
        rx = LinearRegression(fit_intercept=True).fit(Z[tr_idx], X[tr_idx])
        Xr[te_idx] = X[te_idx] - alpha * rx.predict(Z[te_idx])
    return Xr


def _leading_variates(X: np.ndarray, Y: np.ndarray, Z: np.ndarray | None,
                       alpha: float, n_components: int = 8,
                       n_splits: int = 5, random_state: int = 0,
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Leading canonical variates U_A(t), U_B(t) with alpha-fractional Z-partialling."""
    Xa = _shrink_residualize(X, Z, alpha, n_splits=n_splits, random_state=random_state)
    Ya = _shrink_residualize(Y, Z, alpha, n_splits=n_splits, random_state=random_state)
    T = X.shape[0]
    n_comp = min(n_components, Xa.shape[1], Ya.shape[1])
    U_A = np.full(T, np.nan); U_B = np.full(T, np.nan)
    a_ref = None
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for tr_idx, te_idx in kf.split(Xa):
        Xtr, Xte = Xa[tr_idx], Xa[te_idx]
        Ytr, Yte = Ya[tr_idx], Ya[te_idx]
        _, A_x, B_y = cca_svd(Xtr, Ytr, n_components=n_comp)
        Xs_tr = (Xtr - Xtr.mean(axis=0)) @ A_x[:, 0]
        Ys_tr = (Ytr - Ytr.mean(axis=0)) @ B_y[:, 0]
        if Xs_tr.std() > 0 and Ys_tr.std() > 0:
            if np.corrcoef(Xs_tr, Ys_tr)[0, 1] < 0:
                B_y[:, 0] *= -1
        if a_ref is None:
            a_ref = A_x[:, 0].copy()
        elif np.dot(A_x[:, 0], a_ref) < 0:
            A_x[:, 0] *= -1
            B_y[:, 0] *= -1
        Xc = Xte - Xte.mean(axis=0)
        Yc = Yte - Yte.mean(axis=0)
        U_A[te_idx] = Xc @ A_x[:, 0]
        U_B[te_idx] = Yc @ B_y[:, 0]
    return U_A, U_B


def _load_z(cov: pd.DataFrame, cols) -> np.ndarray | None:
    use = [c for c in cols if c in cov.columns and cov[c].std() > 0]
    if not use:
        return None
    Z = cov[use].to_numpy(dtype=float)
    return (Z - Z.mean(axis=0)) / (Z.std(axis=0) + 1e-9)


def _diag_slope(a: np.ndarray, b: np.ndarray, z: np.ndarray) -> float:
    keep = np.isfinite(a) & np.isfinite(b) & np.isfinite(z)
    a, b, z = a[keep], b[keep], z[keep]
    d = (a + b) / np.sqrt(2)
    if d.std() == 0:
        return 0.0
    return float(np.cov(d, z)[0, 1] / np.var(d))


def _heatmap_data(a: np.ndarray, b: np.ndarray, z: np.ndarray,
                    lim: float = 2.4, nbin: int = 18) -> np.ndarray:
    keep = np.isfinite(a) & np.isfinite(b) & np.isfinite(z)
    a, b, z = a[keep], b[keep], z[keep]
    edges = np.linspace(-lim, lim, nbin + 1)
    sum_z, _, _ = np.histogram2d(a, b, bins=[edges, edges], weights=z)
    cnt, _, _ = np.histogram2d(a, b, bins=[edges, edges])
    return np.where(cnt >= 30, sum_z / np.maximum(cnt, 1), np.nan)


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- 1. Aggregate sessions per pair ----
    pair_data: dict[tuple[str, str], dict] = {p: {"Sa": [], "Sb": [], "Z": [], "wh": []}
                                                for p in PAIRS}
    for cca_csv in sorted(glob.glob(str(CACHE / "*_cca_results.csv"))):
        eid = Path(cca_csv).name.split("_cca_results.csv")[0]
        df = pd.read_csv(cca_csv)
        for a, b in PAIRS:
            if not any((df.pair_a == a) & (df.pair_b == b)):
                continue
            sa_p = CACHE / f"{eid}_svca_scores_{a}.npy"
            sb_p = CACHE / f"{eid}_svca_scores_{b}.npy"
            cov_p = CACHE / f"{eid}_covariates.parquet"
            if not (sa_p.exists() and sb_p.exists() and cov_p.exists()):
                continue
            sa = np.load(sa_p); sb = np.load(sb_p)
            cov = pd.read_parquet(cov_p)
            Z = _load_z(cov, Z_RICH_COLS)
            if Z is None:
                continue
            wh = (cov["wheel_velocity"].to_numpy()
                  if "wheel_velocity" in cov.columns else np.zeros(len(cov)))
            T = min(sa.shape[0], sb.shape[0], Z.shape[0], len(wh))
            pair_data[(a, b)]["Sa"].append(sa[:T])
            pair_data[(a, b)]["Sb"].append(sb[:T])
            pair_data[(a, b)]["Z"].append(Z[:T])
            pair_data[(a, b)]["wh"].append(wh[:T])

    for p in PAIRS:
        n = len(pair_data[p]["Sa"])
        print(f"  {PAIR_LABEL_AB[p][0]}↔{PAIR_LABEL_AB[p][1]}: {n} session(s)")

    # Wheel z-score PER SESSION first, then concatenate. Matches the static
    # figure's aggregation, which evens out per-session wheel-velocity scales
    # so vigorous runners don't dominate the per-pair gradient.
    wh_std_by_pair: dict[tuple[str, str], np.ndarray] = {}
    for p in PAIRS:
        per_sess_std = [(w - w.mean()) / (w.std() + 1e-9)
                          for w in pair_data[p]["wh"] if w.size]
        wh_std_by_pair[p] = (np.concatenate(per_sess_std)
                              if per_sess_std else np.array([]))

    # ---- 2. Precompute U_A, U_B at each alpha for every pair ----
    alphas = np.linspace(0.0, 1.0, N_FRAMES)
    per_frame_by_pair: dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]] = {
        p: [] for p in PAIRS
    }
    for i, alpha in enumerate(alphas):
        print(f"  alpha = {alpha:0.2f}  ({i + 1}/{N_FRAMES})",
                end="\r", flush=True)
        for p in PAIRS:
            d = pair_data[p]
            if not d["Sa"]:
                per_frame_by_pair[p].append((np.array([]), np.array([])))
                continue
            U_A_all, U_B_all = [], []
            for sa, sb, Z in zip(d["Sa"], d["Sb"], d["Z"]):
                u_a, u_b = _leading_variates(sa, sb, Z, alpha, n_components=8)
                U_A_all.append(u_a); U_B_all.append(u_b)
            # Raw concatenation — same as the static figure's aggregation.
            per_frame_by_pair[p].append(
                (np.concatenate(U_A_all), np.concatenate(U_B_all)))
    print()

    slopes_by_pair: dict[tuple[str, str], list[float]] = {
        p: [_diag_slope(a, b, wh_std_by_pair[p])
            for (a, b) in per_frame_by_pair[p]]
        for p in PAIRS
    }

    # ---- 3. Render frames: 3 pair panels in a row + a progress bar at top ----
    lim = 2.4
    fig = plt.figure(figsize=(12.4, 5.2), dpi=80)
    gs = fig.add_gridspec(2, 3, height_ratios=[0.10, 1.0],
                            top=0.90, bottom=0.10, left=0.05, right=0.97,
                            hspace=0.32, wspace=0.22)
    bar_ax = fig.add_subplot(gs[0, :])
    panel_axes = [fig.add_subplot(gs[1, c]) for c in range(3)]

    bar_ax.set_xlim(0, 1); bar_ax.set_ylim(0, 1)
    bar_ax.set_xticks([0, 0.5, 1.0])
    bar_ax.set_xticklabels(["raw CCA  (α = 0)", "α = 0.5",
                              "rich-Z pCCA  (α = 1)"], fontsize=10)
    bar_ax.set_yticks([])
    for s in ["top", "right", "left"]:
        bar_ax.spines[s].set_visible(False)
    bar_ax.tick_params(length=2, width=0.8, pad=2)
    bar_ax.fill_between([0, 1], 0, 1, color="#e6e6e6")
    bar_ax.fill_between([0, 0], 0, 1, color="#c44536", alpha=0.85)
    bar_ax.axvline(0, color="#1a1a2e", linewidth=1.6)

    def render(frame_idx):
        alpha = alphas[frame_idx]
        for ax, p in zip(panel_axes, PAIRS):
            a, b = per_frame_by_pair[p][frame_idx]
            ax.clear()
            if a.size and b.size:
                mean_z = _heatmap_data(a, b, wh_std_by_pair[p], lim=lim, nbin=18)
                ax.imshow(mean_z.T, origin="lower",
                            extent=(-lim, lim, -lim, lim),
                            cmap="RdBu_r", vmin=-1.0, vmax=1.0,
                            interpolation="bilinear", aspect="equal")
            ax.plot([-lim, lim], [-lim, lim], color="black",
                      ls="--", lw=0.7, alpha=0.5)
            ax.axhline(0, color="black", lw=0.4, alpha=0.35)
            ax.axvline(0, color="black", lw=0.4, alpha=0.35)
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
            la, lb = PAIR_LABEL_AB[p]
            ax.set_xlabel(f"$U_A$  ({la})", labelpad=3, fontsize=11)
            ax.set_ylabel(f"$U_B$  ({lb})", labelpad=3, fontsize=11)
            ax.set_title(PAIR_LABEL[p], color=PAIR_COLOR[p], pad=6, fontsize=13)
            slope = slopes_by_pair[p][frame_idx]
            ax.text(0.03, 0.97, f"diag slope = {slope:+.2f}",
                      transform=ax.transAxes, ha="left", va="top",
                      fontsize=9, family="monospace",
                      bbox=dict(boxstyle="round,pad=0.25", fc="white",
                                  ec="#cccccc", lw=0.5, alpha=0.9))

        # progress bar
        for c in list(bar_ax.collections):
            c.remove()
        bar_ax.fill_between([0, 1], 0, 1, color="#e6e6e6")
        bar_ax.fill_between([0, alpha], 0, 1, color="#c44536", alpha=0.85)
        for line in bar_ax.lines:
            line.set_xdata([alpha, alpha])
        fig.suptitle(f"Partialling morph   ·   α = {alpha:.2f}",
                       y=0.985, fontsize=14, fontweight="bold")
        return ()

    frame_order = list(range(len(alphas)))
    if PING_PONG:
        frame_order += list(range(len(alphas) - 2, 0, -1))

    anim = animation.FuncAnimation(
        fig, render, frames=frame_order, interval=70, blit=False)
    out_path = OUT / "geometry_morph.gif"
    anim.save(str(out_path), writer=animation.PillowWriter(fps=14), dpi=85)
    print(f"Wrote {out_path}  ({out_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
