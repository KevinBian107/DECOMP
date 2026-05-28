"""Matplotlib figures for the representation-geometry section.

Each function takes a per-pair dict already aggregated across sessions by
`scripts/run_geometry.py` and writes a PNG to `outputs/`. Style mirrors
`figures.py` (sans-serif, no top/right spines, region-pair colours).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
# Reuse the global rcParams already set by .figures import side effect when present.
from .figures import _BASE_STYLE, _save, _strip_box, _pair_color  # noqa: F401
mpl.rcParams.update(_BASE_STYLE)


PAIR_LABEL = {
    ("VIS", "CB"): r"V1 $\leftrightarrow$ CB",
    ("VIS", "MO"): r"V1 $\leftrightarrow$ M1",
    ("CB",  "MO"): r"CB $\leftrightarrow$ M1",
}
PAIR_ORDER = [("VIS", "CB"), ("VIS", "MO"), ("CB", "MO")]


# ---------------------------------------------------------------------------
# Joint state-space density coloured by behaviour
# ---------------------------------------------------------------------------

def fig_state_space_density(per_pair: dict, out_dir: Path) -> None:
    """3 rows × 2 cols: 2-D histogram of (U_A, U_B) coloured by *mean wheel velocity
    per bin*, raw CCA vs pCCA (rich-Z partial).

    Under raw CCA, a wheel-driven shared mode shows as a diverging colour gradient
    running along the diagonal (low wheel at one end, high wheel at the other).
    Under pCCA, the gradient flattens to neutral grey while the cloud's overall
    shape is preserved — wheel-driven coupling has been removed but residual
    geometry has not.
    """
    fig, axes = plt.subplots(3, 2, figsize=(12, 13.6),
                              gridspec_kw={"hspace": 0.46, "wspace": 0.30,
                                            "left": 0.09, "right": 0.93,
                                            "top": 0.91, "bottom": 0.06})
    im = None  # last imshow handle, for shared colorbar

    for row, pair in enumerate(PAIR_ORDER):
        for col, condition in enumerate(["raw", "rich"]):
            ax = axes[row, col]
            d = per_pair.get(pair, {}).get(f"density_{condition}")
            if d is None or len(d.get("a", [])) == 0:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                          transform=ax.transAxes, color="gray")
                ax.set_xticks([]); ax.set_yticks([])
                continue

            a, b, zstd = d["a"], d["b"], d["z_std"]
            if len(a) > 60000:
                idx = np.random.default_rng(0).choice(len(a), 60000, replace=False)
                a, b, zstd = a[idx], b[idx], zstd[idx]

            lim = 2.4
            # Bin a, b on a square grid; per bin compute mean of zstd
            nbin = 18
            edges = np.linspace(-lim, lim, nbin + 1)
            sum_z, _, _ = np.histogram2d(a, b, bins=[edges, edges], weights=zstd)
            cnt,   _, _ = np.histogram2d(a, b, bins=[edges, edges])
            mean_z = np.where(cnt >= 30, sum_z / np.maximum(cnt, 1), np.nan)

            # Soft Gaussian smooth (manual 3x3 kernel) of NaN-aware mean
            mean_z = _nan_smooth(mean_z, sigma=0.8)

            im = ax.imshow(mean_z.T, origin="lower", extent=(-lim, lim, -lim, lim),
                            cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="equal",
                            interpolation="bilinear")
            # Outline the populated region with the cloud's 2σ contour
            _outline_2sigma(ax, a, b, n_sigma=2.0)

            label_a, label_b = pair
            label_a = "V1" if label_a == "VIS" else label_a
            label_b = "M1" if label_b == "MO" else label_b

            ax.plot([-lim, lim], [-lim, lim],
                     color="black", linestyle="--", linewidth=0.7, alpha=0.5, zorder=1)
            ax.axhline(0, color="black", linewidth=0.4, alpha=0.35, zorder=1)
            ax.axvline(0, color="black", linewidth=0.4, alpha=0.35, zorder=1)
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
            ax.set_aspect("equal", adjustable="box")
            _strip_box(ax)
            ax.set_xlabel(f"$U_A$  ({label_a})", labelpad=4)
            ax.set_ylabel(f"$U_B$  ({label_b})", labelpad=4)

            cond_label = "raw CCA" if condition == "raw" else "pCCA (rich Z)"
            ax.set_title(f"{PAIR_LABEL[pair]} — {cond_label}",
                          color=_pair_color(*pair), pad=8, fontsize=14)

            # Gradient strength: slope of mean_z vs diagonal coordinate
            grad = _diagonal_wheel_slope(a, b, zstd)
            ax.text(0.03, 0.97, f"ρ = {d['rho']:+.2f}\ndiag slope (z-wheel/diag) = {grad:+.2f}",
                     transform=ax.transAxes, ha="left", va="top",
                     fontsize=9.5, family="monospace",
                     bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                ec="#cccccc", lw=0.6, alpha=0.9))

    # One shared colorbar (only if at least one panel rendered an image)
    if im is not None:
        cbar = fig.colorbar(im, ax=axes, location="right",
                              shrink=0.55, aspect=22, pad=0.02)
        cbar.set_label("mean wheel velocity in bin  (z-score)",
                         rotation=270, labelpad=18)

    fig.suptitle("Joint state-space geometry, coloured by mean wheel velocity per bin",
                   y=0.985, fontsize=16, fontweight="bold")
    _save(fig, "geometry_state_space", out_dir)


def _nan_smooth(arr: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """NaN-aware Gaussian smoothing of a 2-D array via a small fixed kernel."""
    if not np.any(np.isfinite(arr)):
        return arr
    s2 = sigma * sigma
    r = max(1, int(round(2 * sigma)))
    yy, xx = np.mgrid[-r:r+1, -r:r+1]
    kern = np.exp(-(xx ** 2 + yy ** 2) / (2 * s2))
    kern /= kern.sum()
    H, W = arr.shape
    out = np.full_like(arr, np.nan)
    valid = np.isfinite(arr)
    for i in range(H):
        for j in range(W):
            i0 = max(0, i - r); i1 = min(H, i + r + 1)
            j0 = max(0, j - r); j1 = min(W, j + r + 1)
            k = kern[r - (i - i0):r + (i1 - i), r - (j - j0):r + (j1 - j)]
            w = arr[i0:i1, j0:j1]
            v = valid[i0:i1, j0:j1]
            if not np.any(v):
                continue
            num = np.sum(np.where(v, w * k, 0.0))
            den = np.sum(np.where(v, k, 0.0))
            if den > 0:
                out[i, j] = num / den
    return out


def _outline_2sigma(ax, a: np.ndarray, b: np.ndarray, n_sigma: float = 2.0) -> None:
    from matplotlib.patches import Ellipse
    if len(a) < 50:
        return
    mu = (float(a.mean()), float(b.mean()))
    cov = np.cov(a, b)
    w, V = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1]
    w = w[order]; V = V[:, order]
    angle = float(np.degrees(np.arctan2(V[1, 0], V[0, 0])))
    width, height = 2 * n_sigma * np.sqrt(np.clip(w, 1e-12, None))
    e = Ellipse(xy=mu, width=width, height=height, angle=angle,
                  facecolor="none", edgecolor="#333", linewidth=1.0,
                  linestyle=":", zorder=3, alpha=0.7)
    ax.add_patch(e)


def _diagonal_wheel_slope(a: np.ndarray, b: np.ndarray, z: np.ndarray) -> float:
    """Slope of z (wheel z-score) regressed on the diagonal coordinate (U_A + U_B)/√2.

    Captures how strongly the shared mode aligns with wheel velocity in one scalar.
    Under raw CCA on a wheel-driven shared subspace this is large; under pCCA it
    drops to near zero.
    """
    d = (a + b) / np.sqrt(2)
    if d.std() == 0:
        return 0.0
    return float(np.cov(d, z)[0, 1] / np.var(d))


def _plot_cov_ellipse(ax, x: np.ndarray, y: np.ndarray, color: str,
                       n_sigma: float = 2.0) -> None:
    """Plot the n-sigma covariance ellipse of a 2-D point cloud, plus its centroid."""
    from matplotlib.patches import Ellipse
    if len(x) < 10:
        return
    mu = (float(np.mean(x)), float(np.mean(y)))
    cov = np.cov(x, y)
    w, V = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1]
    w = w[order]; V = V[:, order]
    angle = float(np.degrees(np.arctan2(V[1, 0], V[0, 0])))
    width, height = 2 * n_sigma * np.sqrt(np.clip(w, 1e-12, None))
    e = Ellipse(xy=mu, width=width, height=height, angle=angle,
                  facecolor=color, edgecolor=color, alpha=0.18,
                  linewidth=2.2, zorder=3)
    ax.add_patch(e)
    e2 = Ellipse(xy=mu, width=width, height=height, angle=angle,
                   facecolor="none", edgecolor=color, linewidth=2.0, zorder=4)
    ax.add_patch(e2)
    ax.scatter([mu[0]], [mu[1]], s=44, color=color, edgecolor="white",
                linewidths=1.0, zorder=5)


# ---------------------------------------------------------------------------
# Cross-region RSA
# ---------------------------------------------------------------------------

def fig_cross_region_rsa(per_pair: dict, out_dir: Path) -> None:
    """3 rows × 4 cols (RDM_A, RDM_B, colorbar, RSA bar)."""
    fig = plt.figure(figsize=(16, 12.5))
    gs = fig.add_gridspec(3, 5, hspace=0.55, wspace=0.55,
                            width_ratios=[1.0, 1.0, 0.05, 0.18, 1.25],
                            left=0.06, right=0.96, top=0.93, bottom=0.06)
    im_last = None

    for row, pair in enumerate(PAIR_ORDER):
        rsa = per_pair.get(pair, {}).get("rsa_raw", {})
        rsa_p = per_pair.get(pair, {}).get("rsa_rich", {})
        label_a, label_b = pair
        label_a = "V1" if label_a == "VIS" else label_a
        label_b = "M1" if label_b == "MO" else label_b

        # Region A RDM — display as Pearson similarity (r) so bright = similar
        ax = fig.add_subplot(gs[row, 0])
        if rsa and rsa.get("rdm_a") is not None and rsa["rdm_a"].size:
            sim_a = 1.0 - rsa["rdm_a"]
            im = ax.imshow(sim_a, cmap="RdBu_r", aspect="equal",
                            vmin=-1.0, vmax=1.0)
            im_last = im
            n_cond = sim_a.shape[0]
            ax.set_xticks(range(n_cond))
            ax.set_yticks(range(n_cond))
            ax.set_xticklabels([str(c) for c in rsa["conds"]],
                                 rotation=45, fontsize=9, ha="right")
            ax.set_yticklabels([str(c) for c in rsa["conds"]], fontsize=9)
        ax.set_title(f"{label_a} similarity matrix",
                      color=_pair_color(*pair), fontsize=12, pad=6)
        ax.tick_params(direction="out", length=2.5, width=0.8, pad=2)
        for s in ax.spines.values():
            s.set_visible(True); s.set_linewidth(0.8)

        # Region B RDM
        ax = fig.add_subplot(gs[row, 1])
        if rsa and rsa.get("rdm_b") is not None and rsa["rdm_b"].size:
            sim_b = 1.0 - rsa["rdm_b"]
            im = ax.imshow(sim_b, cmap="RdBu_r", aspect="equal",
                            vmin=-1.0, vmax=1.0)
            im_last = im
            n_cond = sim_b.shape[0]
            ax.set_xticks(range(n_cond))
            ax.set_yticks(range(n_cond))
            ax.set_xticklabels([str(c) for c in rsa["conds"]],
                                 rotation=45, fontsize=9, ha="right")
            ax.set_yticklabels([str(c) for c in rsa["conds"]], fontsize=9)
        ax.set_title(f"{label_b} similarity matrix",
                      color=_pair_color(*pair), fontsize=12, pad=6)
        ax.tick_params(direction="out", length=2.5, width=0.8, pad=2)
        for s in ax.spines.values():
            s.set_visible(True); s.set_linewidth(0.8)

        # Shared colorbar in col 2 — label sits on the LEFT, away from the bar chart
        cax = fig.add_subplot(gs[row, 2])
        if im_last is not None:
            cbar = fig.colorbar(im_last, cax=cax)
            cbar.ax.yaxis.set_label_position("left")
            cbar.ax.yaxis.set_ticks_position("right")
            cbar.set_label("Pearson $r$", labelpad=6, fontsize=10)

        # gs[row, 3] is a spacer column — leave empty for breathing room

        # RSA bar in col 4 — comfortably separated from the colorbar
        ax = fig.add_subplot(gs[row, 4])
        rsa_raw_r = rsa.get("spearman_r", np.nan) if rsa else np.nan
        rsa_par_r = rsa_p.get("spearman_r", np.nan) if rsa_p else np.nan
        bars = ax.bar(["raw", "Z-partialled"],
                       [rsa_raw_r, rsa_par_r],
                       color=["#5a6f8a", "#c44536"], alpha=0.9,
                       edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, [rsa_raw_r, rsa_par_r]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                     val + (0.03 if val > 0 else -0.06),
                     f"{val:+.2f}", ha="center", va="bottom" if val > 0 else "top",
                     fontsize=11, family="monospace")
        ax.set_ylim(-0.2, 1.0)
        ax.axhline(0, color="black", linewidth=0.6, alpha=0.5)
        ax.set_ylabel("Spearman $r$", labelpad=8)
        ax.set_title(f"{PAIR_LABEL[pair]}  cross-region RSA",
                      color=_pair_color(*pair), pad=10, fontsize=13)
        _strip_box(ax)

    fig.suptitle("Representational similarity analysis: basis-free geometric overlap "
                  "across trial conditions",
                   y=0.985, fontsize=16, fontweight="bold")
    _save(fig, "geometry_rsa", out_dir)


# ---------------------------------------------------------------------------
# Schematic manifold cartoon
# ---------------------------------------------------------------------------

def fig_manifold_schematic(out_dir: Path) -> None:
    """Three 3D panels: ellipsoid manifolds + shared-subspace glyph per pair.

    Pure illustration — no data. Anchors the geometric vocabulary used in the
    abstract and the question section: each region is a low-d state manifold,
    the cross-region shared subspace is a plane connecting them, and the glyph
    on the plane encodes what kind of motion the shared mode supports.
    """
    fig = plt.figure(figsize=(16, 8.4))
    pair_glyphs = [
        (r"V1 $\leftrightarrow$ CB", "#3a7ca5", "#c44536",
            "slow, task-flat, zero-lag",
            "global drive we still don't measure",  "smooth"),
        (r"V1 $\leftrightarrow$ M1", "#3a7ca5", "#3a8a4d",
            "zero-lag, strong task transients",
            "shared cognitive state, no transmission", "pulse"),
        (r"CB $\leftrightarrow$ M1", "#c44536", "#3a8a4d",
            "directed, $\\approx$ +40 ms cerebellar lead",
            "real cortico-cerebellar transmission", "arrow"),
    ]

    for col, (label, c1, c2, sub, caption, glyph) in enumerate(pair_glyphs):
        ax = fig.add_subplot(1, 3, col + 1, projection="3d")

        # Region manifolds
        _draw_ellipsoid(ax, center=(-1.8, 0, 0),
                          radii=(0.95, 0.65, 0.5), color=c1, alpha=0.22)
        _draw_ellipsoid(ax, center=(+1.8, 0, 0),
                          radii=(0.95, 0.65, 0.5), color=c2, alpha=0.22)

        # Shared subspace plane
        xs = np.array([[-1.8, +1.8], [-1.8, +1.8]])
        ys = np.array([[-0.5, -0.5], [+0.5, +0.5]])
        zs = np.array([[0, 0], [0, 0]])
        ax.plot_surface(xs, ys, zs, color="#cfa645", alpha=0.30,
                          linewidth=0, antialiased=True)

        # Pair-specific glyph drawn slightly above the plane
        zg = 0.04
        if glyph == "smooth":
            t = np.linspace(-1.7, 1.7, 120)
            ax.plot(t, 0.16 * np.sin(t * 1.6),
                     np.full_like(t, zg),
                     color="#1a1a2e", linewidth=2.4, solid_capstyle="round")
        elif glyph == "pulse":
            t = np.linspace(-1.7, 1.7, 200)
            pulse = np.exp(-(t**2) / 0.18) * 0.36 - 0.05
            ax.plot(t, pulse,
                     np.full_like(t, zg),
                     color="#1a1a2e", linewidth=2.4)
            ax.plot(t, -pulse,
                     np.full_like(t, zg),
                     color="#1a1a2e", linewidth=2.4, alpha=0.55)
        else:  # arrow
            ax.plot([-1.4, 1.4], [0, 0], [zg, zg],
                     color="#1a1a2e", linewidth=2.6, solid_capstyle="round")
            # arrowhead
            ax.plot([1.4, 1.05], [0, 0.18], [zg, zg],
                     color="#1a1a2e", linewidth=2.6, solid_capstyle="round")
            ax.plot([1.4, 1.05], [0, -0.18], [zg, zg],
                     color="#1a1a2e", linewidth=2.6, solid_capstyle="round")
            ax.text(0, 0.50, zg + 0.02, "$+40$ ms",
                      ha="center", va="bottom", fontsize=11, color="#1a1a2e")

        ax.set_xlim(-2.9, 2.9); ax.set_ylim(-1.3, 1.3); ax.set_zlim(-0.85, 0.85)
        ax.set_box_aspect((4.2, 1.6, 1.0))
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_axis_off()
        ax.view_init(elev=22, azim=-58)

        # Region labels at ellipsoid centres
        name_a, name_b = label.split(r"$\leftrightarrow$")
        ax.text(-1.8, 0, 0.95, name_a.strip(), ha="center", va="bottom",
                  fontsize=15, fontweight="bold", color=c1)
        ax.text(+1.8, 0, 0.95, name_b.strip(), ha="center", va="bottom",
                  fontsize=15, fontweight="bold", color=c2)
        ax.text2D(0.5, 0.05, "shared subspace", transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=9.5, color="#7a5a14",
                    style="italic")

        # Panel title block (two stacked lines, large breathing room)
        ax.set_title(f"{label}", fontsize=15, fontweight="bold", pad=14)
        ax.text2D(0.5, 0.95, sub, transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=11.5, color="#444")
        ax.text2D(0.5, 0.88, caption, transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=10.5, color="#888",
                    style="italic")

    fig.suptitle("Three pairs, three representational geometries on a shared substrate",
                   y=0.99, fontsize=18, fontweight="bold")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.04, wspace=0.04)
    _save(fig, "geometry_schematic", out_dir)


def _draw_ellipsoid(ax, center, radii, color, alpha=0.2, n=30):
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    cu, cv = np.meshgrid(u, v)
    x = center[0] + radii[0] * np.cos(cu) * np.sin(cv)
    y = center[1] + radii[1] * np.sin(cu) * np.sin(cv)
    z = center[2] + radii[2] * np.cos(cv)
    ax.plot_surface(x, y, z, color=color, alpha=alpha,
                       linewidth=0, antialiased=True, shade=True)
