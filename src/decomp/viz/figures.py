"""Matplotlib figure factory for the DECOMP pipeline.

Layout:
    outputs/
    ├── fig01_glm_dr2_per_region.png       per-region GLM ΔR² (top-level, all 4 ROIs)
    ├── fig_pair_distribution.png          per-session survival ratio across pairs (boxplot)
    └── by_pair/
        ├── VIS_CB/
        │   ├── correlations.png           grid of per-session canonical correlations vs null
        │   ├── survival.png               per-session ΣρCCA, ΣρpCCA, survival ratio bars
        │   └── reliability.png            grid of per-session SVCA reliability
        ├── VIS_MO/
        │   └── ... (same files)
        └── CB_MO/
            └── ... (same files)

All region/pair colors are converted to hex strings to avoid mixed-type c=array bugs.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..data.sessions import ROI_ORDER

# ---- Style ---------------------------------------------------------------------------------
_BASE_STYLE = {
    "figure.dpi": 110,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 14,
    "axes.titlesize": 17,
    "axes.titleweight": "regular",
    "axes.labelsize": 14,
    "axes.labelweight": "regular",
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "axes.linewidth": 1.2,
    "axes.edgecolor": "#222222",
    "axes.labelcolor": "#222222",
    "xtick.color": "#222222",
    "ytick.color": "#222222",
    "axes.grid": False,
    "axes.axisbelow": True,
    "lines.linewidth": 2.0,
    "patch.linewidth": 1.0,
    "patch.edgecolor": "#222222",
    "legend.frameon": False,
    "axes.titlepad": 14,
    "axes.labelpad": 8,
}

mpl.rcParams.update(_BASE_STYLE)

_REGION_COLORS = {
    "VISp": "#3a7ca5",
    "VIS":  "#3a7ca5",
    "CB":   "#c44536",
    "MO":   "#3a8a4d",
    "CA1":  "#7a5db8",
}

_KERNEL_LABEL = {
    "dR2_movement": "Movement\n(wheel + ME + lick + fmove)",
    "dR2_stim":     "Stimulus",
    "dR2_choice":   "Choice",
}

# MVP-anchor session highlighted in V1vsCB panels for backward-compat sanity checking.
MVP_ANCHOR_EID_PREFIX = "41431f53"


def _save(fig: plt.Figure, name: str, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png")
    plt.close(fig)


def _strip_box(ax: plt.Axes) -> None:
    ax.tick_params(direction="out", length=4, width=1.0, pad=4)
    ax.spines["bottom"].set_position(("outward", 6))
    ax.spines["left"].set_position(("outward", 6))


def _pair_color(a: str, b: str) -> str:
    """Color for a region pair; defaults to gray."""
    if "MO" in (a, b):
        return _REGION_COLORS["MO"]
    if "CB" in (a, b):
        return _REGION_COLORS["CB"]
    return _REGION_COLORS["VIS"]


def _slug(a: str, b: str) -> str:
    return f"{a}_{b}"


# ============================================================================================
# fig01 — per-region GLM ΔR²  (top-level, all 4 ROIs)
# ============================================================================================
def fig01_glm_dr2_per_region(df: pd.DataFrame, out_dir: Path = Path("outputs")) -> None:
    kernels = ["dR2_movement", "dR2_stim", "dR2_choice"]
    rois = [r for r in ROI_ORDER if r in df["region"].unique()]
    fig, axes = plt.subplots(1, len(kernels), figsize=(5.4 * len(kernels), 5.0), sharey=True)

    for ax, k in zip(axes, kernels):
        data = [df.loc[df["region"] == r, k].dropna().to_numpy() for r in rois]
        positions = np.arange(len(rois)) * 1.2
        bp = ax.boxplot(
            data, positions=positions, widths=0.7,
            showfliers=False, patch_artist=True,
            medianprops=dict(color="#111111", linewidth=2.2),
            whiskerprops=dict(color="#444444", linewidth=1.0),
            capprops=dict(color="#444444", linewidth=1.0),
        )
        for patch, r in zip(bp["boxes"], rois):
            patch.set_facecolor(_REGION_COLORS.get(r, "0.6"))
            patch.set_alpha(0.55)
            patch.set_edgecolor("#222222")
        for j, (r, vals) in enumerate(zip(rois, data)):
            if len(vals):
                jitter = (np.random.default_rng(j + hash(k) % 1000)
                           .normal(0, 0.06, size=len(vals)))
                ax.scatter(
                    np.full(len(vals), positions[j]) + jitter, vals,
                    s=8, color="#222222", alpha=0.18, linewidths=0,
                )
        ax.axhline(0, color="#888888", lw=0.8, zorder=0)
        ax.set_xticks(positions)
        ax.set_xticklabels(rois, fontsize=13)
        ax.set_title(_KERNEL_LABEL.get(k, k))
        if k == kernels[0]:
            ax.set_ylabel("Cross-validated $\\Delta R^2$")
        _strip_box(ax)

    fig.suptitle(
        "Per-region variance partition  ·  GLM leave-one-group-out $\\Delta R^2$",
        y=1.02, fontsize=18, fontweight="regular",
    )
    fig.tight_layout()
    _save(fig, "fig01_glm_dr2_per_region", out_dir)


# ============================================================================================
# fig_pair_distribution — top-level cross-pair survival comparison (boxplot)
# ============================================================================================
def fig_pair_distribution(df: pd.DataFrame, out_dir: Path = Path("outputs"),
                           name: str = "survival") -> None:
    """Two-panel cross-pair box-and-whisker (the headline answer figure).

    Left panel: per-pair raw vs partialled ΣρCCA distributions (paired boxes per pair,
    showing what is being preserved).
    Right panel: per-pair survival ratio distributions (one box per pair, the headline).
    Per-session dots overlaid on every box. MVP-anchor V1↔CB session ringed in gold.
    """
    from matplotlib.patches import Patch

    summary = (
        df.groupby(["eid", "pair_a", "pair_b"])
        .agg(rho_cca=("rho_cca", "sum"), rho_pcca=("rho_pcca", "sum"))
        .reset_index()
    )
    summary["survival"] = summary["rho_pcca"] / summary["rho_cca"].replace(0, np.nan)
    summary["pair_label"] = summary["pair_a"] + " vs " + summary["pair_b"]
    if summary.empty:
        return

    # Order pairs by sample size (largest first) — keeps V1↔CB on the left
    pair_order = (
        summary.groupby("pair_label")["eid"].nunique()
        .sort_values(ascending=False).index.tolist()
    )
    n_pairs = len(pair_order)
    positions = np.arange(n_pairs) * 1.0
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(
        1, 2, figsize=(13.0, 5.4), gridspec_kw={"width_ratios": [1.4, 1.0]}
    )

    # ---- LEFT PANEL: ΣρCCA vs ΣρpCCA paired boxes per pair --------------------------------
    ax = axes[0]
    box_w = 0.34
    for i, plabel in enumerate(pair_order):
        sub = summary[summary["pair_label"] == plabel].reset_index(drop=True)
        cca_vals = sub["rho_cca"].dropna().to_numpy()
        pcca_vals = sub["rho_pcca"].dropna().to_numpy()
        bp_c = ax.boxplot([cca_vals], positions=[positions[i] - box_w / 2 - 0.02],
                           widths=box_w, patch_artist=True, showfliers=False,
                           medianprops=dict(color="#111", linewidth=2.0),
                           whiskerprops=dict(color="#555", linewidth=0.9),
                           capprops=dict(color="#555", linewidth=0.9))
        bp_p = ax.boxplot([pcca_vals], positions=[positions[i] + box_w / 2 + 0.02],
                           widths=box_w, patch_artist=True, showfliers=False,
                           medianprops=dict(color="#111", linewidth=2.0),
                           whiskerprops=dict(color="#555", linewidth=0.9),
                           capprops=dict(color="#555", linewidth=0.9))
        bp_c["boxes"][0].set_facecolor("#5e5e5e"); bp_c["boxes"][0].set_alpha(0.55)
        bp_p["boxes"][0].set_facecolor(_REGION_COLORS.get("CB", "#c44536"))
        bp_p["boxes"][0].set_alpha(0.55)
        jitter = rng.normal(0, 0.025, size=len(cca_vals))
        ax.scatter(np.full(len(cca_vals), positions[i] - box_w / 2 - 0.02) + jitter,
                   cca_vals, s=28, color="#222", alpha=0.55,
                   edgecolors="white", linewidths=0.5, zorder=3)
        ax.scatter(np.full(len(pcca_vals), positions[i] + box_w / 2 + 0.02) + jitter,
                   pcca_vals, s=28, color="#222", alpha=0.55,
                   edgecolors="white", linewidths=0.5, zorder=3)
        for j, eid in enumerate(sub["eid"].tolist()):
            if eid.startswith(MVP_ANCHOR_EID_PREFIX):
                ax.scatter(positions[i] - box_w / 2 - 0.02 + jitter[j], cca_vals[j],
                           s=120, edgecolors="#d4a017", facecolors="none",
                           linewidths=2.2, zorder=4)
                ax.scatter(positions[i] + box_w / 2 + 0.02 + jitter[j], pcca_vals[j],
                           s=120, edgecolors="#d4a017", facecolors="none",
                           linewidths=2.2, zorder=4)

    ax.set_xticks(positions)
    ax.set_xticklabels([
        f"{p}\n(n={int(summary.loc[summary['pair_label']==p, 'eid'].nunique())})"
        for p in pair_order
    ], fontsize=12)
    ax.set_ylabel(r"$\sum_k\,\rho_k$ (cross-validated)")
    ax.set_title("Raw vs partialled shared variance, per pair")
    ax.legend(handles=[
        Patch(facecolor="#5e5e5e", alpha=0.55, edgecolor="#222",
              label=r"$\sum_k\,\rho_k^{\mathrm{CCA}}$"),
        Patch(facecolor=_REGION_COLORS.get("CB", "#c44536"), alpha=0.55, edgecolor="#222",
              label=r"$\sum_k\,\rho_k^{\mathrm{pCCA}}$ (out: wheel + pupil)"),
    ], loc="upper right", frameon=False, fontsize=10)
    ax.set_ylim(0, max(1.4, summary["rho_cca"].max() * 1.1))
    _strip_box(ax)

    # ---- RIGHT PANEL: survival ratio box-and-whisker per pair ------------------------------
    ax = axes[1]
    survival_groups = {p: summary.loc[summary["pair_label"] == p, "survival"]
                          .dropna().to_numpy() for p in pair_order}
    bp = ax.boxplot(
        [survival_groups[p] for p in pair_order],
        positions=positions, widths=0.55, patch_artist=True, showfliers=False,
        medianprops=dict(color="#111", linewidth=2.4),
        whiskerprops=dict(color="#555", linewidth=1.0),
        capprops=dict(color="#555", linewidth=1.0),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(_REGION_COLORS.get("MO", "#3a8a4d"))
        patch.set_alpha(0.45)
        patch.set_edgecolor("#222")

    for i, plabel in enumerate(pair_order):
        vals = survival_groups[plabel]
        sub = summary[summary["pair_label"] == plabel].reset_index(drop=True)
        if not len(vals):
            continue
        jitter = rng.normal(0, 0.04, size=len(vals))
        ax.scatter(np.full(len(vals), positions[i]) + jitter, vals,
                   s=46, color="#222", alpha=0.55, edgecolors="white",
                   linewidths=0.6, zorder=3)
        for j, eid in enumerate(sub["eid"].tolist()):
            if eid.startswith(MVP_ANCHOR_EID_PREFIX):
                ax.scatter(positions[i] + jitter[j], vals[j],
                           s=160, edgecolors="#d4a017", facecolors="none",
                           linewidths=2.4, zorder=4, label="MVP anchor session")
        m = float(np.median(vals))
        ax.text(positions[i], m + 0.05, f"med = {m:.2f}", ha="center", va="bottom",
                fontsize=11, color="#222")

    ax.axhline(1.0, color="#888", lw=0.7, ls="-")
    ax.axhline(0.0, color="#222", lw=0.7)
    ax.set_xticks(positions)
    ax.set_xticklabels([
        f"{p}\n(n={int(summary.loc[summary['pair_label']==p, 'eid'].nunique())})"
        for p in pair_order
    ], fontsize=12)
    ax.set_ylabel(r"Survival ratio  $\rho^{\mathrm{pCCA}} / \rho^{\mathrm{CCA}}$")
    ax.set_ylim(0, 1.25)
    ax.set_title("Survival of shared subspace under wheel + pupil partialling")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        seen = set(); uniq = []
        for h, l in zip(handles, labels):
            if l not in seen:
                uniq.append((h, l)); seen.add(l)
        ax.legend([h for h, _ in uniq], [l for _, l in uniq], loc="lower right",
                   frameon=False, fontsize=10)
    _strip_box(ax)

    fig.suptitle(
        "Cross-pair survival of the V1 vs CB / V1 vs M1 / CB vs M1 shared subspace",
        y=1.01, fontsize=17,
    )
    fig.tight_layout()
    _save(fig, name, out_dir)


# ============================================================================================
# Top-level aggregated cross-pair figures (companion to fig_pair_distribution)
# ============================================================================================
def fig_correlations_aggregated(df: pd.DataFrame, out_dir: Path = Path("outputs"),
                                  name: str = "correlations_aggregated") -> None:
    """Cross-pair canonical-correlation curves: median ± IQR per component, per pair.

    Two panels: left = ρ_k^CCA, right = ρ_k^pCCA, both with phase-shuffle null bands
    (median across sessions). One colored line per pair; ribbons show IQR across sessions.
    """
    if df.empty:
        return

    df = df.copy()
    df["pair_label"] = df["pair_a"] + " vs " + df["pair_b"]
    pair_order = (
        df.groupby("pair_label")["eid"].nunique()
        .sort_values(ascending=False).index.tolist()
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), sharey=True)

    for which, ax, title in [
        ("rho_cca", axes[0], r"Raw  $\rho_k^{\mathrm{CCA}}$"),
        ("rho_pcca", axes[1], r"Partialled  $\rho_k^{\mathrm{pCCA}}$  (out: wheel + pupil)"),
    ]:
        for plabel in pair_order:
            sub = df[df["pair_label"] == plabel]
            n_sess = sub["eid"].nunique()
            agg = (sub.groupby("component")[which]
                      .agg(["median",
                            lambda s: float(np.percentile(s, 25)),
                            lambda s: float(np.percentile(s, 75))])
                      .rename(columns={"<lambda_0>": "q25", "<lambda_1>": "q75"})
                      .reset_index())
            x = agg["component"].to_numpy() + 1
            a, b = plabel.split(" vs ")
            color = _pair_color(a, b)
            ax.fill_between(x, agg["q25"], agg["q75"], color=color, alpha=0.18, lw=0)
            ax.plot(x, agg["median"], "o-", color=color, lw=2.2, ms=6,
                    mec="white", mew=0.8, label=f"{plabel}  (n={n_sess})")
        # null band: median across all sessions and pairs (phase-shuffle 99%)
        null_med = df.groupby("component")["null_cca_99"].median().reset_index()
        ax.fill_between(null_med["component"] + 1, null_med["null_cca_99"],
                         np.zeros_like(null_med["null_cca_99"]),
                         color="#bbbbbb", alpha=0.30, label="phase-shuffle 99% null")
        ax.set_xlabel("Canonical component $k$")
        ax.set_title(title)
        ax.set_xticks(np.arange(1, df["component"].max() + 2))
        ax.set_ylim(-0.05, 1.0)
        _strip_box(ax)

    axes[0].set_ylabel(r"Cross-validated $\rho_k$  (median across sessions, IQR shaded)")
    axes[0].legend(fontsize=10, loc="upper right")
    fig.suptitle("Cross-pair canonical correlations  ·  V1 vs CB / V1 vs M1 / CB vs M1",
                 y=1.02, fontsize=17)
    fig.tight_layout()
    _save(fig, name, out_dir)


def fig_reliability_aggregated(svca_per_session: list, cca_df: pd.DataFrame,
                                 out_dir: Path = Path("outputs"),
                                 name: str = "reliability_aggregated") -> None:
    """Cross-pair SVCA reliability comparison: leading-component reliability ρ_1 per
    region per pair-context, as box-and-whisker with per-session dots.

    Each pair contributes two boxes (one per region in the pair). Boxes are colored by
    region. Reveals the V1↔CB asymmetric reliability pattern called out in the writeup.
    """
    if cca_df.empty or not svca_per_session:
        return
    rows = []
    pair_keys = cca_df[["pair_a", "pair_b"]].drop_duplicates().itertuples(index=False)
    pair_keys = list(pair_keys)
    eids_by_pair = {(a, b): set(cca_df.loc[(cca_df["pair_a"] == a)
                                              & (cca_df["pair_b"] == b), "eid"].unique())
                      for a, b in pair_keys}

    for (a, b), pair_eids in eids_by_pair.items():
        for eid, _n_units, regions in svca_per_session:
            if eid not in pair_eids:
                continue
            for region in (a, b):
                if region not in regions:
                    continue
                rel = np.asarray(regions[region].reliability)
                if not len(rel):
                    continue
                rows.append({"pair_label": f"{a} vs {b}", "pair_a": a, "pair_b": b,
                              "region": region, "eid": eid, "rho1": float(rel[0]),
                              "n_above_05": int((rel > 0.5).sum())})
    if not rows:
        return
    df = pd.DataFrame(rows)

    pair_order = (df.groupby("pair_label")["eid"].nunique()
                    .sort_values(ascending=False).index.tolist())

    # Build flat box positions: per pair, two side-by-side boxes (region A then B).
    box_w = 0.36
    positions: list[float] = []
    box_data: list[np.ndarray] = []
    box_colors: list[str] = []
    box_labels: list[str] = []
    pair_centers: list[float] = []

    cur_x = 0.0
    for plabel in pair_order:
        sub = df[df["pair_label"] == plabel]
        a, b = plabel.split(" vs ")
        pair_centers.append(cur_x + 0.5)
        for j, region in enumerate((a, b)):
            vals = sub.loc[sub["region"] == region, "rho1"].to_numpy()
            positions.append(cur_x + j * (box_w + 0.04))
            box_data.append(vals)
            box_colors.append(_REGION_COLORS.get(region, "#777"))
            box_labels.append(region)
        cur_x += 1.2  # gap between pairs

    fig, ax = plt.subplots(figsize=(max(7.0, 1.6 * len(pair_order) + 4.0), 5.2))
    bp = ax.boxplot(box_data, positions=positions, widths=box_w,
                     patch_artist=True, showfliers=False,
                     medianprops=dict(color="#111", linewidth=2.0),
                     whiskerprops=dict(color="#555", linewidth=0.9),
                     capprops=dict(color="#555", linewidth=0.9))
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color); patch.set_alpha(0.55); patch.set_edgecolor("#222")

    rng = np.random.default_rng(1)
    for pos, vals, region in zip(positions, box_data, box_labels):
        if not len(vals):
            continue
        jitter = rng.normal(0, 0.025, size=len(vals))
        ax.scatter(np.full(len(vals), pos) + jitter, vals,
                   s=32, color="#222", alpha=0.6, edgecolors="white",
                   linewidths=0.5, zorder=3)

    ax.axhline(0.5, ls="--", color="#666", lw=1.0,
               label=r"Stringer threshold ($\rho_1 = 0.5$)")
    ax.set_xticks(positions)
    ax.set_xticklabels(box_labels, fontsize=11)
    for tick, region in zip(ax.get_xticklabels(), box_labels):
        tick.set_color(_REGION_COLORS.get(region, "#444"))
        tick.set_fontweight("bold")
    pair_n = {p: int(df.loc[df["pair_label"] == p, "eid"].nunique()) for p in pair_order}
    # Group label below each pair (in axes-fraction y so it sits under the region ticks)
    for center, plabel in zip(pair_centers, pair_order):
        ax.text(center, -0.13, f"{plabel}  (n={pair_n[plabel]})",
                ha="center", va="top", fontsize=12, color="#222",
                transform=ax.get_xaxis_transform())
    ax.set_ylabel(r"Leading-component reliability  $\rho_1^{\mathrm{SVCA}}$")
    ax.set_ylim(-0.05, 1.0)
    ax.set_title("Per-region SVCA reliability across pair-sessions")
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    _strip_box(ax)
    fig.suptitle("Cross-pair SVCA reliability  ·  region quality feeding the CCA",
                 y=1.01, fontsize=17)
    fig.tight_layout()
    _save(fig, name, out_dir)


# ============================================================================================
# fig_richz_comparison — survival under minimal Z vs richer Z (the discriminating test)
# ============================================================================================
def fig_richz_comparison(richz_df: pd.DataFrame, out_dir: Path = Path("outputs"),
                           name: str = "richz_comparison") -> None:
    """Per-pair box-and-whisker comparing survival ratios under Z_min vs Z_rich.

    Z_min  = [wheel_velocity, pupil]                           (2-D, the headline result)
    Z_rich = [wheel_velocity, wheel_acceleration, me_whisker, (me_body,) pupil, lick_rate]
             (5-6 D, adds the IBL-shipped uninstructed-movement scalars)

    Survival drops materially under Z_rich → wheel+pupil were too narrow a Z and the
    surviving subspace was partly capturable behavioral state.
    Survival stays near 1 under Z_rich → the surviving subspace is genuinely orthogonal
    to behaviorally measurable global state.
    """
    if richz_df.empty:
        return
    from matplotlib.patches import Patch

    summary = (
        richz_df.groupby(["eid", "pair_a", "pair_b"])
        .agg(rho_cca=("rho_cca", "sum"),
             rho_pcca_min=("rho_pcca_min", "sum"),
             rho_pcca_rich=("rho_pcca_rich", "sum"))
        .reset_index()
    )
    summary["surv_min"]  = summary["rho_pcca_min"]  / summary["rho_cca"].replace(0, np.nan)
    summary["surv_rich"] = summary["rho_pcca_rich"] / summary["rho_cca"].replace(0, np.nan)
    summary["pair_label"] = summary["pair_a"] + " vs " + summary["pair_b"]

    pair_order = (summary.groupby("pair_label")["eid"].nunique()
                    .sort_values(ascending=False).index.tolist())
    n_pairs = len(pair_order)
    positions = np.arange(n_pairs) * 1.0
    rng = np.random.default_rng(2)

    box_w = 0.34
    fig, ax = plt.subplots(figsize=(max(7.5, 1.9 * n_pairs + 4.0), 5.6))
    color_min  = "#5e5e5e"
    color_rich = _REGION_COLORS.get("CB", "#c44536")

    for i, plabel in enumerate(pair_order):
        sub = summary[summary["pair_label"] == plabel].reset_index(drop=True)
        v_min  = sub["surv_min"].dropna().to_numpy()
        v_rich = sub["surv_rich"].dropna().to_numpy()

        bp_m = ax.boxplot([v_min],  positions=[positions[i] - box_w / 2 - 0.02],
                            widths=box_w, patch_artist=True, showfliers=False,
                            medianprops=dict(color="#111", linewidth=2.0),
                            whiskerprops=dict(color="#555", linewidth=0.9),
                            capprops=dict(color="#555", linewidth=0.9))
        bp_r = ax.boxplot([v_rich], positions=[positions[i] + box_w / 2 + 0.02],
                            widths=box_w, patch_artist=True, showfliers=False,
                            medianprops=dict(color="#111", linewidth=2.0),
                            whiskerprops=dict(color="#555", linewidth=0.9),
                            capprops=dict(color="#555", linewidth=0.9))
        bp_m["boxes"][0].set_facecolor(color_min);  bp_m["boxes"][0].set_alpha(0.55)
        bp_r["boxes"][0].set_facecolor(color_rich); bp_r["boxes"][0].set_alpha(0.55)

        # paired dots + connecting lines (per session)
        jitter = rng.normal(0, 0.018, size=len(v_min))
        x_m = positions[i] - box_w / 2 - 0.02 + jitter
        x_r = positions[i] + box_w / 2 + 0.02 + jitter
        for xm, xr, vm, vr in zip(x_m, x_r, v_min, v_rich):
            ax.plot([xm, xr], [vm, vr], color="#999", lw=0.8, alpha=0.6, zorder=2)
        ax.scatter(x_m, v_min,  s=32, color="#222", alpha=0.7, zorder=3,
                   edgecolors="white", linewidths=0.5)
        ax.scatter(x_r, v_rich, s=32, color="#222", alpha=0.7, zorder=3,
                   edgecolors="white", linewidths=0.5)
        for j, eid in enumerate(sub["eid"].tolist()):
            if eid.startswith(MVP_ANCHOR_EID_PREFIX):
                ax.scatter(x_m[j], v_min[j],  s=140, edgecolors="#d4a017",
                           facecolors="none", linewidths=2.4, zorder=4)
                ax.scatter(x_r[j], v_rich[j], s=140, edgecolors="#d4a017",
                           facecolors="none", linewidths=2.4, zorder=4)

        # median annotations
        if len(v_min):
            ax.text(positions[i] - box_w / 2 - 0.02, float(np.median(v_min)) + 0.05,
                    f"{np.median(v_min):.2f}", ha="center", va="bottom",
                    fontsize=10, color="#222")
        if len(v_rich):
            ax.text(positions[i] + box_w / 2 + 0.02, float(np.median(v_rich)) + 0.05,
                    f"{np.median(v_rich):.2f}", ha="center", va="bottom",
                    fontsize=10, color="#222")

    ax.axhline(1.0, color="#888", lw=0.7, ls="-")
    ax.axhline(0.0, color="#222", lw=0.7)
    ax.set_xticks(positions)
    ax.set_xticklabels([
        f"{p}\n(n={int(summary.loc[summary['pair_label']==p, 'eid'].nunique())})"
        for p in pair_order
    ], fontsize=12)
    ax.set_ylabel(r"Survival ratio  $\rho^{\mathrm{pCCA}} / \rho^{\mathrm{CCA}}$")
    ax.set_ylim(-0.05, 1.25)
    ax.set_title("Survival of shared subspace under minimal vs richer Z")

    n_min  = int(richz_df["n_z_min"].iloc[0])
    n_rich = int(richz_df["n_z_rich"].iloc[0])
    ax.legend(handles=[
        Patch(facecolor=color_min, alpha=0.55, edgecolor="#222",
              label=f"Z_min ({n_min}D): wheel + pupil"),
        Patch(facecolor=color_rich, alpha=0.55, edgecolor="#222",
              label=f"Z_rich ({n_rich}D): + wheel acc, whisker ME, lick rate" +
                    (", body ME" if n_rich >= 6 else "")),
    ], loc="lower left", frameon=False, fontsize=10)
    _strip_box(ax)
    fig.suptitle("Richer-Z stress test  ·  what does survival look like under fuller Z?",
                 y=1.01, fontsize=17)
    fig.tight_layout()
    _save(fig, name, out_dir)


# ============================================================================================
# Per-pair detailed figures — written into outputs/by_pair/<pair>/
# ============================================================================================
def per_pair_correlations(pair_df: pd.DataFrame, pair_a: str, pair_b: str,
                          out_dir: Path) -> None:
    """Grid of per-session canonical correlations with phase-shuffle null bands.

    pair_df should be filtered to one (pair_a, pair_b). One subplot per session.
    Layout: cap at 3 columns so 4-session figures lay out 3+1 in two rows rather than
    a single very-wide row. Single-session figures are kept small (one panel).
    """
    if pair_df.empty:
        return
    eids = sorted(pair_df["eid"].unique(), key=lambda e: -float(
        pair_df.loc[pair_df["eid"] == e, "rho_cca"].sum()
    ))
    n = len(eids)
    cols = min(3, max(1, n))
    rows = int(np.ceil(n / cols))
    fig_w = max(5.0, 3.8 * cols)
    fig_h = max(3.6, 3.4 * rows)
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h),
                              squeeze=False, sharey=True)

    for idx, eid in enumerate(eids):
        ax = axes.ravel()[idx]
        sub = pair_df[pair_df["eid"] == eid].sort_values("component")
        x = sub["component"].to_numpy() + 1
        ax.fill_between(x, sub["null_cca_99"], np.zeros_like(sub["null_cca_99"]),
                        color="#bbbbbb", alpha=0.35, label="phase-shuffle 99% null")
        ax.plot(x, sub["rho_cca"], "o-", color=_REGION_COLORS.get(pair_a, "#3a7ca5"),
                lw=2.2, ms=7, mec="white", mew=0.8, label=r"$\rho_k^{\mathrm{CCA}}$")
        ax.plot(x, sub["rho_pcca"], "s--", color=_REGION_COLORS.get(pair_b, "#c44536"),
                lw=2.0, ms=7, mec="white", mew=0.8, label=r"$\rho_k^{\mathrm{pCCA}}$")
        title = eid[:8]
        if eid.startswith(MVP_ANCHOR_EID_PREFIX):
            title += "  (MVP anchor)"
            ax.set_facecolor("#fff8d8")
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Canonical component $k$")
        if idx % cols == 0:
            ax.set_ylabel(r"Cross-validated $\rho_k$")
        ax.set_ylim(-0.05, 1.0)
        ax.set_xticks(x)
        if idx == 0:
            ax.legend(fontsize=10, loc="upper right")
        _strip_box(ax)

    for ax in axes.ravel()[n:]:
        ax.axis("off")

    fig.suptitle(
        f"Canonical correlations vs phase-shuffle null  ·  {pair_a} vs {pair_b}  "
        f"(n = {n} sessions)", y=1.005, fontsize=18,
    )
    fig.tight_layout()
    _save(fig, "correlations", out_dir)


def per_pair_survival(pair_df: pd.DataFrame, pair_a: str, pair_b: str,
                       out_dir: Path) -> None:
    """Per-session ΣρCCA, ΣρpCCA, and survival ratio bars for one pair."""
    if pair_df.empty:
        return
    summary = (
        pair_df.groupby("eid")
        .agg(rho_cca=("rho_cca", "sum"), rho_pcca=("rho_pcca", "sum"),
             n_above_null=("rho_cca",
                           lambda s: int((s.values
                                            > pair_df.loc[s.index, "null_cca_99"].values).sum())))
        .reset_index()
    )
    summary["survival"] = summary["rho_pcca"] / summary["rho_cca"].replace(0, np.nan)
    summary = summary.sort_values("rho_cca", ascending=False).reset_index(drop=True)
    n = len(summary)

    # Width scales with bar count so 1-bar figures aren't huge squares.
    fig_w = max(4.6, 0.85 * n + 3.2)
    fig, axes = plt.subplots(2, 1, figsize=(fig_w, 7.0))
    x = np.arange(n)
    width = 0.38

    # Top: ΣρCCA vs ΣρpCCA
    ax_top = axes[0]
    ax_top.bar(x - width / 2, summary["rho_cca"], width,
               label=r"$\sum_k\,\rho_k^{\mathrm{CCA}}$",
               color="#5e5e5e", edgecolor="#222")
    ax_top.bar(x + width / 2, summary["rho_pcca"], width,
               label=r"$\sum_k\,\rho_k^{\mathrm{pCCA}}$" + "\n(out: wheel + pupil)",
               color=_pair_color(pair_a, pair_b), edgecolor="#222")

    for i, eid in enumerate(summary["eid"]):
        if eid.startswith(MVP_ANCHOR_EID_PREFIX):
            ax_top.add_patch(plt.Rectangle(
                (i - 0.5, 0), 1.0, ax_top.get_ylim()[1] or 1.0,
                fc="#ffe999", ec="none", alpha=0.4, zorder=0,
            ))

    ax_top.set_xticks(x)
    ax_top.set_xticklabels(summary["eid"].str.slice(0, 8), fontsize=11,
                            rotation=30, ha="right")
    ax_top.set_ylabel(r"Sum of cross-validated $\rho_k$")
    ax_top.set_title(f"{pair_a} vs {pair_b}  ·  shared structure: raw vs partialled  "
                     f"(n = {n} sessions)", fontsize=15)
    ax_top.legend(loc="upper right", framealpha=1.0, frameon=False, fontsize=11)
    _strip_box(ax_top)

    # Bottom: survival ratio
    ax_bot = axes[1]
    ax_bot.bar(x, summary["survival"], width=0.62,
               color=_pair_color(pair_a, pair_b), edgecolor="#222")
    ax_bot.axhline(1.0, color="#888", lw=0.8, ls="-", zorder=0)
    ax_bot.axhline(0.0, color="#222", lw=0.8)
    for i, (v, eid) in enumerate(zip(summary["survival"], summary["eid"])):
        if pd.notnull(v):
            ax_bot.text(i, v + 0.03, f"{v:.2f}", ha="center", va="bottom",
                         fontsize=11, color="#222")
        if eid.startswith(MVP_ANCHOR_EID_PREFIX):
            ax_bot.add_patch(plt.Rectangle(
                (i - 0.5, 0), 1.0, 1.2,
                fc="#ffe999", ec="none", alpha=0.4, zorder=0,
            ))

    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(summary["eid"].str.slice(0, 8), fontsize=11,
                            rotation=30, ha="right")
    ax_bot.set_ylabel(r"Survival ratio   $\rho^{\mathrm{pCCA}} / \rho^{\mathrm{CCA}}$")
    ax_bot.set_ylim(0, 1.2)
    median_s = float(summary["survival"].median())
    ax_bot.set_title(f"Survival ratio per session  ·  median = {median_s:.2f}  ·  "
                     f"yellow band = MVP anchor", fontsize=14)
    _strip_box(ax_bot)

    fig.tight_layout()
    _save(fig, "survival", out_dir)


def per_pair_reliability(svca_per_session: list, pair_a: str, pair_b: str,
                          out_dir: Path) -> None:
    """Grid of per-session SVCA reliability for the two regions in this pair.

    svca_per_session: list of (eid, n_units_dict, regions_dict) for sessions in this pair.
    """
    sessions = [(eid, n_u, regs) for (eid, n_u, regs) in svca_per_session
                if pair_a in regs and pair_b in regs]
    if not sessions:
        return

    n = len(sessions)
    cols = min(3, max(1, n))
    rows = int(np.ceil(n / cols))
    fig_w = max(5.0, 4.0 * cols)
    fig_h = max(4.0, 3.7 * rows)
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h),
                              squeeze=False, sharey=True)

    for idx, (eid, n_units, regions) in enumerate(sessions):
        ax = axes.ravel()[idx]
        max_k = 0
        for region in (pair_a, pair_b):
            if region not in regions:
                continue
            res = regions[region]
            rel = np.asarray(res.reliability)
            x = np.arange(1, len(rel) + 1)
            ax.plot(x, rel, color=_REGION_COLORS.get(region, "#444"), lw=2.0,
                    marker="o", ms=5, mec="white", mew=0.6, label=region)
            n_above = int((rel > 0.5).sum())
            top1 = float(rel[0]) if len(rel) else float("nan")
            ax.text(
                0.04, 0.95 - 0.08 * (1 if region == pair_b else 0),
                f"{region}: $\\rho_1$={top1:.2f}, {n_above} > 0.5",
                transform=ax.transAxes, ha="left", va="top",
                color=_REGION_COLORS.get(region, "#444"), fontsize=11,
            )
            max_k = max(max_k, len(rel))

        ax.axhline(0.5, ls="--", color="#666", lw=1.0)
        title = eid[:8]
        if eid.startswith(MVP_ANCHOR_EID_PREFIX):
            title += "  (MVP)"
            ax.set_facecolor("#fff8d8")
        if n_units:
            title += f"\n({pair_a}: {n_units.get(pair_a, '?')}u, " \
                     f"{pair_b}: {n_units.get(pair_b, '?')}u)"
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Component $k$")
        if idx % cols == 0:
            ax.set_ylabel(r"Reliability $\rho^{\mathrm{SVCA}}_k$")
        ax.set_xlim(0.5, max(15, max_k + 0.5))
        ax.set_ylim(-0.05, 1.0)
        if idx == 0:
            ax.legend(loc="upper right", fontsize=10)
        _strip_box(ax)

    for ax in axes.ravel()[n:]:
        ax.axis("off")

    fig.suptitle(f"SVCA reliability per session  ·  {pair_a} vs {pair_b}", y=1.005,
                 fontsize=18)
    fig.tight_layout()
    _save(fig, "reliability", out_dir)


# ============================================================================================
# Top-level orchestrators called by run_all.py
# ============================================================================================
def write_per_pair_figures(cca_df: pd.DataFrame, svca_per_session: list,
                            out_dir: Path = Path("outputs")) -> None:
    """For each (pair_a, pair_b) in cca_df, write per-pair correlations + reliability
    figures into out_dir/by_pair/<slug>/. Survival is rendered top-level only (combined
    cross-pair box-and-whisker), not per pair, since the comparison is the point."""
    if cca_df.empty:
        return
    for (a, b), pair_df in cca_df.groupby(["pair_a", "pair_b"]):
        pair_dir = Path(out_dir) / "by_pair" / _slug(a, b)
        per_pair_correlations(pair_df, a, b, pair_dir)
        if svca_per_session:
            per_pair_reliability(svca_per_session, a, b, pair_dir)


def write_top_level_figures(glm_df: pd.DataFrame, cca_df: pd.DataFrame,
                              out_dir: Path = Path("outputs"),
                              svca_per_session: list | None = None) -> None:
    """Top-level overview figures: GLM ΔR², cross-pair survival, cross-pair canonical
    correlations, cross-pair SVCA reliability."""
    if not glm_df.empty:
        fig01_glm_dr2_per_region(glm_df, out_dir)
    if not cca_df.empty:
        fig_pair_distribution(cca_df, out_dir)
        fig_correlations_aggregated(cca_df, out_dir)
        if svca_per_session:
            fig_reliability_aggregated(svca_per_session, cca_df, out_dir)


# ============================================================================================
# Backward-compat shims (so the MVP-named callers keep working)
# ============================================================================================
def fig02_svca_reliability(svca_results, out_dir=Path("outputs")) -> None:
    """Backward-compat: emits per-pair reliability figures into by_pair/ subdirs.

    Accepts either the legacy (single dict[region -> SVCAResult]) form or the new
    (list[(eid, n_units, regions)]) per-pair-session form.
    """
    if isinstance(svca_results, dict):
        # single-session legacy form: emit one figure
        regions = svca_results
        fig, ax = plt.subplots(figsize=(8.2, 5.4))
        for region, res in regions.items():
            rel = np.asarray(res.reliability)
            x = np.arange(1, len(rel) + 1)
            ax.plot(x, rel, label=region,
                    color=_REGION_COLORS.get(region, "#444444"),
                    lw=2.2, marker="o", ms=5, mec="white", mew=0.8)
        ax.axhline(0.5, ls="--", color="#888", lw=1.2, label="reliability = 0.5")
        ax.set_xlabel("Component index $k$")
        ax.set_ylabel(r"Reliability  $\rho^{\mathrm{SVCA}}_k$")
        ax.set_title("SVCA reliability")
        ax.legend(loc="upper right")
        _strip_box(ax)
        fig.tight_layout()
        _save(fig, "fig02_svca_reliability", out_dir)


def fig03_cca_canonical_correlations(df, out_dir=Path("outputs")) -> None:
    """Backward-compat shim — emits per-pair correlations into by_pair/ subdirs."""
    if df is None or df.empty:
        return
    for (a, b), pair_df in df.groupby(["pair_a", "pair_b"]):
        pair_dir = Path(out_dir) / "by_pair" / _slug(a, b)
        per_pair_correlations(pair_df, a, b, pair_dir)


def fig04_pcca_vs_cca(df, out_dir=Path("outputs")) -> None:
    """Backward-compat shim — emits per-pair survival into by_pair/ subdirs and the
    top-level fig_pair_distribution."""
    if df is None or df.empty:
        return
    for (a, b), pair_df in df.groupby(["pair_a", "pair_b"]):
        pair_dir = Path(out_dir) / "by_pair" / _slug(a, b)
        per_pair_survival(pair_df, a, b, pair_dir)
    fig_pair_distribution(df, out_dir)


def fig05_pair_distribution(df, out_dir=Path("outputs")) -> None:
    """Backward-compat alias for fig_pair_distribution."""
    fig_pair_distribution(df, out_dir)
