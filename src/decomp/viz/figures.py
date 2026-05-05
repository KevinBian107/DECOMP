"""Matplotlib figure factory for the four MVP deliverables.

Modern, publication-style plots — sans-serif, 14pt+ fonts, restrained palette, no chartjunk.
All functions accept the relevant results dict / DataFrame and write png + pdf to `outputs/`.
"""

from __future__ import annotations

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

# Region palette — restrained but distinguishable, slightly desaturated
_REGION_COLORS = {
    "VISp": "#3a7ca5",   # blue
    "CB":   "#c44536",   # red-brown
    "MO":   "#3a8a4d",   # green
    "CA1":  "#7a5db8",   # purple
}

_KERNEL_LABEL = {
    "dR2_movement": "Movement\n(wheel + ME + lick + fmove)",
    "dR2_stim":     "Stimulus",
    "dR2_choice":   "Choice",
}


def _save(fig: plt.Figure, name: str, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png")
    plt.close(fig)


def _strip_box(ax: plt.Axes) -> None:
    """Tidy spines + ticks, the way most journal figures look."""
    ax.tick_params(direction="out", length=4, width=1.0, pad=4)
    ax.spines["bottom"].set_position(("outward", 6))
    ax.spines["left"].set_position(("outward", 6))


# ---- fig01: per-region GLM ΔR² -----------------------------------------------------------
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

        # Strip plot of medians of each session within each region (gives a sense of spread)
        for j, (r, vals) in enumerate(zip(rois, data)):
            if len(vals):
                jitter = (np.random.default_rng(j + hash(k) % 1000).normal(0, 0.06, size=len(vals)))
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


# ---- fig02: SVCA reliability spectrum (per pair session) ---------------------------------
def fig02_svca_reliability(svca_results: dict | list,
                            out_dir: Path = Path("outputs")) -> None:
    """SVCA reliability spectrum per region, faceted by pair session.

    `svca_results` accepts either:
      - dict[region -> SVCAResult-like]              (single-session legacy form)
      - list[(eid, n_units_dict, dict[region -> SVCAResult-like])]  (per-session faceted)

    Each panel shows VISp and CB reliability spectra for one pair session, with
    Stringer's 0.5 threshold as a reference and an annotation of how many components
    clear it.
    """
    # normalize to per-session list
    if isinstance(svca_results, dict):
        sessions = [(None, {}, svca_results)]
    else:
        sessions = list(svca_results)

    n = len(sessions)
    fig, axes = plt.subplots(1, n, figsize=(5.6 * n, 5.0), sharey=True, squeeze=False)

    for ax, (eid, n_units, regions) in zip(axes.ravel(), sessions):
        max_k = 0
        legend_lines: list[str] = []
        for region, res in regions.items():
            rel = np.asarray(res.reliability)
            x = np.arange(1, len(rel) + 1)
            color = _REGION_COLORS.get(region, "#444444")
            ax.plot(x, rel, color=color, lw=2.2, marker="o", ms=6,
                    mec="white", mew=0.8, label=region)
            n_above = int((rel > 0.5).sum())
            max_k = max(max_k, len(rel))
            top1 = float(rel[0]) if len(rel) else float("nan")
            legend_lines.append((color, region, n_above, top1))

        ax.axhline(0.5, ls="--", color="#666666", lw=1.0)
        ax.text(0.985, 0.515, "Stringer 0.5",
                transform=ax.get_yaxis_transform(),
                fontsize=10, color="#666666", ha="right", va="bottom")

        # right-side info block: per-region #components above 0.5 and top-1 reliability
        for j, (color, region, n_above, top1) in enumerate(legend_lines):
            ax.text(
                0.04, 0.95 - 0.08 * j,
                f"{region}:  $\\rho_1$ = {top1:.2f}   ·   {n_above} comp > 0.5",
                transform=ax.transAxes, ha="left", va="top",
                color=color, fontsize=11.5, fontweight="regular",
            )

        ax.set_xlabel("Component index $k$")
        if ax is axes.ravel()[0]:
            ax.set_ylabel("Reliability  $\\rho^{\\mathrm{SVCA}}_k$")
        ax.set_xlim(0.5, max(15, max_k + 0.5))
        ax.set_ylim(-0.05, 1.0)

        title = ""
        if eid is not None:
            title = eid[:8]
            if n_units:
                pieces = " · ".join(f"{r}: {nu}u" for r, nu in n_units.items() if r in regions)
                title = f"{title}   ({pieces})"
        ax.set_title(title or "SVCA reliability", fontsize=14)
        _strip_box(ax)

    fig.suptitle(
        "SVCA reliability per pair session  ·  V1 vs CB",
        y=1.02, fontsize=18,
    )
    fig.tight_layout()
    _save(fig, "fig02_svca_reliability", out_dir)


# ---- fig03: cross-validated canonical correlations with phase-shuffle null ---------------
def fig03_cca_canonical_correlations(df: pd.DataFrame, out_dir: Path = Path("outputs")) -> None:
    pairs = list(df[["pair_a", "pair_b"]].drop_duplicates().itertuples(index=False, name=None))
    n_pairs = len(pairs)
    if n_pairs == 0:
        return

    eids = sorted(df["eid"].unique()) if "eid" in df.columns else [None]
    n_panels = len(eids) * n_pairs
    cols = min(3, max(1, n_panels))
    rows = int(np.ceil(n_panels / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.4 * cols, 4.4 * rows), squeeze=False)

    panel = 0
    for eid in eids:
        for (a, b) in pairs:
            ax = axes.ravel()[panel]
            sub = df[(df["pair_a"] == a) & (df["pair_b"] == b)]
            if eid is not None:
                sub = sub[sub["eid"] == eid]
            sub = sub.sort_values("component")
            x = sub["component"].to_numpy() + 1

            ax.fill_between(
                x, sub["null_cca_99"], np.zeros_like(sub["null_cca_99"]),
                color="#bbbbbb", alpha=0.35, label="phase-shuffle 99% null",
            )
            ax.plot(x, sub["rho_cca"], "o-", color=_REGION_COLORS["VISp"], lw=2.2, ms=7,
                    mec="white", mew=0.8, label="$\\rho_k^{\\mathrm{CCA}}$")
            ax.plot(x, sub["rho_pcca"], "s--", color=_REGION_COLORS["CB"], lw=2.0, ms=7,
                    mec="white", mew=0.8, label="$\\rho_k^{\\mathrm{pCCA}}$")

            tag = f"{a}–{b}"
            if eid is not None:
                tag += f"  ·  {eid[:8]}"
            ax.set_title(tag, fontsize=14)
            ax.set_xlabel("Canonical component $k$")
            ax.set_ylabel("Cross-validated $\\rho_k$")
            ax.set_ylim(-0.05, 1.0)
            ax.set_xticks(x)
            ax.legend(fontsize=10, loc="upper right")
            _strip_box(ax)
            panel += 1
    for ax in axes.ravel()[panel:]:
        ax.axis("off")
    fig.suptitle(
        "Canonical correlations with phase-shuffle null  ·  V1 vs CB",
        y=1.02, fontsize=18,
    )
    fig.tight_layout()
    _save(fig, "fig03_cca_canonical_correlations", out_dir)


# ---- fig04: the answer figure — pCCA vs CCA survival -------------------------------------
def fig04_pcca_vs_cca(df: pd.DataFrame, out_dir: Path = Path("outputs")) -> None:
    summary = (
        df.groupby(["eid", "pair_a", "pair_b"])
        .agg(rho_cca=("rho_cca", "sum"), rho_pcca=("rho_pcca", "sum"))
        .reset_index()
    )
    summary["survival"] = summary["rho_pcca"] / summary["rho_cca"].replace(0, np.nan)
    summary["label"] = (summary["pair_a"] + "vs" + summary["pair_b"]
                        + "\n" + summary["eid"].str.slice(0, 8))

    x = np.arange(len(summary))
    width = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), gridspec_kw={"width_ratios": [1.4, 1]})

    bars_cca = axes[0].bar(
        x - width / 2, summary["rho_cca"], width,
        label="$\\sum_k \\rho_k^{\\mathrm{CCA}}$", color="#5e5e5e", edgecolor="#222222",
    )
    bars_pcca = axes[0].bar(
        x + width / 2, summary["rho_pcca"], width,
        label="$\\sum_k \\rho_k^{\\mathrm{pCCA}}$\n(out: wheel + pupil)",
        color=_REGION_COLORS["CB"], edgecolor="#222222",
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(summary["label"], fontsize=11)
    axes[0].set_ylabel("Sum of cross-validated $\\rho_k$")
    axes[0].set_title("Shared structure: raw vs partialled")
    axes[0].legend(loc="upper left", framealpha=1.0, frameon=False)
    _strip_box(axes[0])

    # Right panel: survival ratio with annotation
    bars_surv = axes[1].bar(
        x, summary["survival"], width=0.6,
        color=[_REGION_COLORS["MO"]] * len(summary), edgecolor="#222222",
    )
    axes[1].axhline(1.0, color="#888888", lw=0.8, ls="-", zorder=0)
    axes[1].axhline(0.0, color="#222222", lw=0.8, zorder=0)
    for xb, v in zip(x, summary["survival"]):
        if pd.notnull(v):
            axes[1].text(xb, v + 0.03, f"{v:.2f}", ha="center", va="bottom",
                         fontsize=12, color="#222222", fontweight="regular")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(summary["label"], fontsize=11)
    axes[1].set_ylabel("Survival ratio   $\\rho^{\\mathrm{pCCA}} / \\rho^{\\mathrm{CCA}}$")
    axes[1].set_ylim(0, max(1.15, (summary["survival"].max() if len(summary) else 1.0) * 1.15))
    axes[1].set_title("Low: inherited global state  /  High: region-specific structure")
    _strip_box(axes[1])

    fig.suptitle(
        "Does the V1vsCB shared subspace survive partialling out wheel + pupil?",
        y=1.02, fontsize=18,
    )
    fig.tight_layout()
    _save(fig, "fig04_pcca_vs_cca", out_dir)
