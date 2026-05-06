"""Post-run interpretation: per-pair survival summary + branching narrative.

Reads the on-disk caches written by run_all.py (per-session CCA csvs) and produces:
  - per_pair_summary  : one row per (pair, eid)
  - cross_pair_comparison : one row per pair, with median / IQR / paired Wilcoxon vs other pairs
  - auto_interpret    : a markdown narrative branching on the observed pattern

The narrative is a *first draft* — meant to be read, edited by hand for the final
docs/expansion_analysis.md. Branches on:
  (A) all pairs survival ≈ 1: claim is generalized ("any cortical pair preserves")
  (B) V1↔CB > V1↔M1: V1's coupling to CB is special
  (C) V1↔M1 ≥ V1↔CB: V1 is more aligned with M1 than CB
  (D) high heterogeneity across sessions: report distributions, not means
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def load_cca_results(cache_dir: Path = Path("data/cache")) -> pd.DataFrame:
    """Concatenate every per-session CCA csv in cache_dir."""
    paths = sorted(glob.glob(str(Path(cache_dir) / "*_cca_results.csv")))
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def per_pair_summary(cca_df: pd.DataFrame) -> pd.DataFrame:
    """Per (pair, eid): Σρ_CCA, Σρ_pCCA, survival ratio, # components above null."""
    if cca_df.empty:
        return pd.DataFrame()
    g = (
        cca_df.groupby(["pair_a", "pair_b", "eid"])
        .agg(
            rho_cca_sum=("rho_cca", "sum"),
            rho_pcca_sum=("rho_pcca", "sum"),
            n_components=("component", "size"),
            n_above_null_cca=(
                "rho_cca",
                lambda s: int((s.values
                               > cca_df.loc[s.index, "null_cca_99"].values).sum()),
            ),
        )
        .reset_index()
    )
    g["survival"] = g["rho_pcca_sum"] / g["rho_cca_sum"].replace(0, np.nan)
    g["pair"] = g["pair_a"] + ":" + g["pair_b"]
    return g.sort_values(["pair", "rho_cca_sum"], ascending=[True, False]).reset_index(drop=True)


def cross_pair_comparison(per_pair: pd.DataFrame) -> pd.DataFrame:
    """Per pair: median survival, IQR, n_sessions, n_above_null. Pairwise Wilcoxon
    is computed on sessions where two pairs both appear (rare in practice — usually
    no overlap because pair sessions for different pairs are different sessions)."""
    if per_pair.empty:
        return pd.DataFrame()
    g = (
        per_pair.groupby("pair")
        .agg(
            n_sessions=("eid", "nunique"),
            survival_median=("survival", "median"),
            survival_q25=("survival", lambda s: float(np.nanquantile(s, 0.25))),
            survival_q75=("survival", lambda s: float(np.nanquantile(s, 0.75))),
            survival_min=("survival", "min"),
            survival_max=("survival", "max"),
            cca_median=("rho_cca_sum", "median"),
            n_components_median=("n_components", "median"),
            n_above_null_median=("n_above_null_cca", "median"),
        )
        .reset_index()
    )
    return g.sort_values("pair").reset_index(drop=True)


def paired_wilcoxon_across_pairs(per_pair: pd.DataFrame) -> dict:
    """For each pair of pairs, paired Wilcoxon on sessions where both pairs were run.
    Returns dict[(pair_x, pair_y) -> {n_paired, statistic, p_value}]. Empty if no overlap.
    """
    out = {}
    pairs = sorted(per_pair["pair"].unique())
    for i, px in enumerate(pairs):
        for py in pairs[i + 1:]:
            sx = per_pair[per_pair["pair"] == px][["eid", "survival"]].rename(
                columns={"survival": f"surv_{px}"}
            )
            sy = per_pair[per_pair["pair"] == py][["eid", "survival"]].rename(
                columns={"survival": f"surv_{py}"}
            )
            merged = sx.merge(sy, on="eid", how="inner").dropna()
            if len(merged) >= 3:
                try:
                    stat, p = wilcoxon(merged[f"surv_{px}"], merged[f"surv_{py}"])
                    out[(px, py)] = {"n_paired": len(merged),
                                      "statistic": float(stat), "p_value": float(p)}
                except ValueError:
                    out[(px, py)] = {"n_paired": len(merged), "statistic": float("nan"),
                                      "p_value": float("nan")}
            else:
                out[(px, py)] = {"n_paired": len(merged)}
    return out


def auto_interpret(per_pair: pd.DataFrame, cross: pd.DataFrame,
                   paired: dict, summary_path: Path | None = None) -> str:
    """Return a markdown narrative interpreting the cross-pair survival pattern."""
    if per_pair.empty:
        return "# Expansion analysis\n\nNo CCA results found. Did the run finish?\n"

    lines: list[str] = ["# Expansion analysis — interpretation\n"]
    if summary_path and summary_path.exists():
        s = json.loads(Path(summary_path).read_text())
        lines.append(f"_Run scope_: {s.get('strategy', '(no strategy note)')}\n")
        lines.append("\n## Run scope\n")
        for k, v in s.get("expansion", {}).items():
            lines.append(f"- **{k}**: {v.get('n_sessions', '?')} sessions "
                         f"(min_units = {v.get('min_units', '?')})")
        lines.append("")

    # Per-pair summary table
    lines.append("\n## Per-pair survival ratio summary\n")
    lines.append("| Pair | n sessions | Median survival | IQR | n components above null (median) |")
    lines.append("|---|---|---|---|---|")
    for _, row in cross.iterrows():
        lines.append(
            f"| {row['pair']} | {row['n_sessions']} | "
            f"{row['survival_median']:.2f} | "
            f"[{row['survival_q25']:.2f}, {row['survival_q75']:.2f}] | "
            f"{row['n_above_null_median']:.0f} |"
        )

    # Branching narrative
    lines.append("\n## Branching interpretation\n")
    medians = dict(zip(cross["pair"], cross["survival_median"]))

    def _get(*candidates):
        for c in candidates:
            if c in medians:
                return medians[c]
        return None

    v1cb = _get("VIS:CB", "VISp:CB")
    v1m1 = _get("VIS:MO", "VISp:MO")
    cbm1 = _get("CB:MO")

    iqrs = dict(zip(cross["pair"], cross["survival_q75"] - cross["survival_q25"]))
    max_iqr = max(iqrs.values()) if iqrs else 0.0

    branch_text = ""
    if max_iqr > 0.4:
        branch_text = (
            "**Branch (D): high heterogeneity across sessions.** Within at least one pair, the "
            "survival ratio varies by more than 0.4 across sessions. The point-median "
            "comparison below should be read with caution — distributions matter more than "
            "summary statistics here."
        )
    elif v1cb is not None and v1m1 is not None and abs(v1cb - v1m1) < 0.07:
        if v1cb >= 0.85 and (cbm1 is None or cbm1 >= 0.85):
            branch_text = (
                "**Branch (A): all cortical pairs preserve shared structure under wheel + pupil "
                "partialling.** Median survival is comparable across V1↔CB, V1↔M1, and CB↔M1, "
                "and all sit close to 1. The original V1↔CB result is **generalized**: this "
                "is a property of cortical region pairs in the BWM dataset, not specific to "
                "V1's relationship with cerebellum."
            )
        else:
            branch_text = (
                "Survival ratios across the available pairs are similar but not all near 1. "
                "Inspect per-session distributions in fig05 before concluding."
            )
    elif v1cb is not None and v1m1 is not None and v1cb > v1m1 + 0.07:
        branch_text = (
            "**Branch (B): V1↔CB shared structure survives partialling more than V1↔M1's does.** "
            f"Median survival V1↔CB = {v1cb:.2f} vs V1↔M1 = {v1m1:.2f}. This is consistent with "
            "V1 having a relationship with cerebellum that is qualitatively different from its "
            "relationship with motor cortex — strengthens the original V1↔CB claim by giving it "
            "a comparison anchor."
        )
    elif v1cb is not None and v1m1 is not None and v1m1 > v1cb + 0.07:
        branch_text = (
            "**Branch (C): V1↔M1 shared structure survives partialling more than V1↔CB's does.** "
            f"Median survival V1↔M1 = {v1m1:.2f} vs V1↔CB = {v1cb:.2f}. V1's coupling to motor "
            "cortex appears stronger than to cerebellum after global state is removed. This "
            "would reframe the project's headline: V1's movement signal is more motor-cortex-like "
            "than cerebellum-like at the population-axis level."
        )
    else:
        branch_text = (
            "Pattern does not fit a clean A/B/C/D branch. Inspect fig04 and fig05 directly."
        )
    lines.append(branch_text)

    # MVP regression check
    lines.append("\n## MVP regression check\n")
    mvp_anchor = "41431f53-69fd-4e3b-80ce-ea62e03bf9c7"
    anchor_row = per_pair[
        ((per_pair["pair_a"] == "VIS") | (per_pair["pair_a"] == "VISp"))
        & (per_pair["pair_b"] == "CB")
        & (per_pair["eid"] == mvp_anchor)
    ]
    if len(anchor_row):
        s = float(anchor_row["survival"].iloc[0])
        lines.append(
            f"MVP-anchor session `{mvp_anchor[:8]}` survival ratio in this run: **{s:.3f}** "
            f"(MVP reported 0.98). "
            f"{'✓ matches within tolerance' if abs(s - 0.98) < 0.05 else '⚠ deviates from MVP — investigate'}."
        )
    else:
        lines.append(
            f"Anchor session `{mvp_anchor[:8]}` not in V1↔CB results. Either the run skipped it "
            "or the cache was cleared — investigate before trusting other comparisons."
        )

    if paired:
        lines.append("\n## Paired Wilcoxon (sessions appearing in both pairs)\n")
        lines.append("| Pair X | Pair Y | n paired | statistic | p |")
        lines.append("|---|---|---|---|---|")
        for (px, py), v in paired.items():
            n = v.get("n_paired", 0)
            stat = v.get("statistic", float("nan"))
            p = v.get("p_value", float("nan"))
            lines.append(
                f"| {px} | {py} | {n} | "
                f"{stat:.2f} | {p:.4f} |"
            )

    # Caveats
    lines.append("\n## Caveats\n")
    lines.append(
        "- The 3-region partial CCA `CCA(V1, CB | M1)` is **not** in this analysis — "
        "0 sessions in BWM 2023_12 have simultaneous V1+CB+M1 recordings. We cannot test "
        "the 'is V1↔CB shared structure just an echo of motor cortex's command' hypothesis "
        "directly within-session on this dataset."
    )
    lines.append(
        "- The wide 'VIS' definition includes higher visual areas (VISa, VISam, VISpm, etc.). "
        "This trades the MVP-precise V1-only claim for a 3× larger sample. The MVP-anchor row "
        "above checks that the original V1-only claim is preserved."
    )
    lines.append(
        "- Pair sample sizes are small (V1↔M1: 4–8, CB↔M1: 1–3). Across-pair comparisons are "
        "indicative, not statistically sharp."
    )

    return "\n".join(lines) + "\n"


def write_expansion_doc(
    cache_dir: Path = Path("data/cache"),
    out_path: Path = Path("docs/expansion_analysis.md"),
    summary_path: Path = Path("outputs/summary.json"),
) -> str:
    """Top-level: load CCA results, build summaries, write docs/expansion_analysis.md.

    Returns the written content (also writes to disk).
    """
    cca = load_cca_results(cache_dir)
    pp = per_pair_summary(cca)
    cross = cross_pair_comparison(pp)
    paired = paired_wilcoxon_across_pairs(pp)
    text = auto_interpret(pp, cross, paired, summary_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(text)
    return text


if __name__ == "__main__":
    print(write_expansion_doc())
