"""Render the README banner: a stripped-down, banner-aspect population activity heatmap
spanning the three regions analysed in this project (VIS, MO, CB).

No single BWM 2023_12 session has simultaneous Neuropixels coverage of all three regions
(see docs/expansion_analysis.md), so the banner composites three regions from two sessions:
VIS and CB from the V1↔CB anchor (41431f53), MO from a V1↔M1 session (4aa1d525). Each row
uses its own 60-s time window.

Output: docs/figures/banner.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import numpy as np

CACHE = Path("data/cache")

# (region, eid, vmax, window_seconds_within_session)
ROWS = [
    ("VIS", "41431f53-69fd-4e3b-80ce-ea62e03bf9c7", 1.0, (2635.0, 2695.0)),
    ("MO",  "4aa1d525-5c7d-4c50-a147-ec53a9014812", 2.0, (1500.0, 1560.0)),
    ("CB",  "41431f53-69fd-4e3b-80ce-ea62e03bf9c7", 2.0, (2635.0, 2695.0)),
]

OUT = Path("docs/figures/banner.png")


def _load_window(eid: str, region: str, t0: float, t1: float):
    bc = np.load(CACHE / f"{eid}_bin_centers.npy")
    sp = np.load(CACHE / f"{eid}_spikes_{region}.npy")
    mask = (bc >= t0) & (bc <= t1)
    rate = sp.mean(axis=1)
    sp_sorted = sp[np.argsort(rate)]
    return bc[mask], sp_sorted[:, mask]


def main() -> None:
    rows = []
    for region, eid, vmax, (t0, t1) in ROWS:
        bc, mat = _load_window(eid, region, t0, t1)
        rows.append({"region": region, "bc": bc, "mat": mat, "vmax": vmax})

    height_ratios = [r["mat"].shape[0] for r in rows]
    fig, axes = plt.subplots(
        len(rows), 1, figsize=(15.0, 4.0),
        gridspec_kw={"height_ratios": height_ratios, "hspace": 0.04},
        sharex=False,
    )

    for ax, r in zip(axes, rows):
        n_units = r["mat"].shape[0]
        bc = r["bc"]
        extent = (bc[0], bc[-1], 0, n_units)
        ax.imshow(r["mat"], aspect="auto", origin="lower", extent=extent,
                   cmap="magma", vmin=0.0, vmax=r["vmax"], interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.text(0.005, 0.5, r["region"], transform=ax.transAxes,
                fontsize=18, fontweight="bold", color="white",
                ha="left", va="center",
                path_effects=[
                    patheffects.Stroke(linewidth=2.0, foreground="black"),
                    patheffects.Normal(),
                ])

    # Time scale bar on the bottom panel
    bc = rows[-1]["bc"]
    ax = axes[-1]
    ax.plot([bc[-1] - 10.0, bc[-1] - 0.5], [-2.5, -2.5],
              color="white", lw=2.5,
              path_effects=[patheffects.Stroke(linewidth=4.0, foreground="black"),
                              patheffects.Normal()],
              clip_on=False)
    ax.text(bc[-1] - 5.25, -5.5, "10 s", color="#222",
              fontsize=11, ha="center", va="top",
              transform=ax.transData)

    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=220, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    n_per = ", ".join(f"{r['region']}: {r['mat'].shape[0]}u" for r in rows)
    print(f"Wrote {OUT}  ({n_per})")


if __name__ == "__main__":
    main()
