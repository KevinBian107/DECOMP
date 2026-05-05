"""Re-render all four figures from cached per-session csvs in data/cache/.

Usage:
    PYTHONPATH=src python scripts/render_figures.py

Does not re-run any analysis -- only redraws figures from existing GLM/SVCA/CCA caches.
Useful for tweaking visual style without paying the data-load + compute cost.
"""

from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from decomp.viz.figures import (
    fig01_glm_dr2_per_region,
    fig02_svca_reliability,
    fig03_cca_canonical_correlations,
    fig04_pcca_vs_cca,
)

CACHE = Path("data/cache")
OUT = Path("outputs")


@dataclass
class _SVCAStub:
    """Minimal SVCA result for fig02 — only reliability is needed."""
    region: str
    reliability: np.ndarray


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    glm = pd.concat(
        [pd.read_csv(p) for p in sorted(glob.glob(str(CACHE / "*_glm_results.csv")))],
        ignore_index=True,
    )
    cca = pd.concat(
        [pd.read_csv(p) for p in sorted(glob.glob(str(CACHE / "*_cca_results.csv")))],
        ignore_index=True,
    )

    # SVCA reliability per region, faceted by pair session.
    PAIR_FOCUS = ("VISp", "CB")
    pair_eids = sorted({Path(p).stem.split("_svca_scov_")[0] for p in
                        glob.glob(str(CACHE / "*_svca_scov_VISp.npy"))})
    # only keep eids that have BOTH VISp and CB (i.e. real V1+CB pair sessions)
    pair_eids = [
        e for e in pair_eids
        if (CACHE / f"{e}_svca_scov_CB.npy").exists()
    ]

    svca_per_session = []
    cca = pd.concat(
        [pd.read_csv(p) for p in sorted(glob.glob(str(CACHE / "*_cca_results.csv")))],
        ignore_index=True,
    ) if list(glob.glob(str(CACHE / "*_cca_results.csv"))) else None
    cca_eids_order = list(cca["eid"].unique()) if cca is not None else pair_eids
    pair_eids = sorted(pair_eids, key=lambda e: cca_eids_order.index(e) if e in cca_eids_order else 99)

    for eid in pair_eids:
        regions = {}
        n_units = {}
        for roi in PAIR_FOCUS:
            scov_path = CACHE / f"{eid}_svca_scov_{roi}.npy"
            var_path = CACHE / f"{eid}_svca_varcov_{roi}.npy"
            spike_path = CACHE / f"{eid}_spikes_{roi}.npy"
            if not scov_path.exists():
                continue
            scov = np.load(scov_path)
            varcov = np.load(var_path)
            rel = np.divide(scov, varcov, out=np.zeros_like(scov, dtype=float), where=varcov > 0)
            regions[roi] = _SVCAStub(region=roi, reliability=rel)
            if spike_path.exists():
                n_units[roi] = int(np.load(spike_path, mmap_mode="r").shape[0])
        if regions:
            svca_per_session.append((eid, n_units, regions))

    print(f"GLM rows: {len(glm)}  CCA rows: {len(cca) if cca is not None else 0}  "
          f"SVCA pair sessions: {len(svca_per_session)}")

    fig01_glm_dr2_per_region(glm, OUT)
    if svca_per_session:
        fig02_svca_reliability(svca_per_session, OUT)
    if cca is not None and len(cca):
        fig03_cca_canonical_correlations(cca, OUT)
        fig04_pcca_vs_cca(cca, OUT)
    print(f"Wrote 4 figures to {OUT.resolve()}")


if __name__ == "__main__":
    main()
