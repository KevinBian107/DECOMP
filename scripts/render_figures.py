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

    # SVCA reliability per region. Restrict to {VISp, CB} — those are the regions that feed
    # into the V1+CB cross-region CCA (fig03/04). Pull from the strongest pair session.
    svca: dict[str, _SVCAStub] = {}
    PAIR_FOCUS = ("VISp", "CB")
    STRONG_EID_PREFIX = "41431f53"  # CSH_ZAD_022, 63 V1 + 46 CB
    for roi in PAIR_FOCUS:
        candidates = sorted(glob.glob(str(CACHE / f"{STRONG_EID_PREFIX}*_svca_scov_{roi}.npy")))
        if not candidates:
            candidates = sorted(glob.glob(str(CACHE / f"*_svca_scov_{roi}.npy")))
        if not candidates:
            continue
        scov_path = candidates[0]
        var_path = scov_path.replace("_scov_", "_varcov_")
        scov = np.load(scov_path)
        varcov = np.load(var_path)
        rel = np.divide(scov, varcov, out=np.zeros_like(scov, dtype=float), where=varcov > 0)
        svca[roi] = _SVCAStub(region=roi, reliability=rel)

    print(f"GLM rows: {len(glm)}  CCA rows: {len(cca)}  SVCA regions: {list(svca)}")

    fig01_glm_dr2_per_region(glm, OUT)
    if svca:
        fig02_svca_reliability(svca, OUT)
    fig03_cca_canonical_correlations(cca, OUT)
    fig04_pcca_vs_cca(cca, OUT)
    print(f"Wrote 4 figures to {OUT.resolve()}")


if __name__ == "__main__":
    main()
