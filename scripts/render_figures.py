"""Re-render all figures from cached per-session csvs in data/cache/.

Usage:
    PYTHONPATH=src python scripts/render_figures.py

Does not re-run any analysis -- only redraws figures from existing GLM/SVCA/CCA caches.
Useful for tweaking visual style without paying the data-load + compute cost.

Output layout (mirrors run_all.py):
    outputs/fig01_glm_dr2_per_region.png
    outputs/fig_pair_distribution.png
    outputs/by_pair/<a>_<b>/correlations.png
    outputs/by_pair/<a>_<b>/survival.png
    outputs/by_pair/<a>_<b>/reliability.png
"""

from __future__ import annotations

import glob
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from decomp.viz.figures import write_per_pair_figures, write_top_level_figures

CACHE = Path("data/cache")
OUT = Path("outputs")


@dataclass
class _SVCAStub:
    """Minimal SVCA result for the per-pair reliability figure — only reliability is needed."""
    region: str
    reliability: np.ndarray
    scov: np.ndarray = None
    varcov: np.ndarray = None
    k_reliable: int = 0
    scores: np.ndarray = None
    full_components: np.ndarray = None


def _load_svca_per_session(cca: pd.DataFrame) -> list:
    """For every (eid, region) pair on disk, load reliability and pack into the per-pair-
    session structure expected by write_per_pair_figures.

    Returns: list[(eid, n_units_dict, regions_dict)]  where regions_dict maps region->_SVCAStub.
    Only sessions present in cca["eid"] are included.
    """
    if cca is None or cca.empty:
        return []

    out: dict[str, dict] = {}  # eid -> {"regions": {roi: stub}, "n_units": {roi: int}}
    for path in sorted(glob.glob(str(CACHE / "*_svca_scov_*.npy"))):
        m = re.match(r"^(?P<eid>[a-f0-9-]+)_svca_scov_(?P<roi>.+)\.npy$",
                      Path(path).name)
        if not m:
            continue
        eid = m["eid"]; roi = m["roi"]
        if eid not in cca["eid"].values:
            continue
        scov = np.load(path)
        varcov = np.load(path.replace("_scov_", "_varcov_"))
        rel = np.divide(scov, varcov,
                         out=np.zeros_like(scov, dtype=float), where=varcov > 0)
        out.setdefault(eid, {"regions": {}, "n_units": {}})
        out[eid]["regions"][roi] = _SVCAStub(
            region=roi, reliability=rel, scov=scov, varcov=varcov
        )
        spike_path = CACHE / f"{eid}_spikes_{roi}.npy"
        if spike_path.exists():
            out[eid]["n_units"][roi] = int(np.load(spike_path, mmap_mode="r").shape[0])

    # Order by CCA-row order so figures match the rest of the run
    eid_order = list(cca["eid"].unique())
    return [
        (eid, out[eid]["n_units"], out[eid]["regions"])
        for eid in eid_order
        if eid in out
    ]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    glm_paths = sorted(glob.glob(str(CACHE / "*_glm_results.csv")))
    cca_paths = sorted(glob.glob(str(CACHE / "*_cca_results.csv")))
    glm = (pd.concat([pd.read_csv(p) for p in glm_paths], ignore_index=True)
           if glm_paths else pd.DataFrame())
    cca = (pd.concat([pd.read_csv(p) for p in cca_paths], ignore_index=True)
           if cca_paths else pd.DataFrame())

    svca_per_session = _load_svca_per_session(cca)
    print(f"GLM rows: {len(glm)}  CCA rows: {len(cca)}  "
          f"SVCA sessions: {len(svca_per_session)}")

    write_top_level_figures(glm, cca, OUT, svca_per_session=svca_per_session)
    write_per_pair_figures(cca, svca_per_session, OUT)
    print(f"Wrote figures to {OUT.resolve()}")


if __name__ == "__main__":
    main()
