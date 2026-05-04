"""DECOMP MVP entry point — runs the full pipeline end-to-end and writes outputs/.

Two-track design:
  - Pool sessions  -> per-region GLM ΔR² (literature-reproduction figure across all 4 ROIs).
                       Each session contributes only the ROIs actually present in it.
  - Pair sessions  -> within-region SVCA + cross-region V1↔CB CCA + pCCA (the answer figure).
                       BWM 2023_12 has 3 V1+CB sessions; one strong (63V/46CB) and two
                       borderline. M1+V1 simultaneous coverage is zero across the freeze.

Usage:
    python run_all.py
    python run_all.py --rerun-from svca
    python run_all.py --max-pool 3 --max-pair 3
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("run_all")

CACHE_DIR = Path("data/cache")
OUT_DIR = Path("outputs")


def set_seeds(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)


def get_one():
    from one.api import ONE  # type: ignore[import-not-found]
    return ONE(base_url="https://openalyx.internationalbrainlab.org",
               password="international", silent=True)


def main(args: argparse.Namespace) -> None:
    set_seeds(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    from decomp.pipeline.stage_data import load_and_bin, select_sessions
    from decomp.pipeline.stage_svca import run_session_svca
    from decomp.pipeline.stage_cca import run_session_cca
    from decomp.pipeline.stage_glm import run_session_glm
    from decomp.viz.figures import (
        fig01_glm_dr2_per_region,
        fig02_svca_reliability,
        fig03_cca_canonical_correlations,
        fig04_pcca_vs_cca,
    )
    from brainwidemap import bwm_query  # type: ignore[import-not-found]

    one = get_one()

    # ---- Stage 1: Session selection ----------------------------------------------------------
    log.info("Stage 1: session selection")
    plan = select_sessions(one, target_n=3, freeze=args.freeze, cache_dir=CACHE_DIR)
    log.info("  %s", plan.strategy_note)

    pair_eids = plan.pair_eids[: args.max_pair]
    pool_eids = {roi: eids[: args.max_pool] for roi, eids in plan.pool_eids.items()}
    union_eids = sorted(set(pair_eids) | {e for v in pool_eids.values() for e in v})
    log.info("  pair sessions (V1+CB): %s", pair_eids)
    log.info("  pool sessions per ROI: %s", pool_eids)
    log.info("  union: %d sessions to load", len(union_eids))

    bwm = bwm_query(one, freeze=args.freeze)
    eid_to_pids = bwm.groupby("eid")["pid"].apply(list).to_dict()

    glm_rows: list[pd.DataFrame] = []
    cca_rows: list[pd.DataFrame] = []
    svca_for_fig: dict[str, object] = {}

    # Track which ROIs each session should contribute to GLM ΔR²
    session_glm_rois: dict[str, set[str]] = {}
    for roi, eids in pool_eids.items():
        for eid in eids:
            session_glm_rois.setdefault(eid, set()).add(roi)
    # Pair sessions always contribute V1 and CB
    for eid in pair_eids:
        session_glm_rois.setdefault(eid, set()).update({"VISp", "CB"})

    for eid in union_eids:
        pids = eid_to_pids.get(eid, [])
        if not pids:
            log.warning("  no pids for %s, skipping", eid)
            continue

        glm_rois = sorted(session_glm_rois.get(eid, set()))
        log.info("Session %s with %d probes (GLM ROIs: %s)", eid, len(pids), glm_rois)

        # Load + bin: load all 4 ROIs even if only a subset are needed (cheap union load)
        try:
            sd, binned = load_and_bin(one, eid, pids, rois=plan.rois_used, cache_dir=CACHE_DIR)
        except Exception as e:
            log.warning("  load_and_bin failed for %s (%s): skipping session", eid, type(e).__name__)
            continue
        if not binned.spikes_by_roi:
            log.warning("  no usable spike data for %s, skipping", eid)
            continue

        # Stage 3: GLM (only for ROIs this session contributes to)
        if args.rerun_from in (None, "data", "glm") and glm_rois:
            glm_df = run_session_glm(sd, binned, rois=glm_rois, cache_dir=CACHE_DIR,
                                      random_state=args.seed)
            glm_rows.append(glm_df)

        # Stage 4 + 5: only on V1+CB pair sessions
        if eid in pair_eids and args.rerun_from in (None, "data", "glm", "svca", "cca"):
            svca_results = run_session_svca(binned, cache_dir=CACHE_DIR,
                                             random_state=args.seed)
            for roi, res in svca_results.items():
                svca_for_fig.setdefault(roi, res)

            if args.rerun_from in (None, "data", "glm", "svca", "cca"):
                pair_rois = ["VISp", "CB"]
                pair_svca = {r: svca_results[r] for r in pair_rois if r in svca_results}
                if len(pair_svca) == 2:
                    cca_df = run_session_cca(binned, pair_svca, cache_dir=CACHE_DIR,
                                             n_components=args.n_components,
                                             n_surrogates=args.n_surrogates,
                                             random_state=args.seed)
                    cca_rows.append(cca_df)

    glm_all = pd.concat(glm_rows, ignore_index=True) if glm_rows else pd.DataFrame()
    cca_all = pd.concat(cca_rows, ignore_index=True) if cca_rows else pd.DataFrame()

    if not glm_all.empty:
        fig01_glm_dr2_per_region(glm_all, OUT_DIR)
    if svca_for_fig:
        fig02_svca_reliability(svca_for_fig, OUT_DIR)
    if not cca_all.empty:
        fig03_cca_canonical_correlations(cca_all, OUT_DIR)
        fig04_pcca_vs_cca(cca_all, OUT_DIR)

    summary = {
        "strategy": plan.strategy_note,
        "pair_eids": pair_eids,
        "pool_eids": pool_eids,
        "union_eids": union_eids,
        "rois_used": plan.rois_used,
        "n_glm_rows": int(len(glm_all)),
        "n_cca_rows": int(len(cca_all)),
        "seed": args.seed,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Wrote %s", OUT_DIR / "summary.json")
    log.info("Done.")


def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--rerun-from", choices=["data", "glm", "svca", "cca"], default=None)
    p.add_argument("--max-pair", type=int, default=3,
                   help="max V1+CB pair sessions for cross-region CCA")
    p.add_argument("--max-pool", type=int, default=10,
                   help="max sessions per ROI for the per-region GLM pool")
    p.add_argument("--freeze", default="2023_12_bwm_release")
    p.add_argument("--n-components", type=int, default=8)
    p.add_argument("--n-surrogates", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    main(cli())
