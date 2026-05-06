"""DECOMP pipeline entry point — runs the full pipeline end-to-end and writes outputs/.

Multi-pair design (post-MVP):
  - Pool sessions  -> per-region GLM ΔR² across the 4-region ROI_ORDER (VISp, CB, MO, CA1)
  - Pair sessions  -> within-region SVCA + cross-region CCA + pCCA, looped over an
                       arbitrary list of region pairs (e.g. VIS:CB, VIS:MO, CB:MO)

Usage:
    python run_all.py
    python run_all.py --pairs "VISp:CB"                              # MVP-equivalent
    python run_all.py --pairs "VIS:CB,VIS:MO,CB:MO"                  # post-MVP expansion
    python run_all.py --max-pair 10 --max-pool 10
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

# MVP frozen reference: the 3 V1+CB pair sessions used in the MVP analysis. Surfaced in
# summary.json for downstream interpretation / regression checking.
MVP_V1CB_EIDS = [
    "41431f53-69fd-4e3b-80ce-ea62e03bf9c7",  # CSH_ZAD_022 (strong)
    "09b2c4d1-058d-4c84-9fd4-97530f85baf6",  # ZFM-01577   (mixed)
    "a7763417-e0d6-4f2a-aa55-e382fd9b5fb8",  # ibl_witten_20 (uninformative)
]


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

    from decomp.data.sessions import parse_pair_specs
    from decomp.pipeline.stage_data import load_and_bin, select_sessions
    from decomp.pipeline.stage_pair_runner import run_pair_session
    from decomp.pipeline.stage_glm import run_session_glm
    from decomp.viz.figures import (
        write_top_level_figures,
        write_per_pair_figures,
    )
    from brainwidemap import bwm_query  # type: ignore[import-not-found]

    one = get_one()

    # ---- Stage 1: Session selection ----------------------------------------------------------
    log.info("Stage 1: session selection")
    pair_specs = parse_pair_specs(args.pairs)
    plan = select_sessions(one, target_n=3, freeze=args.freeze, cache_dir=CACHE_DIR,
                            n_per_region=args.max_pool,
                            pair_specs=pair_specs,
                            pair_min_units=args.pair_min_units,
                            pair_n_max=args.max_pair)
    log.info("  %s", plan.strategy_note)

    # Per-pair eid lists (truncated to --max-pair) and per-ROI pool eid lists
    pair_eids_by_pair = {k: v[: args.max_pair] for k, v in plan.pair_eids_by_pair.items()}
    pool_eids = {roi: eids[: args.max_pool] for roi, eids in plan.pool_eids.items()}
    union_eids = sorted(
        {e for v in pair_eids_by_pair.values() for e in v}
        | {e for v in pool_eids.values() for e in v}
    )
    log.info("  pair sessions per pair:")
    for k, v in pair_eids_by_pair.items():
        log.info("    %s: %d sessions", k, len(v))
    log.info("  pool sessions per ROI: %s",
             {roi: len(eids) for roi, eids in pool_eids.items()})
    log.info("  union: %d unique sessions to load", len(union_eids))

    bwm = bwm_query(one, freeze=args.freeze)
    eid_to_pids = bwm.groupby("eid")["pid"].apply(list).to_dict()

    # Reverse map: which pairs does each pair-eid belong to?
    eid_to_pairs: dict[str, list[tuple[str, str]]] = {}
    for pair_key, eids in pair_eids_by_pair.items():
        a, b = pair_key.split(":")
        for eid in eids:
            eid_to_pairs.setdefault(eid, []).append((a, b))

    # GLM ROIs each session contributes to (always strict ROI_ORDER for the literature
    # reproduction figure)
    session_glm_rois: dict[str, set[str]] = {}
    for roi, eids in pool_eids.items():
        for eid in eids:
            session_glm_rois.setdefault(eid, set()).add(roi)
    # Pair sessions also contribute their pair regions to the GLM pool when those regions
    # are in ROI_ORDER (i.e. only for VISp/CB/MO/CA1 directly; "VIS" wide doesn't add to
    # the strict GLM panel)
    for eid, pairs in eid_to_pairs.items():
        for (a, b) in pairs:
            for r in (a, b):
                if r in plan.rois_used:
                    session_glm_rois.setdefault(eid, set()).add(r)

    glm_rows: list[pd.DataFrame] = []
    cca_rows: list[pd.DataFrame] = []
    svca_per_session: list[tuple[str, dict[str, int], dict[str, object]]] = []

    # All ROIs that may appear in any pair (union across pair_specs) — used to choose
    # which ROIs load_and_bin should bin for. Always include the 4-region ROI_ORDER too.
    all_roi_keys = set(plan.rois_used)
    for (a, b) in pair_specs:
        all_roi_keys.update([a, b])
    rois_to_load = sorted(all_roi_keys)

    for eid in union_eids:
        pids = eid_to_pids.get(eid, [])
        if not pids:
            log.warning("  no pids for %s, skipping", eid)
            continue

        glm_rois = sorted(session_glm_rois.get(eid, set()))
        these_pairs = eid_to_pairs.get(eid, [])

        # Optimization: pool-only sessions whose GLM csv is already cached can skip
        # load_and_bin entirely. Pair sessions always re-load (SVCA/CCA need fresh data).
        glm_cache = CACHE_DIR / f"{eid}_glm_results.csv"
        if (
            not these_pairs
            and glm_cache.exists()
            and args.rerun_from in (None, "glm")
        ):
            log.info("Session %s — GLM cached, no pair work, skipping load.", eid)
            glm_rows.append(pd.read_csv(glm_cache))
            continue

        log.info("Session %s with %d probes (GLM ROIs: %s, pairs: %s)",
                 eid, len(pids), glm_rois, [f"{a}:{b}" for (a, b) in these_pairs])

        try:
            sd, binned = load_and_bin(one, eid, pids, rois=rois_to_load, cache_dir=CACHE_DIR)
        except Exception as e:
            log.warning("  load_and_bin failed for %s (%s): skipping session",
                        eid, type(e).__name__)
            continue
        if not binned.spikes_by_roi:
            log.warning("  no usable spike data for %s, skipping", eid)
            continue

        # GLM stage on all glm_rois present in this session
        if args.rerun_from in (None, "data", "glm") and glm_rois:
            glm_df = run_session_glm(sd, binned, rois=glm_rois, cache_dir=CACHE_DIR,
                                      random_state=args.seed)
            glm_rows.append(glm_df)

        # SVCA + multi-pair CCA + pCCA stages, run only on pair sessions
        if these_pairs and args.rerun_from in (None, "data", "glm", "svca", "cca"):
            svca_results, cca_df = run_pair_session(
                eid, binned, pair_rois=these_pairs, cache_dir=CACHE_DIR,
                n_components=args.n_components, n_surrogates=args.n_surrogates,
                n_splits=5, random_state=args.seed,
            )

            if svca_results:
                # bundle per-pair-session SVCA for fig02 (only the pair regions, not the
                # full ROI_ORDER set, to keep fig02 focused on what feeds into fig03/04)
                pair_regions = sorted({r for (a, b) in these_pairs for r in (a, b)})
                pair_svca = {r: svca_results[r] for r in pair_regions
                             if r in svca_results}
                n_units = {r: int(binned.spikes_by_roi[r].shape[0])
                           for r in pair_regions if r in binned.spikes_by_roi}
                if pair_svca:
                    svca_per_session.append((eid, n_units, pair_svca))

            if not cca_df.empty:
                cca_rows.append(cca_df)

    glm_all = pd.concat(glm_rows, ignore_index=True) if glm_rows else pd.DataFrame()
    cca_all = pd.concat(cca_rows, ignore_index=True) if cca_rows else pd.DataFrame()

    if cca_rows and svca_per_session:
        eid_order = list(cca_all["eid"].unique())
        svca_per_session.sort(key=lambda t: eid_order.index(t[0]) if t[0] in eid_order else 99)

    write_top_level_figures(glm_all, cca_all, OUT_DIR)
    write_per_pair_figures(cca_all, svca_per_session, OUT_DIR)

    # ---- summary.json ----
    expansion_block = {}
    for pair_key, eids in pair_eids_by_pair.items():
        expansion_block[pair_key] = {
            "eids": eids,
            "n_sessions": len(eids),
            "min_units": args.pair_min_units,
        }
    n_cca_by_pair = {}
    if not cca_all.empty:
        for (a, b), grp in cca_all.groupby(["pair_a", "pair_b"]):
            n_cca_by_pair[f"{a}:{b}"] = int(grp["eid"].nunique())

    summary = {
        "strategy": plan.strategy_note,
        "pair_specs": [f"{a}:{b}" for (a, b) in pair_specs],
        "expansion": expansion_block,
        "mvp_v1cb": {
            "eids": MVP_V1CB_EIDS,
            "preserved": True,
            "note": "MVP V1+CB pair sessions; preserved here so the new run can be "
                    "regression-checked against the MVP result.",
        },
        "pool_eids": pool_eids,
        "union_eids": union_eids,
        "rois_used": plan.rois_used,
        "n_glm_rows": int(len(glm_all)),
        "n_cca_rows_by_pair": n_cca_by_pair,
        "seed": args.seed,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Wrote %s", OUT_DIR / "summary.json")
    log.info("Done.")


def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--rerun-from", choices=["data", "glm", "svca", "cca"], default=None)
    p.add_argument("--pairs", default="VIS:CB,VIS:MO,CB:MO",
                   help="comma-separated region pairs for cross-region CCA "
                        "(e.g. 'VIS:CB,VIS:MO'). Use 'VISp:CB' for MVP-equivalent.")
    p.add_argument("--max-pair", type=int, default=10,
                   help="max sessions per pair for cross-region CCA")
    p.add_argument("--max-pool", type=int, default=10,
                   help="max sessions per ROI for the per-region GLM pool")
    p.add_argument("--pair-min-units", type=int, default=5,
                   help="min units per region for a session to count as a pair session")
    p.add_argument("--freeze", default="2023_12_bwm_release")
    p.add_argument("--n-components", type=int, default=8)
    p.add_argument("--n-surrogates", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    main(cli())
