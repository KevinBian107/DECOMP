"""Re-run pCCA across all cached pair-sessions under a richer Z, and emit the
cross-pair comparison figure.

Usage:
    PYTHONPATH=src python scripts/run_richz.py

Reads cached SVCA scores + binned covariates per session — no IBL re-download, no SVCA
recompute. Writes:
    data/cache/{eid}_cca_richz.csv            per session
    outputs/richz_comparison.png              cross-pair survival under Z_min vs Z_rich
"""
from __future__ import annotations

from pathlib import Path

from decomp.cca.richz import rerun_all
from decomp.viz.figures import fig_richz_comparison

CACHE = Path("data/cache")
OUT = Path("outputs")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # n_surrogates=0 — the headline figure uses survival ratios only; per-component nulls
    # for the richer-Z panel are out of scope for this stress test.
    df = rerun_all(CACHE, n_surrogates=0)
    if df.empty:
        print("No cached sessions found.")
        return
    print(f"\nDone. {df['eid'].nunique()} sessions, "
          f"{df[['pair_a', 'pair_b']].drop_duplicates().shape[0]} pair types.")
    fig_richz_comparison(df, OUT)
    print(f"Wrote {OUT / 'richz_comparison.png'}")


if __name__ == "__main__":
    main()
