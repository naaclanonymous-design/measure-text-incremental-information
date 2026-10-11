"""Chronos-2 baseline for the influenza dataset: multivariate rolling forecasts of weekly
influenza hospital admissions (NHSN) for each target state. These forecasts are the
`chronos2_pred_3w` column of data/Influenza/{STATE}.csv.

How Chronos-2 is called, per target state and per reference date (= first forecast week h0):
  - MULTIVARIATE: the target column is the state's own series; the covariates are every
    OTHER column of the national panel (the 50 states + DC + the national series "USA")
    that has >= COV_MIN_COVERAGE (95 %) non-missing weeks in the context window
    (missing values linearly interpolated), as PAST-only covariates
  - context = every week from CONTEXT_START (2020-08-01, or the state's first report if
    later) up to the cutoff h0 - 7 days (growing context)
  - --max-covariates K keeps only the K covariates with the highest |Pearson corr| with
    the target over the context (the national "USA" target was run with K = 20 to bound
    GPU memory; the state targets with no cap)
  - prediction_length = PRED_LEN (3), quantile 0.5 (median) -> h0, h1, h2, in admissions
  - reference dates: every week in [--start, --end] (default 2022-01-08 .. 2026-04-04) except
    the 3-week train/test buffer (--skip-weeks)
  - resumable: (state, reference_date) pairs already in the cache CSV are skipped

Input : data/raw/influenza_admissions_panel.csv  (week_ending_date x 52 series)
Output: data/Influenza/chronos2/chronos2_influenza_predictions.csv  (state, reference_date, h0, h1, h2)

Model: CHRONOS2_MODEL_PATH (env) or <paper-content>/checkpoints/chronos-2; needs torch + CUDA
and the `chronos` package (chronos-forecasting >= 2.0).

Run on a GPU node:   python chronos2_influenza.py                       (the 8 released states)
                     python chronos2_influenza.py --states USA --max-covariates 20
                     sbatch run_chronos2.sbatch influenza
Schedule only:       python chronos2_influenza.py --dry-run
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR.parent))                     # <paper-content>/code -> paper_paths
import paper_paths as P   # noqa: E402

PANEL_PATH = P.RAW_DATA_DIR / "influenza_admissions_panel.csv"
CACHE_PATH = P.INFLUENZA_DATA_DIR / "chronos2" / "chronos2_influenza_predictions.csv"
CHRONOS2_MODEL_PATH = os.environ.get("CHRONOS2_MODEL_PATH", P.CHRONOS2_MODEL_PATH)

DEFAULT_STATES = "FL,GA,IA,MS,NJ,NY,VA,USA"   # the released data/Influenza/{STATE}.csv
CONTEXT_START = "2020-08-01"
COV_MIN_COVERAGE = 0.95
HISTORY_LEN = 12
PRED_LEN = 3
DEFAULT_START, DEFAULT_END = "2022-01-08", "2026-04-04"
DEFAULT_SKIP = "2025-05-10,2025-05-17,2025-05-24"   # the 3-week train/test buffer (never forecast)
FLUSH_EVERY = 10
DEFAULT_PIPELINE_RELOAD_EVERY = 25


def load_panel() -> pd.DataFrame:
    wide = pd.read_csv(PANEL_PATH, parse_dates=["week_ending_date"]).set_index("week_ending_date").sort_index()
    return wide.apply(pd.to_numeric, errors="coerce")


def reference_dates_for(panel: pd.DataFrame, state: str, start: str, end: str) -> List[pd.Timestamp]:
    actuals = panel[state].dropna()
    return [d for d in actuals.index if pd.Timestamp(start) <= d <= pd.Timestamp(end)]


def load_cache(path: Path) -> Dict[Tuple[str, str], np.ndarray]:
    cache: Dict[Tuple[str, str], np.ndarray] = {}
    if path.exists():
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            cache[(str(r["state"]), str(r["reference_date"]))] = np.array([r["h0"], r["h1"], r["h2"]], float)
        print(f"[flu-mv] loaded {len(cache)} cached forecasts from {path}")
    return cache


def flush_cache(cache: Dict[Tuple[str, str], np.ndarray], path: Path) -> None:
    rows = [{"state": s, "reference_date": r, "h0": v[0], "h1": v[1], "h2": v[2]}
            for (s, r), v in sorted(cache.items())]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def chronos2_for_pairs(panel: pd.DataFrame, pairs: List[Tuple[str, pd.Timestamp]], cache_path: Path,
                       max_covariates: Optional[int], pipeline_reload_every: Optional[int]
                       ) -> Dict[Tuple[str, str], np.ndarray]:
    cache = load_cache(cache_path)
    missing = [(s, r) for (s, r) in pairs if (s, r.strftime("%Y-%m-%d")) not in cache]
    if not missing:
        return cache
    print(f"[flu-mv] need {len(missing)} forecasts (model={CHRONOS2_MODEL_PATH}, "
          f"max_covariates={max_covariates})")
    import gc
    import torch
    from chronos import BaseChronosPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    load = lambda: BaseChronosPipeline.from_pretrained(CHRONOS2_MODEL_PATH, device_map=device)
    pipeline = load()
    floor = pd.Timestamp(CONTEXT_START)
    n_done = 0
    for state, ref_date in missing:
        cutoff = ref_date - pd.Timedelta(days=7)
        own = panel[state].dropna()
        if own.empty:
            print(f"[flu-mv skip] {state} {ref_date.date()}: no data"); continue
        train = panel.loc[max(floor, own.index.min()):cutoff].copy()
        if len(train) < HISTORY_LEN:
            print(f"[flu-mv skip] {state} {ref_date.date()}: only {len(train)} weeks"); continue
        coverage = train.notna().mean()
        keep = coverage[coverage >= COV_MIN_COVERAGE].index.tolist()
        if state not in keep:
            print(f"[flu-mv skip] {state} {ref_date.date()}: target coverage {coverage[state]:.0%}"); continue
        train = train[keep].interpolate("linear").ffill().bfill()
        if max_covariates is not None:
            cov_cols = [c for c in train.columns if c != state]
            if len(cov_cols) > max_covariates:
                corr = train[cov_cols].corrwith(train[state]).abs().fillna(0.0).sort_values(ascending=False)
                train = train[[state] + corr.index[:max_covariates].tolist()]
        ctx = train.reset_index().rename(columns={"week_ending_date": "timestamp"})
        ctx.insert(0, "item_id", state)
        ctx = ctx[["item_id", "timestamp", state] + [c for c in ctx.columns if c not in ("item_id", "timestamp", state)]]
        pred = pipeline.predict_df(ctx, future_df=None, prediction_length=PRED_LEN, quantile_levels=[0.5],
                                   id_column="item_id", timestamp_column="timestamp", target=state)
        pred = pred[(pred["item_id"] == state) & (pred["target_name"] == state)].copy()
        pred["timestamp"] = pd.to_datetime(pred["timestamp"])
        medians = pred.sort_values("timestamp")["0.5"].to_numpy(dtype=float)[:PRED_LEN]
        cache[(state, ref_date.strftime("%Y-%m-%d"))] = medians
        n_done += 1
        if n_done % 25 == 0 or n_done == len(missing):
            print(f"[flu-mv] {n_done}/{len(missing)}  {state} {ref_date.date()}: "
                  f"n_cov={train.shape[1] - 1} median={medians.tolist()}")
        del pred, ctx, train
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if n_done % FLUSH_EVERY == 0:
            flush_cache(cache, cache_path)
        if pipeline_reload_every and n_done % pipeline_reload_every == 0:
            del pipeline; gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            pipeline = load()
    flush_cache(cache, cache_path)
    return cache


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--states", default=DEFAULT_STATES, help="comma list of target columns of the panel")
    p.add_argument("--start", default=DEFAULT_START); p.add_argument("--end", default=DEFAULT_END)
    p.add_argument("--skip-weeks", default=DEFAULT_SKIP, help="comma list of reference dates to leave out")
    p.add_argument("--max-covariates", type=int, default=None,
                   help="keep the K covariates most correlated with the target (default: all; 20 was used for USA)")
    p.add_argument("--pipeline-reload-every", type=int, default=DEFAULT_PIPELINE_RELOAD_EVERY)
    p.add_argument("--cache", type=Path, default=CACHE_PATH)
    p.add_argument("--dry-run", action="store_true", help="print the schedule; do not load the model")
    args = p.parse_args()
    panel = load_panel()
    states = [s.strip().upper() for s in args.states.split(",") if s.strip()]
    missing_cols = [s for s in states if s not in panel.columns]
    if missing_cols:
        sys.exit(f"not in the panel: {missing_cols}")
    pairs: List[Tuple[str, pd.Timestamp]] = []
    for st in states:
        skip = {pd.Timestamp(x) for x in args.skip_weeks.split(",") if x.strip()}
        refs = [r for r in reference_dates_for(panel, st, args.start, args.end) if r not in skip]
        print(f"[flu-mv] {st}: {len(refs)} reference dates ({refs[0].date()} .. {refs[-1].date()})" if refs
              else f"[flu-mv] {st}: no reference dates")
        pairs += [(st, r) for r in refs]
    print(f"[flu-mv] panel {panel.shape[0]} weeks x {panel.shape[1]} series; {len(pairs)} windows")
    if args.dry_run:
        cache = load_cache(args.cache)
        todo = [x for x in pairs if (x[0], x[1].strftime("%Y-%m-%d")) not in cache]
        print(f"[flu-mv] dry run: {len(todo)} still to forecast; cache={args.cache}")
        return
    cache = chronos2_for_pairs(panel, pairs, args.cache, args.max_covariates, args.pipeline_reload_every)
    print(f"[flu-mv] done: {len(cache)} forecasts in {args.cache}")


if __name__ == "__main__":
    main()
