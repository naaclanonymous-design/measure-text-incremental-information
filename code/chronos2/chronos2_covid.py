"""Chronos-2 baseline for the COVID dataset: univariate rolling forecasts of weekly
COVID-19 hospital admissions (NHSN) for each state, on the raw counts and on the
trailing-26-week-window z-score (z26w) series. The z26w forecasts are the
`chronos2_pred_3w` column of data/COVID/{STATE}.csv.

How Chronos-2 is called, per state and per reference date (= first forecast week h0):
  - UNIVARIATE: the context is the target state's own series only
  - context = every week from the context floor (HISTORY_LEN = 12 weeks before the
    first reference date, DEFAULT_START 2022-01-01) up to the cutoff h0 - 7 days,
    i.e. a growing context (--context-weeks N caps it instead)
  - prediction_length = PRED_LEN (3), quantile 0.5 (median) -> h0, h1, h2
  - z26w: z_w = (x_w - mu) / sigma with mu, sigma over the Z_WINDOW (26) weeks ending
    AT w (ddof=0; sigma=0 -> NaN); the model sees and predicts z units
  - resumable: windows already in the cache CSV are skipped, so adding a state to
    STATES and re-running computes only that state

Input : data/raw/covid-hospitalizations-nhsn.csv  (state, location, target_end_date, hospitalizations)
Output: data/COVID/chronos2/chronos2_nhsn_zscore26w_predictions.csv  (state, reference_date, h0, h1, h2)
        data/COVID/chronos2/chronos2_nhsn_raw_predictions.csv

Model: CHRONOS2_MODEL_PATH (env) or <paper-content>/checkpoints/chronos-2; needs torch + CUDA
and the `chronos` package (chronos-forecasting >= 2.0).

Run on a GPU node:   python chronos2_covid.py --series both      (or: sbatch run_chronos2.sbatch covid)
Schedule only:       python chronos2_covid.py --dry-run
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

BASE_DIR  = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR.parent))                     # <paper-content>/code -> paper_paths
import paper_paths as P   # noqa: E402
OUT_DIR   = P.COVID_CHRONOS2_DIR                              # data/COVID/chronos2: the prediction caches only
DATA_PATH = P.RAW_DATA_DIR / "covid-hospitalizations-nhsn.csv"

STATE_COL, DATE_COL, TARGET_COL = "state", "target_end_date", "hospitalizations"
STATES = {"NJ": "New Jersey", "VT": "Vermont", "TX": "Texas", "AZ": "Arizona",
          "MS": "Mississippi", "MI": "Michigan", "NY": "New York", "VA": "Virginia",
          "IA": "Iowa"}
TARGET_STATES = list(STATES)

Z_WINDOW    = 26   # trailing z-score window, INCLUDING the point
HISTORY_LEN = 12   # minimum weeks of history before a reference date
PRED_LEN    = 3    # forecast horizon (weeks)
DEFAULT_CONTEXT_WEEKS: Optional[int] = None   # None = all available history
DEFAULT_START = "2022-01-01"   # first reference_date (week-ending Saturday);
                               # context still reaches back to 2020-08-08

CHRONOS2_MODEL_PATH = os.environ.get("CHRONOS2_MODEL_PATH", P.CHRONOS2_MODEL_PATH)
DEFAULT_PIPELINE_RELOAD_EVERY = 25
FLUSH_EVERY = 10

SERIES_SPECS = {
    "raw":  {"tag": "raw",       "title": "raw counts"},
    "z26w": {"tag": "zscore26w", "title": f"trailing {Z_WINDOW}-week window z-score"},
}



# ────────────────────────────────────────────────────────────────────────────
# Data
# ────────────────────────────────────────────────────────────────────────────
def load_raw_panel() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    wide = (df.pivot_table(index=DATE_COL, columns=STATE_COL,
                           values=TARGET_COL, aggfunc="first")
              .sort_index())
    missing = [s for s in TARGET_STATES if s not in wide.columns]
    if missing:
        sys.exit(f"[nhsn-uni] states missing from panel: {missing}")
    return wide[TARGET_STATES]


def to_z26w(wide: pd.DataFrame) -> pd.DataFrame:
    z = pd.DataFrame(index=wide.index)
    for st in wide.columns:
        a = wide[st].dropna()
        mu = a.rolling(Z_WINDOW).mean()
        sd = a.rolling(Z_WINDOW).std(ddof=0).replace(0.0, np.nan)
        z[st] = (a - mu) / sd
    return z


def load_panel(series: str) -> pd.DataFrame:
    raw = load_raw_panel()
    return raw if series == "raw" else to_z26w(raw)


def context_floor_for(start: Optional[str]) -> Optional[pd.Timestamp]:
    """Context floor = HISTORY_LEN weeks before the first reference date, so
    that window sees exactly HISTORY_LEN weeks of history (its cutoff is the
    week before the reference date)."""
    if start is None:
        return None
    return pd.Timestamp(start) - pd.Timedelta(weeks=HISTORY_LEN)


def reference_dates_for(panel: pd.DataFrame, state: str,
                        start: Optional[str], end: Optional[str]
                        ) -> List[pd.Timestamp]:
    """All weeks in the state's series with >= HISTORY_LEN prior weeks
    (counted from the context floor), optionally restricted to [start, end]."""
    actuals = panel[state].dropna()
    floor = context_floor_for(start)
    if floor is not None:
        actuals = actuals.loc[floor:]
    out = []
    for d in actuals.index:
        cutoff = d - pd.Timedelta(days=7)
        if len(actuals.loc[:cutoff]) < HISTORY_LEN:
            continue
        if start is not None and d < pd.Timestamp(start):
            continue
        if end is not None and d > pd.Timestamp(end):
            continue
        out.append(d)
    return out


def _flush_cache(cache: Dict[Tuple[str, str], np.ndarray], path: Path) -> None:
    rows = [{"state": s, "reference_date": r,
             "h0": v[0], "h1": v[1], "h2": v[2]}
            for (s, r), v in sorted(cache.items())]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def load_cache(path: Path) -> Dict[Tuple[str, str], np.ndarray]:
    cache: Dict[Tuple[str, str], np.ndarray] = {}
    if path.exists():
        df = pd.read_csv(path)
        if {"state", "reference_date", "h0", "h1", "h2"} <= set(df.columns):
            for _, row in df.iterrows():
                cache[(str(row["state"]), str(row["reference_date"]))] = (
                    np.array([row["h0"], row["h1"], row["h2"]], dtype=float))
            print(f"[nhsn-uni] loaded {len(cache)} cached forecasts from {path}")
        else:
            print(f"[nhsn-uni] cache schema mismatch at {path}; regenerating.")
    return cache


# ────────────────────────────────────────────────────────────────────────────
# Chronos-2 forecasting (univariate, full-history context)
# ────────────────────────────────────────────────────────────────────────────
def chronos2_for_pairs(
    panel: pd.DataFrame,
    state_ref_pairs: List[Tuple[str, pd.Timestamp]],
    cache_path: Path,
    context_floor: Optional[pd.Timestamp] = None,
    context_weeks: Optional[int] = None,
    pipeline_reload_every: Optional[int] = None,
) -> Dict[Tuple[str, str], np.ndarray]:
    cache = load_cache(cache_path)
    missing = [(s, r) for (s, r) in state_ref_pairs
               if (s, r.strftime("%Y-%m-%d")) not in cache]
    if not missing:
        return cache

    print(f"[nhsn-uni] need to generate {len(missing)} forecasts "
          f"(model={CHRONOS2_MODEL_PATH}, context_floor="
          f"{context_floor.date() if context_floor is not None else None}, "
          f"context_weeks={context_weeks}, univariate, "
          f"pipeline_reload_every={pipeline_reload_every})")
    import gc
    import torch
    from chronos import BaseChronosPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[nhsn-uni] device: {device}")

    def _load_pipeline():
        return BaseChronosPipeline.from_pretrained(
            CHRONOS2_MODEL_PATH, device_map=device)

    pipeline = _load_pipeline()
    n_done = 0
    for state, ref_date in missing:
        cutoff = ref_date - pd.Timedelta(days=7)
        series = panel[state].dropna()
        train = series.loc[context_floor:cutoff] if context_floor is not None \
            else series.loc[:cutoff]
        if context_weeks is not None and len(train) > context_weeks:
            train = train.iloc[-context_weeks:]
        if len(train) < HISTORY_LEN:
            print(f"[nhsn-uni skip] {state} {ref_date.date()}: only "
                  f"{len(train)} weeks of history.")
            continue
        ctx = train.rename(state).reset_index()
        ctx.columns = ["timestamp", state]
        ctx.insert(0, "item_id", state)
        pred = pipeline.predict_df(
            ctx, future_df=None, prediction_length=PRED_LEN,
            quantile_levels=[0.5], id_column="item_id",
            timestamp_column="timestamp", target=state,
        )
        pred = pred[(pred["item_id"] == state)
                    & (pred["target_name"] == state)].copy()
        pred["timestamp"] = pd.to_datetime(pred["timestamp"])
        pred = pred.sort_values("timestamp")
        medians = pred["0.5"].to_numpy(dtype=float)[:PRED_LEN]
        cache[(state, ref_date.strftime("%Y-%m-%d"))] = medians
        n_done += 1
        if n_done % 10 == 0 or n_done == len(missing):
            print(f"[nhsn-uni] {n_done}/{len(missing)}  {state} "
                  f"{ref_date.date()}: ctx={len(train)}w "
                  f"median={medians.tolist()}")
        del pred, ctx, train
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if n_done % FLUSH_EVERY == 0:
            _flush_cache(cache, cache_path)
        if (pipeline_reload_every is not None and pipeline_reload_every > 0
                and n_done % pipeline_reload_every == 0):
            print(f"[nhsn-uni] reloading pipeline after {n_done} forecasts")
            del pipeline
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            pipeline = _load_pipeline()
    _flush_cache(cache, cache_path)
    return cache


# ────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--series", choices=["raw", "z26w", "both"], default="both")
    p.add_argument("--start", default=DEFAULT_START,
                   help=f"first reference_date to forecast (YYYY-MM-DD); "
                        f"default {DEFAULT_START}, 'none' = earliest possible")
    p.add_argument("--end", default=None,
                   help="last reference_date to forecast (YYYY-MM-DD)")
    p.add_argument("--context-weeks", type=int, default=None,
                   help="cap the context to the last N weeks; default/-1/0 = "
                        "all available history (the influenza setting)")
    p.add_argument("--pipeline-reload-every", type=int,
                   default=DEFAULT_PIPELINE_RELOAD_EVERY)
    p.add_argument("--dry-run", action="store_true",
                   help="print the forecast schedule; do not load the model")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    context_weeks = (None if args.context_weeks in (None, -1, 0)
                     else args.context_weeks)
    if args.start is not None and args.start.lower() == "none":
        args.start = None
    series_list = ["raw", "z26w"] if args.series == "both" else [args.series]

    for series in series_list:
        spec = SERIES_SPECS[series]
        panel = load_panel(series)
        cache_path = OUT_DIR / f"chronos2_nhsn_{spec['tag']}_predictions.csv"
        print(f"\n===== series: {series} ({spec['title']}) =====")

        floor = context_floor_for(args.start)
        print(f"[nhsn-uni] context floor: "
              f"{floor.date() if floor is not None else None} "
              f"(first window = {HISTORY_LEN} weeks, then growing)")
        pairs: List[Tuple[str, pd.Timestamp]] = []
        for st in TARGET_STATES:
            refs = reference_dates_for(panel, st, args.start, args.end)
            s = panel[st].dropna()
            print(f"[nhsn-uni] {st}: {len(s)} weeks "
                  f"({s.index.min().date()} .. {s.index.max().date()}), "
                  f"{len(refs)} reference dates"
                  + (f" ({refs[0].date()} .. {refs[-1].date()})" if refs else ""))
            pairs += [(st, r) for r in refs]

        if args.dry_run:
            cache = load_cache(cache_path)
            todo = [(s, r) for (s, r) in pairs
                    if (s, r.strftime("%Y-%m-%d")) not in cache]
            print(f"[nhsn-uni] dry run: {len(pairs)} windows total, "
                  f"{len(todo)} still to forecast; cache={cache_path.name}")
            continue
        else:
            cache = chronos2_for_pairs(panel, pairs, cache_path,
                                       context_floor=floor,
                                       context_weeks=context_weeks,
                                       pipeline_reload_every=args.pipeline_reload_every)
        print(f"[nhsn-uni] {series}: {len(cache)} forecasts in {cache_path}")


if __name__ == "__main__":
    main()
