"""Build the COVID dataset: one CSV per state (data/COVID/{STATE}.csv) with the causal
26-week z-score normalisation, read directly by code/cot_generation (--data-mode
covid_TSFM_updated) and code/finetune (--data-mode covid_TSFM_bitong*).

Inputs
  data/raw/covid-hospitalizations-nhsn.csv   NHSN weekly COVID-19 hospital admissions,
                                                 long format (state, location, target_end_date,
                                                 hospitalizations); the 9 dataset states are read
  data/raw/target-hospital-admissions.csv    FluSight target file -> state population
  data/raw/holdout_splits.json        date-based train / buffer / val / test windows
  data/COVID/chronos2/chronos2_nhsn_zscore26w_predictions.csv
                                                 Chronos-2 univariate forecasts ON the z26w series
                                                 (code/chronos2/chronos2_covid.py); keyed
                                                 by (state, h0 = first forecast week)
  data/COVID/text_reports/<State>/<State>_covid_report_week<W>_<YYYY>.txt   (optional)
                                                 weekly state health-department reports (W = MMWR
                                                 week of the origin week; Texas: the Summary-only
                                                 copies in TX_modified/). The release does not ship
                                                 this folder: the reports are the weekly_text column
                                                 of data/COVID/{STATE}.csv, and when the folder is
                                                 absent the builder reuses that column (keyed by
                                                 reference_date) from the existing CSVs.

Numbers
  - Normalisation: CAUSAL trailing 26-week window z-score z26w (the point itself + 25
    prior weeks, ddof=0, sigma=0 -> NaN), identical to code/chronos2/chronos2_covid.py.
  - hist_12w, chronos2_pred_3w, stat_pred_3w, future_3w : 100*z26w, 2 dp (model-visible).
  - future_3w_full  : 100*z26w at full precision (exact ground truth)
  - future_3w_count : the raw weekly admission counts (exact ground truth)
  - denorm_mean_*/denorm_std_* : per-week window mu/sigma. Back-transform
        actual count = (v / 100) * sigma + mu   (every 100*z column)
  - stat_pred_3w = mean of the 2-dp hist_12w, repeated 3x.

Rows / dates
  - reference_date = ORIGIN week (last observed week, = last hist_12w point).
    hist = origin-11w..origin, future = origin+1w..origin+3w, h0_date = origin + 1w.
  - Every COMPLETE origin week with a Chronos-2 row is kept (no text-based filter);
    weekly_text = the origin week's report, "" when the week has no report.

Splits (holdout_splits.json, matched on h0_date)
  - split  : FINAL setting  -> train / gap / test
  - fold_A, fold_B, fold_C : expanding-window folds -> train / gap / val / unused

Output: data/COVID/{STATE}.csv (19 columns; row / split / text-coverage counts go to
$PAPER_OUTPUT_ROOT/covid_export_summary.json). The finetune code derives each fold's
13-column view (fold label column, future_3w as plain z) from these files at run time.

Run (any Python with pandas / numpy; no GPU):
    python build_covid_dataset.py                     # all 9 states
    python build_covid_dataset.py --states TX,VT,NJ
A state without Chronos-2 rows aborts with a hint (run `sbatch code/chronos2/run_chronos2.sbatch covid`
first) unless --skip-missing-cache is given.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import statistics
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # <paper-content>/code
import paper_paths as P   # noqa: E402

BASE_DIR    = P.COVID_DATA_ROOT                                # data/COVID (output)
RAW_LONG    = P.RAW_DATA_DIR / "covid-hospitalizations-nhsn.csv"
POP_PATH    = P.RAW_DATA_DIR / "target-hospital-admissions.csv"
SPLITS_PATH = P.RAW_DATA_DIR / "holdout_splits.json"
CACHE_PATH  = P.COVID_CHRONOS2_DIR / "chronos2_nhsn_zscore26w_predictions.csv"
TEXT_ROOT   = P.COVID_TEXT_DIR
OUT_DIR     = BASE_DIR

STATES_ALL = ["AZ", "IA", "MI", "MS", "NJ", "NY", "TX", "VA", "VT"]
STATE_DIRS = {"AZ": "Arizona", "IA": "Iowa", "MI": "Michigan", "MS": "Mississippi", "NJ": "NewJersey",
              "NY": "NewYork", "TX": "Texas", "VA": "Virginia", "VT": "Vermont"}   # file-name prefix
TEXT_DIRS  = dict(STATE_DIRS, TX="TX_modified")   # where weekly_text is READ from
STATE_LOC  = {"AZ": "04", "IA": "19", "MI": "26", "MS": "28", "NJ": "34", "NY": "36", "TX": "48", "VA": "51", "VT": "50"}

Z_WINDOW, HISTORY_LEN, PRED_LEN = 26, 12, 3
ZDP, SCALE = 2, 100.0
W = pd.Timedelta(weeks=1)

PIPE_COLUMNS = ["state", "reference_date", "hist_12w", "future_3w", "weekly_text",
                "chronos2_pred_3w", "stat_pred_3w", "split", "population",
                "denorm_mean_hist_12w", "denorm_std_hist_12w",
                "denorm_mean_future_3w", "denorm_std_future_3w"]
COLUMNS = PIPE_COLUMNS + ["h0_date", "fold_A", "fold_B", "fold_C", "future_3w_full", "future_3w_count"]


# ── small helpers (copies of cot_common.escape_multiline / build_population_map and an MMWR
#    week function, so this builder does not import the pipeline code) ─────────────────────
def escape_multiline(text: str) -> str:
    """One-physical-line CSV encoding of a multi-line text field (reversed exactly by
    cot_common.unescape_multiline)."""
    return text.replace("\\", "\\\\").replace("\r", "\\r").replace("\n", "\\n")


def mmwr_week(d: dt.date) -> tuple[int, int]:
    """(mmwr_year, mmwr_week) of the Sun-Sat week containing d."""
    sat = d + dt.timedelta(days=(5 - d.weekday()) % 7)
    year = (sat - dt.timedelta(days=3)).year
    jan1 = dt.date(year, 1, 1)
    first_sat = jan1 + dt.timedelta(days=(5 - jan1.weekday()) % 7)
    if first_sat.day < 4:
        first_sat += dt.timedelta(days=7)
    return year, (sat - first_sat).days // 7 + 1


def build_population_map(states, csv_path: Path = POP_PATH) -> dict:
    """FluSight population = median over weeks of value / weekly_rate * 1e5."""
    df = pd.read_csv(csv_path, dtype={"location": str})
    df = df[df["weekly_rate"] > 0]
    pops = {}
    for code in states:
        sub = df[df["location"] == STATE_LOC[code]]
        if sub.empty:
            sys.exit(f"no population rows for {code} in {csv_path}")
        pops[code] = float(statistics.median((sub["value"] / sub["weekly_rate"] * 1e5).tolist()))
    return pops


_BIWEEKLY = re.compile(r"_week(\d+)-(\d+)_(\d{4})\.txt$")


def report_path(state: str, yy: int, ww: int):
    """The origin week's report file. Exact weekly file first; then a biweekly file that
    covers the week (Arizona: ..._week26-27_2025.txt); then New York's 'week0_<Y+1>' naming
    of MMWR week 52/53 of the previous year."""
    sdir = TEXT_ROOT / TEXT_DIRS[state]
    prefix = STATE_DIRS[state]
    p = sdir / f"{prefix}_covid_report_week{ww}_{yy}.txt"
    if p.exists():
        return p
    for q in sdir.glob(f"{prefix}_covid_report_week*-*_{yy}.txt"):
        m = _BIWEEKLY.search(q.name)
        if m and int(m.group(1)) <= ww <= int(m.group(2)):
            return q
    if ww >= 52:
        q = sdir / f"{prefix}_covid_report_week0_{yy + 1}.txt"
        if q.exists():
            return q
    return None


def load_panels(states):
    long = pd.read_csv(RAW_LONG, parse_dates=["target_end_date"])
    long["hospitalizations"] = pd.to_numeric(long["hospitalizations"], errors="coerce")
    raw = (long[long["state"].isin(states)]
           .pivot_table(index="target_end_date", columns="state", values="hospitalizations", aggfunc="first")
           .sort_index())
    missing = [s for s in states if s not in raw.columns]
    if missing:
        sys.exit(f"states missing from {RAW_LONG}: {missing}")
    raw = raw[states]
    z, mu_p, sd_p = raw.copy(), raw.copy(), raw.copy()
    for col in states:
        a = raw[col].dropna()
        mu = a.rolling(Z_WINDOW).mean()
        sd = a.rolling(Z_WINDOW).std(ddof=0).replace(0.0, np.nan)
        z[col], mu_p[col], sd_p[col] = (a - mu) / sd, mu, sd
    return raw, z, mu_p, sd_p


class SplitTable:
    """h0_date -> labels (split, fold_A, fold_B, fold_C)."""

    def __init__(self, path: Path):
        js = json.loads(path.read_text())
        self.final, self.folds = {}, {k: {} for k in js["folds"]}
        rd = js["reference_dates"]
        for d in rd["final"]["train"]:  self.final[d] = "train"
        for d in rd["final"]["buffer"]: self.final[d] = "gap"
        for d in rd["final"]["test"]:   self.final[d] = "test"
        for k, v in rd["folds"].items():
            for d in v["train"]:      self.folds[k][d] = "train"
            for d in v["buffer"]:     self.folds[k][d] = "gap"
            for d in v["validation"]: self.folds[k][d] = "val"
        self.fold_val_end = {k: pd.Timestamp(js["folds"][k]["validation"]["end"]) for k in js["folds"]}

    def labels(self, h0: pd.Timestamp):
        d = h0.strftime("%Y-%m-%d")
        out = {"split": self.final.get(d, "gap")}
        for k in self.folds:
            lab = self.folds[k].get(d)
            if lab is None:
                lab = "unused" if h0 > self.fold_val_end[k] else "gap"
            out[f"fold_{k}"] = lab
        return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--states", default=",".join(STATES_ALL))
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--skip-missing-cache", action="store_true",
                   help="skip states that have no Chronos-2 rows instead of aborting")
    args = p.parse_args()
    states = [s.strip().upper() for s in args.states.split(",") if s.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    raw, z, mu_p, sd_p = load_panels(states)
    pops = build_population_map(states)
    splits = SplitTable(SPLITS_PATH)
    cache_df = pd.read_csv(CACHE_PATH)
    cache = {(r["state"], r["reference_date"]): np.array([r["h0"], r["h1"], r["h2"]], float)
             for _, r in cache_df.iterrows()}
    no_cache = [s for s in states if not any(k[0] == s for k in cache)]
    if no_cache:
        msg = (f"no Chronos-2 rows for {no_cache} in {CACHE_PATH}; run "
               f"`sbatch code/chronos2/run_chronos2.sbatch covid` first")
        if not args.skip_missing_cache:
            sys.exit(msg)
        print("WARNING:", msg, "-> skipped")
        states = [s for s in states if s not in no_cache]

    summary = {}
    for state in states:
        series = z[state].dropna()
        mu_s, sd_s = mu_p[state], sd_p[state]
        have_reports = (TEXT_ROOT / TEXT_DIRS[state]).is_dir()
        prev_text = {}
        if not have_reports:
            prev = args.out_dir / f"{state}.csv"
            if not prev.exists():
                sys.exit(f"neither {TEXT_ROOT / TEXT_DIRS[state]} nor {prev} exists: no report text for {state}")
            pdf = pd.read_csv(prev, keep_default_na=False)
            prev_text = dict(zip(pdf["reference_date"], pdf["weekly_text"]))   # already escaped
            print(f"[{state}] text_reports/ absent -> weekly_text reused from {prev}")
        rows = []
        for i, ts in enumerate(series.index):
            hist = series.iloc[max(0, i - HISTORY_LEN + 1): i + 1]
            fut  = series.iloc[i + 1: i + 1 + PRED_LEN]
            h0   = ts + W
            pred = cache.get((state, h0.strftime("%Y-%m-%d")))
            if len(hist) < HISTORY_LEN or len(fut) < PRED_LEN or pred is None:
                continue
            assert (hist.index[-1] - hist.index[0]) == (HISTORY_LEN - 1) * W and fut.index[0] == h0
            if have_reports:
                yy, ww = mmwr_week(ts.date())
                tpath = report_path(state, yy, ww)
                text = tpath.read_text(encoding="utf-8", errors="replace") if tpath else ""
                text_escaped = escape_multiline(text) if text else ""
            else:
                text_escaped = prev_text.get(ts.date().isoformat(), "")
            hist_r = [round(float(v) * SCALE, ZDP) for v in hist]
            fut_full = [float(v) * SCALE for v in fut]
            row = {
                "state": state,
                "reference_date": ts.date().isoformat(),
                "hist_12w": json.dumps(hist_r),
                "future_3w": json.dumps([round(v, ZDP) for v in fut_full]),
                "weekly_text": text_escaped,
                "chronos2_pred_3w": json.dumps([round(float(v) * SCALE, ZDP) for v in pred]),
                "stat_pred_3w": json.dumps([round(sum(hist_r) / len(hist_r), ZDP)] * PRED_LEN),
                "population": pops[state],
                "denorm_mean_hist_12w": json.dumps([float(mu_s[w]) for w in hist.index]),
                "denorm_std_hist_12w":  json.dumps([float(sd_s[w]) for w in hist.index]),
                "denorm_mean_future_3w": json.dumps([float(mu_s[w]) for w in fut.index]),
                "denorm_std_future_3w":  json.dumps([float(sd_s[w]) for w in fut.index]),
                "h0_date": h0.date().isoformat(),
                "future_3w_full": json.dumps(fut_full),
                "future_3w_count": json.dumps([float(raw[state][w]) for w in fut.index]),
            }
            row.update(splits.labels(h0))
            rows.append(row)
            back = hist_r[-1] / SCALE * sd_s[ts] + mu_s[ts]
            assert abs(back - raw[state][ts]) <= 0.005 / SCALE * sd_s[ts] + 1e-9, (state, ts)
            exact = fut_full[0] / SCALE * sd_s[fut.index[0]] + mu_s[fut.index[0]]
            assert abs(exact - raw[state][fut.index[0]]) < 1e-6, (state, ts)

        out = pd.DataFrame(rows, columns=COLUMNS)
        path = args.out_dir / f"{state}.csv"
        out.to_csv(path, index=False, float_format="%.17g")
        has_text = out["weekly_text"].str.len() > 0
        cov = {lab: f"{int((has_text & (out['split'] == lab)).sum())}/{int((out['split'] == lab).sum())}"
               for lab in ["train", "test"]}
        fold_cov = {f"fold_{k}_val": f"{int((has_text & (out[f'fold_{k}'] == 'val')).sum())}/{int((out[f'fold_{k}'] == 'val').sum())}"
                    for k in "ABC"}
        summary[state] = dict(rows=len(out), text=int(has_text.sum()),
                              first_origin=out["reference_date"].iloc[0], last_origin=out["reference_date"].iloc[-1],
                              split=out["split"].value_counts().to_dict(),
                              **{f"fold_{k}": out[f"fold_{k}"].value_counts().to_dict() for k in "ABC"},
                              text_cov={**cov, **fold_cov})
        print(f"[{state}] {len(out)} rows ({int(has_text.sum())} with text) {out['reference_date'].iloc[0]}.."
              f"{out['reference_date'].iloc[-1]} split={summary[state]['split']} text/split={summary[state]['text_cov']} -> {path}")
    summary_path = P.OUTPUT_ROOT / "covid_export_summary.json"      # not part of data/
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=1))
    print(f"summary -> {summary_path}")


if __name__ == "__main__":
    main()
