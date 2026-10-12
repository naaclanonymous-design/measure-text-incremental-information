"""Shared helpers for the Semantic-Residual-Operator two-step CoT pipeline.

This module is written FROM SCRATCH for the CoT_generate_updated pipeline and
imports nothing from the existing A3_CoT_* scripts. It bundles everything the four
pipeline stages need:

    step1_generate.py  →  step1_filter.py  →  step2_generate.py  →  step2_filter.py

Contents
--------
* config / paths / model names / windows                              (CONSTANTS)
* raw NHSN load, Chronos-2 cache, leak-safe `last_observed` windowing  (data layer)
* per-10,000 population normalisation                                  (to_rates)
* MMWR FluView text store + wrong-week guard + relative-distance block (FluView)
* date scrubbing (redact calendar dates / week numbers / years)        (scrub_dates)
* recent-residual (TEFL) block + momentum block builders               (input blocks)
* residual targets / direction labels / utility                       (offline scoring)
* one-directional |r| quantile bins + cold-start lower-edge m_init     (bins / grounding)
* self-consistency confidence α₁                                       (confidence)
* verbatim quote extraction + faithfulness check                       (faithfulness)
* Gemini client, JSONL resume I/O, optional scratch symlink            (infra)

The 6-stage framework is summarised in code/README.md.
"""
from __future__ import annotations

import json
import logging
import os
import re
import statistics
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # <paper-content>/code -> paper_paths
import paper_paths as P   # noqa: E402
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("cot_common")

# ══════════════════════════════════════════════════════════════════════════════
# PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
DATA_PATH = P.RAW_SOURCES_DIR / "NHSN_weekly_hospital_respiratory_data.csv"   # raw NHSN export (provenance only, not shipped)
DATE_COL   = "Week Ending Date"
TARGET_COL = "Total Influenza Admissions"
GEO_COL    = "Geographic aggregation"

CACHE_PATH_USA    = P.RAW_SOURCES_DIR / "chronos2_USA_cache.csv"      # provenance only (influenza CSVs embed the forecasts)
CACHE_PATH_STATES = P.RAW_SOURCES_DIR / "chronos2_cache.csv"          # provenance only

CDC_DIR   = P.RAW_SOURCES_DIR / "CDC_website_data_processed"         # provenance only (texts are inside the CSVs)
CRED_PATH = P.GEMINI_CRED_PATH          # {"api_key": "..."}; or export GEMINI_API_KEY
TARGET_HOSP_CSV = P.TARGET_HOSP_CSV

# Output: physically on /scratch, symlinked back into /home (per-state).
OUTPUT_DIR          = P.COT_OUTPUT_DIR
SCRATCH_OUTPUT_ROOT = P.COT_SCRATCH_ROOT     # == OUTPUT_DIR unless PAPER_SCRATCH_ROOT is set
PROMPT_DIR          = P.PROMPT_DIR_INFLUENZA   # default; covid modes resolve to P.PROMPT_DIR_COVID (prompt_dir_for)

HISTORY_LEN = 12
PRED_LEN    = 3            # horizon H
LOOKBACK_DEFAULT = 5       # FluView report weeks (how many recent reports are shown + gated)
RESID_LOOKBACK_DEFAULT = 3  # recent-residual weeks (the Chronos-2 / GT-change error window)

# Leak-safe date windows (identical convention to the A2/A3 pipelines).
TRAIN_END            = "2025-05-03"
TEST_START           = "2025-05-31"
CHRONOS2_TRAIN_START = "2020-08-01"
VAL_START            = "2024-06-01"
VAL_END              = "2025-05-03"
GAP_WEEKS            = 3

# Per-10,000-resident rate conversion / precision. EVERYTHING the model sees and the whole
# pipeline reasons in (history, baseline, residuals, bins, utility) is rounded to
# RATE_ROUND_DECIMALS dp AFTER the per-10,000 conversion. The full-precision history and
# ground truth are still recorded (x_hist_full / x_gt_full / x_hat_full) but never used.
RATE_PER          = 10000.0
RATE_ROUND_DECIMALS = 2    # the working precision (model-visible; drives all computation)
RATE_DECIMALS = 2          # display precision for rate blocks (match the working precision)
RESID_DECIMALS = 2         # residual display + rounding precision (rate units)

# No-change band ε (per 10,000). |r| ≤ ε ⇒ direction = no-change (§2.2). Data-relative
# noise floor; ≈0.01 for the flu proportion corpus. Configurable per run.
NOCHANGE_BAND = 0.01

# Bins: one-directional |r| quantile size classes (§2). M=4 by default.
BIN_NAMES_DEFAULT = ["small", "medium", "large", "very-large"]

# Self-consistency / abstention.
K_DEFAULT   = 5
WORKERS_DEFAULT = 8        # concurrent Gemini calls (network-bound; raise/lower per rate limits)
TAU_DEFAULT = 0.6          # ≥3/5 of the k CoTs must agree on the direction (§4)
UTILITY_MARGIN = 0.0       # accept a Step-2 size if U > margin (§6)

# Gemini models. Generation + the "complex" grounding judge use Gemini-3-pro; the
# "easy" verbatim-quote faithfulness check uses Gemini-2.5-flash.
PRO_MODEL   = "gemini-3.1-pro-preview"   # gemini-3-pro-preview returns 404 (delisted); 3.1 is the live Gemini-3 pro
FLASH_MODEL = "gemini-2.5-flash"

# ── Model PROVIDER (--model gemini|GPT) ───────────────────────────────────────
# "gemini" (default): the current setting — PRO_MODEL for generation/momentum/complex
# judges, FLASH_MODEL for the easy faithfulness judge. "gpt": the same task split on
# OpenAI models — the most advanced GPT model stands in for every Gemini-3.1-pro task
# and a lighter GPT model stands in for every Gemini-2.5-flash task.
PROVIDER_DEFAULT = "gemini"
PROVIDERS = ("gemini", "gpt")
GPT_PRO_MODEL   = "gpt-5.5"       # ↔ gemini-3.1-pro-preview tasks (most advanced GPT)
GPT_FLASH_MODEL = "gpt-5.4-mini"  # ↔ gemini-2.5-flash tasks (slightly weaker GPT)
OPENAI_CRED_PATH = P.OPENAI_CRED_PATH   # {"api_key": "..."}; or export OPENAI_API_KEY


def normalize_provider(name: Optional[str]) -> str:
    p = (name or PROVIDER_DEFAULT).strip().lower()
    if p not in PROVIDERS:
        raise ValueError(f"--model must be one of {PROVIDERS} (case-insensitive), got {name!r}")
    return p


def provider_pro_model(provider: str) -> str:
    return PRO_MODEL if normalize_provider(provider) == "gemini" else GPT_PRO_MODEL


def provider_flash_model(provider: str) -> str:
    return FLASH_MODEL if normalize_provider(provider) == "gemini" else GPT_FLASH_MODEL


def provider_cred_path(provider: str) -> Path:
    return CRED_PATH if normalize_provider(provider) == "gemini" else OPENAI_CRED_PATH
GEN_TEMPERATURE = 0.9
GEN_TOP_P       = 0.95
GEN_MAX_TOKENS  = 8192
JUDGE_MAX_TOKENS = 1024

STATES_DEFAULT = "GA,NJ,NY"
# CoT is generated for the TRAIN split only by default (finetuning consumes train CoTs;
# val/test evaluation needs only the input blocks + ground truth, which are built
# offline). Pass --splits train,val,test to also generate reference CoTs there.
SPLITS_DEFAULT = "train"

# ── Generation MODE ───────────────────────────────────────────────────────────
# chronos2 : the original framework — 3-week horizon [ref+1,ref+2,ref+3], Chronos-2
#            baseline, 1-step recent residuals; one generation run.
# flat     : predict ONLY the 1-week-ahead (ref+1) and 3-week-ahead (ref+3) weeks (the
#            2-week-ahead ref+2 is skipped); baseline = the FLAT value actual(ref) (the
#            current level) for both; Step-1 direction is generated in TWO single-week runs
#            (one per predicted week) with HORIZON-MATCHED recent residuals defined as the
#            ground-truth CHANGE — GT(w)-GT(w-1) for the ref+1 run, GT(w)-GT(w-3) for the
#            ref+3 run — then the two runs are combined into one 2-week direction result on
#            which Filter-1 / Step-2 / Filter-2 run unchanged.
# chronos2_full : chronos2 windowing/baseline (3-week horizon, Chronos-2 baseline, two-horizon
#            recent residuals) but NO per-10,000 transform — each state's ACTUAL weekly
#            hospitalization counts, kept at FULL precision everywhere (no rounded working
#            copy, no x_*_full duplicates in output files; prompts merely DISPLAY whole
#            numbers). The no-change band and the size bins are RELATIVE (proportional to
#            the baseline): no-change ⇔ |gt − baseline| / baseline ≤ NOCHANGE_REL_BAND
#            (10%); each size class is a % change of the baseline, with QUANTILE bins
#            over the moving change-proportions pooled across ALL states of the run,
#            so each class holds ~the same share of moving cases (~25% each for 4).
MODE_DEFAULT = "chronos2"
MODES = ("chronos2", "flat", "chronos2_full")
FLAT_PRED_OFFSETS    = (1, 3)   # weeks ahead of the origin that flat mode predicts (ref+1, ref+3)
FLAT_RESID_STEPS     = (1, 3)   # GT-change horizon for each predicted week's recent residuals
FLAT_BASELINE_OFFSET = 0        # flat baseline = actual(ref - FLAT_BASELINE_OFFSET) = actual(ref)

# chronos2_full working precision + relative no-change band.
COUNT_DECIMALS    = 0      # counts are shown to the model as whole numbers
NOCHANGE_REL_BAND = 0.10   # no-change ⇔ relative change within ±10% of the baseline
REL_RESID_DECIMALS = 4     # relative-residual working precision (fraction of baseline)


def mode_is_count(mode: str) -> bool:
    """True for modes that reason in raw hospitalization counts (no rate transform)."""
    return mode == "chronos2_full"


def mode_value_decimals(mode: str, data_mode: str = "influenza") -> int:
    """Display/working decimals for model-visible values (0 for admission counts,
    2 for rates, finance prices and the covid 100x-z index)."""
    if mode_is_count(mode):
        if data_mode in FINANCE_DATA_MODES:
            return FINANCE_DECIMALS
        if data_mode in COVID_DATA_MODES:
            return COVID_DECIMALS
        return COUNT_DECIMALS
    return RATE_DECIMALS


def mode_units(mode: str, data_mode: str = "influenza") -> str:
    if data_mode in FINANCE_DATA_MODES:
        # normalized prices: 100 = the ticker's train-split mean weekly open
        # (per-ticker `price_scale` column in the organized CSVs inverts it)
        return "price_index"
    if data_mode in COVID_DATA_MODES:
        # 100 x smoothed per-year z-score of the weekly COVID ED-visit proportion, +300
        # index shift (see datasets/SHIFT300_MANIFEST.md; denorm_mean/std_future_3w in the
        # dataset CSVs invert it via ((v-300)/100)*sigma+mu)
        return "covid_z100_index"
    return "counts" if mode_is_count(mode) else "per_10000"


def default_nochange_band(mode: str, given: Optional[float] = None,
                          data_mode: str = "influenza") -> float:
    """The mode's no-change band ε: RELATIVE (fraction of baseline, 0.10) for count modes,
    ABSOLUTE index points (20) for covid_TSFM, absolute rate units (0.01 per 10,000)
    otherwise. A non-None CLI override wins."""
    if given is not None:
        return float(given)
    if data_mode in COVID_DATA_MODES:
        return COVID_ABS_BAND
    return NOCHANGE_REL_BAND if mode_is_count(mode) else NOCHANGE_BAND


def fmt_nochange_band(mode: str, band: float, data_mode: str = "influenza") -> str:
    """Band as rendered in prompts: '10%' for relative modes, '20 index points' for
    covid_TSFM, '0.01' for rates."""
    if data_mode in COVID_DATA_MODES:
        return f"{band:g} index points"
    return f"{band * 100:g}%" if mode_is_count(mode) else f"{band:.{RATE_DECIMALS}f}"

# Per-state FluView text: 2-letter code -> (subdir under CDC_DIR, filename prefix).
STATE_FLUVIEW: Dict[str, Tuple[str, str]] = {
    "USA": ("USA",         "fluview_week"),
    "FL":  ("Florida",     "FL_flu_report_week"),
    "GA":  ("Georgia",     "flu_report_week"),
    "IA":  ("Iowa",        "flu_report_week"),
    "MS":  ("Mississippi", "MS_flu_report_week"),
    "NJ":  ("New_Jersey",  "flu_report_week"),
    "NY":  ("New_York",    "flu_report_week"),
    "VA":  ("Virginia",    "week"),
}

# 2-letter geo -> FluSight location code (for population lookup).
STATE_LOC: Dict[str, str] = {
    "US": "US", "USA": "US",
    "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06", "CO": "08",
    "CT": "09", "DE": "10", "DC": "11", "FL": "12", "GA": "13", "HI": "15",
    "ID": "16", "IL": "17", "IN": "18", "IA": "19", "KS": "20", "KY": "21",
    "LA": "22", "ME": "23", "MD": "24", "MA": "25", "MI": "26", "MN": "27",
    "MS": "28", "MO": "29", "MT": "30", "NE": "31", "NV": "32", "NH": "33",
    "NJ": "34", "NM": "35", "NY": "36", "NC": "37", "ND": "38", "OH": "39",
    "OK": "40", "OR": "41", "PA": "42", "RI": "44", "SC": "45", "SD": "46",
    "TN": "47", "TX": "48", "UT": "49", "VT": "50", "VA": "51", "WA": "53",
    "WV": "54", "WI": "55", "WY": "56",
}

FLU_SEASON_START_WEEK = 40
FLU_SEASON_END_WEEK   = 20

# FluView report files to TREAT AS "no report" (data-quality exclusions). The USA
# national reports below embed a mis-scraped activity-map table from a LATER report
# (future weeks). Date-scrubbing hides the future *date* but not the future activity
# *values*, so these would forward-leak if USA were used. Keyed by (state code, filename).
FLUVIEW_EXCLUDE = {
    ("USA", "fluview_week08_2025.txt"),   # narrative wk08 but table = wk14–15 (Apr 2025)
    ("USA", "fluview_week17_2022.txt"),   # content date +13d after the filename week
    ("USA", "fluview_week21_2023.txt"),   # content date +139d (Oct 2023 table)
    ("USA", "fluview_week22_2023.txt"),   # content date +132d
    ("USA", "fluview_week23_2022.txt"),   # content date +125d
}


# ── DATA MODE (input-data source; the generation METHOD is identical in both) ─
# influenza : per-state CSVs exported from the NHSN + Chronos-2 cache + FluView text
#             sources (export_influenza_csv.py) — one row per week: value, chronos2_h0-2,
#             weekly_text, population. Splits/windowing/gating are byte-identical to the
#             original NHSN path.
# finance_stat : per-ticker CSVs (organized weekly_with_news) — one row per origin week:
#             hist_open_12w / future_open_3w (JSON lists), news text, split. The
#             BASELINE being corrected is the STATISTICAL one: stat_pred_3w (the
#             12-week-mean carried flat; falls back to chronos2_pred_3w only for old
#             CSV snapshots that predate the stat column, whose chronos2_pred_3w held
#             those same mean values). Requires --mode chronos2_full (raw values,
#             relative band); prompts are the unmodified chronos2_full set.
# finance_TSFM : same organized CSVs and method as finance_stat, but the BASELINE
#             being corrected is the REAL Chronos-2 forecast (chronos2_pred_3w,
#             strictly required — no fallback), and ONLY the far horizon is
#             predicted: the single target period is origin+3 (t+3); t+1 and t+2
#             are dropped from x_hat / x_gt / target_dates right after sample
#             building (subset_samples_to_t3). Prompts come from the
#             chronos2_full_t3 prompt dir (prompt_mode()).
# covid_TSFM : per-state COVID CSVs (legacy ED-visit index, not part of this release) —
#             same windowed layout as the influenza CSVs; states GA,NJ,NY. Values are
#             the NORMALIZED weekly COVID ED-visit proportion: 100 x the smoothed
#             per-year z-score + 300 index shift (SHIFT300_MANIFEST.md; keeps the
#             relative band/bins away from zero). future_3w is stored as z + 3.0 and
#             rescaled x100 by the loader so all working values share one unit.
#             BASELINE = the real Chronos-2 forecast (chronos2_pred_3w). Same
#             generation/finetuning METHOD as influenza, but requires --mode
#             chronos2_full (values used exactly as preserved in the CSVs — no
#             per-10,000 transform), ONLY origin+2 and origin+3 predicted
#             (COVID_PRED_OFFSETS via subset_samples_to_offsets; origin+1 is inside
#             the band ~2/3 of the time), and — unlike the other chronos2_full modes —
#             an ABSOLUTE no-change band of ±20 index
#             points (COVID_ABS_BAND; band_is_relative() is False): residual blocks
#             are shown in index points, direction labels use the absolute band, and
#             size bins are |residual| quantile classes in points (kind =
#             absolute_index). Prompts come from the chronos2_full_abs prompt dir
#             (prompt_mode()). Split windows come from the CSVs' split column
#             (3-week gap buffers).
DATA_MODES = ("influenza", "finance_stat", "finance_TSFM", "covid_TSFM", "covid_TSFM_updated")
FINANCE_DATA_MODES = ("finance_stat", "finance_TSFM")   # both read the organized ticker CSVs
TSFM_PRED_OFFSETS = (3,)      # finance_TSFM predicts only origin+3
DATA_MODE_DEFAULT = "influenza"
INFLUENZA_DATA_DIR = P.INFLUENZA_DATA_DIR
FINANCE_DATA_DIR = P.DATA_ROOT / "Finance" / "organized"      # not part of the paper release
FINANCE_TICKERS_DEFAULT = "AAPL,GOOGL,NVDA"
FINANCE_DECIMALS = 2       # prices are shown/rounded to cents (vs whole admission counts)
COVID_DATA_DIR = P.DATA_ROOT / "COVID_legacy"                 # the old ED-visit index data (not released)
# covid_TSFM_updated: NHSN hospitalization z26w CSVs (export_covid_nhsn_z26w_csv.py) —
# same layout, 100*z with NO +300 shift, states GA/FL/MS, causal 26-wk-window z.
COVID_UPDATED_DATA_DIR = P.COVID_DATA_ROOT        # data/COVID/{STATE}.csv (19 columns; `split` = final train/gap/test)
COVID_DATA_MODES = ("covid_TSFM", "covid_TSFM_updated")
COVID_DECIMALS = 2         # covid values are 100 x z-score, stored/shown to 2 decimals
# covid_TSFM no-change band is ABSOLUTE, in index points (100 points = one per-year sigma):
# the relative ±10% band collapses near the (shifted) series level and admits level-
# dependent tolerance; ±20 points sits just above the index's median weekly jitter (~15).
# Bins for this mode are |residual| quantile classes in index points (kind=absolute_index).
COVID_ABS_BAND = 20.0
COVID_BAND_DECIMALS = 1    # bin-edge display precision for index-point classes
BINS_KIND_ABS_INDEX = "absolute_index"


def band_is_relative(mode: str, data_mode: str = "influenza") -> bool:
    """True when the no-change band / residual blocks / size bins are RELATIVE (fraction of
    the baseline): chronos2_full for influenza + finance. covid_TSFM runs chronos2_full
    but uses an ABSOLUTE index-point band; the rate modes use an absolute rate band."""
    return mode_is_count(mode) and data_mode not in COVID_DATA_MODES


def default_states(data_mode: str) -> str:
    if data_mode in FINANCE_DATA_MODES:
        return FINANCE_TICKERS_DEFAULT
    if data_mode == "covid_TSFM_updated":
        return "GA,FL,MS"
    return STATES_DEFAULT


def _iso_minus_weeks(iso: str, weeks: int) -> str:
    return (datetime.strptime(iso, "%Y-%m-%d").date() - timedelta(weeks=weeks)).isoformat()


GAP_START = _iso_minus_weeks(VAL_START, GAP_WEEKS)


# ══════════════════════════════════════════════════════════════════════════════
# Infra: api key, JSONL, resume, scratch symlink, template fill
# ══════════════════════════════════════════════════════════════════════════════
def load_api_key(path: Path = CRED_PATH) -> str:
    """API key from the environment (GEMINI_API_KEY / OPENAI_API_KEY, chosen by the
    credential file's name) or else from the JSON file {"api_key": "..."}."""
    env_name = "OPENAI_API_KEY" if "openai" in Path(path).name.lower() else "GEMINI_API_KEY"
    if os.environ.get(env_name):
        return os.environ[env_name]
    if not Path(path).exists():
        raise FileNotFoundError(f"LLM credentials not found at {path} and ${env_name} is not set")
    obj = json.loads(Path(path).read_text())
    for key_name in ("api_key", "aki_key", "GEMINI_API_KEY", "OPENAI_API_KEY"):
        if obj.get(key_name):
            return str(obj[key_name])
    raise RuntimeError(f"No api_key field in {path}: keys={list(obj.keys())}")


def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def iter_jsonl(path: Path):
    path = Path(path)
    if not path.exists():
        return
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_done_pairs(path: Path) -> Set[Tuple[str, int]]:
    return {(r["id"], int(r["candidate_index"])) for r in iter_jsonl(path)
            if "id" in r and "candidate_index" in r}


def append_jsonl(path: Path, row: Dict) -> None:
    with Path(path).open("a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()


def run_parallel(items: Sequence, work_fn: Callable, workers: int,
                 on_done: Callable, log_every: int = 10, label: str = "work",
                 state_of: Optional[Callable] = None) -> None:
    """Map `work_fn` over `items` with up to `workers` threads (for network-bound Gemini
    calls), passing each result to `on_done` ON THE MAIN THREAD — so `on_done` is the sole
    writer and JSONL appends stay safe/ordered. work_fn must NOT write files. workers<=1
    runs sequentially. Items already filtered for resume are passed in by the caller.
    If `state_of(result)` is given, each progress line also shows cumulative per-state
    completion (so a single log makes each state's progress visible)."""
    total = len(items)
    if total == 0:
        return
    by_state: Dict[str, int] = {}

    def _progress(i):
        extra = ""
        if by_state:
            extra = " | by-state " + " ".join(f"{k}={v}" for k, v in sorted(by_state.items()))
        logger.info("[%s %d/%d] %d-way%s", label, i, total, workers, extra)

    def _handle(res):
        on_done(res)
        if state_of is not None:
            try:
                st = state_of(res)
            except Exception:
                st = None
            if st:
                by_state[st] = by_state.get(st, 0) + 1

    if workers <= 1:
        for i, it in enumerate(items, 1):
            try:
                _handle(work_fn(it))
            except Exception as e:
                logger.error("%s item failed: %s", label, e)
            if i % log_every == 0 or i == total:
                _progress(i)
        return
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(work_fn, it) for it in items]
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                _handle(fut.result())
            except Exception as e:
                logger.error("%s item failed: %s", label, e)
            if i % log_every == 0 or i == total:
                _progress(i)


def run_dir_name(states: Sequence[str]) -> str:
    """The per-RUN output folder name = the requested states joined, in order, e.g.
    ['GA','NJ','NY'] -> 'GA_NJ_NY'; ['USA'] -> 'USA'. All states in one run share this
    single directory (rows are keyed by id, which embeds the state, so they never clash)."""
    return "_".join(s.strip().upper() for s in states if s.strip())


def _scratch_dir(name: str, home_output_dir: Path = OUTPUT_DIR,
                 scratch_root: Path = SCRATCH_OUTPUT_ROOT) -> Path:
    """Ensure home_output_dir/<name> is a symlink into scratch_root/<name> so the CoT
    corpus physically lives on /scratch while staying visible from /home. `name` may
    contain a sub-path (e.g. 'chronos2/GA_NJ_NY')."""
    scratch = Path(scratch_root) / name
    scratch.mkdir(parents=True, exist_ok=True)
    link = Path(home_output_dir) / name
    link.parent.mkdir(parents=True, exist_ok=True)
    if Path(scratch_root).resolve() == Path(home_output_dir).resolve():
        return link                       # no separate scratch: the corpus lives in the output dir
    if link.is_symlink():
        if link.resolve() != scratch.resolve():
            logger.warning("%s links to %s (expected %s); leaving as-is",
                           link, link.resolve(), scratch)
    elif link.exists():
        logger.warning("%s exists and is NOT a symlink — writing through it", link)
    else:
        link.symlink_to(scratch, target_is_directory=True)
        logger.info("symlinked %s -> %s", link, scratch)
    return link


def scratch_run_dir(states: Sequence[str], mode: str = MODE_DEFAULT,
                    home_output_dir: Path = OUTPUT_DIR,
                    scratch_root: Path = SCRATCH_OUTPUT_ROOT,
                    provider: str = PROVIDER_DEFAULT,
                    data_mode: str = DATA_MODE_DEFAULT) -> Path:
    """The single output dir for a run: <output>/<data_mode>/<mode>/<states>-<provider>/
    (on /scratch, symlinked into /home). The provider suffix (gemini/gpt) keeps corpora
    generated by different model families from mixing in one directory; the data-mode
    level (influenza/finance_stat) keeps the two corpora apart."""
    name = f"{run_dir_name(states)}-{normalize_provider(provider)}"
    return _scratch_dir(f"{data_mode}/{mode}/{name}", home_output_dir, scratch_root)


def fill_template(template: str, mapping: Dict[str, object]) -> str:
    """Robust placeholder fill: replace every '{key}' token. Unlike str.format this
    never raises on stray literal braces or missing keys."""
    out = template
    for k, v in mapping.items():
        out = out.replace("{" + k + "}", str(v))
    return out


def load_prompt(name: str, prompt_dir: Path = PROMPT_DIR, mode: str = MODE_DEFAULT) -> str:
    """Load a prompt from <prompt_dir>/<name>. The release ships one FLAT directory per
    prompt set (prompts/influenza, prompts/covid); a legacy
    <prompt_dir>/<mode>/<name> layout is still honoured when that subdirectory exists."""
    sub = Path(prompt_dir) / mode / name
    return (sub if sub.exists() else Path(prompt_dir) / name).read_text()


def prompt_dir_for(data_mode: str) -> Path:
    """prompts/covid for the covid modes, prompts/influenza otherwise (see paper_paths)."""
    return P.prompt_dir_for(data_mode)


def prompt_mode(mode: str, data_mode: str = "influenza") -> str:
    """The prompt SUBDIR for a run. finance_TSFM uses the far-horizon-only prompt set
    <mode>_t3 (single predicted period = origin+3); covid_TSFM uses <mode>_abs (ABSOLUTE
    index-point no-change band / residuals / size classes); every other data mode uses
    <mode>."""
    if data_mode == "finance_TSFM":
        return f"{mode}_t3"
    if data_mode == "covid_TSFM_updated":
        return f"{mode}_abs_t3"          # single target (origin+3), absolute band
    if data_mode == "covid_TSFM":
        return f"{mode}_abs"
    return mode


def subset_samples_to_t3(samples: Sequence[Dict]) -> None:
    """finance_TSFM: in place, keep ONLY the far horizon (origin+3) in every sample —
    x_hat / x_gt / target_dates collapse to length 1 and pred_offsets records the true
    weeks-ahead offset [3] for block rendering (fmt_baseline_block). Must run BEFORE any
    rounding/scaling step so the *_full copies are consistent."""
    for s in samples:
        if len(s["x_hat"]) < 3:
            raise ValueError(f"sample {s.get('id')}: expected 3-horizon baseline, "
                             f"got {len(s['x_hat'])}")
        s["x_hat"] = [s["x_hat"][2]]
        s["x_gt"] = [s["x_gt"][2]]
        if s.get("target_dates") is not None:
            s["target_dates"] = [s["target_dates"][2]]
        s["pred_offsets"] = list(TSFM_PRED_OFFSETS)


# covid_TSFM predicts ONLY origin+2 and origin+3: at origin+1 the Chronos-2 baseline is
# inside the ±20-point band ~2/3 of the time (move rate 33%, vs 54% / 67% at +2 / +3), so
# the near week carries little correction signal and would dominate the no-change prior.
COVID_PRED_OFFSETS = (2, 3)
# covid_TSFM_updated predicts ONLY origin+3 (single far-horizon target, like finance_TSFM);
# the origin+1/+2 residuals stay observable evidence in the recent-error blocks.
COVID_UPDATED_PRED_OFFSETS = (3,)


def covid_pred_offsets(data_mode: str) -> tuple:
    return (COVID_UPDATED_PRED_OFFSETS if data_mode == "covid_TSFM_updated"
            else COVID_PRED_OFFSETS)


def subset_samples_to_offsets(samples: Sequence[Dict], offsets: Sequence[int]) -> None:
    """In place, keep only the horizons `offsets` (weeks ahead of the origin, 1-based) in
    every sample — x_hat / x_gt / target_dates are sliced to those positions and
    pred_offsets records the true offsets for block rendering (fmt_baseline_block with
    label_by_offset). Generalizes subset_samples_to_t3. Must run BEFORE any
    rounding/scaling step so the *_full copies are consistent."""
    idx = [int(o) - 1 for o in offsets]
    for s in samples:
        if len(s["x_hat"]) < max(idx) + 1:
            raise ValueError(f"sample {s.get('id')}: expected {max(idx) + 1}-horizon "
                             f"baseline, got {len(s['x_hat'])}")
        s["x_hat"] = [s["x_hat"][i] for i in idx]
        s["x_gt"] = [s["x_gt"][i] for i in idx]
        if s.get("target_dates") is not None:
            s["target_dates"] = [s["target_dates"][i] for i in idx]
        s["pred_offsets"] = [int(o) for o in offsets]


# ── Step-1 momentum retry (round 2) ───────────────────────────────────────────
# Dates that ABSTAIN after the round-1 Step-1 filter are regenerated ONCE more with an
# LLM-written ~50-word momentum summary of the recent-residual block injected into the
# Step-1 prompt as item 5 ({momentum_block} in the templates; empty in round 1). Only
# dates that abstain in BOTH rounds stay abstained. Round-2 files sit next to the
# round-1 ones with a _r2 suffix; the merged view is what Step 2 consumes.
MOMENTUM_PROMPT_NAME = "momentum_summary.txt"
MOMENTUM_TEMPERATURE = 0.3     # near-deterministic summarization
MOMENTUM_MAX_TOKENS  = 8192    # reasoning/thinking tokens count toward the cap (both GPT and
                               # Gemini pro); 2048 was observed to truncate the paragraph
MOMENTUM_CACHE_NAME  = "momentum_r2.jsonl"
STEP1_ROUNDS = (1, 2)


def round_suffix(rnd: int) -> str:
    """File-name suffix for a Step-1 round: '' for round 1, '_r2' for round 2."""
    return "" if int(rnd) <= 1 else "_r2"


def momentum_section(momentum_text: Optional[str]) -> str:
    """Render the momentum paragraph as prompt-input item 5 (fills {momentum_block}).
    Empty text (round 1) renders to '' so the round-1 prompt is unchanged."""
    t = (momentum_text or "").strip()
    if not t:
        return ""
    return ("\n5. MOMENTUM & DIRECTIONAL-SIGNAL SUMMARY — REFERENCE ONLY. A brief synthesis "
            "of several\n   derived directional signals (recent errors vs. the baseline, "
            "similar past episodes,\n   surveillance indicators, level/phase, and related "
            "context) and the aggregated lean they\n   imply for where ground truth is "
            "heading (derived context, NOT report text — do NOT quote\n   from it). This "
            "summary is secondary background: still infer each period's direction from\n   "
            "the PRIMARY inputs — the value history (item 1), the baseline (item 2), the "
            "recent\n   residuals (item 3), and the report text (item 4) — and use this "
            "summary only to\n   cross-check that inference. Do NOT take a direction from "
            "this summary that the primary\n   inputs do not themselves support; when it "
            "conflicts with them, the primary inputs win:\n" + t + "\n")


# ── Similar-past-episode analogs (round-2 assist, fused into the momentum summary) ────
# For an abstained origin, find the k historical 12-period windows of the SAME series
# whose shape most resembles the current window, and report the actual values' realized
# next-H-period changes. Everything is causally pre-origin; matching is on GROUND-TRUTH
# counts only (scale-free, so one implementation serves all modes).
ANALOG_TOP_K    = 3
ANALOG_MIN_SEP  = 4     # selected episodes must end ≥4 periods apart (no shifted copies)
ANALOG_MIN_GAP  = 16    # candidate window must end ≥16 periods before the origin
ANALOG_FLAT_TV  = 0.10  # skip when the query shape's total variation is below this
ANALOG_RANK_LABELS = ("most similar episode", "2nd most similar episode",
                      "3rd most similar episode", "4th most similar episode",
                      "5th most similar episode")


def build_analog_episodes(hist: "pd.Series", reference_date: str,
                          L: int = HISTORY_LEN, H: int = PRED_LEN,
                          k: int = ANALOG_TOP_K, min_sep: int = ANALOG_MIN_SEP,
                          min_gap: int = ANALOG_MIN_GAP,
                          flat_tv: float = ANALOG_FLAT_TV) -> List[Dict]:
    """Top-k similar past episodes for one origin. `hist` is the state's full weekly
    ground-truth count series (build_histories). Windows are normalized to SHAPE by
    dividing by the window's final value (floored at 1, matching the mode's residual
    convention); similarity is recency-weighted Euclidean distance. Returns [] when the
    origin's window is too flat to match meaningfully or history is too short."""
    vals = hist.values.astype(float)
    idx = hist.index
    ref_ts = pd.Timestamp(parse_date(reference_date))
    pos = idx.searchsorted(ref_ts)
    if pos >= len(idx) or idx[pos] != ref_ts or pos < L - 1:
        return []
    t = int(pos)

    def shape(end: int) -> np.ndarray:
        w = vals[end - L + 1: end + 1]
        return w / max(abs(w[-1]), 1.0)

    q = shape(t)
    if float(np.abs(np.diff(q)).sum()) < flat_tv:      # near-flat query → no analogs
        return []
    weights = np.linspace(0.5, 1.0, L)
    scored = []
    for i in range(L - 1, t - min_gap + 1):
        if i + H > t:                                   # outcomes must be observed
            continue
        d = float(np.sqrt((weights * (q - shape(i)) ** 2).sum()))
        scored.append((d, i))
    scored.sort()
    picked: List[Tuple[float, int]] = []
    for d, i in scored:
        if all(abs(i - j) >= min_sep for _d, j in picked):
            picked.append((d, i))
        if len(picked) >= k:
            break
    return [{"rank": r + 1, "distance": round(d, 4),
             "end_date": str(idx[i].date()),            # provenance only — NEVER shown
             "outcomes": [round((vals[i + h] - vals[i]) / max(abs(vals[i]), 1.0), 4)
                          for h in range(1, H + 1)]}
            for r, (d, i) in enumerate(picked)]


def fmt_analog_block(episodes: List[Dict]) -> str:
    """Deterministic date-free rendering of the similar-past-episode outcomes, with an
    honest majority/median summary line. '' when there are no episodes."""
    if not episodes:
        return ""
    H = len(episodes[0]["outcomes"])
    lines = []
    for e in episodes:
        label = ANALOG_RANK_LABELS[min(e["rank"] - 1, len(ANALOG_RANK_LABELS) - 1)]
        pct = ", ".join(f"{o:+.0%}" for o in e["outcomes"])
        lines.append(f"  {label}: the value then changed by {pct} over the next {H} periods")
    finals = [e["outcomes"][-1] for e in episodes]
    n = len(finals)
    n_low = sum(1 for f in finals if f < 0)
    med = float(np.median(finals))
    nth = {1: "1st", 2: "2nd", 3: "3rd"}.get(H, f"{H}th")
    if n_low == n or n_low == 0:
        word = "LOWER" if n_low == n else "HIGHER"
        lines.append(f"  Overall: in {n} of {n} similar episodes the value was {word} by "
                     f"the {nth} following period (median change {med:+.0%}).")
    else:
        lines.append(f"  Overall: the episodes disagree — {n_low} of {n} were lower and "
                     f"{n - n_low} of {n} were higher by the {nth} following period "
                     f"(median change {med:+.0%}).")
    return "\n".join(lines)


ANALOG_EMPTY_NOTE = ("[No sufficiently similar past episodes were found — summarize the "
                     "recent errors only.]")


# ── Round-2 directional signals + strength-weighted vote ──────────────────────
# The round-2 momentum block is built from several INDEPENDENT signals, each reduced
# to the SAME standardized read — a single direction for where ground truth is likely
# to head (rise / fall / hold) plus a rule-based strength (strong / moderate / weak).
# A deterministic strength-weighted vote turns the per-signal reads into one verdict
# (direction + confidence + supporting/dissenting signals); the momentum-summary LLM
# then renders that verdict + the signals into one short trend paragraph. Every signal
# is a pure function of data at/before the origin, so the whole stage is reproducible.
# A signal that cannot be computed (missing data) returns None and is simply omitted;
# the vote adapts to whichever signals are present.
STRENGTH_WEIGHT = {"strong": 3, "moderate": 2, "weak": 1}
_DIR_WORD = {"rise": "RISE", "fall": "FALL", "hold": "HOLD"}
# vote thresholds (tunable against the round-2 rescue rate)
VOTE_TIE_MARGIN   = 1     # winning − losing weight ≤ this ⇒ mixed/hold
VOTE_HIGH_MARGIN  = 4     # winning − losing weight ≥ this (and ≥2× the loser) ⇒ high


def _clean(s):
    return {"key": s["key"], "name": s["name"], "evidence": s["evidence"].strip(),
            "direction": s["direction"], "strength": s["strength"]}


def signal_recent_momentum(recent1, recent3, is_flat) -> Optional[Dict]:
    """SIGNAL A — recent momentum vs. the baseline. `recent1`/`recent3` are the
    build_recent_residuals[_diff] entry lists (oldest→newest) at the near and far horizon.
    Non-flat modes: residual = actual − forecast, so a residual running (increasingly)
    below zero means ground truth is softer than the trend model expects → FALL. Flat mode:
    the residual IS the ground-truth change, read directly. Strength rises with sign
    agreement across the window and is capped to weak when the two horizons disagree."""
    vals1 = [float(e["residual"]) for e in (recent1 or [])]
    if len(vals1) < 2:
        return None
    recent = vals1[-3:]
    score = sum(recent) / len(recent)
    signs = [1 if v > 0 else -1 if v < 0 else 0 for v in vals1]
    nz = [s for s in signs if s]
    if not nz:
        return None
    gt_sign = 1 if score > 0 else -1 if score < 0 else 0
    if gt_sign == 0:
        return None
    direction = "rise" if gt_sign > 0 else "fall"
    lead = 1 if sum(nz) > 0 else -1
    agree = sum(1 for s in nz if s == lead) / len(nz)
    strength = "strong" if agree >= 0.8 else "moderate" if agree >= 0.6 else "weak"
    vals3 = [float(e["residual"]) for e in (recent3 or [])]
    if vals3:
        s3 = sum(vals3[-3:]) / len(vals3[-3:])
        corro = (s3 > 0) == (score > 0) if s3 != 0 else None
        if corro is False and strength != "weak":
            strength = "weak"
        elif corro is True and strength == "moderate":
            strength = "strong"
    if is_flat:
        ev = ("the series has been rising over the recent weeks"
              if direction == "rise" else "the series has been falling over the recent weeks")
    else:
        ev = ("actual values have been coming in above the baseline, i.e. ground truth is "
              "running stronger than the trend model expects"
              if direction == "rise" else
              "actual values have been coming in below the baseline, i.e. ground truth is "
              "running softer than the trend model expects")
    return _clean({"key": "A", "name": "Recent momentum vs. baseline",
                   "evidence": ev, "direction": direction, "strength": strength})


def signal_analogs(episodes) -> Optional[Dict]:
    """SIGNAL B — similar past episodes (own-series shape analogs). Direction from the
    median final-horizon outcome; strength is capped to weak whenever the episodes
    disagree among themselves (the abstained-date regime)."""
    if not episodes:
        return None
    finals = [float(e["outcomes"][-1]) for e in episodes]
    n = len(finals)
    n_low = sum(1 for f in finals if f < 0)
    n_high = sum(1 for f in finals if f > 0)
    med = float(np.median(finals))
    if med == 0 and n_low == n_high:
        return None
    direction = "rise" if med > 0 else "fall" if med < 0 else "hold"
    disagree = not (n_low == n or n_high == n)
    strength = "weak" if disagree else "moderate"
    move = "higher" if direction == "rise" else "lower" if direction == "fall" else "little"
    ev = (f"past stretches shaped like the current one mostly drifted {move} afterward"
          + (", though they disagree with one another" if disagree else " and largely agree"))
    return _clean({"key": "B", "name": "Similar past episodes",
                   "evidence": ev, "direction": direction, "strength": strength})


# The CURRENT-season weekly %-ILI is stated once in each report's prose ("… X% of patient
# visits were for ILI …", often wrapped across a line break). Collecting that single figure
# from each report across the lookback window yields a genuine WITHIN-season weekly series —
# unlike the "Percent ILI Visits" table, whose rows are DIFFERENT SEASONS' same-week values.
_ILI_PROSE_RE = re.compile(r"([\d.]+)\s*%\s*of\s+patient\s+visits\s+were\s+for\s+ILI", re.I | re.S)
_ILI_BASE_RE  = re.compile(r"baseline(?:\s*of|\s*:)?\s*([\d.]+)\s*%", re.I)


def extract_surveillance(entries) -> Optional[Dict]:
    """Best-effort ILINet %-ILI weekly series (oldest→newest) + regional baseline, one figure
    per report across the lookback window. `entries` = [(date, label, text)] as returned by
    cdc_store.for_dates (raw text). Returns None when < 2 reports are parseable (omit signal)."""
    dated = sorted(((d, t) for (d, _l, t) in (entries or []) if t), key=lambda x: x[0])
    series = []
    for _d, t in dated:
        m = _ILI_PROSE_RE.search(t)
        if m:
            series.append(float(m.group(1)))
    if len(series) < 2:
        return None
    b = _ILI_BASE_RE.search(dated[-1][1])
    return {"ili": series, "baseline": float(b.group(1)) if b else None}


def signal_surveillance(surv) -> Optional[Dict]:
    """SIGNAL C — surveillance leading indicators. Reads the recent trend of the ILINet
    %-ILI series: cooling from a recent peak → FALL, rising → RISE. Strength is strong when
    the recent trend is monotone, and the 'still above baseline' qualifier is added when the
    latest value exceeds the regional baseline."""
    if not surv:
        return None
    ili = surv.get("ili") or []
    base = surv.get("baseline")
    if len(ili) < 3:
        return None
    recent = ili[-4:]
    last, first, peak = recent[-1], recent[0], max(recent)
    if last < peak * 0.9 and last <= recent[-2]:
        direction, trend = "fall", "cooling from a recent peak"
    elif last > first * 1.1 and last >= recent[-2]:
        direction, trend = "rise", "rising"
    else:
        return None
    above = base is not None and last > base
    mono = (all(recent[i] <= recent[i - 1] for i in range(1, len(recent)))
            or all(recent[i] >= recent[i - 1] for i in range(1, len(recent))))
    strength = "strong" if mono else "moderate"
    ev = (f"surveillance activity is {trend}"
          + (", though still above its seasonal baseline" if above else ""))
    return _clean({"key": "C", "name": "Surveillance indicators",
                   "evidence": ev, "direction": direction, "strength": strength})


# ── Signal C, FINANCE variant (finance_stat / finance_TSFM ONLY) ──────────────
# The flu Signal C above (extract_surveillance / signal_surveillance) parses ILINet
# %-ILI phrasing and stays UNCHANGED for influenza. The finance modes instead read
# the market TONE of the weekly news reports: bullish vs bearish stock phrasing.
_BULL_RE = re.compile(
    r"\b(rall(?:y|ied|ies)|surg(?:e|ed|es|ing)|soar(?:s|ed|ing)?|jump(?:s|ed|ing)?|"
    r"gain(?:s|ed|ing)?|ros[e]|ris(?:e|es|ing)|climb(?:s|ed|ing)?|rebound(?:s|ed|ing)?|"
    r"record high|all-time high|beat(?:s)? (?:estimates|expectations|forecasts)|"
    r"upgrad(?:e|ed|es)|outperform(?:s|ed)?|buy rating|strong (?:demand|growth|results|sales|quarter)|"
    r"bullish|optimis(?:m|tic)|breakthrough|boost(?:s|ed|ing)?)\b", re.I)
_BEAR_RE = re.compile(
    r"\b(fell|fall(?:s|ing)?|drop(?:s|ped|ping)?|declin(?:e|ed|es|ing)|plung(?:e|ed|es|ing)|"
    r"slump(?:s|ed|ing)?|sank|sink(?:s|ing)?|tumbl(?:e|ed|es|ing)|sell-?off|"
    r"downgrad(?:e|ed|es)|miss(?:es|ed)? (?:estimates|expectations|forecasts)|"
    r"underperform(?:s|ed)?|bearish|pessimis(?:m|tic)|layoffs?|"
    r"cuts? (?:guidance|outlook|forecast)|warn(?:s|ed|ing)?|recession|lawsuit|probe|antitrust)\b", re.I)


def extract_market_sentiment(entries) -> Optional[Dict]:
    """FINANCE Signal-C source: per weekly news report, the net bullish-vs-bearish tone
    score (bull − bear)/(bull + bear) ∈ [−1, 1] from stock-related phrasing. Reports with
    no matched phrasing are skipped. Returns {"scores": [oldest→newest]} or None when
    fewer than 2 reports scored (signal omitted)."""
    dated = sorted(((d, t) for (d, _l, t) in (entries or []) if t), key=lambda x: x[0])
    scores = []
    for _d, t in dated:
        b = len(_BULL_RE.findall(t))
        r = len(_BEAR_RE.findall(t))
        if b + r == 0:
            continue
        scores.append((b - r) / (b + r))
    if len(scores) < 2:
        return None
    return {"scores": scores}


def signal_market_sentiment(sent) -> Optional[Dict]:
    """SIGNAL C (finance) — news market tone. Reads the recent net bullish/bearish tone of
    the weekly reports: predominantly bullish and latest report bullish → RISE; bearish →
    FALL; mixed → omitted. Strength is strong when the recent tone is one-sided (same sign
    throughout and decisively away from neutral), else moderate."""
    if not sent:
        return None
    scores = sent.get("scores") or []
    if len(scores) < 2:
        return None
    recent = scores[-3:]
    m = sum(recent) / len(recent)
    last = recent[-1]
    if m >= 0.15 and last > 0:
        direction, tone = "rise", "bullish"
    elif m <= -0.15 and last < 0:
        direction, tone = "fall", "bearish"
    else:
        return None
    one_sided = all((s > 0) == (m > 0) for s in recent if s != 0)
    strength = "strong" if one_sided and abs(m) >= 0.4 else "moderate"
    ev = (f"the recent news coverage tone is predominantly {tone} "
          f"(net tone {m:+.2f} over the last {len(recent)} reports)")
    return _clean({"key": "C", "name": "News market tone",
                   "evidence": ev, "direction": direction, "strength": strength})


def _hist_to_origin(hist, reference_date):
    ref_ts = pd.Timestamp(parse_date(reference_date))
    return hist[hist.index <= ref_ts]


def signal_level(hist, reference_date) -> Optional[Dict]:
    """SIGNAL D — level / phase context. Where the current value sits in the series' own
    historical range and which limb it is on. High and declining → FALL (mean-reversion);
    low and rising → RISE. Strength moderate at the extremes, weak mid-range, omitted when
    ambiguous."""
    if hist is None:
        return None
    s = _hist_to_origin(hist, reference_date)
    if len(s) < 8:
        return None
    arr = s.values.astype(float)
    current = float(arr[-1])
    pct = float((arr <= current).mean())
    if len(s) <= 3:
        return None
    slope = current - float(arr[-4])
    if slope < 0 and pct > 0.5:
        direction, where = "fall", "high in" if pct > 0.66 else "above the middle of"
        strength = "moderate" if pct > 0.66 else "weak"
    elif slope > 0 and pct < 0.5:
        direction, where = "rise", "low in" if pct < 0.34 else "below the middle of"
        strength = "moderate" if pct < 0.34 else "weak"
    else:
        return None
    limb = "declining" if slope < 0 else "rising"
    ev = f"the series sits {where} its historical range and is on the {limb} limb"
    return _clean({"key": "D", "name": "Level context",
                   "evidence": ev, "direction": direction, "strength": strength})


def signal_acceleration(hist, reference_date) -> Optional[Dict]:
    """SIGNAL E — turning-point / acceleration. Whether the most recent change is speeding
    up or losing steam relative to the prior change. A rising move losing steam points down;
    a falling move losing steam points up; a persistent acceleration confirms its direction.
    Always weak — a leading hint, not a strong vote."""
    if hist is None:
        return None
    s = _hist_to_origin(hist, reference_date)
    if len(s) < 4:
        return None
    v = s.values.astype(float)[-4:]
    d_mid, d_new = v[2] - v[1], v[3] - v[2]
    lvl = max(abs(v[3]), 1.0)
    accel = (d_new - d_mid) / lvl
    if d_new > 0 and accel < -0.05:
        direction, note = "fall", "a rising move that is losing steam"
    elif d_new < 0 and accel > 0.05:
        direction, note = "rise", "a falling move that is losing steam"
    elif d_new > 0 and accel >= 0.05:
        direction, note = "rise", "an accelerating rise"
    elif d_new < 0 and accel <= -0.05:
        direction, note = "fall", "an accelerating decline"
    else:
        return None
    return _clean({"key": "E", "name": "Turning-point (acceleration)",
                   "evidence": f"the latest movement looks like {note}",
                   "direction": direction, "strength": "weak"})


def signal_neighbors(state, neighbor_hists, reference_date, lookback: int = 4) -> Optional[Dict]:
    """SIGNAL F — co-modeled neighboring states. The majority recent direction of the OTHER
    states in this run. Always weak (the least causal signal); omitted when the neighbors
    split evenly or none are available."""
    if not neighbor_hists:
        return None
    dirs = []
    for st, h in neighbor_hists.items():
        if st == state:
            continue
        s = _hist_to_origin(h, reference_date)
        if len(s) < lookback + 1:
            continue
        slope = float(s.values[-1]) - float(s.values[-1 - lookback])
        if slope:
            dirs.append(1 if slope > 0 else -1)
    if not dirs:
        return None
    up, down = dirs.count(1), dirs.count(-1)
    if up == down:
        return None
    direction = "rise" if up > down else "fall"
    move = "rising" if direction == "rise" else "falling"
    return _clean({"key": "F", "name": "Neighboring states",
                   "evidence": f"the other co-modeled states are mostly {move} this week",
                   "direction": direction, "strength": "weak"})


def vote_signals(signals: List[Dict]) -> Dict:
    """Strength-weighted tally over the per-signal reads. Returns the verdict:
    {direction, confidence, weights, supporting, dissenting, note}. Direction is the heavier
    of rise/fall unless the margin is within one weak signal (or hold outweighs it), in which
    case it is 'hold' (mixed). Confidence is high when the winner is ≥2× the loser AND leads
    by ≥VOTE_HIGH_MARGIN, medium for a clear-but-modest lead, low for a near-tie."""
    w = {"rise": 0, "fall": 0, "hold": 0}
    for s in signals:
        w[s["direction"]] += STRENGTH_WEIGHT[s["strength"]]
    lead_dir = "rise" if w["rise"] > w["fall"] else "fall" if w["fall"] > w["rise"] else "hold"
    lead, other = max(w["rise"], w["fall"]), min(w["rise"], w["fall"])
    margin = lead - other
    if lead == 0 or margin <= VOTE_TIE_MARGIN or lead <= w["hold"]:
        direction, confidence = "hold", "low"
    elif lead >= 2 * other and margin >= VOTE_HIGH_MARGIN:
        direction, confidence = lead_dir, "high"
    else:
        direction, confidence = lead_dir, "medium"
    tag = lambda s: f'{s["key"]} ({s["strength"]})'
    if direction == "hold":
        supporting: List[str] = []
        dissenting = [tag(s) for s in signals]
        note = "signals are split; slight tilt at best, no confident direction"
    else:
        supporting = [tag(s) for s in signals if s["direction"] == direction]
        dissenting = [tag(s) for s in signals if s["direction"] != direction]
        note = (f"{confidence} confidence {direction} — "
                + ("clear, well-supported lean" if confidence == "high"
                   else "lean with meaningful counter-evidence"))
    return {"direction": direction, "confidence": confidence, "weights": w,
            "supporting": supporting, "dissenting": dissenting, "note": note}


def fmt_signal_block(signals: List[Dict]) -> str:
    """Render the signals in the standardized read format for the momentum prompt."""
    out = []
    for s in signals:
        out.append(f"SIGNAL {s['key']} — {s['name']}. {s['evidence']}.\n"
                   f"  -> Ground truth likely to {_DIR_WORD[s['direction']]}  "
                   f"(strength: {s['strength']}).")
    return "\n\n".join(out)


def fmt_vote_block(verdict: Dict) -> str:
    """Render the computed verdict object for the momentum prompt."""
    w = verdict["weights"]
    return ("AGGREGATED VERDICT (computed from the signals above by a strength-weighted vote)\n"
            f"  Direction: {_DIR_WORD[verdict['direction']]}    Confidence: {verdict['confidence']}\n"
            f"  Weighted support — fall: {w['fall']}, rise: {w['rise']}, hold: {w['hold']}\n"
            f"  Supporting signals: {', '.join(verdict['supporting']) or 'none'}\n"
            f"  Dissenting / caveats: {', '.join(verdict['dissenting']) or 'none'}\n"
            f"  Note: {verdict['note']}")


def assemble_momentum(*, state, reference_date, recent1, recent3, is_flat, episodes,
                      surveillance, hist, neighbor_hists, lookback) -> Optional[Dict]:
    """Build every available signal for one abstained date, run the vote, and render the
    signal + verdict blocks fed to the momentum-summary LLM. Returns None if no signal could
    be computed (the date then gets no momentum injection). `episodes` = build_analog_episodes
    output; `surveillance` = extract_surveillance output; `hist` = this state's GT series;
    `neighbor_hists` = {state: GT series} for the co-modeled states."""
    signals = [s for s in (
        signal_recent_momentum(recent1, recent3, is_flat),
        signal_analogs(episodes),
        (signal_market_sentiment(surveillance)
         if surveillance and "scores" in surveillance
         else signal_surveillance(surveillance)),
        signal_level(hist, reference_date),
        signal_acceleration(hist, reference_date),
        signal_neighbors(state, neighbor_hists, reference_date),
    ) if s]
    if not signals:
        return None
    verdict = vote_signals(signals)
    return {"signals": signals, "verdict": verdict,
            "signal_block": fmt_signal_block(signals),
            "vote_block": fmt_vote_block(verdict)}


def step1_filtered_view(out_dir: Path) -> Path:
    """The Step-1 filter file downstream stages should read: the two-round merged view
    when the momentum retry has run, else the plain round-1 file."""
    merged = Path(out_dir) / "step1_filtered_merged.jsonl"
    return merged if merged.exists() else Path(out_dir) / "step1_filtered.jsonl"


# ══════════════════════════════════════════════════════════════════════════════
# Date windows
# ══════════════════════════════════════════════════════════════════════════════
def history_dates_for(reference_date: str, history_len: int = HISTORY_LEN) -> List[date]:
    ref = parse_date(reference_date)
    return [ref - timedelta(weeks=(history_len - 1 - i)) for i in range(history_len)]


def target_dates_for(reference_date: str, pred_len: int = PRED_LEN) -> List[date]:
    ref = parse_date(reference_date)
    return [ref + timedelta(weeks=i + 1) for i in range(pred_len)]


def lookback_dates_for(reference_date: str, k: int) -> List[date]:
    return history_dates_for(reference_date, history_len=k)


# ══════════════════════════════════════════════════════════════════════════════
# MMWR week / FluView text store
# ══════════════════════════════════════════════════════════════════════════════
def mmwr_week_year(d: date) -> Tuple[int, int]:
    """(MMWR year, MMWR week) — the CDC epiweek convention the FluView files are named
    by (Sun–Sat weeks; week 1 = first week whose Wednesday is in the new year)."""
    wday = (d.weekday() + 1) % 7
    sat  = d + timedelta(days=6 - wday)
    year = (sat - timedelta(days=3)).year
    jan1 = date(year, 1, 1)
    j1sat = jan1 + timedelta(days=6 - ((jan1.weekday() + 1) % 7))
    wk1sat = j1sat if (j1sat - timedelta(days=3)).year == year else j1sat + timedelta(days=7)
    return year, (sat - wk1sat).days // 7 + 1


def in_flu_season(reference_date: str) -> bool:
    _yr, wk = mmwr_week_year(parse_date(reference_date))
    return wk >= FLU_SEASON_START_WEEK or wk <= FLU_SEASON_END_WEEK


_FLUVIEW_MONTHS = {m: i for i, m in enumerate(
    ("January", "February", "March", "April", "May", "June", "July", "August",
     "September", "October", "November", "December"), 1)}
_FLUVIEW_DATE_RE     = re.compile(r"([A-Z][a-z]+)\s+(\d{1,2}),?\s+(\d{4})")
_FLUVIEW_ENDING_RE   = re.compile(r"ending\s+([A-Z][a-z]+)\s+(\d{1,2}),?\s+(\d{4})", re.IGNORECASE)
_FLUVIEW_WEEKLINE_RE = re.compile(r"week|ending|mmwr", re.IGNORECASE)
FLUVIEW_DATE_TOL_DAYS = 6


def _fluview_mk_date(mon: str, day: str, yr: str) -> Optional[date]:
    mi = _FLUVIEW_MONTHS.get(mon)
    if mi is None:
        return None
    try:
        return date(int(yr), mi, int(day))
    except ValueError:
        return None


def fluview_header_week_ending(text: str) -> Optional[date]:
    head = text[:500]
    for mon, day, yr in _FLUVIEW_ENDING_RE.findall(head):
        d = _fluview_mk_date(mon, day, yr)
        if d:
            return d
    best: Optional[date] = None
    for line in head.splitlines():
        if not _FLUVIEW_WEEKLINE_RE.search(line):
            continue
        ds = [d for d in (_fluview_mk_date(*x) for x in _FLUVIEW_DATE_RE.findall(line)) if d]
        if len(ds) >= 2:
            cand = max(ds)
            if best is None or cand > best:
                best = cand
    return best


@dataclass
class CDCTextStore:
    cdc_dir: Path
    file_prefix: str = "fluview_week"
    state: str = ""
    _cache: Dict[Tuple[int, int], Optional[str]] = field(default_factory=dict)

    def __post_init__(self):
        if not Path(self.cdc_dir).exists():
            logger.warning("CDC dir %s does not exist — all weeks 'no report'", self.cdc_dir)

    def _read(self, year: int, week: int) -> Optional[str]:
        key = (year, week)
        if key in self._cache:
            return self._cache[key]
        path = Path(self.cdc_dir) / f"{self.file_prefix}{week:02d}_{year}.txt"
        if not path.exists() and week == 53:
            alt = Path(self.cdc_dir) / f"{self.file_prefix}{week:02d}_{year + 1}.txt"
            if alt.exists():
                path = alt
        if (self.state, path.name) in FLUVIEW_EXCLUDE:
            logger.info("FluView exclude (data quality): %s [%s] → treated as no report",
                        path.name, self.state)
            self._cache[key] = None
            return None
        if not path.exists():
            self._cache[key] = None
            return None
        try:
            text = path.read_text(errors="replace").strip() or None
        except Exception as e:
            logger.warning("failed to read %s: %s", path, e)
            text = None
        self._cache[key] = text
        return text

    def for_dates(self, dates: Sequence[date]) -> List[Tuple[date, str, Optional[str]]]:
        out = []
        for d in sorted(dates):
            year, week = mmwr_week_year(d)
            label = f"{self.file_prefix}{week:02d}_{year}"
            text = self._read(year, week)
            if text is not None:
                e = fluview_header_week_ending(text)
                if e is not None and abs((e - d).days) > FLUVIEW_DATE_TOL_DAYS:
                    logger.warning("FluView guard: %s declares %s (>%dd from %s) — dropping.",
                                   label, e, FLUVIEW_DATE_TOL_DAYS, d)
                    text = None
            out.append((d, label, text))
        return out


def cdc_store_for_state(state: str, cdc_base: Path = CDC_DIR) -> CDCTextStore:
    code = state.strip().upper()
    if code not in STATE_FLUVIEW:
        raise SystemExit(f"state {state!r} has no FluView text; known: {sorted(STATE_FLUVIEW)}")
    subdir, prefix = STATE_FLUVIEW[code]
    return CDCTextStore(cdc_dir=Path(cdc_base) / subdir, file_prefix=prefix, state=code)


# ══════════════════════════════════════════════════════════════════════════════
# CSV data layer — DataBundle + per-state/per-ticker CSV inputs (both data modes)
# ══════════════════════════════════════════════════════════════════════════════
def escape_multiline(text: str) -> str:
    """One-physical-line CSV encoding of a multi-line text field: backslash-escape
    backslashes and newlines. Reversed exactly by unescape_multiline (content is
    preserved byte-for-byte through the CSV round trip)."""
    return text.replace("\\", "\\\\").replace("\r", "\\r").replace("\n", "\\n")


_UNESCAPE_RE = re.compile(r"\\(\\|n|r)")
_UNESCAPE_MAP = {"\\": "\\", "n": "\n", "r": "\r"}


def unescape_multiline(text: str) -> str:
    return _UNESCAPE_RE.sub(lambda m: _UNESCAPE_MAP[m.group(1)], text)


@dataclass
class CsvTextStore:
    """CSV-backed weekly-text store with the same .for_dates interface as CDCTextStore.
    Texts are baked into the input CSVs (influenza: FluView report text, with the
    wrong-week guard + data-quality exclusions already applied at export time;
    finance: the week's news headlines + article bodies)."""
    state: str
    texts: Dict[str, Optional[str]]     # iso week date -> text (or None = no report)

    def for_dates(self, dates: Sequence[date]) -> List[Tuple[date, str, Optional[str]]]:
        out = []
        for d in sorted(dates):
            iso = d.isoformat()
            out.append((d, f"report_{iso}", self.texts.get(iso) or None))
        return out


@dataclass
class DataBundle:
    """Everything the sample builders need, loaded from the mode's input CSVs:
    full per-geo series, the Chronos-2 forecast cache (keyed by (geo, first forecast
    week) exactly like the original cache files), weekly text per (geo, week), the
    population map (influenza only) and, for finance_stat, explicit split date ranges."""
    data_mode: str
    histories: Dict[str, "pd.Series"]
    cache: Dict[Tuple[str, str], np.ndarray]
    texts: Dict[Tuple[str, str], Optional[str]]
    pop_map: Dict[str, float]
    split_ranges: Optional[Dict[str, Tuple[str, str]]] = None

    def store_for(self, state: str) -> CsvTextStore:
        code = state.strip().upper()
        return CsvTextStore(state=code,
                            texts={iso: t for (s, iso), t in self.texts.items() if s == code})


def load_influenza_bundle(states: Sequence[str],
                          data_dir: Path = INFLUENZA_DATA_DIR) -> DataBundle:
    """Per-state CSVs (export_influenza_csv.py), in the SAME windowed layout as the
    finance ticker CSVs: one row per complete origin week with state, reference_date,
    hist_12w / future_3w / chronos2_pred_3w (JSON lists), weekly_text (empty string
    when no usable FluView report), split (train/val/test/gap) and population. The
    weekly series is reconstructed from the overlapping hist/future windows; the
    Chronos-2 cache is keyed by (state, first forecast week) = reference_date + 1w;
    the split windows stay the pipeline's leak-safe date constants (the split column
    is informational, mirroring what the builder assigns)."""
    histories, cache, texts, pops = {}, {}, {}, {}
    for raw in states:
        code = raw.strip().upper()
        p = Path(data_dir) / f"{code}.csv"
        if not p.exists():
            raise SystemExit(f"influenza input CSV missing for {code!r}: {p} "
                             f"(run export_influenza_csv.py first)")
        # round_trip: exact float64 parse (the default fast parser can be 1 ulp off,
        # which would make CSV-loaded samples differ from the original NHSN path)
        df = pd.read_csv(p, float_precision="round_trip", keep_default_na=False)
        series: Dict[pd.Timestamp, float] = {}
        for _, r in df.iterrows():
            ref = pd.Timestamp(r["reference_date"])
            iso = ref.strftime("%Y-%m-%d")
            hist = json.loads(r["hist_12w"])
            fut = json.loads(r["future_3w"])
            for i, v in enumerate(hist):
                series[ref - pd.Timedelta(weeks=len(hist) - 1 - i)] = float(v)
            for i, v in enumerate(fut):
                series[ref + pd.Timedelta(weeks=i + 1)] = float(v)
            cache[(code, (ref + pd.Timedelta(weeks=1)).strftime("%Y-%m-%d"))] = \
                np.array(json.loads(r["chronos2_pred_3w"]), dtype=float)
            t = unescape_multiline(str(r["weekly_text"]))
            texts[(code, iso)] = t if t.strip() else None
        histories[code] = pd.Series(series).sort_index()
        pops[code] = float(df["population"].iloc[0])
    return DataBundle("influenza", histories, cache, texts, pops, split_ranges=None)


def load_finance_bundle(tickers: Sequence[str],
                        data_dir: Path = FINANCE_DATA_DIR,
                        baseline_col: Optional[str] = None,
                        label: str = "finance_stat") -> DataBundle:
    """Per-ticker organized CSVs: one row per origin week with hist_open_12w /
    future_open_3w (JSON lists), article_titles / articles ('|||'-separated news of
    the origin week) and split. The weekly Open series is reconstructed from the
    overlapping windows. The BASELINE cache holds the STATISTICAL prediction
    stat_pred_3w (12-week mean carried flat); chronos2_pred_3w is the fallback only
    for old CSV snapshots without the stat column (their chronos2_pred_3w contained
    those same mean values). Keyed by (ticker, origin+1w) = the first forecast week,
    matching the influenza cache convention; split date ranges come from the CSVs'
    split column."""
    histories, cache, texts = {}, {}, {}
    lo: Dict[str, str] = {}
    hi: Dict[str, str] = {}
    for raw in tickers:
        code = raw.strip().upper()
        p = Path(data_dir) / f"{code}.csv"
        if not p.exists():
            raise SystemExit(f"finance input CSV missing for {code!r}: {p}")
        df = pd.read_csv(p, keep_default_na=False)
        # the organized CSVs have appeared with both column namings
        hist_col = "hist_open_12w" if "hist_open_12w" in df.columns else "hist_12w"
        fut_col = "future_open_3w" if "future_open_3w" in df.columns else "future_3w"
        if baseline_col is not None:                 # finance_TSFM: strict column
            if baseline_col not in df.columns:
                raise SystemExit(f"{code}: required baseline column {baseline_col!r} "
                                 f"missing from {p}")
            base_col = baseline_col
        else:
            # finance_stat baseline = the statistical (12-wk-mean) prediction; old CSV
            # snapshots without stat_pred_3w stored those same values in chronos2_pred_3w
            base_col = "stat_pred_3w" if "stat_pred_3w" in df.columns else "chronos2_pred_3w"
        series: Dict[pd.Timestamp, float] = {}
        for _, r in df.iterrows():
            ref = pd.Timestamp(r["origin_week"])
            iso = ref.strftime("%Y-%m-%d")
            hist = json.loads(r[hist_col])
            fut = json.loads(r[fut_col])
            for i, v in enumerate(hist):
                series[ref - pd.Timedelta(weeks=len(hist) - 1 - i)] = float(v)
            for i, v in enumerate(fut):
                series[ref + pd.Timedelta(weeks=i + 1)] = float(v)
            cache[(code, (ref + pd.Timedelta(weeks=1)).strftime("%Y-%m-%d"))] = \
                np.array(json.loads(r[base_col]), dtype=float)
            titles = [t.strip() for t in str(r["article_titles"]).split("|||") if t.strip()]
            bodies = [b.strip() for b in str(r["articles"]).split("|||")]
            parts = []
            for j, t in enumerate(titles):
                body = bodies[j] if j < len(bodies) else ""
                parts.append(f"HEADLINE: {t}" + (f"\n{body}" if body else ""))
            texts[(code, iso)] = "\n\n".join(parts) if parts else None
            sp = str(r["split"]).strip().lower()
            if sp:
                lo[sp] = min(lo.get(sp, iso), iso)
                hi[sp] = max(hi.get(sp, iso), iso)
        histories[code] = pd.Series(series).sort_index()
    split_ranges = {sp: (lo[sp], hi[sp]) for sp in lo}
    return DataBundle(label, histories, cache, texts, {}, split_ranges=split_ranges)


def load_covid_bundle(states: Sequence[str],
                      data_dir: Path = COVID_DATA_DIR,
                      label: str = "covid_TSFM") -> DataBundle:
    """Per-state COVID CSVs (data/COVID/{STATE}.csv, code/data_build/build_covid_dataset.py), the influenza windowed
    layout but with MIXED units on disk: hist_12w / chronos2_pred_3w / stat_pred_3w are
    100 x the per-year z-score + 300 index shift (2dp; SHIFT300_MANIFEST.md) while
    future_3w is the full-precision z-score + 3.0. The future values are rescaled here
    (x100, round 2dp — the exporter's exact hist transform) so the reconstructed weekly
    series, ground truth and baseline all share the same shifted 100x-z units. Split date ranges come from the CSVs' split column
    (train/val/test with the 3-week 'gap' buffers already carved out by the exporter),
    like the finance loader; population from the CSV column. The denorm_mean/std_*
    columns (restore to the original ED-visit proportion) are not needed for generation
    and are consumed by finetune.py's original-scale MAE instead."""
    histories, cache, texts, pops = {}, {}, {}, {}
    lo: Dict[str, str] = {}
    hi: Dict[str, str] = {}
    for raw in states:
        code = raw.strip().upper()
        p = Path(data_dir) / f"{code}.csv"
        if not p.exists():
            raise SystemExit(f"covid input CSV missing for {code!r}: {p} "
                             f"(run code/data_build/build_covid_dataset.py first)")
        df = pd.read_csv(p, float_precision="round_trip", keep_default_na=False)
        series: Dict[pd.Timestamp, float] = {}
        for _, r in df.iterrows():
            ref = pd.Timestamp(r["reference_date"])
            iso = ref.strftime("%Y-%m-%d")
            hist = json.loads(r["hist_12w"])                    # already 100*z, 2dp
            if "future_3w_full" in df.columns:                  # release layout: 100*z at full precision
                fut = [round(float(v), 2) for v in json.loads(r["future_3w_full"])]
            else:                                               # pipeline layout: plain z
                fut = [round(float(v) * 100.0, 2) for v in json.loads(r["future_3w"])]
            for i, v in enumerate(hist):
                series[ref - pd.Timedelta(weeks=len(hist) - 1 - i)] = float(v)
            for i, v in enumerate(fut):
                series[ref + pd.Timedelta(weeks=i + 1)] = float(v)
            cache[(code, (ref + pd.Timedelta(weeks=1)).strftime("%Y-%m-%d"))] = \
                np.array(json.loads(r["chronos2_pred_3w"]), dtype=float)
            t = unescape_multiline(str(r["weekly_text"]))
            texts[(code, iso)] = t if t.strip() else None
            sp = str(r["split"]).strip().lower()
            if sp in ("train", "val", "test"):
                lo[sp] = min(lo.get(sp, iso), iso)
                hi[sp] = max(hi.get(sp, iso), iso)
        histories[code] = pd.Series(series).sort_index()
        pops[code] = float(df["population"].iloc[0])
    split_ranges = {sp: (lo[sp], hi[sp]) for sp in lo}
    return DataBundle(label, histories, cache, texts, pops,
                      split_ranges=split_ranges)


def load_data_bundle(data_mode: str, states: Sequence[str],
                     data_dir: Optional[Path] = None) -> DataBundle:
    if data_mode == "finance_stat":
        return load_finance_bundle(states, data_dir or FINANCE_DATA_DIR)
    if data_mode == "covid_TSFM":
        return load_covid_bundle(states, data_dir or COVID_DATA_DIR)
    if data_mode == "covid_TSFM_updated":
        return load_covid_bundle(states, data_dir or COVID_UPDATED_DATA_DIR,
                                 label="covid_TSFM_updated")
    if data_mode == "finance_TSFM":
        return load_finance_bundle(states, data_dir or FINANCE_DATA_DIR,
                                   baseline_col="chronos2_pred_3w",
                                   label="finance_TSFM")
    return load_influenza_bundle(states, data_dir or INFLUENZA_DATA_DIR)


# ══════════════════════════════════════════════════════════════════════════════
# Date scrubbing — redact absolute dates / week numbers / years; keep relative terms
# ══════════════════════════════════════════════════════════════════════════════
_MONTHS_RE = (r"(?:January|February|March|April|May|June|July|August|September|October|"
              r"November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)")
# Order matters: most specific first. URLs are redacted whole (they routinely encode the
# week number / date / publication id in any format), then explicit date shapes (both
# Month-first and Day-first), numeric dates, year ranges, week numbers, digit runs, and
# finally bare years. Relative terms ("the previous week", "this season") are KEPT.
_SCRUB_RULES = [
    # whole URLs → "[source]" (kills week-numbers / dates / ids embedded in any encoding)
    (re.compile(r"\b(?:https?://|www\.)\S+", re.I), "[source]"),
    # "week ending <Month DD, YYYY>" / "week ending <DD Month YYYY>"
    (re.compile(rf"\bweek\s+ending\s+(?:{_MONTHS_RE}\.?\s+\d{{1,2}}|\d{{1,2}}\s+{_MONTHS_RE})"
                rf"(?:,?\s+\d{{2,4}})?", re.I), "the report week"),
    # Day-first dates: "20 January 2023" / "20 Jan 23" / "20-Jan-2023". The trailing \b
    # after the month name stops abbrevs matching inside words ("8 declined" ≠ a date).
    (re.compile(rf"\b\d{{1,2}}[\s\-/]+(?:{_MONTHS_RE})\b\.?(?:[\s\-/,]+\d{{2,4}})?", re.I), "[date]"),
    # Month-first dates: "January 18, 2025" / "Jan 18" / "January 2025"
    (re.compile(rf"\b{_MONTHS_RE}\.?\s+\d{{1,2}}(?:,?\s+\d{{2,4}})?\b", re.I), "[date]"),
    (re.compile(rf"\b{_MONTHS_RE}\.?[\s,]+\d{{4}}\b", re.I), "[date]"),
    # numeric dates: 2025-01-18, 01/18/2025, 1/18/25, 01-18-2025. The trailing (?!\d)
    # (not \b) also catches dates run together with the next word by PDF text
    # extraction ("10/12/24Week Ending Date" in the covid respiratory-era reports).
    (re.compile(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}(?!\d)"), "[date]"),
    (re.compile(r"\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}(?!\d)"), "[date]"),
    # year / season ranges (with OR without the word 'season'): 2024-2025, 2022-23, 2024–25
    (re.compile(r"\b(?:19|20)\d{2}\s*[-–/]\s*(?:19|20)?\d{2}\b"), "the current season"),
    # MMWR / epi / surveillance week N (or a week RANGE) → it IS the report's own week.
    (re.compile(r"\b(?:MMWR|epi(?:demiological)?|surveillance)\s+weeks?\s+#?\d{1,3}"
                r"(?:\s*(?:[-–]|to|through|and)\s*#?\d{1,3})?\b", re.I), "the report week"),
    # "Week202608" / "Week_1" embedded in filenames/labels
    (re.compile(r"[Ww]eek[_-]?\d{3,8}"), "the report week"),
    # bare in-prose week NUMBER or week range → neutral "[week]" (absolute number removed,
    # but not falsely renamed "the report week", since it's a seasonal position not the origin)
    (re.compile(r"\b[Ww]eeks?[_\s\-]*#?\s*\d{1,3}(?:\s*(?:[-–]|to|through|and)\s*#?\d{1,3})?\b"),
     "[week]"),
    # date-shaped digit RUNS that slip past \b (YYYYMMDD=8, YYYYWW=6)
    (re.compile(r"(?<!\d)(?:19|20)\d{4,6}(?!\d)"), "[date]"),
    # any remaining standalone 4-digit year
    (re.compile(r"\b(?:19|20)\d{2}\b"), "[year]"),
    # tidy leftovers: "week ending [date]" / "the report week [date]" → "the report week"
    (re.compile(r"\bweek\s+ending\s+\[date\]", re.I), "the report week"),
    (re.compile(r"\bthe report week[\s,]+\[(?:date|year|week)\]", re.I), "the report week"),
]
# A leak = any ABSOLUTE date/week-number/year/url token surviving the scrub. Bare phrases
# like "week ending" (date already redacted to [date]) are NOT leaks; the \b after a month
# abbrev avoids flagging "declined"/"decreased" as a date.
_LEAK_RE = re.compile(
    rf"\bhttps?://|\bwww\.|\b(?:19|20)\d{{2}}\b|(?<!\d)(?:19|20)\d{{4,6}}(?!\d)|"
    rf"\b{_MONTHS_RE}\.?\s*,?\s*\d|\b\d\s+(?:{_MONTHS_RE})\b|\b\d{{1,2}}[/\-]\d{{1,2}}[/\-]\d|"
    rf"\b[Ww]eeks?[_\s\-]*#?\s*\d", re.I)


def scrub_dates(text: str) -> str:
    if not text:
        return text
    out = text
    for pattern, repl in _SCRUB_RULES:
        out = pattern.sub(repl, out)
    out = re.sub(r"\b([Tt]he)\s+the\b", r"\1", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r"\s+([.,;:])", r"\1", out)
    return out.strip()


def date_leaks(text: str) -> List[str]:
    return _LEAK_RE.findall(text or "")


# ══════════════════════════════════════════════════════════════════════════════
# Data layer — raw NHSN, Chronos-2 cache, leak-safe windowing, population, rates
# ══════════════════════════════════════════════════════════════════════════════
def load_raw() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, thousands=",", low_memory=False)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    return df


def build_histories(df_raw: pd.DataFrame, geos: Sequence[str]) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    for raw in geos:
        code = raw.strip().upper()
        sub = df_raw[df_raw[GEO_COL] == code][[DATE_COL, TARGET_COL]].dropna()
        if sub.empty:
            logger.warning("no NHSN rows for geo %r — skipping", code)
            continue
        out[code] = sub.sort_values(DATE_COL).set_index(DATE_COL)[TARGET_COL]
    return out


def load_chronos2_cache(cache_paths: Sequence[Path]) -> Dict[Tuple[str, str], np.ndarray]:
    required = {"state", "reference_date", "h0", "h1", "h2"}
    out: Dict[Tuple[str, str], np.ndarray] = {}
    loaded = 0
    for p in cache_paths:
        p = Path(p)
        if not p.exists():
            logger.warning("Chronos-2 cache %s not found — skipping", p)
            continue
        df = pd.read_csv(p)
        if not required <= set(df.columns):
            raise RuntimeError(f"cache {p} has unexpected columns: {list(df.columns)}")
        for _, row in df.iterrows():
            out[(str(row["state"]), str(row["reference_date"]))] = np.array(
                [row["h0"], row["h1"], row["h2"]], dtype=float)
        loaded += 1
    if not loaded:
        raise FileNotFoundError(f"none of the Chronos-2 caches exist: {[str(p) for p in cache_paths]}")
    return out


def build_examples(histories: Dict[str, pd.Series], start: str, end: str,
                   chronos2_cache: Dict[Tuple[str, str], np.ndarray]) -> List[Dict]:
    """Leak-safe `last_observed` windowing: reference_date = last observed week;
    history = 12 weeks ending at ref; targets = [ref+1..ref+3]; baseline =
    cache[(state, ref+1week)] (context ends at ref → no peek past ref)."""
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    examples: List[Dict] = []
    for state, actuals in histories.items():
        candidates = actuals.index[(actuals.index >= start_ts) & (actuals.index <= end_ts)]
        for ref_date in candidates:
            cutoff = ref_date
            target_dates = pd.DatetimeIndex([ref_date + pd.Timedelta(weeks=i + 1)
                                             for i in range(PRED_LEN)])
            base_ref = ref_date + pd.Timedelta(weeks=1)
            hist = actuals.loc[:cutoff].dropna()
            if len(hist) < HISTORY_LEN:
                continue
            if not all(d in actuals.index for d in target_dates):
                continue
            history = hist.tail(HISTORY_LEN).to_numpy(dtype=float)
            target  = np.array([float(actuals.loc[d]) for d in target_dates])
            baseline = chronos2_cache.get((state, base_ref.strftime("%Y-%m-%d")))
            if baseline is None:
                continue
            examples.append({
                "state": state,
                "reference_date": ref_date.strftime("%Y-%m-%d"),
                "target_dates": [d.strftime("%Y-%m-%d") for d in target_dates],
                "history": history.tolist(),
                "baseline": baseline.tolist(),
                "target": target.tolist(),
            })
    return examples


def build_chronos2_samples(cache_paths: Sequence[Path], include_states: Sequence[str],
                           splits: Sequence[str],
                           train_start: str = CHRONOS2_TRAIN_START,
                           bundle: Optional[DataBundle] = None) -> List[Dict]:
    split_ranges = {
        "train": (train_start, TRAIN_END),
        "test":  (TEST_START, date.today().isoformat()),
    }
    if bundle is not None and bundle.split_ranges:
        split_ranges = dict(bundle.split_ranges)
    splits_norm = [s.strip().lower() for s in splits if s.strip()]
    unknown = [s for s in splits_norm if s not in split_ranges]
    if unknown:
        raise ValueError(f"unknown split(s): {unknown}; expected {sorted(split_ranges)}")
    include = {s.strip().upper() for s in include_states}

    if bundle is not None:
        histories = {s: bundle.histories[s] for s in sorted(include) if s in bundle.histories}
        chronos2_cache = bundle.cache
    else:
        logger.info("Loading NHSN data + Chronos-2 cache…")
        df_raw = load_raw()
        histories = build_histories(df_raw, sorted(include))
        chronos2_cache = load_chronos2_cache(cache_paths)

    samples: List[Dict] = []
    for split in splits_norm:
        start, end = split_ranges[split]
        for ex in build_examples(histories, start, end, chronos2_cache):
            state = ex["state"]
            if state not in include:
                continue
            ref = ex["reference_date"]
            samples.append({
                "id": f"{split}_{state}_{ref}",
                "split": split, "state": state, "reference_date": ref,
                "target_dates": ex["target_dates"],
                "x_hist": list(map(float, ex["history"])),
                "x_hat":  list(map(float, ex["baseline"])),
                "x_gt":   list(map(float, ex["target"])),
            })

    # The VAL_START/GAP_START carve-out is the INFLUENZA split convention (val carved out
    # of the raw train range by module constants). Bundles whose CSVs carry their own
    # split column (finance, covid) are authoritative — carving would silently re-label
    # or drop their train dates (it did: covid variant-C train dates in Jun–Nov 2024).
    carve = bundle is None or not bundle.split_ranges
    kept, n_val, n_gap = [], 0, 0
    for s in samples:
        if carve and s["split"] == "train":
            rd = s["reference_date"]
            if VAL_START <= rd <= VAL_END:
                s["split"] = "val"
                s["id"] = f"val_{s['state']}_{rd}"
                n_val += 1
            elif GAP_START <= rd < VAL_START:
                n_gap += 1
                continue
        kept.append(s)
    logger.info("Built %d samples (val=%d, dropped gap=%d) for %s",
                len(kept), n_val, n_gap, sorted(include))
    return kept, chronos2_cache


def build_flat_examples(histories: Dict[str, pd.Series], start: str, end: str) -> List[Dict]:
    """FLAT-mode windowing: reference_date = last observed week; history = 12 weeks ending
    at ref; predicted weeks = [ref + FLAT_PRED_OFFSETS] (ref+1, ref+3); baseline = the FLAT
    value actual(ref - FLAT_BASELINE_OFFSET) = actual(ref-1), repeated for each predicted
    week. (The 2-week-ahead ref+2 is intentionally not predicted.)"""
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    examples: List[Dict] = []
    for state, actuals in histories.items():
        candidates = actuals.index[(actuals.index >= start_ts) & (actuals.index <= end_ts)]
        for ref_date in candidates:
            base_date = ref_date - pd.Timedelta(weeks=FLAT_BASELINE_OFFSET)
            target_dates = [ref_date + pd.Timedelta(weeks=o) for o in FLAT_PRED_OFFSETS]
            hist = actuals.loc[:ref_date].dropna()
            if len(hist) < HISTORY_LEN:
                continue
            if base_date not in actuals.index:
                continue
            if not all(d in actuals.index for d in target_dates):
                continue
            flat = float(actuals.loc[base_date])
            examples.append({
                "state": state, "reference_date": ref_date.strftime("%Y-%m-%d"),
                "target_dates": [d.strftime("%Y-%m-%d") for d in target_dates],
                "history": hist.tail(HISTORY_LEN).to_numpy(dtype=float).tolist(),
                "baseline": [flat for _ in FLAT_PRED_OFFSETS],
                "target": [float(actuals.loc[d]) for d in target_dates],
            })
    return examples


def build_flat_samples(cache_paths: Sequence[Path], include_states: Sequence[str],
                       splits: Sequence[str], train_start: str = CHRONOS2_TRAIN_START,
                       bundle: Optional[DataBundle] = None):
    """FLAT-mode samples (2 predicted weeks; flat ref-1 baseline). Same split/val/gap logic
    as chronos2; the Chronos-2 cache is still loaded (used for the horizon-matched recent
    residuals during generation)."""
    split_ranges = {"train": (train_start, TRAIN_END),
                    "test": (TEST_START, date.today().isoformat())}
    if bundle is not None and bundle.split_ranges:
        split_ranges = dict(bundle.split_ranges)
    splits_norm = [s.strip().lower() for s in splits if s.strip()]
    unknown = [s for s in splits_norm if s not in split_ranges]
    if unknown:
        raise ValueError(f"unknown split(s): {unknown}; expected {sorted(split_ranges)}")
    include = {s.strip().upper() for s in include_states}

    if bundle is not None:
        histories = {s: bundle.histories[s] for s in sorted(include) if s in bundle.histories}
        chronos2_cache = bundle.cache
    else:
        logger.info("Loading NHSN data + Chronos-2 cache (flat mode)…")
        df_raw = load_raw()
        histories = build_histories(df_raw, sorted(include))
        chronos2_cache = load_chronos2_cache(cache_paths)

    samples: List[Dict] = []
    for split in splits_norm:
        start, end = split_ranges[split]
        for ex in build_flat_examples(histories, start, end):
            state = ex["state"]
            if state not in include:
                continue
            ref = ex["reference_date"]
            samples.append({
                "id": f"{split}_{state}_{ref}", "split": split, "state": state,
                "reference_date": ref, "target_dates": ex["target_dates"],
                "x_hist": list(map(float, ex["history"])),
                "x_hat":  list(map(float, ex["baseline"])),
                "x_gt":   list(map(float, ex["target"])),
            })

    kept, n_val, n_gap = [], 0, 0
    for s in samples:
        if s["split"] == "train":
            rd = s["reference_date"]
            if VAL_START <= rd <= VAL_END:
                s["split"] = "val"; s["id"] = f"val_{s['state']}_{rd}"; n_val += 1
            elif GAP_START <= rd < VAL_START:
                n_gap += 1; continue
        kept.append(s)
    logger.info("Built %d FLAT samples (val=%d, dropped gap=%d) for %s",
                len(kept), n_val, n_gap, sorted(include))
    return kept, chronos2_cache


def build_population_map(states: Sequence[str], csv_path: Path = TARGET_HOSP_CSV) -> Dict[str, float]:
    df = pd.read_csv(csv_path, dtype={"location": str})
    df = df[df["weekly_rate"] > 0]
    pops: Dict[str, float] = {}
    for raw in states:
        code = raw.strip().upper()
        loc = STATE_LOC.get(code)
        if loc is None:
            sys.exit(f"no FluSight location code for geo {code!r} (add to STATE_LOC)")
        sub = df[df["location"] == loc]
        if sub.empty:
            sys.exit(f"no population row for geo {code!r} (loc {loc!r}) in {csv_path}")
        pop = statistics.median((sub["value"] / sub["weekly_rate"] * 1e5).tolist())
        pops[code] = float(pop)
        logger.info("population[%s] (loc %s) = %s", code, loc, f"{pop:,.0f}")
    return pops


def to_rates(samples: List[Dict], population: float) -> None:
    """In place: convert x_hist/x_hat/x_gt counts → per-RATE_PER residents."""
    scale = RATE_PER / population
    for s in samples:
        s["x_hist"] = [float(v) * scale for v in s["x_hist"]]
        s["x_hat"]  = [float(v) * scale for v in s["x_hat"]]
        s["x_gt"]   = [float(v) * scale for v in s["x_gt"]]


def round_sample_rates(samples: List[Dict], decimals: int = RATE_ROUND_DECIMALS) -> None:
    """In place, AFTER to_rates: stash the full-precision rate copies as *_full (recorded
    but NEVER shown to the model or used in any computation), then replace the working
    x_hist/x_hat/x_gt with values rounded to `decimals` dp. Every downstream block,
    residual, direction label, bin, and utility is then computed on these rounded values
    (baseline rounded per date, residuals derived from the rounded numbers)."""
    for s in samples:
        s["x_hist_full"] = [float(v) for v in s["x_hist"]]
        s["x_hat_full"]  = [float(v) for v in s["x_hat"]]
        s["x_gt_full"]   = [float(v) for v in s["x_gt"]]
        s["x_hist"] = [round(float(v), decimals) for v in s["x_hist"]]
        s["x_hat"]  = [round(float(v), decimals) for v in s["x_hat"]]
        s["x_gt"]   = [round(float(v), decimals) for v in s["x_gt"]]


def round_baseline_counts(samples: List[Dict], decimals: int = COUNT_DECIMALS) -> None:
    """chronos2_full: round the Chronos-2 BASELINE forecasts (x_hat) to the mode's working
    precision (whole admissions for influenza; cents for finance prices), so the stored/
    computed baseline matches the prompt display. The actuals (x_hist / x_gt) stay untouched;
    the recent residuals are rounded to the same precision at build time."""
    for s in samples:
        s["x_hat"] = [float(round(float(v), decimals)) for v in s["x_hat"]]


FULL_VALUE_KEYS = ("x_hist_full", "x_hat_full", "x_gt_full")


def full_value_fields(src: Dict) -> Dict:
    """The x_*_full duplicate fields, ONLY when present (rate modes stash rounded working
    copies + full originals). chronos2_full keeps x_hist/x_hat/x_gt at FULL precision
    directly — no rounded working copy, no *_full duplicates in any output file."""
    return {k: src[k] for k in FULL_VALUE_KEYS if src.get(k) is not None}


def tag_and_gate_fluview(samples: List[Dict], cdc_store: CDCTextStore,
                         lookback_weeks: int, min_present: int) -> List[Dict]:
    """Tag every sample with n_fluview_reports + in_season, then keep only samples
    with ≥min_present reports (default 1 ⇒ drop only fully text-less weeks)."""
    need = max(0, min(int(min_present), lookback_weeks))
    kept: List[Dict] = []
    for s in samples:
        lb = lookback_dates_for(s["reference_date"], lookback_weeks)
        n = sum(text is not None for _, _, text in cdc_store.for_dates(lb))
        s["n_fluview_reports"] = int(n)
        s["in_season"] = in_flu_season(s["reference_date"])
        if n >= need:
            kept.append(s)
    by = {sp: sum(1 for s in kept if s["split"] == sp) for sp in ("train", "val", "test")}
    logger.info("FluView gate (>=%d of %d): kept %d/%d (train=%d val=%d test=%d)",
                need, lookback_weeks, len(kept), len(samples), by["train"], by["val"], by["test"])
    return kept


# ══════════════════════════════════════════════════════════════════════════════
# Input blocks — numeric history / baseline / recent residuals / momentum / text
# ══════════════════════════════════════════════════════════════════════════════
def fmt_rate_block(dates: Sequence[date], values: Sequence[float],
                   show_dates: bool = False, decimals: int = RATE_DECIMALS) -> str:
    """One value per line, relative week index (no calendar dates unless show_dates).
    `decimals`: 2 for rates, 0 for chronos2_full counts."""
    n = len(values)
    lines = []
    for i, v in enumerate(values):
        if show_dates and i < len(dates):
            lines.append(f"  {dates[i].isoformat()}  {v:.{decimals}f}")
        else:
            wk = n - i           # weeks before origin (1 = origin week for history)
            tag = "origin week" if wk == 1 else f"{wk - 1} weeks before origin"
            lines.append(f"  (t-{n - i}, {tag}): {v:.{decimals}f}")
    return "\n".join(lines)


def fmt_baseline_block(values: Sequence[float], offsets: Optional[Sequence[int]] = None,
                       decimals: int = RATE_DECIMALS,
                       label_by_offset: bool = False) -> str:
    """Baseline per predicted week. `offsets` = weeks-ahead of the origin for each value
    (default 1..H for the chronos2 horizon; flat mode passes e.g. [1, 3]).
    label_by_offset=True numbers each week BY ITS OFFSET — finance_TSFM uses this so its
    single origin+3 target reads "predicted week 3 (origin + 3)", never "week 1"."""
    if offsets is None:
        offsets = list(range(1, len(values) + 1))
    return "\n".join(f"  predicted week {offsets[i] if label_by_offset else i + 1} "
                     f"(origin + {offsets[i]}): {v:.{decimals}f}"
                     for i, v in enumerate(values))


def rate_list(values: Sequence[float]) -> str:
    return ", ".join(f"{v:.{RATE_DECIMALS}f}" for v in values)


CHRONOS2_RESID_STEPS = (1, 3)   # chronos2 shows recent residuals at BOTH the 1-week-ahead and
                                # 3-week-ahead Chronos-2 horizons


def build_recent_residuals(state: str, reference_date: str, x_hist_rate: Sequence[float],
                           chronos2_cache: Dict[Tuple[str, str], np.ndarray],
                           scale: float, lookback: int, step_h: int = 1,
                           decimals: int = RATE_ROUND_DECIMALS,
                           relative: bool = False) -> List[Dict]:
    """The TEFL signal: for each of the last `lookback` OBSERVED weeks w, the residual
    (actual − Chronos-2's step_h-ahead forecast of w) in rate units. step_h=1 → the forecast
    keyed at w (h0, the NEXT-WEEK prediction); step_h=3 → the forecast keyed at w-2 (h2, the
    3-WEEK-AHEAD prediction). Causally available at the origin. Returns oldest→newest list of
    {weeks_ago, residual} for the weeks a forecast exists; baseline rounded to 2dp then
    subtracted, so residuals are 2dp; sign + = forecast too low."""
    hist_dates = history_dates_for(reference_date, len(x_hist_rate))
    ref = parse_date(reference_date)
    out: List[Dict] = []
    for i in range(len(x_hist_rate) - lookback, len(x_hist_rate)):
        if i < 0:
            continue
        w = hist_dates[i]
        if step_h <= 1:
            key, hidx = w.isoformat(), 0
        else:
            key, hidx = (w - timedelta(weeks=step_h - 1)).isoformat(), step_h - 1
        arr = chronos2_cache.get((state, key))
        if arr is None or hidx >= len(arr):
            continue
        base_rate = round(float(arr[hidx]) * scale, decimals)
        if relative:
            # chronos2_full: the residual as a CHANGING RATIO of the forecast —
            # (actual − forecast) / forecast, denominator floored at 1 admission.
            resid = round((float(x_hist_rate[i]) - base_rate) / max(abs(base_rate), 1.0),
                          REL_RESID_DECIMALS)
        else:
            resid = round(float(x_hist_rate[i]) - base_rate, decimals)
        out.append({"weeks_ago": round((ref - w).days / 7), "residual": resid})
    return out


def build_recent_residuals_diff(reference_date: str, x_hist_rate: Sequence[float],
                                lookback: int, step: int) -> List[Dict]:
    """FLAT-mode recent residuals = the ground-truth CHANGE over `step` weeks, for each of the
    last `lookback` observed weeks w:  GT(w) − GT(w − step)  (rate units, 2dp). This matches
    the flat baseline (the current level actual(ref)) and the prediction horizon: step=1 for
    the 1-week-ahead target, step=3 for the 3-week-ahead target. Sign: + = the rate rose over
    the period. Uses only the (already 2dp) history — no Chronos-2 cache."""
    hist_dates = history_dates_for(reference_date, len(x_hist_rate))
    ref = parse_date(reference_date)
    n = len(x_hist_rate)
    out: List[Dict] = []
    for i in range(n - lookback, n):
        if i < 0 or i - step < 0:
            continue
        resid = round(float(x_hist_rate[i]) - float(x_hist_rate[i - step]), RATE_ROUND_DECIMALS)
        out.append({"weeks_ago": round((ref - hist_dates[i]).days / 7), "residual": resid})
    return out


def fmt_recent_change_block(entries: List[Dict], step_weeks: int) -> str:
    """FLAT-mode recent-change block: each line is the GT change over `step_weeks` at a recent
    week, with a rose/fell annotation."""
    if not entries:
        return "[No recent changes are available for the lookback window.]"
    period = "week" if step_weeks <= 1 else f"{step_weeks}-week period"
    lines = []
    for e in entries:
        wa = e["weeks_ago"]
        rel = "the origin week" if wa <= 0 else (
            "1 week before origin" if wa == 1 else f"{wa} weeks before origin")
        r = e["residual"]
        d = "rose" if r > 0 else ("fell" if r < 0 else "no change")
        lines.append(f"  {rel}: {r:+.{RESID_DECIMALS}f}  (rate {d} over the prior {period})")
    return "\n".join(lines)


def fmt_recent_residual_block(entries: List[Dict], decimals: int = RESID_DECIMALS,
                              relative: bool = False) -> str:
    """relative=True (chronos2_full): entries are fractions of the forecast, shown as
    signed percentages (e.g. '+68%')."""
    if not entries:
        return "[No recent Chronos-2 residuals are available for the lookback window.]"
    lines = []
    for e in entries:
        wa = e["weeks_ago"]
        rel = "the origin week" if wa <= 0 else (
            "1 week before origin" if wa == 1 else f"{wa} weeks before origin")
        r = e["residual"]
        sign = "baseline too LOW (under-predicted)" if r > 0 else (
            "baseline too HIGH (over-predicted)" if r < 0 else "baseline on target")
        val = f"{r * 100:+.0f}%" if relative else f"{r:+.{decimals}f}"
        lines.append(f"  {rel}: {val}  ({sign})")
    return "\n".join(lines)


def build_momentum(x_hist_rate: Sequence[float]) -> Dict[str, Optional[float]]:
    """Relative-change momentum only (NO season): week-over-week Δ and 3-week slope."""
    h = list(x_hist_rate)
    wow = (h[-1] - h[-2]) if len(h) >= 2 else None
    slope3 = ((h[-1] - h[-4]) / 3.0) if len(h) >= 4 else None
    return {"wow": wow, "slope3": slope3}


def fmt_momentum_block(mom: Dict[str, Optional[float]]) -> str:
    def f(v):
        return "n/a" if v is None else f"{v:+.{RESID_DECIMALS}f}"
    return (f"  week-over-week change (most recent week minus the one before): {f(mom['wow'])} per 10,000\n"
            f"  recent 3-week slope (average weekly change over the last 4 weeks): {f(mom['slope3'])} per 10,000")


def report_kind(data_mode: str = "influenza") -> str:
    """The report label rendered in the weekly text block ('CDC FluView report' /
    'financial news report' / 'state respiratory surveillance report') — data wording
    only; the block structure is identical."""
    if data_mode in FINANCE_DATA_MODES:
        return "financial news report"
    if data_mode in COVID_DATA_MODES:
        return "state respiratory surveillance report"
    return "CDC FluView report"


def fmt_weekly_text_block(entries: List[Tuple[date, str, Optional[str]]],
                          reference_date: str,
                          kind: str = "CDC FluView report") -> str:
    """Render AVAILABLE weekly reports, date-SCRUBBED, labelled ONLY by relative
    distance from the origin. Weeks with no report are omitted."""
    ref = parse_date(reference_date)
    blocks = []
    for d, _label, text in entries:
        if text is None:
            continue
        scrubbed = scrub_dates(text.strip())
        leaks = date_leaks(scrubbed)
        if leaks:
            logger.warning("residual date leak after scrub (%s): %s", _label, leaks[:3])
        weeks_ago = round((ref - d).days / 7)
        if weeks_ago <= 0:
            rel = "the most recent surveillance week (the forecast-origin week)"
        elif weeks_ago == 1:
            rel = "1 week before the forecast origin"
        else:
            rel = f"{weeks_ago} weeks before the forecast origin"
        blocks.append(f"--- {kind} for {rel} ---\n{scrubbed}\n")
    if blocks:
        return "\n".join(blocks)
    return ("[No financial news reports are available for the recent weeks.]"
            if kind == "financial news report" else
            "[No CDC FluView surveillance reports are available for the recent weeks.]")


def assembled_text_for_faithfulness(weekly_text_block: str) -> str:
    """Normalised (lowercase, single-spaced) view of the scrubbed report block, used
    to verify that CoT quotes appear verbatim."""
    return re.sub(r"\s+", " ", weekly_text_block.lower()).strip()


# ══════════════════════════════════════════════════════════════════════════════
# Offline scoring — residual targets, direction labels, utility
# ══════════════════════════════════════════════════════════════════════════════
def residuals(x_hat: Sequence[float], x_gt: Sequence[float]) -> List[float]:
    """True residual r = gt − baseline per horizon week (rate units, 4-decimal)."""
    return [round(float(g) - float(b), RESID_DECIMALS) for b, g in zip(x_hat, x_gt)]


def direction_labels(x_hat: Sequence[float], x_gt: Sequence[float],
                     eps: float = NOCHANGE_BAND) -> List[int]:
    """+1 / −1 / 0 per week with the ε no-change dead-band (|r| ≤ ε ⇒ 0)."""
    return [0 if abs(r) <= eps else (1 if r > 0 else -1) for r in residuals(x_hat, x_gt)]


def relative_residuals(x_hat: Sequence[float], x_gt: Sequence[float]) -> List[float]:
    """chronos2_full: relative residual per week, (gt − baseline)/baseline (fraction of the
    baseline count). The denominator is floored at 1 admission so near-zero summer baselines
    cannot blow up the proportion."""
    return [round((float(g) - float(b)) / max(abs(float(b)), 1.0), REL_RESID_DECIMALS)
            for b, g in zip(x_hat, x_gt)]


def direction_labels_rel(x_hat: Sequence[float], x_gt: Sequence[float],
                         eps_rel: float = NOCHANGE_REL_BAND) -> List[int]:
    """chronos2_full labels: +1 / −1 / 0 with a RELATIVE dead-band — no-change ⇔ the change
    is within ±eps_rel (default 10%) of the baseline count."""
    return [0 if abs(r) <= eps_rel else (1 if r > 0 else -1)
            for r in relative_residuals(x_hat, x_gt)]


def direction_labels_for_mode(mode: str, x_hat: Sequence[float], x_gt: Sequence[float],
                              band: float, data_mode: str = "influenza") -> List[int]:
    """Mode dispatch: relative band for influenza/finance count modes; ABSOLUTE band
    (index points for covid_TSFM, rate units for chronos2/flat) otherwise."""
    if band_is_relative(mode, data_mode):
        return direction_labels_rel(x_hat, x_gt, band)
    return direction_labels(x_hat, x_gt, band)


def utility(delta: Sequence[float], r: Sequence[float]) -> float:
    """U = 2⟨δ,r⟩ − ‖δ‖² (squared-error reduction)."""
    dd = float(np.dot(delta, r))
    nn = float(np.dot(delta, delta))
    return 2.0 * dd - nn


# ══════════════════════════════════════════════════════════════════════════════
# Bins (Stage 2) + cold-start grounding m_init (Stage 5)
# ══════════════════════════════════════════════════════════════════════════════
def design_bins(abs_resids: Sequence[float], names: Sequence[str] = BIN_NAMES_DEFAULT,
                eps: float = NOCHANGE_BAND, kind: Optional[str] = None) -> Dict:
    """One-directional |r| quantile size classes over MOVING cases (|r| > ε).
    lower_edges[i] is the cold-start conservative magnitude m_init for class i (so any
    case correctly in class i has |r| ≥ lower_edges[i] ⇒ δ ≤ |r| ⇒ no overshoot).
    kind: None for per-10,000 rate residuals (chronos2/flat); BINS_KIND_ABS_INDEX for
    covid_TSFM index-point residuals (edges rounded to COVID_BAND_DECIMALS)."""
    moving = sorted(float(a) for a in abs_resids if float(a) > eps)
    M = len(names)
    nd = COVID_BAND_DECIMALS if kind == BINS_KIND_ABS_INDEX else None
    rnd = (lambda v: round(float(v), nd)) if nd is not None else float
    if len(moving) < M:
        # Degenerate corpus — fall back to ε-spaced edges so the pipeline still runs.
        lower = [eps * (1 + i) for i in range(M)]
        upper = lower[1:] + [None]            # None = open top (JSON-safe; no Infinity)
    else:
        qs = [i / M for i in range(M)]                 # 0, 1/M, …, (M-1)/M
        lower = [rnd(np.quantile(moving, q)) for q in qs]
        lower[0] = eps                                  # smallest moving bin starts at ε
        upper = lower[1:] + [rnd(moving[-1])]
    out = {
        "names": list(names), "lower_edges": lower, "upper_edges": upper,
        "epsilon": eps, "n_moving": len(moving),
    }
    if kind:
        out["kind"] = kind
    return out


def design_bins_relative(rel_abs_resids: Sequence[float],
                         names: Sequence[str] = BIN_NAMES_DEFAULT,
                         eps_rel: float = NOCHANGE_REL_BAND) -> Dict:
    """chronos2_full size classes: RELATIVE (fraction-of-baseline) QUANTILE bins.
    Change-proportions |r_rel| ≤ eps_rel (10%) are no-change; the moving cases,
    pooled across ALL states of the run, are split at the i/M quantiles so each of
    the M classes holds ~the same share of moving cases (~25% each for 4 classes).
    lower_edges[i] is the cold-start proportion m_init for class i (δ = m_init ×
    baseline; conservative, no overshoot). Edges are fractions (0.10 = 10%)."""
    moving = sorted(float(a) for a in rel_abs_resids if float(a) > eps_rel)
    M = len(names)
    if len(moving) < M:
        # Degenerate corpus — fall back to eps-spaced relative edges so the pipeline runs.
        lower = [eps_rel * (1 + i) for i in range(M)]
        upper = lower[1:] + [None]
    else:
        qs = [i / M for i in range(M)]                 # 0, 1/M, …, (M-1)/M
        lower = [round(float(np.quantile(moving, q)), REL_RESID_DECIMALS) for q in qs]
        lower[0] = eps_rel                              # smallest moving class starts at ε
        upper = lower[1:] + [round(moving[-1], REL_RESID_DECIMALS)]
    return {
        "names": list(names), "lower_edges": lower, "upper_edges": upper,
        "epsilon": eps_rel, "n_moving": len(moving), "kind": "relative",
    }


def m_init(bins: Dict, bin_index: int) -> float:
    """Cold-start lower-edge magnitude for a size class (conservative; no overshoot).
    For relative bins this is a FRACTION of the baseline (apply as m_init × baseline)."""
    return float(bins["lower_edges"][bin_index])


def bins_are_relative(bins: Dict) -> bool:
    return bool(bins) and bins.get("kind") == "relative"


def bin_index_of(bins: Dict, name: str) -> Optional[int]:
    name = (name or "").strip().lower()
    for i, nm in enumerate(bins["names"]):
        if nm.lower() == name:
            return i
    return None


def fmt_bin_definitions(bins: Dict) -> str:
    if bins_are_relative(bins):
        lines = []
        for i, nm in enumerate(bins["names"]):
            lo, hi = bins["lower_edges"][i], bins["upper_edges"][i]
            hi_s = ("and larger" if (hi is None or hi == float("inf"))
                    else f"up to ~{hi * 100:.0f}%")
            lines.append(f"  {nm:<11}: |change| roughly {lo * 100:.0f}% {hi_s} "
                         "of the baseline forecast")
        return "\n".join(lines)
    if bins.get("kind") == BINS_KIND_ABS_INDEX:
        # covid_TSFM: ABSOLUTE index-point classes (same width at every level)
        lines = []
        nd = COVID_BAND_DECIMALS
        for i, nm in enumerate(bins["names"]):
            lo, hi = bins["lower_edges"][i], bins["upper_edges"][i]
            hi_s = ("and larger" if (hi is None or hi == float("inf"))
                    else f"up to ~{hi:.{nd}f}")
            lines.append(f"  {nm:<11}: |change| roughly {lo:.{nd}f} {hi_s} index points "
                         "away from the baseline forecast")
        return "\n".join(lines)
    lines = []
    for i, nm in enumerate(bins["names"]):
        lo, hi = bins["lower_edges"][i], bins["upper_edges"][i]
        hi_s = "and larger" if (hi is None or hi == float("inf")) else f"up to ~{hi:.{RATE_DECIMALS}f}"
        lines.append(f"  {nm:<11}: |change| roughly {lo:.{RATE_DECIMALS}f} {hi_s} per 10,000")
    return "\n".join(lines)


def bin_names_csv(bins: Dict) -> str:
    return " | ".join(bins["names"])


def pool_abs_residuals(filtered_path: Path,
                       bins_split: Optional[str] = "train") -> List[float]:
    """Pool |r| from a run's step1_filtered.jsonl. Reads the per-date SUMMARY rows
    (candidate_index == -1), which carry one (x_hat, x_gt) per date across ALL states in
    the run — so the bins are pooled GLOBALLY over every state of the run. Prefer
    `bins_split` (train) to avoid test leakage; fall back to all splits if empty."""
    absr = []
    for r in iter_jsonl(filtered_path):
        if int(r.get("candidate_index", 0)) != -1:
            continue
        if bins_split and r.get("split") != bins_split:
            continue
        if "x_hat" in r and "x_gt" in r:
            absr.extend(abs(rr) for rr in residuals(r["x_hat"], r["x_gt"]))
    if not absr and bins_split is not None:
        return pool_abs_residuals(filtered_path, bins_split=None)
    return absr


def pool_rel_residuals(filtered_path: Path,
                       bins_split: Optional[str] = "train") -> List[float]:
    """chronos2_full: pool the |RELATIVE residual| (|gt − baseline| / baseline) from a run's
    step1_filtered.jsonl per-date SUMMARY rows — pooled GLOBALLY over every state of the
    run, so the smallest/largest changing proportion across all states set the bin range."""
    relr = []
    for r in iter_jsonl(filtered_path):
        if int(r.get("candidate_index", 0)) != -1:
            continue
        if bins_split and r.get("split") != bins_split:
            continue
        if "x_hat" in r and "x_gt" in r:
            relr.extend(abs(rr) for rr in relative_residuals(r["x_hat"], r["x_gt"]))
    if not relr and bins_split is not None:
        return pool_rel_residuals(filtered_path, bins_split=None)
    return relr


def build_or_load_run_bins(out_dir: Path, names: Sequence[str] = BIN_NAMES_DEFAULT,
                           eps: float = NOCHANGE_BAND, bins_split: Optional[str] = "train",
                           rebuild: bool = False, relative: bool = False,
                           abs_index: bool = False) -> Dict:
    """The run's GLOBAL bins: pooled over every state's residuals in this run's
    step1_filtered.jsonl, cached to out_dir/bins.json so Step-2 generation and filtering
    share identical size-class definitions across all states of the run.
    relative=True (influenza/finance chronos2_full): pool RELATIVE residuals and build
    QUANTILE fraction-of-baseline classes, each holding ~the same share of moving cases.
    abs_index=True (covid_TSFM): ABSOLUTE |residual| quantile classes in index points."""
    bins_path = Path(out_dir) / "bins.json"
    if bins_path.exists() and not rebuild:
        return json.loads(bins_path.read_text())
    flt = step1_filtered_view(out_dir)   # merged two-round view when the retry has run
    if relative:
        bins = design_bins_relative(pool_rel_residuals(flt, bins_split),
                                    names=names, eps_rel=eps)
    else:
        bins = design_bins(pool_abs_residuals(flt, bins_split), names=names, eps=eps,
                           kind=BINS_KIND_ABS_INDEX if abs_index else None)
    bins["scope"] = "global_over_run"
    bins_path.write_text(json.dumps(bins, indent=2))
    logger.info("designed GLOBAL run bins (%d classes, kind=%s, n_moving=%d) → %s",
                len(names), bins.get("kind", "absolute"), bins["n_moving"], bins_path)
    return bins


# ══════════════════════════════════════════════════════════════════════════════
# Parsing CoT outputs
# ══════════════════════════════════════════════════════════════════════════════
_DIR_MAP = {"increase": 1, "up": 1, "decrease": -1, "down": -1,
            "no change": 0, "no-change": 0, "nochange": 0, "none": 0, "no": 0}
DIRECTIONS_TAG = "Directions:"
SIZES_TAG = "Sizes:"


def _last_tagged_line(text: str, tag: str) -> Optional[str]:
    if not text or tag.lower() not in text.lower():
        return None
    tail = re.split(re.escape(tag), text, flags=re.IGNORECASE)[-1]
    return tail.splitlines()[0] if tail.strip() else ""


def parse_directions(cot: str, horizon: int) -> Optional[List[int]]:
    line = _last_tagged_line(cot, DIRECTIONS_TAG)
    if line is None:
        return None
    toks = [t.strip().lower() for t in line.split(",")]
    out: List[int] = []
    for t in toks:
        if t in _DIR_MAP:
            out.append(_DIR_MAP[t])
    if len(out) < horizon:
        return None
    return out[:horizon]


def parse_sizes(cot: str, horizon: int, bins: Dict) -> Optional[List[Optional[int]]]:
    """Parse the 'Sizes:' line → per-week bin index, or None for 'none' (no-change)."""
    line = _last_tagged_line(cot, SIZES_TAG)
    if line is None:
        return None
    toks = [t.strip().lower() for t in line.split(",")]
    out: List[Optional[int]] = []
    for t in toks:
        if t in ("none", "no", "no-change", "no change", "0", ""):
            out.append(None)
        else:
            bi = bin_index_of(bins, t)
            if bi is None:
                out.append(None)
            else:
                out.append(bi)
    if len(out) < horizon:
        return None
    return out[:horizon]


# ── Faithfulness: verbatim quote check ────────────────────────────────────────
_QUOTE_RE = re.compile(r"[\"“‘']([^\"“”‘’']{3,})['\"”’]")


def extract_quotes(cot: str) -> List[str]:
    """Phrases the CoT presents in quotation marks (≥3 chars)."""
    return [m.group(1).strip() for m in _QUOTE_RE.finditer(cot or "")]


def faithfulness(cot: str, weekly_text_block: str) -> Dict:
    """Verbatim quote-grounding: every quoted phrase must appear in the scrubbed text.
    Returns {n_quotes, n_found, missing, faithful}. faithful ⇔ ≥1 quote and all found."""
    norm_text = assembled_text_for_faithfulness(weekly_text_block)
    quotes = extract_quotes(cot)
    missing = []
    for q in quotes:
        nq = re.sub(r"\s+", " ", q.lower()).strip()
        if nq and nq not in norm_text:
            missing.append(q)
    n_found = len(quotes) - len(missing)
    return {"n_quotes": len(quotes), "n_found": n_found, "missing": missing,
            "faithful": len(quotes) >= 1 and len(missing) == 0}


# ══════════════════════════════════════════════════════════════════════════════
# Self-consistency confidence
# ══════════════════════════════════════════════════════════════════════════════
def alpha1_direction(dir_lists: List[List[int]]) -> Tuple[float, List[int]]:
    """α₁ = mean over horizon weeks of (#agreeing with the modal direction)/k. Also
    returns the modal direction per week."""
    if not dir_lists:
        return 0.0, []
    H = min(len(d) for d in dir_lists)
    k = len(dir_lists)
    fracs, modal = [], []
    for h in range(H):
        col = [d[h] for d in dir_lists]
        m = statistics.mode(col)
        modal.append(m)
        fracs.append(col.count(m) / k)
    return (statistics.fmean(fracs) if fracs else 0.0), modal


# ══════════════════════════════════════════════════════════════════════════════
# LLM clients (Gemini / OpenAI) — one .generate() interface
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class GeminiClient:
    """Vertex AI Gemini client authenticated via Application Default Credentials (ADC)
    — NO API key. Credentials come from `gcloud auth application-default login` or a
    service-account JSON pointed to by GOOGLE_APPLICATION_CREDENTIALS; the project is
    GOOGLE_CLOUD_PROJECT (or the `project` field), location GOOGLE_CLOUD_LOCATION
    (default 'global'). Optional thinking controls (applied to every call):
      thinking_budget  int; 0 disables thinking on gemini-2.5-flash-tier models
                       (2.5-pro has a 128-token floor; cannot be fully disabled)
      thinking_level   'low' | 'high' for gemini-3* models (they have no off switch;
                       a thinking_budget<=0 request is mapped to level 'low')"""
    project: Optional[str] = None
    location: Optional[str] = None
    thinking_budget: Optional[int] = None
    thinking_level: Optional[str] = None
    _client: object = None

    def __post_init__(self):
        import os
        from google import genai
        project = self.project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        # interactive calls ALWAYS default to 'global' — the only endpoint serving
        # gemini-3.x. Deliberately ignore GOOGLE_CLOUD_LOCATION (a stale regional
        # export once sent a whole run to us-central1 -> 404s); override only via
        # the explicit field or GEMINI_INTERACTIVE_LOCATION. Batch jobs pin their
        # own region separately (GEMINI_BATCH_LOCATION).
        location = (self.location or os.environ.get("GEMINI_INTERACTIVE_LOCATION")
                    or "global")
        if not project:
            raise SystemExit(
                "Vertex/ADC Gemini needs a Google Cloud project.\n"
                "  1) authenticate once:  gcloud auth application-default login\n"
                "     (or set GOOGLE_APPLICATION_CREDENTIALS to a service-account JSON)\n"
                "  2) export GOOGLE_CLOUD_PROJECT=<your-project-id>\n"
                "     (optional: GOOGLE_CLOUD_LOCATION, default 'global')")
        self.project, self.location = project, location
        self._client = genai.Client(vertexai=True, project=project, location=location)

    def _thinking_config(self, model: str):
        from google.genai import types as gtypes
        if model.startswith("gemini-3"):
            level = self.thinking_level
            if level is None and self.thinking_budget is not None and self.thinking_budget <= 0:
                level = "low"     # gemini-3 has no off switch; 'low' is the minimum
            return gtypes.ThinkingConfig(thinking_level=level) if level else None
        if self.thinking_budget is not None:
            budget = int(self.thinking_budget)
            if "pro" in model and 0 <= budget < 128:
                budget = 128      # 2.5-pro cannot disable thinking; 128 is its floor
            return gtypes.ThinkingConfig(thinking_budget=budget)
        return None

    def generate(self, model: str, prompt: str, *, temperature: float = 0.8,
                 top_p: float = 0.95, max_output_tokens: int = 2048,
                 retries: int = 3, backoff_s: float = 4.0) -> str:
        from google.genai import types as gtypes
        tc = self._thinking_config(model)
        cfg = gtypes.GenerateContentConfig(
            temperature=temperature, top_p=top_p, max_output_tokens=max_output_tokens,
            **({"thinking_config": tc} if tc is not None else {}))
        last_err = None
        for attempt in range(retries):
            try:
                resp = self._client.models.generate_content(
                    model=model, contents=prompt, config=cfg)
                txt = (resp.text or "").strip()
                if not txt:
                    raise RuntimeError("empty response text")
                return txt
            except Exception as e:
                last_err = e
                wait = backoff_s * (2 ** attempt)
                logger.warning("Gemini call failed (%d/%d, model=%s): %s — retry in %.1fs",
                               attempt + 1, retries, model, e, wait)
                time.sleep(wait)
        raise RuntimeError(f"Gemini call to {model} failed after {retries}: {last_err}")


BATCH_GCS_ENV = "GEMINI_BATCH_GCS_BUCKET"
BATCH_LINGER_S = 30.0        # flush once no new request arrived for this long
BATCH_POLL_S = 60.0          # batch-job status poll interval
BATCH_MAX_REQUESTS = 20000   # flush immediately at this queue size
BATCH_WORKERS = 4096         # thread cap so every pending call lands in ONE job
BATCH_FALLBACK_LOCATION = "global"   # interactive endpoint for un-batchable models
BATCH_FALLBACK_WORKERS = 16          # concurrency of the interactive fallback
_BATCH_DONE = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED",
               "JOB_STATE_EXPIRED", "JOB_STATE_PAUSED"}


def _batch_unavailable(err: Exception) -> bool:
    """True when a batch job cannot exist for this model/region at all (model not
    published in the batch region, or the org policy forbids the job's location) —
    the cases where retrying batch is pointless and interactive is the answer."""
    msg = str(err)
    return "PublisherModel" in msg or "resourceLocations" in msg


@dataclass
class GeminiBatchClient(GeminiClient):
    """Vertex AI BATCH-mode Gemini client (~50% token price vs interactive). Same
    blocking .generate() interface as GeminiClient, so every pipeline stage runs
    UNCHANGED: each worker thread enqueues its prompt and blocks; a dispatcher
    thread coalesces everything that arrives within BATCH_LINGER_S into one batch
    job per (model, generation-config), uploads the requests JSONL to GCS, waits
    for the job, downloads the predictions and wakes each caller with its text.
    Callers whose line failed get an exception -> the stage leaves that candidate
    pending, exactly like an interactive failure (rerun to fill).

    Models that CANNOT be batched in this project's allowed region (e.g.
    gemini-3.x-pro, which is served only from the org-policy-blocked 'global'
    endpoint) fall back automatically to INTERACTIVE calls at 'global' — full
    price for those calls, batch price for everything else.

    Needs a staging bucket: export GEMINI_BATCH_GCS_BUCKET=<bucket[/prefix]>.
    Run the stage with a worker cap >= the number of pending calls (the scripts
    raise --workers to BATCH_WORKERS automatically when --batch is on)."""
    gcs_bucket: Optional[str] = None
    linger_s: float = BATCH_LINGER_S
    poll_s: float = BATCH_POLL_S

    def __post_init__(self):
        import os
        # batch JOBS are regional (org policy also forbids 'global' jobs here) — pin
        # the job region independently of the interactive endpoint, which stays
        # 'global' so gemini-3.x models resolve for GeminiClient / the fallback
        if self.location is None:
            self.location = os.environ.get("GEMINI_BATCH_LOCATION", "us-central1")
        super().__post_init__()
        bucket = (self.gcs_bucket or os.environ.get(BATCH_GCS_ENV) or "").strip()
        if not bucket:
            raise SystemExit(
                "--batch needs a Cloud Storage staging bucket for the request/response "
                f"files: export {BATCH_GCS_ENV}=<bucket-name[/prefix]> (the ADC identity "
                "must be allowed to read/write it, and Vertex AI must be enabled).")
        bucket = bucket[5:] if bucket.startswith("gs://") else bucket
        self._bucket_name, _, self._bucket_prefix = bucket.partition("/")
        self._lock = threading.Lock()
        self._queue: List[dict] = []
        self._last_add = 0.0
        self._dispatcher = None
        self._interactive_models: Set[str] = set()   # models proven un-batchable here
        self._fallback: Optional[GeminiClient] = None
        self._fallback_lock = threading.Lock()

    def _fallback_client(self) -> GeminiClient:
        """Interactive client at BATCH_FALLBACK_LOCATION for un-batchable models."""
        with self._fallback_lock:
            if self._fallback is None:
                self._fallback = GeminiClient(
                    project=self.project, location=BATCH_FALLBACK_LOCATION,
                    thinking_budget=self.thinking_budget,
                    thinking_level=self.thinking_level)
            return self._fallback

    # ── caller side ───────────────────────────────────────────────────────────
    def generate(self, model: str, prompt: str, *, temperature: float = 0.8,
                 top_p: float = 0.95, max_output_tokens: int = 2048,
                 retries: int = 3, backoff_s: float = 4.0) -> str:
        cfg = {"temperature": float(temperature), "topP": float(top_p),
               "maxOutputTokens": int(max_output_tokens)}
        tc = self._thinking_config(model)
        if tc is not None:
            if tc.thinking_level is not None:
                cfg["thinkingConfig"] = {"thinkingLevel": str(tc.thinking_level.value)}
            elif tc.thinking_budget is not None:
                cfg["thinkingConfig"] = {"thinkingBudget": int(tc.thinking_budget)}
        req = {"model": model, "cfg": json.dumps(cfg, sort_keys=True), "prompt": prompt,
               "event": threading.Event(), "text": None, "error": None}
        with self._lock:
            self._queue.append(req)
            self._last_add = time.monotonic()
            if self._dispatcher is None:
                self._dispatcher = threading.Thread(target=self._dispatch_loop,
                                                    daemon=True, name="gemini-batch")
                self._dispatcher.start()
        if not req["event"].wait(timeout=30 * 3600):
            raise RuntimeError(f"batch call to {model} timed out")
        if req["error"] is not None:
            raise RuntimeError(f"batch call to {model} failed: {req['error']}")
        return req["text"]

    # ── dispatcher side ───────────────────────────────────────────────────────
    def _dispatch_loop(self):
        while True:
            time.sleep(1.0)
            with self._lock:
                if not self._queue:
                    continue
                idle = time.monotonic() - self._last_add
                if idle < self.linger_s and len(self._queue) < BATCH_MAX_REQUESTS:
                    continue
                batch, self._queue = self._queue, []
            groups: Dict[Tuple[str, str], List[dict]] = {}
            for r in batch:
                groups.setdefault((r["model"], r["cfg"]), []).append(r)
            logger.info("BATCH flush: %d requests -> %d job(s): %s", len(batch),
                        len(groups), {m: len(v) for (m, _c), v in groups.items()})
            for (model, cfg), reqs in groups.items():
                threading.Thread(target=self._run_group, args=(model, cfg, reqs),
                                 daemon=True).start()

    def _run_group(self, model: str, cfg_json: str, reqs: List[dict]):
        try:
            if model in self._interactive_models:
                self._run_group_interactive(model, cfg_json, reqs)
                return
            try:
                by_prompt = self._execute_batch(model, json.loads(cfg_json),
                                                [r["prompt"] for r in reqs])
            except Exception as e:
                if not _batch_unavailable(e):
                    raise
                self._interactive_models.add(model)
                logger.warning("BATCH unavailable for %s (%s) — falling back to "
                               "INTERACTIVE calls at %s (full price) for this and all "
                               "future %s requests", model, str(e).splitlines()[0][:100],
                               BATCH_FALLBACK_LOCATION, model)
                self._run_group_interactive(model, cfg_json, reqs)
                return
            for r in reqs:
                texts = by_prompt.get(r["prompt"])
                if texts:
                    r["text"] = texts.pop()
                else:
                    r["error"] = "no (non-empty) response line for this request"
        except Exception as e:
            logger.error("BATCH job failed (model=%s, n=%d): %s", model, len(reqs), e)
            for r in reqs:
                r["error"] = str(e)
        finally:
            for r in reqs:
                r["event"].set()

    def _run_group_interactive(self, model: str, cfg_json: str, reqs: List[dict]):
        """Interactive fallback: the same requests, one online call each, bounded
        concurrency. Fills text/error per request; events are set by _run_group."""
        cfg = json.loads(cfg_json)
        cli = self._fallback_client()

        def one(r):
            try:
                r["text"] = cli.generate(
                    model=model, prompt=r["prompt"],
                    temperature=cfg.get("temperature", 0.8),
                    top_p=cfg.get("topP", 0.95),
                    max_output_tokens=cfg.get("maxOutputTokens", 2048))
            except Exception as e:
                r["error"] = str(e)

        logger.info("INTERACTIVE fallback: %d calls (model=%s, %d-way)",
                    len(reqs), model, BATCH_FALLBACK_WORKERS)
        with ThreadPoolExecutor(max_workers=BATCH_FALLBACK_WORKERS) as ex:
            list(ex.map(one, reqs))

    def _execute_batch(self, model: str, cfg: dict,
                       prompts: List[str]) -> Dict[str, List[str]]:
        """Upload one requests JSONL, run one Vertex batch job, download predictions.
        Returns {prompt: [response texts]} (a list: identical prompts, e.g. the k
        self-consistency candidates of one date, each consume one response)."""
        from google.cloud import storage
        from google.genai import types as gtypes
        job_tag = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        prefix = "/".join(p for p in (self._bucket_prefix, "cot_batch", job_tag) if p)
        gs = storage.Client(project=self.project)
        bucket = gs.bucket(self._bucket_name)

        lines = [json.dumps({"request": {
                    "contents": [{"role": "user", "parts": [{"text": p}]}],
                    "generationConfig": cfg}}) for p in prompts]
        bucket.blob(f"{prefix}/input.jsonl").upload_from_string("\n".join(lines))
        src = f"gs://{self._bucket_name}/{prefix}/input.jsonl"
        dest = f"gs://{self._bucket_name}/{prefix}/output"
        job = self._client.batches.create(model=model, src=src,
                                          config=gtypes.CreateBatchJobConfig(dest=dest))
        logger.info("BATCH job %s: %d requests (model=%s) -> %s",
                    job.name, len(prompts), model, dest)
        state = ""
        while True:
            time.sleep(self.poll_s)
            job = self._client.batches.get(name=job.name)
            new_state = getattr(job.state, "name", str(job.state))
            if new_state != state:
                state = new_state
                logger.info("BATCH job %s: %s", job.name, state)
            if state in _BATCH_DONE:
                break
        if state != "JOB_STATE_SUCCEEDED":
            raise RuntimeError(f"batch job {job.name} ended in {state}")

        out: Dict[str, List[str]] = {}
        for blob in gs.list_blobs(self._bucket_name, prefix=f"{prefix}/output"):
            if not blob.name.endswith(".jsonl"):
                continue
            for line in blob.download_as_text().splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                try:
                    req_text = obj["request"]["contents"][0]["parts"][0]["text"]
                except (KeyError, IndexError, TypeError):
                    continue
                text = ""
                resp = obj.get("response") or {}
                for cand in (resp.get("candidates") or [])[:1]:
                    for part in ((cand.get("content") or {}).get("parts") or []):
                        text += part.get("text") or ""
                if text.strip():
                    out.setdefault(req_text, []).append(text.strip())
        return out


@dataclass
class OpenAIClient:
    api_key: str
    _client: object = None

    def __post_init__(self):
        from openai import OpenAI
        self._client = OpenAI(api_key=self.api_key)

    def generate(self, model: str, prompt: str, *, temperature: float = 0.8,
                 top_p: float = 0.95, max_output_tokens: int = 2048,
                 retries: int = 3, backoff_s: float = 4.0) -> str:
        # gpt-5* / o* are reasoning models: the API rejects temperature/top_p overrides,
        # and hidden reasoning tokens count toward max_completion_tokens — so the cap is
        # floored to keep short judge budgets from being consumed before any output.
        reasoning = model.startswith(("gpt-5", "o1", "o3", "o4"))
        kwargs = {"model": model,
                  "messages": [{"role": "user", "content": prompt}]}
        if reasoning:
            kwargs["max_completion_tokens"] = max(int(max_output_tokens), 4096)
        else:
            kwargs["temperature"] = temperature
            kwargs["top_p"] = top_p
            kwargs["max_tokens"] = int(max_output_tokens)
        last_err = None
        for attempt in range(retries):
            try:
                resp = self._client.chat.completions.create(**kwargs)
                txt = (resp.choices[0].message.content or "").strip()
                if not txt:
                    raise RuntimeError("empty response text")
                return txt
            except Exception as e:
                last_err = e
                # a billing/quota error will not clear on retry — fail the run fast
                if "insufficient_quota" in str(e):
                    raise RuntimeError(
                        f"OpenAI account has no quota (add billing credit): {e}") from e
                wait = backoff_s * (2 ** attempt)
                logger.warning("OpenAI call failed (%d/%d, model=%s): %s — retry in %.1fs",
                               attempt + 1, retries, model, e, wait)
                time.sleep(wait)
        raise RuntimeError(f"OpenAI call to {model} failed after {retries}: {last_err}")


def make_llm_client(provider: str, cred_path: Optional[Path] = None,
                    thinking_budget: Optional[int] = None,
                    thinking_level: Optional[str] = None,
                    use_batch: bool = False):
    """The run's LLM client for --model gemini|GPT. Gemini uses Vertex AI with
    Application Default Credentials (no API key; cred_path is ignored); use_batch
    switches it to the ~50%-priced Batch API transport. GPT still reads its API
    key from cred_path (default: the provider's credentials file)."""
    provider = normalize_provider(provider)
    if provider == "gemini":
        cls = GeminiBatchClient if use_batch else GeminiClient
        return cls(thinking_budget=thinking_budget, thinking_level=thinking_level)
    if use_batch:
        raise SystemExit("--batch is implemented for the gemini provider only")
    key = load_api_key(Path(cred_path) if cred_path else provider_cred_path(provider))
    return OpenAIClient(api_key=key)


# ── Small parsers for the LLM judges ──────────────────────────────────────────
_SCORE_RE = re.compile(r"Score\s*[:\-]\s*([0-9]+)", re.IGNORECASE)
_VERDICT_RE = re.compile(r"Faithful\s*[:\-]\s*(yes|no)", re.IGNORECASE)


def parse_judge_score(text: str) -> Optional[int]:
    m = _SCORE_RE.search(text or "")
    return int(m.group(1)) if m else None


def parse_faithful_verdict(text: str) -> Optional[bool]:
    m = _VERDICT_RE.search(text or "")
    if not m:
        return None
    return m.group(1).lower() == "yes"


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s — %(message)s", datefmt="%H:%M:%S")
