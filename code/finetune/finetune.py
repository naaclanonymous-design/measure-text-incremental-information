"""Finetune a Qwen3 model to emit the SEMANTIC OPERATOR (direction + magnitude bin) on the
CoT corpus (chronos2 / flat). Self-contained rewrite (imports NOTHING from the A3_CoT_finetune*
scripts).

Unlike a numeric-forecast finetune, the model is trained to output, per predicted week, a
categorical OPERATOR — a Direction (increase | decrease | no change) and a Size bin
(small | medium | large | very-large | none) — after its reasoning. There is NO number / MAE /
MSE. Evaluation is CATEGORICAL: does the model's per-week direction and magnitude bin match the
test set's ACTUAL result (the true direction sign(gt−baseline) and the true bin of |gt−baseline|)?

Two modes (the corpus's two horizons):
  chronos2 : 3 predicted weeks [ref+1, ref+2, ref+3].
  flat     : 2 predicted weeks [ref+1, ref+3].
Two finetune methods (the loss recipe):
  original : completion loss with the operator (Direction/Size) lines up-weighted.
  scale    : SCALe scheduled per-segment curriculum loss (scale_tags + scale_cot).

Data modes (--data-mode, mirroring the generation pipeline):
  influenza : corpora under <input-root>/influenza/... ; states are geo codes; unchanged behavior.
  finance_stat : corpora under <input-root>/finance_stat/... ; states are tickers (AAPL,GOOGL,NVDA);
              BASELINE = the statistical stat_pred_3w (12-wk mean);
              mode must be chronos2_full. The corpus is TRAIN-only, so val/test EVALUATION
              examples are built OFFLINE from the organized ticker CSVs with the generation
              pipeline's own block renderer (identical prompts; no reference CoT — val is then
              inference-eval only, not a training eval set). Values are the normalized price
              index; the rough MAE/MSE is computed in ORIGINAL USD by back-transforming the
              baseline via each ticker's `price_scale` and scoring against `future_3w_orig`.
  covid_TSFM : corpora under <input-root>/covid_TSFM/... ; states GA,NJ,NY; same method as
              influenza but mode must be chronos2_full and the values are the normalized
              COVID ED-visit index (100 x per-year z-score + 300, 2dp; see the datasets'
              SHIFT300_MANIFEST.md). Unlike the other chronos2_full modes the no-change
              band is ABSOLUTE — ±20 index points (NOCHANGE_ABS_INDEX_BAND) — and the
              corpus bins are |residual| quantile classes in index points (bins.json
              kind = absolute_index); the operator's δ is the bin lower edge in points.
              The finetune prompt lives in <prompt-dir>/chronos2_full_abs/. Only TWO
              weeks are predicted — origin+2 and origin+3 (horizon 2; origin+1 is dropped
              by cot_common.subset_samples_to_offsets; the per-week denorm vectors are
              sliced to match). Legacy relative-band covid corpora (kind = relative)
              still evaluate correctly.
              Like finance the corpus is TRAIN-only, so val/test
              evaluation examples are built OFFLINE from the covid dataset CSVs. The rough
              MAE/MSE is computed on the ORIGINAL scale (weekly proportion of ED visits):
              the operator is applied in normalized units, then the predicted result is
              back-transformed via ((v-300)/100)*sigma+mu (per-week denorm columns) and
              scored against the exactly-restored ground truth.
  covid_TSFM_bitong : the TX,VT,NJ NHSN z26w setting with the EXPANDING-WINDOW HOLD-OUT
              FOLDS of dataset_bitong (TEXT_REPORT_QUALITY_SUMMARY.pdf §7 /
              holdout_splits_VT_TX_NJ.json) instead of one fixed train/val/test split.
              Same units, horizon (origin+3 only), band (±20 index points), prompt
              (chronos2_full_abs_t3) and corpus format as covid_TSFM_updated; the corpus
              itself is the covid_TSFM_updated/<mode>/<STATES>-<corpus-model>/ run (all
              rows split=train, origins 2023-05-06 .. 2025-09-06 = the FINAL training
              window). For every fold F in --folds (default A,B,C):
                * the corpus is filtered to the rows whose (state, reference_date) is a
                  TRAIN row of that fold's view of data/COVID/<STATE>.csv (rows in that fold's
                  3-week buffer / validation / test blocks are dropped — the same origin
                  filter applies to the aug clones, which share the origin's date);
                * a LoRA is trained on that subset with EVERY epoch's checkpoint kept; each
                  checkpoint is scored on the fold's 20-week VALIDATION block (built offline
                  from the fold CSVs, as for covid_TSFM) and the best epoch by --epoch-metric
                  (default val_operator_accuracy; ties -> better MAE improvement, then the
                  earlier epoch) becomes the fold's model and its `best_epoch`
                  (<fold>/epoch_selection.{json,csv}; --no-epoch-select = last epoch only).
                  Only epochs > --epoch-skip (default 3) every --epoch-stride (default 2)
                  are scored — with the default 12-epoch cap that is epochs 4,6,8,10,12;
                  the test block is NOT scored on fold models (it is never used for tuning);
                * --folds final trains on the full corpus (the `split` column) with the SAME
                  epoch cap and schedule as the folds (so the SCALe loss weights and the
                  cosine learning rate sit at the same position at every epoch), keeps
                  every epoch checkpoint and takes the one at epoch
                  round(mean of the three folds' best epochs) (`final_epochs` in
                  folds_summary.json; --final-epochs N overrides) as THE final model, which
                  is scored on the test block once — run it for the configuration that
                  wins the folds. (Training only N epochs would compress the whole
                  curriculum and LR decay into N epochs — a different recipe from the
                  validated fold checkpoint.)
              Fold results land in <run>/fold_A|fold_B|fold_C|final/ and the run's
              <run>/folds_summary.json holds the per-fold validation metrics plus their
              MEAN over the folds present (overall and per state — NJ has text for only
              2/20 val-C weeks, so per-state numbers matter). Every run also refreshes
              <combo>/<method>/fold_cv_summary.csv, one row per configuration ranked by
              the mean validation --rank-by metric (default val_operator_accuracy), which
              is how the best configuration is picked; --summarize-only rebuilds that CSV
              without training. Default --max-seq-len is 24576 here (44 NJ prompts exceed
              16,384 Qwen3 tokens). --data-dir is the fold ROOT (default
              data/COVID, the 19-column master CSVs; bitong_fold_dir derives the fold views).
              Speed defaults specific to this mode (each is a plain flag): epoch cap
              --scale-epochs 12 (past runs peak on validation around epoch 6 at lr 1e-4 and
              memorise from ~15 on), training --batch-size 1 x --grad-accum 16 (batch 2 OOMs
              on 80 GB at the ~16-20k-token NJ prompts), and BATCHED validation/test
              generation --infer-batch-size 8
              (swift infer --max_batch_size). A non-default epoch cap is appended to the
              run dir name (_ep<N>) so caps never collide.
  covid_TSFM_bitong_org : the same TX,VT,NJ corpus, fold CSVs, units, prompt and speed
              defaults as covid_TSFM_bitong, but ONE fixed split and the plain
              covid_TSFM_updated-style single finetune: TRAIN = fold B's training origins
              (2023-05-06 .. 2024-11-30), VAL = fold B's + fold C's validation blocks
              (40 weeks/state, 2025-01 .. 2025-10, minus the text gate), TEST = the final
              test block (26 weeks/state). Every epoch checkpoint (skip/stride as above) is
              scored on VAL, the best epoch (--epoch-metric) becomes THE model and is scored
              on TEST (+ zero-shot). One run per weight/lr setting; metrics.json carries
              best_epoch, the val metrics of that epoch and the test metrics, and
              <combo>/<method>/org_summary.csv lists every setting run so far ranked by
              the validation --rank-by (--summarize-only rebuilds it).

Input  : output/CoT_generation_updated/<data_mode>/<mode>/<state_combo>/accepted.jsonl  (one operator row per
         date). Newer corpora are SELF-CONTAINED: each row embeds the input blocks, the full
         Step-1 basis CoT, the winning Step-2 size CoT, and the bin definitions. For older
         corpora the blocks / size CoT / bins are joined from step1_filtered.jsonl,
         step2_generated.jsonl, and bins.json as a fallback. ABSTAIN (no-correction) dates ARE
         included: their target is "no change" / "none" (keep the baseline) with a canned
         reasoning.
Output : output/CoT_generation_updated_finetune/<data_mode>/<mode>/<state_combo>/<method>/<run>/  with
         train/val/test.jsonl, splits.json, the loss plugin, the swift adapter, swift predictions,
         predictions.csv, and metrics.json (direction / bin accuracies).

Class weighting (--class-weights move:no_change:abstain, default 1:1:1): applied by
OVERSAMPLING train.jsonl — each training example is written repeat[class] times (2:1:1 =
every move example twice; 2:1:0.5 -> repeats 4/2/1 so nothing is dropped). Equivalent to
per-class loss weights; val/test, loss plugins, prompts and evaluation are untouched. Non-
neutral weights get their own run dir suffix (_cw2-1-1).

Run:
  python finetune.py --mode chronos2 --method original --states GA,NY,NJ
  python finetune.py --mode flat     --method scale    --states GA,NY,NJ --lr 1e-5 --epochs 20
  python finetune.py --data-mode covid_TSFM --mode chronos2_full --method scale --class-weights 2:1:1
  python finetune.py --data-mode covid_TSFM_bitong --mode chronos2_full --method scale \
         --corpus-model gemini-aug2 --folds A,B,C,final --lr 1e-4 --scale-schedule "4:1>0.5:2"
         # ONE hyper-parameter setting, ONE job: folds A,B,C (best epoch each) -> final trained
         # for round(mean best epochs) -> test; fold_cv_summary.csv ranks every setting run so far
  python finetune.py --data-mode covid_TSFM_bitong --mode chronos2_full --method scale \
         --corpus-model gemini-aug2 --summarize-only         # just rebuild that ranking
  python finetune.py --data-mode covid_TSFM_bitong_org --mode chronos2_full --method scale \
         --corpus-model gemini-aug2 --lr 1e-4 --scale-schedule "4:1>0.5:2"
         # fixed split: train = fold B train, val = val B + val C, test = final test;
         # best epoch on val -> test; org_summary.csv ranks every setting run so far
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_CODE_ROOT = Path(__file__).resolve().parents[1]          # <paper-content>/code
for _d in ("cot_generation", "finetune", "zero_shot"):
    sys.path.insert(0, str(_CODE_ROOT / _d))
sys.path.insert(0, str(_CODE_ROOT))                         # paper_paths.py
import paper_paths as P                                     # noqa: E402

# ══════════════════════════════════════════════════════════════════════════════
# Defaults (every value has a matching CLI flag)
# ══════════════════════════════════════════════════════════════════════════════
QWEN_MODEL_PATH = P.QWEN_MODEL_PATH          # <PAPER_MODEL_ROOT>/Qwen3-8B
INPUT_ROOT  = P.COT_OUTPUT_DIR               # the CoT corpora (code/cot_generation output)
OUTPUT_ROOT = P.FINETUNE_OUTPUT_ROOT
TARGET_HOSP_CSV = P.TARGET_HOSP_CSV
PROMPT_DIR = P.PROMPT_DIR_INFLUENZA   # <prompt_dir>/finetune_prompt.txt; covid modes -> prompts/covid (parse_args)

MODES = ("chronos2", "flat", "chronos2_full")
METHODS = ("original", "scale")
HORIZON = {"chronos2": 3, "flat": 2, "chronos2_full": 3}
DATA_MODES = ("influenza", "finance_stat", "finance_TSFM", "covid_TSFM", "covid_TSFM_updated",
              "covid_TSFM_bitong", "covid_TSFM_bitong_org")
INFLUENZA_MODES = ("influenza",)
# "no residuals": the COVID finetune prompt (prompts/covid/finetune_prompt.txt) shows NO recent-error
# item — history, baseline, reports only — as in the reported COVID run; the residual block is still
# loaded (used by the offline evaluation) but never rendered into a COVID prompt
NORES_MODES = ("covid_TSFM_updated", "covid_TSFM_bitong", "covid_TSFM_bitong_org")
COVID_MODES = ("covid_TSFM", "covid_TSFM_updated", "covid_TSFM_bitong", "covid_TSFM_bitong_org")
# the z26w (trailing-26-week z-score, origin+3 only) covid modes: identical units/prompt/horizon
COVID_Z26W_MODES = ("covid_TSFM_updated", "covid_TSFM_bitong", "covid_TSFM_bitong_org")
# the TX/VT/NJ COVID modes: same corpus, fold views and speed defaults
BITONG_MODES = ("covid_TSFM_bitong", "covid_TSFM_bitong_org")
BITONG_NORES_MODES = ()
BITONG_CV_MODES = ("covid_TSFM_bitong",)          # expanding folds A, B, C + final
BITONG_ORG_MODES = ("covid_TSFM_bitong_org",)     # one fixed split
FINANCE_MODES = ("finance_stat", "finance_TSFM")   # both read the organized ticker CSVs
DATA_MODE_STATES = {"influenza": "GA,NY,NJ", "finance_stat": "AAPL,GOOGL,NVDA",
                    "finance_TSFM": "AAPL,GOOGL,NVDA", "covid_TSFM": "GA,NJ,NY",
                    "covid_TSFM_updated": "GA,FL,MS", "covid_TSFM_bitong": "TX,VT,NJ",
                    "covid_TSFM_bitong_org": "TX,VT,NJ"}
# --text-stub: the report block every prompt (train/val/test) gets instead of the reports —
# the no-text control arm (same stub as finetune_residual_cot.py / inference_text_ablation.py)
NOTEXT_STUB = "(no weekly reports available for this state)"
TEXT_STUB: Optional[str] = None     # set from --text-stub in main(); read by build_user_prompt
# --reasoning-style: what the assistant TARGET's reasoning section is (train + val targets):
#   cot      = the generated Step-1/Step-2 CoT (default, all existing runs)
#   residual = the example's recent_residual_block verbatim (numbers-only supervision; the
#              finetune_residual_cot.py ablation, now available inside every driver)
#   none     = no Reasoning section at all (Direction/Size lines only)
REASONING_STYLES = ("cot", "residual", "none")
REASONING_STYLE = "cot"             # set from --reasoning-style in main(); read by build_target_text
# val/test evaluation states, when different from the TRAIN states: covid_TSFM_updated
# trains on the full southern cluster but evaluates only on GA+FL (MS text coverage is
# too sparse in val/test for a like-for-like split — see the split-design analysis).
DATA_MODE_EVAL_STATES = {"covid_TSFM_updated": "GA,FL"}
FINANCE_ORG_DIR = P.DATA_ROOT / "Finance" / "organized"      # not part of the paper release
COVID_DATA_DIR = P.DATA_ROOT / "COVID_legacy"                # the old ED-visit index data (not released)
# The covid dataset CSVs carry a +300 index shift (future_3w: +3.0 in z units) so the
# relative band/bins never hit a near-zero baseline — see datasets/SHIFT300_MANIFEST.md.
# Working values are 100*z + 300; restore originals with ((v - COVID_SHIFT)/100)*sigma + mu.
COVID_SHIFT = 300.0
# covid_TSFM_updated (NHSN z26w, dataset_updated): plain 100*z, NO shift, ONLY origin+3.
COVID_UPDATED_DATA_DIR = P.COVID_DATA_ROOT                    # data/COVID/{STATE}.csv (19 columns)
# covid_TSFM_bitong: expanding-window hold-out folds (dataset_bitong, TEXT_REPORT_QUALITY_SUMMARY.pdf
# §7). <fold root>/foldA|foldB|foldC|final/<STATE>.csv carry that fold's split column
# (train / gap / val / test); the CoT corpus (all rows split=train) is filtered per fold to
# the fold's TRAIN origins, validation examples come from the fold's val block.
BITONG_FOLD_ROOT = P.COVID_DATA_ROOT                          # data/COVID/{STATE}.csv; fold views derived by bitong_fold_dir()
BITONG_FOLD_DIRS = {"A": "foldA", "B": "foldB", "C": "foldC", "final": "final"}
BITONG_FOLD_COLUMN = {"A": "fold_A", "B": "fold_B", "C": "fold_C", "final": "split"}   # master-CSV column per fold
# the 13 pipeline columns of a fold view (cot_common.load_covid_bundle / load_covid_orig read them)
COVID_PIPE_COLUMNS = ["state", "reference_date", "hist_12w", "future_3w", "weekly_text",
                      "chronos2_pred_3w", "stat_pred_3w", "split", "population",
                      "denorm_mean_hist_12w", "denorm_std_hist_12w",
                      "denorm_mean_future_3w", "denorm_std_future_3w"]
BITONG_CV_FOLDS = ("A", "B", "C")            # the folds averaged for configuration selection
BITONG_DEFAULT_FOLDS = "A,B,C"
BITONG_CORPUS_DATA_MODE = "covid_TSFM_updated"   # the TX/VT/NJ corpus was generated under this mode
BITONG_MAX_SEQ_LEN = 24576                       # 44 NJ prompts exceed 16,384 tokens; none 22,000
BITONG_RANK_BY_DEFAULT = "val_operator_accuracy"
# metrics averaged over folds (val = the fold's validation block); MAE-type keys rank ascending
BITONG_FOLD_METRICS = ("val_n", "val_n_parsed", "val_direction_accuracy",
                       "val_bin_exact_accuracy_moves", "val_operator_accuracy",
                       "val_rough_MAE_counts", "val_baseline_MAE_counts",
                       "val_improvement_MAE_vs_baseline")
BITONG_LOWER_IS_BETTER = ("val_rough_MAE_counts",)
BITONG_EPOCH_METRIC_DEFAULT = "val_operator_accuracy"   # picks each fold's best epoch
# covid_TSFM_bitong speed defaults (all overridable by the matching CLI flag)
BITONG_EPOCHS = 12                     # epoch cap: val optimum ~6 at lr 1e-4, memorisation from ~15
# batch 2 OOMs even on an 80 GB A100 (the fp32 full-vocab logits of the cross-entropy on a
# 2 x ~16k-token batch alone need ~19 GB on top of ~66 GB) -> 1 x 16, as the other modes
BITONG_BATCH_SIZE, BITONG_GRAD_ACCUM = 1, 16
BITONG_INFER_BATCH_SIZE = 8            # batched generation for the val/test passes
BITONG_EPOCH_SKIP, BITONG_EPOCH_STRIDE = 3, 2   # score epochs 4, 6, 8, ... only
# covid_TSFM_bitong_org: the one fixed split (train fold / val folds / test fold)
BITONG_ORG_TRAIN_FOLD = "B"
BITONG_ORG_VAL_FOLDS = ("B", "C")
BITONG_ORG_TEST_FOLD = "final"
VAL_KEYS = ("n", "n_parsed", "direction_accuracy", "bin_exact_accuracy_moves",
            "operator_accuracy", "rough_MAE_counts", "baseline_MAE_counts",
            "improvement_MAE_vs_baseline")


def covid_shift_for(data_mode: str) -> float:
    return COVID_SHIFT if data_mode == "covid_TSFM" else 0.0


def covid_dir_for(data_mode: str) -> Path:
    if data_mode == "covid_TSFM":
        return COVID_DATA_DIR
    if data_mode in BITONG_MODES:
        return bitong_fold_dir(BITONG_FOLD_ROOT, "final")
    return COVID_UPDATED_DATA_DIR


def pipeline_data_mode(data_mode: str) -> str:
    """The data-mode name the GENERATION pipeline (cot_common / step1_generate) knows:
    covid_TSFM_bitong reuses the covid_TSFM_updated loaders, offsets, units and prompts."""
    return BITONG_CORPUS_DATA_MODE if data_mode in BITONG_MODES else data_mode
HISTORY_LEN = 12
NOCHANGE_BAND = 0.01                       # |gt−baseline| ≤ ε ⇒ true direction = no-change / bin none
                                           # (ABSOLUTE rate units — chronos2 / flat)
NOCHANGE_REL_BAND = 0.10                    # chronos2_full: RELATIVE band — |gt−baseline|/|baseline| ≤ 10%
NOCHANGE_ABS_INDEX_BAND = 20.0              # covid_TSFM: ABSOLUTE band in index points (bins kind absolute_index)
BINS_KIND_ABS_INDEX = "absolute_index"
DEFAULT_BIN_NAMES = ["small", "medium", "large", "very-large"]

# LoRA / optimisation (all CLI-overridable)
LORA_RANK, LORA_ALPHA, LORA_DROPOUT = 8, 16, 0.05
LR = 2e-5
EPOCHS = 15
SCALE_EPOCHS = 20
BATCH_SIZE, GRAD_ACCUM = 1, 16
MAX_SEQ_LEN, MAX_NEW_TOKENS, SEED = 16384, 1536, 42

# original-method loss weights (operator lines up-weighted vs reasoning)
REASONING_LOSS_WEIGHT, OPERATOR_LOSS_WEIGHT = 1.0, 8.0
# scale-method schedule: (w_think, w_answer) START -> END, cosine
SCALE_START, SCALE_END = (1.0, 1.0), (0.5, 2.0)
SCALE_MODE, SCALE_MIN_THINK, SCALE_OTHER_W = "cosine", 0.05, 1.0
THINK_TAG, ANSWER_TAG = 1.0, 2.0

NO_THINK = "/no_think"
REASONING_TAG = "Reasoning:"
DIRECTION_TAG = "Direction:"               # the operator answer starts here (loss split point)
SIZE_TAG = "Size:"
SYSTEM_PROMPT = ("You are a senior epidemiological time-series analyst. You reason step by step "
                 "from the data, then commit to a categorical direction and size for each week. "
                 + NO_THINK)
SYSTEM_PROMPT_FINANCE = ("You are a senior financial time-series analyst. You reason step by step "
                         "from the data, then commit to a categorical direction and size for each "
                         "predicted week. " + NO_THINK)


def system_prompt_for(data_mode: str) -> str:
    return SYSTEM_PROMPT_FINANCE if data_mode in FINANCE_MODES else SYSTEM_PROMPT

DIR_WORD = {1: "increase", -1: "decrease", 0: "no change"}


def run_dir_name(states: List[str]) -> str:
    return "_".join(s.strip().upper() for s in states if s.strip())


COVID_PRED_OFFSETS = (2, 3)      # covid_TSFM predicts origin+2 and origin+3 only (cot_common)
COVID_UPDATED_PRED_OFFSETS = (3,)   # covid_TSFM_updated: ONLY origin+3 (cot_common)


def covid_offsets_for(data_mode: str):
    return COVID_UPDATED_PRED_OFFSETS if data_mode in COVID_Z26W_MODES else COVID_PRED_OFFSETS


def horizon_for(data_mode: str, mode: str) -> int:
    """Number of predicted weeks: finance_TSFM = 1 (origin+3 only), covid_TSFM = 2
    (origin+2, origin+3), else the mode's full horizon."""
    if data_mode == "finance_TSFM":
        return 1
    if data_mode in COVID_MODES:
        return len(covid_offsets_for(data_mode))
    return HORIZON[mode]


def _covid_slice(seq, horizon: int):
    """The per-week denorm/ground-truth vectors in the covid CSVs cover origin+1..+3;
    keep the trailing `horizon` entries (origin+2, +3 for the 2-week operator)."""
    return list(seq)[len(seq) - horizon:]


# FluSight 2-letter → location code, for the per-10,000 → count population lookup.
STATE_LOC = {
    "US": "US", "USA": "US", "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06",
    "CO": "08", "CT": "09", "DE": "10", "DC": "11", "FL": "12", "GA": "13", "HI": "15",
    "ID": "16", "IL": "17", "IN": "18", "IA": "19", "KS": "20", "KY": "21", "LA": "22",
    "ME": "23", "MD": "24", "MA": "25", "MI": "26", "MN": "27", "MS": "28", "MO": "29",
    "MT": "30", "NE": "31", "NV": "32", "NH": "33", "NJ": "34", "NM": "35", "NY": "36",
    "NC": "37", "ND": "38", "OH": "39", "OK": "40", "OR": "41", "PA": "42", "RI": "44",
    "SC": "45", "SD": "46", "TN": "47", "TX": "48", "UT": "49", "VT": "50", "VA": "51",
    "WA": "53", "WV": "54", "WI": "55", "WY": "56",
}


def rate_to_count_scale(states, csv_path: Path = TARGET_HOSP_CSV) -> Dict[str, float]:
    """{STATE: population/10000} so count = rate * (population/10000). Population is the canonical
    FluSight value (median of value/weekly_rate*1e5) — the exact inverse of the count→rate step."""
    import statistics
    df = pd.read_csv(csv_path, dtype={"location": str})
    df = df[df["weekly_rate"] > 0]
    out: Dict[str, float] = {}
    for raw in {s.strip().upper() for s in states}:
        loc = STATE_LOC.get(raw)
        if loc is None:
            sys.exit(f"no FluSight location code for geo {raw!r}")
        sub = df[df["location"] == loc]
        if sub.empty:
            sys.exit(f"no population row for geo {raw!r} (loc {loc!r})")
        pop = statistics.median((sub["value"] / sub["weekly_rate"] * 1e5).tolist())
        out[raw] = float(pop) / 10000.0
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Bins: the size-class definitions the operator maps to (from the run's bins.json)
# ══════════════════════════════════════════════════════════════════════════════
def load_bins(run_dir: Path) -> Optional[Dict]:
    p = run_dir / "bins.json"
    if p.exists():
        return json.loads(p.read_text())
    # fallback: newer corpora embed the run bins in every accepted.jsonl row
    acc = run_dir / "accepted.jsonl"
    if acc.exists():
        with acc.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    return json.loads(line).get("bins")
    return None


def bin_names(bins: Optional[Dict]) -> List[str]:
    return bins["names"] if bins else list(DEFAULT_BIN_NAMES)


def bin_of(absr: float, bins: Optional[Dict]) -> Optional[int]:
    """The size-bin index |r| falls into (highest lower-edge ≤ |r|); None if bins missing."""
    if bins is None:
        return None
    edges = bins["lower_edges"]
    idx = 0
    for i, e in enumerate(edges):
        if absr >= e:
            idx = i
        else:
            break
    return idx


def bins_are_relative(bins: Optional[Dict]) -> bool:
    """chronos2_full bins are RELATIVE: edges/m_init are FRACTIONS of the baseline, and the
    residual is classified as |gt−baseline|/|baseline| (not the absolute rate residual)."""
    return bool(bins) and bins.get("kind") == "relative"


# ══════════════════════════════════════════════════════════════════════════════
# Data: read accepted.jsonl (operator per date) + join blocks + include abstain
# ══════════════════════════════════════════════════════════════════════════════
ABSTAIN_REASONING = (
    "The available CDC FluView surveillance text and the recent forecast-error pattern do not "
    "provide clear, consistent evidence for which way the baseline should be revised over the "
    "predicted weeks. Rather than over-read weak or ambiguous signals, the baseline is treated as "
    "already the best available forecast and is kept unchanged (no change, no size).")
# finance modes must not mention FluView; finance_TSFM has a single target week (origin+3)
ABSTAIN_REASONING_BY_MODE = {
    "influenza": ABSTAIN_REASONING,
    "finance_stat": (
        "The available financial news reports and the recent forecast-error pattern do not "
        "provide clear, consistent evidence for which way the baseline should be revised over "
        "the predicted weeks. Rather than over-read weak or ambiguous signals, the baseline is "
        "treated as already the best available forecast and is kept unchanged (no change, no size)."),
    "finance_TSFM": (
        "The available financial news reports and the recent forecast-error pattern do not "
        "provide clear, consistent evidence for which way the baseline should be revised for "
        "predicted week 3 (origin + 3). Rather than over-read weak or ambiguous signals, the "
        "baseline is treated as already the best available forecast and is kept unchanged "
        "(no change, no size)."),
    "covid_TSFM": (
        "The available state respiratory surveillance text and the recent forecast-error "
        "pattern do not provide clear, consistent evidence for which way the baseline should "
        "be revised for predicted weeks 2 and 3 (origin + 2 and origin + 3). Rather than "
        "over-read weak or ambiguous signals, the baseline is treated as already the best "
        "available forecast and is kept unchanged (no change, no size)."),
    "covid_TSFM_updated": (
        "The available state respiratory surveillance text and the recent forecast-error "
        "pattern do not provide clear, consistent evidence for which way the baseline should "
        "be revised for predicted week 3 (origin + 3). Rather than over-read weak or "
        "ambiguous signals, the baseline is treated as already the best available forecast "
        "and is kept unchanged (no change, no size)."),
}
# the COVID prompt shows no recent-error block, so the canned fallback must not mention it
# (the released -gemini-final corpus carries an input-specific abstain_cot anyway)
ABSTAIN_REASONING_BY_MODE["covid_TSFM_updated"] = (
    "The available state respiratory surveillance text does not provide clear, consistent "
    "evidence for which way the baseline should be revised for predicted week 3 (origin + 3). "
    "Rather than over-read weak or ambiguous signals, the baseline is treated as already the "
    "best available forecast and is kept unchanged (no change, no size).")
ABSTAIN_REASONING_BY_MODE["covid_TSFM_bitong"] = ABSTAIN_REASONING_BY_MODE["covid_TSFM_updated"]
ABSTAIN_REASONING_BY_MODE["covid_TSFM_bitong_org"] = ABSTAIN_REASONING_BY_MODE["covid_TSFM_updated"]

_TRAILING_TAGS_RE = re.compile(r"(?is)\s*(?:Directions|Sizes)\s*:[^\n]*\s*$")
_BLOCK_KEYS = ("history_block", "baseline_block", "recent_residual_block",
               "weekly_text_block")


def _clean_cot(cot: str) -> str:
    cot = (cot or "").strip()
    prev = None
    while cot and cot != prev:
        prev = cot
        cot = _TRAILING_TAGS_RE.sub("", cot).strip()
    return cot


def _fit(seq, horizon, fill):
    seq = list(seq or [])
    return (seq + [fill] * horizon)[:horizon]


# ── Class weighting by OVERSAMPLING the training set ──────────────────────────
# --class-weights move:no_change:abstain (e.g. 2:1:1). Implemented at the DATA level: each
# training example is written into train.jsonl `repeat[class]` times (identical copies,
# shuffled), which is equivalent to multiplying that class's loss by its weight. Fractional
# weights are scaled to whole repeat counts (2:1:0.5 -> 4/2/1) so no example is dropped.
# Val/test are never expanded. Loss plugins, prompts and evaluation are untouched.
CLASS_ORDER = ("move", "no_change", "abstain")
CLASS_WEIGHTS_DEFAULT = "1:1:1"


def parse_class_weights(spec: str) -> Dict[str, float]:
    parts = [p.strip() for p in str(spec).split(":")]
    if len(parts) != 3:
        sys.exit(f"--class-weights must be move:no_change:abstain, got {spec!r}")
    try:
        w = {k: float(v) for k, v in zip(CLASS_ORDER, parts)}
    except ValueError:
        sys.exit(f"--class-weights values must be numbers, got {spec!r}")
    if any(v <= 0 for v in w.values()):
        sys.exit(f"--class-weights values must be > 0, got {spec!r}")
    return w


def class_repeats(weights: Dict[str, float]) -> Dict[str, int]:
    """Whole-number repeat counts with the same ratios: divide by the smallest weight and
    round to the nearest integer (2:1:0.5 -> 4:2:1; 1.5:1:1 -> 3:2:2 after doubling)."""
    lo = min(weights.values())
    rep = {k: v / lo for k, v in weights.items()}
    # scale up (x2, x4) while any ratio is still non-integral, to keep exact ratios
    for mult in (1, 2, 4):
        cand = {k: round(v * mult, 6) for k, v in rep.items()}
        if all(abs(v - round(v)) < 1e-6 for v in cand.values()):
            return {k: int(round(v)) for k, v in cand.items()}
    return {k: max(1, int(round(v))) for k, v in rep.items()}


def class_weights_tag(weights: Dict[str, float]) -> str:
    """Run-dir suffix, e.g. 'cw2-1-0.5'; '' for the neutral 1:1:1."""
    if all(abs(v - 1.0) < 1e-9 for v in weights.values()):
        return ""
    return "cw" + "-".join(f"{weights[k]:g}" for k in CLASS_ORDER)


def expand_train(train: List[Dict], repeats: Dict[str, int], seed: int) -> List[Dict]:
    """Oversample the training examples by class and shuffle (seeded) so duplicate copies
    are not adjacent in train.jsonl. Examples without a class (offline eval rows) count 1."""
    out = [ex for ex in train for _ in range(repeats.get(ex.get("cls") or "", 1))]
    random.Random(seed).shuffle(out)
    return out


def _blocks_by_id(run_dir: Path) -> Dict[str, Dict]:
    """Fallback join for OLDER corpora whose accepted.jsonl rows do not embed the input
    blocks. Newer corpora are self-contained and never need this map."""
    flt = run_dir / "step1_filtered.jsonl"
    if not flt.exists():
        return {}
    out: Dict[str, Dict] = {}
    with flt.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if int(r.get("candidate_index", 0)) != -1:
                continue
            out[r["id"]] = {k: r.get(k, "") for k in _BLOCK_KEYS}
    return out


def _size_cot_by_key(run_dir: Path) -> Dict[Tuple[str, int], str]:
    """The Step-2 magnitude CoT text per (id, candidate_index), from step2_generated.jsonl.
    Used to fold the size reasoning of the WINNING size candidate into a move date's target
    reasoning, so the model learns direction AND magnitude from the CoT."""
    p = run_dir / "step2_generated.jsonl"
    out: Dict[Tuple[str, int], str] = {}
    if not p.exists():
        return out
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[(r.get("id"), int(r.get("candidate_index", -1)))] = r.get("cot", "")
    return out


def load_examples(run_dir: Path, horizon: int, bins: Optional[Dict],
                  keep_states: Optional[set] = None,
                  data_mode: str = "influenza") -> Dict[str, List[Dict]]:
    """Each example carries the operator TARGET (per-week direction word + size word) and the
    rate arrays (for the true direction/bin at eval). correction → basis operator; no-change and
    abstain → 'no change' / 'none'. Train/val need a reasoning CoT (abstain uses a canned one)."""
    acc = run_dir / "accepted.jsonl"
    if not acc.exists():
        raise FileNotFoundError(f"corpus not found: {acc} (run the pipeline for this mode/states)")
    blocks = _blocks_by_id(run_dir)
    size_cots = _size_cot_by_key(run_dir)      # winning Step-2 size CoT, folded into move reasoning
    names = bin_names(bins)
    by_split: Dict[str, List[Dict]] = defaultdict(list)
    n_total = n_skip = n_abstain = n_missing = n_abstain_reasoned = 0
    with acc.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            n_total += 1
            if keep_states is not None and r.get("state") not in keep_states:
                continue
            split = r.get("split")
            x_hist, x_hat, x_gt = r.get("x_hist"), r.get("x_hat"), r.get("x_gt")
            ok = (split in ("train", "val", "test")
                  and isinstance(x_hist, list) and len(x_hist) == HISTORY_LEN
                  and isinstance(x_hat, list) and len(x_hat) == horizon
                  and isinstance(x_gt, list) and len(x_gt) == horizon)
            if not ok:
                n_skip += 1
                continue
            op = r.get("operator")
            if op == "no-correction" or r.get("date_status") == "abstain":
                cls = "abstain"
                dir_target = [0] * horizon
                size_target = ["none"] * horizon
                # abstain-fix corpora (build_abstain_fix_corpus.py) embed a real, input-
                # specific reasoning per abstain row; the canned text is only the fallback
                cot = _clean_cot(r.get("abstain_cot") or "")
                if cot:
                    n_abstain_reasoned += 1
                else:
                    cot = ABSTAIN_REASONING_BY_MODE.get(data_mode, ABSTAIN_REASONING)
                n_abstain += 1
            else:
                cls = "move" if op == "correction" else "no_change"
                # reasoning = the Step-1 DIRECTION CoT, plus (for moves) the winning Step-2
                # MAGNITUDE CoT — so the model learns direction AND magnitude from the CoT.
                cot = _clean_cot(r.get("step1_basis_cot") or "")
                if op == "correction":
                    # prefer the size CoT embedded in accepted.jsonl (self-contained corpora);
                    # fall back to the step2_generated.jsonl join for older corpora
                    scot = _clean_cot(r.get("step2_size_cot")
                                      or size_cots.get((r.get("id"),
                                                        int(r.get("winning_candidate_index", -1))), ""))
                    if scot:
                        cot = (cot + "\n\n" + scot).strip() if cot else scot
                if split in ("train", "val") and not cot:
                    n_skip += 1
                    continue
                dir_target = _fit(r.get("basis_directions"), horizon, 0)
                sn = r.get("size_names")
                if op == "correction" and sn:
                    size_target = _fit([s.strip() for s in sn.split(",")], horizon, "none")
                else:                                   # no-change (or missing sizes)
                    size_target = ["none"] * horizon
            # input blocks: prefer the copies embedded in accepted.jsonl (self-contained
            # corpora); fall back to the step1_filtered.jsonl join for older corpora
            blk = {k: r.get(k) for k in _BLOCK_KEYS}
            if not all(blk.values()):
                blk = blocks.get(r.get("id"))
                if blk is None:
                    n_missing += 1
                    blk = {k: "" for k in _BLOCK_KEYS}
            by_split[split].append({
                "id": r.get("id"), "state": r.get("state"), "split": split,
                "reference_date": r.get("reference_date"),
                "baseline": [float(v) for v in x_hat], "target": [float(v) for v in x_gt],
                # full-precision arrays for the count-based rough MAE/MSE
                "baseline_full": [float(v) for v in (r.get("x_hat_full") or x_hat)],
                "target_full": [float(v) for v in (r.get("x_gt_full") or x_gt)],
                "dir_target": dir_target, "size_target": size_target, "cot": cot,
                "operator": op, "cls": cls, **blk,
            })
    for split in by_split:
        by_split[split].sort(key=lambda e: (e.get("state") or "", e["reference_date"] or "", e["id"] or ""))
    msg = (f"Loaded {n_total} dates ({n_skip} skipped; {n_abstain} abstain as no-change/none"
           + (f", {n_abstain_reasoned} with embedded reasoning" if n_abstain_reasoned else "")
           + ")")
    if n_missing:
        msg += f"; WARNING {n_missing} missing blocks"
    print(msg + "; splits: " + ", ".join(f"{k}={len(v)}" for k, v in by_split.items())
          + f"; size bins: {names}")
    return by_split


# ══════════════════════════════════════════════════════════════════════════════
# Prompt / target (reason, then a Direction line and a Size line)
# ══════════════════════════════════════════════════════════════════════════════
# Default finetune USER-prompt templates (preserved to prompts/CoT_generation_updated/<mode>/
# finetune_prompt.txt on first run; edit that file to change the prompt). Neither mode uses a
# momentum block. Placeholders: {intro} {history_header} {baseline_header} {reports_header}
# (domain wording, filled per data-mode from PROMPT_WORDING) plus {history_len} {horizon}
# {history_block} {baseline_block} {recent_residual_block} {weekly_text_block} {bin_names}.
# influenza and finance share the SAME template file (both run mode chronos2_full); only the
# wording table differs between them.
_TASK = (
    "TASK: Reason step by step, then for EACH predicted week decide (a) the DIRECTION the baseline "
    "should be revised, and (b) the SIZE of that revision (a magnitude class).\n\n"
    "OUTPUT FORMAT — return EXACTLY three parts:\n"
    f"  {REASONING_TAG} <one flowing reasoning paragraph>\n\n"
    f"  {DIRECTION_TAG} {{horizon}} comma-separated values, each one of: increase | decrease | no change\n"
    f"  {SIZE_TAG} {{horizon}} comma-separated values, each one of: {{bin_names}} | none  "
    "(use \"none\" for exactly the weeks whose Direction is \"no change\").")
# One SHARED template body per mode; the domain wording (intro + the three item headers)
# lives in placeholders, filled per data-mode from PROMPT_WORDING below. influenza and
# finance therefore use the SAME chronos2_full template file.
DEFAULT_FINETUNE_PROMPT = {
    "chronos2": (
        "{intro}\n\n"
        "1. {history_header}\n{history_block}\n\n"
        "2. {baseline_header}\n{baseline_block}\n\n"
        "3. Recent residuals / changes over the recent observed weeks (sign: + = the baseline was "
        "too low / the rate rose):\n{recent_residual_block}\n\n"
        "4. {reports_header}\n{weekly_text_block}\n\n" + _TASK),
    "flat": (
        "{intro}\n\n"
        "1. {history_header}\n{history_block}\n\n"
        "2. {baseline_header}\n{baseline_block}\n\n"
        "3. Recent residuals / changes over the recent observed weeks (sign: + = the baseline was "
        "too low / the rate rose):\n{recent_residual_block}\n\n"
        "4. {reports_header}\n{weekly_text_block}\n\n" + _TASK),
    "chronos2_full": (
        "{intro}\n\n"
        "1. {history_header}\n{history_block}\n\n"
        "2. {baseline_header}\n{baseline_block}\n\n"
        "3. Recent baseline errors at two horizons over the last 3 observed weeks — the RELATIVE "
        "residual, (actual − baseline) as a PERCENTAGE of the baseline (sign: + = baseline too low / "
        "under-predicted), directly comparable to the ±10% no-change band:\n{recent_residual_block}\n\n"
        "4. {reports_header}\n{weekly_text_block}\n\n" + _TASK),
    # covid_TSFM (prompt subdir chronos2_full_abs): ABSOLUTE index-point band / residuals
    # covid z26w modes (prompt file prompts/covid/finetune_prompt.txt): NO recent-error item,
    # the reports are item 3. Fallback only, written when that file is absent.
    "chronos2_full_abs_t3": "{intro}\n\n1. {history_header}\n{history_block}\n\n2. {baseline_header}\n{baseline_block}\n\n3. {reports_header}\n{weekly_text_block}\n\nTASK: Reason step by step, then for the SINGLE predicted week — the far horizon, origin + 3 (the nearer weeks origin+1 and origin+2 are NOT being predicted) — decide (a) the DIRECTION the baseline should be revised, and (b) the SIZE of that revision (a magnitude class). The baseline forecast extrapolates the recent numeric trend of the index. Read the reports first and decide which phase the epidemic is in (rising / near a peak / declining / low and stable), then judge whether the baseline forecast for predicted week 3 is consistent with that phase given the value history. The reports are your primary evidence; the value history is the context they are read against. Only call increase or decrease when the evidence supports a move clearly larger than the ±20-index-point no-change band away from the baseline; otherwise say no change.\n\nOUTPUT FORMAT — return EXACTLY three parts:\n  " + REASONING_TAG + " <one flowing reasoning paragraph>\n\n  " + DIRECTION_TAG + " {horizon} comma-separated values, each one of: increase | decrease | no change\n  " + SIZE_TAG + " {horizon} comma-separated values, each one of: {bin_names} | none  (use \"none\" if and only if the Direction is \"no change\").",
    "chronos2_full_abs": (
        "{intro}\n\n"
        "1. {history_header}\n{history_block}\n\n"
        "2. {baseline_header}\n{baseline_block}\n\n"
        "3. Recent baseline errors at two horizons over the last 3 observed weeks — the ABSOLUTE "
        "residual, (actual − baseline) in index points (sign: + = baseline too low / "
        "under-predicted), directly comparable to the ±20-index-point no-change band:\n"
        "{recent_residual_block}\n\n"
        "4. {reports_header}\n{weekly_text_block}\n\n" + _TASK),
}

# The domain wording per (data_mode, mode). Filled values reproduce the previous
# hard-coded prompts byte-for-byte.
PROMPT_WORDING = {
    ("influenza", "chronos2"): {
        "intro": ("You are given, for one US state at a forecast origin, weekly influenza "
                  "hospital-admission RATE data — admissions per 10,000 residents (2 decimals). "
                  "The reports are date-scrubbed; refer to them only by relative order."),
        "history_header": "{history_len}-week admission-rate history (most recent last):",
        "baseline_header": ("Baseline forecast for the next {horizon} predicted weeks — the "
                            "object being corrected:"),
        "reports_header": "CDC FluView weekly surveillance reports (relative order):",
    },
    ("influenza", "chronos2_full"): {
        "intro": ("You are given, for one US state at a forecast origin, weekly influenza "
                  "hospital-admission COUNT data — whole-number weekly admissions. The reports "
                  "are date-scrubbed; refer to them only by relative order."),
        "history_header": "{history_len}-week admission-count history (most recent last):",
        "baseline_header": ("Baseline forecast (counts) for the next {horizon} predicted weeks "
                            "— the object being corrected:"),
        "reports_header": "CDC FluView weekly surveillance reports (relative order):",
    },
    ("finance_stat", "chronos2_full"): {
        "intro": ("You are given, for one stock at a forecast origin, weekly OPENING-PRICE data "
                  "as a normalized price index (2 decimals; ~1000 = the stock's training-period "
                  "average price). The news reports are date-scrubbed; refer to them only by "
                  "relative order."),
        "history_header": "{history_len}-week opening-price history (most recent last):",
        "baseline_header": ("Baseline forecast (price index) for the next {horizon} predicted "
                            "weeks — the object being corrected:"),
        "reports_header": "Weekly financial news reports (relative order):",
    },
    ("finance_TSFM", "chronos2_full"): {
        "intro": ("You are given, for one stock at a forecast origin, weekly OPENING-PRICE data "
                  "as a normalized price index (2 decimals; ~1000 = the stock's training-period "
                  "average price). The news reports are date-scrubbed; refer to them only by "
                  "relative order."),
        "history_header": "{history_len}-week opening-price history (most recent last):",
        "baseline_header": ("Baseline forecast (price index) for the SINGLE target — "
                            "predicted week 3 (origin + 3; weeks origin+1 and origin+2 are "
                            "NOT predicted) — the object being corrected:"),
        "reports_header": "Weekly financial news reports (relative order):",
    },
    ("covid_TSFM", "chronos2_full"): {
        "intro": ("You are given, for one US state at a forecast origin, weekly COVID-19 "
                  "emergency-department-visit ACTIVITY data as a normalized index (2 "
                  "decimals; 300 = the year's average activity level, values above 300 = "
                  "above-average activity, values below 300 = below-average activity). The "
                  "reports are date-scrubbed; refer to them only by relative order."),
        "history_header": "{history_len}-week activity-index history (most recent last):",
        "baseline_header": ("Baseline forecast (activity index) for the TWO targets — "
                            "predicted week 2 (origin + 2) and predicted week 3 (origin + 3); "
                            "week origin+1 is NOT predicted — the object being corrected:"),
        "reports_header": "State weekly respiratory surveillance reports (relative order):",
    },
    ("covid_TSFM_updated", "chronos2_full"): {
        "intro": ("You are given, for one US state at a forecast origin, weekly COVID-19 "
                  "hospital-admission ACTIVITY data as a normalized index — NOT raw "
                  "admission counts: each week's admission count is standardized against "
                  "the trailing 26-WEEK window ending at that week (z-score = (count − "
                  "26-week mean) / 26-week standard deviation) and multiplied by 100 "
                  "(2 decimals). So 0 = the average level of the trailing half-year, "
                  "+100 / −100 = one standard deviation above / below that recent average; "
                  "negative values are normal and the index has no floor at zero. Every "
                  "number below (history, baseline forecast, the no-change "
                  "band) is in these index units. The reports are date-scrubbed and speak in "
                  "real-world quantities; refer to them only by relative order and use them "
                  "for the direction and strength of the trend, never to match numbers "
                  "against the index."),
        "history_header": "{history_len}-week activity-index history (most recent last):",
        "baseline_header": ("Baseline forecast (activity index) for the SINGLE target — "
                            "predicted week 3 (origin + 3); weeks origin+1 and origin+2 "
                            "are NOT predicted — the object being corrected:"),
        "reports_header": "State weekly respiratory surveillance reports (relative order):",
    },
}
PROMPT_WORDING[("influenza", "flat")] = PROMPT_WORDING[("influenza", "chronos2")]
# covid_TSFM_bitong: same z26w units/horizon as covid_TSFM_updated (same prompt file too)
PROMPT_WORDING[("covid_TSFM_bitong", "chronos2_full")] = PROMPT_WORDING[("covid_TSFM_updated", "chronos2_full")]
PROMPT_WORDING[("covid_TSFM_bitong_org", "chronos2_full")] = PROMPT_WORDING[("covid_TSFM_updated", "chronos2_full")]


def ensure_finetune_prompt(mode: str, prompt_dir: Path, data_mode: str = "influenza") -> str:
    """Preserve the SHARED template body to <prompt_dir>/<mode>/finetune_prompt.txt (writing
    the default if absent), then return its text. Both data modes read the same file; the
    {intro}/{history_header}/{baseline_header}/{reports_header} placeholders are filled per
    data-mode from PROMPT_WORDING at prompt-build time. Edit the file to change the shared
    structure; edit PROMPT_WORDING to change a data-mode's wording."""
    # finance_TSFM: <mode>_t3 (origin+3 only); covid_TSFM: <mode>_abs (absolute index-point
    # band) — same subdir convention as cot_common.prompt_mode()
    subdir = (f"{mode}_t3" if data_mode == "finance_TSFM"
              else f"{mode}_abs_t3" if data_mode in COVID_Z26W_MODES
              else f"{mode}_abs" if data_mode == "covid_TSFM" else mode)
    # the release ships FLAT prompt directories (prompts/influenza, prompts/covid,
    # chosen by paper_paths.prompt_dir_for); a legacy <prompt_dir>/<subdir>/ layout is honoured if present
    d = Path(prompt_dir) / subdir if (Path(prompt_dir) / subdir).is_dir() else Path(prompt_dir)
    d.mkdir(parents=True, exist_ok=True)
    p = d / "finetune_prompt.txt"
    if not p.exists():
        # subdir-specific body if one is defined (chronos2_full_abs), else the base mode's
        p.write_text(DEFAULT_FINETUNE_PROMPT.get(subdir, DEFAULT_FINETUNE_PROMPT[mode]))
        print(f"  preserved finetune prompt template → {p}")
    return p.read_text()


def _fill(template: str, mapping: Dict) -> str:
    out = template
    for k, v in mapping.items():
        out = out.replace("{" + k + "}", str(v))
    return out


def build_user_prompt(ex: Dict, horizon: int, bin_names_str: str, template: str,
                      wording: Dict[str, str]) -> str:
    # wording first: headers may themselves contain {history_len}/{horizon}, which the
    # later entries then resolve (_fill replaces in mapping order)
    return _fill(template, {
        **wording,
        "history_len": HISTORY_LEN, "horizon": horizon,
        "history_block": ex["history_block"], "baseline_block": ex["baseline_block"],
        "recent_residual_block": ex["recent_residual_block"],
        # --text-stub (no-text control arm): every split sees the stub instead of the reports
        "weekly_text_block": TEXT_STUB if TEXT_STUB else ex["weekly_text_block"],
        "bin_names": bin_names_str,
    })


def build_target_text(ex: Dict) -> str:
    dirs = ", ".join(DIR_WORD.get(d, "no change") for d in ex["dir_target"])
    sizes = ", ".join(ex["size_target"])
    answer = f"{DIRECTION_TAG} {dirs}\n{SIZE_TAG} {sizes}"
    if REASONING_STYLE == "none":        # direct-answer ablation: no Reasoning section
        return answer
    if REASONING_STYLE == "residual":    # numbers-only supervision: restate the error block
        reasoning = (ex.get("recent_residual_block") or "").strip()
    else:
        reasoning = ex["cot"]
    return f"{REASONING_TAG} {reasoning}\n\n{answer}"


def write_jsonl(path: Path, examples: List[Dict], horizon: int, bin_names_str: str,
                template: str, wording: Dict[str, str], include_assistant: bool,
                sys_prompt: str = SYSTEM_PROMPT) -> None:
    with path.open("w") as f:
        for ex in examples:
            msgs = [{"role": "system", "content": sys_prompt},
                    {"role": "user", "content": build_user_prompt(ex, horizon, bin_names_str,
                                                                  template, wording)}]
            if include_assistant:
                msgs.append({"role": "assistant", "content": build_target_text(ex)})
            f.write(json.dumps({"messages": msgs}) + "\n")
    print(f"  wrote {len(examples)} examples → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Loss-scale plugins (up-weight the operator lines; split at 'Direction:')
# ══════════════════════════════════════════════════════════════════════════════
def write_original_plugin(path: Path, operator_weight: float, reasoning_weight: float) -> None:
    path.write_text(f'''\
"""Auto-generated (finetune.py, original method). Up-weights the operator (Direction/Size) lines."""
from swift.loss_scale import loss_scale_map, LossScale

DIRECTION_TAG         = {DIRECTION_TAG!r}
REASONING_LOSS_WEIGHT = {reasoning_weight!r}
OPERATOR_LOSS_WEIGHT  = {operator_weight!r}


class OperatorEmphasisLossScale(LossScale):
    is_binary = False

    def get_loss_scale(self, context, **kwargs):
        if isinstance(context, str) and DIRECTION_TAG in context:
            idx = context.index(DIRECTION_TAG)
            head, tail = context[:idx], context[idx:]
            segments, weights = [], []
            if head:
                segments.append(head); weights.append(REASONING_LOSS_WEIGHT)
            segments.append(tail); weights.append(OPERATOR_LOSS_WEIGHT)
            return segments, weights
        return super().get_loss_scale(context, **kwargs)


loss_scale_map['operator_emphasis'] = OperatorEmphasisLossScale
''')
    print(f"  wrote original loss plugin (operator_weight={operator_weight}) → {path}")


def write_scale_plugin(path: Path, start, end, mode: str, min_think: float, other_w: float) -> None:
    path.write_text(f'''\
"""Auto-generated (finetune.py, scale method). SCALe scheduled per-segment curriculum loss."""
import math, torch
from swift.loss_scale import loss_scale_map, LossScale
from swift.loss.base import BaseLoss
from swift.loss.mapping import loss_map
try:
    from swift.trainers import per_token_loss_func
except Exception:
    from swift.trainers.utils import per_token_loss_func

DIRECTION_TAG = {DIRECTION_TAG!r}
THINK_TAG  = {float(THINK_TAG)!r}
ANSWER_TAG = {float(ANSWER_TAG)!r}
START = ({float(start[0])!r}, {float(start[1])!r})
END   = ({float(end[0])!r}, {float(end[1])!r})
MODE  = {mode!r}
MIN_THINK = {float(min_think)!r}
OTHER_W   = {float(other_w)!r}


class ScaleSegmentTagsLossScale(LossScale):
    is_binary = False

    def get_loss_scale(self, context, **kwargs):
        if isinstance(context, str) and DIRECTION_TAG in context:
            idx = context.index(DIRECTION_TAG)
            head, tail = context[:idx], context[idx:]
            segments, weights = [], []
            if head:
                segments.append(head); weights.append(THINK_TAG)
            segments.append(tail); weights.append(ANSWER_TAG)
            return segments, weights
        return super().get_loss_scale(context, **kwargs)


def _interp(p):
    if MODE == "cosine":
        p = 0.5 * (1.0 - math.cos(math.pi * p))
    tw = START[0] + p * (END[0] - START[0])
    aw = START[1] + p * (END[1] - START[1])
    return max(MIN_THINK, tw), aw


class ScaleCotLoss(BaseLoss):
    _log_every = 50

    def __call__(self, outputs, labels, *, num_items_in_batch=None, loss_scale=None, **kwargs):
        ce = per_token_loss_func(outputs, labels)
        B, T = labels.shape[0], labels.shape[1]
        ce = ce.view(B, T)
        if loss_scale is not None:
            ls = loss_scale.view(B, T)
            think_mask  = (ls > 0.5) & (ls < 1.5)
            answer_mask = (ls >= 1.5)
            other_mask  = (ls > 0) & ~think_mask & ~answer_mask
        else:
            active = (torch.roll(labels, shifts=-1, dims=-1) != -100)
            think_mask, answer_mask, other_mask = active, torch.zeros_like(active), torch.zeros_like(active)

        def seg_mean(mask):
            mask = mask.to(ce.dtype)
            cnt = mask.sum(dim=1); summ = (ce * mask).sum(dim=1); has = cnt > 0
            return (summ[has] / cnt[has]).mean() if has.any() else ce.new_zeros(())

        think_loss, answer_loss, other_loss = seg_mean(think_mask), seg_mean(answer_mask), seg_mean(other_mask)
        state = getattr(getattr(self, "trainer", None), "state", None)
        step  = float(getattr(state, "global_step", 0) or 0)
        total = float(getattr(state, "max_steps", 0) or 0) or 1.0
        tw, aw = _interp(min(1.0, max(0.0, step / total)))
        loss = tw * think_loss + aw * answer_loss + OTHER_W * other_loss
        self._n = getattr(self, "_n", 0) + 1
        if self._n % self._log_every == 1:
            print(f"[scale_cot] step={{int(step)}}/{{int(total)}} w_think={{tw:.3f}} w_answer={{aw:.3f}} "
                  f"L={{float(loss):.4f}}", flush=True)
        return loss


loss_scale_map['scale_tags'] = ScaleSegmentTagsLossScale
loss_map['scale_cot'] = ScaleCotLoss
''')
    print(f"  wrote scale loss plugin (start={start} -> end={end}, {mode}) → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# ms-swift wrappers
# ══════════════════════════════════════════════════════════════════════════════
def _check_swift() -> None:
    if shutil.which("swift") is None:
        raise RuntimeError("`swift` CLI not found on PATH. Install with `pip install ms-swift`.")


def _model_type_args(args) -> List[str]:
    """--model_type/--template passthrough (empty for Qwen3 checkpoints, which swift
    auto-detects from the dir name; required for Llama-3.1-8B-Instruct, whose
    LlamaForCausalLM architecture maps to several swift model types)."""
    out: List[str] = []
    if getattr(args, "model_type", None):
        out += ["--model_type", args.model_type]
    if getattr(args, "template", None):
        out += ["--template", args.template]
    return out


def run_swift_sft(args, train_jsonl: Path, swift_out: Path, plugin_path: Path,
                  val_jsonl: Optional[Path]) -> None:
    swift_out.mkdir(parents=True, exist_ok=True)
    epochs = args.epochs if args.method == "original" else args.scale_epochs
    cmd = ["swift", "sft", "--model", args.model, *_model_type_args(args),
           "--dataset", str(train_jsonl),
           "--tuner_type", "lora", "--lora_rank", str(args.lora_rank),
           "--lora_alpha", str(args.lora_alpha), "--lora_dropout", str(args.lora_dropout),
           "--target_modules", "all-linear", "--custom_register_path", str(plugin_path)]
    if args.method == "original":
        cmd += ["--loss_scale", "operator_emphasis"]
    else:
        cmd += ["--loss_scale", "scale_tags", "--loss_type", "scale_cot"]
    cmd += ["--learning_rate", str(args.lr), "--num_train_epochs", str(epochs),
            "--per_device_train_batch_size", str(args.batch_size),
            "--gradient_accumulation_steps", str(args.grad_accum),
            "--max_length", str(args.max_seq_len), "--packing", "false", "--attn_impl", "sdpa",
            "--torch_dtype", "bfloat16", "--gradient_checkpointing", "true",
            "--save_strategy", "epoch", "--logging_steps", "10", "--output_dir", str(swift_out),
            "--seed", str(args.seed), "--save_only_model", "true"]
    if val_jsonl is not None:
        cmd += ["--val_dataset", str(val_jsonl), "--eval_strategy", "epoch",
                "--load_best_model_at_end", "true", "--metric_for_best_model", "loss",
                "--greater_is_better", "false", "--save_total_limit", "2"]
    else:
        # covid_TSFM_bitong CV folds keep EVERY per-epoch checkpoint (epoch selection on the
        # validation block afterwards); otherwise only the last one
        keep = epochs if getattr(args, "keep_all_checkpoints", False) else 1
        cmd += ["--split_dataset_ratio", "0", "--save_total_limit", str(keep)]
    cmd += ["--report_to", "wandb"]
    print("\n$ " + " ".join(cmd) + "\n")
    env = dict(os.environ); env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env["WANDB_PROJECT"] = "CoT_generation_updated_finetune"
    # the run_name leaf dir (fold runs: <run_name>_fold<F>, set by run_bitong_folds)
    env.setdefault("WANDB_NAME", getattr(args, "wandb_name", None) or swift_out.parent.name)
    subprocess.run(cmd, check=True, env=env)


def find_best_adapter(swift_out: Path) -> Path:
    states = sorted(swift_out.glob("**/trainer_state.json"), key=lambda p: p.stat().st_mtime)
    for st in reversed(states):
        try:
            best = json.loads(st.read_text()).get("best_model_checkpoint")
        except Exception:
            best = None
        if best and Path(best).is_dir():
            print(f"  using best checkpoint: {best}")
            return Path(best)
    cands = [c for c in swift_out.glob("**/checkpoint-*") if c.is_dir()]
    if not cands:
        raise RuntimeError(f"no checkpoint under {swift_out}")
    cands.sort(key=lambda p: p.stat().st_mtime)
    print(f"  using latest checkpoint: {cands[-1]}")
    return cands[-1]


def list_epoch_checkpoints(swift_out: Path) -> List[Tuple[int, Path]]:
    """[(epoch, checkpoint_dir)] for every per-epoch LoRA checkpoint of the NEWEST swift run
    under swift_out (swift writes swift_out/v<N>-<timestamp>/checkpoint-<step>/). The epoch
    is read from the checkpoint's trainer_state.json (fallback: its rank by step)."""
    versions = sorted((p for p in swift_out.glob("v*") if p.is_dir()),
                      key=lambda p: p.stat().st_mtime)
    roots = [versions[-1]] if versions else [swift_out]
    cands = [c for r in roots for c in r.glob("checkpoint-*")
             if c.is_dir() and (c / "adapter_config.json").exists()]

    def step_of(c: Path) -> int:
        try:
            return int(c.name.rsplit("-", 1)[-1])
        except ValueError:
            return -1
    cands.sort(key=step_of)
    out: List[Tuple[int, Path]] = []
    for i, c in enumerate(cands, 1):
        ep = None
        st = c / "trainer_state.json"
        if st.exists():
            try:
                ep = json.loads(st.read_text()).get("epoch")
            except Exception:
                ep = None
        out.append((int(round(float(ep))) if ep is not None else i, c))
    return out


def select_best_epoch(args, out_dir: Path, swift_out: Path, val: List[Dict], val_infer: Path,
                      horizon: int, bins: Optional[Dict]) -> Optional[Dict]:
    """covid_TSFM_bitong CV folds: score EVERY per-epoch checkpoint on the fold's validation
    block and pick the best epoch by args.epoch_metric (ties: better MAE improvement vs the
    baseline, then the EARLIER epoch). Writes <out_dir>/epoch_selection.{json,csv}; the per-
    epoch predictions live in <out_dir>/epoch_val/. Returns None when there is nothing to
    choose from (a single checkpoint)."""
    all_ckpts = list_epoch_checkpoints(swift_out)
    skip = int(getattr(args, "epoch_skip", 0) or 0)
    stride = max(1, int(getattr(args, "epoch_stride", 1) or 1))
    # score epochs skip+1, skip+1+stride, ... (default 4, 6, 8, ...); never the first `skip`
    ckpts = [(ep, ck) for ep, ck in all_ckpts if ep > skip and (ep - skip - 1) % stride == 0]
    if not ckpts and all_ckpts:
        print(f"  epoch selection: skip={skip}/stride={stride} leaves no epoch out of "
              f"{[e for e, _ in all_ckpts]} — falling back to the last one")
        ckpts = all_ckpts[-1:]
    if len(ckpts) < 2:
        print(f"  epoch selection: {len(ckpts)} candidate checkpoint(s) under {swift_out} — "
              f"nothing to choose from, using the last one")
        return None
    print(f"  epoch selection: {len(all_ckpts)} checkpoints, scoring epochs "
          f"{[e for e, _ in ckpts]} (skip first {skip}, stride {stride})")
    metric = args.epoch_metric
    lower = metric in BITONG_LOWER_IS_BETTER
    ep_dir = out_dir / "epoch_val"
    ep_dir.mkdir(exist_ok=True)
    print(f"  epoch selection over {len(ckpts)} checkpoints by {metric} "
          f"({'lower' if lower else 'higher'} is better) on {len(val)} val examples")
    rows = []
    for ep, ck in ckpts:
        res = ep_dir / f"swift_predictions_val_ep{ep:02d}.jsonl"
        run_swift_infer(args, val_infer, res, adapter_dir=ck)
        _, m = evaluate(val, load_swift_predictions(res), horizon, bins, args.nochange_band)
        rows.append({"epoch": ep, "checkpoint": str(ck), "predictions": str(res),
                     **{f"val_{k}": m[k] for k in VAL_KEYS}})
        print(f"    epoch {ep:2d}: dir={m['direction_accuracy']:.3f} "
              f"bin_exact={m['bin_exact_accuracy_moves']:.3f} op={m['operator_accuracy']:.3f} "
              f"MAE={m['rough_MAE_counts']:.2f} (baseline {m['baseline_MAE_counts']:.2f}) "
              f"parsed {m['n_parsed']}/{m['n']}")

    def _num(v) -> float:
        return float("-inf") if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)

    def score(r):
        prim = _num(r.get(metric))
        if lower and prim != float("-inf"):
            prim = -prim
        return (prim, _num(r.get("val_improvement_MAE_vs_baseline")), -r["epoch"])
    best = max(rows, key=score)
    sel = {"metric": metric, "lower_is_better": lower, "n_checkpoints": len(rows),
           "n_checkpoints_total": len(all_ckpts), "epoch_skip": skip, "epoch_stride": stride,
           "epochs_scored": [r["epoch"] for r in rows],
           "best_epoch": best["epoch"], "best": best, "per_epoch": rows}
    (out_dir / "epoch_selection.json").write_text(json.dumps(sel, indent=2))
    pd.DataFrame(rows).drop(columns=["checkpoint", "predictions"]).to_csv(
        out_dir / "epoch_selection.csv", index=False)
    print(f"  best epoch by {metric}: {best['epoch']} ({best[metric]:.4f}) → {best['checkpoint']}")
    return sel


def run_swift_infer(args, dataset_jsonl: Path, result_path: Path, adapter_dir: Optional[Path]) -> None:
    if result_path.exists():
        if args.eval_only:
            print(f"REUSE {result_path} (--eval-only: keeping existing predictions)")
            return
        result_path.unlink()
    cmd = ["swift", "infer"]
    cmd += (["--adapters", str(adapter_dir)] if adapter_dir is not None
            else ["--model", args.model, *_model_type_args(args)])
    cmd += ["--val_dataset", str(dataset_jsonl), "--max_new_tokens", str(args.max_new_tokens),
            "--temperature", "0", "--top_p", "1.0", "--repetition_penalty", "1.0",
            "--infer_backend", "pt", "--stream", "false", "--result_path", str(result_path)]
    ibs = int(getattr(args, "infer_batch_size", 1) or 1)
    if ibs > 1:                        # batched generation (pt backend); 1 = the old one-by-one
        cmd += ["--max_batch_size", str(ibs)]
    print("\n$ " + " ".join(cmd) + "\n")
    env = dict(os.environ); env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    subprocess.run(cmd, check=True, env=env)


def load_swift_predictions(path: Path) -> List[str]:
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            resp = rec.get("response")
            if resp is None and "messages" in rec:
                for m in reversed(rec["messages"]):
                    if m.get("role") == "assistant":
                        resp = m.get("content", ""); break
            out.append(resp or "")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Parse the model's operator + CATEGORICAL evaluation vs the true result
# ══════════════════════════════════════════════════════════════════════════════
def _parse_dir_token(t: str) -> Optional[int]:
    t = t.strip().lower()
    if "increase" in t or t == "up":
        return 1
    if "decrease" in t or t == "down":
        return -1
    if "no change" in t or "no-change" in t or t in ("none", "no", "flat", "same"):
        return 0
    return None


def parse_operator(text: str, horizon: int, names: List[str]) -> Tuple[Optional[List[int]], List[Optional[int]]]:
    """Return (directions, sizes) parsed from the model's Direction:/Size: lines. directions is
    None if unparseable; sizes[h] is a bin index, or None for 'none'/no-change."""
    if not text or DIRECTION_TAG.lower() not in text.lower():
        return None, [None] * horizon
    # (or [""]) guards responses truncated right at the tag: split()[-1] is "" and splitlines() is []
    dline = (re.split(re.escape(DIRECTION_TAG), text, flags=re.IGNORECASE)[-1].splitlines() or [""])[0]
    dirs = [d for d in (_parse_dir_token(t) for t in dline.split(",")) if d is not None]
    if len(dirs) < horizon:
        return None, [None] * horizon
    dirs = dirs[:horizon]
    sizes: List[Optional[int]] = [None] * horizon
    if SIZE_TAG.lower() in text.lower():
        sline = (re.split(re.escape(SIZE_TAG), text, flags=re.IGNORECASE)[-1].splitlines() or [""])[0]
        low = [n.lower() for n in names]
        toks = [t.strip().lower() for t in sline.split(",")]
        for h in range(min(horizon, len(toks))):
            t = toks[h]
            if t in ("none", "no", "no change", "no-change", ""):
                sizes[h] = None
            elif t in low:
                sizes[h] = low.index(t)
    return dirs, sizes


def load_finance_orig(states: List[str]) -> Dict[Tuple[str, str], Tuple[float, List[float]]]:
    """(ticker, origin_week) -> (price_scale, future_3w_orig): the per-ticker back-transform
    factor and the ORIGINAL-scale (USD) ground truth, from the organized CSVs."""
    out = {}
    for st in states:
        df = pd.read_csv(FINANCE_ORG_DIR / f"{st}.csv", keep_default_na=False)
        for _, r in df.iterrows():
            out[(st, r["origin_week"])] = (float(r["price_scale"]),
                                           [float(v) for v in json.loads(r["future_3w_orig"])])
    return out


def to_original_scale(by_split: Dict[str, List[Dict]], orig) -> None:
    """Finance: point baseline_full/target_full at the ORIGINAL USD scale, so the rough
    MAE/MSE is computed on back-transformed values (baseline / price_scale) against the
    untouched original ground truth. The categorical evaluation stays on the normalized
    `baseline`/`target` (the relative band and bins are scale-invariant)."""
    for exs in by_split.values():
        for ex in exs:
            scale, f_orig = orig[(ex["state"], ex["reference_date"])]
            ex["baseline_full"] = [float(v) / scale for v in ex["baseline"]]
            # t+3-only corpora carry length-1 vectors; slice the 3-week USD truth to match
            ex["target_full"] = f_orig[2:] if len(ex["baseline"]) == 1 and len(f_orig) == 3 else f_orig


def build_finance_eval_examples(states: List[str], splits: List[str], horizon: int,
                                orig, data_mode: str = "finance_stat") -> Dict[str, List[Dict]]:
    """Finance val/test EVALUATION examples, built OFFLINE from the organized CSVs (the
    finance corpus generates CoT for the train split only). Reuses the generation
    pipeline's own sample builder, news gate and block renderer, so the prompts are
    identical to what generation would produce; cot='' (no reference reasoning — these
    examples are inference-eval only and never train)."""
    from types import SimpleNamespace
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import cot_common as C
    import step1_generate as S1
    ns = SimpleNamespace(mode="chronos2_full", data_mode=data_mode,
                         lookback=C.LOOKBACK_DEFAULT,
                         resid_lookback=C.RESID_LOOKBACK_DEFAULT)
    bundle = C.load_data_bundle(data_mode, states)
    out: Dict[str, List[Dict]] = {sp: [] for sp in splits}
    for state in states:
        store = bundle.store_for(state)
        samples, cache = C.build_chronos2_samples([], [state], splits, bundle=bundle)
        if data_mode == "finance_TSFM":
            C.subset_samples_to_t3(samples)          # keep ONLY the origin+3 horizon
        samples = C.tag_and_gate_fluview(samples, store, ns.lookback, 1)
        C.round_baseline_counts(samples, C.mode_value_decimals("chronos2_full", data_mode))
        for s in sorted(samples, key=lambda x: x["reference_date"]):
            blocks = S1.build_blocks(s, store, cache, 1.0, ns)
            scale, f_orig = orig[(state, s["reference_date"])]
            if data_mode == "finance_TSFM":
                f_orig = f_orig[2:]                  # USD ground truth for origin+3 only
            out[s["split"]].append({
                "id": s["id"], "state": state, "split": s["split"],
                "reference_date": s["reference_date"],
                "baseline": [float(v) for v in s["x_hat"]],
                "target": [float(v) for v in s["x_gt"]],
                "baseline_full": [float(v) / scale for v in s["x_hat"]],
                "target_full": f_orig,
                "dir_target": [0] * horizon, "size_target": ["none"] * horizon,
                "cot": "", "operator": None,
                **{k: blocks[k] for k in ("history_block", "baseline_block",
                                          "recent_residual_block", "weekly_text_block")},
            })
    for sp in splits:
        print(f"  built {len(out[sp])} offline finance {sp} eval examples")
    return out


def load_covid_orig(states: List[str], data_mode: str = "covid_TSFM",
                    data_dir: Optional[Path] = None
                    ) -> Dict[Tuple[str, str], Tuple[List[float], List[float], List[float]]]:
    """(state, reference_date) -> (mu_3w, sigma_3w, target_orig_3w): the per-week denorm
    parameters for the 3 future weeks and the ORIGINAL-scale ground truth (weekly proportion
    of ED visits due to COVID-19), from the covid dataset CSVs. future_3w is the full-
    precision z-score + COVID_SHIFT/100, so (z − COVID_SHIFT/100)*sigma+mu restores the
    source observation exactly."""
    out = {}
    shift = covid_shift_for(data_mode)
    for st in states:
        df = pd.read_csv((data_dir or covid_dir_for(data_mode)) / f"{st}.csv", keep_default_na=False)
        for _, r in df.iterrows():
            mu = [float(v) for v in json.loads(r["denorm_mean_future_3w"])]
            sd = [float(v) for v in json.loads(r["denorm_std_future_3w"])]
            if "future_3w_full" in df.columns:          # master layout: 100*z at full precision
                z = [float(v) / 100.0 for v in json.loads(r["future_3w_full"])]
            else:                                       # pipeline layout: z + COVID_SHIFT/100
                z = [float(v) - shift / 100.0 for v in json.loads(r["future_3w"])]
            out[(st, r["reference_date"])] = (mu, sd,
                                              [zz * s + m for zz, s, m in zip(z, sd, mu)])
    return out


def to_covid_original_scale(by_split: Dict[str, List[Dict]], orig,
                            data_mode: str = "covid_TSFM") -> None:
    """covid_TSFM: attach the per-week denorm parameters and point baseline_full/target_full
    at the ORIGINAL ED-visit-proportion scale, so the rough MAE/MSE is computed on the
    back-transformed values (((v − COVID_SHIFT)/100)*sigma + mu — see evaluate()'s denorm
    branch: the operator is applied in normalized units, THEN the predicted result is
    transformed). The categorical evaluation stays on the normalized shifted-100x-z
    `baseline`/`target` (the relative band and bins live in that shifted space)."""
    shift = covid_shift_for(data_mode)
    for exs in by_split.values():
        for ex in exs:
            mu, sd, tgt = orig[(ex["state"], ex["reference_date"])]
            H = len(ex["baseline"])              # horizon: 2 (covid_TSFM) / 1 (updated)
            mu, sd, tgt = _covid_slice(mu, H), _covid_slice(sd, H), _covid_slice(tgt, H)
            ex["denorm_mu"], ex["denorm_sigma"] = mu, sd
            ex["covid_shift"] = shift
            ex["baseline_full"] = [((float(b) - shift) / 100.0) * s + m
                                   for b, s, m in zip(ex["baseline"], sd, mu)]
            ex["target_full"] = tgt


def build_covid_eval_examples(states: List[str], splits: List[str], horizon: int,
                              orig, data_mode: str = "covid_TSFM",
                              data_dir: Optional[Path] = None) -> Dict[str, List[Dict]]:
    """covid val/test EVALUATION examples, built OFFLINE from the covid dataset CSVs (the
    covid corpus generates CoT for the train split only, like finance). Reuses the
    generation pipeline's own sample builder, text gate and block renderer, so the prompts
    are identical to what generation would produce; cot='' (inference-eval only)."""
    from types import SimpleNamespace
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import cot_common as C
    import step1_generate as S1
    pdm = pipeline_data_mode(data_mode)      # covid_TSFM_bitong -> covid_TSFM_updated loaders
    ns = SimpleNamespace(mode="chronos2_full", data_mode=pdm,
                         lookback=C.LOOKBACK_DEFAULT,
                         resid_lookback=C.RESID_LOOKBACK_DEFAULT)
    shift = covid_shift_for(data_mode)
    bundle = C.load_data_bundle(pdm, states, data_dir)   # data_dir: text-ablation variants / fold dir
    if bundle.split_ranges:
        # a CSV without one of the requested splits (e.g. a final train/test setting with no
        # val block) yields no examples for it instead of failing in build_chronos2_samples
        missing = [sp for sp in splits if sp not in bundle.split_ranges]
        if missing:
            print(f"  covid eval: split(s) {missing} absent from the dataset CSVs -> 0 examples")
            splits = [sp for sp in splits if sp in bundle.split_ranges]
    out: Dict[str, List[Dict]] = {sp: [] for sp in splits}
    for state in states:
        store = bundle.store_for(state)
        samples, cache = C.build_chronos2_samples([], [state], splits, bundle=bundle)
        C.subset_samples_to_offsets(samples, C.covid_pred_offsets(pdm))
        samples = C.tag_and_gate_fluview(samples, store, ns.lookback, 1)
        C.round_baseline_counts(samples, C.mode_value_decimals("chronos2_full", pdm))
        for s in sorted(samples, key=lambda x: x["reference_date"]):
            blocks = S1.build_blocks(s, store, cache, 1.0, ns)
            mu, sd, tgt_orig = orig[(state, s["reference_date"])]
            mu, sd, tgt_orig = (_covid_slice(mu, horizon), _covid_slice(sd, horizon),
                                _covid_slice(tgt_orig, horizon))
            out[s["split"]].append({
                "id": s["id"], "state": state, "split": s["split"],
                "reference_date": s["reference_date"],
                "baseline": [float(v) for v in s["x_hat"]],
                "target": [float(v) for v in s["x_gt"]],
                "baseline_full": [((float(v) - shift) / 100.0) * ss + m
                                  for v, ss, m in zip(s["x_hat"], sd, mu)],
                "target_full": tgt_orig,
                "denorm_mu": mu, "denorm_sigma": sd, "covid_shift": shift,
                "dir_target": [0] * horizon, "size_target": ["none"] * horizon,
                "cot": "", "operator": None,
                **{k: blocks[k] for k in ("history_block", "baseline_block",
                                          "recent_residual_block", "weekly_text_block")},
            })
    for sp in splits:
        print(f"  built {len(out[sp])} offline covid {sp} eval examples")
    return out


def evaluate(examples: List[Dict], responses: List[str], horizon: int,
             bins: Optional[Dict], eps: float) -> Tuple[pd.DataFrame, Dict]:
    """Categorical accuracy vs the true operator, PLUS a rough count-based MAE/MSE: each predicted
    (direction, bin) is grounded to a number δ[h] = sign(dir) × bin-LOWER-EDGE (0 on no-change),
    the forecast = full baseline + δ is converted per-10,000 → counts, and scored vs the ground
    truth and the baseline. All MAE/MSE are in ABSOLUTE HOSPITALIZATION COUNTS."""
    names = bin_names(bins)
    lower = bins["lower_edges"] if bins else None
    rel = bins_are_relative(bins)   # influenza/finance chronos2_full: counts + relative band/bins
    # covid_TSFM: ABSOLUTE index-point band/bins (no rate→count scaling; denorm for the MAE)
    abs_idx = bool(bins) and bins.get("kind") == BINS_KIND_ABS_INDEX
    scale_map = ({} if (rel or abs_idx)
                 else (rate_to_count_scale({ex["state"] for ex in examples}) if examples else {}))
    rows = []
    dir_hits = dir_tot = 0
    bin_exact = bin_within1 = bin_move_tot = 0
    op_hits = op_tot = 0
    n_parsed = 0
    r_ae, r_se, b_ae, b_se = [], [], [], []      # rough forecast / baseline errors (counts), parsed rows
    for ex, resp in zip(examples, responses):
        r = [round(g - b, 2) for g, b in zip(ex["target"], ex["baseline"])]   # absolute residual (reported)
        if rel:
            # chronos2_full: classify the RELATIVE residual (fraction of baseline); band ε is relative
            rr = [(g - b) / max(abs(b), 1.0) for g, b in zip(ex["target"], ex["baseline"])]
        else:
            rr = r          # absolute: rate units (chronos2/flat) or index points (covid_TSFM)
        true_dir = [0 if abs(x) <= eps else (1 if x > 0 else -1) for x in rr]
        true_bin = [None if abs(x) <= eps else bin_of(abs(x), bins) for x in rr]
        md, ms = parse_operator(resp, horizon, names)
        parsed = md is not None
        n_parsed += int(parsed)
        row = {"id": ex["id"], "state": ex["state"], "reference_date": ex["reference_date"],
               "operator": ex.get("operator"), "residual": r,
               "true_direction": true_dir, "true_bin": true_bin,
               "pred_direction": md, "pred_bin": ms, "parsed": parsed}
        if parsed:
            for h in range(horizon):
                dir_tot += 1
                d_ok = (md[h] == true_dir[h])
                dir_hits += int(d_ok)
                if true_dir[h] == 0:
                    s_ok = (ms[h] is None)
                else:
                    bin_move_tot += 1
                    s_ok = (ms[h] == true_bin[h])
                    bin_exact += int(s_ok)
                    if ms[h] is not None and abs(ms[h] - true_bin[h]) <= 1:
                        bin_within1 += 1
                op_tot += 1
                op_hits += int(d_ok and s_ok)
            # rough NUMBER forecast: δ = sign(dir) × bin lower-edge, applied to the full baseline.
            if lower is not None:
                bf = np.asarray(ex["baseline_full"], float)
                tf = np.asarray(ex["target_full"], float)
                if ex.get("denorm_sigma") is not None:
                    # covid_TSFM: ground the operator in NORMALIZED (shifted 100·z + 300)
                    # units — δ = sign × bin-lower-edge, where the edge is an ABSOLUTE number
                    # of index points (absolute_index bins) or, for legacy relative-band
                    # corpora, a fraction × |normalized baseline| — then TRANSFORM the
                    # predicted result to the ORIGINAL ED-visit proportion
                    # (((v − COVID_SHIFT)/100)·σ + μ) and compute the MAE/MSE there.
                    # baseline_full/target_full are already on the original scale.
                    bn = np.asarray(ex["baseline"], float)
                    mu = np.asarray(ex["denorm_mu"], float)
                    sd = np.asarray(ex["denorm_sigma"], float)
                    delta = np.array([(float(md[h]) * float(lower[ms[h]])
                                       * (abs(bn[h]) if rel else 1.0))
                                      if (md[h] != 0 and ms[h] is not None) else 0.0
                                      for h in range(horizon)])
                    fc = (bn + delta - float(ex.get("covid_shift", COVID_SHIFT))) / 100.0 * sd + mu
                    tgt, base = tf, bf
                elif rel:
                    # relative bins: δ (counts) = sign × m_init(fraction) × |baseline|; values are
                    # already whole counts, so no rate→count scaling.
                    delta = np.array([(float(md[h]) * float(lower[ms[h]]) * abs(bf[h]))
                                      if (md[h] != 0 and ms[h] is not None) else 0.0
                                      for h in range(horizon)])
                    fc, tgt, base = bf + delta, tf, bf
                else:
                    sc = scale_map[ex["state"].upper()]
                    delta = np.array([(float(md[h]) * float(lower[ms[h]]))
                                      if (md[h] != 0 and ms[h] is not None) else 0.0
                                      for h in range(horizon)])
                    fc, tgt, base = (bf + delta) * sc, tf * sc, bf * sc
                r_ae.append(float(np.mean(np.abs(tgt - fc)))); r_se.append(float(np.mean((tgt - fc) ** 2)))
                b_ae.append(float(np.mean(np.abs(tgt - base)))); b_se.append(float(np.mean((tgt - base) ** 2)))
                row["rough_delta_rate"] = delta.tolist()
                row["rough_forecast_count"] = fc.tolist()
                row["rough_mae_count"] = r_ae[-1]
                row["baseline_mae_count"] = b_ae[-1]
        rows.append(row)
    mean = lambda a: (float(np.mean(a)) if a else float("nan"))
    summ = {
        "n": len(examples), "n_parsed": n_parsed,
        "direction_accuracy": (dir_hits / dir_tot) if dir_tot else float("nan"),
        "bin_exact_accuracy_moves": (bin_exact / bin_move_tot) if bin_move_tot else float("nan"),
        "bin_within1_accuracy_moves": (bin_within1 / bin_move_tot) if bin_move_tot else float("nan"),
        "operator_accuracy": (op_hits / op_tot) if op_tot else float("nan"),
        # rough number metrics (counts), over parsed rows; baseline over the SAME rows (fair compare)
        "rough_MAE_counts": mean(r_ae), "rough_MSE_counts": mean(r_se),
        "baseline_MAE_counts": mean(b_ae), "baseline_MSE_counts": mean(b_se),
        "improvement_MAE_vs_baseline": (mean(b_ae) - mean(r_ae)) if (r_ae and b_ae) else float("nan"),
    }
    return pd.DataFrame(rows), summ


# ══════════════════════════════════════════════════════════════════════════════
# One finetune + eval run
# ══════════════════════════════════════════════════════════════════════════════
def run_one(args, out_dir: Path, by_split: Dict[str, List[Dict]], horizon: int,
            bins: Optional[Dict]) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists() and not args.force and not args.eval_only:
        print(f"SKIP {out_dir.name} (metrics.json present; --force to redo)")
        return json.loads(metrics_path.read_text())
    names_str = " | ".join(bin_names(bins))
    template = ensure_finetune_prompt(args.mode, args.prompt_dir, args.data_mode)
    wording = PROMPT_WORDING[(args.data_mode, args.mode)]

    train, val, test = by_split.get("train", []), by_split.get("val", []), by_split.get("test", [])
    if not train:
        sys.exit("no train examples — nothing to finetune.")
    # finance's offline val examples carry no reference CoT -> they are inference-eval only
    # (written without assistant text, never passed to swift as a training eval set)
    val_has_cot = bool(val) and all(ex.get("cot") for ex in val)
    train_jsonl, val_jsonl, test_jsonl = out_dir / "train.jsonl", out_dir / "val.jsonl", out_dir / "test.jsonl"
    sysp = system_prompt_for(args.data_mode)
    # class weighting = oversampling of the TRAIN set only (val/test never expanded)
    repeats = class_repeats(args.class_weights)
    n_cls = {k: sum(1 for ex in train if ex.get("cls") == k) for k in CLASS_ORDER}
    train_w = expand_train(train, repeats, args.seed)
    print(f"  class weights {args.class_weights_spec} → repeats "
          + "/".join(f"{k}×{repeats[k]}" for k in CLASS_ORDER)
          + f" | train classes {n_cls} | train rows {len(train)} → {len(train_w)}")
    write_jsonl(train_jsonl, train_w, horizon, names_str, template, wording,
                include_assistant=True, sys_prompt=sysp)
    if val:
        write_jsonl(val_jsonl, val, horizon, names_str, template, wording,
                    include_assistant=val_has_cot, sys_prompt=sysp)
    write_jsonl(test_jsonl, test, horizon, names_str, template, wording,
                include_assistant=False, sys_prompt=sysp)
    (out_dir / "splits.json").write_text(json.dumps(
        {"mode": args.mode, "method": args.method, "bins": bin_names(bins),
         "n_train": len(train), "n_train_expanded": len(train_w), "train_classes": n_cls,
         "class_weights": args.class_weights, "class_repeats": repeats,
         "n_val": len(val), "n_test": len(test)}, indent=2))
    if args.data_only:
        print("--data-only: wrote datasets, skipping training."); return {}

    plugin = out_dir / "loss_plugin.py"
    if args.method == "original":
        write_original_plugin(plugin, args.operator_loss_weight, REASONING_LOSS_WEIGHT)
    else:
        write_scale_plugin(plugin, args.scale_start, args.scale_end,
                           args.scale_mode, args.scale_min_think, args.scale_other_w)

    swift_out = out_dir / "swift_sft"
    _check_swift()
    if not args.eval_only and args.adapter is None:
        run_swift_sft(args, train_jsonl, swift_out, plugin,
                      val_jsonl if (val and val_has_cot) else None)
    adapter = args.adapter if args.adapter is not None else find_best_adapter(swift_out)

    val_infer = out_dir / "val_infer.jsonl"
    if val:
        write_jsonl(val_infer, val, horizon, names_str, template, wording,
                    include_assistant=False, sys_prompt=sysp)
    # covid_TSFM_bitong CV folds: score every per-epoch checkpoint on the validation block and
    # make the best epoch THE fold model (its best_epoch feeds the final run's epoch count)
    epoch_sel: Optional[Dict] = None
    if val and getattr(args, "epoch_select", False) and args.adapter is None:
        epoch_sel = select_best_epoch(args, out_dir, swift_out, val, val_infer, horizon, bins)
        if epoch_sel is not None:
            adapter = Path(epoch_sel["best"]["checkpoint"])
    # covid_TSFM_bitong final stage: the full cap was trained (same schedule as the folds);
    # THE model is the checkpoint at the epoch the folds agreed on
    pick = getattr(args, "pick_epoch", None)
    if pick is not None and args.adapter is None:
        ckpts = list_epoch_checkpoints(swift_out)
        hit = [ck for ep, ck in ckpts if ep == int(pick)]
        if not hit:
            sys.exit(f"pick_epoch={pick}: no checkpoint for that epoch under {swift_out} "
                     f"(epochs present: {[e for e, _ in ckpts]})")
        adapter = hit[0]
        print(f"  using the epoch-{pick} checkpoint of {len(ckpts)} "
              f"(same {args.scale_epochs if args.method == 'scale' else args.epochs}-epoch "
              f"schedule as the folds): {adapter}")

    if test:
        lora_res = out_dir / "swift_predictions_lora.jsonl"
        run_swift_infer(args, test_jsonl, lora_res, adapter_dir=adapter)
        lora_resp = load_swift_predictions(lora_res)
        zs_res = out_dir / "swift_predictions_zero_shot.jsonl"
        run_swift_infer(args, test_jsonl, zs_res, adapter_dir=None)
        zs_resp = load_swift_predictions(zs_res)

        df, lora_m = evaluate(test, lora_resp, horizon, bins, args.nochange_band)
        _, zs_m = evaluate(test, zs_resp, horizon, bins, args.nochange_band)
        df.to_csv(out_dir / "predictions.csv", index=False)
    else:
        # covid_TSFM_bitong fold runs: the fold model is scored on its VALIDATION block only
        # (the test block is reserved for the final model) — no test inference at all
        print("  no test examples in this run (fold validation run): test inference skipped")
        lora_m, zs_m = {}, {}

    val_m: Dict = {}
    val_by_state: Dict[str, Dict] = {}
    if val:
        val_res = out_dir / "swift_predictions_val.jsonl"
        if epoch_sel is not None:
            # the best epoch's predictions ARE the fold model's validation predictions
            shutil.copyfile(epoch_sel["best"]["predictions"], val_res)
        else:
            run_swift_infer(args, val_infer, val_res, adapter_dir=adapter)
        val_resp = load_swift_predictions(val_res)
        dfv, val_m = evaluate(val, val_resp, horizon, bins, args.nochange_band)
        dfv.to_csv(out_dir / "predictions_val.csv", index=False)
        # per-state validation metrics (bitong: NJ has text for only 2/20 val-C weeks, so the
        # pooled number hides a lot; the fold summary averages these too)
        for st in sorted({ex["state"] for ex in val}):
            idx = [i for i, ex in enumerate(val) if ex["state"] == st]
            _, m_st = evaluate([val[i] for i in idx], [val_resp[i] for i in idx],
                               horizon, bins, args.nochange_band)
            val_by_state[st] = {k: m_st[k] for k in VAL_KEYS}

    metrics = {
        "mode": args.mode, "method": args.method, "states": args.states, "model": Path(args.model).name,
        "fold": getattr(args, "fold", None),          # covid_TSFM_bitong: A / B / C / final
        "n_test": len(test), "n_parsed_lora": lora_m.get("n_parsed"),
        "lora_direction_accuracy": lora_m.get("direction_accuracy"),
        "lora_bin_exact_accuracy_moves": lora_m.get("bin_exact_accuracy_moves"),
        "lora_bin_within1_accuracy_moves": lora_m.get("bin_within1_accuracy_moves"),
        "lora_operator_accuracy": lora_m.get("operator_accuracy"),
        # rough count-based forecast error from grounding the operator at each bin's lower edge:
        # forecast = baseline (± predicted-bin LOWER edge), compared to ground truth, in raw counts
        "units_rough_mae": ("USD (baseline back-transformed via price_scale; ground truth = "
                            "original future_3w_orig)" if args.data_mode in FINANCE_MODES
                            else "ED-visit proportion (operator applied in shifted 100x-z "
                            "units, predicted result back-transformed via "
                            "((v-300)/100)*sigma+mu; x100 = percent of ED visits)"
                            if args.data_mode == "covid_TSFM"
                            else "hospital admissions (operator applied in 100x-z units, "
                            "back-transformed via (v/100)*sigma+mu)"
                            if args.data_mode in COVID_Z26W_MODES
                            else "counts (raw hospitalizations; baseline × predicted-bin lower edge)"
                            if bins_are_relative(bins)
                            else "counts(from full-precision per_10000)"),
        "lora_rough_MAE_counts": lora_m.get("rough_MAE_counts"),
        "lora_rough_MSE_counts": lora_m.get("rough_MSE_counts"),
        "baseline_MAE_counts": lora_m.get("baseline_MAE_counts"),
        "baseline_MSE_counts": lora_m.get("baseline_MSE_counts"),
        "improvement_MAE_vs_baseline": lora_m.get("improvement_MAE_vs_baseline"),
        "zero_shot_direction_accuracy": zs_m.get("direction_accuracy"),
        "zero_shot_operator_accuracy": zs_m.get("operator_accuracy"),
        "zero_shot_rough_MAE_counts": zs_m.get("rough_MAE_counts"),
        "val_n": val_m.get("n"), "val_n_parsed": val_m.get("n_parsed"),
        "val_direction_accuracy": val_m.get("direction_accuracy"),
        "val_bin_exact_accuracy_moves": val_m.get("bin_exact_accuracy_moves"),
        "val_operator_accuracy": val_m.get("operator_accuracy"),
        "val_rough_MAE_counts": val_m.get("rough_MAE_counts"),
        "val_baseline_MAE_counts": val_m.get("baseline_MAE_counts"),
        "val_improvement_MAE_vs_baseline": val_m.get("improvement_MAE_vs_baseline"),
        "val_by_state": val_by_state,
        "lr": args.lr, "epochs": (args.epochs if args.method == "original" else args.scale_epochs),
        # covid_TSFM_bitong: CV folds -> the epoch kept after validation-block selection;
        # final -> where its epoch count came from (round(mean of fold best epochs) / flag)
        "best_epoch": epoch_sel["best_epoch"] if epoch_sel else None,
        "epoch_metric": epoch_sel["metric"] if epoch_sel else None,
        "n_epoch_checkpoints": epoch_sel["n_checkpoints"] if epoch_sel else None,
        "epochs_scored": epoch_sel["epochs_scored"] if epoch_sel else None,
        # final stage: the epoch whose checkpoint is THE model (trained for the full `epochs`)
        "picked_epoch": (int(pick) if pick is not None else None),
        "epochs_source": getattr(args, "epochs_source", None),
        "class_weights": args.class_weights, "class_repeats": repeats,
        "n_train": len(train), "n_train_expanded": len(train_w), "train_classes": n_cls,
        "lora_rank": args.lora_rank, "bins": bin_names(bins),
        "batch_size": args.batch_size, "grad_accum": args.grad_accum,
        "infer_batch_size": int(getattr(args, "infer_batch_size", 1) or 1),
        "max_new_tokens": args.max_new_tokens, "adapter": str(adapter),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))
    if test:
        print(f"\n[{out_dir.name}] LoRA  direction_acc={lora_m['direction_accuracy']:.3f}  "
              f"bin_exact(moves)={lora_m['bin_exact_accuracy_moves']:.3f}  "
              f"operator_acc={lora_m['operator_accuracy']:.3f}  "
              f"| rough MAE={lora_m['rough_MAE_counts']:.2f} vs baseline {lora_m['baseline_MAE_counts']:.2f} "
              f"({lora_m['improvement_MAE_vs_baseline']:+.2f})  "
              f"(parsed {lora_m['n_parsed']}/{len(test)})")
    if val_m:
        print(f"[{out_dir.name}] VAL   direction_acc={val_m['direction_accuracy']:.3f}  "
              f"bin_exact(moves)={val_m['bin_exact_accuracy_moves']:.3f}  "
              f"operator_acc={val_m['operator_accuracy']:.3f}  "
              f"| rough MAE={val_m['rough_MAE_counts']:.2f} vs baseline {val_m['baseline_MAE_counts']:.2f} "
              f"({val_m['improvement_MAE_vs_baseline']:+.2f})  "
              f"(parsed {val_m['n_parsed']}/{len(val)})"
              + "".join(f"\n    {st}: n={m['n']} dir={m['direction_accuracy']:.3f} "
                        f"op={m['operator_accuracy']:.3f} MAE={m['rough_MAE_counts']:.2f} "
                        f"vs base {m['baseline_MAE_counts']:.2f}"
                        for st, m in val_by_state.items()))
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
def parse_schedule(spec: str):
    a, b = spec.split(">")
    return tuple(float(x) for x in a.split(":")), tuple(float(x) for x in b.split(":"))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=MODES, default="chronos2")
    p.add_argument("--data-mode", choices=DATA_MODES, default="influenza",
                   help="influenza (default; unchanged behavior) or finance_stat (tickers, "
                        "normalized prices, USD-back-transformed rough MAE/MSE; "
                        "requires --mode chronos2_full)")
    p.add_argument("--method", choices=METHODS, default="original")
    p.add_argument("--states", default=None,
                   help="geo codes (influenza, default GA,NY,NJ) or tickers (finance, "
                        "default AAPL,GOOGL,NVDA)")
    p.add_argument("--eval-states", default=None,
                   help="geo codes for the OFFLINE val/test evaluation examples, when "
                        "different from the training states (default: "
                        "DATA_MODE_EVAL_STATES for the data mode, else --states; "
                        "covid_TSFM_updated defaults to GA,FL)")
    p.add_argument("--model", default=QWEN_MODEL_PATH)
    p.add_argument("--model-type", default=None,
                   help="ms-swift model_type passthrough (e.g. llama). Needed when swift cannot "
                        "infer it from the checkpoint dir name (Llama-3.1-8B-Instruct). Default: "
                        "None = let swift auto-detect (Qwen3-* checkpoints)")
    p.add_argument("--template", default=None,
                   help="ms-swift template passthrough (e.g. llama3 = Llama-3 chat format WITHOUT "
                        "the 'Today Date' line that llama3_2 injects). Default: swift auto-detect")
    p.add_argument("--corpus-model", default="gemini",
                   help="corpus tag; the run dir is <STATES>-<corpus-model> (e.g. gemini, gpt, "
                        "gemini-complete, gpt-complete). NOT the finetune base model (--model)")
    p.add_argument("--input-root", type=Path, default=INPUT_ROOT)
    p.add_argument("--data-dir", type=Path, default=None,
                   help="covid modes: override the dataset CSV directory used for the "
                        "denorm parameters and the offline val/test eval examples "
                        "(default: data/COVID; the fold views are derived from its master CSVs)")
    p.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    p.add_argument("--prompt-dir", type=Path, default=None,
                   help="finetune prompt lives at <prompt-dir>/<mode>/finetune_prompt.txt")
    p.add_argument("--text-stub", action="store_true",
                   help="no-text control arm: replace the weekly reports with "
                        f"{NOTEXT_STUB!r} in EVERY prompt (train, val, test); the run dir "
                        "gets a _notext suffix (walkthrough §12.5 arm D)")
    p.add_argument("--reasoning-style", choices=REASONING_STYLES, default="cot",
                   help="assistant TARGET reasoning: cot (generated CoT, default) | residual "
                        "(the recent_residual_block verbatim — numbers-only supervision, run "
                        "dir _residualcot) | none (Direction/Size lines only, run dir _nocot). "
                        "With --text-stub this is the finetune_residual_cot.py ablation inside "
                        "every driver, including the bitong folds (walkthrough §12.9)")
    p.add_argument("--nochange-band", type=float, default=None,
                   help="ε no-change band; default 0.01 ABSOLUTE (chronos2/flat), 0.10 RELATIVE "
                        "fraction of the baseline (chronos2_full influenza/finance), or 20 ABSOLUTE "
                        "index points (covid_TSFM). |residual| ≤ ε ⇒ no-change / bin none")
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--epochs", type=int, default=None,
                   help=f"original method epochs (default {EPOCHS}; covid_TSFM_bitong {BITONG_EPOCHS})")
    p.add_argument("--scale-epochs", type=int, default=None,
                   help=f"scale method epoch cap (default {SCALE_EPOCHS}; covid_TSFM_bitong "
                        f"{BITONG_EPOCHS} — the fold's best epoch is picked below the cap)")
    p.add_argument("--lora-rank", type=int, default=LORA_RANK)
    p.add_argument("--lora-alpha", type=int, default=LORA_ALPHA)
    p.add_argument("--lora-dropout", type=float, default=LORA_DROPOUT)
    p.add_argument("--batch-size", type=int, default=None,
                   help=f"per-device train batch (default {BATCH_SIZE}; covid_TSFM_bitong {BITONG_BATCH_SIZE})")
    p.add_argument("--grad-accum", type=int, default=None,
                   help=f"gradient accumulation (default {GRAD_ACCUM}; covid_TSFM_bitong {BITONG_GRAD_ACCUM})")
    p.add_argument("--infer-batch-size", type=int, default=None,
                   help="swift infer --max_batch_size for the val/test generation passes "
                        f"(default 1 = one by one; covid_TSFM_bitong {BITONG_INFER_BATCH_SIZE})")
    p.add_argument("--max-seq-len", type=int, default=None,
                   help=f"default {MAX_SEQ_LEN}; covid_TSFM_bitong defaults to {BITONG_MAX_SEQ_LEN}")
    p.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--operator-loss-weight", type=float, default=OPERATOR_LOSS_WEIGHT,
                   help="original method: up-weight the Direction/Size lines vs reasoning")
    p.add_argument("--scale-schedule", type=str, default="1:1>0.5:2")
    p.add_argument("--scale-mode", choices=["cosine", "linear"], default=SCALE_MODE)
    p.add_argument("--scale-min-think", type=float, default=SCALE_MIN_THINK)
    p.add_argument("--scale-other-w", type=float, default=SCALE_OTHER_W)
    p.add_argument("--class-weights", type=str, default=CLASS_WEIGHTS_DEFAULT,
                   help="move:no_change:abstain training weights, applied by OVERSAMPLING "
                        "train.jsonl (e.g. 2:1:1 writes every move example twice; fractional "
                        "weights are scaled to whole repeats, 2:1:0.5 -> 4/2/1, nothing is "
                        "dropped). Val/test are never expanded. Default 1:1:1 = unchanged.")
    p.add_argument("--adapter", type=Path, default=None,
                   help="reuse this trained LoRA checkpoint dir and skip training — for re-running "
                        "inference/eval (e.g. with a larger --max-new-tokens) in a new run dir")
    p.add_argument("--force", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--data-only", action="store_true")
    # covid_TSFM_bitong: expanding-window hold-out folds
    p.add_argument("--folds", default=BITONG_DEFAULT_FOLDS,
                   help="covid_TSFM_bitong: comma list of folds to run — A,B,C (validation "
                        "folds, scored on their val block) and/or final (full training window, "
                        "scored on the test block). Default A,B,C")
    p.add_argument("--rank-by", default=BITONG_RANK_BY_DEFAULT, choices=BITONG_FOLD_METRICS,
                   help="covid_TSFM_bitong: the mean-over-folds validation metric that ranks "
                        "configurations in fold_cv_summary.csv (default val_operator_accuracy; "
                        "val_rough_MAE_counts ranks ascending)")
    p.add_argument("--epoch-metric", default=BITONG_EPOCH_METRIC_DEFAULT, choices=BITONG_FOLD_METRICS,
                   help="covid_TSFM_bitong: validation metric that picks each CV fold's best "
                        "epoch checkpoint (default val_operator_accuracy; ties -> better MAE "
                        "improvement, then the earlier epoch)")
    p.add_argument("--epoch-skip", type=int, default=BITONG_EPOCH_SKIP,
                   help="covid_TSFM_bitong: never score the first N epochs (default 3)")
    p.add_argument("--epoch-stride", type=int, default=BITONG_EPOCH_STRIDE,
                   help="covid_TSFM_bitong: score every k-th epoch after --epoch-skip "
                        "(default 2 -> epochs 4, 6, 8, ...)")
    p.add_argument("--no-epoch-select", action="store_true",
                   help="covid_TSFM_bitong: skip per-epoch validation scoring and keep the "
                        "last epoch (the final run then needs --final-epochs)")
    p.add_argument("--final-epochs", type=int, default=None,
                   help="covid_TSFM_bitong: epoch count for --folds final (default: "
                        "round(mean of the A/B/C best epochs) from folds_summary.json)")
    p.add_argument("--summarize-only", action="store_true",
                   help="covid_TSFM_bitong: only rebuild <combo>/<method>/fold_cv_summary.csv "
                        "from the fold metrics.json files already on disk (no training)")
    args = p.parse_args()
    if args.prompt_dir is None:            # prompts/covid for the covid modes, prompts/influenza otherwise
        args.prompt_dir = P.prompt_dir_for(args.data_mode)
    return args


def run_name_for(args) -> str:
    """The run (leaf) dir name: base model + method + lr + method-specific tags."""
    run_name = f"{Path(args.model).name}_{args.method}_lr{args.lr:g}"
    if args.method == "original":   # keep distinct operator weights from colliding at the same lr
        run_name += f"_w{args.operator_loss_weight:g}"
    elif args.method == "scale":    # keep distinct schedules from colliding at the same lr
        s, e = args.scale_start, args.scale_end
        run_name += f"_sched{s[0]:g}-{s[1]:g}to{e[0]:g}-{e[1]:g}"
    if args.max_new_tokens != MAX_NEW_TOKENS:   # non-default generation budget → its own run dir
        run_name += f"_mnt{args.max_new_tokens}"
    n_ep = args.epochs if args.method == "original" else args.scale_epochs
    if n_ep != getattr(args, "epochs_default", n_ep):   # non-default epoch cap → its own run dir
        run_name += f"_ep{n_ep}"
    cw_tag = class_weights_tag(args.class_weights)
    if cw_tag:                                  # non-neutral class weights → its own run dir
        run_name += f"_{cw_tag}"
    style = getattr(args, "reasoning_style", "cot")
    if style == "residual":                     # numbers-only supervision → its own run dir
        run_name += "_residualcot"
    elif style == "none":
        run_name += "_nocot"
    if getattr(args, "text_stub", False):       # no-text control arm → its own run dir
        run_name += "_notext"
    return run_name


# ══════════════════════════════════════════════════════════════════════════════
# covid_TSFM_bitong: expanding-window hold-out folds (A, B, C) + final
# ══════════════════════════════════════════════════════════════════════════════
def bitong_fold_dir(fold_root: Path, fold: str) -> Path:
    """The directory of this fold's per-state CSVs in the 13-column pipeline layout.

    Release layout: fold_root holds the 19-column master data/COVID/{STATE}.csv (one file per
    state, with the fold_A / fold_B / fold_C / split label columns). The fold view is derived
    from them here — the fold's label column (unused -> gap; the FINAL test block exposed as
    test), future_3w as plain z from future_3w_full — and cached under
    <output root>/covid_folds/<fold>/ (rebuilt when a master file is newer).
    A pre-split <fold_root>/<fold>/ directory, if present, is used as is."""
    if fold not in BITONG_FOLD_DIRS:
        sys.exit(f"unknown fold {fold!r}; choose from {', '.join(BITONG_FOLD_DIRS)}")
    d = Path(fold_root) / BITONG_FOLD_DIRS[fold]
    if d.is_dir():
        return d
    masters = sorted(p for p in Path(fold_root).glob("*.csv") if re.fullmatch(r"[A-Z]{2,3}", p.stem))
    if not masters:
        sys.exit(f"no master {{STATE}}.csv files in {fold_root} (and no {d} directory)")
    out = P.OUTPUT_ROOT / "covid_folds" / BITONG_FOLD_DIRS[fold]
    out.mkdir(parents=True, exist_ok=True)
    col = BITONG_FOLD_COLUMN[fold]
    for m in masters:
        dst = out / m.name
        if dst.exists() and dst.stat().st_mtime >= m.stat().st_mtime:
            continue
        df = pd.read_csv(m, keep_default_na=False, float_precision="round_trip")
        if col not in df.columns or "future_3w_full" not in df.columns:
            sys.exit(f"{m} lacks the {col!r} / future_3w_full columns of the master layout")
        lab = df[col].replace({"unused": "gap"})
        if col != "split":                        # folds: expose the FINAL test block as test
            lab = lab.where(df["split"] != "test", "test")
        view = df[COVID_PIPE_COLUMNS].copy()
        view["split"] = lab
        view["future_3w"] = [json.dumps([v / 100.0 for v in json.loads(x)]) for x in df["future_3w_full"]]
        view.to_csv(dst, index=False, float_format="%.17g")
    return out


def bitong_split_map(fold_dir: Path, states: List[str]) -> Dict[Tuple[str, str], str]:
    """(state, reference_date) -> split label (train / gap / val / test) of that fold's CSVs.
    reference_date is the ORIGIN week (last observed), the same key the corpus rows carry."""
    out: Dict[Tuple[str, str], str] = {}
    for st in states:
        df = pd.read_csv(fold_dir / f"{st}.csv", keep_default_na=False,
                         usecols=["state", "reference_date", "split"])
        for _, r in df.iterrows():
            out[(str(r["state"]).strip().upper(), str(r["reference_date"]))] = \
                str(r["split"]).strip().lower()
    return out


def bitong_filter_train(train: List[Dict], split_map: Dict[Tuple[str, str], str],
                        fold: str) -> List[Dict]:
    """Keep the corpus rows whose origin is a TRAIN row of this fold; report what was dropped
    (the fold's buffer/val/test origins — and their aug clones, which share the date)."""
    kept, dropped = [], defaultdict(int)
    for ex in train:
        sp = split_map.get((ex["state"], ex["reference_date"]), "MISSING")
        if sp == "train":
            kept.append(ex)
        else:
            dropped[sp] += 1
    cls = {k: sum(1 for ex in kept if ex.get("cls") == k) for k in CLASS_ORDER}
    rng = ((min(ex["reference_date"] for ex in kept), max(ex["reference_date"] for ex in kept))
           if kept else ("-", "-"))
    print(f"  fold {fold}: corpus {len(train)} rows -> {len(kept)} train rows "
          f"(origins {rng[0]} .. {rng[1]}; classes {cls}); dropped "
          + (", ".join(f"{k}={v}" for k, v in sorted(dropped.items())) or "none"))
    if dropped.get("MISSING"):
        print(f"  WARNING fold {fold}: {dropped['MISSING']} corpus rows have no row in the fold "
              f"CSVs (state/date mismatch?)")
    return kept


def _mean(vals: List) -> Optional[float]:
    vals = [float(v) for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(vals)) if vals else None


def bitong_fold_means(fold_metrics: Dict[str, Dict]) -> Dict:
    """Mean of the validation metrics over the CV folds present (A/B/C), overall and per state."""
    cv = {f: m for f, m in fold_metrics.items() if f in BITONG_CV_FOLDS and m}
    means = {k: _mean([m.get(k) for m in cv.values()]) for k in BITONG_FOLD_METRICS}
    states = sorted({st for m in cv.values() for st in (m.get("val_by_state") or {})})
    by_state = {}
    for st in states:
        per = [m["val_by_state"][st] for m in cv.values() if st in (m.get("val_by_state") or {})]
        by_state[st] = {k: _mean([p.get(k) for p in per])
                        for k in ("n", "direction_accuracy", "operator_accuracy",
                                  "rough_MAE_counts", "baseline_MAE_counts",
                                  "improvement_MAE_vs_baseline")}
        by_state[st]["n_folds"] = len(per)
    # the final run's epoch count: round(mean of the CV folds' best epochs), only when every
    # CV fold is present and went through epoch selection
    best_epochs = {f: cv[f].get("best_epoch") for f in sorted(cv)}
    complete = (len(cv) == len(BITONG_CV_FOLDS)
                and all(v is not None for v in best_epochs.values()))
    final_epochs = (int(np.floor(np.mean([float(v) for v in best_epochs.values()]) + 0.5))
                    if complete else None)
    return {"folds": sorted(cv), "n_folds": len(cv), "mean": means, "mean_by_state": by_state,
            "best_epochs": best_epochs, "final_epochs": final_epochs,
            "final_epochs_rule": "round(mean of the A/B/C best epochs); None until all three "
                                 "folds have a best_epoch"}


def write_bitong_folds_summary(run_root: Path, run_name: str) -> Optional[Dict]:
    """<run>/folds_summary.json: per-fold validation metrics (+ the final run's test metrics
    when present) and the mean over the CV folds. Rebuilt from the metrics.json files on disk."""
    fold_metrics: Dict[str, Dict] = {}
    for fold, sub in (("A", "fold_A"), ("B", "fold_B"), ("C", "fold_C"), ("final", "final")):
        mp = run_root / sub / "metrics.json"
        if mp.exists():
            fold_metrics[fold] = json.loads(mp.read_text())
    if not fold_metrics:
        return None
    summary = {"run_name": run_name, **bitong_fold_means(fold_metrics),
               "per_fold": {f: {k: m.get(k) for k in ("n_train", "train_classes", "epochs",
                                                       "best_epoch", "epoch_metric",
                                                       "n_epoch_checkpoints",
                                                       *BITONG_FOLD_METRICS, "val_by_state")}
                            for f, m in fold_metrics.items() if f in BITONG_CV_FOLDS},
               "final": ({k: fold_metrics["final"].get(k) for k in
                          ("n_train", "train_classes", "epochs", "picked_epoch", "epochs_source",
                           "n_test", "n_parsed_lora",
                           "lora_direction_accuracy", "lora_bin_exact_accuracy_moves",
                           "lora_operator_accuracy", "lora_rough_MAE_counts",
                           "baseline_MAE_counts", "improvement_MAE_vs_baseline",
                           "zero_shot_direction_accuracy", "zero_shot_operator_accuracy",
                           "zero_shot_rough_MAE_counts")}
                         if "final" in fold_metrics else None)}
    (run_root / "folds_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def write_bitong_cv_summary(method_root: Path, rank_by: str) -> Optional[pd.DataFrame]:
    """<combo>/<method>/fold_cv_summary.csv: one row per configuration (run dir) with the
    mean-over-folds validation metrics, the per-fold values, the per-state means and the
    final run's test metrics when present; ranked by `rank_by` (the selection rule)."""
    rows = []
    for run_root in sorted(p for p in method_root.iterdir() if p.is_dir()):
        s = write_bitong_folds_summary(run_root, run_root.name)
        if s is None:
            continue
        row = {"run_name": run_root.name, "n_folds": s["n_folds"], "folds": "".join(s["folds"])}
        for k in BITONG_FOLD_METRICS:
            row[f"mean_{k}"] = s["mean"].get(k)
        for f in BITONG_CV_FOLDS:
            m = s["per_fold"].get(f) or {}
            row[f"{f}_val_operator_accuracy"] = m.get("val_operator_accuracy")
            row[f"{f}_val_direction_accuracy"] = m.get("val_direction_accuracy")
            row[f"{f}_val_improvement_MAE"] = m.get("val_improvement_MAE_vs_baseline")
            row[f"{f}_best_epoch"] = m.get("best_epoch")
            row[f"{f}_n_train"] = m.get("n_train")
        row["final_epochs_planned"] = s.get("final_epochs")
        for st, m in s["mean_by_state"].items():
            row[f"{st}_mean_val_operator_accuracy"] = m.get("operator_accuracy")
            row[f"{st}_mean_val_direction_accuracy"] = m.get("direction_accuracy")
            row[f"{st}_mean_val_improvement_MAE"] = m.get("improvement_MAE_vs_baseline")
        fin = s.get("final") or {}
        row["final_epochs_used"] = fin.get("picked_epoch") or fin.get("epochs")
        row["final_test_operator_accuracy"] = fin.get("lora_operator_accuracy")
        row["final_test_direction_accuracy"] = fin.get("lora_direction_accuracy")
        row["final_test_improvement_MAE"] = fin.get("improvement_MAE_vs_baseline")
        rows.append(row)
    if not rows:
        print(f"  no fold metrics under {method_root} yet — nothing to summarize")
        return None
    df = pd.DataFrame(rows)
    for c in df.columns:                      # epoch counts as whole numbers (nullable)
        if c.endswith("_best_epoch") or c.startswith("final_epochs"):
            df[c] = pd.to_numeric(df[c], errors="coerce").round().astype("Int64")
    key = f"mean_{rank_by}"
    # complete CV (all 3 folds) first, then the ranking metric
    df["_complete"] = df["n_folds"] >= len(BITONG_CV_FOLDS)
    df = df.sort_values(["_complete", key],
                        ascending=[False, rank_by in BITONG_LOWER_IS_BETTER],
                        na_position="last").drop(columns="_complete")
    df.insert(0, "rank", range(1, len(df) + 1))
    out = method_root / "fold_cv_summary.csv"
    df.to_csv(out, index=False)
    print(f"\n=== fold CV summary ({len(df)} configuration(s), ranked by mean {rank_by}) → {out}")
    show = ["rank", "run_name", "n_folds", key, "mean_val_direction_accuracy",
            "mean_val_improvement_MAE_vs_baseline", "final_epochs_planned",
            "final_test_operator_accuracy"]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(df[[c for c in show if c in df.columns]].to_string(index=False))
    return df


def run_bitong_folds(args, states: List[str], eval_states: List[str], horizon: int,
                     combo: str, folds: Optional[List[str]] = None) -> Optional[Dict]:
    """covid_TSFM_bitong driver: one finetune per fold on the fold's TRAIN origins, scored on
    the fold's VALIDATION block (A/B/C) or the TEST block (final); then the fold summaries."""
    run_name = run_name_for(args)
    method_root = args.output_root / args.data_mode / args.mode / combo / args.method
    run_root = method_root / run_name
    if args.summarize_only:
        write_bitong_cv_summary(method_root, args.rank_by)
        return None
    if folds is None:
        folds = [f.strip() for f in args.folds.split(",") if f.strip()]
    folds = [("final" if f.lower() == "final" else f.upper()) for f in folds]
    for f in folds:
        if f not in BITONG_FOLD_DIRS:
            sys.exit(f"--folds: unknown fold {f!r}; choose from {', '.join(BITONG_FOLD_DIRS)}")
    fold_root = Path(args.data_dir) if args.data_dir is not None else BITONG_FOLD_ROOT

    # the corpus: generated under covid_TSFM_updated for TX/VT/NJ (all rows split=train, the
    # FINAL training window); a corpus under covid_TSFM_bitong/ takes precedence if present
    run_dir = args.input_root / args.data_mode / args.mode / combo
    if not (run_dir / "accepted.jsonl").exists():
        alt = args.input_root / BITONG_CORPUS_DATA_MODE / args.mode / combo
        if (alt / "accepted.jsonl").exists():
            run_dir = alt
    if not (run_dir / "accepted.jsonl").exists():      # released corpus under data/cot_corpora
        found = P.find_corpus(args.input_root, args.data_mode, args.mode, combo)
        if found is not None:
            run_dir = found[1]
    print(f"input corpus: {run_dir}")
    print(f"fold CSV root: {fold_root}   folds: {','.join(folds)}")
    bins = load_bins(run_dir)
    corpus = load_examples(run_dir, horizon, bins, keep_states=set(states),
                           data_mode=args.data_mode)
    all_train = corpus.get("train", [])
    if corpus.get("val") or corpus.get("test"):
        print(f"  NOTE corpus carries val/test rows ({len(corpus.get('val', []))}/"
              f"{len(corpus.get('test', []))}); only its train rows are used for the folds")
    if not all_train:
        sys.exit("no train rows in the corpus — nothing to finetune.")

    def run_fold(fold: str) -> None:
        fold_dir = bitong_fold_dir(fold_root, fold)
        print(f"\n══════ fold {fold}  (CSVs: {fold_dir}) ══════")
        split_map = bitong_split_map(fold_dir, sorted(set(states) | set(eval_states)))
        train = bitong_filter_train(all_train, split_map, fold)
        if not train:
            print(f"  fold {fold}: no training rows — skipped"); return
        by_split: Dict[str, List[Dict]] = {"train": train}
        orig = load_covid_orig(sorted(set(states) | set(eval_states)), args.data_mode, fold_dir)
        to_covid_original_scale(by_split, orig, args.data_mode)
        # A/B/C: the fold's validation block only (the test block is never used for tuning);
        # final: the test block only (final has no val block)
        eval_splits = ["test"] if fold == "final" else ["val"]
        by_split.update(build_covid_eval_examples(eval_states, eval_splits, horizon, orig,
                                                  args.data_mode, fold_dir))
        args.fold = fold
        args.wandb_name = f"{run_name}_fold{fold}"
        out_dir = run_root / ("final" if fold == "final" else f"fold_{fold}")
        print(f"output dir: {out_dir}")
        run_one(args, out_dir, by_split, horizon, bins)

    def print_summary(s: Dict) -> None:
        print(f"\n=== {run_name}: mean over folds {''.join(s['folds'])} "
              f"(n={s['n_folds']}) → {run_root / 'folds_summary.json'}")
        for k in BITONG_FOLD_METRICS:
            v = s["mean"].get(k)
            if v is not None:
                print(f"    {k:36s} {v:.4f}")
        for st, m in s["mean_by_state"].items():
            print(f"    {st}: dir={m['direction_accuracy'] or float('nan'):.3f} "
                  f"op={m['operator_accuracy'] or float('nan'):.3f} "
                  f"MAE={m['rough_MAE_counts'] or float('nan'):.2f} "
                  f"vs base {m['baseline_MAE_counts'] or float('nan'):.2f} "
                  f"(n_folds={m['n_folds']})")
        print(f"    best epochs {s['best_epochs']} → final_epochs = {s['final_epochs']}")

    # 1) the CV folds: every epoch checkpoint kept, best epoch chosen on the validation block
    cv_folds = [f for f in folds if f != "final"]
    args.keep_all_checkpoints = True
    args.epoch_select = not args.no_epoch_select
    args.epochs_source = None
    for fold in cv_folds:
        run_fold(fold)
    s = None if args.data_only else write_bitong_folds_summary(run_root, run_name)
    if s:
        print_summary(s)

    # 2) final: the full training window for round(mean of the fold best epochs) epochs
    if "final" in folds:
        if args.final_epochs is not None:
            n_ep, src = args.final_epochs, "--final-epochs"
        elif s and s.get("final_epochs") is not None:
            n_ep, src = s["final_epochs"], f"round(mean of fold best epochs {s['best_epochs']})"
        elif args.data_only:
            n_ep, src = (args.epochs if args.method == "original" else args.scale_epochs), \
                "default (--data-only)"
        else:
            have = s["best_epochs"] if s else {}
            sys.exit(f"final: no epoch count available — fold best epochs so far: {have}. "
                     f"Run --folds A,B,C for this configuration first (epoch selection on), "
                     f"or pass --final-epochs N.")
        cap = args.epochs if args.method == "original" else args.scale_epochs
        if n_ep > cap:
            sys.exit(f"final: picked epoch {n_ep} exceeds the epoch cap {cap}")
        # SAME cap/schedule as the folds (loss weights + LR at the same position per epoch);
        # every epoch checkpoint kept; the epoch-n_ep checkpoint becomes THE model
        args.pick_epoch = n_ep
        args.epochs_source = src
        args.keep_all_checkpoints = True
        args.epoch_select = False
        print(f"\nfinal: training the full {cap}-epoch schedule, taking the epoch-{n_ep} "
              f"checkpoint ({src})")
        run_fold("final")
        args.pick_epoch = None
        s = None if args.data_only else write_bitong_folds_summary(run_root, run_name)

    if args.data_only:
        return None
    write_bitong_cv_summary(method_root, args.rank_by)     # ranks every setting run so far
    return s


# ══════════════════════════════════════════════════════════════════════════════
# covid_TSFM_bitong_org: ONE fixed split — train fold B / val B+C / test final
# ══════════════════════════════════════════════════════════════════════════════
ORG_SUMMARY_KEYS = ("best_epoch", "epochs", "epochs_scored", "val_n", "val_n_parsed",
                    "val_direction_accuracy", "val_bin_exact_accuracy_moves",
                    "val_operator_accuracy", "val_rough_MAE_counts", "val_baseline_MAE_counts",
                    "val_improvement_MAE_vs_baseline", "n_test", "n_parsed_lora",
                    "lora_direction_accuracy", "lora_bin_exact_accuracy_moves",
                    "lora_operator_accuracy", "lora_rough_MAE_counts", "baseline_MAE_counts",
                    "improvement_MAE_vs_baseline", "zero_shot_direction_accuracy",
                    "zero_shot_operator_accuracy", "zero_shot_rough_MAE_counts",
                    "lr", "class_weights", "batch_size", "grad_accum", "n_train")


def build_bitong_org_splits(args, corpus_train: List[Dict], states: List[str],
                            eval_states: List[str], horizon: int) -> Dict[str, List[Dict]]:
    """train = the corpus rows whose origin is a TRAIN row of fold B; val = fold B's + fold
    C's validation blocks (built offline from each fold's CSVs); test = the final test block."""
    fold_root = Path(args.data_dir) if args.data_dir is not None else BITONG_FOLD_ROOT
    all_states = sorted(set(states) | set(eval_states))
    train_dir = bitong_fold_dir(fold_root, BITONG_ORG_TRAIN_FOLD)
    print(f"fold CSV root: {fold_root}   train fold {BITONG_ORG_TRAIN_FOLD}, "
          f"val folds {'+'.join(BITONG_ORG_VAL_FOLDS)}, test fold {BITONG_ORG_TEST_FOLD}")
    train = bitong_filter_train(corpus_train, bitong_split_map(train_dir, all_states),
                                BITONG_ORG_TRAIN_FOLD)
    if not train:
        sys.exit("covid_TSFM_bitong_org: no training rows after the fold-B filter")
    by_split: Dict[str, List[Dict]] = {"train": train}
    # all fold CSVs hold the same rows (only the split column differs) -> denorm from `final`
    orig = load_covid_orig(all_states, args.data_mode, bitong_fold_dir(fold_root, "final"))
    to_covid_original_scale(by_split, orig, args.data_mode)
    val: List[Dict] = []
    for f in BITONG_ORG_VAL_FOLDS:
        part = build_covid_eval_examples(eval_states, ["val"], horizon, orig, args.data_mode,
                                         bitong_fold_dir(fold_root, f)).get("val", [])
        for ex in part:
            ex["val_fold"] = f
        per_state = {st: sum(1 for e in part if e["state"] == st) for st in eval_states}
        print(f"  val fold {f}: {len(part)} examples {per_state}")
        val.extend(part)
    seen = set()
    for ex in val:
        if ex["id"] in seen:
            sys.exit(f"covid_TSFM_bitong_org: duplicate val example {ex['id']} across folds")
        seen.add(ex["id"])
    val.sort(key=lambda e: (e["state"], e["reference_date"]))
    test = build_covid_eval_examples(eval_states, ["test"], horizon, orig, args.data_mode,
                                     bitong_fold_dir(fold_root, BITONG_ORG_TEST_FOLD)).get("test", [])
    # no held-out origin may be a training origin (val/test start after fold B's buffer)
    tr_keys = {(e["state"], e["reference_date"]) for e in train}
    leak = [e["id"] for e in val + test if (e["state"], e["reference_date"]) in tr_keys]
    if leak:
        sys.exit(f"covid_TSFM_bitong_org: {len(leak)} val/test origins are also training "
                 f"origins: {leak[:5]}")
    last_train = max(e["reference_date"] for e in train)
    first_val = min(e["reference_date"] for e in val) if val else "-"
    first_test = min(e["reference_date"] for e in test) if test else "-"
    print(f"  split: train {len(train)} rows (origins .. {last_train}) | val {len(val)} "
          f"(from {first_val}) | test {len(test)} (from {first_test})")
    by_split["val"], by_split["test"] = val, test
    return by_split


def write_bitong_org_summary(method_root: Path, rank_by: str) -> Optional[pd.DataFrame]:
    """<combo>/<method>/org_summary.csv: one row per setting (run dir) — best epoch, its val
    metrics, the test metrics — ranked by the validation `rank_by`."""
    rows = []
    for run_root in sorted(p for p in method_root.iterdir() if p.is_dir()):
        mp = run_root / "metrics.json"
        if not mp.exists():
            continue
        m = json.loads(mp.read_text())
        row = {"run_name": run_root.name}
        for k in ORG_SUMMARY_KEYS:
            v = m.get(k)
            row[k] = json.dumps(v) if isinstance(v, (dict, list)) else v
        for st, sm in (m.get("val_by_state") or {}).items():
            row[f"{st}_val_operator_accuracy"] = sm.get("operator_accuracy")
            row[f"{st}_val_improvement_MAE"] = sm.get("improvement_MAE_vs_baseline")
        rows.append(row)
    if not rows:
        print(f"  no metrics under {method_root} yet — nothing to summarize")
        return None
    df = pd.DataFrame(rows)
    df = df.sort_values(rank_by, ascending=rank_by in BITONG_LOWER_IS_BETTER, na_position="last")
    df.insert(0, "rank", range(1, len(df) + 1))
    out = method_root / "org_summary.csv"
    df.to_csv(out, index=False)
    print(f"\n=== org summary ({len(df)} setting(s), ranked by {rank_by}) → {out}")
    show = ["rank", "run_name", "best_epoch", rank_by, "val_direction_accuracy",
            "val_improvement_MAE_vs_baseline", "lora_operator_accuracy",
            "lora_direction_accuracy", "improvement_MAE_vs_baseline"]
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(df[[c for c in show if c in df.columns]].to_string(index=False))
    return df


def main():
    args = parse_args()
    args.scale_start, args.scale_end = parse_schedule(args.scale_schedule)
    bitong = args.data_mode in BITONG_MODES
    if args.text_stub:                      # no-text control arm (read by build_user_prompt)
        global TEXT_STUB
        TEXT_STUB = NOTEXT_STUB
        print(f"[text-stub] every prompt's report block is replaced by {NOTEXT_STUB!r}")
    if args.reasoning_style != "cot":       # target reasoning ablation (read by build_target_text)
        global REASONING_STYLE
        REASONING_STYLE = args.reasoning_style
        if args.reasoning_style == "residual" and args.data_mode in NORES_MODES:
            sys.exit("--reasoning-style residual restates the recent_residual_block, which the "
                     f"{args.data_mode} prompt does not show")
        print(f"[reasoning-style] assistant targets use style {args.reasoning_style!r}")
    if args.max_seq_len is None:
        args.max_seq_len = BITONG_MAX_SEQ_LEN if bitong else MAX_SEQ_LEN
    # per-mode defaults for the speed knobs (covid_TSFM_bitong: 12 epochs, 2x8, batched infer)
    args.epochs_default = (BITONG_EPOCHS if bitong
                           else (EPOCHS if args.method == "original" else SCALE_EPOCHS))
    if args.epochs is None:
        args.epochs = BITONG_EPOCHS if bitong else EPOCHS
    if args.scale_epochs is None:
        args.scale_epochs = BITONG_EPOCHS if bitong else SCALE_EPOCHS
    if args.batch_size is None:
        args.batch_size = BITONG_BATCH_SIZE if bitong else BATCH_SIZE
    if args.grad_accum is None:
        args.grad_accum = BITONG_GRAD_ACCUM if bitong else GRAD_ACCUM
    if args.infer_batch_size is None:
        args.infer_batch_size = BITONG_INFER_BATCH_SIZE if bitong else 1
    if args.nochange_band is None:      # covid: ±20 index pts; relative for chronos2_full; else rate
        args.nochange_band = (NOCHANGE_ABS_INDEX_BAND if args.data_mode in COVID_MODES
                              else NOCHANGE_REL_BAND if args.mode == "chronos2_full"
                              else NOCHANGE_BAND)
    if (args.data_mode in FINANCE_MODES or args.data_mode in COVID_MODES) \
            and args.mode != "chronos2_full":
        sys.exit(f"--data-mode {args.data_mode} requires --mode chronos2_full")
    states_arg = args.states or DATA_MODE_STATES[args.data_mode]
    states = [s.strip().upper() for s in states_arg.split(",") if s.strip()]
    if not states:
        sys.exit("--states must list at least one state")
    eval_states_arg = (args.eval_states or DATA_MODE_EVAL_STATES.get(args.data_mode)
                       or states_arg)
    eval_states = [s.strip().upper() for s in eval_states_arg.split(",") if s.strip()]
    horizon = horizon_for(args.data_mode, args.mode)
    combo = f"{run_dir_name(states)}-{args.corpus_model.lower()}"
    args.class_weights_spec = args.class_weights
    args.class_weights = parse_class_weights(args.class_weights)
    if args.data_mode in BITONG_CV_MODES:
        # expanding-window folds: one run per fold (+ final), then the fold summaries
        print(f"=== operator finetune  data-mode={args.data_mode}  mode={args.mode}  "
              f"method={args.method}  states={combo}  eval-states={','.join(eval_states)}  "
              f"horizon={horizon}  folds={args.folds}  model={Path(args.model).name}  "
              f"epochs<={args.scale_epochs if args.method == 'scale' else args.epochs}  "
              f"batch={args.batch_size}x{args.grad_accum}  infer-batch={args.infer_batch_size}  "
              f"epochs scored: skip {args.epoch_skip} / stride {args.epoch_stride} ===")
        run_bitong_folds(args, states, eval_states, horizon, combo)
        return
    # corpus dir: <input-root>/<data_mode>/... ; the influenza corpora live under
    # influenza_TSFM/ (mirrored on the output side, next to the existing runs) — same
    # fallback rule as finetune_residual_cot.py
    data_dir = args.data_mode
    run_dir = args.input_root / data_dir / args.mode / combo
    if not (run_dir / "accepted.jsonl").exists() and args.data_mode in INFLUENZA_MODES:
        alt = args.input_root / "influenza_TSFM" / args.mode / combo
        if (alt / "accepted.jsonl").exists():
            data_dir = "influenza_TSFM"          # outputs mirror the corpus dir name
            run_dir = alt
    if not (run_dir / "accepted.jsonl").exists() and args.data_mode in BITONG_MODES:
        # the TX/VT/NJ corpus was generated under covid_TSFM_updated (outputs stay under
        # the bitong mode's own subtree)
        alt = args.input_root / BITONG_CORPUS_DATA_MODE / args.mode / combo
        if (alt / "accepted.jsonl").exists():
            run_dir = alt
    if not (run_dir / "accepted.jsonl").exists():      # released corpus under data/cot_corpora
        found = P.find_corpus(args.input_root, args.data_mode, args.mode, combo)
        if found is not None:
            data_dir = found[0] if args.data_mode in INFLUENZA_MODES else data_dir
            run_dir = found[1]
    org = args.data_mode in BITONG_ORG_MODES
    if org:
        method_root = args.output_root / data_dir / args.mode / combo / args.method
        if args.summarize_only:
            write_bitong_org_summary(method_root, args.rank_by)
            return
        # fixed split + best-epoch-on-val selection (the fold machinery of covid_TSFM_bitong)
        args.keep_all_checkpoints = True
        args.epoch_select = not args.no_epoch_select
        args.fold = f"train{BITONG_ORG_TRAIN_FOLD}_val{''.join(BITONG_ORG_VAL_FOLDS)}_test{BITONG_ORG_TEST_FOLD}"
    print(f"=== operator finetune  data-mode={args.data_mode}  mode={args.mode}  "
          f"method={args.method}  states={combo}  eval-states={",".join(eval_states)}  "
          f"horizon={horizon}  "
          f"model={Path(args.model).name}"
          + (f"  epochs<={args.scale_epochs if args.method == 'scale' else args.epochs}  "
             f"batch={args.batch_size}x{args.grad_accum}  infer-batch={args.infer_batch_size}  "
             f"epochs scored: skip {args.epoch_skip} / stride {args.epoch_stride}" if org else "")
          + " ===")
    print(f"input corpus: {run_dir}")
    bins = load_bins(run_dir)
    by_split = load_examples(run_dir, horizon, bins, keep_states=set(states),
                             data_mode=args.data_mode)
    if args.data_mode in FINANCE_MODES:
        orig = load_finance_orig(states)
        to_original_scale(by_split, orig)          # USD metrics for any corpus-loaded rows
        if not by_split.get("val") and not by_split.get("test"):
            # train-only corpus: build val/test evaluation examples offline
            by_split.update(build_finance_eval_examples(states, ["val", "test"], horizon,
                                                        orig, args.data_mode))
    elif org:
        by_split = build_bitong_org_splits(args, by_split.get("train", []), states,
                                           eval_states, horizon)
    elif args.data_mode in COVID_MODES:
        orig = load_covid_orig(sorted(set(states) | set(eval_states)), args.data_mode,
                               args.data_dir)
        to_covid_original_scale(by_split, orig, args.data_mode)   # original-scale metrics
        if not by_split.get("val") and not by_split.get("test"):
            # train-only corpus: build val/test evaluation examples offline
            # (eval_states may differ from the train states, e.g. GA,FL vs GA,FL,MS)
            by_split.update(build_covid_eval_examples(eval_states, ["val", "test"], horizon,
                                                      orig, args.data_mode, args.data_dir))
    run_name = run_name_for(args)
    out_dir = args.output_root / data_dir / args.mode / combo / args.method / run_name
    print(f"output dir: {out_dir}")
    args.wandb_name = run_name
    run_one(args, out_dir, by_split, horizon, bins)
    if org and not args.data_only:
        write_bitong_org_summary(out_dir.parent, args.rank_by)


if __name__ == "__main__":
    main()
