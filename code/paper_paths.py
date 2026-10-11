"""Every filesystem location used by the paper-content code, in one place.

The layout is relative to the paper-content root (the directory holding data/, code/,
prompts/). Override pieces with environment variables when the heavy outputs or the
model checkpoints live elsewhere:

    PAPER_ROOT              the paper-content directory (default: parent of code/)
    PAPER_OUTPUT_ROOT       where CoT corpora / finetune runs / zero-shot runs are written
                            (default: <PAPER_ROOT>/output)
    PAPER_SCRATCH_ROOT      physical location of the CoT corpora (default: same as the
                            output dir, i.e. no scratch symlinks)
    PAPER_MODEL_ROOT        directory holding Qwen3-8B/ and chronos-2/ (default:
                            <PAPER_ROOT>/checkpoints)
    PAPER_CREDENTIALS_DIR   directory holding gemini_api_key.json / openai_api_key.json
                            (default: <PAPER_ROOT>/credentials; never committed).
                            GEMINI_API_KEY / OPENAI_API_KEY environment variables take
                            precedence over the files.
"""
from __future__ import annotations

import os
from pathlib import Path

PAPER_ROOT = Path(os.environ.get("PAPER_ROOT", Path(__file__).resolve().parents[1]))

# ── data ──────────────────────────────────────────────────────────────────────
DATA_ROOT = PAPER_ROOT / "data"
INFLUENZA_DATA_DIR = DATA_ROOT / "Influenza"                 # {STATE}.csv, 9 columns
COVID_DATA_ROOT = DATA_ROOT / "COVID"                        # {STATE}.csv (19 columns: values, text, splits, truth)
COVID_FOLD_ROOT = COVID_DATA_ROOT                            # finetune derives each fold's view from the same files
COVID_FINAL_DATA_DIR = COVID_DATA_ROOT                       # CoT generation reads the `split` column (final setting)
COVID_CHRONOS2_DIR = COVID_DATA_ROOT / "chronos2"            # Chronos-2 prediction caches (code/chronos2 writes here)
COVID_TEXT_DIR = COVID_DATA_ROOT / "text_reports"            # optional raw reports (not shipped; the text is in the CSVs)
RAW_DATA_DIR = DATA_ROOT / "raw"                             # NHSN admissions (COVID long table, influenza panel), FluSight populations, split json
DATA_BUILD_RAW_DIR = RAW_DATA_DIR                            # alias
TARGET_HOSP_CSV = RAW_DATA_DIR / "target-hospital-admissions.csv"   # FluSight (population)
INFLUENZA_PANEL_CSV = RAW_DATA_DIR / "influenza_admissions_panel.csv"   # NHSN weekly influenza admissions, all jurisdictions
# raw per-state influenza report text is NOT shipped separately (it is embedded in data/Influenza/{STATE}.csv)
RAW_SOURCES_DIR = RAW_DATA_DIR / "influenza_sources"

# ── prompts ───────────────────────────────────────────────────────────────────
PROMPT_ROOT = PAPER_ROOT / "prompts"
PROMPT_DIR_INFLUENZA = PROMPT_ROOT / "influenza"             # CoT generation + judges + finetune prompt (influenza)
PROMPT_DIR_COVID = PROMPT_ROOT / "covid"                     # CoT generation + judges + finetune prompt (COVID)
PROMPT_DIR_DIRECT = PROMPT_ROOT / "zero_shot_direct"         # influenza/, covid/ (zero_shot_api_direct.py)

# ── outputs ───────────────────────────────────────────────────────────────────
OUTPUT_ROOT = Path(os.environ.get("PAPER_OUTPUT_ROOT", PAPER_ROOT / "output"))
COT_OUTPUT_DIR = OUTPUT_ROOT / "cot_corpora"                 # <data_mode>/<mode>/<STATES>-<provider>/
# the released corpora (data/cot_corpora/<influenza|covid>/<STATES>-<provider>/); finetune.py /
# zero_shot_api_notext.py fall back to this tree when a corpus is not found under the output dir
SHIPPED_CORPORA_DIR = DATA_ROOT / "cot_corpora"
COT_SCRATCH_ROOT = Path(os.environ.get("PAPER_SCRATCH_ROOT", COT_OUTPUT_DIR))
FINETUNE_OUTPUT_ROOT = OUTPUT_ROOT / "finetune"              # <data_mode>/<mode>/<combo>/<method>/<run>/
ZERO_SHOT_OUTPUT_ROOT = OUTPUT_ROOT / "zero_shot"            # <data_mode>/<mode>/<combo>/<variant>/<model_tag>/

# ── models / credentials ──────────────────────────────────────────────────────
MODEL_ROOT = Path(os.environ.get("PAPER_MODEL_ROOT", PAPER_ROOT / "checkpoints"))
QWEN_MODEL_PATH = str(MODEL_ROOT / "Qwen3-8B")
CHRONOS2_MODEL_PATH = str(MODEL_ROOT / "chronos-2")
CREDENTIALS_DIR = Path(os.environ.get("PAPER_CREDENTIALS_DIR", PAPER_ROOT / "credentials"))
GEMINI_CRED_PATH = CREDENTIALS_DIR / "gemini_api_key.json"
OPENAI_CRED_PATH = CREDENTIALS_DIR / "openai_api_key.json"


def find_corpus(input_root, data_mode: str, mode: str, combo: str):
    """<root>/<data_mode or its aliases>/<mode>/<combo>/accepted.jsonl under the output root
    first, then under the shipped corpora. Returns (data_dir_name, run_dir) or None."""
    aliases = {"influenza": ("influenza", "influenza_TSFM")}
    names = aliases.get(data_mode, (data_mode,))
    if str(data_mode).startswith("covid"):
        names = (data_mode, "covid_TSFM_updated", "covid")
    for root in (Path(input_root), SHIPPED_CORPORA_DIR):
        for dm in names:
            for d in (root / dm / mode / combo, root / dm / combo):   # pipeline layout, then the flat release layout
                if (d / "accepted.jsonl").exists():
                    return dm, d
    return None


def prompt_dir_for(data_mode: str) -> Path:
    """The (flat) prompt directory of a data mode: covid* modes -> prompts/covid, everything
    else -> prompts/influenza."""
    return PROMPT_DIR_COVID if str(data_mode).startswith("covid") else PROMPT_DIR_INFLUENZA
