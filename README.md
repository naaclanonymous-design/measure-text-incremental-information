# Text-aware correction of a time-series foundation model: data, prompts and code

Self-contained release behind the paper's influenza and COVID experiments: the two datasets,
the prompt templates that produced the reported results, the chain-of-thought (CoT) generation
pipeline, the LoRA finetuning code (SCALe and original loss) and the zero-shot API tests
(Gemini-3.1-pro / GPT-5.4, with and without text).

```
paper-content/
├── README.md, requirements.txt, .gitignore
├── data/                     DATA ONLY (no scripts): state CSVs, report text, Chronos-2 forecasts
│   ├── Influenza/            {FL,GA,IA,MS,NJ,NY,USA,VA}.csv (weekly influenza admissions, Chronos-2
│   │                         forecasts and the weekly report text, all inside the CSV),
│   │                         chronos2/chronos2_influenza_predictions.csv                 -> README
│   ├── COVID/                {AZ,IA,MI,MS,NJ,NY,TX,VA,VT}.csv (26-week z-score index, Chronos-2
│   │                         forecasts, report text and all split labels inside the CSV),
│   │                         chronos2/*.csv                                               -> README
│   ├── raw/                  NHSN admissions (COVID long table, influenza panel), FluSight
│   │                         populations, the split-date json                           -> README
│   └── cot_corpora/          CoT corpora: <STATES>-gemini = the plain 6-stage pipeline output,
│                             <STATES>-gemini-final = the corpus of the reported run           -> README
├── prompts/                  one flat directory per prompt set                             -> README
│   ├── influenza/            CoT generation + judges + finetune prompt (influenza, 153.7 MAE)
│   ├── covid/                CoT generation + judges + finetune prompt (COVID)
│   └── zero_shot_direct/     influenza/, covid/: prompts of zero_shot_api_direct.py
├── code/
│   ├── paper_paths.py        every path used by the code (override with env vars)
│   ├── data_build/           build_covid_dataset.py: the 26-week z-score normalisation and the
│   │                         COVID CSV layout (reads data/raw/ + data/COVID/chronos2/)
│   ├── chronos2/             how Chronos-2 is called per state: chronos2_covid.py (univariate),
│   │                         chronos2_influenza.py (multivariate), run_chronos2.sbatch
│   ├── cot_generation/       cot_common.py, step1_generate/filter.py, step2_generate/filter.py,
│   │                         build_global_bins.py, run_cot_pipeline.sh (the 6 stages)
│   ├── data_augmentation/    how the -gemini-final corpora were derived: influenza training-set
│   │                         expansion, COVID abstain fix + nores rewrite (+ their prompts)   -> README
│   ├── finetune/             finetune.py (--method scale | original), train_grounding.py,
│   │                         apply_state_gate.py, finetune_residual_cot.py, 4 sbatch scripts
│   └── zero_shot/            zero_shot_api_notext.py (operator test, --text keep|drop),
│                             zero_shot_api_direct.py, ground_zero_shot_api.py, run_zero_shot.sh
├── credentials/              API keys go here (git-ignored; see credentials/README.md)
├── checkpoints/              Qwen3-8B/ and chronos-2/ (git-ignored; symlink or download)
└── output/                   created at run time for everything the code writes (git-ignored)
```

## The task in one paragraph

For every state and forecast origin `t` (the last fully observed week) the model sees the 12
weekly values `t-11..t`, the Chronos-2 forecast for the target week(s), and the weekly
surveillance reports of weeks `t-4..t` (date-scrubbed). A teacher LLM (Gemini-3.1-pro) writes
CoTs that decide whether the Chronos-2 forecast should be revised *up*, *down* or *kept*, and
by which *size class*; offline checks and LLM judges keep only correct, quote-faithful,
grounded CoTs. An open 8B model (Qwen3-8B, LoRA) is then finetuned to emit the same
reasoning + `Direction:` / `Size:` operator, and its operator is turned back into a numerical
forecast (baseline ± the class's cold-start magnitude) and scored in original units against
Chronos-2 on the same test weeks.

| dataset | series | target | no-change band | states (CoT / eval) | test block |
|---|---|---|---|---|---|
| Influenza | NHSN weekly influenza admissions (counts) | weeks t+1..t+3 | relative ±10 % of the baseline | GA, NY, NJ | 2025-05-31 .. 2026-03-28 (101 origins) |
| COVID | NHSN weekly COVID admissions, 100 × causal 26-week z-score | week t+3 only | absolute ±20 index points | TX, VT, NJ | 2025-10-04 .. 2026-03-28 origins (78 rows) |

## Quick start

```bash
# 0. environment
pip install -r requirements.txt
export PYTHON=$(which python)              # the run scripts call $PYTHON (default: python)
ln -s /path/to/Qwen3-8B  checkpoints/Qwen3-8B
ln -s /path/to/chronos-2 checkpoints/chronos-2
export GOOGLE_CLOUD_PROJECT=<gcp-project>  # Gemini via Vertex AI (gcloud auth application-default login)
export OPENAI_API_KEY=...                  # GPT-5.4 (zero-shot tests only)

# 1. data (already built; only needed to add states or after raw updates)
cd code/chronos2
sbatch run_chronos2.sbatch covid        # GPU; Chronos-2 per state -> data/COVID/chronos2/ (missing states only)
sbatch run_chronos2.sbatch influenza    # GPU; Chronos-2 per state -> data/Influenza/chronos2/
cd ../..
python code/data_build/build_covid_dataset.py   # z26w normalisation -> data/COVID/{STATE}.csv

# 2. CoT generation (6 stages, Gemini) -> output/cot_corpora/<data_mode>/chronos2_full/<STATES>-gemini/
code/cot_generation/run_cot_pipeline.sh influenza 8
code/cot_generation/run_cot_pipeline.sh covid 4
#    (skip to use the released corpora in data/cot_corpora/ — all code falls back to them;
#     --corpus-model gemini = plain pipeline output, gemini-final = the reported run's corpus,
#     derived from the plain output by code/data_augmentation/, see its README)

# 3. finetuning (GPU, ms-swift LoRA) -> output/finetune/...
cd code/finetune
sbatch run_finetune_influenza_scale.sbatch      # SCALe loss (the reported configuration)
sbatch run_finetune_influenza_original.sbatch   # original loss
sbatch run_finetune_covid_scale.sbatch          # SCALe loss + per-state gate (the reported configuration)
sbatch run_finetune_covid_original.sbatch       # original loss

# 4. zero-shot API tests (Gemini-3.1-pro + GPT-5.4, with / without text) -> output/zero_shot/...
code/zero_shot/run_zero_shot.sh influenza both gemini,gpt
code/zero_shot/run_zero_shot.sh covid     both gemini,gpt
```

All paths come from `code/paper_paths.py`; set `PAPER_OUTPUT_ROOT`, `PAPER_MODEL_ROOT`,
`PAPER_CREDENTIALS_DIR` or `PAPER_SCRATCH_ROOT` to move the heavy directories. Every stage is
resumable.

## Notes

* **No API keys are stored anywhere in this tree.** Keys come from the environment or the
  git-ignored `credentials/` directory (`credentials/README.md`).
* **Data formats are the ones the code reads**: `data/Influenza/{STATE}.csv` ↔ `--data-mode
  influenza`; `data/COVID/{STATE}.csv` ↔ `--data-mode covid_TSFM_updated` (generation, `split`
  column) and `covid_TSFM_bitong` (finetuning; each fold's view is derived at run time from the
  `fold_A` / `fold_B` / `fold_C` / `split` columns into `output/covid_folds/`).
