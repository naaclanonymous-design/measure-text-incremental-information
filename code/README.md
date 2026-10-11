# Code

| directory | purpose |
|---|---|
| `data_build/` | `build_covid_dataset.py`: the causal 26-week z-score normalisation of the NHSN COVID admissions and the 19-column `data/COVID/{STATE}.csv` layout (reads `data/raw/` + `data/COVID/chronos2/`; the report text is reused from the existing CSVs) |
| `chronos2/` | how Chronos-2 is called for every state: `chronos2_covid.py` (univariate, raw + z26w series → `data/COVID/chronos2/`), `chronos2_influenza.py` (multivariate: target state + the other states + USA as covariates → `data/Influenza/chronos2/`), `run_chronos2.sbatch covid\|influenza` |
| `cot_generation/` | the 6-stage CoT pipeline |
| `finetune/` | LoRA finetuning (SCALe / original loss), grounding, per-state gate |
| `zero_shot/` | hosted-model (Gemini-3.1-pro / GPT-5.4) tests with and without text |

All scripts import `paper_paths.py` (one level up) for their locations and add the sibling
directories to `sys.path`, so every script runs from its own directory or by absolute path.
`$PYTHON` (default `python`) is the interpreter the shell / sbatch wrappers call.

## `cot_generation/` — the 6-stage CoT pipeline (teacher: Gemini-3.1-pro, judges: Gemini-2.5-flash)

`run_cot_pipeline.sh influenza|covid [WORKERS]` runs the six stages in order; each stage is
also a stand-alone resumable script (`--dry-run` builds the samples and prompts without API
calls, `--help` lists every knob):

| stage | script | what it does |
|---|---|---|
| 1 | `step1_generate.py` | builds one sample per train origin (12-week history, Chronos-2 baseline, recent 3-week-ahead baseline errors, the reports of weeks t−4..t, date-scrubbed; ≥1 report required), samples k=5 direction-only CoTs at T=0.9 |
| 2 | `step1_filter.py` | offline direction correctness vs the realised residual (±band), self-consistency α₁ ≥ 0.6 else abstain, verbatim quote check, Flash faithfulness judge, grounding judge (score ≥ 3); the best passer becomes the origin's basis CoT; status move / no_change / abstain |
| 3 | `step1_generate.py --round 2` | re-samples only the abstained origins with a momentum paragraph (six deterministic signals rendered by the LLM) injected as item 5 |
| 4 | `step1_filter.py --round 2` | filters the retries and merges both rounds (abstain only if both rounds abstained) |
| 5 | `step2_generate.py` | designs the global size bins (quantiles of the pooled moving |residual|, first edge = the band) and samples k=5 size-class CoTs per move origin with the direction fixed |
| 6 | `step2_filter.py` | grounds every candidate at its class's cold-start magnitude **δ = sign(d)·m_init(bin)**, keeps it iff the utility U = 2δr − δ² > 0 (no overshoot), runs the faithfulness + size-grounding judges, fixes the highest-U size per origin and writes `accepted.jsonl` (move / no_change / abstain records with all input blocks) |

`build_global_bins.py` rebuilds / inspects the bins of a run. `cot_common.py` holds the data
loaders (`load_influenza_bundle`, `load_covid_bundle`), block renderers, date scrubbing,
judges' parsers, the LLM clients (Vertex AI Gemini, OpenAI) and the run-dir layout
`output/cot_corpora/<data_mode>/chronos2_full/<STATES>-<provider>/`. Wrapper logs go to `output/logs/`.

Data modes used here: `influenza` (reads `data/Influenza/{STATE}.csv`, prompts/influenza,
relative ±10 % band, 3 target weeks) and `covid_TSFM_updated` (reads
`data/COVID/{STATE}.csv`, prompts/covid, absolute ±20-point band, target =
origin + 3 only). `--model GPT` switches the teacher and
judges to OpenAI models.

## `finetune/` — LoRA finetuning of Qwen3-8B (ms-swift) and evaluation

`finetune.py --data-mode {influenza, covid_TSFM_bitong, ...} --mode chronos2_full
--method {scale, original} --corpus-model <suffix> ...`

* **Target**: `Reasoning: <Step-1 basis CoT> [<Step-2 size CoT>]` then `Direction: ...` /
  `Size: ...`; the prompt is `prompts/<set>/finetune_prompt.txt` with the four input blocks.
* `--method scale`: the SCALe loss — the target is split at the first `Direction:` into a
  reasoning segment and an answer segment, each averaged over its own tokens, and combined
  with cosine-scheduled weights `a:b>c:d` (`--scale-schedule`, start → end).
* `--method original`: plain completion loss with the Direction/Size tokens up-weighted
  (`--operator-loss-weight`, 16 in the reported ablations).
* Evaluation: parse Direction / Size, categorical accuracies, and the **rough forecast**
  `ŷ = baseline ± lower_edge(bin)` (× |baseline| for relative bins) back-transformed to
  admissions → MAE vs Chronos-2 on the same rows (`metrics.json`, `predictions.csv`), plus a
  zero-shot pass of the bare base model on the same prompts.
* COVID modes `covid_TSFM_bitong` (expanding folds A, B, C + final; `--folds`) and
  `covid_TSFM_bitong_org` (one fixed split) read `data/COVID/{STATE}.csv` and derive each
  fold's view from its `fold_A` / `fold_B` / `fold_C` / `split` columns (`bitong_fold_dir`,
  cached under `output/covid_folds/`); each fold's epoch is chosen by `--epoch-metric` on its
  validation block and the final model trains for the mean best epoch. `apply_state_gate.py`
  then keeps a state's moves on test only if the model beat the baseline on that state's
  pooled validation rows.
* `train_grounding.py <run_dir>` fits per-predicted-bin magnitudes on validation and adds
  `lora_grounded_*` fields; `finetune_residual_cot.py` is the numbers-only ablation
  (`--reasoning-style residual|none`, `--text drop`).

sbatch wrappers (submit from `code/finetune/`; slurm output lands next to the script): `run_finetune_influenza_scale.sbatch`,
`run_finetune_influenza_original.sbatch`, `run_finetune_covid_scale.sbatch`,
`run_finetune_covid_original.sbatch`. They train on the released original corpora; the reported
numbers used derived training corpora / a prompt variant that are not part of this release (see
the root README). Append `--data-only` to only write the train/val/test
prompt files. Defaults: LoRA r 8 / α 16 / dropout 0.05, batch 1 × 16, max 16 384 tokens
(24 576 for COVID), bf16, seed 42, greedy inference with 1 536 new tokens.

## `zero_shot/` — hosted-model tests on the same prompts

* `zero_shot_api_notext.py` — the **operator** zero-shot test: Gemini-3.1-pro (thinking level
  *low*, its floor) and GPT-5.4 (`reasoning_effort=none`) answer the finetune prompt of the
  test rows. `--text keep` = history + baseline + recent errors + the last 5 weeks of reports; `--text drop` = the report block replaced by the notext stub. Scored
  exactly like the finetuned operator (direction / size accuracy, rough MAE vs Chronos-2).
  `run_zero_shot.sh influenza|covid [keep|drop|both] [gemini,gpt]`.
* `zero_shot_api_direct.py` — the **direct** zero-shot test: no baseline in the prompt, the
  model outputs the three admission values itself (`prompts/zero_shot_direct/{influenza,covid}/`).
* `ground_zero_shot_api.py` — fits the per-bin magnitudes on the `--split val` responses and
  adds `grounded_*` MAE fields to the test metrics.

Outputs: `output/zero_shot/<data_mode>/chronos2_full/<combo>/<variant>/<model_tag>/{responses.jsonl, predictions.csv, metrics.json}`.
