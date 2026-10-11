# Prompt templates

Only the prompt sets behind the two reported results are released, one **flat directory per
set**. The code picks the directory by data mode (`paper_paths.prompt_dir_for`): `influenza` →
`prompts/influenza`, every covid mode → `prompts/covid`; `--prompt-dir` overrides it.

## `influenza/` — influenza (data mode `influenza`)

| file | used by | role |
|---|---|---|
| `step1_direction.txt` | step1_generate (rounds 1 + 2) | direction CoT: history, baseline, recent errors, reports (+ momentum paragraph in round 2) → `Directions: ...` |
| `momentum_summary.txt` | step1_generate --round 2 | renders the deterministic momentum signals into the paragraph injected as item 5 |
| `step1_faithfulness_flash.txt` | step1_filter | semantic quote-faithfulness judge (Gemini Flash, T=0) |
| `step1_grounding_pro.txt` | step1_filter | direction-grounding judge, `Score: 1-5` |
| `step2_magnitude.txt` | step2_generate | size-class CoT with the direction fixed → `Evidence:` line + `Sizes: ...` |
| `step2_faithfulness_flash.txt` | step2_filter | quote-faithfulness judge for the size CoT |
| `step2_grounding_pro.txt` | step2_filter | size-grounding judge |
| `finetune_prompt.txt` | finetune.py, zero_shot_api_notext.py | the user-prompt body (items 1-4 + task + output format); data-mode wording from `finetune.PROMPT_WORDING` |

## `covid/` — COVID (data mode `covid_TSFM_updated` for generation, `covid_TSFM_bitong` for finetuning)

The same eight files, worded for the 100 × 26-week z-score index, the absolute ±20-point band
and the single far-horizon target (origin + 3); `finetune_prompt.txt` holds the four input items
(history, baseline, recent 3-week-ahead baseline errors, reports) followed by the task and the
output format.

## `zero_shot_direct/` — the direct-forecast zero-shot test

Used only by `code/zero_shot/zero_shot_api_direct.py`, a secondary zero-shot experiment in which
the hosted model (Gemini-3.1-pro / GPT-5.4) is given the history and the reports but **no
Chronos-2 baseline**, and must output the three admission values itself
(`Forecast: w1, w2, w3`) instead of revising the baseline. `influenza/` and `covid/` each hold
the system prompt, the user template and an optional recent-errors item.

Placeholders are `{name}` tokens filled by `cot_common.fill_template` / `finetune._fill`
(`{history_block}`, `{baseline_block}`, `{recent_residual_block}`, `{weekly_text_block}`,
`{momentum_block}`, `{bin_definitions}`, `{nochange_band}`, ...).
