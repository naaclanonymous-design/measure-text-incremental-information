# Released CoT corpora (original pipeline output, no augmentation)

`accepted.jsonl` + `bins.json` exactly as written by stage 6 of `code/cot_generation`
(`step2_filter.py`): one JSON object per origin, self-contained (the four rendered input
blocks, `x_hist / x_hat / x_gt`, the labels, the Step-1 basis CoT, the Step-2 size CoT for
moves, the run's bins). `finetune.py` and `zero_shot_api_notext.py` fall back to this
directory when a corpus is not found under `output/cot_corpora/` (`--corpus-model gemini`).

| corpus | rows | content |
|---|---|---|
| `influenza/GA_NY_NJ-gemini/` | 523 = train 310 (261 move / 17 no_change / 245 abstain overall) + val 112 + test 101, one row per origin | `prompts/influenza/`, Gemini-3.1-pro teacher, Gemini-2.5-flash judges |
| `covid/TX_VT_NJ-gemini/` | 358 train rows: 102 move / 87 no_change / 169 abstain | `prompts/covid/`, same models; origins 2023-05-06 .. 2025-09-06 of TX/VT/NJ |

These are the corpora the released finetune scripts train on. (The paper's reported runs used
derived versions — an expanded influenza corpus and a repaired/rewritten COVID corpus — that are
not part of this release; see the root README.)

Key fields: `id` (`{split}_{STATE}_{origin}`), `state`, `split`, `reference_date`,
`date_status` (move / no_change / abstain), `operator`, `basis_directions`, `size_names`,
`step1_basis_cot`, `step2_size_cot` (moves), `history_block`, `baseline_block`,
`recent_residual_block`, `weekly_text_block`, `bins`.
