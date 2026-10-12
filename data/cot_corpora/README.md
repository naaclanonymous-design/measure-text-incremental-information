# Released CoT corpora

Each corpus is `accepted.jsonl` + `bins.json`: one JSON object per training row, self-contained
(the four rendered input blocks, `x_hist / x_hat / x_gt`, the labels, the Step-1 basis CoT, the
Step-2 size CoT for moves, the run's bins). `finetune.py` and `zero_shot_api_notext.py` resolve
`--corpus-model <tag>` to `<dataset>/<STATES>-<tag>/` here when the corpus is not found under
`output/cot_corpora/`.

| corpus | rows | what it is |
|---|---|---|
| `influenza/GA_NY_NJ-gemini/` | 523 = train 310 (261 move / 17 no_change / 245 abstain overall) + val 112 + test 101, one row per origin | the plain output of the 6-stage pipeline (`code/cot_generation`, `prompts/influenza/`, Gemini-3.1-pro teacher, Gemini-2.5-flash judges) |
| `influenza/GA_NY_NJ-gemini-final/` | 866 = train 653 (468 move / 41 no_change / 144 abstain) + the same val 112 + test 101 | **the corpus of the reported influenza run (test MAE 153.7)**: every passing Step-1 direction CoT of a train origin (round 1 and the momentum round 2) becomes its own training row sharing the origin's Step-2 size CoT (`source_id`, `basis_candidate_index` point to the origin row); val/test rows are unchanged |
| `covid/TX_VT_NJ-gemini/` | 358 train rows: 102 move / 87 no_change / 169 abstain | the plain output of the 6-stage pipeline (`prompts/covid/`, same models; origins 2023-05-06 .. 2025-09-06 of TX/VT/NJ) |
| `covid/TX_VT_NJ-gemini-final/` | 257 train rows: 92 move / 85 no_change / 80 abstain | **the corpus of the reported COVID run (test MAE 31.9 after the per-state gate)**, derived from the plain output in two Gemini passes: (1) every kept abstain origin receives an input-specific `abstain_cot` (the best faithful "no change" candidate, or a regenerated and judged hard negative, `abstain_group` / `hard_negative`), 89 abstain rows without one are dropped; (2) every Step-1 / Step-2 CoT is rewritten so that it no longer cites the recent-error block (`*_orig` keeps the text before the rewrite, `rewrite_status` / `rewrite_checks` the judge outcome). `prompts/covid/finetune_prompt.txt` is that matching prompt without the recent-error item (the plain `-gemini` corpus still cites the error block, so it does not match the released COVID finetune prompt). |

The `-final` corpora are produced from the plain output by `code/data_augmentation/` (see its README).
The released sbatch scripts in `code/finetune/` use `--corpus-model gemini-final`; pass
`--corpus-model gemini` to train on the plain pipeline output instead. `bins.json` is identical
within each dataset.

Key fields: `id` (`{split}_{STATE}_{origin}`, with a `__c<k>` candidate suffix in the influenza
final corpus), `state`, `split`, `reference_date`, `date_status` (move / no_change / abstain),
`operator` (correction / no-change / no-correction), `direction_labels`, `delta_cold_start`,
`basis_directions`, `size_names`, `step1_basis_cot`, `step2_size_cot` (moves), `abstain_cot`
(COVID final, abstain rows), `history_block`, `baseline_block`, `recent_residual_block`,
`weekly_text_block`, `bins`.
