# `data_augmentation/` — how the released `-gemini-final` corpora were derived

The 6-stage pipeline (`code/cot_generation`) writes one row per origin
(`data/cot_corpora/<dataset>/<STATES>-gemini/`). The corpora the reported runs train on
(`<STATES>-gemini-final/`) are derived from those runs by the scripts here. All of them leave
the source untouched and write a new `accepted.jsonl` + `bins.json` in the same schema, so the
result is a drop-in `--corpus-model` tag for `finetune.py`.

| dataset | script | what it does | API | output tag |
|---|---|---|---|---|
| Influenza | `build_complete_corpus.py` | **training-set expansion**: every Step-1 direction CoT that passed stage 2 / stage 4 (round 1 and the momentum round 2, from `step1_filtered_merged.jsonl`) becomes its own training row sharing the origin's Step-2 size CoT (`id = <origin>__c<k>`, `source_id`, `basis_candidate_index`); abstain origins keep one row; val / test stay one row per origin | none | `GA_NY_NJ-gemini-complete` = `data/cot_corpora/influenza/GA_NY_NJ-gemini-final` (310 → 653 train rows) |
| COVID | `build_abstain_fix_corpus.py` | **abstain fix**: the 169 abstain origins all carried one canned reasoning. Each is classified from its rejected Step-1 candidates (`step1_filtered_merged.jsonl`, `step1_generated*.jsonl`): group 1 (every faithful candidate said "no change") keeps the best candidate's own reasoning; group 2 (candidates argued a move, truth stayed in the ±20 band) gets a regenerated, judged hard negative (`data_augmentation/prompts/abstain_hardneg.txt`, faithfulness + grounding judges, dropped if no candidate passes); groups 3-4 (label contradicts the evidence) are dropped. Move / no-change rows are copied | Gemini (group 2) | `TX_VT_NJ-gemini-abstainfix` (358 → 270 rows) |
| COVID | `rewrite_cot_corpus.py` | **nores rewrite**: every training target (Step-1 / Step-2 / abstain CoT) is edited so it no longer cites the recent-baseline-error block (`data_augmentation/prompts/rewrite_cot.txt`, temperature 0), then checked offline (no residual reference left, every double-quoted phrase verbatim in the reports, move rows keep both sections) and by the faithfulness and no-residual grounding judges (`data_augmentation/prompts/rewrite_*_flash.txt`); a failed edit is retried once with feedback, then the row is dropped; `*_orig` keeps the text before the edit | Gemini | `TX_VT_NJ-gemini-abstainfix-nores` = `data/cot_corpora/covid/TX_VT_NJ-gemini-final` (270 → 257 rows) |

The nores rewrite is the corpus side of the "no residuals" protocol; the prompt side is the
released COVID finetune prompt (`prompts/covid/finetune_prompt.txt`: history, baseline, reports —
no recent-error item), which `finetune.py`'s COVID modes (`covid_TSFM_bitong`,
`covid_TSFM_bitong_org`) and `zero_shot_api_notext.py` use.

## Running

```bash
# inputs: the full pipeline runs (the released corpora hold accepted.jsonl only)
code/cot_generation/run_cot_pipeline.sh influenza 8      # -> output/cot_corpora/influenza/chronos2_full/GA_NY_NJ-gemini/
code/cot_generation/run_cot_pipeline.sh covid 4          # -> output/cot_corpora/covid_TSFM_updated/chronos2_full/TX_VT_NJ-gemini/
cd code/data_augmentation
python build_complete_corpus.py                           # influenza -> .../GA_NY_NJ-gemini-complete/
export GOOGLE_CLOUD_PROJECT=<gcp-project>                 # Gemini via Vertex AI ADC
python build_abstain_fix_corpus.py --dry-run              # classify + group 1 only, no API calls
python build_abstain_fix_corpus.py                        # covid -> .../TX_VT_NJ-gemini-abstainfix/
python rewrite_cot_corpus.py --dry-run                    # residual-reference statistics only
python rewrite_cot_corpus.py                              # covid -> .../TX_VT_NJ-gemini-abstainfix-nores/
```

`--src` / `--dst` override the defaults (e.g. `--dst .../GA_NY_NJ-gemini-final` to reproduce the
released tag). Both API stages are resumable (`group2_*.jsonl`, `rewrite_*.jsonl` caches) and write
a decisions file and a JSON report next to the corpus. The judge prompts of the abstain fix
(`step1_faithfulness_flash.txt`, `step1_grounding_pro.txt`) are the pipeline's own in `prompts/covid/`.
