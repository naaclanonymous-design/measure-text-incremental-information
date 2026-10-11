# COVID dataset — weekly COVID-19 hospital admissions (NHSN) + state surveillance reports

Nine states: **AZ, IA, MI, MS, NJ, NY, TX, VA, VT**. This directory holds data only; it is built by
`code/data_build/build_covid_dataset.py` from `data/raw/` (NHSN admissions, FluSight
populations, split definitions), `chronos2/` and the weekly state reports.

```
{STATE}.csv              one file per state, 19 columns (see below) — the only file the code needs
chronos2/
  chronos2_nhsn_zscore26w_predictions.csv   Chronos-2 univariate forecasts ON the z26w series = the
                                            baseline in the CSVs (state, reference_date = h0, h0, h1, h2)
  chronos2_nhsn_raw_predictions.csv         the same model on raw counts (reference only)
```

The weekly report of every origin week is the `weekly_text` column of the CSV (one physical
line; `\n` escapes restored by `cot_common.unescape_multiline`), so each state file is
complete on its own. Texas text is the Summary section of the DSHS report (metadata header and
the laboratory / variant / mortality tables dropped).

The paper's CoT corpus and finetunes use **TX, VT, NJ**; the other states are released with the
same columns, the same date-based splits and their own report text.

## Units

* Working series: the **causal trailing-26-week z-score** of weekly admissions,
  `z_w = (x_w − μ_w) / σ_w` (μ, σ over the 26 weeks ending at `w`, ddof = 0), stored as
  **100 × z rounded to 2 dp** in every model-visible column (`hist_12w`, `chronos2_pred_3w`,
  `stat_pred_3w`, `future_3w`). 100 points = one local standard deviation.
* `denorm_mean_*` / `denorm_std_*` hold the per-week μ, σ: `count = (v / 100) · σ + μ`.
  `future_3w_full` (100×z, full precision) and `future_3w_count` (raw counts) are the exact
  ground truth.
* Chronos-2 is run **on the z26w series** (univariate, growing context from 2021-10-09,
  horizon 3, median); the cache's `reference_date` is the first forecast week
  `h0 = origin + 1 week`. Script: `code/chronos2/chronos2_covid.py`.
* No-change band: **absolute ±20 index points**; size classes = |residual| quantiles in index
  points (`bins.json` of the corpus: 20 / 37.5 / 62.1 / 114.7).

## Columns (19)

`state, reference_date (origin week t), hist_12w (t−11..t), future_3w (t+1..t+3, 2 dp),
weekly_text (origin week's report, newline-escaped; "" if none), chronos2_pred_3w,
stat_pred_3w (mean of hist_12w ×3), split, population, denorm_mean_hist_12w,
denorm_std_hist_12w, denorm_mean_future_3w, denorm_std_future_3w, h0_date (= t + 1 week),
fold_A, fold_B, fold_C, future_3w_full, future_3w_count`.

CoT generation (`--data-mode covid_TSFM_updated`) reads these files directly (`split` column =
the final setting). Finetuning (`--data-mode covid_TSFM_bitong`) derives each fold's view at run
time — the first 13 columns, `split` := the fold's label column (`unused` → `gap`, the final
test block exposed as `test`), `future_3w` := `future_3w_full / 100` (plain z) — into
`output/covid_folds/<fold>/`; nothing else is stored in the data directory.

## Rows and splits

Every complete origin week with a Chronos-2 row: **223 rows per state, origins 2021-12-25 ..
2026-03-28**; no text-based row filter (the pipeline's text gate — ≥1 report in weeks t−4..t —
applies at sample time). Splits are date-based and identical for every state
(`data/raw/holdout_splits.json`, dates = h0; origin = h0 − 1 week):

| column | train (origins) | n | buffer | held-out (origins) | n |
|---|---|---|---|---|---|
| fold_A | 2023-05-06 .. 2024-07-13 | 63 | 3 weeks | val 2024-08-10 .. 2024-12-21 | 20 |
| fold_B | 2023-05-06 .. 2024-11-30 | 83 | 3 weeks | val 2024-12-28 .. 2025-05-10 | 20 |
| fold_C | 2023-05-06 .. 2025-04-19 | 103 | 3 weeks | val 2025-05-17 .. 2025-09-27 | 20 |
| split  | 2023-05-06 .. 2025-09-06 | 123 | 3 weeks | test 2025-10-04 .. 2026-03-28 | 26 |

Labels: `split` = train / gap / test; `fold_X` = train / gap / val / unused. Origins before
2023-05-06 are `gap` everywhere. Protocol: choose the configuration by the mean validation score
over folds A, B, C; finetune once on `split == train`, score once on `split == test`.

## Text coverage (rows with a report / rows)

| state | report source | total | train | test | fold A val | fold B val | fold C val |
|---|---|---|---|---|---|---|---|
| TX | Texas DSHS weekly COVID-19 surveillance report (Summary section only) | 152 | 123/123 | 26/26 | 20/20 | 20/20 | 20/20 |
| VT | Vermont DOH weekly respiratory surveillance report | 201 | 113/123 | 26/26 | 18/20 | 19/20 | 20/20 |
| NJ | NJDOH weekly respiratory illness surveillance report | 198 | 105/123 | 25/26 | 20/20 | 20/20 | 2/20 |
| VA | VDH weekly respiratory disease surveillance report | 176 | 83/123 | 26/26 | 14/20 | 20/20 | 5/20 |
| NY | NYSDOH respiratory report / Erie County DOH report | 144 | 69/123 | 24/26 | 19/20 | 15/20 | 2/20 |
| MI | MDHHS "MI COVID Response" weekly update | 109 | 47/123 | 17/26 | 11/20 | 20/20 | 1/20 |
| AZ | ADHS weekly respiratory report (biweekly files cover both weeks) | 108 | 79/123 | 26/26 | 11/20 | 11/20 | 20/20 |
| MS | MSDH weekly report (from 2024-12) | 49 | 23/123 | 26/26 | 2/20 | 20/20 | 1/20 |
| IA | Iowa HHS weekly respiratory report (from 2024) | 131 | 99/123 | 26/26 | 19/20 | 19/20 | 5/20 |

NJ, MS, NY, MI and AZ pause for ~18–22 weeks in summer 2025 (fold C), so fold-C text methods
should be scored on TX, VT, AZ (and IA / VA with partial coverage).
