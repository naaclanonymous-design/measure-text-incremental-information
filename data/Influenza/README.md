# Influenza dataset — weekly influenza hospital admissions + surveillance reports

One CSV per state — the format read by `code/cot_generation` and `code/finetune`
(`--data-mode influenza`). Every row carries the weekly report text (`weekly_text`) and the
Chronos-2 forecast (`chronos2_pred_3w`), so each state file is complete on its own;
`chronos2/chronos2_influenza_predictions.csv` is the same Chronos-2 output as a separate cache
(state, reference_date = first forecast week, h0, h1, h2), produced by
`code/chronos2/chronos2_influenza.py` from the NHSN influenza panel in `data/raw/`.

| file | rows | origins | text rows train / val / test | report source |
|---|---|---|---|---|
| GA.csv | 219 | 2022-01-01 .. 2026-03-28 | 123/123, 48/48, 44/44 | Georgia DPH weekly influenza report |
| NY.csv | 219 | same | 85/123, 28/48, 24/44 | NYSDOH weekly influenza surveillance report |
| NJ.csv | 219 | same | 85/123, 30/48, 26/44 | NJDOH weekly respiratory illness surveillance report |
| FL.csv | 219 | same | 77/123, 39/48, 29/44 | Florida Flu Review |
| IA.csv | 219 | same | 67/123, 46/48, 26/44 | Iowa HHS weekly respiratory report |
| MS.csv | 219 | same | 38/123, 21/48, 17/44 | MSDH weekly influenza report |
| VA.csv | 219 | same | 0/123, 30/48, 30/44 | VDH weekly respiratory report (from 2024) |
| USA.csv | 219 | same | 110/123, 29/48, 25/44 | CDC FluView national summary |

The paper's CoT corpus and finetunes use **GA, NY, NJ**; the other states are released for
transfer / additional-state experiments (`--states FL,IA,VA`).

## Columns (9)

| column | content |
|---|---|
| `state` | 2-letter code (USA = national) |
| `reference_date` | week-ending Saturday of the origin week `t` (last observed week) |
| `hist_12w` | JSON list, admissions of weeks `t-11 .. t` (whole counts) |
| `future_3w` | JSON list, ground-truth admissions of `t+1, t+2, t+3` |
| `weekly_text` | the origin week's report, one physical line (newlines escaped as `\n`; `cot_common.unescape_multiline` restores it); empty when the week has no usable report |
| `chronos2_pred_3w` | Chronos-2 forecast of `t+1..t+3` with context ending at `t` (multivariate: the state's series plus the other states and the national series as past-only covariates, context from 2020-08-01; see `code/chronos2/chronos2_influenza.py`) |
| `stat_pred_3w` | mean of `hist_12w`, repeated 3× |
| `split` | `train` (2022-01-01 .. 2024-05-04, 123 origins), `val` (2024-06-01 .. 2025-04-26, 48), `test` (2025-05-31 .. 2026-03-28, 44), `gap` (3-week buffers) |
| `population` | FluSight population (median of value / weekly_rate) |

A sample at origin `t` uses `hist_12w`, the reports of weeks `t-4 .. t` (from the five
consecutive rows), `chronos2_pred_3w` and `future_3w`, and is kept only when at least one of
the five weeks has a report. Values are raw counts; the no-change band is ±10 % of the baseline
and the size classes are fractions of the baseline (corpus `bins.json`: 0.10 / 0.2155 / 0.3649 /
0.6449).
