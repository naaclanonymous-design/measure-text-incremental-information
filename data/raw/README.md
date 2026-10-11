# Raw inputs

The source tables behind the two datasets (data only; the scripts that consume them live in
`code/`).

| file | content | used by |
|---|---|---|
| `covid-hospitalizations-nhsn.csv` | NHSN weekly COVID-19 hospital admissions, long format (`state, location, target_end_date, hospitalizations`), every jurisdiction, 2020-08-08 .. 2026-04-18 | `code/chronos2/chronos2_covid.py`, `code/data_build/build_covid_dataset.py` |
| `influenza_admissions_panel.csv` | NHSN weekly influenza hospital admissions (*Total Influenza Admissions*), wide format: `week_ending_date` × 50 states + DC + `USA`, 2020-08-08 .. 2026-04-18 | `code/chronos2/chronos2_influenza.py` (the target state's series plus the other columns as covariates) |
| `target-hospital-admissions.csv` | FluSight target file; the `population` column of the dataset CSVs is the median of `value / weekly_rate × 1e5` per location | `build_covid_dataset.py`, `cot_common.build_population_map` |
| `holdout_splits.json` | the COVID train / buffer / validation / test windows (folds A, B, C and final), as first-forecast-week dates (`h0 = origin + 1 week`) | `build_covid_dataset.py` (writes the `split` / `fold_*` columns) |
