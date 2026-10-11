Direct numeric forecasting prompts for covid_TSFM_bitong_org (TX/VT/NJ, z26w activity index,
single target origin+3; no Chronos-2 baseline shown), used by zero_shot_api_direct.py
--data-mode covid_TSFM_bitong_org. Same structure as ../influenza (influenza):
history + reports + TASK; output = ONE "Forecast:" line; the error item only with --errors keep.
The forecast is an INDEX value; the evaluator back-transforms it to admissions with the target
week's mu/sigma and compares with Chronos-2 on the same rows (±20-point band for direction).
