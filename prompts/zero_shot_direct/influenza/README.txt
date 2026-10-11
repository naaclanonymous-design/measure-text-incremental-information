Direct numeric forecasting prompts (influenza; no Chronos-2 baseline shown), used by
zero_shot_api_direct.py (Gemini / GPT) and qwen_direct_forecast.py (Qwen3-8B).
  direct_forecast_prompt.txt  user template: history + reports + TASK; output = ONE "Forecast:" line
                              placeholders {history_len} {history_block} {errors_item} {reports_num}
                              {weekly_text_block} {horizon}
  direct_errors_item.txt      the optional recent-baseline-error item (--errors keep only; default drop)
  direct_system_prompt.txt    system prompt (Qwen runs append the /no_think token)
v1 (2026-09-28) asked for "Reasoning: ..." + "Forecast: ..." and included the error item by default;
v2 (2026-09-29, this set) drops the error item and the reasoning section.
