#!/usr/bin/env python3
"""DIRECT numeric forecasting with hosted LLMs (Gemini-3.1 / GPT-5.4), influenza test set —
the closed-source counterpart of zero_shot_api_notext.py WITHOUT the Chronos-2 baseline.

zero_shot_api_notext.py asks the model to REVISE a given Chronos-2 forecast (direction + size
class per predicted week). This script removes the baseline item from the prompt and asks the
model to output the admission COUNT for each of the next three weeks itself. Everything else
is held fixed: the same corpus rows and split, the same 12-week history block, the same
recent-baseline-error block (errors of the statistical forecaster on past weeks — its
forecasts for the coming weeks are NOT shown), the same report block (--text keep: plain
FluView reports of the last --text-weeks weeks; --text drop: the notext stub), the same
models, thinking settings, temperature and output budget.

Prompt (v2, 2026-09-29; files in prompts/zero_shot_direct/influenza/):
  1. 12-week admission-count history
  2. weekly reports (or the stub)
  TASK: forecast the admission count for predicted weeks 1, 2, 3; output is ONE line
        "Forecast: <w1>, <w2>, <w3>" — no reasoning section.
  --errors keep re-inserts the recent-baseline-error block as item 2 (reports become item 3;
  output dir suffix _witherr). v1 (2026-09-28) had the error item and a "Reasoning:" section
  by default; its results were archived as <variant>_v1_err_reasoning/.

Evaluation (counts, same rows as the operator runs):
  * MAE / MSE of the model's three numbers vs truth, overall, per horizon and per state
  * the Chronos-2 baseline MAE on the same rows (the reference the operator runs correct)
  * a naive persistence baseline (last observed week repeated) for scale
  * implied direction accuracy vs the Chronos-2 baseline with the ±10% band (sign of
    model − baseline vs sign of truth − baseline), so it is comparable with the operator
    runs' direction accuracy
  * parse failures fall back to persistence and are counted (n_parsed)

Output (default --output-root <paper-content>/output/zero_shot):
  <output-root>/influenza_TSFM/chronos2_full/GA_NY_NJ-gemini/direct_<text5wk|notext>_input[_noerr][/val]/
      test.jsonl                      the exact prompts sent
      <model_tag>/responses.jsonl     raw responses + token usage (resumable, one row per id)
      <model_tag>/predictions.csv     parsed forecasts vs truth vs baseline, per row
      <model_tag>/metrics.json        overall / per-horizon / per-state metrics + config
      summary.json / summary.csv      side-by-side table across the models run

Run (forecast conda env; GOOGLE_CLOUD_PROJECT exported for Vertex ADC):
  python zero_shot_api_direct.py --providers gemini,gpt                 # with text, 5 weeks
  python zero_shot_api_direct.py --providers gemini,gpt --text drop     # without text
  ... --split val | --limit 2 | --errors drop | --workers 8
  Gemini-2.5 (thinking_budget instead of thinking_level; 128 = 2.5-pro floor, 'low'):
  python zero_shot_api_direct.py --providers gemini --gemini-model gemini-2.5-flash,gemini-2.5-pro \
         --gemini-thinking-budget 128 [--text drop]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_CODE_ROOT = Path(__file__).resolve().parents[1]          # <paper-content>/code
for _d in ("cot_generation", "finetune", "zero_shot"):
    sys.path.insert(0, str(_CODE_ROOT / _d))
sys.path.insert(0, str(_CODE_ROOT))                         # paper_paths.py
import paper_paths as P                                     # noqa: E402
import finetune as ft                                   # noqa: E402
import zero_shot_api_notext as Z                        # noqa: E402  (clients, corpus, run loop)

# Prompt files (v2, 2026-09-29): prompts/zero_shot_direct/influenza/
#   direct_forecast_prompt.txt  user template — history + reports + TASK; output = ONE Forecast line
#   direct_errors_item.txt      optional recent-baseline-error item (--errors keep; default drop)
#   direct_system_prompt.txt    system prompt
PROMPT_SUBDIR = "influenza"
PROMPT_FILES = {"template": "direct_forecast_prompt.txt", "errors": "direct_errors_item.txt",
                "system": "direct_system_prompt.txt"}
# per data mode: prompt set, band for the implied direction, and the value scale the model
# forecasts in (influenza: admission counts; covid bitong: the 100x z-score activity index,
# back-transformed to admissions with the example's per-week mu/sigma for the MAE)
DIRECT_PROFILES = {
    "influenza": dict(subdir="influenza", band=ft.NOCHANGE_REL_BAND, relative=True,
                      scale="counts"),
    "covid_TSFM_bitong_org": dict(subdir="covid", band=ft.NOCHANGE_ABS_INDEX_BAND,
                                  relative=False, scale="index"),
}
_FORECAST_RE = re.compile(r"Forecast\s*:\s*([^\n]+)", re.IGNORECASE)
_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")
INPUT_SUBDIR = {"drop": "direct_notext_input", "keep": "direct_text{n}wk_input"}
# module-level prompt set (loaded by load_direct_prompts; qwen_direct_forecast.py reuses it)
PROMPTS: Dict[str, str] = {}
SYSTEM_PROMPT_DIRECT = ""
ALLOW_NEGATIVE = False      # set from the profile in main(): True for the covid z-score index


# ══════════════════════════════════════════════════════════════════════════════
# Prompts
# ══════════════════════════════════════════════════════════════════════════════
def load_direct_prompts(prompt_dir: Path, subdir: str = PROMPT_SUBDIR) -> Dict[str, str]:
    """Read the three prompt files into PROMPTS (and SYSTEM_PROMPT_DIRECT)."""
    global SYSTEM_PROMPT_DIRECT
    d = Path(prompt_dir) / subdir
    for k, name in PROMPT_FILES.items():
        p = d / name
        if not p.exists():
            sys.exit(f"prompt file missing: {p}")
        PROMPTS[k] = p.read_text().rstrip("\n")
    SYSTEM_PROMPT_DIRECT = PROMPTS["system"].strip()
    return PROMPTS


def build_prompt(ex: Dict, horizon: int, errors: str) -> str:
    """Fill the user template. With errors == 'keep' the error item is inserted as item 2 and
    the reports become item 3; by default (drop) the reports are item 2."""
    if not PROMPTS:
        load_direct_prompts(P.PROMPT_DIR_DIRECT)
    if errors == "keep":
        errors_item = ft._fill(PROMPTS["errors"], {"recent_residual_block": ex["recent_residual_block"]})
        reports_num = 3
    else:
        errors_item, reports_num = "", 2
    return ft._fill(PROMPTS["template"], {
        "history_len": ft.HISTORY_LEN, "history_block": ex["history_block"],
        "errors_item": errors_item, "reports_num": reports_num,
        "weekly_text_block": ex["weekly_text_block"], "horizon": horizon})


def build_test_prompts(args, states: List[str]):
    data_dir, combo, run_dir, horizon, bins, test = Z.load_split_examples(args, states)
    n_rep = []
    for ex in test:
        if args.text == "drop":
            ex["weekly_text_block"] = Z.NOTEXT_STUB
        else:
            ex["weekly_text_block"] = Z.trim_text_block(ex["weekly_text_block"], args.text_weeks)
            n_rep.append(len(Z._REPORT_HDR.findall(ex["weekly_text_block"])))
    if n_rep:
        from collections import Counter
        print(f"  [text keep] reports per row (last {args.text_weeks} weeks): "
              f"{sorted(Counter(n_rep).items())}")
    prompts = [build_prompt(ex, horizon, args.errors) for ex in test]
    for p in prompts:                                   # the baseline must never leak in
        assert "predicted week 1 (origin + 1):" not in p and "Baseline forecast" not in p
    return data_dir, combo, run_dir, horizon, test, prompts


# ══════════════════════════════════════════════════════════════════════════════
# Parsing + evaluation
# ══════════════════════════════════════════════════════════════════════════════
def parse_forecast(text: str, horizon: int) -> Optional[List[float]]:
    """The LAST 'Forecast:' line → horizon non-negative numbers; None if absent/short."""
    ms = _FORECAST_RE.findall(text or "")
    if not ms:
        return None
    nums = [float(x.replace(",", "")) for x in _NUM_RE.findall(ms[-1])]
    if not ALLOW_NEGATIVE:                 # admission counts cannot be negative; the z-score index can
        nums = [max(0.0, v) for v in nums]
    return nums[:horizon] if len(nums) >= horizon else None


def direction(v: float, base: float, band: float) -> int:
    if base == 0:
        return 0 if abs(v) <= 1 else (1 if v > 0 else -1)
    r = (v - base) / abs(base)
    return 0 if abs(r) <= band else (1 if r > 0 else -1)


def evaluate_direct(tag: str, client_cfg: Dict, test: List[Dict], responses: Dict[str, Dict],
                    horizon: int, band: float, out_dir: Path, states: List[str],
                    input_desc: str, split: str) -> Dict:
    rows = []
    for ex in test:
        resp = responses.get(ex["id"], {}).get("response", "")
        pred = parse_forecast(resp, horizon)
        truth = [float(v) for v in ex["target_full"]][:horizon]
        base = [float(v) for v in ex["baseline_full"]][:horizon]
        persist = [ex["persist"]] * horizon
        parsed = pred is not None
        if not parsed:
            pred = persist
        denorm = ex.get("denorm_sigma") is not None      # covid bitong: values are index points
        for h in range(horizon):
            if denorm:
                # the model forecasts the 100x z-score index; MAE is scored in admissions via
                # the target week's mu/sigma ((v - shift)/100 * sigma + mu, shift = 0 here);
                # the implied direction uses the INDEX values and the absolute ±band
                mu, sd = float(ex["denorm_mu"][h]), float(ex["denorm_sigma"][h])
                shift = float(ex.get("covid_shift", 0.0))
                to_count = lambda v: (v - shift) / 100.0 * sd + mu
                pred_c, persist_c = to_count(pred[h]), to_count(persist[h])
                base_idx, truth_idx = float(ex["baseline"][h]), float(ex["target"][h])
                p_dir = 0 if abs(pred[h] - base_idx) <= band else (1 if pred[h] > base_idx else -1)
                t_dir = 0 if abs(truth_idx - base_idx) <= band else (1 if truth_idx > base_idx else -1)
            else:
                pred_c, persist_c = pred[h], persist[h]
                p_dir = direction(pred[h], base[h], band)
                t_dir = direction(truth[h], base[h], band)
            rows.append({"id": ex["id"], "state": ex["state"], "reference_date": ex["reference_date"],
                         "h": h + 1, "parsed": parsed,
                         "pred_raw": pred[h], "pred": pred_c, "truth": truth[h],
                         "baseline": base[h], "persist": persist_c,
                         "abs_err": abs(pred_c - truth[h]),
                         "baseline_abs_err": abs(base[h] - truth[h]),
                         "persist_abs_err": abs(persist_c - truth[h]),
                         "pred_dir_vs_baseline": p_dir, "true_dir_vs_baseline": t_dir,
                         "response": resp if h == 0 else ""})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "predictions.csv", index=False)

    def block(d: pd.DataFrame) -> Dict:
        return {"n_rows": int(d.id.nunique()), "n_slots": int(len(d)),
                "MAE_counts": float(d.abs_err.mean()), "MSE_counts": float((d.abs_err ** 2).mean()),
                "baseline_MAE_counts": float(d.baseline_abs_err.mean()),
                "baseline_MSE_counts": float((d.baseline_abs_err ** 2).mean()),
                "persistence_MAE_counts": float(d.persist_abs_err.mean()),
                "improvement_MAE_vs_baseline": float(d.baseline_abs_err.mean() - d.abs_err.mean()),
                "improvement_MAE_vs_persistence": float(d.persist_abs_err.mean() - d.abs_err.mean()),
                "direction_accuracy_vs_baseline": float((d.pred_dir_vs_baseline == d.true_dir_vs_baseline).mean()),
                "MAE_by_horizon": [float(d[d.h == h].abs_err.mean()) for h in range(1, horizon + 1)],
                "baseline_MAE_by_horizon": [float(d[d.h == h].baseline_abs_err.mean()) for h in range(1, horizon + 1)]}

    overall = block(df)
    per_state = {st: block(df[df.state == st]) for st in states if (df.state == st).any()}
    usage = [responses[ex["id"]]["usage"] for ex in test if ex["id"] in responses]
    tok = {k: (sum(u[k] for u in usage) if usage else 0)
           for k in ("prompt_tokens", "thinking_tokens", "output_tokens")}
    fin: Dict[str, int] = {}
    for ex in test:
        if ex["id"] in responses:
            k = str(responses[ex["id"]].get("finish_reason"))
            fin[k] = fin.get(k, 0) + 1
    metrics = {"tag": tag, **client_cfg, "zero_shot": True, "task": "direct numeric forecast",
               "split": split, "input": input_desc, "horizon": horizon,
               "nochange_band_relative": band,
               "n_test": len(test), "n_responded": len(usage),
               "n_parsed": int(df[df.h == 1].parsed.sum()),
               "units": "counts (weekly influenza hospital admissions)",
               **overall, "per_state": per_state, "token_totals": tok,
               "mean_thinking_tokens_per_call": (tok["thinking_tokens"] / len(usage)) if usage else None,
               "finish_reasons": fin}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\n[{tag}] MAE={overall['MAE_counts']:.2f} vs Chronos-2 {overall['baseline_MAE_counts']:.2f} "
          f"({overall['improvement_MAE_vs_baseline']:+.2f}) vs persistence "
          f"{overall['persistence_MAE_counts']:.2f} | by horizon "
          f"{[round(x, 1) for x in overall['MAE_by_horizon']]} | implied direction acc vs baseline "
          f"{overall['direction_accuracy_vs_baseline']:.3f} | parsed {metrics['n_parsed']}/{len(test)} "
          f"| mean thinking tokens/call {metrics['mean_thinking_tokens_per_call']}", flush=True)
    return metrics


def write_summary(base: Path, all_metrics: List[Dict]) -> None:
    prev = {}
    sp = base / "summary.json"
    if sp.exists():
        prev = {m["tag"]: m for m in json.loads(sp.read_text())}
    for m in all_metrics:
        prev[m["tag"]] = m
    rows = list(prev.values())
    sp.write_text(json.dumps(rows, indent=2))
    cols = ["tag", "n_test", "n_parsed", "MAE_counts", "baseline_MAE_counts",
            "improvement_MAE_vs_baseline", "persistence_MAE_counts",
            "direction_accuracy_vs_baseline", "MAE_by_horizon", "mean_thinking_tokens_per_call"]
    flat = []
    for m in rows:
        r = {c: m.get(c) for c in cols}
        for st, ms in (m.get("per_state") or {}).items():
            r[f"{st}_MAE_counts"] = ms["MAE_counts"]
            r[f"{st}_baseline_MAE_counts"] = ms["baseline_MAE_counts"]
            r[f"{st}_improvement_MAE_vs_baseline"] = ms["improvement_MAE_vs_baseline"]
        flat.append(r)
    pd.DataFrame(flat).to_csv(base / "summary.csv", index=False)
    print(f"\nsummary → {sp}\n" + pd.DataFrame(flat)[cols].to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--providers", default="gemini,gpt", help="comma list of gemini,gpt")
    p.add_argument("--data-mode", default="influenza", choices=sorted(DIRECT_PROFILES),
                   help="influenza (counts, 3 weeks) or covid_TSFM_bitong_org (z26w index, origin+3)")
    p.add_argument("--mode", default="chronos2_full", choices=("chronos2_full",))
    p.add_argument("--split", choices=("test", "val"), default="test")
    p.add_argument("--states", default=None, help="default per data mode")
    p.add_argument("--corpus-model", default=None,
                   help="corpus tag (<STATES>-<tag>); default per data mode (only bins/rows are read)")
    p.add_argument("--input-root", type=Path, default=ft.INPUT_ROOT)
    p.add_argument("--output-root", type=Path, default=Z.OUTPUT_ROOT)
    p.add_argument("--text", choices=("keep", "drop"), default="keep",
                   help="keep = plain FluView reports of the last --text-weeks weeks; drop = stub")
    p.add_argument("--text-weeks", type=int, default=Z.TEXT_WEEKS)
    p.add_argument("--errors", choices=("keep", "drop"), default="drop",
                   help="drop (default, v2) = no recent-baseline-error item; keep = insert it as "
                        "item 2 (output dir gets a _witherr suffix)")
    p.add_argument("--prompt-dir", type=Path, default=P.PROMPT_DIR_DIRECT)
    p.add_argument("--prompt-subdir", default=None,
                   help=f"subdir holding {', '.join(PROMPT_FILES.values())} (default per data mode)")
    p.add_argument("--gemini-model", default=Z.GEMINI_MODEL,
                   help="comma list; each model is run and evaluated in turn")
    p.add_argument("--gemini-thinking-level", default=Z.GEMINI_THINKING_LEVEL,
                   choices=("minimal", "low", "medium", "high"))
    p.add_argument("--gemini-thinking-budget", type=int, default=None,
                   help="gemini-2.5-* only: integer thinking budget (replaces thinking_level; "
                        "128 = 2.5-pro floor). Run tag becomes <model>_budget-<N>")
    p.add_argument("--gpt-model", default=Z.GPT_MODEL)
    p.add_argument("--gpt-reasoning-effort", default=Z.GPT_REASONING_EFFORT,
                   choices=("none", "minimal", "low", "medium", "high"))
    p.add_argument("--openai-cred", type=Path, default=Z.OPENAI_CRED_PATH)
    p.add_argument("--temperature", type=float, default=Z.TEMPERATURE)
    p.add_argument("--max-output-tokens", type=int, default=ft.MAX_NEW_TOKENS)
    p.add_argument("--nochange-band", type=float, default=None,
                   help="band for the implied direction accuracy vs the Chronos-2 baseline "
                        "(default per data mode: ±10% relative / ±20 index points)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=None, help="first N rows (smoke test)")
    p.add_argument("--force", action="store_true", help="discard cached responses and re-query")
    return p.parse_args()


def main():
    args = parse_args()
    prof = DIRECT_PROFILES[args.data_mode]
    if args.states is None:
        args.states = ft.DATA_MODE_STATES[args.data_mode]
    if args.corpus_model is None:
        args.corpus_model = Z.DEFAULT_CORPUS[args.data_mode]
    if args.nochange_band is None:
        args.nochange_band = prof["band"]
    if args.prompt_subdir is None:
        args.prompt_subdir = prof["subdir"]
    global ALLOW_NEGATIVE
    ALLOW_NEGATIVE = prof["scale"] == "index"     # the z-score index may be negative
    states = [s.strip().upper() for s in args.states.split(",") if s.strip()]
    providers = [s.strip().lower() for s in args.providers.split(",") if s.strip()]
    for pv in providers:
        if pv not in ("gemini", "gpt"):
            sys.exit(f"unknown provider {pv}")

    load_direct_prompts(args.prompt_dir, args.prompt_subdir)
    data_dir, combo, run_dir, horizon, test, prompts = build_test_prompts(args, states)
    # persistence reference = the origin week's observed value (last history value; counts for
    # influenza, index points for covid — converted to admissions in evaluate_direct)
    for ex in test:
        m = re.findall(r"\):\s*(-?\d[\d,]*\.?\d*)", ex["history_block"])
        ex["persist"] = float(m[-1].replace(",", "")) if m else float("nan")
    input_desc = ("history + "
                  + ("recent baseline errors + " if args.errors == "keep" else "")
                  + ("PLAIN FluView reports of the last "
                     f"{args.text_weeks} weeks" if args.text == "keep" else "notext stub")
                  + " — NO Chronos-2 baseline, NO reasoning section; the model outputs one "
                    "Forecast line with the counts")
    print(f"=== DIRECT-FORECAST API inference  data={data_dir}  corpus={run_dir}  states={combo}  "
          f"horizon={horizon}  split={args.split}  n={len(test)}  providers={providers} ===")
    print(f"  input: {input_desc}")
    if args.limit:
        test, prompts = test[:args.limit], prompts[:args.limit]
        print(f"  --limit {args.limit}: using the first {len(test)} rows")

    variant = INPUT_SUBDIR[args.text].format(n=args.text_weeks) + ("_witherr" if args.errors == "keep" else "")
    base = args.output_root / data_dir / args.mode / combo / variant
    if args.split != "test":
        base = base / args.split
    if args.limit:
        base = base / f"smoke_limit{args.limit}"
    base.mkdir(parents=True, exist_ok=True)
    sysp = SYSTEM_PROMPT_DIRECT
    with (base / "test.jsonl").open("w") as f:
        for ex, p in zip(test, prompts):
            f.write(json.dumps({"id": ex["id"], "state": ex["state"],
                                "reference_date": ex["reference_date"],
                                "messages": [{"role": "system", "content": sysp},
                                             {"role": "user", "content": p}]}) + "\n")
    print(f"  wrote {len(test)} {args.split} prompts → {base / 'test.jsonl'}")
    print(f"  system prompt: {sysp!r}")

    clients = []
    for pv in providers:
        if pv == "gemini":
            for gm in [m.strip() for m in args.gemini_model.split(",") if m.strip()]:
                clients.append(Z.GeminiZS(gm, args.gemini_thinking_level,
                                          args.max_output_tokens, args.temperature,
                                          thinking_budget=args.gemini_thinking_budget))
        else:
            clients.append(Z.GPTZS(args.gpt_model, args.gpt_reasoning_effort,
                                   args.max_output_tokens, args.temperature, args.openai_cred))
    all_metrics = []
    for client in clients:
        tag = client.tag()
        out_dir = base / tag
        print(f"\n--- {tag} → {out_dir} ---")
        responses = Z.run_model(client, tag, test, prompts, sysp, out_dir, args.workers, args.force)
        all_metrics.append(evaluate_direct(tag, client.config(), test, responses, horizon,
                                           args.nochange_band, out_dir, states, input_desc,
                                           args.split))
    write_summary(base, all_metrics)


if __name__ == "__main__":
    main()
