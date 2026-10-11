#!/usr/bin/env python3
"""Grounded MAE for the ZERO-SHOT API runs (zero_shot_api_notext.py), using EXACTLY the
train_grounding.py method applied to the finetune runs:

    grounding  : fit on the setting's own VAL predictions — for each PREDICTED bin k, the
                 median true |relative residual| over val week-slots where the model predicted
                 bin k (fallback: the bin's lower edge if never predicted)   [fit_grounding]
    grounded   : rescore the TEST predictions with those magnitudes instead of the rough
                 bin LOWER edges                                                   [score]
    grounded_improvement_MAE_vs_rough = rough_MAE − grounded_MAE  (same rows, same parser)

NEW script; imports fit_grounding/score/parse_pred from train_grounding.py unchanged. For each
<variant>/<model_tag> it reads <variant>/val/<tag>/responses.jsonl (val) and
<variant>/<tag>/responses.jsonl (test), aligns responses to the corpus examples BY ID (the
API responses are written in completion order), writes <variant>/<tag>/grounding.json, and
ADDS grounded_* fields to <variant>/<tag>/metrics.json (backing the original up once to
metrics.json.bak). A cross-setting table goes to <combo-root>/grounding_summary.{json,csv}.

Usage:
  python ground_zero_shot_api.py            # all 4 settings
  ... --variants text5wk_input --tags gpt-5.4_reasoning-none
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

_CODE_ROOT = Path(__file__).resolve().parents[1]          # <paper-content>/code
for _d in ("cot_generation", "finetune", "zero_shot"):
    sys.path.insert(0, str(_CODE_ROOT / _d))
sys.path.insert(0, str(_CODE_ROOT))                         # paper_paths.py
import paper_paths as P                                     # noqa: E402
import finetune as F                                              # noqa: E402
from train_grounding import fit_grounding, parse_pred, score     # noqa: E402

ZS_ROOT = P.ZERO_SHOT_OUTPUT_ROOT / "influenza" / "chronos2_full" / "GA_NY_NJ-gemini"
VARIANTS = ("text5wk_input", "scale_residualcot_notext_input")
TAGS = ("gemini-3.1-pro-preview_think-low", "gpt-5.4_reasoning-none")


def responses_by_id(path: Path) -> dict:
    out = {}
    for l in path.read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            out[r["id"]] = r.get("response", "")
    return out


def aligned(examples, resp_map: dict, what: str):
    missing = [e["id"] for e in examples if e["id"] not in resp_map]
    if missing:
        sys.exit(f"{what}: {len(missing)} examples without a response (e.g. {missing[:3]})")
    return [resp_map[e["id"]] for e in examples]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=ZS_ROOT)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--tags", default=",".join(TAGS))
    ap.add_argument("--data-mode", default="influenza")
    ap.add_argument("--mode", default="chronos2_full")
    ap.add_argument("--states", default="GA,NY,NJ")
    args = ap.parse_args()

    states_list = [s.strip().upper() for s in args.states.split(",") if s.strip()]
    combo = f"{F.run_dir_name(states_list)}-gemini"
    corpus = F.INPUT_ROOT / args.data_mode / args.mode / combo
    horizon = F.horizon_for("influenza", args.mode)
    bins = F.load_bins(corpus)
    relative = F.bins_are_relative(bins)
    names = F.bin_names(bins)
    lower = bins["lower_edges"]
    by = F.load_examples(corpus, horizon, bins, keep_states=set(states_list), data_mode="influenza")
    scale_map = {} if relative else F.rate_to_count_scale(set(states_list))
    print(f"corpus={corpus}  horizon={horizon}  relative={relative}  val={len(by['val'])}  test={len(by['test'])}")

    rows = []
    for variant in [v for v in args.variants.split(",") if v]:
        for tag in [t for t in args.tags.split(",") if t]:
            run = args.root / variant / tag
            val_dir = args.root / variant / "val" / tag
            mpath = run / "metrics.json"
            if not (mpath.exists() and (run / "responses.jsonl").exists()
                    and (val_dir / "responses.jsonl").exists()):
                print(f"SKIP {variant}/{tag}: missing test or val responses / metrics.json")
                continue
            meta = json.loads(mpath.read_text())
            val_resp = aligned(by["val"], responses_by_id(val_dir / "responses.jsonl"), f"{variant}/{tag} val")
            test_resp = aligned(by["test"], responses_by_id(run / "responses.jsonl"), f"{variant}/{tag} test")
            val_pred = parse_pred(val_resp, by["val"], horizon, names)
            test_pred = parse_pred(test_resp, by["test"], horizon, names)

            m_g, n_val = fit_grounding(val_pred, horizon, bins, relative)
            r_hmae, r_hmse, r_mae, r_mse = score(test_pred, horizon, bins, relative, lambda k: lower[k], scale_map)
            g_hmae, g_hmse, g_mae, g_mse = score(test_pred, horizon, bins, relative, lambda k: m_g[k], scale_map)
            base_mae = meta.get("baseline_MAE_counts")
            if abs(r_mae - meta["rough_MAE_counts"]) > 1e-6:
                print(f"  WARNING rough MAE recomputed {r_mae:.4f} != metrics.json {meta['rough_MAE_counts']:.4f}")

            print(f"\n=== {variant} / {tag} ===")
            for k, nm in enumerate(names):
                print(f"  {nm:11s} lower(rough)={lower[k]:.3f}  grounded={m_g[k]:.3f}  (n_val={n_val[k]})")
            print(f"  rough  MAE {r_mae:.2f}  per-horizon {[round(x, 1) for x in r_hmae]}")
            print(f"  GROUND MAE {g_mae:.2f}  per-horizon {[round(x, 1) for x in g_hmae]}")
            print(f"  baseline MAE {base_mae:.2f} | grounded improvement vs baseline {base_mae - g_mae:+.2f} "
                  f"| vs rough {r_mae - g_mae:+.2f}")

            (run / "grounding.json").write_text(json.dumps(
                {"method": "val_predicted_bin_median", "relative": relative, "bin_names": names,
                 "lower_edges": lower, "magnitudes": m_g, "n_val_per_bin": n_val, "fit_split": "val",
                 "fit_source": str(val_dir / "responses.jsonl"), "combo": combo, "mode": args.mode},
                indent=2))
            if not (run / "metrics.json.bak").exists():
                shutil.copy(mpath, run / "metrics.json.bak")
            upd = {
                "grounding_method": "val_predicted_bin_median",
                "grounding_bin_magnitudes": {names[k]: round(m_g[k], 4) for k in range(len(names))},
                "grounding_n_val_per_bin": {names[k]: n_val[k] for k in range(len(names))},
                "grounded_MAE_counts": g_mae, "grounded_MSE_counts": g_mse,
                "grounded_MAE_by_horizon": [round(x, 2) for x in g_hmae],
                "rough_MAE_by_horizon": [round(x, 2) for x in r_hmae],
                "grounded_improvement_MAE_vs_baseline": (base_mae - g_mae) if base_mae else None,
                "grounded_improvement_MAE_vs_rough": r_mae - g_mae,
            }
            meta.update(upd)
            mpath.write_text(json.dumps(meta, indent=2))
            rows.append({"variant": variant, "tag": tag, "rough_MAE_counts": r_mae,
                         "grounded_MAE_counts": g_mae, "baseline_MAE_counts": base_mae,
                         "grounded_improvement_MAE_vs_rough": r_mae - g_mae,
                         "grounded_improvement_MAE_vs_baseline": base_mae - g_mae,
                         **{f"m_{names[k]}": round(m_g[k], 4) for k in range(len(names))},
                         **{f"nval_{names[k]}": n_val[k] for k in range(len(names))},
                         "grounded_MAE_by_horizon": [round(x, 2) for x in g_hmae]})
    if rows:
        sp = args.root / "grounding_summary.json"
        prev = {}
        if sp.exists():
            prev = {(r["variant"], r["tag"]): r for r in json.loads(sp.read_text())}
        for r in rows:
            prev[(r["variant"], r["tag"])] = r
        allrows = list(prev.values())
        sp.write_text(json.dumps(allrows, indent=2))
        df = pd.DataFrame(allrows)
        df.to_csv(args.root / "grounding_summary.csv", index=False)
        print(f"\nsummary → {sp}\n" + df[["variant", "tag", "rough_MAE_counts", "grounded_MAE_counts",
                                        "baseline_MAE_counts", "grounded_improvement_MAE_vs_rough",
                                        "grounded_improvement_MAE_vs_baseline"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
