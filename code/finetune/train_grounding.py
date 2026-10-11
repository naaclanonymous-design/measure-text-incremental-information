#!/usr/bin/env python3
"""Train a GROUNDING MODULE for a finetuned run and write the accurate MAE into metrics.json.

The finetuned LLM emits a categorical operator (per-week direction + size bin). The pipeline's
"rough" MAE grounds every predicted bin at its LOWER edge (conservative, systematically
under-corrects). This module instead LEARNS, per PREDICTED bin, the magnitude that best matches
reality — calibrated on the run's VALIDATION predictions (data the LLM did NOT train on, so its
predicted bins reflect genuine test-time bin-confusion). It is deliberately fit on the model's
*predicted* bins (not the true bins), which makes it robust to the model's low bin accuracy.

Pipeline (disjoint splits, no leakage):
    LLM operator : trained on TRAIN
    grounding    : fit on     VAL   (median true |residual| among weeks the model PREDICTED bin k)
    accurate MAE : scored on  TEST  (LLM test predictions + this grounding)

Outputs (in the run dir):
    grounding.json     — the learned per-bin magnitudes + provenance
    metrics.json       — ADDS lora_grounded_* fields (keeps the existing rough fields); backs up
                         the original to metrics.json.bak once.

Usage:  python train_grounding.py <run_dir>
        <run_dir> = .../CoT_generation_updated_finetune/<mode>/<combo>/<method>/<run_name>
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

_CODE_ROOT = Path(__file__).resolve().parents[1]          # <paper-content>/code
for _d in ("cot_generation", "finetune", "zero_shot"):
    sys.path.insert(0, str(_CODE_ROOT / _d))
sys.path.insert(0, str(_CODE_ROOT))                         # paper_paths.py
import paper_paths as P                                     # noqa: E402
import finetune as F


def _residual(gt, base, relative):
    """Signed residual used for binning: relative fraction (chronos2_full) or absolute (rate)."""
    return (gt - base) / max(abs(base), 1.0) if relative else (gt - base)


def parse_pred(resp_list, examples, horizon, names):
    """Align model responses (in file order) with examples and parse per-week (dir, bin)."""
    out = []
    for e, resp in zip(examples, resp_list):
        md, ms = F.parse_operator(resp, horizon, names)
        out.append((e, md, ms))
    return out


def fit_grounding(val_pred, horizon, bins, relative):
    """m_grounded[k] = median |true residual| over VAL weeks where the model PREDICTED bin k.
    Falls back to the bin's lower edge when a bin was never predicted on val."""
    lower = bins["lower_edges"]
    pools = {k: [] for k in range(len(lower))}
    for e, md, ms in val_pred:
        if md is None:
            continue
        for h in range(horizon):
            if md[h] != 0 and ms[h] is not None:
                r = abs(_residual(e["target"][h], e["baseline"][h], relative))
                pools[ms[h]].append(r)
    m = [float(np.median(pools[k])) if pools[k] else float(lower[k]) for k in range(len(lower))]
    n = [len(pools[k]) for k in range(len(lower))]
    return m, n


def score(test_pred, horizon, bins, relative, mfun, scale_map):
    """Return (per-horizon MAE, per-horizon MSE, overall MAE, overall MSE) in counts."""
    ae = [[] for _ in range(horizon)]
    se = [[] for _ in range(horizon)]
    for e, md, ms in test_pred:
        if md is None:
            continue
        bf = np.asarray(e["baseline_full"], float)
        tf = np.asarray(e["target_full"], float)
        for h in range(horizon):
            if md[h] != 0 and ms[h] is not None:
                m = mfun(ms[h])
                if relative:
                    d = (1 if md[h] > 0 else -1) * m * abs(bf[h])
                    fc = bf[h] + d
                else:
                    sc = scale_map[e["state"].upper()]
                    fc = (bf[h] + (1 if md[h] > 0 else -1) * m) * sc
            else:
                fc = bf[h] * (1.0 if relative else scale_map[e["state"].upper()])
            tgt = tf[h] * (1.0 if relative else scale_map[e["state"].upper()])
            ae[h].append(abs(tgt - fc)); se[h].append((tgt - fc) ** 2)
    hmae = [float(np.mean(a)) for a in ae]
    hmse = [float(np.mean(s)) for s in se]
    allmae = float(np.mean([x for a in ae for x in a]))
    allmse = float(np.mean([x for s in se for x in s]))
    return hmae, hmse, allmae, allmse


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path, help="finetune run directory (contains metrics.json, "
                                               "swift_predictions_val.jsonl, swift_predictions_lora.jsonl)")
    args = ap.parse_args()
    run = args.run_dir
    mpath = run / "metrics.json"
    if not mpath.exists():
        sys.exit(f"no metrics.json in {run}")
    meta = json.loads(mpath.read_text())

    # .../<data_mode>/<mode>/<combo>/<method>/<run_name>
    mode = run.parent.parent.parent.name
    data_mode = run.parent.parent.parent.parent.name
    combo = run.parent.parent.name
    horizon = F.horizon_for(data_mode, mode)
    states_list = [s.strip().upper() for s in meta.get("states", F.DATA_MODE_STATES.get(data_mode, "GA,NY,NJ")).split(",") if s.strip()]
    states = set(states_list)
    corpus = F.INPUT_ROOT / data_mode / mode / combo
    print(f"run    : {run.name}")
    print(f"data_mode={data_mode}  mode={mode}  combo={combo}  horizon={horizon}  corpus={corpus}")

    bins = F.load_bins(corpus)
    relative = F.bins_are_relative(bins)
    names = F.bin_names(bins)
    by = F.load_examples(corpus, horizon, bins, keep_states=states, data_mode=data_mode)
    if data_mode in ("finance_stat", "finance_TSFM"):
        # same data path as finetune.py: USD-scale full values + offline val/test examples
        orig = F.load_finance_orig(states_list)
        F.to_original_scale(by, orig)
        if not by.get("val") and not by.get("test"):
            by.update(F.build_finance_eval_examples(states_list, ["val", "test"], horizon,
                                                    orig, data_mode))
    # covid_TSFM bins are absolute index points (no rate→count scaling either)
    scale_map = {} if (relative or data_mode == "covid_TSFM") else F.rate_to_count_scale(states)

    val_resp = F.load_swift_predictions(run / "swift_predictions_val.jsonl")
    test_resp = F.load_swift_predictions(run / "swift_predictions_lora.jsonl")
    if len(val_resp) != len(by["val"]) or len(test_resp) != len(by["test"]):
        sys.exit(f"prediction/example count mismatch: val {len(val_resp)}/{len(by['val'])} "
                 f"test {len(test_resp)}/{len(by['test'])}")
    val_pred = parse_pred(val_resp, by["val"], horizon, names)
    test_pred = parse_pred(test_resp, by["test"], horizon, names)

    # 1) FIT grounding on VAL predicted bins
    m_grounded, n_val = fit_grounding(val_pred, horizon, bins, relative)
    lower = bins["lower_edges"]
    print("\nlearned grounding magnitudes (per PREDICTED bin, from val):")
    for k, nm in enumerate(names):
        print(f"  {nm:11s} lower(rough)={lower[k]:.3f}  grounded={m_grounded[k]:.3f}  (n_val={n_val[k]})")

    # 2) SCORE test: rough (lower edge) vs grounded
    r_hmae, r_hmse, r_mae, r_mse = score(test_pred, horizon, bins, relative, lambda k: lower[k], scale_map)
    g_hmae, g_hmse, g_mae, g_mse = score(test_pred, horizon, bins, relative, lambda k: m_grounded[k], scale_map)
    base_mae = meta.get("baseline_MAE_counts")
    print(f"\nrough  MAE {r_mae:.2f}  (reproduces evaluate())   per-horizon {[round(x,1) for x in r_hmae]}")
    print(f"GROUND MAE {g_mae:.2f}                            per-horizon {[round(x,1) for x in g_hmae]}")
    if base_mae:
        print(f"baseline MAE {base_mae:.2f} | grounded improvement vs baseline {base_mae-g_mae:+.2f} "
              f"| vs rough {r_mae-g_mae:+.2f}")

    # 3) SAVE grounding module
    grounding = {"method": "val_predicted_bin_median", "relative": relative,
                 "bin_names": names, "lower_edges": lower, "magnitudes": m_grounded,
                 "n_val_per_bin": n_val, "fit_split": "val", "combo": combo, "mode": mode}
    (run / "grounding.json").write_text(json.dumps(grounding, indent=2))

    # 4) UPDATE metrics.json (additive; back up original once)
    if not (run / "metrics.json.bak").exists():
        shutil.copy(mpath, run / "metrics.json.bak")
    meta.update({
        "grounding_method": "val_predicted_bin_median",
        "grounding_bin_magnitudes": {names[k]: round(m_grounded[k], 4) for k in range(len(names))},
        "lora_grounded_MAE_counts": g_mae,
        "lora_grounded_MSE_counts": g_mse,
        "lora_grounded_MAE_by_horizon": [round(x, 2) for x in g_hmae],
        "grounded_improvement_MAE_vs_baseline": (base_mae - g_mae) if base_mae else None,
        "grounded_improvement_MAE_vs_rough": r_mae - g_mae,
    })
    mpath.write_text(json.dumps(meta, indent=2))
    print(f"\nwrote grounding.json and updated {mpath} (backup: metrics.json.bak)")


if __name__ == "__main__":
    main()
