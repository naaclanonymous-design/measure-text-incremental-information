#!/usr/bin/env python3
"""Per-state move-or-not gate for covid_TSFM_bitong CV runs (walkthrough §11.9 rule 2,
applied to the fold design). NEW script; reads only existing prediction CSVs.

For each run dir <root>/<run>/:
  1. FIT on validation: pool fold_A/fold_B/fold_C/predictions_val.csv (each fold model scored on
     its own validation block). Per state, compare the mean rough MAE of the model's moves
     (rough_mae_count: move applied at the predicted bin's lower edge, back-transformed to
     admissions) with the mean baseline MAE (baseline_mae_count: keep Chronos-2). Gate = KEEP the
     state's moves iff baseline_MAE − model_MAE > 0 on validation.
  2. APPLY on test: final/predictions.csv. Rows of a KEEP state keep the model's forecast; rows
     of a HOLD state revert to the baseline (delta 0). Nothing else changes.
Writes <run>/final/predictions_gated_state.csv and <run>/final/metrics_gated_state.json and prints
one line per run. Usage:
  python apply_state_gate.py --root <...>/covid_TSFM_bitong/chronos2_full/TX_VT_NJ-gemini/scale
"""
import argparse, ast, json
from pathlib import Path
import numpy as np, pandas as pd

def load(p: Path) -> pd.DataFrame:
    d = pd.read_csv(p)
    for c in ("true_direction", "pred_direction"):
        d[c] = d[c].apply(lambda s: ast.literal_eval(s)[0] if isinstance(s, str) and s.startswith("[") else s)
    return d

def gate_run(run: Path) -> dict | None:
    vals = [load(run / f"fold_{f}" / "predictions_val.csv").assign(fold=f)
            for f in "ABC" if (run / f"fold_{f}" / "predictions_val.csv").exists()]
    tp = run / "final" / "predictions.csv"
    if not vals or not tp.exists():
        return None
    val = pd.concat(vals)
    fit = val.groupby("state").agg(n=("id", "size"), moves=("pred_direction", lambda s: int((s != 0).sum())),
                                   model_MAE=("rough_mae_count", "mean"), baseline_MAE=("baseline_mae_count", "mean"))
    fit["gain"] = fit.baseline_MAE - fit.model_MAE
    fit["keep"] = fit.gain > 0
    te = load(tp)
    keep = te.state.map(fit.keep).fillna(False).astype(bool)
    te["gated"] = keep.values
    te["gated_mae_count"] = np.where(keep, te.rough_mae_count, te.baseline_mae_count)
    te["gated_pred_direction"] = np.where(keep, te.pred_direction, 0)
    te.to_csv(run / "final" / "predictions_gated_state.csv", index=False)
    per_state = {}
    for st, g in te.groupby("state"):
        per_state[st] = {"n": int(len(g)), "keep_moves": bool(fit.keep.get(st, False)),
                         "val_gain": round(float(fit.gain.get(st, np.nan)), 3),
                         "raw_gain": round(float((g.baseline_mae_count - g.rough_mae_count).mean()), 3),
                         "gated_gain": round(float((g.baseline_mae_count - g.gated_mae_count).mean()), 3)}
    res = {"rule": "per-state: keep the model's moves iff its pooled fold-A/B/C validation MAE beats the baseline for that state",
           "baseline_MAE": round(float(te.baseline_mae_count.mean()), 3),
           "raw_MAE": round(float(te.rough_mae_count.mean()), 3),
           "gated_MAE": round(float(te.gated_mae_count.mean()), 3),
           "raw_gain": round(float((te.baseline_mae_count - te.rough_mae_count).mean()), 3),
           "gated_gain": round(float((te.baseline_mae_count - te.gated_mae_count).mean()), 3),
           "moves_before": int((te.pred_direction != 0).sum()), "moves_after": int((te.gated_pred_direction != 0).sum()),
           "per_state": per_state, "val_fit": fit.round(3).reset_index().to_dict(orient="records")}
    (run / "final" / "metrics_gated_state.json").write_text(json.dumps(res, indent=2))
    return res

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="the <combo>/scale dir holding the run dirs")
    a = ap.parse_args()
    for run in sorted(p for p in a.root.iterdir() if p.is_dir() and p.name.startswith(("Qwen", "Llama"))):
        r = gate_run(run)
        if r is None:
            continue
        keep = ",".join(s for s, v in r["per_state"].items() if v["keep_moves"]) or "-"
        print(f"{run.name:48s} baseline {r['baseline_MAE']:6.2f}  raw gain {r['raw_gain']:+6.2f}  "
              f"gated gain {r['gated_gain']:+6.2f}  moves {r['moves_before']:2d}->{r['moves_after']:2d}  keep: {keep}")

if __name__ == "__main__":
    main()
