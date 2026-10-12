#!/usr/bin/env bash
#SBATCH --job-name=covid-scale
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100_80gb
#SBATCH --partition=gpu
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=3-00:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
# COVID, SCALe loss — the COVID operator finetune:
#   data mode covid_TSFM_bitong: expanding folds A, B, C + final, derived at run time from the
#   fold_A / fold_B / fold_C / split columns of data/COVID/{STATE}.csv (TX, VT, NJ);
#   corpus = data/cot_corpora/covid/TX_VT_NJ-gemini-final (the corpus of the reported 31.9 run: 257
#   train rows, abstain rows carry an input-specific CoT; --corpus-model gemini = the plain 6-stage
#   output, 358 train rows);
#   prompt prompts/covid/finetune_prompt.txt (history + Chronos-2 baseline + the reports of the
#   last 5 weeks; NO recent-error item); LoRA r8/a16, lr 1e-4, batch 1x16,
#   12-epoch cap, SCALe cosine schedule 3:1>0.5:4, class weights 1:1:1; each fold's epoch is
#   picked by the validation MAE gain, the final model trains for round(mean best epoch); then
#   the per-state move-or-not gate (apply_state_gate.py) fitted on the pooled fold validation
#   predictions decides, per state, whether the model's moves are applied on test.
# This is the configuration of the reported 31.9 run.
# 80 GB card: the epoch-selection inference runs 8 prompts per batch (use INFER_BS=4 on 40 GB).
# Output: $PAPER_OUTPUT_ROOT/finetune/covid_TSFM_bitong/chronos2_full/TX_VT_NJ-gemini-final/scale/
#         Qwen3-8B_scale_lr0.0001_sched3-1to0.5-4/{fold_A,fold_B,fold_C,final}/metrics.json,
#         final/metrics_gated_state.json, folds_summary.json
#   sbatch run_finetune_covid_scale.sbatch
#   sbatch run_finetune_covid_scale.sbatch --data-only     # only write the train/val/test jsonl
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}"      # submit from code/finetune/
PYTHON="${PYTHON:-python}"                                      # python with torch + ms-swift
export PATH="$(dirname "$(command -v "$PYTHON")"):$PATH"       # the `swift` CLI next to it
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ROOT="$("$PYTHON" -c 'import sys; sys.path.insert(0, ".."); import paper_paths as P; print(P.FINETUNE_OUTPUT_ROOT)')/covid_TSFM_bitong/chronos2_full/TX_VT_NJ-gemini/scale"
echo "=== COVID SCALe finetune (folds A,B,C,final) start $(date) on $(hostname) ==="
"$PYTHON" -u finetune.py --data-mode covid_TSFM_bitong --mode chronos2_full --method scale \
    --corpus-model gemini-final --states TX,VT,NJ --folds A,B,C,final \
    --lr 1e-4 --scale-schedule "3:1>0.5:4" --class-weights 1:1:1 --scale-epochs 12 \
    --rank-by val_improvement_MAE_vs_baseline --epoch-metric val_improvement_MAE_vs_baseline \
    --infer-batch-size "${INFER_BS:-8}" "$@"
if [[ " $* " != *" --data-only "* ]]; then
    echo "=== per-state move-or-not gate ==="
    "$PYTHON" -u apply_state_gate.py --root "$ROOT"
fi
echo "=== done $(date) ==="
