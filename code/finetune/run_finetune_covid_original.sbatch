#!/usr/bin/env bash
#SBATCH --job-name=covid-original
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100_80gb
#SBATCH --partition=gpu
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=3-00:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
# COVID, ORIGINAL loss — the loss ablation of run_finetune_covid_scale.sbatch: identical data,
# prompt (prompts/covid/finetune_prompt.txt), corpus, LoRA, lr, batch, 12-epoch cap, folds, epoch
# rule and per-state gate, but --method original = constant per-segment weights (reasoning x1,
# Direction/Size tokens x16) instead of the SCALe ramp 3:1>0.5:4.
# Output: $PAPER_OUTPUT_ROOT/finetune/covid_TSFM_bitong/chronos2_full/TX_VT_NJ-gemini/original/
#         Qwen3-8B_original_lr0.0001_w16/{fold_A,fold_B,fold_C,final}/metrics.json, final/metrics_gated_state.json
#   sbatch run_finetune_covid_original.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}"      # submit from code/finetune/
PYTHON="${PYTHON:-python}"                                      # python with torch + ms-swift
export PATH="$(dirname "$(command -v "$PYTHON")"):$PATH"       # the `swift` CLI next to it
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ROOT="$("$PYTHON" -c 'import sys; sys.path.insert(0, ".."); import paper_paths as P; print(P.FINETUNE_OUTPUT_ROOT)')/covid_TSFM_bitong/chronos2_full/TX_VT_NJ-gemini/original"
echo "=== COVID ORIGINAL-loss finetune (folds A,B,C,final) start $(date) on $(hostname) ==="
"$PYTHON" -u finetune.py --data-mode covid_TSFM_bitong --mode chronos2_full --method original \
    --corpus-model gemini --states TX,VT,NJ --folds A,B,C,final \
    --lr 1e-4 --operator-loss-weight 16 --class-weights 1:1:1 --epochs 12 \
    --rank-by val_improvement_MAE_vs_baseline --epoch-metric val_improvement_MAE_vs_baseline \
    --infer-batch-size "${INFER_BS:-8}" "$@"
if [[ " $* " != *" --data-only "* ]]; then
    echo "=== per-state move-or-not gate ==="
    "$PYTHON" -u apply_state_gate.py --root "$ROOT"
fi
echo "=== done $(date) ==="
