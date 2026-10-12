#!/usr/bin/env bash
#SBATCH --job-name=flu-original
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=3-00:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
# INFLUENZA, ORIGINAL loss — the loss ablation of the best run: identical data, prompt, target,
# LoRA, lr, batch, 20 epochs and class weights, but --method original = constant per-segment
# weights (reasoning tokens x1, Direction/Size tokens x16) instead of the SCALe cosine ramp.
# Output: $PAPER_OUTPUT_ROOT/finetune/influenza/chronos2_full/GA_NY_NJ-gemini-final/original/
#         Qwen3-8B_original_lr0.0001_w16_ep20/metrics.json
#   sbatch run_finetune_influenza_original.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}"      # submit from code/finetune/
PYTHON="${PYTHON:-python}"                                      # python with torch + ms-swift
export PATH="$(dirname "$(command -v "$PYTHON")"):$PATH"       # the `swift` CLI next to it
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== influenza ORIGINAL-loss finetune start $(date) on $(hostname) ==="
"$PYTHON" -u finetune.py --data-mode influenza --mode chronos2_full --method original \
    --corpus-model gemini-final --states GA,NY,NJ \
    --lr 1e-4 --operator-loss-weight 16 --epochs 20 --class-weights 1:1:1 "$@"
echo "=== done $(date) ==="
