#!/usr/bin/env bash
#SBATCH --job-name=flu-scale
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=3-00:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
# INFLUENZA, SCALe loss — the influenza operator finetune (Qwen3-8B; GA/NY/NJ; 101 test origins):
#   corpus = the released 6-stage output data/cot_corpora/influenza/GA_NY_NJ-gemini (310 train rows;
#   the paper's 153.7 run used an expanded version of it, 653 rows, not part of this release)
#   prompt prompts/influenza/chronos2_full/finetune_prompt.txt (history + baseline + recent errors
#   + 5 weeks of reports), LoRA r8/a16/d0.05, lr 1e-4, batch 1x16, 20 epochs, class weights 1:1:1,
#   SCALe cosine schedule 4:1>0.5:4 (reasoning:answer segment weights, start > end).
# Checkpoint by validation loss; greedy inference on val + test; zero-shot Qwen3-8B control.
# Output: $PAPER_OUTPUT_ROOT/finetune/influenza/chronos2_full/GA_NY_NJ-gemini-complete/scale/
#         Qwen3-8B_scale_lr0.0001_sched4-1to0.5-4/{metrics.json,predictions.csv,grounding.json}
#   sbatch run_finetune_influenza_scale.sbatch
#   sbatch run_finetune_influenza_scale.sbatch --data-only     # only write train/val/test jsonl
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}"      # submit from code/finetune/
PYTHON="${PYTHON:-python}"                                      # python with torch + ms-swift
export PATH="$(dirname "$(command -v "$PYTHON")"):$PATH"       # the `swift` CLI next to it
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUN="$("$PYTHON" -c 'import sys; sys.path.insert(0, ".."); import paper_paths as P; print(P.FINETUNE_OUTPUT_ROOT)')/influenza/chronos2_full/GA_NY_NJ-gemini/scale/Qwen3-8B_scale_lr0.0001_sched4-1to0.5-4"
echo "=== influenza SCALe finetune start $(date) on $(hostname) ==="
"$PYTHON" -u finetune.py --data-mode influenza --mode chronos2_full --method scale \
    --corpus-model gemini --states GA,NY,NJ \
    --lr 1e-4 --scale-schedule "4:1>0.5:4" --scale-epochs 20 --class-weights 1:1:1 "$@"
if [[ " $* " != *" --data-only "* ]]; then
    echo "=== post-hoc grounding (bin -> magnitude fitted on val) ==="
    "$PYTHON" -u train_grounding.py "$RUN"
fi
echo "=== done $(date) ==="
