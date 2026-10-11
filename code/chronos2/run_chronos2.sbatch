#!/bin/bash
#SBATCH --job-name=chronos2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
# Chronos-2 baselines of the two datasets (submit from code/chronos2/):
#   sbatch run_chronos2.sbatch covid       univariate, raw + z26w series -> data/COVID/chronos2/*.csv
#   sbatch run_chronos2.sbatch influenza   multivariate (state + other states + USA as covariates)
#                                          -> data/Influenza/chronos2/chronos2_influenza_predictions.csv
# Both are resumable: only (state, reference_date) windows missing from the caches are computed.
# Environment knobs: PYTHON (python with torch + chronos-forecasting; default `python`),
#                    CHRONOS2_MODEL_PATH (default <paper-content>/checkpoints/chronos-2).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}"
PYTHON="${PYTHON:-python}"
DATASET="${1:?usage: run_chronos2.sbatch covid|influenza [extra args]}"; shift
echo "=== chronos2 $DATASET start $(date) on $(hostname); python=$PYTHON ==="
case "$DATASET" in
  covid)     "$PYTHON" chronos2_covid.py --series both "$@" ;;
  influenza) "$PYTHON" chronos2_influenza.py "$@"
             "$PYTHON" chronos2_influenza.py --states USA --max-covariates 20 "$@" ;;
  *) echo "unknown dataset: $DATASET"; exit 1 ;;
esac
echo "=== done $(date) ==="
