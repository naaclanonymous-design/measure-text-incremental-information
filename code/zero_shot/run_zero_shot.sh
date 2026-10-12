#!/usr/bin/env bash
# Zero-shot OPERATOR test of the hosted models on the same test prompts the finetuned
# operator sees — Gemini-3.1-pro (thinking level low, the model's floor) and GPT-5.4
# (reasoning_effort none) — WITH the weekly reports (--text keep: history + Chronos-2 baseline
# [+ recent errors on influenza] + the last 5 weeks of reports) and WITHOUT them (--text drop:
# the report block replaced by the notext stub).
#
# Usage: run_zero_shot.sh influenza|covid [keep|drop|both] [gemini,gpt] [extra args...]
#   influenza : --data-mode influenza (GA/NY/NJ test split of the released corpus, 101 rows)
#   covid     : --data-mode covid_TSFM_bitong_org (TX/VT/NJ final test block, 78 rows; the
#               finetune prompt of prompts/covid, which has NO recent-error item)
# Credentials: Gemini via Vertex AI ADC (GOOGLE_CLOUD_PROJECT); GPT via OPENAI_API_KEY or
#              credentials/openai_api_key.json.
# Output: $PAPER_OUTPUT_ROOT/zero_shot/<data_mode>/chronos2_full/<combo>/<text5wk_input|
#         scale_residualcot_notext_input>/<model_tag>/{responses.jsonl,predictions.csv,metrics.json}
# Add --split val to score the validation rows too, then ground_zero_shot_api.py fits the
# per-bin magnitudes on val and adds grounded_* MAE fields to the test metrics.
set -euo pipefail
DATASET="${1:?usage: run_zero_shot.sh influenza|covid [keep|drop|both] [providers]}"
TEXT="${2:-both}"; PROVIDERS="${3:-gemini,gpt}"; shift $(( $# >= 3 ? 3 : $# ))
PYTHON="${PYTHON:-python}"
cd "$(dirname "${BASH_SOURCE[0]}")"
case "$DATASET" in
  influenza) ARGS=(--data-mode influenza --corpus-model gemini-final --states GA,NY,NJ) ;;
  covid)     ARGS=(--data-mode covid_TSFM_bitong_org --corpus-model gemini-final --states TX,VT,NJ) ;;
  *) echo "unknown dataset: $DATASET"; exit 1 ;;
esac
[[ "$TEXT" == "both" ]] && VARIANTS=(keep drop) || VARIANTS=("$TEXT")
LOGDIR="$("$PYTHON" -c 'import sys; sys.path.insert(0, ".."); import paper_paths as P; print(P.OUTPUT_ROOT / "logs")')"
mkdir -p "$LOGDIR"
for v in "${VARIANTS[@]}"; do
  echo "=== zero-shot $DATASET text=$v providers=$PROVIDERS $(date) ==="
  "$PYTHON" -u zero_shot_api_notext.py "${ARGS[@]}" --providers "$PROVIDERS" --text "$v" "$@" \
      2>&1 | tee -a "$LOGDIR/zero_shot_${DATASET}_${v}_$(date +%Y%m%d_%H%M%S).log"
done
