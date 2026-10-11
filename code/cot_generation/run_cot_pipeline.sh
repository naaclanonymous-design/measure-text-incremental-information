#!/usr/bin/env bash
# The 6-stage CoT generation pipeline for one dataset:
#   [1] step1_generate            direction CoTs (k=5 per train origin, Gemini-3.1-pro, T=0.9)
#   [2] step1_filter              offline direction check + verbatim quotes + Flash faithfulness
#                                 + grounding judges; alpha_1 self-consistency; basis selection
#   [3] step1_generate --round 2  momentum retry of the round-1 abstains (item 5 = momentum summary)
#   [4] step1_filter --round 2    refilter the retries + merge the two rounds
#   [5] step2_generate            size-class CoTs for every move date (designs the global bins)
#   [6] step2_filter              cold-start grounding (delta = sign(d) * class lower edge),
#                                 utility gate, judges, corpus finalisation -> accepted.jsonl
#
# Usage:  run_cot_pipeline.sh influenza|covid [WORKERS]
#   influenza : --data-mode influenza           --states GA,NY,NJ  (data/Influenza/{STATE}.csv,
#               prompts/influenza/chronos2_full/)
#   covid     : --data-mode covid_TSFM_updated  --states TX,VT,NJ  (data/COVID/cot_ready/final/{STATE}.csv,
#               prompts/covid/chronos2_full_abs_t3/)
# Credentials: Gemini via Vertex AI ADC (export GOOGLE_CLOUD_PROJECT=<project>; gcloud auth
#              application-default login or GOOGLE_APPLICATION_CREDENTIALS); see code/README.md.
# Output: $PAPER_OUTPUT_ROOT/cot_corpora/<data_mode>/chronos2_full/<STATES>-gemini/accepted.jsonl
# Every stage is resumable (re-running skips finished work). Logs: $PAPER_OUTPUT_ROOT/logs/cot_<dataset>_<ts>.log
set -euo pipefail
DATASET="${1:?usage: run_cot_pipeline.sh influenza|covid [WORKERS]}"
WORKERS="${2:-8}"
PYTHON="${PYTHON:-python}"
cd "$(dirname "${BASH_SOURCE[0]}")"
case "$DATASET" in
  influenza) DATA_MODE=influenza;          STATES="GA,NY,NJ" ;;
  covid)     DATA_MODE=covid_TSFM_updated; STATES="TX,VT,NJ" ;;
  *) echo "unknown dataset: $DATASET (influenza|covid)"; exit 1 ;;
esac
MODE=chronos2_full
COMMON=(--mode "$MODE" --states "$STATES" --workers "$WORKERS" --model gemini --data-mode "$DATA_MODE")
LOGDIR="$("$PYTHON" -c 'import sys; sys.path.insert(0, ".."); import paper_paths as P; print(P.OUTPUT_ROOT / "logs")')"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/cot_${DATASET}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "=== PIPELINE START dataset=$DATASET data-mode=$DATA_MODE states=$STATES workers=$WORKERS $(date) ==="
echo "---- [1/6] step1_generate            $(date) ----"; "$PYTHON" -u step1_generate.py "${COMMON[@]}"
echo "---- [2/6] step1_filter              $(date) ----"; "$PYTHON" -u step1_filter.py   "${COMMON[@]}"
echo "---- [3/6] step1_generate --round 2  $(date) ----"; "$PYTHON" -u step1_generate.py "${COMMON[@]}" --round 2
echo "---- [4/6] step1_filter --round 2    $(date) ----"; "$PYTHON" -u step1_filter.py   "${COMMON[@]}" --round 2
echo "---- [5/6] step2_generate            $(date) ----"; "$PYTHON" -u step2_generate.py "${COMMON[@]}"
echo "---- [6/6] step2_filter              $(date) ----"; "$PYTHON" -u step2_filter.py   "${COMMON[@]}"
RUN="$("$PYTHON" -c 'import sys; sys.path.insert(0, ".."); import paper_paths as P; print(P.COT_OUTPUT_DIR)')/$DATA_MODE/$MODE/$(echo "$STATES" | tr ',' '_')-gemini"
echo "=== PIPELINE DONE $(date) -> $RUN/accepted.jsonl ==="
"$PYTHON" - "$RUN" <<'PY'
import json, sys
from collections import Counter
rows = [json.loads(l) for l in open(f"{sys.argv[1]}/accepted.jsonl")]
print(f"accepted.jsonl: {len(rows)} rows  by state: {dict(Counter(r['state'] for r in rows))}  "
      f"by class: {dict(Counter(r['date_status'] for r in rows))}")
PY
