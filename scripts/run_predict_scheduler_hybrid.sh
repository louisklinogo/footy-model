#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_scheduler_common.sh
source "$SCRIPT_DIR/_scheduler_common.sh"

JOB_LOG="artifacts/logs/predict_scheduler_hybrid.log"
HYBRID_MODEL_NAME="${HYBRID_MODEL_NAME:-market_outcome_v2}"
HYBRID_MODEL_VERSION="${HYBRID_MODEL_VERSION:-hybrid_v1}"
HYBRID_EXPORT_OUT="${HYBRID_EXPORT_OUT:-storage/reports/market_predictions_hybrid_v1.csv}"

scheduler_prepare "predict_scheduler_hybrid" 360
scheduler_acquire_lock "predict scheduler hybrid"
scheduler_run \
  "Predict scheduler hybrid" \
  "$JOB_LOG" \
  "$FOOTY_PYTHON_BIN" \
  "src/jobs/tick_due_fixtures_v1.py" \
  "--skip-settle" \
  "--skip-score" \
  "--predict-runtime" "hybrid_v2" \
  "--hybrid-model-name" "$HYBRID_MODEL_NAME" \
  "--hybrid-model-version" "$HYBRID_MODEL_VERSION" \
  "--hybrid-export-out" "$HYBRID_EXPORT_OUT" \
  "--max-predict" "100" \
  "--predict-days" "3" \
  "--log-file" "$JOB_LOG"