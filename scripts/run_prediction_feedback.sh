#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_scheduler_common.sh
source "$SCRIPT_DIR/_scheduler_common.sh"

JOB_LOG="artifacts/logs/prediction_feedback.log"

scheduler_prepare "prediction_feedback" 1440
scheduler_acquire_lock "prediction feedback scheduler"
scheduler_run \
  "Prediction feedback scheduler" \
  "$JOB_LOG" \
  "$FOOTY_PYTHON_BIN" \
  "src/analysis/prediction_feedback.py"
