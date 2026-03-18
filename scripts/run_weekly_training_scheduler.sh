#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_scheduler_common.sh
source "$SCRIPT_DIR/_scheduler_common.sh"

JOB_LOG="artifacts/logs/weekly_training_scheduler.log"

scheduler_prepare "weekly_training_scheduler" 1440
scheduler_acquire_lock "weekly training scheduler"
scheduler_run \
  "Weekly training scheduler" \
  "$JOB_LOG" \
  "$FOOTY_PYTHON_BIN" \
  "src/jobs/train_weekly_v2.py" \
  "--log-file" "$JOB_LOG"
