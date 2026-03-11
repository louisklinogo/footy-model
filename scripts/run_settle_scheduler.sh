#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_scheduler_common.sh
source "$SCRIPT_DIR/_scheduler_common.sh"

JOB_LOG="artifacts/logs/settle_scheduler.log"

scheduler_prepare "settle_scheduler" 360
scheduler_acquire_lock "settle scheduler"
scheduler_run \
  "Settle scheduler" \
  "$JOB_LOG" \
  "$FOOTY_PYTHON_BIN" \
  "src/jobs/tick_due_fixtures_v1.py" \
  "--skip-predict" \
  "--skip-score" \
  "--max-settle" "100" \
  "--log-file" "$JOB_LOG"

