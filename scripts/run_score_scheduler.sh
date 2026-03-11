#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_scheduler_common.sh
source "$SCRIPT_DIR/_scheduler_common.sh"

JOB_LOG="artifacts/logs/score_scheduler.log"

scheduler_prepare "score_scheduler" 360
scheduler_acquire_lock "score scheduler"
scheduler_run \
  "Score scheduler" \
  "$JOB_LOG" \
  "$FOOTY_PYTHON_BIN" \
  "src/jobs/tick_due_fixtures_v1.py" \
  "--skip-settle" \
  "--skip-predict" \
  "--max-score" "300" \
  "--score-since-days" "30" \
  "--log-file" "$JOB_LOG"

