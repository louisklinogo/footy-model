#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_scheduler_common.sh
source "$SCRIPT_DIR/_scheduler_common.sh"

JOB_LOG="artifacts/logs/odds_polling_scheduler.log"

scheduler_prepare "odds_polling_scheduler" 360
scheduler_acquire_lock "odds polling scheduler"
scheduler_run \
  "Odds polling scheduler" \
  "$JOB_LOG" \
  "$FOOTY_PYTHON_BIN" \
  "src/jobs/sofascore_odds_polling.py" \
  "--days" "3" \
  "--sleep-sec" "0.5" \
  "--log-file" "$JOB_LOG"

