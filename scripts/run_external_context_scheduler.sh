#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_scheduler_common.sh
source "$SCRIPT_DIR/_scheduler_common.sh"

JOB_LOG="artifacts/logs/external_context_scheduler.log"
EXTERNAL_CONTEXT_LIMIT="${EXTERNAL_CONTEXT_LIMIT:-100}"
EXTERNAL_CONTEXT_STATUS="${EXTERNAL_CONTEXT_STATUS:-scheduled}"
EXTERNAL_CONTEXT_START_HOURS="${EXTERNAL_CONTEXT_START_HOURS:-0}"
EXTERNAL_CONTEXT_END_HOURS="${EXTERNAL_CONTEXT_END_HOURS:-72}"

scheduler_prepare "external_context_scheduler" 360
scheduler_acquire_lock "external context scheduler"
scheduler_run \
  "External context scheduler" \
  "$JOB_LOG" \
  "$FOOTY_PYTHON_BIN" \
  "src/ingest/ingest_external_context.py" \
  "--context-type" "all" \
  "--status" "$EXTERNAL_CONTEXT_STATUS" \
  "--limit" "$EXTERNAL_CONTEXT_LIMIT" \
  "--scheduled-start-hours" "$EXTERNAL_CONTEXT_START_HOURS" \
  "--scheduled-end-hours" "$EXTERNAL_CONTEXT_END_HOURS" \
  "--skip-bootstrap"
