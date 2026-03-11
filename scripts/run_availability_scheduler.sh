#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_scheduler_common.sh
source "$SCRIPT_DIR/_scheduler_common.sh"

scheduler_prepare "availability_scheduler" 360
scheduler_acquire_lock "availability scheduler"
scheduler_run \
  "Availability scheduler" \
  "" \
  "$FOOTY_PYTHON_BIN" \
  "src/ingest/ingest_sofascore_availability.py" \
  "--status" "scheduled" \
  "--limit" "250" \
  "--scheduled-start-hours" "0" \
  "--scheduled-end-hours" "72" \
  "--scheduled-order" "asc" \
  "--retry-404-cooldown-hours" "6"

