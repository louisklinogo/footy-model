#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_scheduler_common.sh
source "$SCRIPT_DIR/_scheduler_common.sh"

scheduler_prepare "formations_scheduler" 360
scheduler_acquire_lock "formations scheduler"
scheduler_run \
  "Formations scheduler" \
  "" \
  "$FOOTY_PYTHON_BIN" \
  "src/ingest/ingest_sofascore_formations.py" \
  "--status" "scheduled" \
  "--limit" "250"
