#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_scheduler_common.sh
source "$SCRIPT_DIR/_scheduler_common.sh"

JOB_LOG="artifacts/logs/score_scheduler_hybrid.log"
HYBRID_MODEL_NAME="${HYBRID_MODEL_NAME:-market_outcome_v2}"
HYBRID_MODEL_VERSION="${HYBRID_MODEL_VERSION:-hybrid_v1}"

scheduler_prepare "score_scheduler_hybrid" 360
scheduler_acquire_lock "score scheduler hybrid"
scheduler_run \
  "Score scheduler hybrid" \
  "$JOB_LOG" \
  "$FOOTY_PYTHON_BIN" \
  "src/jobs/tick_due_fixtures_v1.py" \
  "--skip-settle" \
  "--skip-predict" \
  "--predict-runtime" "hybrid_v2" \
  "--hybrid-model-name" "$HYBRID_MODEL_NAME" \
  "--hybrid-model-version" "$HYBRID_MODEL_VERSION" \
  "--max-score" "300" \
  "--score-since-days" "30" \
  "--log-file" "$JOB_LOG"
