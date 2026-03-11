#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_scheduler_common.sh
source "$SCRIPT_DIR/_scheduler_common.sh"

scheduler_prepare "validation_loop_scheduler" 360
scheduler_prepare_validation_dirs
scheduler_acquire_lock "validation loop"

RUN_TS="$(date -u +"%Y%m%d_%H%M%S")"
HEALTH_OUT="artifacts/reports/layer2_feature_health/${RUN_TS}"
MONITOR_OUT="artifacts/reports/monitoring/${RUN_TS}"
LEAKAGE_SOURCE=".sisyphus/evidence/task-7-leakage-audit.json"
LEAKAGE_COPY="artifacts/reports/leakage_audit/leakage_audit_${RUN_TS}.json"

run_validation_step() {
  local -a command=("$@")
  local rc=0

  printf 'COMMAND: %s\n' "$(printf '%q ' "${command[@]}")" >> "$SCHEDULER_STDOUT_LOG"
  set +e
  (
    cd "$FOOTY_ROOT"
    "${command[@]}" >> "$SCHEDULER_STDOUT_LOG" 2>&1
  )
  rc=$?
  set -e
  return "$rc"
}

scheduler_log "Validation loop start run_ts=${RUN_TS}."
rc=0

mkdir -p "$FOOTY_ROOT/$HEALTH_OUT" "$FOOTY_ROOT/$MONITOR_OUT"

run_validation_step "$FOOTY_PYTHON_BIN" "src/modeling/layer2_situational/audit_feature_health.py" "--output-dir" "$HEALTH_OUT" || rc=$?
run_validation_step "$FOOTY_PYTHON_BIN" "src/db/audit_layer2_situational_leakage.py" "--days" "14" "--limit" "500" || rc=$?
if [[ -f "$FOOTY_ROOT/$LEAKAGE_SOURCE" ]]; then
  cp -f "$FOOTY_ROOT/$LEAKAGE_SOURCE" "$FOOTY_ROOT/$LEAKAGE_COPY"
fi
run_validation_step "$FOOTY_PYTHON_BIN" "src/modeling/layer2_situational/monitor_layer2_signals.py" "--days" "14" "--output-dir" "$MONITOR_OUT" || rc=$?

scheduler_log "Validation loop end run_ts=${RUN_TS} rc=${rc}."
exit "$rc"

