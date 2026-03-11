#!/usr/bin/env bash
set -euo pipefail

FOOTY_ROOT="${FOOTY_ROOT:-/mnt/c/Developer/soccer/footy-model}"
FOOTY_PYTHON_BIN="${FOOTY_PYTHON_BIN:-/mnt/c/Python313/python.exe}"

scheduler_ts() {
  date --iso-8601=seconds
}

scheduler_log() {
  printf '[%s] %s\n' "$(scheduler_ts)" "$1" >> "$SCHEDULER_STDOUT_LOG"
}

scheduler_prepare() {
  local scheduler_name="$1"
  local stale_minutes="$2"

  SCHEDULER_NAME="$scheduler_name"
  STALE_MINUTES="$stale_minutes"
  SCHEDULER_LOG_DIR="$FOOTY_ROOT/artifacts/logs"
  SCHEDULER_LOCK_DIR="$FOOTY_ROOT/artifacts/locks"

  mkdir -p "$SCHEDULER_LOG_DIR" "$SCHEDULER_LOCK_DIR"

  SCHEDULER_STDOUT_LOG="$SCHEDULER_LOG_DIR/${SCHEDULER_NAME}.stdout.log"
  SCHEDULER_LOCKFILE="$SCHEDULER_LOCK_DIR/${SCHEDULER_NAME}.lock"
}

scheduler_prepare_validation_dirs() {
  mkdir -p \
    "$FOOTY_ROOT/artifacts/reports/layer2_feature_health" \
    "$FOOTY_ROOT/artifacts/reports/monitoring" \
    "$FOOTY_ROOT/artifacts/reports/leakage_audit"
}

scheduler_is_stale_lock() {
  find "$SCHEDULER_LOCKFILE" -mmin "+${STALE_MINUTES}" -print -quit 2>/dev/null | grep -q .
}

scheduler_acquire_lock() {
  local job_label="$1"

  if [[ -e "$SCHEDULER_LOCKFILE" ]]; then
    if scheduler_is_stale_lock; then
      rm -f "$SCHEDULER_LOCKFILE"
      scheduler_log "Removed stale ${SCHEDULER_NAME} lock older than ${STALE_MINUTES} minutes."
    else
      scheduler_log "Skip: ${job_label} already running."
      exit 0
    fi
  fi

  : > "$SCHEDULER_LOCKFILE"
  trap 'rm -f "$SCHEDULER_LOCKFILE"' EXIT
}

scheduler_run() {
  local job_label="$1"
  local job_log="$2"
  shift 2

  local -a command=("$@")
  local -a runner=()
  local rc=0

  scheduler_log "${job_label} start."
  if [[ -n "$job_log" ]]; then
    printf 'Log file: %s\n' "$job_log" >> "$SCHEDULER_STDOUT_LOG"
  fi

  if [[ -n "${MAX_RUNTIME_MINUTES:-}" ]]; then
    runner=(timeout --foreground "${MAX_RUNTIME_MINUTES}m")
  fi

  set +e
  (
    cd "$FOOTY_ROOT"
    "${runner[@]}" "${command[@]}" >> "$SCHEDULER_STDOUT_LOG" 2>&1
  )
  rc=$?
  set -e

  scheduler_log "${job_label} end rc=${rc}."
  return "$rc"
}

