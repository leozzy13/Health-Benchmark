#!/usr/bin/env bash
set -euo pipefail

VLLM_PID_FILE="${VLLM_PID_FILE:?}"
RAY_CLUSTER_PID_FILE="${RAY_CLUSTER_PID_FILE:-}"

terminate_pid_file() {
  local pid_file="$1"
  if [[ -z "$pid_file" || ! -f "$pid_file" ]]; then
    return 0
  fi
  local pid=""
  pid="$(cat "$pid_file")"
  if [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1; then
    kill "$pid" >/dev/null 2>&1 || true
    wait "$pid" >/dev/null 2>&1 || true
  fi
  rm -f "$pid_file"
}

terminate_pid_file "$VLLM_PID_FILE"
terminate_pid_file "$RAY_CLUSTER_PID_FILE"
