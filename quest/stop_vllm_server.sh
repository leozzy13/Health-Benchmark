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
    for _ in $(seq 1 20); do
      if ! kill -0 "$pid" >/dev/null 2>&1; then
        break
      fi
      sleep 1
    done
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill -9 "$pid" >/dev/null 2>&1 || true
      for _ in $(seq 1 5); do
        if ! kill -0 "$pid" >/dev/null 2>&1; then
          break
        fi
        sleep 1
      done
    fi
  fi
  rm -f "$pid_file"
}

terminate_pid_file "$VLLM_PID_FILE"
terminate_pid_file "$RAY_CLUSTER_PID_FILE"
