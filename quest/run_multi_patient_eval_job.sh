#!/usr/bin/env bash
set -euo pipefail

: "${REPO_ROOT:?}"
: "${OUTPUT_ROOT:?}"
: "${PROJECT_ROOT:?}"
: "${SINGULARITY_BIN:?}"
: "${VLLM_IMAGE:?}"

PATIENT_MANIFEST="${PATIENT_MANIFEST:-}"
QUEST_JOB_OUTPUT_ROOT="${QUEST_JOB_OUTPUT_ROOT:-$OUTPUT_ROOT/quest_job_outputs/${SLURM_JOB_ID:-manual}}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
SERVER_READY_TIMEOUT_S="${SERVER_READY_TIMEOUT_S:-1800}"
EVAL_MODEL_PRESET="${EVAL_MODEL_PRESET:-trio}"
ANSWER_GPU_DEVICE_IDS="${ANSWER_GPU_DEVICE_IDS:-0}"
JUDGE_MODEL="Qwen/Qwen3.5-27B"
JUDGE_MODEL_SLUG="qwen3.5-27b-judge"
ACTIVE_SERVER_SLUG=""

mkdir -p "$QUEST_JOB_OUTPUT_ROOT"

if [[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]]; then
  DEFAULT_ANSWER_VLLM_PORT="$((20000 + (SLURM_JOB_ID % 10000) * 4))"
else
  DEFAULT_ANSWER_VLLM_PORT="8000"
fi
ANSWER_VLLM_PORT="${ANSWER_VLLM_PORT:-${VLLM_PORT:-$DEFAULT_ANSWER_VLLM_PORT}}"

validate_tcp_port() {
  local label="$1"
  local value="$2"

  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "${label} must be a numeric TCP port, got: ${value}" >&2
    exit 2
  fi
  if (( value < 1 || value > 65535 )); then
    echo "${label} must be between 1 and 65535, got: ${value}" >&2
    exit 2
  fi
}

validate_tcp_port "ANSWER_VLLM_PORT" "$ANSWER_VLLM_PORT"
DEFAULT_JUDGE_VLLM_PORT="$((ANSWER_VLLM_PORT + 2))"
JUDGE_VLLM_PORT="${JUDGE_VLLM_PORT:-$DEFAULT_JUDGE_VLLM_PORT}"
validate_tcp_port "JUDGE_VLLM_PORT" "$JUDGE_VLLM_PORT"
if [[ "$ANSWER_VLLM_PORT" == "$JUDGE_VLLM_PORT" ]]; then
  echo "ANSWER_VLLM_PORT and JUDGE_VLLM_PORT must differ, got: ${ANSWER_VLLM_PORT}" >&2
  exit 2
fi

echo "Using answer vLLM port: $ANSWER_VLLM_PORT"
echo "Using judge vLLM port: $JUDGE_VLLM_PORT"

case "$EVAL_MODEL_PRESET" in
  small)
    DEFAULT_JUDGE_TENSOR_PARALLEL_SIZE="2"
    DEFAULT_JUDGE_MAX_MODEL_LEN="131072"
    DEFAULT_JUDGE_GPU_DEVICE_IDS=""
    ;;
  small_1gpu)
    DEFAULT_JUDGE_TENSOR_PARALLEL_SIZE="1"
    DEFAULT_JUDGE_MAX_MODEL_LEN="32768"
    DEFAULT_JUDGE_GPU_DEVICE_IDS="0"
    ;;
  *)
    DEFAULT_JUDGE_TENSOR_PARALLEL_SIZE="2"
    DEFAULT_JUDGE_MAX_MODEL_LEN="131072"
    DEFAULT_JUDGE_GPU_DEVICE_IDS=""
    ;;
esac

JUDGE_TENSOR_PARALLEL_SIZE="${JUDGE_TENSOR_PARALLEL_SIZE:-$DEFAULT_JUDGE_TENSOR_PARALLEL_SIZE}"
JUDGE_MAX_MODEL_LEN="${JUDGE_MAX_MODEL_LEN:-$DEFAULT_JUDGE_MAX_MODEL_LEN}"
JUDGE_GPU_DEVICE_IDS="${JUDGE_GPU_DEVICE_IDS:-$DEFAULT_JUDGE_GPU_DEVICE_IDS}"

echo "Using judge tensor parallel size: $JUDGE_TENSOR_PARALLEL_SIZE"
echo "Using judge max model len: $JUDGE_MAX_MODEL_LEN"
echo "Using judge GPU device ids: ${JUDGE_GPU_DEVICE_IDS:-all visible GPUs}"

resolve_model_records() {
  case "${1:-}" in
    trio)
      echo "EVAL_MODEL_PRESET=trio is retired for judged evaluation. Use qwen_open_eval_small_models.slurm for 4B/9B and qwen_open_eval_27b_2gpu.slurm for 27B." >&2
      exit 2
      ;;
    small)
      cat <<'EOF'
Qwen/Qwen3.5-4B|qwen3.5-4b|1|262144
Qwen/Qwen3.5-9B|qwen3.5-9b|1|262144
EOF
      ;;
    small_1gpu)
      cat <<'EOF'
Qwen/Qwen3.5-4B|qwen3.5-4b|1|262144
Qwen/Qwen3.5-9B|qwen3.5-9b|1|262144
EOF
      ;;
    27b_2gpu)
      cat <<'EOF'
Qwen/Qwen3.5-27B|qwen3.5-27b|2|131072
EOF
      ;;
    *)
      echo "Unknown EVAL_MODEL_PRESET: ${1:-}" >&2
      exit 2
      ;;
  esac
}

if [[ -z "$PATIENT_MANIFEST" ]]; then
  if [[ "${1:-}" == "--patient-manifest" ]]; then
    if [[ -z "${2:-}" ]]; then
      echo "--patient-manifest requires a path argument" >&2
      exit 2
    fi
    PATIENT_MANIFEST="$2"
    shift 2
  fi
  if [[ -z "$PATIENT_MANIFEST" && "$#" -eq 0 ]]; then
    echo "Pass subject_ids or set PATIENT_MANIFEST=/path/to/manifest.txt" >&2
    exit 2
  fi
  if [[ -z "$PATIENT_MANIFEST" ]]; then
    PATIENT_MANIFEST="$QUEST_JOB_OUTPUT_ROOT/patient_manifest_snapshot.txt"
    printf '%s\n' "$@" >"$PATIENT_MANIFEST"
  else
    cp "$PATIENT_MANIFEST" "$QUEST_JOB_OUTPUT_ROOT/patient_manifest_snapshot.txt"
  fi
else
  cp "$PATIENT_MANIFEST" "$QUEST_JOB_OUTPUT_ROOT/patient_manifest_snapshot.txt"
fi

mapfile -t MODELS < <(resolve_model_records "$EVAL_MODEL_PRESET")

job_summary_json="$QUEST_JOB_OUTPUT_ROOT/job_summary.json"
job_summary_csv="$QUEST_JOB_OUTPUT_ROOT/job_summary.csv"
tmp_rows="$QUEST_JOB_OUTPUT_ROOT/.job_rows.jsonl"
: >"$tmp_rows"

server_log_path() {
  local server_slug="$1"
  printf '%s\n' "$QUEST_JOB_OUTPUT_ROOT/${server_slug}_vllm_${SLURM_JOB_ID:-manual}.log"
}

server_pid_path() {
  local server_slug="$1"
  printf '%s\n' "$QUEST_JOB_OUTPUT_ROOT/.${server_slug}_vllm.pid"
}

server_ray_pid_path() {
  local server_slug="$1"
  printf '%s\n' "$QUEST_JOB_OUTPUT_ROOT/.${server_slug}_ray.pid"
}

server_ray_log_dir() {
  local server_slug="$1"
  printf '%s\n' "$QUEST_JOB_OUTPUT_ROOT/${server_slug}_ray_logs"
}

stop_named_server() {
  local server_slug="$1"
  if [[ -z "$server_slug" ]]; then
    return 0
  fi
  export VLLM_PID_FILE="$(server_pid_path "$server_slug")"
  export RAY_CLUSTER_PID_FILE="$(server_ray_pid_path "$server_slug")"
  "$REPO_ROOT/quest/stop_vllm_server.sh"
}

cleanup() {
  stop_named_server "$ACTIVE_SERVER_SLUG"
}
trap cleanup EXIT

start_named_server() {
  local model="$1"
  local server_slug="$2"
  local tensor_parallel_size="$3"
  local max_model_len="$4"
  local port="$5"
  local gpu_device_ids="${6:-}"
  local vllm_log
  local vllm_pid_file
  local ray_cluster_pid_file
  local ray_log_dir

  vllm_log="$(server_log_path "$server_slug")"
  vllm_pid_file="$(server_pid_path "$server_slug")"
  ray_cluster_pid_file="$(server_ray_pid_path "$server_slug")"
  ray_log_dir="$(server_ray_log_dir "$server_slug")"

  if [[ -n "$gpu_device_ids" ]]; then
    MODEL="$model" \
      MODEL_SLUG="$server_slug" \
      TENSOR_PARALLEL_SIZE="$tensor_parallel_size" \
      MAX_MODEL_LEN="$max_model_len" \
      REASONING_PARSER="$REASONING_PARSER" \
      VLLM_PORT="$port" \
      VLLM_LOG="$vllm_log" \
      VLLM_PID_FILE="$vllm_pid_file" \
      RAY_CLUSTER_PID_FILE="$ray_cluster_pid_file" \
      RAY_LOG_DIR="$ray_log_dir" \
      CUDA_VISIBLE_DEVICES_OVERRIDE="$gpu_device_ids" \
      "$REPO_ROOT/quest/launch_vllm_server.sh"
  else
    MODEL="$model" \
      MODEL_SLUG="$server_slug" \
      TENSOR_PARALLEL_SIZE="$tensor_parallel_size" \
      MAX_MODEL_LEN="$max_model_len" \
      REASONING_PARSER="$REASONING_PARSER" \
      VLLM_PORT="$port" \
      VLLM_LOG="$vllm_log" \
      VLLM_PID_FILE="$vllm_pid_file" \
      RAY_CLUSTER_PID_FILE="$ray_cluster_pid_file" \
      RAY_LOG_DIR="$ray_log_dir" \
      "$REPO_ROOT/quest/launch_vllm_server.sh"
  fi
  if ! python "$REPO_ROOT/quest/wait_for_server.py" \
    --base-url "http://127.0.0.1:${port}/v1" \
    --expected-model "$model" \
    --timeout-seconds "$SERVER_READY_TIMEOUT_S"; then
    VLLM_PID_FILE="$vllm_pid_file" \
      RAY_CLUSTER_PID_FILE="$ray_cluster_pid_file" \
      "$REPO_ROOT/quest/stop_vllm_server.sh"
    return 1
  fi
  ACTIVE_SERVER_SLUG="$server_slug"
  return 0
}

run_answers_stage() {
  local model="$1"
  local model_slug="$2"
  local answer_port="$3"
  set +e
  python "$REPO_ROOT/main.py" evaluate \
    --stage answers \
    --provider vllm \
    --base-url "http://127.0.0.1:${answer_port}/v1" \
    --output-root "$OUTPUT_ROOT" \
    --patient-manifest "$PATIENT_MANIFEST" \
    --models "$model" \
    --replace-existing \
    >"$QUEST_JOB_OUTPUT_ROOT/${model_slug}_answers.log" 2>&1
  local exit_code=$?
  set -e
  return "$exit_code"
}

run_judge_stage() {
  local model="$1"
  local model_slug="$2"
  local judge_port="$3"
  set +e
  python "$REPO_ROOT/main.py" evaluate \
    --stage judge \
    --provider vllm \
    --judge-base-url "http://127.0.0.1:${judge_port}/v1" \
    --output-root "$OUTPUT_ROOT" \
    --patient-manifest "$PATIENT_MANIFEST" \
    --models "$model" \
    --replace-existing \
    >"$QUEST_JOB_OUTPUT_ROOT/${model_slug}_judge.log" 2>&1
  local exit_code=$?
  set -e
  return "$exit_code"
}

run_full_stage() {
  local model="$1"
  local model_slug="$2"
  local answer_port="$3"
  set +e
  python "$REPO_ROOT/main.py" evaluate \
    --provider vllm \
    --base-url "http://127.0.0.1:${answer_port}/v1" \
    --output-root "$OUTPUT_ROOT" \
    --patient-manifest "$PATIENT_MANIFEST" \
    --models "$model" \
    --replace-existing \
    >"$QUEST_JOB_OUTPUT_ROOT/${model_slug}_evaluation.log" 2>&1
  local exit_code=$?
  set -e
  return "$exit_code"
}

for model_record in "${MODELS[@]}"; do
  IFS='|' read -r MODEL MODEL_SLUG TENSOR_PARALLEL_SIZE MAX_MODEL_LEN <<<"$model_record"

  if [[ "$EVAL_MODEL_PRESET" == "small" || "$EVAL_MODEL_PRESET" == "small_1gpu" ]]; then
    if ! start_named_server "$MODEL" "$MODEL_SLUG" "$TENSOR_PARALLEL_SIZE" "$MAX_MODEL_LEN" "$ANSWER_VLLM_PORT" "$ANSWER_GPU_DEVICE_IDS"; then
      echo "{\"model_slug\": \"${MODEL_SLUG}\", \"model_name\": \"${MODEL}\", \"status\": \"answer_server_failed\"}" >>"$tmp_rows"
      continue
    fi
    if run_answers_stage "$MODEL" "$MODEL_SLUG" "$ANSWER_VLLM_PORT"; then
      answers_exit_code=0
    else
      answers_exit_code=$?
    fi
    stop_named_server "$MODEL_SLUG"
    ACTIVE_SERVER_SLUG=""

    if [[ "$answers_exit_code" -ne 0 ]]; then
      echo "{\"model_slug\": \"${MODEL_SLUG}\", \"model_name\": \"${MODEL}\", \"status\": \"answers_failed\", \"exit_code\": ${answers_exit_code}}" >>"$tmp_rows"
      continue
    fi

    if ! start_named_server "$JUDGE_MODEL" "$JUDGE_MODEL_SLUG" "$JUDGE_TENSOR_PARALLEL_SIZE" "$JUDGE_MAX_MODEL_LEN" "$JUDGE_VLLM_PORT" "$JUDGE_GPU_DEVICE_IDS"; then
      echo "{\"model_slug\": \"${MODEL_SLUG}\", \"model_name\": \"${MODEL}\", \"status\": \"judge_server_failed\"}" >>"$tmp_rows"
      continue
    fi
    if run_judge_stage "$MODEL" "$MODEL_SLUG" "$JUDGE_VLLM_PORT"; then
      judge_exit_code=0
    else
      judge_exit_code=$?
    fi
    stop_named_server "$JUDGE_MODEL_SLUG"
    ACTIVE_SERVER_SLUG=""

    if [[ "$judge_exit_code" -ne 0 ]]; then
      echo "{\"model_slug\": \"${MODEL_SLUG}\", \"model_name\": \"${MODEL}\", \"status\": \"judge_failed\", \"exit_code\": ${judge_exit_code}}" >>"$tmp_rows"
      continue
    fi

    echo "{\"model_slug\": \"${MODEL_SLUG}\", \"model_name\": \"${MODEL}\", \"status\": \"completed\", \"exit_code\": 0}" >>"$tmp_rows"
    continue
  fi

  if ! start_named_server "$MODEL" "$MODEL_SLUG" "$TENSOR_PARALLEL_SIZE" "$MAX_MODEL_LEN" "$ANSWER_VLLM_PORT"; then
    echo "{\"model_slug\": \"${MODEL_SLUG}\", \"model_name\": \"${MODEL}\", \"status\": \"server_failed\"}" >>"$tmp_rows"
    continue
  fi
  if run_full_stage "$MODEL" "$MODEL_SLUG" "$ANSWER_VLLM_PORT"; then
    exit_code=0
  else
    exit_code=$?
  fi
  stop_named_server "$MODEL_SLUG"
  ACTIVE_SERVER_SLUG=""

  if [[ "$exit_code" -ne 0 ]]; then
    echo "{\"model_slug\": \"${MODEL_SLUG}\", \"model_name\": \"${MODEL}\", \"status\": \"failed\", \"exit_code\": ${exit_code}}" >>"$tmp_rows"
    continue
  fi
  echo "{\"model_slug\": \"${MODEL_SLUG}\", \"model_name\": \"${MODEL}\", \"status\": \"completed\", \"exit_code\": 0}" >>"$tmp_rows"
done

python - <<'PY' "$tmp_rows" "$job_summary_json" "$job_summary_csv" "$PATIENT_MANIFEST"
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

rows_path = Path(sys.argv[1])
summary_json = Path(sys.argv[2])
summary_csv = Path(sys.argv[3])
manifest_path = Path(sys.argv[4])
rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]
summary = {
    "patient_manifest": str(manifest_path),
    "rows": rows,
    "failed_models": [row["model_slug"] for row in rows if row.get("exit_code", 1) != 0 or row["status"] != "completed"],
}
summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
with summary_csv.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["model_slug", "model_name", "status", "exit_code"])
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                "model_slug": row.get("model_slug"),
                "model_name": row.get("model_name"),
                "status": row.get("status"),
                "exit_code": row.get("exit_code", ""),
            }
        )
sys.exit(1 if summary["failed_models"] else 0)
PY
