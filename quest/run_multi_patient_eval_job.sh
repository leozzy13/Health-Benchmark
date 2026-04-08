#!/usr/bin/env bash
set -euo pipefail

: "${REPO_ROOT:?}"
: "${OUTPUT_ROOT:?}"
: "${PROJECT_ROOT:?}"
: "${SINGULARITY_BIN:?}"
: "${VLLM_IMAGE:?}"

PATIENT_MANIFEST="${PATIENT_MANIFEST:-}"
QUEST_JOB_OUTPUT_ROOT="${QUEST_JOB_OUTPUT_ROOT:-$OUTPUT_ROOT/quest_job_outputs/${SLURM_JOB_ID:-manual}}"
VLLM_PORT="${VLLM_PORT:-8000}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
SERVER_READY_TIMEOUT_S="${SERVER_READY_TIMEOUT_S:-1800}"

mkdir -p "$QUEST_JOB_OUTPUT_ROOT"

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

MODELS=(
  "Qwen/Qwen3.5-4B|qwen3.5-4b|1|262144"
  "Qwen/Qwen3.5-9B|qwen3.5-9b|1|262144"
  "Qwen/Qwen3.5-27B|qwen3.5-27b|8|262144"
)

job_summary_json="$QUEST_JOB_OUTPUT_ROOT/job_summary.json"
job_summary_csv="$QUEST_JOB_OUTPUT_ROOT/job_summary.csv"
tmp_rows="$QUEST_JOB_OUTPUT_ROOT/.job_rows.jsonl"
: >"$tmp_rows"

for model_record in "${MODELS[@]}"; do
  IFS='|' read -r MODEL MODEL_SLUG TENSOR_PARALLEL_SIZE MAX_MODEL_LEN <<<"$model_record"
  export MODEL MODEL_SLUG TENSOR_PARALLEL_SIZE MAX_MODEL_LEN REASONING_PARSER VLLM_PORT
  export VLLM_LOG="$QUEST_JOB_OUTPUT_ROOT/${MODEL_SLUG}_vllm_${SLURM_JOB_ID:-manual}.log"
  export VLLM_PID_FILE="$QUEST_JOB_OUTPUT_ROOT/.${MODEL_SLUG}_vllm.pid"
  export RAY_CLUSTER_PID_FILE="$QUEST_JOB_OUTPUT_ROOT/.${MODEL_SLUG}_ray.pid"
  export RAY_LOG_DIR="$QUEST_JOB_OUTPUT_ROOT/${MODEL_SLUG}_ray_logs"

  "$REPO_ROOT/quest/launch_vllm_server.sh"
  if ! python "$REPO_ROOT/quest/wait_for_server.py" --base-url "http://127.0.0.1:${VLLM_PORT}/v1" --timeout-seconds "$SERVER_READY_TIMEOUT_S"; then
    "$REPO_ROOT/quest/stop_vllm_server.sh"
    echo "{\"model_slug\": \"${MODEL_SLUG}\", \"model_name\": \"${MODEL}\", \"status\": \"server_failed\"}" >>"$tmp_rows"
    continue
  fi

  set +e
  python "$REPO_ROOT/main.py" evaluate \
    --provider vllm \
    --base-url "http://127.0.0.1:${VLLM_PORT}/v1" \
    --output-root "$OUTPUT_ROOT" \
    --patient-manifest "$PATIENT_MANIFEST" \
    --models "$MODEL" \
    --replace-existing \
    >"$QUEST_JOB_OUTPUT_ROOT/${MODEL_SLUG}_evaluation.log" 2>&1
  exit_code=$?
  set -e

  "$REPO_ROOT/quest/stop_vllm_server.sh"
  echo "{\"model_slug\": \"${MODEL_SLUG}\", \"model_name\": \"${MODEL}\", \"status\": \"completed\", \"exit_code\": ${exit_code}}" >>"$tmp_rows"
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
PY
