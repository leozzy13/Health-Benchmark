#!/usr/bin/env bash
set -euo pipefail

: "${MODEL:?}"
: "${MODEL_SLUG:?}"
: "${TENSOR_PARALLEL_SIZE:?}"
: "${MAX_MODEL_LEN:?}"
: "${VLLM_PORT:?}"
: "${SINGULARITY_BIN:?}"
: "${VLLM_IMAGE:?}"
: "${VLLM_LOG:?}"
: "${VLLM_PID_FILE:?}"

RAY_PORT="${RAY_PORT:-6379}"
RAY_CLUSTER_PID_FILE="${RAY_CLUSTER_PID_FILE:-}"
SLURM_NODE_COUNT="${SLURM_JOB_NUM_NODES:-1}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
RAY_LOG_DIR="${RAY_LOG_DIR:-$(dirname "$VLLM_LOG")/ray_${MODEL_SLUG}_${SLURM_JOB_ID:-manual}}"
CUDA_VISIBLE_DEVICES_OVERRIDE="${CUDA_VISIBLE_DEVICES_OVERRIDE:-}"

launch_local_server() {
  if [[ -n "$CUDA_VISIBLE_DEVICES_OVERRIDE" ]]; then
    export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_OVERRIDE"
  else
    unset CUDA_VISIBLE_DEVICES
  fi
  exec "$SINGULARITY_BIN" exec --nv -B /projects:/projects "$VLLM_IMAGE" \
    vllm serve "$MODEL" \
    --host 127.0.0.1 \
    --port "$VLLM_PORT" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --reasoning-parser "$REASONING_PARSER"
}

if [[ "$SLURM_NODE_COUNT" -gt 1 && "$TENSOR_PARALLEL_SIZE" -gt 4 ]]; then
  if [[ -n "$CUDA_VISIBLE_DEVICES_OVERRIDE" ]]; then
    echo "CUDA_VISIBLE_DEVICES_OVERRIDE is not supported for distributed ray launches." >&2
    exit 2
  fi
  HEAD_NODE="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
  mkdir -p "$RAY_LOG_DIR"
  export HEAD_NODE RAY_PORT SINGULARITY_BIN VLLM_IMAGE
  srun --nodes="$SLURM_NODE_COUNT" --ntasks="$SLURM_NODE_COUNT" --ntasks-per-node=1 bash -lc '
    set -euo pipefail
    NODE_NAME="$(hostname -s)"
    export NODE_NAME
    "$SINGULARITY_BIN" exec --nv -B /projects:/projects "$VLLM_IMAGE" bash -lc "
set -euo pipefail
if [[ \"\$NODE_NAME\" == \"$HEAD_NODE\" ]]; then
  ray start --head --port \"$RAY_PORT\" --disable-usage-stats
else
  ray start --address \"$HEAD_NODE:$RAY_PORT\" --disable-usage-stats
fi
trap \"ray stop --force >/dev/null 2>&1 || true; exit 0\" TERM INT
while true; do sleep 30; done
"
  ' >"$RAY_LOG_DIR/cluster.log" 2>&1 &
  RAY_CLUSTER_PID=$!
  if [[ -n "$RAY_CLUSTER_PID_FILE" ]]; then
    printf '%s\n' "$RAY_CLUSTER_PID" >"$RAY_CLUSTER_PID_FILE"
  fi
  sleep 15
  srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" bash -lc "
    exec \"$SINGULARITY_BIN\" exec --nv -B /projects:/projects \"$VLLM_IMAGE\" \
      vllm serve \"$MODEL\" \
      --host 127.0.0.1 \
      --port \"$VLLM_PORT\" \
      --tensor-parallel-size \"$TENSOR_PARALLEL_SIZE\" \
      --max-model-len \"$MAX_MODEL_LEN\" \
      --reasoning-parser \"$REASONING_PARSER\" \
      --distributed-executor-backend ray
  " >"$VLLM_LOG" 2>&1 &
  VLLM_PID=$!
else
  ( launch_local_server ) >"$VLLM_LOG" 2>&1 &
  VLLM_PID=$!
fi

printf '%s\n' "$VLLM_PID" >"$VLLM_PID_FILE"
