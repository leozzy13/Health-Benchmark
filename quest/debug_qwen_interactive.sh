#!/usr/bin/env bash

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-235B-A22B-Instruct-2507-FP8}"
VLLM_PORT="${VLLM_PORT:-8000}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-49152}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-3600}"
READY_CHECK_SLEEP_SECONDS="${READY_CHECK_SLEEP_SECONDS:-2}"
READY_CHECK_ATTEMPTS="${READY_CHECK_ATTEMPTS:-$(( (VLLM_ENGINE_READY_TIMEOUT_S + READY_CHECK_SLEEP_SECONDS - 1) / READY_CHECK_SLEEP_SECONDS ))}"
DEFAULT_PROJECT_ROOT="/projects/p33194/health-benchmark"
DEFAULT_MIMICIV_DIR="/projects/p33194/health-benchmark/data/mimic-iv"
DEFAULT_MIMICIV_NOTE_DIR="/projects/p33194/health-benchmark/data/mimic-iv-notes"
DEFAULT_OUTPUT_ROOT="/projects/p33194/medbench-output"
DEFAULT_HF_HOME="/projects/p33194/hf_cache"
DEFAULT_MAMBA_ROOT="/hpc/software/mamba/24.3.0"
DEFAULT_ENV_PREFIX="/projects/p33194/envs/medbench-qwen"
DEFAULT_SINGULARITY_BIN="/software/singularity/3.8.1/bin/singularity"
DEFAULT_VLLM_IMAGE="/projects/p33194/containers/vllm-openai_latest.sif"

normalize_quest_path() {
  local raw_path="$1"
  local normalized="$raw_path"

  if [[ "$raw_path" == "/gpfs/projects" ]]; then
    normalized="/projects"
  elif [[ "$raw_path" == /gpfs/projects/* ]]; then
    normalized="/projects/${raw_path#/gpfs/projects/}"
  fi

  if [[ "$normalized" != "$raw_path" && -e "$normalized" ]]; then
    printf '%s\n' "$normalized"
    return 0
  fi

  printf '%s\n' "$raw_path"
}

canonicalize_dir() {
  local raw_path="$1"
  local candidate=""
  if [[ -z "$raw_path" ]]; then
    return 1
  fi
  builtin cd -L -- "$raw_path" >/dev/null 2>&1 || return 1
  candidate="$(pwd -L)"
  normalize_quest_path "$candidate"
}

is_valid_repo_root() {
  local candidate="$1"
  [[ -n "$candidate" && -f "$candidate/main.py" && -d "$candidate/health_benchmark" ]]
}

resolve_required_env() {
  local preferred_name="$1"
  local alias_name="$2"
  local default_value="$3"
  local label="$4"
  local value="${!preferred_name-}"
  if [[ -n "$value" ]]; then
    printf '%s\n' "$value"
    return 0
  fi
  value="${!alias_name-}"
  if [[ -n "$value" ]]; then
    printf '%s\n' "$value"
    return 0
  fi
  if [[ -n "$default_value" ]]; then
    printf '%s\n' "$default_value"
    return 0
  fi
  echo "Missing required ${label}. Export ${preferred_name}=... (preferred) or ${alias_name}=... before running this script." >&2
  exit 1
}

resolve_optional_env() {
  local name="$1"
  local default_value="$2"
  local value="${!name-}"
  if [[ -n "$value" ]]; then
    printf '%s\n' "$value"
    return 0
  fi
  printf '%s\n' "$default_value"
}

activate_runtime_env() {
  local hook_file="$MAMBA_ROOT/etc/profile.d/mamba.sh"
  local conda_bin="$MAMBA_ROOT/bin/conda"
  local python_path=""

  if [[ ! -d "$MAMBA_ROOT" ]]; then
    echo "Resolved mamba root does not exist: $MAMBA_ROOT" >&2
    exit 2
  fi
  if [[ ! -x "$conda_bin" ]]; then
    echo "Could not find conda hook binary at: $conda_bin" >&2
    exit 2
  fi
  if [[ ! -f "$hook_file" ]]; then
    echo "Could not find mamba activation script at: $hook_file" >&2
    exit 2
  fi
  if [[ ! -d "$ENV_PREFIX" ]]; then
    echo "Resolved Quest env prefix does not exist: $ENV_PREFIX" >&2
    exit 2
  fi

  export PATH="$MAMBA_ROOT/bin:$PATH"
  eval "$("$conda_bin" shell.bash hook 2>/dev/null)" || {
    echo "Failed to initialize the mamba shell hook from: $conda_bin" >&2
    exit 2
  }
  # shellcheck disable=SC1090
  source "$hook_file"
  mamba activate "$ENV_PREFIX" >/dev/null 2>&1 || {
    echo "Failed to activate Quest env prefix: $ENV_PREFIX" >&2
    exit 2
  }

  python_path="$(which python 2>/dev/null)" || {
    echo "python is not available after activating Quest env: $ENV_PREFIX" >&2
    exit 2
  }

  echo "Using mamba root: $MAMBA_ROOT"
  echo "Using env prefix: $ENV_PREFIX"
  echo "Using python: $python_path"
}

validate_vllm_runtime() {
  local gpu_info=""
  local gpu_count="0"
  local gpu_models=""
  local gpu_summary=""
  local host_name="unknown-host"
  local node_label=""

  if [[ ! -x "$SINGULARITY_BIN" ]]; then
    echo "Resolved Singularity binary does not exist or is not executable: $SINGULARITY_BIN" >&2
    exit 2
  fi
  if [[ ! -f "$VLLM_IMAGE" ]]; then
    echo "Resolved vLLM image does not exist: $VLLM_IMAGE" >&2
    exit 2
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is not available on PATH; cannot verify Quest GPU visibility." >&2
    exit 2
  fi
  if ! gpu_info="$(nvidia-smi -L 2>&1)"; then
    echo "Failed to verify Quest GPU visibility with nvidia-smi -L:" >&2
    printf '%s\n' "$gpu_info" >&2
    exit 2
  fi
  gpu_count="$(printf '%s\n' "$gpu_info" | awk '/^GPU [0-9]+:/ {count++} END {print count+0}')"
  gpu_models="$(printf '%s\n' "$gpu_info" | sed -nE 's/^GPU [0-9]+: ([^(]+) \(UUID:.*$/\1/p')"
  if [[ -z "$gpu_models" ]]; then
    echo "Could not parse GPU model information from nvidia-smi -L output:" >&2
    printf '%s\n' "$gpu_info" >&2
    exit 2
  fi
  gpu_summary="$(printf '%s\n' "$gpu_models" | sed 's/[[:space:]]*$//' | sort -u | paste -sd ',' - | sed 's/,/, /g')"
  if [[ "$gpu_count" -lt "$TENSOR_PARALLEL_SIZE" ]]; then
    echo "Visible GPU count ($gpu_count) is less than tensor parallel size ($TENSOR_PARALLEL_SIZE)." >&2
    printf '%s\n' "$gpu_info" >&2
    exit 2
  fi
  if host_name="$(hostname -s 2>/dev/null)"; then
    :
  elif host_name="$(hostname 2>/dev/null)"; then
    :
  fi
  node_label="${SLURMD_NODENAME:-${SLURM_JOB_NODELIST:-$host_name}}"

  echo "Using singularity: $SINGULARITY_BIN"
  echo "Using vLLM image: $VLLM_IMAGE"
  echo "Using node: $node_label"
  echo "Detected GPU count: $gpu_count"
  echo "Detected GPU model summary: $gpu_summary"
  printf '%s\n' "$gpu_info"
}

resolve_repo_root() {
  local script_candidate=""
  local candidate=""

  if [[ -n "${PROJECT_ROOT:-}" ]]; then
    if ! candidate="$(canonicalize_dir "$PROJECT_ROOT")"; then
      echo "PROJECT_ROOT does not exist or is not accessible: ${PROJECT_ROOT}" >&2
      exit 2
    fi
    if is_valid_repo_root "$candidate"; then
      printf '%s\n' "$candidate"
      return 0
    fi
    echo "PROJECT_ROOT is not a valid repo root: $candidate. Expected main.py and health_benchmark/." >&2
    exit 2
  fi

  if script_candidate="$(canonicalize_dir "$(dirname "${BASH_SOURCE[0]}")/..")" && is_valid_repo_root "$script_candidate"; then
    printf '%s\n' "$script_candidate"
    return 0
  fi

  if candidate="$(canonicalize_dir "$DEFAULT_PROJECT_ROOT")" && is_valid_repo_root "$candidate"; then
    printf '%s\n' "$candidate"
    return 0
  fi

  echo "Could not resolve repo root. Checked PROJECT_ROOT, the script location, and the baked-in default ${DEFAULT_PROJECT_ROOT}. Run this script from the repo checkout or set PROJECT_ROOT=/path/to/health-benchmark." >&2
  exit 2
}

REPO_ROOT="$(resolve_repo_root)"
MIMICIV_DIR="$(resolve_required_env MIMICIV_DIR MIMIC_IV_DIR "$DEFAULT_MIMICIV_DIR" "MIMIC-IV path")"
MIMICIV_NOTE_DIR="$(resolve_required_env MIMICIV_NOTE_DIR MIMIC_IV_NOTE_DIR "$DEFAULT_MIMICIV_NOTE_DIR" "MIMIC-IV-Note path")"
OUTPUT_ROOT="$(resolve_required_env OUTPUT_ROOT MEDBENCH_OUTPUT_ROOT "$DEFAULT_OUTPUT_ROOT" "output root")"
HF_HOME="$(resolve_optional_env HF_HOME "$DEFAULT_HF_HOME")"
HUGGINGFACE_HUB_CACHE="$(resolve_optional_env HUGGINGFACE_HUB_CACHE "$HF_HOME")"
MAMBA_ROOT="$(resolve_optional_env MAMBA_ROOT "$DEFAULT_MAMBA_ROOT")"
ENV_PREFIX="$(resolve_optional_env ENV_PREFIX "$DEFAULT_ENV_PREFIX")"
SINGULARITY_BIN="$(resolve_optional_env SINGULARITY_BIN "$DEFAULT_SINGULARITY_BIN")"
VLLM_IMAGE="$(normalize_quest_path "$(resolve_optional_env VLLM_IMAGE "$DEFAULT_VLLM_IMAGE")")"

if [ "$#" -eq 0 ]; then
  echo "Pass at least one subject_id, for example: quest/debug_qwen_interactive.sh 11826927" >&2
  exit 2
fi

echo "Resolved repo root: $REPO_ROOT"
echo "Using MIMIC-IV dir: $MIMICIV_DIR"
echo "Using MIMIC-IV-Note dir: $MIMICIV_NOTE_DIR"
echo "Using output root: $OUTPUT_ROOT"
echo "Using HF_HOME: $HF_HOME"
echo "Using HUGGINGFACE_HUB_CACHE: $HUGGINGFACE_HUB_CACHE"
activate_runtime_env
validate_vllm_runtime
cd "$REPO_ROOT"
mkdir -p "$OUTPUT_ROOT"
export VLLM_ENGINE_READY_TIMEOUT_S
export HF_HOME
export HUGGINGFACE_HUB_CACHE
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE"

cleanup() {
  if [[ -n "${VLLM_PID:-}" ]]; then
    kill "${VLLM_PID}" >/dev/null 2>&1 || true
    wait "${VLLM_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

VLLM_LOG="${VLLM_LOG:-$OUTPUT_ROOT/vllm_${SLURM_JOB_ID:-interactive_$$}.log}"

"$SINGULARITY_BIN" exec --nv -B /projects:/projects "$VLLM_IMAGE" \
  vllm serve "$MODEL" \
  --host 127.0.0.1 \
  --port "$VLLM_PORT" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --max-model-len "$MAX_MODEL_LEN" \
  --reasoning-parser "$REASONING_PARSER" \
  --language-model-only \
  >"$VLLM_LOG" 2>&1 &
VLLM_PID=$!

for _attempt in $(seq 1 "$READY_CHECK_ATTEMPTS"); do
  if curl -sf "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null; then
    break
  fi
  if ! kill -0 "$VLLM_PID" >/dev/null 2>&1; then
    echo "vLLM exited before becoming ready. See $VLLM_LOG" >&2
    wait "$VLLM_PID" >/dev/null 2>&1 || true
    exit 1
  fi
  sleep "$READY_CHECK_SLEEP_SECONDS"
done

if ! curl -sf "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null; then
  echo "vLLM server did not become ready. See $VLLM_LOG" >&2
  exit 1
fi

python "$REPO_ROOT/main.py" generate-all \
  --provider vllm \
  --model "$MODEL" \
  --base-url "http://127.0.0.1:${VLLM_PORT}/v1" \
  --mimiciv-dir "$MIMICIV_DIR" \
  --mimiciv-note-dir "$MIMICIV_NOTE_DIR" \
  --output-root "$OUTPUT_ROOT" \
  --subject-ids "$@"
