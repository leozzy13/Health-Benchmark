#!/usr/bin/env bash
#SBATCH --job-name=medbench-qwen-eval-27b
#SBATCH --partition=gengpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --constraint=sxm
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --output=%x-%j.out

set -euo pipefail

DEFAULT_PROJECT_ROOT="/projects/p33194/health-benchmark"
DEFAULT_OUTPUT_ROOT="/projects/p33194/medbench-output"
DEFAULT_HF_HOME="/projects/p33194/hf_cache"
DEFAULT_MAMBA_ROOT="/hpc/software/mamba/24.3.0"
DEFAULT_ENV_PREFIX="/projects/p33194/envs/medbench-qwen"
DEFAULT_SINGULARITY_BIN="/software/singularity/3.8.1/bin/singularity"
DEFAULT_VLLM_IMAGE="/projects/p33194/containers/vllm-openai_latest.sif"
EVAL_MODEL_PRESET="${EVAL_MODEL_PRESET:-27b_2gpu}"

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
  if [[ -z "$raw_path" ]]; then
    return 1
  fi
  builtin cd -L -- "$raw_path" >/dev/null 2>&1 || return 1
  normalize_quest_path "$(pwd -L)"
}

is_valid_repo_root() {
  local candidate="$1"
  [[ -n "$candidate" && -f "$candidate/main.py" && -d "$candidate/health_benchmark" ]]
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
  export PATH="$MAMBA_ROOT/bin:$PATH"
  eval "$("$conda_bin" shell.bash hook 2>/dev/null)"
  # shellcheck disable=SC1090
  source "$hook_file"
  mamba activate "$ENV_PREFIX" >/dev/null 2>&1
  python_path="$(which python)"
  echo "Using mamba root: $MAMBA_ROOT"
  echo "Using env prefix: $ENV_PREFIX"
  echo "Using python: $python_path"
}

validate_eval_runtime() {
  local gpu_info=""
  local local_gpu_count="0"
  local total_gpu_count="0"
  local gpu_models=""
  local gpu_summary=""
  local node_label="${SLURMD_NODENAME:-${SLURM_JOB_NODELIST:-unknown-node}}"
  local required_total_gpu_count=""

  if ! gpu_info="$(nvidia-smi -L 2>&1)"; then
    echo "Failed to verify GPU visibility with nvidia-smi -L:" >&2
    printf '%s\n' "$gpu_info" >&2
    exit 2
  fi
  local_gpu_count="$(printf '%s\n' "$gpu_info" | awk '/^GPU [0-9]+:/ {count++} END {print count+0}')"
  total_gpu_count="$(( local_gpu_count * ${SLURM_JOB_NUM_NODES:-1} ))"
  gpu_models="$(printf '%s\n' "$gpu_info" | sed -nE 's/^GPU [0-9]+: ([^(]+) \(UUID:.*$/\1/p')"
  gpu_summary="$(printf '%s\n' "$gpu_models" | sed 's/[[:space:]]*$//' | sort -u | paste -sd ',' - | sed 's/,/, /g')"
  case "$EVAL_MODEL_PRESET" in
    trio)
      required_total_gpu_count=8
      ;;
    small)
      required_total_gpu_count=1
      ;;
    27b_2gpu)
      required_total_gpu_count=2
      ;;
    *)
      echo "Unknown EVAL_MODEL_PRESET: $EVAL_MODEL_PRESET" >&2
      exit 2
      ;;
  esac

  if [[ "$total_gpu_count" -lt "$required_total_gpu_count" ]]; then
    echo "Total allocated GPU count ($total_gpu_count) is less than the required $required_total_gpu_count GPUs for the $EVAL_MODEL_PRESET evaluation profile." >&2
    exit 2
  fi

  echo "Using singularity: $SINGULARITY_BIN"
  echo "Using vLLM image: $VLLM_IMAGE"
  echo "Using node: $node_label"
  echo "Detected local GPU count: $local_gpu_count"
  echo "Detected total allocated GPU count: $total_gpu_count"
  echo "Detected GPU model summary: $gpu_summary"
  printf '%s\n' "$gpu_info"
}

resolve_repo_root() {
  local candidate=""
  if [[ -n "${PROJECT_ROOT:-}" ]]; then
    candidate="$(canonicalize_dir "$PROJECT_ROOT")" || {
      echo "PROJECT_ROOT does not exist or is not accessible: ${PROJECT_ROOT}" >&2
      exit 2
    }
    if is_valid_repo_root "$candidate"; then
      printf '%s\n' "$candidate"
      return 0
    fi
    echo "PROJECT_ROOT is not a valid repo root: $candidate. Expected main.py and health_benchmark/." >&2
    exit 2
  fi
  if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    if candidate="$(canonicalize_dir "$SLURM_SUBMIT_DIR")" && is_valid_repo_root "$candidate"; then
      printf '%s\n' "$candidate"
      return 0
    fi
  fi
  if candidate="$(canonicalize_dir "$DEFAULT_PROJECT_ROOT")" && is_valid_repo_root "$candidate"; then
    printf '%s\n' "$candidate"
    return 0
  fi
  echo "Could not resolve repo root. Checked PROJECT_ROOT, SLURM_SUBMIT_DIR, and ${DEFAULT_PROJECT_ROOT}." >&2
  exit 2
}

REPO_ROOT="$(resolve_repo_root)"
PROJECT_ROOT="$REPO_ROOT"
OUTPUT_ROOT="$(resolve_optional_env OUTPUT_ROOT "$DEFAULT_OUTPUT_ROOT")"
HF_HOME="$(resolve_optional_env HF_HOME "$DEFAULT_HF_HOME")"
HUGGINGFACE_HUB_CACHE="$(resolve_optional_env HUGGINGFACE_HUB_CACHE "$HF_HOME/hub")"
MAMBA_ROOT="$(resolve_optional_env MAMBA_ROOT "$DEFAULT_MAMBA_ROOT")"
ENV_PREFIX="$(resolve_optional_env ENV_PREFIX "$DEFAULT_ENV_PREFIX")"
SINGULARITY_BIN="$(resolve_optional_env SINGULARITY_BIN "$DEFAULT_SINGULARITY_BIN")"
VLLM_IMAGE="$(normalize_quest_path "$(resolve_optional_env VLLM_IMAGE "$DEFAULT_VLLM_IMAGE")")"
QUEST_JOB_OUTPUT_ROOT="$OUTPUT_ROOT/quest_job_outputs/${SLURM_JOB_ID:-manual}"
SERVER_READY_TIMEOUT_S="${SERVER_READY_TIMEOUT_S:-1800}"

if [[ "$#" -eq 0 ]]; then
  echo "Pass subject_ids or --patient-manifest /path/to/patients.txt to sbatch." >&2
  exit 2
fi

echo "Resolved repo root: $REPO_ROOT"
echo "Using output root: $OUTPUT_ROOT"
echo "Using HF_HOME: $HF_HOME"
echo "Using HUGGINGFACE_HUB_CACHE: $HUGGINGFACE_HUB_CACHE"
echo "Using evaluation model preset: $EVAL_MODEL_PRESET"

activate_runtime_env
validate_eval_runtime

mkdir -p "$OUTPUT_ROOT" "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$QUEST_JOB_OUTPUT_ROOT"
export PROJECT_ROOT REPO_ROOT OUTPUT_ROOT HF_HOME HUGGINGFACE_HUB_CACHE MAMBA_ROOT ENV_PREFIX SINGULARITY_BIN VLLM_IMAGE QUEST_JOB_OUTPUT_ROOT SERVER_READY_TIMEOUT_S EVAL_MODEL_PRESET

bash "$REPO_ROOT/quest/run_multi_patient_eval_job.sh" "$@"
