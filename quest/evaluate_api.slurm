#!/usr/bin/env bash
#SBATCH --job-name=medbench-api-eval
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=quest/slurm_logs/%x-%j.out

set -euo pipefail

DEFAULT_PROJECT_ROOT="/projects/p33194/health-benchmark"
DEFAULT_OUTPUT_ROOT="/projects/p33194/health-benchmark/output/benchmark"
DEFAULT_EVALUATION_ROOT="/projects/p33194/health-benchmark/output/evaluation"
DEFAULT_ENV_PREFIX="/projects/p33194/health-benchmark/runtime/envs/medbench"

PROJECT_ROOT="${PROJECT_ROOT:-$DEFAULT_PROJECT_ROOT}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$DEFAULT_OUTPUT_ROOT}"
EVALUATION_ROOT="${EVALUATION_ROOT:-$DEFAULT_EVALUATION_ROOT}"
ENV_PREFIX="${ENV_PREFIX:-$DEFAULT_ENV_PREFIX}"
PYTHON="${PYTHON:-$ENV_PREFIX/bin/python}"
API_KEY_ENV="${API_KEY_ENV:-OPENAI_API_KEY}"

EVAL_ARGS=()
HAS_PROVIDER=0
HAS_API_KEY_ENV=0

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --provider)
      HAS_PROVIDER=1
      if [[ -z "${2:-}" || "${2:-}" == --* ]]; then
        echo "--provider requires an argument" >&2
        exit 2
      fi
      EVAL_ARGS+=("$1" "$2")
      shift 2
      ;;
    --api-key-env)
      HAS_API_KEY_ENV=1
      if [[ -z "${2:-}" || "${2:-}" == --* ]]; then
        echo "--api-key-env requires an environment variable name" >&2
        exit 2
      fi
      API_KEY_ENV="$2"
      EVAL_ARGS+=("$1" "$2")
      shift 2
      ;;
    *)
      EVAL_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ "$HAS_PROVIDER" -eq 0 ]]; then
  EVAL_ARGS=(--provider openai "${EVAL_ARGS[@]}")
fi
if [[ "$HAS_API_KEY_ENV" -eq 0 ]]; then
  EVAL_ARGS=(--api-key-env "$API_KEY_ENV" "${EVAL_ARGS[@]}")
fi

if [[ -z "${!API_KEY_ENV-}" ]]; then
  echo "Missing API key in environment variable ${API_KEY_ENV}." >&2
  exit 2
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "Python executable not found or not executable: $PYTHON" >&2
  exit 2
fi

mkdir -p "$PROJECT_ROOT/quest/slurm_logs"
cd "$PROJECT_ROOT"

"$PYTHON" - <<'PY'
import importlib.util
if importlib.util.find_spec("openai") is None:
    raise SystemExit("The openai Python package is not importable in this environment.")
print("OpenAI Python package import OK.")
PY

echo "Running API-only evaluation with Python: $PYTHON"
echo "Using benchmark root: $OUTPUT_ROOT"
echo "Using evaluation root: $EVALUATION_ROOT"
echo "Using API key env var: $API_KEY_ENV"

exec "$PYTHON" "$PROJECT_ROOT/main.py" evaluate \
  --benchmark-root "$OUTPUT_ROOT" \
  --evaluation-root "$EVALUATION_ROOT" \
  "${EVAL_ARGS[@]}"
