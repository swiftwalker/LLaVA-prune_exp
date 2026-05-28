#!/bin/bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <dataset> [run_official.py args...]" >&2
  exit 2
fi

DATASET="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METHOD_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$METHOD_DIR/../.." && pwd)"

RUNNER_PYTHON="${PYTHON_BIN:-/home/liuyu/miniconda3/envs/llava/bin/python}"
if [[ "$RUNNER_PYTHON" == */* ]]; then
  [[ -x "$RUNNER_PYTHON" ]] || RUNNER_PYTHON="python3"
elif ! command -v "$RUNNER_PYTHON" >/dev/null 2>&1; then
  RUNNER_PYTHON="python3"
fi

args=(
  --method sparsevlm
  --dataset "$DATASET"
  --python-bin "$RUNNER_PYTHON"
)

[[ -n "${MODEL_PATH:-}" ]] && args+=(--model-path "$MODEL_PATH")
[[ -n "${MODEL_NAME:-}" ]] && args+=(--model-name "$MODEL_NAME")
[[ -n "${MODEL_BASE:-}" ]] && args+=(--model-base "$MODEL_BASE")
[[ -n "${QUESTION_FILE:-}" ]] && args+=(--question-file "$QUESTION_FILE")
[[ -n "${IMAGE_FOLDER:-}" ]] && args+=(--image-folder "$IMAGE_FOLDER")
[[ -n "${OUTPUT_DIR:-}" ]] && args+=(--output-dir "$OUTPUT_DIR")
[[ -n "${OFFICIAL_REPO:-}" ]] && args+=(--official-repo "$OFFICIAL_REPO")
[[ -n "${USE_VERSION:-}" ]] && args+=(--use-version "$USE_VERSION")
[[ -n "${RETAIN_TOKN:-}" ]] && args+=(--retain-token "$RETAIN_TOKN")
[[ -n "${CONV_MODE:-}" ]] && args+=(--conv-mode "$CONV_MODE")
[[ -n "${TEMPERATURE:-}" ]] && args+=(--temperature "$TEMPERATURE")
[[ -n "${TOP_P:-}" ]] && args+=(--top-p "$TOP_P")
[[ -n "${NUM_BEAMS:-}" ]] && args+=(--num-beams "$NUM_BEAMS")
[[ -n "${MAX_NEW_TOKENS:-}" ]] && args+=(--max-new-tokens "$MAX_NEW_TOKENS")

has_eval_arg=0
for arg in "$@"; do
  if [[ "$arg" == "--eval" || "$arg" == "--no-eval" ]]; then
    has_eval_arg=1
  fi
done

if [[ "${EVAL_AFTER:-1}" == "1" && "$has_eval_arg" -eq 0 ]]; then
  args+=(--eval)
fi

exec "$RUNNER_PYTHON" "$REPO_ROOT/algo_compare/scripts/run_official.py" "${args[@]}" "$@"
