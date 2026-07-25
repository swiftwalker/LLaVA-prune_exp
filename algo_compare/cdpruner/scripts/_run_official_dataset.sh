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
RUNNER_PYTHON="${PYTHON_BIN:-/data_ssd/liuyu/.conda/envs/llava-next/bin/python}"

args=(--method cdpruner --dataset "$DATASET" --python-bin "$RUNNER_PYTHON")
[[ -n "${MODEL_PATH:-}" ]] && args+=(--model-path "$MODEL_PATH")
[[ -n "${MODEL_NAME:-}" ]] && args+=(--model-name "$MODEL_NAME")
[[ -n "${MODEL_BASE:-}" ]] && args+=(--model-base "$MODEL_BASE")
[[ -n "${QUESTION_FILE:-}" ]] && args+=(--question-file "$QUESTION_FILE")
[[ -n "${IMAGE_FOLDER:-}" ]] && args+=(--image-folder "$IMAGE_FOLDER")
[[ -n "${OUTPUT_DIR:-}" ]] && args+=(--output-dir "$OUTPUT_DIR")
[[ -n "${OFFICIAL_REPO:-}" ]] && args+=(--official-repo "$OFFICIAL_REPO")
[[ -n "${CONV_MODE:-}" ]] && args+=(--conv-mode "$CONV_MODE")
[[ -n "${RETAIN_TOKEN:-}" ]] && args+=(--retain-token "$RETAIN_TOKEN")
[[ -n "${MAX_SAMPLES:-}" ]] && args+=(--max-samples "$MAX_SAMPLES")
[[ -n "${LLAVA_NEXT_COMPAT:-}" ]] && args+=(--llava-next-compat "$LLAVA_NEXT_COMPAT")

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
