#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METHOD_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$METHOD_DIR/../.." && pwd)"
OFFICIAL_REPO="${OFFICIAL_REPO:-$METHOD_DIR/third_party/SparseVLMs}"
PYTHON_BIN="${PYTHON_BIN:-/home/liuyu/miniconda3/envs/llava/bin/python}"
[[ -x "$PYTHON_BIN" ]] || PYTHON_BIN="python3"

MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/entropy_exp/models/llava-v1.5-7b}"
MODEL_NAME="${MODEL_NAME:-$(basename "$MODEL_PATH")}"
QUESTION_FILE="${QUESTION_FILE:-$REPO_ROOT/entropy_exp/eval_questions/MME/llava_mme.jsonl}"
IMAGE_FOLDER="${IMAGE_FOLDER:-$REPO_ROOT/entropy_exp/datasets/MME_Benchmark_release_version}"
USE_VERSION="${USE_VERSION:-2_0}"
RETAIN_TOKN="${RETAIN_TOKN:-192}"
CONV_MODE="${CONV_MODE:-vicuna_v1}"
TEMPERATURE="${TEMPERATURE:-0}"
RUN_LABEL="${RUN_LABEL:-mme_${MODEL_NAME}_v${USE_VERSION}_retain${RETAIN_TOKN}_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$METHOD_DIR/outputs/$RUN_LABEL}"
ANSWERS_FILE="$OUTPUT_DIR/answers.jsonl"
DRY_RUN=0
EVAL_AFTER="${EVAL_AFTER:-1}"

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

cmd=(
  "$PYTHON_BIN" -m llava.eval.model_vqa_loader
  --model-path "$MODEL_PATH"
  --question-file "$QUESTION_FILE"
  --image-folder "$IMAGE_FOLDER"
  --answers-file "$ANSWERS_FILE"
  --temperature "$TEMPERATURE"
  --conv-mode "$CONV_MODE"
)

echo "Official repo: $OFFICIAL_REPO"
echo "USE_VERSION=$USE_VERSION RETAIN_TOKN=$RETAIN_TOKN"
echo "Output dir: $OUTPUT_DIR"
printf 'Command:'
printf ' %q' "${cmd[@]}"
printf '\n'

if [[ "$DRY_RUN" -eq 1 ]]; then
  exit 0
fi

if [[ ! -d "$OFFICIAL_REPO/.git" ]]; then
  echo "Official repo is missing. Run fetch_official.sh first." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
(
  cd "$OFFICIAL_REPO"
  export USE_VERSION RETAIN_TOKN
  "${cmd[@]}"
)

if [[ "$EVAL_AFTER" == "1" ]]; then
  "$PYTHON_BIN" "$REPO_ROOT/entropy_exp/src/eval_datasets.py" \
    --dataset mme \
    --answers-file "$ANSWERS_FILE" \
    --output-dir "$OUTPUT_DIR/eval" \
    --mme-data-path "$IMAGE_FOLDER"
fi

python3 "$METHOD_DIR/scripts/collect_results.py" --run-dir "$OUTPUT_DIR" --dataset mme
