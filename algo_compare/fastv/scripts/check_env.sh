#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METHOD_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$METHOD_DIR/../.." && pwd)"
OFFICIAL_REPO="${OFFICIAL_REPO:-$METHOD_DIR/third_party/FastV}"
MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/entropy_exp/models/llava-v1.5-7b}"
PYTHON_BIN="${PYTHON_BIN:-/home/liuyu/miniconda3/envs/llava/bin/python}"
status=0

check_path() {
  local label="$1"
  local path="$2"
  if [[ -e "$path" ]]; then
    echo "OK      $label: $path"
  else
    echo "MISSING $label: $path"
    status=1
  fi
}

check_path "official repo" "$OFFICIAL_REPO/.git"
check_path "model path" "$MODEL_PATH"
check_path "GQA questions" "$REPO_ROOT/entropy_exp/eval_questions/gqa/llava_gqa_testdev_balanced.jsonl"
check_path "GQA images" "$REPO_ROOT/entropy_exp/datasets/gqa/images"
check_path "MME questions" "$REPO_ROOT/entropy_exp/eval_questions/MME/llava_mme.jsonl"
check_path "MME data" "$REPO_ROOT/entropy_exp/datasets/MME_Benchmark_release_version"
check_path "POPE questions" "$REPO_ROOT/entropy_exp/eval_questions/pope/llava_pope_test.jsonl"
check_path "POPE images" "$REPO_ROOT/entropy_exp/datasets/pope/val2014"
check_path "TextVQA questions" "$REPO_ROOT/entropy_exp/eval_questions/textvqa/llava_textvqa_val_v051_ocr.jsonl"
check_path "TextVQA images" "$REPO_ROOT/entropy_exp/datasets/textvqa/train_images"
check_path "ScienceQA questions" "$REPO_ROOT/entropy_exp/eval_questions/scienceqa/llava_test_CQM-A.json"
check_path "ScienceQA images" "$REPO_ROOT/entropy_exp/eval_questions/scienceqa/test"

if [[ -x "$PYTHON_BIN" ]]; then
  echo "OK      python: $PYTHON_BIN"
elif command -v python3 >/dev/null 2>&1; then
  echo "WARN    python: $PYTHON_BIN missing, python3 is available at $(command -v python3)"
else
  echo "MISSING python: $PYTHON_BIN and python3"
  status=1
fi

if [[ "$status" -eq 0 ]]; then
  echo "Environment check passed."
else
  echo "Environment check found missing paths. This command is read-only."
fi

exit 0
