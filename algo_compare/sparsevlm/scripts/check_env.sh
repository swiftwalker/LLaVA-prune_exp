#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METHOD_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$METHOD_DIR/../.." && pwd)"
OFFICIAL_REPO="${OFFICIAL_REPO:-$METHOD_DIR/third_party/SparseVLMs}"
MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/entropy_exp/models/llava-v1.5-7b}"
MME_DATA_PATH="${MME_DATA_PATH:-$REPO_ROOT/entropy_exp/datasets/MME_Benchmark_release_version}"
GQA_DATA_PATH="${GQA_DATA_PATH:-$REPO_ROOT/entropy_exp/datasets/gqa/images}"
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
check_path "MME data" "$MME_DATA_PATH"
check_path "GQA images" "$GQA_DATA_PATH"

if [[ -x "$PYTHON_BIN" ]]; then
  echo "OK      python: $PYTHON_BIN"
else
  if command -v python3 >/dev/null 2>&1; then
    echo "WARN    python: $PYTHON_BIN missing, python3 is available at $(command -v python3)"
  else
    echo "MISSING python: $PYTHON_BIN and python3"
    status=1
  fi
fi

if [[ "$status" -eq 0 ]]; then
  echo "Environment check passed."
else
  echo "Environment check found missing paths. This command is read-only."
fi

exit 0
