#!/usr/bin/env bash
set -eo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash flops_exp/scripts/run_all.sh <dataset> <run_dir> [sample_limit]" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATASET="$1"
RUN_DIR_INPUT="$2"
SAMPLE_LIMIT="${3:-}"
source "$ROOT_DIR/entropy_exp/scripts/run_dir_common.sh"

activate_llava_env
ensure_python_bin

TEXT_ARGS=(--dataset "$DATASET")
if [[ -n "$SAMPLE_LIMIT" ]]; then
  TEXT_ARGS+=(--sample-limit "$SAMPLE_LIMIT")
fi

RUN_DIR="$(bash "$ROOT_DIR/flops_exp/scripts/run_text_stats.sh" "${TEXT_ARGS[@]}" | tail -n 1)"
bash "$ROOT_DIR/flops_exp/scripts/run_flops.sh" --run-dir "$RUN_DIR_INPUT" --stats-json "$RUN_DIR/summary.json"
