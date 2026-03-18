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

activate_llava_env() {
  if [[ "${CONDA_DEFAULT_ENV:-}" == "llava" ]]; then
    return
  fi
  set +u
  if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate llava
    set -u
    return
  fi
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate llava
    set -u
    return
  fi
  set -u
  echo "Warning: could not activate the 'llava' conda environment; continuing with the current Python." >&2
}

activate_llava_env
set -u

TEXT_ARGS=(--dataset "$DATASET")
if [[ -n "$SAMPLE_LIMIT" ]]; then
  TEXT_ARGS+=(--sample-limit "$SAMPLE_LIMIT")
fi

RUN_DIR="$(bash "$ROOT_DIR/flops_exp/scripts/run_text_stats.sh" "${TEXT_ARGS[@]}" | tail -n 1)"
bash "$ROOT_DIR/flops_exp/scripts/run_flops.sh" --run-dir "$RUN_DIR_INPUT" --stats-json "$RUN_DIR/summary.json"
