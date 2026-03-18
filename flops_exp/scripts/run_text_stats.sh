#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/flops_exp/src:${PYTHONPATH:-}"

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

python "$ROOT_DIR/flops_exp/src/interface.py" --config "$ROOT_DIR/flops_exp/configs/default.yaml" text-stats "$@"
