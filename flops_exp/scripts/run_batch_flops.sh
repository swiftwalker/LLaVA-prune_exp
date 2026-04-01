#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/flops_exp/src:${PYTHONPATH:-}"
source "$ROOT_DIR/entropy_exp/scripts/run_dir_common.sh"

activate_llava_env
ensure_python_bin

"$PYTHON_BIN" "$ROOT_DIR/flops_exp/src/interface.py" --config "$ROOT_DIR/flops_exp/configs/default.yaml" batch-flops "$@"
