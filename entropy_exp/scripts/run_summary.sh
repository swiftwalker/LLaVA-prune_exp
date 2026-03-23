#!/bin/bash
# =============================================================================
# Entropy Experiment: Summarize Evaluated Runs
#
# Usage:
#   bash scripts/run_summary.sh runs
#   bash scripts/run_summary.sh mme
#   bash scripts/run_summary.sh mme_attn_score_20260315_171500
#   bash scripts/run_summary.sh entropy_exp/outputs/runs/.../answers.jsonl
#   bash scripts/run_summary.sh entropy_exp/outputs/runs/.../eval/summary.json
#   bash scripts/run_summary.sh mme entropy_exp/outputs/summary/custom_dir
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LLAVA_ROOT="$(dirname "$PROJECT_DIR")"

cd "$LLAVA_ROOT"

ARG1="${1:-}"
ARG2="${2:-}"
RUNS_DIR="entropy_exp/outputs/runs"

if [ -n "${PYTHON_BIN:-}" ]; then
    :
elif [ -x "/data_ssd/liuyu/miniconda3/envs/llava/bin/python" ]; then
    PYTHON_BIN="/data_ssd/liuyu/miniconda3/envs/llava/bin/python"
elif [ -x "/home/liuyu/miniconda3/envs/llava/bin/python" ]; then
    PYTHON_BIN="/home/liuyu/miniconda3/envs/llava/bin/python"
else
    PYTHON_BIN="$(command -v python)"
fi

source "$SCRIPT_DIR/run_dir_common.sh"

if [ $# -gt 2 ]; then
    echo "Usage: bash entropy_exp/scripts/run_summary.sh [runs|<run>|<prefix>|<answers.jsonl>|<eval/summary.json>] [output_dir]" >&2
    exit 1
fi

SELECTION_LABEL="runs"
OUTPUT_DIR=""
RUN_DIRS=()

if [ $# -eq 0 ] || [ "$ARG1" = "runs" ]; then
    OUTPUT_DIR="$ARG2"
    mapfile -t RUN_DIRS < <(run_dirs_list_all "config.yaml")
elif resolved_run_dir="$(run_dirs_resolve_summary_input "$ARG1" 2>/dev/null)"; then
    SELECTION_LABEL="$(basename "$resolved_run_dir")"
    OUTPUT_DIR="$ARG2"
    RUN_DIRS=("$resolved_run_dir")
else
    SELECTION_LABEL="$ARG1"
    OUTPUT_DIR="$ARG2"
    mapfile -t RUN_DIRS < <(run_dirs_list_prefix "$ARG1" "config.yaml")
fi

if [ ${#RUN_DIRS[@]} -eq 0 ]; then
    echo "No run directories found for: ${ARG1:-runs}" >&2
    exit 1
fi

echo "Found ${#RUN_DIRS[@]} run(s):"
for run_dir in "${RUN_DIRS[@]}"; do
    echo "  $run_dir"
done

CMD=(
    "$PYTHON_BIN"
    entropy_exp/src/summarize_results.py
    --selection-label "$SELECTION_LABEL"
    --run-dir
)
CMD+=("${RUN_DIRS[@]}")

if [ -n "$OUTPUT_DIR" ]; then
    CMD+=(--output-dir "$OUTPUT_DIR")
fi

echo ""
"${CMD[@]}"
