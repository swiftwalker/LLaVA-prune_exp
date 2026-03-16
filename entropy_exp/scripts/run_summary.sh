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

resolve_run_dir() {
    local input_path="$1"

    if [ -d "$input_path" ] && [ -f "$input_path/config.yaml" ]; then
        echo "$input_path"
    elif [ -d "$RUNS_DIR/$input_path" ] && [ -f "$RUNS_DIR/$input_path/config.yaml" ]; then
        echo "$RUNS_DIR/$input_path"
    elif [ -f "$input_path" ] && [ "$(basename "$input_path")" = "answers.jsonl" ]; then
        echo "$(dirname "$input_path")"
    elif [ -f "$input_path" ] && [ "$(basename "$input_path")" = "summary.json" ]; then
        echo "$(dirname "$(dirname "$input_path")")"
    elif [ -d "$input_path" ] && [ -f "$input_path/summary.json" ]; then
        echo "$(dirname "$input_path")"
    else
        return 1
    fi
}

if [ $# -gt 2 ]; then
    echo "Usage: bash entropy_exp/scripts/run_summary.sh [runs|<run>|<prefix>|<answers.jsonl>|<eval/summary.json>] [output_dir]" >&2
    exit 1
fi

SELECTION_LABEL="runs"
OUTPUT_DIR=""
RUN_DIRS=()

if [ $# -eq 0 ] || [ "$ARG1" = "runs" ]; then
    OUTPUT_DIR="$ARG2"
    mapfile -t RUN_DIRS < <(find "$RUNS_DIR" -mindepth 1 -maxdepth 1 -type d -exec test -f "{}/config.yaml" ';' -print | sort)
elif resolved_run_dir="$(resolve_run_dir "$ARG1" 2>/dev/null)"; then
    SELECTION_LABEL="$(basename "$resolved_run_dir")"
    OUTPUT_DIR="$ARG2"
    RUN_DIRS=("$resolved_run_dir")
else
    SELECTION_LABEL="$ARG1"
    OUTPUT_DIR="$ARG2"
    mapfile -t RUN_DIRS < <(find "$RUNS_DIR" -mindepth 1 -maxdepth 1 -type d -name "${ARG1}*" -exec test -f "{}/config.yaml" ';' -print | sort)
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
