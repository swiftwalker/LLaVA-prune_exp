#!/bin/bash
# =============================================================================
# Entropy Experiment: Evaluate Answers
#
# Usage:
#   bash scripts/run_eval.sh runs
#   bash scripts/run_eval.sh mme_attn_score_20260310_153649
#   bash scripts/run_eval.sh entropy_exp/outputs/runs/.../answers.jsonl
#   bash scripts/run_eval.sh gqa entropy_exp/outputs/answers/gqa_*.jsonl
#   bash scripts/run_eval.sh gqa entropy_exp/outputs/answers/gqa_*.jsonl entropy_exp/outputs/eval/custom_dir
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LLAVA_ROOT="$(dirname "$PROJECT_DIR")"

cd "$LLAVA_ROOT"

ARG1="${1:-}"
ARG2="${2:-}"
ARG3="${3:-}"
RUNS_DIR="entropy_exp/outputs/runs"
EVAL_DIR="entropy_exp/outputs/eval"
DEFAULT_LLAVA_PYTHON="/home/liuyu/miniconda3/envs/llava/bin/python"

if [ -x "$DEFAULT_LLAVA_PYTHON" ]; then
    PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_LLAVA_PYTHON}"
else
    PYTHON_BIN="${PYTHON_BIN:-python}"
fi

run_eval_for_run() {
    local run_dir="$1"
    echo "========================================"
    echo " Evaluating run: $run_dir"
    echo " Output dir:     $run_dir/eval"
    echo "========================================"

    "$PYTHON_BIN" entropy_exp/src/eval_datasets.py \
        --run-dir "$run_dir"
}

run_eval_legacy() {
    local dataset="$1"
    local answers_file="$2"
    local output_dir="$3"

    echo "========================================"
    echo " Evaluating: $dataset"
    echo " Answers:    $answers_file"
    echo " Output dir: $output_dir"
    echo "========================================"

    "$PYTHON_BIN" entropy_exp/src/eval_datasets.py \
        --dataset "$dataset" \
        --answers-file "$answers_file" \
        --output-dir "$output_dir"
}

if [ $# -eq 0 ] || [ "$ARG1" = "runs" ]; then
    mapfile -t RUN_DIRS < <(find "$RUNS_DIR" -mindepth 1 -maxdepth 1 -type d -exec test -f "{}/answers.jsonl" ';' -print | sort)
elif [ -d "$ARG1" ] && [ -f "$ARG1/answers.jsonl" ]; then
    RUN_DIRS=("$ARG1")
elif [ -d "$RUNS_DIR/$ARG1" ] && [ -f "$RUNS_DIR/$ARG1/answers.jsonl" ]; then
    RUN_DIRS=("$RUNS_DIR/$ARG1")
elif [ -f "$ARG1" ] && [ "$(basename "$ARG1")" = "answers.jsonl" ]; then
    RUN_DIRS=("$(dirname "$ARG1")")
elif [ $# -eq 1 ]; then
    mapfile -t RUN_DIRS < <(find "$RUNS_DIR" -mindepth 1 -maxdepth 1 -type d -name "${ARG1}*" -exec test -f "{}/answers.jsonl" ';' -print | sort)
else
    DATASET="$ARG1"
    ANSWERS_FILE="$ARG2"
    OUTPUT_DIR="${ARG3:-$EVAL_DIR/$DATASET/$(basename "${ANSWERS_FILE%.jsonl}")}"
    run_eval_legacy "$DATASET" "$ANSWERS_FILE" "$OUTPUT_DIR"
    echo ""
    echo "Evaluation complete!"
    echo "  Output: $OUTPUT_DIR"
    exit 0
fi

if [ ${#RUN_DIRS[@]} -eq 0 ]; then
    echo "No run directories found for: ${ARG1:-runs}"
    echo "Expected answers at: <run_dir>/answers.jsonl"
    exit 1
fi

echo "Found ${#RUN_DIRS[@]} run(s):"
for run_dir in "${RUN_DIRS[@]}"; do
    echo "  $run_dir"
done

for run_dir in "${RUN_DIRS[@]}"; do
    echo ""
    run_eval_for_run "$run_dir"
done

echo ""
echo "Evaluation complete!"
echo "  Run outputs: <run_dir>/eval/"
