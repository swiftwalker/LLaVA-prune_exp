#!/bin/bash
# =============================================================================
# Entropy Experiment: Evaluate Answers
#
# Usage:
#   bash scripts/run_eval.sh gqa entropy_exp/outputs/answers/gqa_*.jsonl
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LLAVA_ROOT="$(dirname "$PROJECT_DIR")"

cd "$LLAVA_ROOT"

DATASET="${1:?Usage: run_eval.sh <dataset> <answers_file>}"
ANSWERS_FILE="${2:?Usage: run_eval.sh <dataset> <answers_file>}"

echo "========================================"
echo " Evaluating: $DATASET"
echo " Answers:    $ANSWERS_FILE"
echo "========================================"

python entropy_exp/src/eval_datasets.py \
    --dataset "$DATASET" \
    --answers-file "$ANSWERS_FILE"
