#!/bin/bash
# =============================================================================
# Entropy Experiment: Full Capture Pipeline
#
# Usage:
#   bash scripts/run_capture.sh [dataset] [max_samples]
#
# Examples:
#   bash scripts/run_capture.sh mme 10        # quick smoke test
#   bash scripts/run_capture.sh gqa 100       # initial analysis
#   bash scripts/run_capture.sh pope           # full dataset
#   bash scripts/run_capture.sh all            # all three datasets
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LLAVA_ROOT="$(dirname "$PROJECT_DIR")"

cd "$LLAVA_ROOT"

export CUDA_VISIBLE_DEVICES=0

DATASET="${1:-mme}"
MAX_SAMPLES="${2:-}"

CONFIG="entropy_exp/configs/default.yaml"

run_dataset() {
    local ds="$1"
    local max_flag=""
    if [ -n "$MAX_SAMPLES" ]; then
        max_flag="--max-samples $MAX_SAMPLES"
    fi

    echo "========================================"
    echo " Dataset: $ds"
    echo " Config:  $CONFIG"
    echo " Max samples: ${MAX_SAMPLES:-all}"
    echo "========================================"

    python entropy_exp/src/inference.py \
        --config "$CONFIG" \
        --dataset "$ds" \
        $max_flag
}

if [ "$DATASET" = "all" ]; then
    for ds in gqa mme pope; do
        run_dataset "$ds"
    done
else
    run_dataset "$DATASET"
fi

echo ""
echo "Capture complete. Raw data in: entropy_exp/outputs/raw/"
echo "Answers in: entropy_exp/outputs/answers/"
