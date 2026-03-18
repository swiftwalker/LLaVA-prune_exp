#!/bin/bash
# =============================================================================
# Pruning Experiment: Run Pruning Inference
#
# Usage:
#   bash scripts/run_prune.sh <strategy> <dataset> [max_samples] [--set key=val ...]
#
# Examples:
#   bash scripts/run_prune.sh attn_score mme 10
#   bash scripts/run_prune.sh pre_attn_score mme 10
#   bash scripts/run_prune.sh entropy pope
#   bash scripts/run_prune.sh random mme 10
#   bash scripts/run_prune.sh sparsevlm mme 10
#   bash scripts/run_prune.sh baseline mme 10
#   bash scripts/run_prune.sh attn_score mme 10 --set pruning.prune_layers=[5] --set pruning.prune_ratio=[0.7]
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LLAVA_ROOT="$(dirname "$PROJECT_DIR")"

cd "$LLAVA_ROOT"

STRATEGY="${1:?Usage: run_prune.sh <strategy|baseline> <dataset|all> [max_samples] [--set ...]}"
DATASET="${2:?Usage: run_prune.sh <strategy|baseline> <dataset|all> [max_samples] [--set ...]}"
shift 2

case "$STRATEGY" in
    baseline|attn_score|pre_attn_score|entropy|random|sparsevlm)
        ;;
    compare)
        echo "Strategy 'compare' has been removed. Please run strategies explicitly." >&2
        exit 1
        ;;
    *)
        echo "Unknown strategy: $STRATEGY" >&2
        echo "Supported strategies: baseline, attn_score, pre_attn_score, entropy, random, sparsevlm" >&2
        exit 1
        ;;
esac

# Parse optional max_samples (first positional arg that is a plain integer)
MAX_SAMPLES=""
EXTRA_SETS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --set)
            EXTRA_SETS+=("--set" "$2")
            shift 2
            ;;
        *)
            if [[ -z "$MAX_SAMPLES" && "$1" =~ ^[0-9]+$ ]]; then
                MAX_SAMPLES="$1"
            else
                echo "Unknown argument: $1" >&2
                exit 1
            fi
            shift
            ;;
    esac
done

CONFIG="entropy_exp/configs/prune.yaml"

run_single() {
    local strategy="$1"
    local ds="$2"
    local flags=()

    if [ -n "$MAX_SAMPLES" ]; then
        flags+=(--max-samples "$MAX_SAMPLES")
    fi

    if [ "$strategy" = "baseline" ]; then
        echo "========================================"
        echo " Baseline (no pruning):  $ds"
        echo "========================================"
        python entropy_exp/src/prune_inference.py \
            --config "$CONFIG" \
            --dataset "$ds" \
            --baseline \
            "${flags[@]}" \
            "${EXTRA_SETS[@]}"
    else
        echo "========================================"
        echo " Strategy: $strategy  |  Dataset: $ds"
        echo "========================================"
        python entropy_exp/src/prune_inference.py \
            --config "$CONFIG" \
            --dataset "$ds" \
            --set "pruning.strategy=$strategy" \
            "${flags[@]}" \
            "${EXTRA_SETS[@]}"
    fi
}

run_dataset() {
    local strategy="$1"
    local ds="$2"

    if [ "$ds" = "all" ]; then
        for d in gqa mme pope; do
            run_single "$strategy" "$d"
        done
    else
        run_single "$strategy" "$ds"
    fi
}

run_dataset "$STRATEGY" "$DATASET"

echo ""
echo "========================================"
echo " All runs completed."
echo "========================================"
