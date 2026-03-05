#!/bin/bash
# =============================================================================
# Entropy Experiment: Analysis Pipeline
#
# Usage:
#   bash scripts/run_analysis.sh                           # analyze all HDF5 files
#   bash scripts/run_analysis.sh gqa                       # analyze by dataset name
#   bash scripts/run_analysis.sh mme_20260305_140915.h5    # analyze specific file
#   bash scripts/run_analysis.sh /absolute/path/to/file.h5 # analyze by absolute path
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LLAVA_ROOT="$(dirname "$PROJECT_DIR")"

cd "$LLAVA_ROOT"

ARG="${1:-}"
RAW_DIR="entropy_exp/outputs/raw"
PROCESSED_DIR="entropy_exp/outputs/processed"
FIGURES_DIR="entropy_exp/outputs/figures"

# Step 1: Compute entropy metrics from HDF5 files
echo "========================================"
echo " Step 1: Computing entropy metrics"
echo "========================================"

if [ -z "$ARG" ]; then
    # No argument: analyze all
    H5_PATTERN="${RAW_DIR}/*.h5"
    shopt -s nullglob
    H5_FILES=($H5_PATTERN)
    shopt -u nullglob
elif [ -f "$ARG" ]; then
    # Argument is an existing file (absolute or relative path)
    H5_FILES=("$ARG")
elif [ -f "${RAW_DIR}/${ARG}" ]; then
    # Argument is a filename inside raw/
    H5_FILES=("${RAW_DIR}/${ARG}")
else
    # Argument is a dataset name prefix
    H5_PATTERN="${RAW_DIR}/${ARG}_*.h5"
    shopt -s nullglob
    H5_FILES=($H5_PATTERN)
    shopt -u nullglob
fi

if [ ${#H5_FILES[@]} -eq 0 ]; then
    echo "No HDF5 files found for: ${ARG:-all}"
    echo "Run capture first: bash scripts/run_capture.sh"
    exit 1
fi

echo "Found ${#H5_FILES[@]} HDF5 file(s):"
for f in "${H5_FILES[@]}"; do echo "  $f"; done

python entropy_exp/analysis/entropy_analysis.py \
    --h5 ${H5_FILES[@]} \
    --output "$PROCESSED_DIR"

echo ""
echo "Analysis complete!"
echo "  Metrics: $PROCESSED_DIR"
