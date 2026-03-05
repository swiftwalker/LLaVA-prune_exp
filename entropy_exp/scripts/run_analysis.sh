#!/bin/bash
# =============================================================================
# Entropy Experiment: Analysis Pipeline
#
# Usage:
#   bash scripts/run_analysis.sh               # analyze all HDF5 files
#   bash scripts/run_analysis.sh gqa            # analyze specific dataset
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LLAVA_ROOT="$(dirname "$PROJECT_DIR")"

cd "$LLAVA_ROOT"

DATASET="${1:-}"
RAW_DIR="entropy_exp/outputs/raw"
PROCESSED_DIR="entropy_exp/outputs/processed"
FIGURES_DIR="entropy_exp/outputs/figures"

# Step 1: Compute entropy metrics from HDF5 files
echo "========================================"
echo " Step 1: Computing entropy metrics"
echo "========================================"

if [ -n "$DATASET" ]; then
    H5_PATTERN="${RAW_DIR}/${DATASET}_*.h5"
else
    H5_PATTERN="${RAW_DIR}/*.h5"
fi

# Check if any files match
shopt -s nullglob
H5_FILES=($H5_PATTERN)
shopt -u nullglob

if [ ${#H5_FILES[@]} -eq 0 ]; then
    echo "No HDF5 files found matching: $H5_PATTERN"
    echo "Run capture first: bash scripts/run_capture.sh"
    exit 1
fi

echo "Found ${#H5_FILES[@]} HDF5 file(s)"

python entropy_exp/analysis/entropy_analysis.py \
    --h5 ${H5_FILES[@]} \
    --output "$PROCESSED_DIR"

# Step 2: Generate visualizations
echo ""
echo "========================================"
echo " Step 2: Generating visualizations"
echo "========================================"

CSV_FILE="${PROCESSED_DIR}/entropy_per_sample_layer.csv"
if [ ! -f "$CSV_FILE" ]; then
    echo "Error: CSV file not found: $CSV_FILE"
    exit 1
fi

python entropy_exp/analysis/visualize.py \
    --csv "$CSV_FILE" \
    --output "$FIGURES_DIR"

echo ""
echo "Analysis complete!"
echo "  Metrics: $PROCESSED_DIR"
echo "  Figures: $FIGURES_DIR"
