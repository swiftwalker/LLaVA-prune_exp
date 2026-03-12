#!/bin/bash
# =============================================================================
# Entropy Experiment: Analysis Pipeline
#
# Usage:
#   bash scripts/run_analysis.sh                            # analyze all HDF5 files in outputs/raw
#   bash scripts/run_analysis.sh gqa                        # analyze raw files by dataset name
#   bash scripts/run_analysis.sh runs                       # analyze all captures.h5 in outputs/runs/*/
#   bash scripts/run_analysis.sh mme_attn_score_20260310_153649
#                                                          # analyze one run dir by name
#   bash scripts/run_analysis.sh entropy_exp/outputs/runs/.../captures.h5
#                                                          # analyze a specific run capture file
#   bash scripts/run_analysis.sh /absolute/path/to/file.h5 # analyze any absolute/relative HDF5 path
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LLAVA_ROOT="$(dirname "$PROJECT_DIR")"

cd "$LLAVA_ROOT"

ARG="${1:-}"
RAW_DIR="entropy_exp/outputs/raw"
PROCESSED_DIR="entropy_exp/outputs/processed"
RUNS_DIR="entropy_exp/outputs/runs"

# Step 1: Compute entropy metrics from HDF5 files
echo "========================================"
echo " Step 1: Computing entropy metrics"
echo "========================================"

if [ -z "$ARG" ]; then
    # No argument: analyze all raw HDF5 files
    H5_PATTERN="${RAW_DIR}/*.h5"
    shopt -s nullglob
    H5_FILES=($H5_PATTERN)
    shopt -u nullglob
elif [ "$ARG" = "runs" ]; then
    # Analyze all captures.h5 files under outputs/runs/*/
    mapfile -t H5_FILES < <(find "$RUNS_DIR" -mindepth 2 -maxdepth 2 -name "captures.h5" | sort)
elif [ -d "$ARG" ] && [ -f "$ARG/captures.h5" ]; then
    # Argument is a run directory path
    H5_FILES=("$ARG/captures.h5")
elif [ -f "$ARG" ]; then
    # Argument is an existing file (absolute or relative path)
    H5_FILES=("$ARG")
elif [ -d "${RUNS_DIR}/${ARG}" ] && [ -f "${RUNS_DIR}/${ARG}/captures.h5" ]; then
    # Argument is a run directory name inside outputs/runs/
    H5_FILES=("${RUNS_DIR}/${ARG}/captures.h5")
elif [ -f "${RAW_DIR}/${ARG}" ]; then
    # Argument is a filename inside raw/
    H5_FILES=("${RAW_DIR}/${ARG}")
else
    # Prefer raw dataset prefix; fall back to run directory prefix
    H5_PATTERN="${RAW_DIR}/${ARG}_*.h5"
    shopt -s nullglob
    H5_FILES=($H5_PATTERN)
    shopt -u nullglob

    if [ ${#H5_FILES[@]} -eq 0 ]; then
        mapfile -t H5_FILES < <(find "$RUNS_DIR" -mindepth 2 -maxdepth 2 -path "${RUNS_DIR}/${ARG}*/captures.h5" | sort)
    fi
fi

if [ ${#H5_FILES[@]} -eq 0 ]; then
    echo "No HDF5 files found for: ${ARG:-all}"
    echo "Run capture first: bash scripts/run_capture.sh"
    exit 1
fi

echo "Found ${#H5_FILES[@]} HDF5 file(s):"
for f in "${H5_FILES[@]}"; do echo "  $f"; done

RAW_H5_FILES=()
RUN_H5_FILES=()

for h5 in "${H5_FILES[@]}"; do
    case "$h5" in
        ${RUNS_DIR}/*/captures.h5|*/outputs/runs/*/captures.h5)
            RUN_H5_FILES+=("$h5")
            ;;
        *)
            RAW_H5_FILES+=("$h5")
            ;;
    esac
done

if [ ${#RAW_H5_FILES[@]} -gt 0 ]; then
    python entropy_exp/analysis/entropy_analysis.py \
        --h5 "${RAW_H5_FILES[@]}" \
        --output "$PROCESSED_DIR"
fi

if [ ${#RUN_H5_FILES[@]} -gt 0 ]; then
    for h5 in "${RUN_H5_FILES[@]}"; do
        RUN_OUTPUT_DIR="$(dirname "$h5")/analysis"
        echo ""
        echo "Writing run-local analysis to: $RUN_OUTPUT_DIR"
        python entropy_exp/analysis/entropy_analysis.py \
            --h5 "$h5" \
            --output "$RUN_OUTPUT_DIR"
    done
fi

echo ""
echo "Analysis complete!"
if [ ${#RAW_H5_FILES[@]} -gt 0 ]; then
    echo "  Raw metrics: $PROCESSED_DIR"
fi
if [ ${#RUN_H5_FILES[@]} -gt 0 ]; then
    echo "  Run metrics: <run_dir>/analysis/"
fi
