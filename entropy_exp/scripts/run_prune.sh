#!/bin/bash
# =============================================================================
# Pruning Experiment: Run Pruning Inference
#
# Usage:
#   bash scripts/run_prune.sh <strategy> <dataset> [max_samples] [--auto-gpu|--no-auto-gpu] [--set key=val ...]
#
# Examples:
#   bash scripts/run_prune.sh attn_score mme 10
#   bash scripts/run_prune.sh pre_attn_score mme 10
#   bash scripts/run_prune.sh masking_attn_score gqa 10
#   bash scripts/run_prune.sh entropy pope
#   bash scripts/run_prune.sh random mme 10
#   bash scripts/run_prune.sh sparsevlm mme 10
#   bash scripts/run_prune.sh baseline mme 10
#   bash scripts/run_prune.sh attn_score mme 10 --auto-gpu
#   CUDA_VISIBLE_DEVICES=2,3 bash scripts/run_prune.sh attn_score mme --auto-gpu
#   bash scripts/run_prune.sh attn_score mme --no-auto-gpu
#   bash scripts/run_prune.sh attn_score mme 10 --set pruning.prune_layers=[5] --set pruning.prune_ratio=[0.7]
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LLAVA_ROOT="$(dirname "$PROJECT_DIR")"

cd "$LLAVA_ROOT"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_prune.sh <strategy|baseline> <dataset|all> [max_samples] [--auto-gpu|--no-auto-gpu] [--set key=val ...]

Examples:
  bash scripts/run_prune.sh attn_score mme 10
  bash scripts/run_prune.sh pre_attn_score mme 10
  bash scripts/run_prune.sh masking_attn_score gqa 10
  bash scripts/run_prune.sh entropy pope
  bash scripts/run_prune.sh random mme 10
  bash scripts/run_prune.sh sparsevlm mme 10
  bash scripts/run_prune.sh baseline mme 10
  bash scripts/run_prune.sh attn_score mme 10 --auto-gpu
  CUDA_VISIBLE_DEVICES=2,3 bash scripts/run_prune.sh attn_score mme --auto-gpu
  bash scripts/run_prune.sh attn_score mme --no-auto-gpu
  bash scripts/run_prune.sh attn_score mme 10 --set pruning.prune_layers=[5] --set pruning.prune_ratio=[0.7]
EOF
}

trim_whitespace() {
    local value="$1"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf '%s' "$value"
}

STRATEGY="${1:?$(usage)}"
DATASET="${2:?$(usage)}"
shift 2

case "$STRATEGY" in
    baseline|attn_score|pre_attn_score|masking_attn_score|entropy|random|sparsevlm)
        ;;
    compare)
        echo "Strategy 'compare' has been removed. Please run strategies explicitly." >&2
        exit 1
        ;;
    *)
        echo "Unknown strategy: $STRATEGY" >&2
        echo "Supported strategies: baseline, attn_score, pre_attn_score, masking_attn_score, entropy, random, sparsevlm" >&2
        exit 1
        ;;
esac

# Parse optional max_samples (first positional arg that is a plain integer)
MAX_SAMPLES=""
EXTRA_SETS=()
AUTO_GPU=0
AUTO_GPU_SAMPLES=10
AUTO_GPU_INTERVAL=1
ORIGINAL_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --auto-gpu)
            AUTO_GPU=1
            shift
            ;;
        --no-auto-gpu)
            AUTO_GPU=0
            shift
            ;;
        --set)
            if [[ $# -lt 2 ]]; then
                echo "Missing value after --set" >&2
                usage >&2
                exit 1
            fi
            EXTRA_SETS+=("--set" "$2")
            shift 2
            ;;
        *)
            if [[ -z "$MAX_SAMPLES" && "$1" =~ ^[0-9]+$ ]]; then
                MAX_SAMPLES="$1"
            else
                echo "Unknown argument: $1" >&2
                usage >&2
                exit 1
            fi
            shift
            ;;
    esac
done

CONFIG="entropy_exp/configs/prune.yaml"

source "$SCRIPT_DIR/run_dir_common.sh"
activate_llava_env
ensure_python_bin

collect_candidate_gpus() {
    local visible="$1"
    local -n out_ref="$2"
    out_ref=()

    if [[ -n "$visible" ]]; then
        local raw_gpu trimmed_gpu
        IFS=',' read -r -a raw_gpus <<< "$visible"
        for raw_gpu in "${raw_gpus[@]}"; do
            trimmed_gpu="$(trim_whitespace "$raw_gpu")"
            if [[ -z "$trimmed_gpu" ]]; then
                continue
            fi
            if [[ ! "$trimmed_gpu" =~ ^[0-9]+$ ]]; then
                echo "[gpu-router] Warning: unsupported CUDA_VISIBLE_DEVICES entry '$trimmed_gpu'; expected numeric GPU indices." >&2
                return 1
            fi
            out_ref+=("$trimmed_gpu")
        done
        return 0
    fi

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "[gpu-router] Warning: nvidia-smi is not available." >&2
        return 1
    fi

    local query_output
    if ! query_output="$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null)"; then
        echo "[gpu-router] Warning: failed to query GPU indices via nvidia-smi." >&2
        return 1
    fi

    local line gpu_index
    while IFS= read -r line; do
        line="$(trim_whitespace "$line")"
        [[ -z "$line" ]] && continue
        gpu_index="${line%%,*}"
        gpu_index="$(trim_whitespace "$gpu_index")"
        [[ -z "$gpu_index" ]] && continue
        out_ref+=("$gpu_index")
    done <<< "$query_output"

    if [[ ${#out_ref[@]} -eq 0 ]]; then
        echo "[gpu-router] Warning: no GPU devices were returned by nvidia-smi." >&2
        return 1
    fi

    return 0
}

select_best_gpu() {
    local sample_count="$1"
    local sample_interval="$2"
    shift 2

    local candidate_gpus=("$@")
    if [[ ${#candidate_gpus[@]} -eq 0 ]]; then
        echo "[gpu-router] Warning: candidate GPU list is empty." >&2
        return 1
    fi

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "[gpu-router] Warning: nvidia-smi is not available." >&2
        return 1
    fi

    declare -A candidate_set=()
    declare -A free_sum=()
    local gpu
    for gpu in "${candidate_gpus[@]}"; do
        candidate_set["$gpu"]=1
        free_sum["$gpu"]=0
    done

    local sample line idx free matched_count
    for ((sample = 1; sample <= sample_count; sample++)); do
        matched_count=0
        local sample_output
        if ! sample_output="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null)"; then
            echo "[gpu-router] Warning: failed to query memory.free via nvidia-smi." >&2
            return 1
        fi

        while IFS= read -r line; do
            line="$(trim_whitespace "$line")"
            [[ -z "$line" ]] && continue

            idx="${line%%,*}"
            idx="$(trim_whitespace "$idx")"
            free="${line#*,}"
            free="$(trim_whitespace "$free")"

            [[ -z "${candidate_set[$idx]:-}" ]] && continue
            [[ ! "$free" =~ ^[0-9]+$ ]] && continue

            free_sum["$idx"]=$((free_sum["$idx"] + free))
            matched_count=$((matched_count + 1))
        done <<< "$sample_output"

        if (( matched_count != ${#candidate_gpus[@]} )); then
            echo "[gpu-router] Warning: failed to read memory.free for all candidate GPUs from nvidia-smi." >&2
            return 1
        fi

        if (( sample < sample_count )); then
            sleep "$sample_interval"
        fi
    done

    local best_gpu=""
    local best_avg=-1
    local avg
    echo "[gpu-router] Average free GPU memory over ${sample_count}s:" >&2
    for gpu in "${candidate_gpus[@]}"; do
        avg=$((free_sum["$gpu"] / sample_count))
        echo "[gpu-router]   GPU $gpu: ${avg} MiB" >&2
        if (( avg > best_avg )) || { (( avg == best_avg )) && [[ -n "$best_gpu" ]] && (( gpu < best_gpu )); }; then
            best_avg="$avg"
            best_gpu="$gpu"
        elif (( avg == best_avg )) && [[ -z "$best_gpu" ]]; then
            best_avg="$avg"
            best_gpu="$gpu"
        fi
    done

    if [[ -z "$best_gpu" ]]; then
        echo "[gpu-router] Warning: unable to select a GPU from sampled data." >&2
        return 1
    fi

    printf '%s\n' "$best_gpu"
}

configure_gpu_router() {
    echo "[gpu-router] Auto router: $([[ "$AUTO_GPU" -eq 1 ]] && echo enabled || echo disabled)"

    if [[ "$AUTO_GPU" -ne 1 ]]; then
        if [[ -n "$ORIGINAL_CUDA_VISIBLE_DEVICES" ]]; then
            echo "[gpu-router] Using existing CUDA_VISIBLE_DEVICES=$ORIGINAL_CUDA_VISIBLE_DEVICES"
        else
            echo "[gpu-router] Auto router disabled; leaving CUDA_VISIBLE_DEVICES unset."
        fi
        return 0
    fi

    local candidate_gpus=()
    if ! collect_candidate_gpus "$ORIGINAL_CUDA_VISIBLE_DEVICES" candidate_gpus; then
        if [[ -n "$ORIGINAL_CUDA_VISIBLE_DEVICES" ]]; then
            export CUDA_VISIBLE_DEVICES="$ORIGINAL_CUDA_VISIBLE_DEVICES"
            echo "[gpu-router] Warning: keeping existing CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2
        else
            export CUDA_VISIBLE_DEVICES="0"
            echo "[gpu-router] Warning: falling back to CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2
        fi
        return 0
    fi

    echo "[gpu-router] Candidate GPUs: ${candidate_gpus[*]}"
    echo "[gpu-router] Sampling GPU free memory once per second for ${AUTO_GPU_SAMPLES}s..."

    local selected_gpu
    if ! selected_gpu="$(select_best_gpu "$AUTO_GPU_SAMPLES" "$AUTO_GPU_INTERVAL" "${candidate_gpus[@]}")"; then
        if [[ -n "$ORIGINAL_CUDA_VISIBLE_DEVICES" ]]; then
            export CUDA_VISIBLE_DEVICES="$ORIGINAL_CUDA_VISIBLE_DEVICES"
            echo "[gpu-router] Warning: keeping existing CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2
        else
            export CUDA_VISIBLE_DEVICES="0"
            echo "[gpu-router] Warning: falling back to CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2
        fi
        return 0
    fi

    export CUDA_VISIBLE_DEVICES="$selected_gpu"
    echo "[gpu-router] Selected GPU: $selected_gpu"
    echo "[gpu-router] Exported CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
}

configure_gpu_router

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
        "$PYTHON_BIN" entropy_exp/src/prune_inference.py \
            --config "$CONFIG" \
            --dataset "$ds" \
            --baseline \
            "${flags[@]}" \
            "${EXTRA_SETS[@]}"
    else
        echo "========================================"
        echo " Strategy: $strategy  |  Dataset: $ds"
        echo "========================================"
        "$PYTHON_BIN" entropy_exp/src/prune_inference.py \
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
