#!/bin/bash
# =============================================================================
# Entropy Experiment: Check Incomplete Run Directories
#
# Usage:
#   bash scripts/check_incomplete_runs.sh
#   bash scripts/check_incomplete_runs.sh --mode delete
#   bash scripts/check_incomplete_runs.sh --dataset gqa
#   bash scripts/check_incomplete_runs.sh --name-prefix gqa_attn_score_
#   bash scripts/check_incomplete_runs.sh --runs-dir entropy_exp/outputs/runs
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LLAVA_ROOT="$(dirname "$PROJECT_DIR")"

cd "$LLAVA_ROOT"

MODE="preview"
RUNS_DIR="entropy_exp/outputs/runs"
DATASET_FILTER=""
NAME_PREFIX=""

usage() {
    cat <<'EOF'
Usage:
  bash scripts/check_incomplete_runs.sh [--mode preview|delete] [--runs-dir DIR] [--dataset NAME] [--name-prefix PREFIX]

Options:
  --mode         preview problematic runs or delete them (default: preview)
  --runs-dir     directory containing run folders
  --dataset      filter run directories by dataset prefix, for example gqa, mme, pope
  --name-prefix  filter run directories by name prefix
  -h, --help     show this help message
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            MODE="${2:-}"
            shift 2
            ;;
        --runs-dir)
            RUNS_DIR="${2:-}"
            shift 2
            ;;
        --dataset)
            DATASET_FILTER="${2:-}"
            shift 2
            ;;
        --name-prefix)
            NAME_PREFIX="${2:-}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ "$MODE" != "preview" && "$MODE" != "delete" ]]; then
    echo "Invalid --mode: $MODE" >&2
    exit 1
fi

if [[ "$RUNS_DIR" != /* ]]; then
    RUNS_DIR="$LLAVA_ROOT/$RUNS_DIR"
fi

if [[ ! -d "$RUNS_DIR" ]]; then
    echo "Runs directory not found: $RUNS_DIR" >&2
    exit 1
fi

declare -A QUESTION_COUNT_CACHE=()
declare -a RUN_DIRS=()
declare -a REPORTS=()

trim_value() {
    local value="$1"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    value="${value%\"}"
    value="${value#\"}"
    value="${value%\'}"
    value="${value#\'}"
    printf '%s' "$value"
}

resolve_repo_path() {
    local path_str="${1:-}"
    if [[ -z "$path_str" ]]; then
        return 1
    fi
    if [[ "$path_str" = /* ]]; then
        printf '%s\n' "$path_str"
    else
        printf '%s\n' "$LLAVA_ROOT/$path_str"
    fi
}

parse_config_metadata() {
    local config_path="$1"
    awk '
        function trim(s) {
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", s)
            gsub(/^["'"'"'"'"'"']|["'"'"'"'"'"']$/, "", s)
            return s
        }

        /^[[:space:]]*#/ || /^[[:space:]]*$/ { next }

        {
            stripped = $0
            sub(/^[[:space:]]+/, "", stripped)

            indent = match($0, /[^ ]/)
            if (indent == 0) {
                indent = 0
            } else {
                indent -= 1
            }

            if (in_run_meta && indent <= run_meta_indent) {
                in_run_meta = 0
            }

            if (in_datasets && indent <= datasets_indent) {
                in_datasets = 0
                current_dataset = ""
                current_dataset_indent = -1
            }

            if (in_run_meta && stripped ~ /^dataset:/) {
                value = stripped
                sub(/^[^:]+:[[:space:]]*/, "", value)
                dataset = trim(value)
                next
            }

            if (in_datasets) {
                if (stripped ~ /:$/ && indent == datasets_indent + 2) {
                    current_dataset = stripped
                    sub(/:$/, "", current_dataset)
                    current_dataset_indent = indent
                    next
                }

                if (current_dataset != "" && indent <= current_dataset_indent) {
                    current_dataset = ""
                }

                if (current_dataset != "" && stripped ~ /^question_file:/) {
                    value = stripped
                    sub(/^[^:]+:[[:space:]]*/, "", value)
                    question_file[current_dataset] = trim(value)
                    next
                }
            }

            if (stripped == "_run_meta:") {
                in_run_meta = 1
                run_meta_indent = indent
                next
            }

            if (stripped == "datasets:") {
                in_datasets = 1
                datasets_indent = indent
                current_dataset = ""
                current_dataset_indent = -1
                next
            }
        }

        END {
            printf("DATASET=%s\n", dataset)
            for (key in question_file) {
                printf("QUESTION_FILE[%s]=%s\n", key, question_file[key])
            }
        }
    ' "$config_path"
}

count_lines_cached() {
    local file_path="$1"
    if [[ -z "${QUESTION_COUNT_CACHE[$file_path]+x}" ]]; then
        QUESTION_COUNT_CACHE["$file_path"]="$(wc -l < "$file_path" | tr -d ' ')"
    fi
    printf '%s\n' "${QUESTION_COUNT_CACHE[$file_path]}"
}

analyze_answers() {
    local answers_path="$1"
    python - "$answers_path" <<'PY'
import json
import sys

answers_path = sys.argv[1]
valid_answers = 0
total_lines = 0
first_bad_line = ""

with open(answers_path, "r", encoding="utf-8") as f:
    for line_no, line in enumerate(f, 1):
        total_lines += 1
        try:
            json.loads(line)
        except json.JSONDecodeError:
            if not first_bad_line:
                first_bad_line = str(line_no)
            continue
        valid_answers += 1

print(f"{valid_answers}\t{total_lines}\t{first_bad_line}")
PY
}

discover_run_dirs() {
    local run_dir
    while IFS= read -r run_dir; do
        [[ -n "$run_dir" ]] || continue

        if [[ -n "$NAME_PREFIX" && "$(basename "$run_dir")" != "$NAME_PREFIX"* ]]; then
            continue
        fi

        if [[ ! -f "$run_dir/config.yaml" && ! -f "$run_dir/answers.jsonl" ]]; then
            continue
        fi

        if [[ -n "$DATASET_FILTER" && "$(basename "$run_dir")" != "${DATASET_FILTER}_"* ]]; then
            continue
        fi

        RUN_DIRS+=("$run_dir")
    done < <(find "$RUNS_DIR" -mindepth 1 -maxdepth 1 -type d | sort)
}

build_report() {
    local run_dir="$1"
    local config_path="$run_dir/config.yaml"
    local answers_path="$run_dir/answers.jsonl"
    local dataset=""
    local expected_answers=""
    local valid_answers=0
    local total_lines=0
    local first_bad_line=""
    local question_file=""
    local status=""
    local -a reasons=()

    if [[ -f "$config_path" ]]; then
        while IFS= read -r line; do
            [[ -n "$line" ]] || continue
            if [[ "$line" == DATASET=* ]]; then
                dataset="$(trim_value "${line#DATASET=}")"
            elif [[ "$line" == QUESTION_FILE\[*\]=* ]]; then
                local key="${line#QUESTION_FILE[}"
                key="${key%%]=*}"
                local value="${line#*=}"
                if [[ "$key" == "$dataset" ]]; then
                    question_file="$(trim_value "$value")"
                fi
            fi
        done < <(parse_config_metadata "$config_path")
    else
        reasons+=("missing config.yaml")
    fi

    if [[ -z "$dataset" ]]; then
        reasons+=("dataset missing in _run_meta")
    fi

    if [[ -n "$dataset" ]]; then
        if [[ -z "$question_file" ]]; then
            while IFS= read -r line; do
                if [[ "$line" == QUESTION_FILE\[*\]=* ]]; then
                    local key="${line#QUESTION_FILE[}"
                    key="${key%%]=*}"
                    if [[ "$key" == "$dataset" ]]; then
                        question_file="$(trim_value "${line#*=}")"
                        break
                    fi
                fi
            done < <(parse_config_metadata "$config_path")
        fi

        if [[ -z "$question_file" ]]; then
            reasons+=("question_file missing for dataset=$dataset")
        else
            local resolved_question_file
            resolved_question_file="$(resolve_repo_path "$question_file")"
            if [[ ! -f "$resolved_question_file" ]]; then
                reasons+=("question_file not found: $resolved_question_file")
            else
                expected_answers="$(count_lines_cached "$resolved_question_file")"
            fi
        fi
    fi

    if [[ -f "$answers_path" ]]; then
        IFS=$'\t' read -r valid_answers total_lines first_bad_line < <(analyze_answers "$answers_path")
        if [[ -n "$first_bad_line" ]]; then
            reasons+=("malformed JSONL at line $first_bad_line")
        fi
    else
        reasons+=("missing answers.jsonl")
    fi

    if [[ -n "$expected_answers" && "$valid_answers" != "$expected_answers" ]]; then
        reasons+=("valid answers $valid_answers/$expected_answers")
    fi

    if [[ -n "$first_bad_line" && -n "$expected_answers" && "$valid_answers" != "$expected_answers" ]]; then
        status="malformed_and_incomplete"
    elif [[ -n "$first_bad_line" ]]; then
        status="malformed"
    elif [[ -n "$expected_answers" && "$valid_answers" != "$expected_answers" ]]; then
        status="incomplete"
    elif [[ ${#reasons[@]} -gt 0 ]]; then
        status="invalid"
    else
        status="complete"
    fi

    local expected_display="${expected_answers:-?}"
    local reasons_text=""
    if [[ ${#reasons[@]} -gt 0 ]]; then
        local reason
        for reason in "${reasons[@]}"; do
            if [[ -n "$reasons_text" ]]; then
                reasons_text="$reasons_text; "
            fi
            reasons_text="$reasons_text$reason"
        done
    fi

    REPORTS+=("$status"$'\t'"$run_dir"$'\t'"${dataset:-?}"$'\t'"$valid_answers"$'\t'"$expected_display"$'\t'"$total_lines"$'\t'"$reasons_text")
}

print_preview() {
    local problematic_count=0
    local report
    for report in "${REPORTS[@]}"; do
        local status="${report%%$'\t'*}"
        if [[ "$status" != "complete" ]]; then
            problematic_count=$((problematic_count + 1))
        fi
    done

    echo "Runs dir: $RUNS_DIR"
    echo "Mode: $MODE"
    echo "Scanned runs: ${#REPORTS[@]}"
    echo "Problematic runs: $problematic_count"

    if [[ "$problematic_count" -eq 0 ]]; then
        echo "No incomplete run directories detected."
        return
    fi

    for report in "${REPORTS[@]}"; do
        IFS=$'\t' read -r status run_dir dataset valid_answers expected_answers total_lines reasons_text <<< "$report"
        if [[ "$status" == "complete" ]]; then
            continue
        fi
        echo "- $run_dir | dataset=$dataset | status=$status | valid=$valid_answers/$expected_answers | lines=$total_lines"
        echo "  reasons: $reasons_text"
    done
}

delete_problematic_runs() {
    local deleted=0
    local report
    for report in "${REPORTS[@]}"; do
        IFS=$'\t' read -r status run_dir _ <<< "$report"
        if [[ "$status" == "complete" ]]; then
            continue
        fi
        rm -rf "$run_dir"
        deleted=$((deleted + 1))
        echo "DELETED $run_dir"
    done
    echo "Deleted $deleted problematic run(s)."
}

discover_run_dirs

if [[ ${#RUN_DIRS[@]} -eq 0 ]]; then
    echo "No matching run directories found."
    exit 0
fi

for run_dir in "${RUN_DIRS[@]}"; do
    build_report "$run_dir"
done

print_preview

if [[ "$MODE" == "delete" ]]; then
    delete_problematic_runs
fi
