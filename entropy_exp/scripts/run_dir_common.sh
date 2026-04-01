#!/bin/bash

preferred_llava_env_name() {
    printf '%s\n' "${PREFERRED_LLAVA_ENV_NAME:-llava}"
}

preferred_conda_sh_path() {
    printf '%s\n' "${HOME}/miniconda3/etc/profile.d/conda.sh"
}

preferred_llava_prefix_path() {
    printf '%s\n' "${HOME}/miniconda3/envs/$(preferred_llava_env_name)"
}

preferred_llava_python_path() {
    printf '%s\n' "$(preferred_llava_prefix_path)/bin/python"
}

activate_llava_env() {
    local env_name preferred_prefix preferred_conda_sh
    env_name="$(preferred_llava_env_name)"
    preferred_prefix="$(preferred_llava_prefix_path)"
    preferred_conda_sh="$(preferred_conda_sh_path)"

    if [[ "${CONDA_PREFIX:-}" == "$preferred_prefix" ]]; then
        return 0
    fi

    set +u
    if [[ -f "$preferred_conda_sh" ]]; then
        # shellcheck disable=SC1090
        if source "$preferred_conda_sh" && conda activate "$preferred_prefix"; then
            set -u
            return 0
        fi
        set -u
        echo "Warning: failed to activate '${env_name}' from ${preferred_conda_sh}; continuing with the current shell." >&2
        return 0
    fi

    if [[ "${CONDA_DEFAULT_ENV:-}" == "$env_name" ]]; then
        set -u
        return 0
    fi

    if command -v conda >/dev/null 2>&1; then
        if eval "$(conda shell.bash hook)" && conda activate "$env_name"; then
            set -u
            return 0
        fi
    fi
    set -u

    echo "Warning: could not activate the '${env_name}' conda environment; continuing with the current shell." >&2
    return 0
}

ensure_python_bin() {
    if [[ -n "${PYTHON_BIN:-}" ]]; then
        return 0
    fi

    local home_python
    home_python="$(preferred_llava_python_path)"
    if [[ -x "$home_python" ]]; then
        PYTHON_BIN="$home_python"
        return 0
    fi

    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python3)"
        return 0
    fi

    if command -v python >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python)"
        return 0
    fi

    echo "Python interpreter not found. Set PYTHON_BIN explicitly." >&2
    return 1
}

run_dir_helper_list() {
    local -a cmd=("${PYTHON_BIN:-python}" "$SCRIPT_DIR/run_dir_helper.py" list --runs-dir "$RUNS_DIR")
    while [[ $# -gt 0 ]]; do
        cmd+=("$1")
        shift
    done
    "${cmd[@]}"
}

run_dirs_list_all() {
    local required_file="${1:-}"
    shift || true

    local -a extra=()
    if [[ -n "$required_file" ]]; then
        extra+=(--require-file "$required_file")
    fi
    run_dir_helper_list "${extra[@]}" "$@"
}

run_dirs_list_prefix() {
    local prefix="$1"
    local required_file="${2:-}"
    shift 2 || true

    local -a extra=(--prefix "$prefix")
    if [[ -n "$required_file" ]]; then
        extra+=(--require-file "$required_file")
    fi
    run_dir_helper_list "${extra[@]}" "$@"
}

run_dirs_resolve_exact_name() {
    local run_name="$1"
    local required_file="${2:-}"
    shift 2 || true

    local -a extra=(--exact-name "$run_name")
    if [[ -n "$required_file" ]]; then
        extra+=(--require-file "$required_file")
    fi

    local -a matches=()
    mapfile -t matches < <(run_dir_helper_list "${extra[@]}" "$@")
    if [[ ${#matches[@]} -eq 1 ]]; then
        printf '%s\n' "${matches[0]}"
        return 0
    fi
    if [[ ${#matches[@]} -gt 1 ]]; then
        echo "Multiple run directories matched '$run_name':" >&2
        printf '  %s\n' "${matches[@]}" >&2
        return 2
    fi
    return 1
}

run_dirs_resolve_input() {
    local input_path="$1"
    local required_file="${2:-}"

    if [[ -d "$input_path" && ( -z "$required_file" || -f "$input_path/$required_file" ) ]]; then
        (cd "$input_path" && pwd)
        return 0
    fi

    if [[ -d "$RUNS_DIR/$input_path" && ( -z "$required_file" || -f "$RUNS_DIR/$input_path/$required_file" ) ]]; then
        (cd "$RUNS_DIR/$input_path" && pwd)
        return 0
    fi

    if [[ -n "$required_file" && -f "$input_path" && "$(basename "$input_path")" = "$required_file" ]]; then
        (cd "$(dirname "$input_path")" && pwd)
        return 0
    fi

    run_dirs_resolve_exact_name "$input_path" "$required_file"
}

run_dirs_resolve_summary_input() {
    local input_path="$1"

    if [[ -d "$input_path" && -f "$input_path/config.yaml" ]]; then
        (cd "$input_path" && pwd)
        return 0
    fi

    if [[ -d "$RUNS_DIR/$input_path" && -f "$RUNS_DIR/$input_path/config.yaml" ]]; then
        (cd "$RUNS_DIR/$input_path" && pwd)
        return 0
    fi

    if [[ -f "$input_path" && "$(basename "$input_path")" = "answers.jsonl" ]]; then
        (cd "$(dirname "$input_path")" && pwd)
        return 0
    fi

    if [[ -f "$input_path" && "$(basename "$input_path")" = "summary.json" ]]; then
        (cd "$(dirname "$(dirname "$input_path")")" && pwd)
        return 0
    fi

    if [[ -d "$input_path" && -f "$input_path/summary.json" ]]; then
        (cd "$(dirname "$input_path")" && pwd)
        return 0
    fi

    run_dirs_resolve_exact_name "$input_path" "config.yaml"
}
