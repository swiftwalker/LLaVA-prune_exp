"""Persistent tmux-based experiment scheduler for entropy_exp."""

from __future__ import annotations

import json
import math
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

try:
    from entropy_exp.src.run_layout import find_run_dirs, resolve_repo_path
except ImportError:  # pragma: no cover - fallback for direct src imports
    from run_layout import find_run_dirs, resolve_repo_path

try:
    from entropy_exp.src.dataset_adapters import count_dataset_samples
except ImportError:  # pragma: no cover - fallback for direct src imports
    from dataset_adapters import count_dataset_samples


LLAVA_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_BASE_DIR = LLAVA_ROOT / "entropy_exp" / "outputs" / "scheduler"
DEFAULT_RUNS_DIR = LLAVA_ROOT / "entropy_exp" / "outputs" / "runs"
DEFAULT_CONDA_SH = Path.home() / "miniconda3" / "etc" / "profile.d" / "conda.sh"
DEFAULT_CONDA_ENV = "llava"

SUPPORTED_DATASETS = {"gqa", "mme", "pope", "textvqa", "scienceqa", "mmbench"}
SUPPORTED_STRATEGIES = {
    "baseline",
    "attn_score",
    "pre_attn_score",
    "masking_attn_score",
    "tail_masking_attn_score",
    "entropy",
    "random",
    "sparsevlm",
    "sparsevlm_adaptive_stratified",
    "sparsevlm_boost",
    "sparsevlm_compensated",
    "sparsevlm_entropy_alpha",
    "sparsevlm_entropy_alpha_global",
}
MIN_FREE_MIB_DEFAULT = 16 * 1024
VISIBLE_GPUS_ENV = "LLAVA_SCHEDULER_VISIBLE_GPUS"


class SchedulerError(RuntimeError):
    """Raised when scheduler configuration or runtime state is invalid."""


@dataclass(frozen=True)
class GPUConfig:
    min_free_gib: int
    selection: str
    sample_seconds: int
    poll_interval_seconds: int


@dataclass(frozen=True)
class RetryConfig:
    budget_ratio: float
    rounding: str


@dataclass(frozen=True)
class TmuxConfig:
    session_name: str
    log_dir: str


@dataclass(frozen=True)
class EnvironmentConfig:
    conda_sh: str
    conda_env: str
    extra_env: List[str]


@dataclass(frozen=True)
class DefaultsConfig:
    max_samples: Optional[int]
    extra_sets: List[str]


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    dataset: str
    strategies: List[str]
    extra_sets: List[str]
    max_samples: Optional[int]


@dataclass(frozen=True)
class SchedulerPlan:
    version: int
    label: str
    pool_size: int
    gpu: GPUConfig
    retry: RetryConfig
    tmux: TmuxConfig
    environment: EnvironmentConfig
    defaults: DefaultsConfig
    experiments: List[ExperimentSpec]


@dataclass(frozen=True)
class JobSpec:
    job_id: str
    experiment_name: str
    dataset: str
    strategy: str
    max_samples: Optional[int]
    extra_sets: List[str]
    run_prefix: str
    window_base_name: str


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def resolve_conda_sh_value(raw_value: Any) -> str:
    text = str(raw_value or "").strip()
    if not text:
        return str(DEFAULT_CONDA_SH)
    return os.path.expandvars(os.path.expanduser(text))


def resolve_conda_activate_target(conda_sh: str, conda_env: str) -> str:
    env_text = str(conda_env).strip()
    if not env_text:
        return env_text

    expanded_env = os.path.expandvars(os.path.expanduser(env_text))
    if expanded_env != env_text or "/" in env_text:
        return expanded_env

    conda_sh_path = Path(conda_sh).expanduser()
    try:
        conda_root = conda_sh_path.parents[2]
    except IndexError:
        return env_text
    return str(conda_root / "envs" / env_text)


def validate_env_assignments(values: List[str], field_name: str) -> List[str]:
    validated: List[str] = []
    for raw_value in values:
        if "=" not in raw_value:
            raise SchedulerError(f"{field_name} entries must be KEY=VALUE assignments: {raw_value!r}")
        key, _value = raw_value.split("=", 1)
        key = key.strip()
        if not key:
            raise SchedulerError(f"{field_name} entries must include a variable name: {raw_value!r}")
        if not (key[0].isalpha() or key[0] == "_"):
            raise SchedulerError(f"{field_name} variable names must start with a letter or underscore: {raw_value!r}")
        if any(not (char.isalnum() or char == "_") for char in key):
            raise SchedulerError(f"{field_name} variable names may only contain letters, digits, and underscores: {raw_value!r}")
        validated.append(f"{key}={raw_value.split('=', 1)[1]}")
    return validated


def ensure_list_of_strings(value: Any, field_name: str) -> List[str]:
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise SchedulerError(f"{field_name} must be a list of strings")
    return list(value)


def merge_extra_sets(*groups: Iterable[str]) -> List[str]:
    merged: List[str] = []
    seen: set[str] = set()
    for group in groups:
        for item in group:
            if item in seen:
                continue
            merged.append(item)
            seen.add(item)
    return merged


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
    tmp_path.replace(path)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_yaml(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
    tmp_path.replace(path)


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise SchedulerError(f"YAML must contain a top-level mapping: {path}")
    return data


def sanitize_name(value: str) -> str:
    cleaned = []
    for char in value:
        if char.isalnum() or char in {"_", "-", "."}:
            cleaned.append(char)
        else:
            cleaned.append("_")
    return "".join(cleaned).strip("._-") or "job"


def parse_override_value(extra_sets: Iterable[str], key: str) -> Any:
    prefix = f"{key}="
    for override in extra_sets:
        if override.startswith(prefix):
            return yaml.safe_load(override[len(prefix) :])
    return None


def encode_run_value(value: Any) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        text = f"{value:.6g}"
    else:
        text = str(value)
    return text.replace("-", "m").replace(".", "p")


def encode_run_list(values: Any) -> str:
    if not isinstance(values, list):
        raise SchedulerError(f"Expected list for run-name encoding, got {values!r}")
    return "-".join(encode_run_value(item) for item in values)


def build_run_prefix(dataset: str, strategy: str, extra_sets: Iterable[str]) -> str:
    run_tag_suffix = parse_override_value(extra_sets, "output.run_tag_suffix")
    tag = "baseline" if strategy == "baseline" else strategy
    if run_tag_suffix is not None and str(run_tag_suffix).strip():
        tag = f"{tag}_{sanitize_name(str(run_tag_suffix))}"
    if strategy == "baseline":
        return f"{dataset}_{tag}_"
    layers = parse_override_value(extra_sets, "pruning.prune_layers")
    ratios = parse_override_value(extra_sets, "pruning.prune_ratio")
    if not isinstance(layers, list) or not isinstance(ratios, list):
        raise SchedulerError(
            f"Job {dataset}/{strategy} is missing pruning.prune_layers or pruning.prune_ratio overrides"
        )
    return f"{dataset}_{tag}_l{encode_run_list(layers)}_r{encode_run_list(ratios)}__"


def build_window_base_name(dataset: str, strategy: str, extra_sets: Iterable[str]) -> str:
    run_tag_suffix = parse_override_value(extra_sets, "output.run_tag_suffix")
    strategy_label = strategy
    if run_tag_suffix is not None and str(run_tag_suffix).strip():
        strategy_label = f"{strategy}_{sanitize_name(str(run_tag_suffix))}"
    layers = parse_override_value(extra_sets, "pruning.prune_layers")
    ratios = parse_override_value(extra_sets, "pruning.prune_ratio")
    if not isinstance(layers, list) or not isinstance(ratios, list):
        raise SchedulerError(
            f"Job {dataset}/{strategy} is missing pruning.prune_layers or pruning.prune_ratio overrides"
        )
    return sanitize_name(
        f"{dataset}_{strategy_label}_l{encode_run_list(layers)}_r{encode_run_list(ratios)}"
    )


def build_run_command(job: JobSpec) -> List[str]:
    command = ["bash", "entropy_exp/scripts/run_prune.sh", job.strategy, job.dataset]
    if job.max_samples is not None:
        command.append(str(job.max_samples))
    command.append("--no-auto-gpu")
    for override in job.extra_sets:
        command.extend(["--set", override])
    return command


def runs_dir_for_extra_sets(extra_sets: Iterable[str], repo_root: Path) -> Path:
    output_base_dir = parse_override_value(extra_sets, "output.base_dir")
    if output_base_dir is None:
        return repo_root / "entropy_exp" / "outputs" / "runs"
    output_base_path = Path(output_base_dir)
    if not output_base_path.is_absolute():
        output_base_path = repo_root / output_base_path
    return output_base_path.resolve() / "runs"


def compute_retry_budget(total_jobs: int, budget_ratio: float, rounding: str) -> int:
    if total_jobs < 0:
        raise SchedulerError("total_jobs must be non-negative")
    raw = total_jobs * budget_ratio
    if rounding == "ceil":
        return int(math.ceil(raw))
    if rounding == "floor":
        return int(math.floor(raw))
    if rounding == "round":
        return int(round(raw))
    raise SchedulerError(f"Unsupported retry.rounding: {rounding}")


def load_scheduler_plan(plan_path: Path) -> SchedulerPlan:
    raw = load_yaml(plan_path)

    version = raw.get("version")
    if version != 1:
        raise SchedulerError("Plan version must be 1")

    label = raw.get("label")
    if not isinstance(label, str) or not label.strip():
        raise SchedulerError("label must be a non-empty string")

    pool_size = raw.get("pool_size", 6)
    if not isinstance(pool_size, int) or pool_size <= 0:
        raise SchedulerError("pool_size must be a positive integer")

    gpu_raw = raw.get("gpu")
    if not isinstance(gpu_raw, dict):
        raise SchedulerError("gpu must be a mapping")
    gpu_cfg = GPUConfig(
        min_free_gib=int(gpu_raw.get("min_free_gib", 16)),
        selection=str(gpu_raw.get("selection", "max_free")),
        sample_seconds=int(gpu_raw.get("sample_seconds", 3)),
        poll_interval_seconds=int(gpu_raw.get("poll_interval_seconds", 15)),
    )
    if gpu_cfg.min_free_gib <= 0:
        raise SchedulerError("gpu.min_free_gib must be positive")
    if gpu_cfg.sample_seconds <= 0:
        raise SchedulerError("gpu.sample_seconds must be positive")
    if gpu_cfg.poll_interval_seconds <= 0:
        raise SchedulerError("gpu.poll_interval_seconds must be positive")
    if gpu_cfg.selection != "max_free":
        raise SchedulerError("gpu.selection must be 'max_free' in v1")

    retry_raw = raw.get("retry")
    if not isinstance(retry_raw, dict):
        raise SchedulerError("retry must be a mapping")
    retry_cfg = RetryConfig(
        budget_ratio=float(retry_raw.get("budget_ratio", 0.1)),
        rounding=str(retry_raw.get("rounding", "ceil")),
    )
    if retry_cfg.budget_ratio < 0:
        raise SchedulerError("retry.budget_ratio must be >= 0")
    if retry_cfg.rounding not in {"ceil", "floor", "round"}:
        raise SchedulerError("retry.rounding must be one of ceil/floor/round")

    tmux_raw = raw.get("tmux")
    if not isinstance(tmux_raw, dict):
        raise SchedulerError("tmux must be a mapping")
    tmux_cfg = TmuxConfig(
        session_name=str(tmux_raw.get("session_name", "")).strip(),
        log_dir=str(tmux_raw.get("log_dir", "")).strip(),
    )
    if not tmux_cfg.session_name:
        raise SchedulerError("tmux.session_name must be provided")
    if not tmux_cfg.log_dir:
        raise SchedulerError("tmux.log_dir must be provided")

    env_raw = raw.get("environment")
    if not isinstance(env_raw, dict):
        raise SchedulerError("environment must be a mapping")
    env_cfg = EnvironmentConfig(
        conda_sh=resolve_conda_sh_value(env_raw.get("conda_sh")),
        conda_env=str(env_raw.get("conda_env", DEFAULT_CONDA_ENV)).strip(),
        extra_env=validate_env_assignments(
            ensure_list_of_strings(env_raw.get("extra_env"), "environment.extra_env"),
            "environment.extra_env",
        ),
    )
    if not env_cfg.conda_env:
        raise SchedulerError("environment.conda_env must be provided")

    defaults_raw = raw.get("defaults", {})
    if not isinstance(defaults_raw, dict):
        raise SchedulerError("defaults must be a mapping")
    max_samples = defaults_raw.get("max_samples")
    if max_samples is not None and (not isinstance(max_samples, int) or max_samples <= 0):
        raise SchedulerError("defaults.max_samples must be null or a positive integer")
    defaults_cfg = DefaultsConfig(
        max_samples=max_samples,
        extra_sets=ensure_list_of_strings(defaults_raw.get("extra_sets"), "defaults.extra_sets"),
    )

    experiments_raw = raw.get("experiments")
    if not isinstance(experiments_raw, list) or not experiments_raw:
        raise SchedulerError("experiments must be a non-empty list")
    experiments: List[ExperimentSpec] = []
    for idx, item in enumerate(experiments_raw, start=1):
        if not isinstance(item, dict):
            raise SchedulerError(f"experiments[{idx}] must be a mapping")
        name = item.get("name")
        dataset = item.get("dataset")
        strategies = item.get("strategies")
        item_max_samples = item.get("max_samples", defaults_cfg.max_samples)
        extra_sets = merge_extra_sets(
            defaults_cfg.extra_sets,
            ensure_list_of_strings(item.get("extra_sets"), f"experiments[{idx}].extra_sets"),
        )

        if not isinstance(name, str) or not name.strip():
            raise SchedulerError(f"experiments[{idx}].name must be a non-empty string")
        if dataset not in SUPPORTED_DATASETS:
            raise SchedulerError(f"experiments[{idx}].dataset must be one of {sorted(SUPPORTED_DATASETS)}")
        if not isinstance(strategies, list) or not strategies:
            raise SchedulerError(f"experiments[{idx}].strategies must be a non-empty list")
        for strategy in strategies:
            if strategy not in SUPPORTED_STRATEGIES:
                raise SchedulerError(f"experiments[{idx}] has unsupported strategy: {strategy}")
        if item_max_samples is not None and (not isinstance(item_max_samples, int) or item_max_samples <= 0):
            raise SchedulerError(f"experiments[{idx}].max_samples must be null or a positive integer")
        experiments.append(
            ExperimentSpec(
                name=name,
                dataset=dataset,
                strategies=list(strategies),
                extra_sets=extra_sets,
                max_samples=item_max_samples,
            )
        )

    return SchedulerPlan(
        version=version,
        label=label,
        pool_size=pool_size,
        gpu=gpu_cfg,
        retry=retry_cfg,
        tmux=tmux_cfg,
        environment=env_cfg,
        defaults=defaults_cfg,
        experiments=experiments,
    )


def expand_jobs(plan: SchedulerPlan) -> List[JobSpec]:
    jobs: List[JobSpec] = []
    counter = 1
    for experiment in plan.experiments:
        for strategy in experiment.strategies:
            job_id = f"job_{counter:04d}"
            jobs.append(
                JobSpec(
                    job_id=job_id,
                    experiment_name=experiment.name,
                    dataset=experiment.dataset,
                    strategy=strategy,
                    max_samples=experiment.max_samples,
                    extra_sets=list(experiment.extra_sets),
                    run_prefix=build_run_prefix(experiment.dataset, strategy, experiment.extra_sets),
                    window_base_name=build_window_base_name(experiment.dataset, strategy, experiment.extra_sets),
                )
            )
            counter += 1
    return jobs


def job_to_payload(job: JobSpec) -> Dict[str, Any]:
    payload = asdict(job)
    payload["command"] = build_run_command(job)
    return payload


def state_file_path(state_dir: Path) -> Path:
    return state_dir / "state.json"


def progress_json_path(state_dir: Path) -> Path:
    return state_dir / "progress.json"


def progress_text_path(state_dir: Path) -> Path:
    return state_dir / "progress.txt"


def job_file_path(state_dir: Path, job_id: str) -> Path:
    return state_dir / "jobs" / f"{job_id}.json"


def attempt_file_path(state_dir: Path, job_id: str, attempt: int) -> Path:
    return state_dir / "attempts" / f"{job_id}__try{attempt}.json"


def launcher_file_path(state_dir: Path, job_id: str, attempt: int) -> Path:
    return state_dir / "launchers" / f"{job_id}__try{attempt}.sh"


def log_file_path(log_dir: Path, label: str, job_id: str, attempt: int) -> Path:
    return log_dir / label / f"{job_id}__try{attempt}.log"


def load_job(state_dir: Path, job_id: str) -> Dict[str, Any]:
    return load_json(job_file_path(state_dir, job_id))


def save_job(state_dir: Path, payload: Dict[str, Any]) -> None:
    write_json(job_file_path(state_dir, payload["job_id"]), payload)


def append_attempt_stub(state_dir: Path, job: JobSpec, attempt: int, running_entry: Dict[str, Any]) -> None:
    job_payload = load_job(state_dir, job.job_id)
    attempts = job_payload.setdefault("attempts", [])
    attempts.append(
        {
            "attempt": attempt,
            "status": "running",
            "gpu": running_entry["gpu"],
            "tmux_session": running_entry["tmux_session"],
            "tmux_window": running_entry["tmux_window"],
            "started_at": running_entry["started_at"],
            "log_path": running_entry["log_path"],
            "launcher_path": running_entry["launcher_path"],
        }
    )
    save_job(state_dir, job_payload)


def update_job_attempt(state_dir: Path, job_id: str, attempt_result: Dict[str, Any]) -> None:
    job_payload = load_job(state_dir, job_id)
    attempts = job_payload.setdefault("attempts", [])
    for item in attempts:
        if item.get("attempt") == attempt_result["attempt"]:
            item.update(attempt_result)
            break
    else:
        attempts.append(dict(attempt_result))
    job_payload["last_status"] = attempt_result["status"]
    save_job(state_dir, job_payload)


def validate_state_totals(state: Dict[str, Any]) -> None:
    total = int(state["total_jobs"])
    completed = len(state["completed"])
    pending = len(state["pending"])
    running = len(state["running"])
    failed = len(state["failed_final"])
    if completed + pending + running + failed != total:
        raise SchedulerError(
            f"State totals mismatch: completed={completed}, pending={pending}, running={running}, failed={failed}, total={total}"
        )


def build_progress_payload(state: Dict[str, Any]) -> Dict[str, Any]:
    total_jobs = int(state["total_jobs"])
    completed_jobs = len(state["completed"])
    running_jobs = len(state["running"])
    pending_jobs = len(state["pending"])
    failed_final_jobs = len(state["failed_final"])
    completion_ratio = (completed_jobs / total_jobs) if total_jobs else 1.0
    active_gpus = sorted({int(entry["gpu"]) for entry in state["running"].values()})
    return {
        "total_jobs": total_jobs,
        "completed_jobs": completed_jobs,
        "running_jobs": running_jobs,
        "pending_jobs": pending_jobs,
        "failed_final_jobs": failed_final_jobs,
        "completion_ratio": round(completion_ratio, 6),
        "retry_budget_total": int(state["retry_budget_total"]),
        "retry_budget_remaining": int(state["retry_budget_remaining"]),
        "active_gpus": active_gpus,
        "updated_at": now_iso(),
    }


def format_progress_line(progress: Dict[str, Any]) -> str:
    total = progress["total_jobs"]
    completed = progress["completed_jobs"]
    ratio = progress["completion_ratio"] * 100.0
    return (
        f"[progress] {completed}/{total} completed ({ratio:.1f}%) | "
        f"running={progress['running_jobs']} pending={progress['pending_jobs']} failed={progress['failed_final_jobs']} | "
        f"retry={progress['retry_budget_remaining']}/{progress['retry_budget_total']} | "
        f"active_gpus={','.join(str(gpu) for gpu in progress['active_gpus']) or '-'}"
    )


def persist_progress(state_dir: Path, state: Dict[str, Any], emit_stdout: bool = True) -> Dict[str, Any]:
    validate_state_totals(state)
    progress = build_progress_payload(state)
    write_json(progress_json_path(state_dir), progress)
    progress_line = format_progress_line(progress)
    progress_text_path(state_dir).write_text(progress_line + "\n", encoding="utf-8")
    if emit_stdout:
        print(progress_line)
    return progress


def save_state(state_dir: Path, state: Dict[str, Any], emit_progress: bool = True) -> None:
    validate_state_totals(state)
    write_json(state_file_path(state_dir), state)
    persist_progress(state_dir, state, emit_stdout=emit_progress)


def reserve_state_dir(state_dir: Path) -> None:
    if state_dir.exists():
        raise SchedulerError(f"State directory already exists; use --resume to continue: {state_dir}")
    (state_dir / "jobs").mkdir(parents=True, exist_ok=False)
    (state_dir / "attempts").mkdir(parents=True, exist_ok=True)
    (state_dir / "launchers").mkdir(parents=True, exist_ok=True)


def initialize_scheduler_state(
    plan: SchedulerPlan,
    state_dir: Path,
    plan_path: Path,
    pool_size_override: Optional[int] = None,
) -> Dict[str, Any]:
    reserve_state_dir(state_dir)
    jobs = expand_jobs(plan)
    retry_budget_total = compute_retry_budget(len(jobs), plan.retry.budget_ratio, plan.retry.rounding)
    for job in jobs:
        payload = job_to_payload(job)
        payload["attempts"] = []
        payload["last_status"] = "pending"
        save_job(state_dir, payload)

    snapshot_payload = {
        "version": plan.version,
        "label": plan.label,
        "pool_size": pool_size_override or plan.pool_size,
        "gpu": asdict(plan.gpu),
        "retry": asdict(plan.retry),
        "tmux": asdict(plan.tmux),
        "environment": asdict(plan.environment),
        "defaults": asdict(plan.defaults),
        "experiments": [asdict(item) for item in plan.experiments],
        "original_plan_path": str(plan_path.resolve()),
    }
    write_yaml(state_dir / "plan.snapshot.yaml", snapshot_payload)

    state = {
        "label": plan.label,
        "created_at": now_iso(),
        "pool_size": int(pool_size_override or plan.pool_size),
        "retry_budget_total": retry_budget_total,
        "retry_budget_remaining": retry_budget_total,
        "total_jobs": len(jobs),
        "pending": [job.job_id for job in jobs],
        "running": {},
        "completed": [],
        "failed_final": [],
    }
    save_state(state_dir, state)
    return state


def load_scheduler_snapshot(state_dir: Path) -> SchedulerPlan:
    snapshot = load_yaml(state_dir / "plan.snapshot.yaml")
    temp_path = state_dir / "__snapshot_parse__.yaml"
    write_yaml(temp_path, snapshot)
    try:
        return load_scheduler_plan(temp_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def load_scheduler_state(state_dir: Path) -> Dict[str, Any]:
    if not state_file_path(state_dir).exists():
        raise SchedulerError(f"Missing scheduler state file: {state_file_path(state_dir)}")
    state = load_json(state_file_path(state_dir))
    validate_state_totals(state)
    return state


def tmux_has_session(session_name: str) -> bool:
    proc = subprocess.run(["tmux", "has-session", "-t", session_name], capture_output=True, text=True)
    return proc.returncode == 0


def list_tmux_windows(session_name: str) -> List[str]:
    if not tmux_has_session(session_name):
        return []
    proc = subprocess.run(
        ["tmux", "list-windows", "-t", session_name, "-F", "#{window_name}"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def ensure_tmux_session(session_name: str) -> None:
    if tmux_has_session(session_name):
        return
    controller_cmd = "bash -lc 'while true; do sleep 3600; done'"
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", session_name, "-n", "__controller", controller_cmd],
        check=True,
    )


def tmux_window_exists(session_name: str, window_name: str) -> bool:
    return window_name in set(list_tmux_windows(session_name))


def cleanup_tmux_session_if_idle(session_name: str) -> bool:
    windows = list_tmux_windows(session_name)
    if not windows:
        return False

    non_controller_windows = [window for window in windows if window != "__controller"]
    if non_controller_windows:
        print(
            "[scheduler] Completed; tmux session "
            f"{session_name} kept because windows remain: {','.join(windows)}"
        )
        return False

    subprocess.run(["tmux", "kill-session", "-t", session_name], check=True)
    print(f"[scheduler] Completed; cleaned up tmux session: {session_name}")
    return True


def build_tmux_window_command(launcher_path: Path, log_path: Path) -> str:
    launcher = shlex.quote(str(launcher_path))
    log = shlex.quote(str(log_path))
    return f"bash -lc 'set -euo pipefail; {launcher} 2>&1 | tee {log}'"


def parse_visible_gpus_env(raw_value: Optional[str]) -> Optional[set[int]]:
    if raw_value is None:
        return None
    text = raw_value.strip()
    if not text or text.lower() == "all":
        return None

    visible_gpus: set[int] = set()
    for raw_item in text.split(","):
        item = raw_item.strip()
        if not item:
            raise SchedulerError(f"{VISIBLE_GPUS_ENV} contains an empty GPU id: {raw_value!r}")
        try:
            gpu_idx = int(item)
        except ValueError as exc:
            raise SchedulerError(f"{VISIBLE_GPUS_ENV} entries must be integer GPU ids: {raw_value!r}") from exc
        if gpu_idx < 0:
            raise SchedulerError(f"{VISIBLE_GPUS_ENV} entries must be non-negative GPU ids: {raw_value!r}")
        visible_gpus.add(gpu_idx)
    return visible_gpus


def filter_visible_gpus(observed_free_mib: Dict[int, int]) -> Dict[int, int]:
    visible_gpus = parse_visible_gpus_env(os.environ.get(VISIBLE_GPUS_ENV))
    if visible_gpus is None:
        return observed_free_mib

    filtered = {gpu: free_mib for gpu, free_mib in observed_free_mib.items() if gpu in visible_gpus}
    if not filtered:
        raise SchedulerError(
            f"{VISIBLE_GPUS_ENV}={os.environ.get(VISIBLE_GPUS_ENV)!r} did not match any GPUs reported by nvidia-smi"
        )
    return filtered


def collect_average_gpu_free_mib(sample_seconds: int) -> Dict[int, int]:
    free_sums: Dict[int, int] = {}
    counts: Dict[int, int] = {}
    for sample_idx in range(sample_seconds):
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
        )
        for raw_line in proc.stdout.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            gpu_str, free_str = [item.strip() for item in line.split(",", 1)]
            gpu_idx = int(gpu_str)
            free_mib = int(free_str)
            free_sums[gpu_idx] = free_sums.get(gpu_idx, 0) + free_mib
            counts[gpu_idx] = counts.get(gpu_idx, 0) + 1
        if sample_idx + 1 < sample_seconds:
            time.sleep(1)
    return {gpu: free_sums[gpu] // counts[gpu] for gpu in free_sums}


def select_gpu_for_dispatch(
    gpu_cfg: GPUConfig,
    provisional_reservations_by_gpu: Optional[Dict[int, int]] = None,
) -> Tuple[Optional[int], Dict[int, int], Dict[int, int]]:
    observed_free_mib = filter_visible_gpus(collect_average_gpu_free_mib(gpu_cfg.sample_seconds))
    reserve_per_job_mib = int(gpu_cfg.min_free_gib * 1024)
    provisional_reservations_by_gpu = dict(provisional_reservations_by_gpu or {})

    projected_free_mib = {
        gpu: free_mib - provisional_reservations_by_gpu.get(gpu, 0)
        for gpu, free_mib in observed_free_mib.items()
    }
    candidates = {
        gpu: projected_free_mib[gpu]
        for gpu in projected_free_mib
        if projected_free_mib[gpu] >= reserve_per_job_mib
    }
    if not candidates:
        return None, observed_free_mib, projected_free_mib
    selected_gpu = max(candidates.items(), key=lambda item: (item[1], -item[0]))[0]
    return selected_gpu, observed_free_mib, projected_free_mib


def validate_answers_file(run_dir: Path, dataset: str, repo_root: Path) -> Dict[str, Any]:
    config_path = run_dir / "config.yaml"
    answers_path = run_dir / "answers.jsonl"
    if not config_path.is_file():
        return {"ok": False, "reason": "missing_config"}
    if not answers_path.is_file():
        return {"ok": False, "reason": "missing_answers"}

    config = load_yaml(config_path)
    dataset_cfg = (config.get("datasets") or {}).get(dataset) or {}
    run_meta = config.get("_run_meta") or {}
    question_file = dataset_cfg.get("question_file")
    if not question_file:
        return {"ok": False, "reason": "missing_question_file"}

    if Path(question_file).is_absolute():
        question_path = Path(question_file).resolve()
    else:
        question_path = (repo_root / question_file).resolve()
    if not question_path.is_file():
        return {"ok": False, "reason": f"missing_question_source:{question_path}"}

    expected_lines = count_dataset_samples(dataset, question_path)
    max_samples = run_meta.get("max_samples")
    if isinstance(max_samples, int) and max_samples > 0:
        expected_lines = min(expected_lines, max_samples)
    valid_lines = 0
    try:
        with answers_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                json.loads(line)
                valid_lines += 1
    except json.JSONDecodeError as exc:
        return {
            "ok": False,
            "reason": f"malformed_answers_line:{exc.lineno or valid_lines + 1}",
            "expected_answers": expected_lines,
            "valid_answers": valid_lines,
        }

    if valid_lines != expected_lines:
        return {
            "ok": False,
            "reason": "incomplete_answers",
            "expected_answers": expected_lines,
            "valid_answers": valid_lines,
        }

    return {
        "ok": True,
        "reason": "complete",
        "expected_answers": expected_lines,
        "valid_answers": valid_lines,
    }


def find_run_candidates(run_prefix: str, started_at: float, runs_dir: Path) -> List[Path]:
    matches = find_run_dirs(runs_dir=runs_dir, name_prefix=run_prefix)
    return [
        path for path in matches
        if path.stat().st_mtime >= started_at - 1.0
    ]


def finalize_attempt_result(
    repo_root: Path,
    state_dir: Path,
    job_id: str,
    attempt: int,
    run_prefix: str,
    dataset: str,
    started_at: float,
    exit_code: int,
    log_path: str,
    gpu: int,
    tmux_session: str,
    tmux_window: str,
    runs_dir: Optional[Path] = None,
) -> Path:
    runs_dir = runs_dir or (repo_root / "entropy_exp" / "outputs" / "runs")
    finished_at = time.time()
    candidates = find_run_candidates(run_prefix=run_prefix, started_at=started_at, runs_dir=runs_dir)

    result: Dict[str, Any] = {
        "job_id": job_id,
        "attempt": attempt,
        "status": "failed_unknown",
        "dataset": dataset,
        "run_prefix": run_prefix,
        "gpu": gpu,
        "tmux_session": tmux_session,
        "tmux_window": tmux_window,
        "started_at": started_at,
        "finished_at": finished_at,
        "exit_code": exit_code,
        "log_path": log_path,
        "candidate_run_dirs": [str(path.resolve()) for path in candidates],
    }

    if exit_code != 0:
        result["status"] = "failed_process_exit"

    if not candidates:
        if exit_code == 0:
            result["status"] = "failed_missing_run_dir"
        result_path = attempt_file_path(state_dir, job_id, attempt)
        write_json(result_path, result)
        return result_path

    if len(candidates) > 1:
        result["status"] = "failed_multiple_run_dirs"
        result_path = attempt_file_path(state_dir, job_id, attempt)
        write_json(result_path, result)
        return result_path

    run_dir = candidates[0].resolve()
    result["run_dir"] = str(run_dir)
    validation = validate_answers_file(run_dir, dataset, repo_root)
    result["validation"] = validation
    if exit_code == 0 and validation["ok"]:
        result["status"] = "completed"
    elif exit_code == 0:
        result["status"] = f"failed_{validation['reason']}"

    result_path = attempt_file_path(state_dir, job_id, attempt)
    write_json(result_path, result)
    return result_path


def create_launcher_script(
    state_dir: Path,
    repo_root: Path,
    scheduler_script: Path,
    scheduler_python: Path,
    job: JobSpec,
    attempt: int,
    gpu: int,
    tmux_session: str,
    tmux_window: str,
    log_path: Path,
    env_cfg: EnvironmentConfig,
) -> Tuple[Path, float]:
    launcher_path = launcher_file_path(state_dir, job.job_id, attempt)
    started_at = time.time()
    command_line = shlex.join(build_run_command(job))
    conda_activate_target = resolve_conda_activate_target(env_cfg.conda_sh, env_cfg.conda_env)
    extra_env_lines = ""
    if env_cfg.extra_env:
        exports = []
        for assignment in env_cfg.extra_env:
            key, value = assignment.split("=", 1)
            exports.append(f"  export {key}={shlex.quote(value)}")
        extra_env_lines = "\n".join(exports) + "\n"
    content = f"""#!/usr/bin/env bash
set -u
JOB_EXIT_CODE=0

if [[ -f {shlex.quote(env_cfg.conda_sh)} ]]; then
  # shellcheck disable=SC1090
  source {shlex.quote(env_cfg.conda_sh)} || JOB_EXIT_CODE=$?
else
  JOB_EXIT_CODE=1
fi

if [[ "$JOB_EXIT_CODE" -eq 0 ]]; then
  conda activate {shlex.quote(conda_activate_target)} || JOB_EXIT_CODE=$?
fi

if [[ "$JOB_EXIT_CODE" -eq 0 ]]; then
  cd {shlex.quote(str(repo_root))} || JOB_EXIT_CODE=$?
fi

if [[ "$JOB_EXIT_CODE" -eq 0 ]]; then
{extra_env_lines}  export CUDA_VISIBLE_DEVICES={gpu}
  {command_line}
  JOB_EXIT_CODE=$?
fi

{shlex.quote(str(scheduler_python))} {shlex.quote(str(scheduler_script))} __finalize_attempt \\
  --state-dir {shlex.quote(str(state_dir))} \\
  --job-id {shlex.quote(job.job_id)} \\
  --attempt {attempt} \\
  --run-prefix {shlex.quote(job.run_prefix)} \\
  --dataset {shlex.quote(job.dataset)} \\
  --started-at {started_at:.6f} \\
  --exit-code "$JOB_EXIT_CODE" \\
  --log-path {shlex.quote(str(log_path))} \\
  --gpu {gpu} \\
  --tmux-session {shlex.quote(tmux_session)} \\
  --tmux-window {shlex.quote(tmux_window)} \\
  --runs-dir {shlex.quote(str(runs_dir_for_extra_sets(job.extra_sets, repo_root)))}

exit "$JOB_EXIT_CODE"
"""
    launcher_path.parent.mkdir(parents=True, exist_ok=True)
    launcher_path.write_text(content, encoding="utf-8")
    launcher_path.chmod(0o755)
    return launcher_path, started_at


class ExperimentScheduler:
    """Persistent FIFO scheduler for pruning jobs."""

    def __init__(
        self,
        repo_root: Path,
        plan: SchedulerPlan,
        state_dir: Path,
        state: Dict[str, Any],
        scheduler_script: Path,
        scheduler_python: Path,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.plan = plan
        self.state_dir = state_dir.resolve()
        self.state = state
        self.scheduler_script = scheduler_script.resolve()
        self.scheduler_python = scheduler_python.resolve()
        self.log_dir = resolve_repo_path(plan.tmux.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def create(
        cls,
        repo_root: Path,
        plan_path: Path,
        state_dir: Optional[Path],
        pool_size_override: Optional[int],
        scheduler_script: Path,
        scheduler_python: Path,
    ) -> "ExperimentScheduler":
        plan = load_scheduler_plan(plan_path)
        resolved_state_dir = state_dir or (DEFAULT_STATE_BASE_DIR / sanitize_name(plan.label))
        state = initialize_scheduler_state(plan, resolved_state_dir, plan_path, pool_size_override=pool_size_override)
        return cls(repo_root, plan, resolved_state_dir, state, scheduler_script, scheduler_python)

    @classmethod
    def resume(
        cls,
        repo_root: Path,
        state_dir: Path,
        scheduler_script: Path,
        scheduler_python: Path,
        pool_size_override: Optional[int] = None,
    ) -> "ExperimentScheduler":
        plan = load_scheduler_snapshot(state_dir)
        state = load_scheduler_state(state_dir)
        if pool_size_override is not None:
            state["pool_size"] = int(pool_size_override)
        scheduler = cls(repo_root, plan, state_dir, state, scheduler_script, scheduler_python)
        scheduler.reconcile_running_jobs()
        save_state(state_dir, scheduler.state)
        return scheduler

    def emit_progress(self) -> None:
        persist_progress(self.state_dir, self.state, emit_stdout=True)

    def reconcile_running_jobs(self) -> None:
        changed = False
        for job_id, running_entry in list(self.state["running"].items()):
            attempt = int(running_entry["attempt"])
            attempt_path = attempt_file_path(self.state_dir, job_id, attempt)
            if attempt_path.exists():
                attempt_result = load_json(attempt_path)
                self._resolve_finished_attempt(job_id, attempt_result)
                changed = True
                continue
            if not tmux_window_exists(running_entry["tmux_session"], running_entry["tmux_window"]):
                failure_result = {
                    "job_id": job_id,
                    "attempt": attempt,
                    "status": "failed_missing_attempt_result",
                    "dataset": running_entry["dataset"],
                    "run_prefix": running_entry["run_prefix"],
                    "gpu": running_entry["gpu"],
                    "tmux_session": running_entry["tmux_session"],
                    "tmux_window": running_entry["tmux_window"],
                    "started_at": running_entry["started_at"],
                    "finished_at": time.time(),
                    "exit_code": None,
                    "log_path": running_entry["log_path"],
                }
                write_json(attempt_path, failure_result)
                self._resolve_finished_attempt(job_id, failure_result)
                changed = True
        if changed:
            save_state(self.state_dir, self.state)

    def _resolve_finished_attempt(self, job_id: str, attempt_result: Dict[str, Any]) -> None:
        running_entry = self.state["running"].pop(job_id, None)
        update_job_attempt(self.state_dir, job_id, attempt_result)
        if running_entry is None:
            return

        if attempt_result["status"] == "completed":
            self.state["completed"].append(job_id)
            return

        if self.state["retry_budget_remaining"] > 0:
            self.state["retry_budget_remaining"] -= 1
            self.state["pending"].append(job_id)
        else:
            self.state["failed_final"].append(job_id)

    def _get_job(self, job_id: str) -> JobSpec:
        payload = load_job(self.state_dir, job_id)
        return JobSpec(
            job_id=payload["job_id"],
            experiment_name=payload["experiment_name"],
            dataset=payload["dataset"],
            strategy=payload["strategy"],
            max_samples=payload.get("max_samples"),
            extra_sets=list(payload.get("extra_sets", [])),
            run_prefix=payload["run_prefix"],
            window_base_name=payload["window_base_name"],
        )

    def _launch_job(self, job_id: str, selected_gpu: int) -> None:
        job = self._get_job(job_id)
        attempt = len(load_job(self.state_dir, job_id).get("attempts", [])) + 1
        window_name = sanitize_name(f"{job.window_base_name}_try{attempt}")
        log_path = log_file_path(self.log_dir, self.plan.label, job.job_id, attempt)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        launcher_path, started_at = create_launcher_script(
            state_dir=self.state_dir,
            repo_root=self.repo_root,
            scheduler_script=self.scheduler_script,
            scheduler_python=self.scheduler_python,
            job=job,
            attempt=attempt,
            gpu=selected_gpu,
            tmux_session=self.plan.tmux.session_name,
            tmux_window=window_name,
            log_path=log_path,
            env_cfg=self.plan.environment,
        )

        ensure_tmux_session(self.plan.tmux.session_name)
        window_command = build_tmux_window_command(launcher_path, log_path)
        subprocess.run(
            ["tmux", "new-window", "-d", "-t", self.plan.tmux.session_name, "-n", window_name, window_command],
            check=True,
        )

        running_entry = {
            "attempt": attempt,
            "gpu": selected_gpu,
            "dataset": job.dataset,
            "run_prefix": job.run_prefix,
            "runs_dir": str(runs_dir_for_extra_sets(job.extra_sets, self.repo_root)),
            "tmux_session": self.plan.tmux.session_name,
            "tmux_window": window_name,
            "started_at": started_at,
            "log_path": str(log_path),
            "launcher_path": str(launcher_path),
        }
        self.state["running"][job.job_id] = running_entry
        append_attempt_stub(self.state_dir, job, attempt, running_entry)

    def dispatch_available_jobs(self) -> None:
        provisional_reservations_by_gpu: Dict[int, int] = {}
        reserve_per_job_mib = int(self.plan.gpu.min_free_gib * 1024)
        while self.state["pending"] and len(self.state["running"]) < int(self.state["pool_size"]):
            selected_gpu, observed_free, projected_free = select_gpu_for_dispatch(
                self.plan.gpu,
                provisional_reservations_by_gpu=provisional_reservations_by_gpu,
            )
            print(f"[scheduler] GPU observed free memory (MiB): {observed_free}")
            print(
                "[scheduler] GPU projected free memory after provisional reservations (MiB): "
                f"{projected_free}"
            )
            if selected_gpu is None:
                print("[scheduler] No GPU currently satisfies the 16 GiB projected-free threshold this pass.")
                break

            job_id = self.state["pending"].pop(0)
            self._launch_job(job_id, selected_gpu)
            provisional_reservations_by_gpu[selected_gpu] = (
                provisional_reservations_by_gpu.get(selected_gpu, 0) + reserve_per_job_mib
            )
            save_state(self.state_dir, self.state)

    def run(self) -> None:
        self.emit_progress()
        while self.state["pending"] or self.state["running"]:
            self.reconcile_running_jobs()
            self.dispatch_available_jobs()
            persist_progress(self.state_dir, self.state, emit_stdout=True)
            if not self.state["pending"] and not self.state["running"]:
                break
            time.sleep(self.plan.gpu.poll_interval_seconds)
        self.cleanup_tmux_session_if_complete()

    def cleanup_tmux_session_if_complete(self) -> None:
        if not self.state["pending"] and not self.state["running"]:
            cleanup_tmux_session_if_idle(self.plan.tmux.session_name)


def scheduler_dry_run(
    repo_root: Path,
    plan: SchedulerPlan,
    pool_size_override: Optional[int] = None,
) -> Dict[str, Any]:
    jobs = expand_jobs(plan)
    effective_pool_size = int(pool_size_override or plan.pool_size)
    retry_budget_total = compute_retry_budget(len(jobs), plan.retry.budget_ratio, plan.retry.rounding)
    payload = {
        "label": plan.label,
        "repo_root": str(repo_root.resolve()),
        "pool_size": effective_pool_size,
        "retry_budget_total": retry_budget_total,
        "total_jobs": len(jobs),
        "jobs": [job_to_payload(job) for job in jobs],
    }
    print(f"Plan label: {plan.label}")
    print(f"Repository root: {repo_root.resolve()}")
    print(f"Pool size: {effective_pool_size}")
    print(f"Retry budget: {retry_budget_total}")
    print(f"Planned jobs: {len(jobs)}")
    for job in jobs:
        print(f"- {job.job_id} | {job.experiment_name} | {job.dataset} | {job.strategy}")
        print(f"  {' '.join(shlex.quote(part) for part in build_run_command(job))}")
    return payload
