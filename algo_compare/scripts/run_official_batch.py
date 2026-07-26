#!/usr/bin/env python3
"""Run a resumable, GPU-balanced batch of official comparison methods."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Job:
    job_id: str
    method: str
    variant: str
    budget: str
    dataset: str
    expected_samples: int
    estimated_weight: float
    output_dir: str
    command: list[str]
    status: str = "pending"
    attempts: int = 0
    gpu: int | None = None
    return_code: int | None = None
    error: str | None = None


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_plan(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        plan = yaml.safe_load(handle)
    if not isinstance(plan, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return plan


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def build_jobs(plan: dict[str, Any]) -> list[Job]:
    python_bin = str(resolve_path(plan["python_bin"]))
    runner = REPO_ROOT / "algo_compare" / "scripts" / "run_official.py"
    model_path = str(resolve_path(plan["model_path"]))
    output_root = resolve_path(plan["output_root"])
    model_name = str(plan["model_name"])
    conv_mode = str(plan.get("conv_mode", "vicuna_v1"))
    jobs: list[Job] = []

    for method, method_cfg in plan["methods"].items():
        method_weight = float(method_cfg.get("weight", 1.0))
        for budget, budget_cfg in plan["budgets"].items():
            budget_weight = float(budget_cfg.get("weight", 1.0))
            for dataset_cfg in plan["datasets"]:
                dataset = str(dataset_cfg["name"])
                expected_samples = int(dataset_cfg["samples"])
                max_samples_value = dataset_cfg.get("max_samples", plan.get("max_samples"))
                max_samples = None if max_samples_value is None else int(max_samples_value)
                if max_samples is not None:
                    if max_samples <= 0:
                        raise ValueError(f"max_samples must be positive, got {max_samples}")
                    expected_samples = min(expected_samples, max_samples)
                output_dir = output_root / method / budget / dataset
                command = [
                    python_bin,
                    str(runner),
                    "--method",
                    method,
                    "--dataset",
                    dataset,
                    "--variant",
                    str(method_cfg["variant"]),
                    "--retain-token",
                    str(int(budget_cfg["one_shot_tokens"])),
                    "--model-path",
                    model_path,
                    "--model-name",
                    model_name,
                    "--conv-mode",
                    conv_mode,
                    "--python-bin",
                    python_bin,
                    "--output-dir",
                    str(output_dir),
                    "--temperature",
                    "0",
                    "--eval",
                ]
                if max_samples is not None:
                    command.extend(["--max-samples", str(max_samples)])
                if method == "divprune":
                    command.extend(
                        [
                            "--divprune-subset-ratio",
                            str(budget_cfg["divprune_subset_ratio"]),
                            "--divprune-visual-token-count",
                            "576",
                        ]
                    )
                elif method == "cdpruner":
                    command.extend(
                        [
                            "--cdpruner-llava-next-compat",
                            str(method_cfg.get("llava_next_compat", "auto")),
                        ]
                    )
                    if bool(method_cfg.get("padding_diagnostics", False)):
                        command.append("--cdpruner-padding-diagnostics")
                job_id = f"{method}_{budget}_{dataset}"
                jobs.append(
                    Job(
                        job_id=job_id,
                        method=method,
                        variant=str(method_cfg["variant"]),
                        budget=budget,
                        dataset=dataset,
                        expected_samples=expected_samples,
                        estimated_weight=expected_samples * method_weight * budget_weight,
                        output_dir=str(output_dir),
                        command=command,
                    )
                )
    return sorted(jobs, key=lambda item: (-item.estimated_weight, item.job_id))


def answer_count(job: Job) -> int:
    answers = Path(job.output_dir) / "answers.jsonl"
    if not answers.is_file():
        return 0
    with answers.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def validate_output(job: Job) -> tuple[bool, str | None]:
    count = answer_count(job)
    if count != job.expected_samples:
        return False, f"answers={count}, expected={job.expected_samples}"
    eval_summary = Path(job.output_dir) / "eval" / "summary.json"
    if not eval_summary.is_file():
        return False, "missing eval/summary.json"
    official_summary = Path(job.output_dir) / "official_summary.json"
    if not official_summary.is_file():
        return False, "missing official_summary.json"
    return True, None


def write_state(path: Path, label: str, jobs: list[Job], started_at: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "label": label,
        "started_at": started_at,
        "updated_at": now(),
        "counts": {
            status: sum(job.status == status for job in jobs)
            for status in ("pending", "running", "completed", "failed")
        },
        "jobs": [asdict(job) for job in jobs],
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def restore_state(path: Path, jobs: list[Job]) -> None:
    previous: dict[str, Any] = {}
    if path.is_file():
        with path.open("r", encoding="utf-8") as handle:
            previous = {item["job_id"]: item for item in json.load(handle).get("jobs", [])}
    for job in jobs:
        valid, _ = validate_output(job)
        if valid:
            job.status = "completed"
            continue
        old = previous.get(job.job_id, {})
        job.attempts = int(old.get("attempts", 0))
        job.status = "pending"


def wait_for_upstream(plan: dict[str, Any]) -> None:
    wait_cfg = plan.get("wait_for")
    if not wait_cfg:
        return
    state_path = resolve_path(wait_cfg["state_file"])
    expected_jobs = int(wait_cfg["expected_jobs"])
    poll_seconds = int(wait_cfg.get("poll_interval_seconds", 60))
    last_report = 0.0
    while True:
        if state_path.is_file():
            try:
                with state_path.open("r", encoding="utf-8") as handle:
                    state = json.load(handle)
                completed = len(state.get("completed", []))
                pending = len(state.get("pending", []))
                running = len(state.get("running", {}))
                failed = len(state.get("failed_final", []))
                if failed:
                    raise RuntimeError(f"Upstream batch has {failed} final failures: {state_path}")
                if completed == expected_jobs and pending == 0 and running == 0:
                    print(f"[{now()}] upstream complete: {completed}/{expected_jobs}", flush=True)
                    return
                if time.time() - last_report >= 60:
                    print(
                        f"[{now()}] waiting upstream: completed={completed}/{expected_jobs} "
                        f"running={running} pending={pending}",
                        flush=True,
                    )
                    last_report = time.time()
            except json.JSONDecodeError:
                pass
        elif time.time() - last_report >= 60:
            print(f"[{now()}] waiting for upstream state: {state_path}", flush=True)
            last_report = time.time()
        time.sleep(poll_seconds)


def gpu_free_gib() -> dict[int, float]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    free: dict[int, float] = {}
    for line in result.stdout.splitlines():
        index, memory_mib = (part.strip() for part in line.split(",", 1))
        free[int(index)] = float(memory_mib) / 1024.0
    return free


def dry_run(plan: dict[str, Any], jobs: list[Job]) -> None:
    print(f"label={plan['label']} jobs={len(jobs)}")
    print(
        f"gpus={plan['gpus']} max_concurrent_per_gpu={plan['max_concurrent_per_gpu']} "
        f"output_root={resolve_path(plan['output_root'])}"
    )
    for job in jobs:
        print(f"{job.job_id}\tweight={job.estimated_weight:.1f}\t{' '.join(job.command)}")


def run_batch(plan: dict[str, Any], jobs: list[Job]) -> int:
    label = str(plan["label"])
    state_path = resolve_path(plan["state_file"])
    log_root = resolve_path(plan["log_dir"])
    log_root.mkdir(parents=True, exist_ok=True)
    restore_state(state_path, jobs)
    started_at = now()
    write_state(state_path, label, jobs, started_at)

    gpus = [int(gpu) for gpu in plan["gpus"]]
    max_per_gpu = int(plan["max_concurrent_per_gpu"])
    min_free_gib = float(plan.get("min_free_gib", 35))
    retry_limit = int(plan.get("retry_limit", 1))
    poll_seconds = int(plan.get("poll_interval_seconds", 10))
    launch_stagger = float(plan.get("launch_stagger_seconds", 5))
    running: dict[int, tuple[subprocess.Popen[Any], Job, int, Any]] = {}

    base_env = os.environ.copy()
    for item in plan.get("environment", []):
        key, value = str(item).split("=", 1)
        base_env[key] = value

    while True:
        for pid, (process, job, gpu, log_handle) in list(running.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            log_handle.close()
            del running[pid]
            job.return_code = return_code
            valid, error = validate_output(job)
            if return_code == 0 and valid:
                job.status = "completed"
                job.error = None
                print(f"[{now()}] completed {job.job_id} on GPU{gpu}", flush=True)
            elif job.attempts <= retry_limit:
                job.status = "pending"
                job.error = error or f"return_code={return_code}"
                print(
                    f"[{now()}] retrying {job.job_id}: attempt={job.attempts} "
                    f"return_code={return_code} error={job.error}",
                    flush=True,
                )
            else:
                job.status = "failed"
                job.error = error or f"return_code={return_code}"
                print(f"[{now()}] failed {job.job_id}: {job.error}", flush=True)
            job.gpu = gpu
            write_state(state_path, label, jobs, started_at)

        pending = [job for job in jobs if job.status == "pending"]
        if not pending and not running:
            break

        running_per_gpu = {
            gpu: sum(running_job_gpu == gpu for _, _, running_job_gpu, _ in running.values())
            for gpu in gpus
        }
        free = gpu_free_gib()
        candidates = [
            gpu
            for gpu in gpus
            if running_per_gpu[gpu] < max_per_gpu and free.get(gpu, 0.0) >= min_free_gib
        ]
        if pending and candidates:
            gpu = min(candidates, key=lambda item: (running_per_gpu[item], -free.get(item, 0.0), item))
            job = pending[0]
            job.attempts += 1
            job.status = "running"
            job.gpu = gpu
            job.error = None
            log_path = log_root / f"{job.job_id}.log"
            log_handle = log_path.open("a", encoding="utf-8")
            log_handle.write(f"\n[{now()}] attempt={job.attempts} gpu={gpu}\n")
            log_handle.flush()
            env = base_env.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            process = subprocess.Popen(
                job.command,
                cwd=REPO_ROOT,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            running[process.pid] = (process, job, gpu, log_handle)
            print(
                f"[{now()}] launched {job.job_id} attempt={job.attempts} "
                f"gpu={gpu} free_gib={free[gpu]:.1f}",
                flush=True,
            )
            write_state(state_path, label, jobs, started_at)
            time.sleep(launch_stagger)
            continue
        time.sleep(poll_seconds)

    write_state(state_path, label, jobs, started_at)
    failed = [job.job_id for job in jobs if job.status == "failed"]
    print(f"[{now()}] batch complete: completed={len(jobs) - len(failed)}/{len(jobs)} failed={failed}")
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-wait", action="store_true")
    args = parser.parse_args()

    plan = load_plan(args.plan)
    jobs = build_jobs(plan)
    expected = int(plan.get("expected_jobs", len(jobs)))
    if len(jobs) != expected:
        raise ValueError(f"Plan expanded to {len(jobs)} jobs; expected {expected}")
    if args.dry_run:
        dry_run(plan, jobs)
        return 0
    if not args.skip_wait:
        wait_for_upstream(plan)
    return run_batch(plan, jobs)


if __name__ == "__main__":
    raise SystemExit(main())
