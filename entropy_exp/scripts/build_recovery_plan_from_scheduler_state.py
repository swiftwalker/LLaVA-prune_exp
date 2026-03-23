#!/usr/bin/env python3
"""Build a recovery scheduler plan from a previous scheduler state."""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

import yaml


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def dump_yaml(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a recovery scheduler plan from failed_final jobs.")
    parser.add_argument("--state-dir", required=True, help="Existing scheduler state directory")
    parser.add_argument("--output", required=True, help="Output plan YAML path")
    parser.add_argument("--label", required=True, help="Label for the recovery plan")
    parser.add_argument("--tmux-session", required=True, help="tmux session name for the recovery plan")
    parser.add_argument("--pool-size", type=int, help="Optional pool size override to write into the recovery plan")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    state_dir = Path(args.state_dir).resolve()
    snapshot_path = state_dir / "plan.snapshot.yaml"
    state_path = state_dir / "state.json"
    jobs_dir = state_dir / "jobs"

    if not snapshot_path.is_file():
        raise FileNotFoundError(f"Missing snapshot plan: {snapshot_path}")
    if not state_path.is_file():
        raise FileNotFoundError(f"Missing scheduler state: {state_path}")
    if not jobs_dir.is_dir():
        raise FileNotFoundError(f"Missing jobs directory: {jobs_dir}")

    snapshot = load_yaml(snapshot_path)
    state = load_json(state_path)
    failed_jobs = set(state.get("failed_final", []))
    if not failed_jobs:
        raise RuntimeError("No failed_final jobs found; recovery plan would be empty.")

    grouped = OrderedDict()
    for job_path in sorted(jobs_dir.glob("job_*.json")):
        job = load_json(job_path)
        job_id = job["job_id"]
        if job_id not in failed_jobs:
            continue

        experiment_name = job["experiment_name"]
        payload = grouped.setdefault(
            experiment_name,
            {
                "name": experiment_name,
                "dataset": job["dataset"],
                "strategies": [],
                "extra_sets": list(job.get("extra_sets", [])),
                "max_samples": job.get("max_samples"),
            },
        )
        payload["strategies"].append(job["strategy"])

    recovery_plan = {
        "version": snapshot["version"],
        "label": args.label,
        "pool_size": args.pool_size or snapshot["pool_size"],
        "gpu": snapshot["gpu"],
        "retry": snapshot["retry"],
        "tmux": {
            **snapshot["tmux"],
            "session_name": args.tmux_session,
        },
        "environment": snapshot["environment"],
        "defaults": snapshot["defaults"],
        "experiments": list(grouped.values()),
    }

    dump_yaml(Path(args.output).resolve(), recovery_plan)

    total_jobs = sum(len(item["strategies"]) for item in recovery_plan["experiments"])
    print(
        json.dumps(
            {
                "output": str(Path(args.output).resolve()),
                "experiments": len(recovery_plan["experiments"]),
                "jobs": total_jobs,
                "label": recovery_plan["label"],
                "tmux_session": recovery_plan["tmux"]["session_name"],
                "pool_size": recovery_plan["pool_size"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
