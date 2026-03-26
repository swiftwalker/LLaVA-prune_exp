#!/usr/bin/env python3
"""List completed run directories from a scheduler state directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List completed run directories from scheduler attempts.")
    parser.add_argument("--state-dir", required=True, help="Scheduler state directory")
    parser.add_argument(
        "--status",
        default="completed",
        help="Attempt status to select from the latest attempt of each job",
    )
    parser.add_argument(
        "--format",
        choices=["lines", "json"],
        default="lines",
        help="Output format",
    )
    parser.add_argument(
        "--existing-only",
        action="store_true",
        help="Only emit run directories that still exist on disk",
    )
    return parser.parse_args()


def load_latest_attempts(state_dir: Path) -> list[dict[str, Any]]:
    attempts_dir = state_dir / "attempts"
    if not attempts_dir.is_dir():
        raise FileNotFoundError(f"Missing attempts directory: {attempts_dir}")

    latest_by_job: dict[str, dict[str, Any]] = {}
    for attempt_path in sorted(attempts_dir.glob("job_*__try*.json")):
        payload = load_json(attempt_path)
        job_id = str(payload.get("job_id", "")).strip()
        attempt = int(payload.get("attempt", 0))
        if not job_id:
            continue

        previous = latest_by_job.get(job_id)
        if previous is None or attempt > int(previous.get("attempt", 0)):
            latest_by_job[job_id] = payload

    return [latest_by_job[job_id] for job_id in sorted(latest_by_job)]


def main() -> int:
    args = parse_args()
    state_dir = Path(args.state_dir).resolve()

    selected_attempts: list[dict[str, Any]] = []
    for payload in load_latest_attempts(state_dir):
        if payload.get("status") != args.status:
            continue
        run_dir = payload.get("run_dir")
        if not run_dir:
            continue
        run_path = Path(run_dir).resolve()
        if args.existing_only and not run_path.is_dir():
            continue

        selected_attempts.append(
            {
                "job_id": payload.get("job_id"),
                "attempt": payload.get("attempt"),
                "status": payload.get("status"),
                "run_dir": str(run_path),
            }
        )

    if args.format == "json":
        print(json.dumps(selected_attempts, indent=2, ensure_ascii=False))
        return 0

    for payload in selected_attempts:
        print(payload["run_dir"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
