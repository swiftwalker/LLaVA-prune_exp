#!/usr/bin/env python3
"""Collect lightweight metadata from an official SparseVLM run directory."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def primary_metric(dataset: str, metrics: dict[str, Any]) -> tuple[str | None, Any]:
    if dataset == "mme":
        return "overall_total_score", metrics.get("overall_total_score")
    if dataset in {"gqa", "textvqa", "scienceqa"}:
        return "accuracy", metrics.get("accuracy")
    if dataset == "pope":
        return "macro_f1", metrics.get("macro_f1")
    return None, None


def git_commit(repo: Path) -> str | None:
    if not (repo / ".git").exists():
        return None
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--official-repo")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    eval_summary = run_dir / "eval" / "summary.json"
    answers_file = run_dir / "answers.jsonl"
    official_repo = Path(args.official_repo).resolve() if args.official_repo else run_dir.parents[1] / "third_party" / "SparseVLMs"

    metrics = {}
    if eval_summary.is_file():
        metrics = load_json(eval_summary).get("metrics", {}) or {}
    metric_name, metric_value = primary_metric(args.dataset, metrics)

    record = {
        "label": run_dir.name,
        "source_type": "official",
        "method": "sparsevlm",
        "dataset": args.dataset,
        "run_dir": str(run_dir),
        "answers_file": str(answers_file) if answers_file.exists() else None,
        "eval_summary": str(eval_summary) if eval_summary.exists() else None,
        "official_repo_commit": git_commit(official_repo),
        "use_version": os.environ.get("USE_VERSION"),
        "retain_token": os.environ.get("RETAIN_TOKN"),
        "metric_name": metric_name,
        "metric_value": metric_value,
        "collected_at": datetime.now().isoformat(timespec="seconds"),
    }

    summary_json = run_dir / "official_summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2, sort_keys=True)

    summary_csv = run_dir / "official_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(record.keys()))
        writer.writeheader()
        writer.writerow(record)

    print(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
