#!/usr/bin/env python3
"""Collect metadata and local eval metrics from an official SparseVLM run."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


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
        commit_marker = repo.parent / f"{repo.name}_FETCHED_COMMIT.txt"
        if commit_marker.is_file():
            return commit_marker.read_text(encoding="utf-8").strip() or None
        return None
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-config")
    parser.add_argument("--official-repo")
    parser.add_argument("--method")
    parser.add_argument("--variant")
    parser.add_argument("--retain-token")
    parser.add_argument("--use-version")
    parser.add_argument("--model-path")
    parser.add_argument("--model-name")
    parser.add_argument("--question-file")
    parser.add_argument("--image-folder")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    eval_summary = run_dir / "eval" / "summary.json"
    answers_file = run_dir / "answers.jsonl"
    run_config = load_yaml(Path(args.run_config).resolve()) if args.run_config else load_yaml(run_dir / "run_config.yaml")
    official_repo_value = (
        args.official_repo
        or run_config.get("official_repo")
        or run_config.get("official", {}).get("repo")
        or run_dir.parents[1] / "third_party" / "SparseVLMs"
    )
    official_repo = Path(official_repo_value).resolve()

    metrics = {}
    eval_summary_payload = {}
    if eval_summary.is_file():
        eval_summary_payload = load_json(eval_summary)
        metrics = eval_summary_payload.get("metrics", {}) or {}
    metric_name, metric_value = primary_metric(args.dataset, metrics)

    model_cfg = run_config.get("model", {}) or {}
    dataset_paths = run_config.get("dataset_paths", {}) or {}
    official_env = run_config.get("official_env", {}) or {}
    fallback_env = {
        "USE_VERSION": args.use_version or os.environ.get("USE_VERSION"),
        "RETAIN_TOKN": args.retain_token or os.environ.get("RETAIN_TOKN"),
    }
    command_file = run_dir / "command.sh"
    command = run_config.get("command_shell") or (
        command_file.read_text(encoding="utf-8").strip() if command_file.is_file() else None
    )

    record = {
        "label": run_dir.name,
        "source_type": "official",
        "method": args.method or run_config.get("method") or "sparsevlm",
        "variant": args.variant or run_config.get("variant"),
        "dataset": args.dataset,
        "model_path": args.model_path or model_cfg.get("path"),
        "model_name": args.model_name or model_cfg.get("name"),
        "question_file": args.question_file or dataset_paths.get("question_file"),
        "image_folder": args.image_folder or dataset_paths.get("image_folder"),
        "run_dir": str(run_dir),
        "answers_file": str(answers_file) if answers_file.exists() else None,
        "eval_summary": str(eval_summary) if eval_summary.exists() else None,
        "official_repo_commit": git_commit(official_repo),
        "official_repo": str(official_repo),
        "command": command,
        "cwd": run_config.get("official_repo") or str(official_repo),
        "env_vars": official_env or fallback_env,
        "use_version": args.use_version or official_env.get("USE_VERSION") or run_config.get("use_version") or os.environ.get("USE_VERSION"),
        "retain_token": args.retain_token or official_env.get("RETAIN_TOKN") or run_config.get("retain_token") or os.environ.get("RETAIN_TOKN"),
        "method_params": run_config.get("method_params") or {},
        "metric_name": metric_name,
        "metric_value": metric_value,
        "metrics": metrics,
        "eval_dataset": eval_summary_payload.get("dataset"),
        "collected_at": datetime.now().isoformat(timespec="seconds"),
    }

    summary_json = run_dir / "official_summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2, sort_keys=True)

    summary_csv = run_dir / "official_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(record.keys()))
        writer.writeheader()
        writer.writerow({key: csv_value(value) for key, value in record.items()})

    print(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
