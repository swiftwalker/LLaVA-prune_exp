#!/usr/bin/env python3
"""Detect and optionally delete incomplete experiment run directories."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_DIR = REPO_ROOT / "entropy_exp" / "outputs" / "runs"


@dataclass
class RunReport:
    run_dir: Path
    dataset: str | None
    status: str
    valid_answers: int
    total_lines: int
    expected_answers: int | None
    first_bad_line: int | None
    reasons: list[str]

    @property
    def is_problematic(self) -> bool:
        return self.status != "complete"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect incomplete or malformed run directories under entropy_exp/outputs/runs."
    )
    parser.add_argument(
        "--mode",
        choices=["preview", "delete"],
        default="preview",
        help="Preview problematic runs or delete them.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help="Directory containing run folders. Defaults to entropy_exp/outputs/runs.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Optional dataset filter, for example gqa, mme, or pope.",
    )
    parser.add_argument(
        "--name-prefix",
        type=str,
        help="Optional run directory name prefix filter.",
    )
    return parser.parse_args()


def resolve_repo_path(path_str: str | None) -> Path | None:
    if not path_str:
        return None
    path = Path(path_str)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def parse_yaml_scalar(line: str) -> str:
    return line.split(":", 1)[1].strip().strip("'\"")


def parse_config_metadata(config_path: Path) -> tuple[str | None, dict[str, str]]:
    dataset = None
    question_files: dict[str, str] = {}

    in_run_meta = False
    run_meta_indent = -1
    in_datasets = False
    datasets_indent = -1
    current_dataset = None
    current_dataset_indent = -1

    with config_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            indent = len(raw_line) - len(raw_line.lstrip(" "))

            if in_run_meta:
                if indent <= run_meta_indent:
                    in_run_meta = False
                elif stripped.startswith("dataset:"):
                    dataset = parse_yaml_scalar(stripped)
                    continue

            if in_datasets:
                if indent <= datasets_indent:
                    in_datasets = False
                    current_dataset = None
                else:
                    if stripped.endswith(":") and indent == datasets_indent + 2:
                        current_dataset = stripped[:-1]
                        current_dataset_indent = indent
                        continue
                    if current_dataset is not None and indent <= current_dataset_indent:
                        current_dataset = None
                    if current_dataset is not None and stripped.startswith("question_file:"):
                        question_files[current_dataset] = parse_yaml_scalar(stripped)
                        continue

            if stripped == "_run_meta:":
                in_run_meta = True
                run_meta_indent = indent
                continue

            if stripped == "datasets:":
                in_datasets = True
                datasets_indent = indent
                current_dataset = None
                continue

    return dataset, question_files


def count_lines(path: Path, cache: dict[Path, int]) -> int:
    path = path.resolve()
    if path not in cache:
        with path.open("r", encoding="utf-8") as f:
            cache[path] = sum(1 for _ in f)
    return cache[path]


def discover_run_dirs(runs_dir: Path, dataset: str | None, name_prefix: str | None) -> list[Path]:
    run_dirs: list[Path] = []
    for path in sorted(runs_dir.iterdir()):
        if not path.is_dir():
            continue
        if name_prefix and not path.name.startswith(name_prefix):
            continue
        if not ((path / "config.yaml").exists() or (path / "answers.jsonl").exists()):
            continue
        if dataset and not path.name.startswith(f"{dataset}_"):
            continue
        run_dirs.append(path)
    return run_dirs


def analyze_answers(answers_path: Path) -> tuple[int, int, int | None]:
    valid_answers = 0
    total_lines = 0
    first_bad_line = None

    with answers_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            total_lines += 1
            try:
                json.loads(line)
            except json.JSONDecodeError:
                if first_bad_line is None:
                    first_bad_line = line_no
                continue
            valid_answers += 1

    return valid_answers, total_lines, first_bad_line


def analyze_run(run_dir: Path, question_count_cache: dict[Path, int]) -> RunReport:
    config_path = run_dir / "config.yaml"
    answers_path = run_dir / "answers.jsonl"

    dataset = None
    expected_answers = None
    reasons: list[str] = []

    if not config_path.exists():
        reasons.append("missing config.yaml")
    else:
        try:
            dataset, question_files = parse_config_metadata(config_path)
        except Exception as exc:
            question_files = {}
            reasons.append(f"config load failed: {exc}")
    if not config_path.exists():
        question_files = {}

    if dataset is None:
        reasons.append("dataset missing in _run_meta")

    question_file = None
    if dataset:
        question_file = resolve_repo_path(question_files.get(dataset))
        if question_file is None:
            reasons.append(f"question_file missing for dataset={dataset}")
        elif not question_file.exists():
            reasons.append(f"question_file not found: {question_file}")
        else:
            expected_answers = count_lines(question_file, question_count_cache)

    if not answers_path.exists():
        reasons.append("missing answers.jsonl")
        valid_answers = 0
        total_lines = 0
        first_bad_line = None
    else:
        valid_answers, total_lines, first_bad_line = analyze_answers(answers_path)
        if first_bad_line is not None:
            reasons.append(f"malformed JSONL at line {first_bad_line}")

    if expected_answers is not None and valid_answers != expected_answers:
        reasons.append(f"valid answers {valid_answers}/{expected_answers}")

    if first_bad_line is not None and expected_answers is not None and valid_answers != expected_answers:
        status = "malformed_and_incomplete"
    elif first_bad_line is not None:
        status = "malformed"
    elif expected_answers is not None and valid_answers != expected_answers:
        status = "incomplete"
    elif reasons:
        status = "invalid"
    else:
        status = "complete"

    return RunReport(
        run_dir=run_dir,
        dataset=dataset,
        status=status,
        valid_answers=valid_answers,
        total_lines=total_lines,
        expected_answers=expected_answers,
        first_bad_line=first_bad_line,
        reasons=reasons,
    )


def print_preview(reports: list[RunReport], runs_dir: Path, mode: str) -> None:
    problematic = [report for report in reports if report.is_problematic]
    print(f"Runs dir: {runs_dir}")
    print(f"Mode: {mode}")
    print(f"Scanned runs: {len(reports)}")
    print(f"Problematic runs: {len(problematic)}")

    if not problematic:
        print("No incomplete run directories detected.")
        return

    for report in problematic:
        expected = report.expected_answers if report.expected_answers is not None else "?"
        print(
            f"- {report.run_dir} | dataset={report.dataset or '?'} | status={report.status} "
            f"| valid={report.valid_answers}/{expected} | lines={report.total_lines}"
        )
        print(f"  reasons: {'; '.join(report.reasons)}")


def delete_runs(reports: list[RunReport]) -> int:
    problematic = [report for report in reports if report.is_problematic]
    deleted = 0

    for report in problematic:
        shutil.rmtree(report.run_dir)
        deleted += 1
        print(f"DELETED {report.run_dir}")

    return deleted


def main() -> int:
    args = parse_args()
    runs_dir = args.runs_dir.resolve()

    if not runs_dir.exists():
        print(f"Runs directory not found: {runs_dir}", file=sys.stderr)
        return 1

    run_dirs = discover_run_dirs(runs_dir, args.dataset, args.name_prefix)
    if not run_dirs:
        print("No matching run directories found.")
        return 0

    question_count_cache: dict[Path, int] = {}
    reports = [analyze_run(run_dir, question_count_cache) for run_dir in run_dirs]

    print_preview(reports, runs_dir, args.mode)

    if args.mode == "delete":
        deleted = delete_runs(reports)
        print(f"Deleted {deleted} problematic run(s).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
