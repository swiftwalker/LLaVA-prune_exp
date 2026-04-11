#!/usr/bin/env python3
"""Migrate entropy_exp run directories to the strategy/dataset layout."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
LLAVA_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(LLAVA_ROOT))

from entropy_exp.src.run_layout import (
    LLAVA_ROOT as RUN_LAYOUT_ROOT,
    build_run_dir,
    build_run_rel_dir,
    find_run_dirs,
    map_run_dirs_by_name,
    read_run_metadata,
    resolve_repo_path,
)


assert RUN_LAYOUT_ROOT == LLAVA_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Relayout entropy_exp run directories")
    parser.add_argument("--runs-dir", default="entropy_exp/outputs/runs")
    parser.add_argument("--summary-dir", default="entropy_exp/outputs/summary")
    parser.add_argument("--summary-archive-dir", default="entropy_exp/outputs/summary_archive")
    parser.add_argument("--logs-dir", default="entropy_exp/outputs/logs/migrations")
    parser.add_argument("--execute", action="store_true", help="Apply the migration; default is dry-run")
    parser.add_argument("--skip-process-check", action="store_true")
    return parser.parse_args()


def timestamp_now() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


def find_active_processes() -> list[str]:
    proc = subprocess.run(["ps", "-eo", "pid=,cmd="], check=True, capture_output=True, text=True)
    matches: list[str] = []
    for raw_line in proc.stdout.splitlines():
        line = raw_line.strip()
        if not line or "migrate_runs_layout.py" in line:
            continue
        if "entropy_exp/scripts/run_prune.sh" in line or "entropy_exp/src/prune_inference.py" in line:
            matches.append(line)
    return matches


def find_tmux_sessions() -> list[str]:
    proc = subprocess.run(["tmux", "ls"], capture_output=True, text=True)
    if proc.returncode != 0:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def build_run_inventory(run_dirs: list[Path]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for run_dir in run_dirs:
        metadata = read_run_metadata(run_dir)
        strategy = metadata.strategy or "unknown"
        dataset = metadata.dataset or "unknown"
        counter[f"{strategy}/{dataset}"] += 1
    return dict(sorted(counter.items()))


def build_legacy_run_alias(run_dir: Path) -> str | None:
    metadata = read_run_metadata(run_dir)
    if not metadata.dataset or not metadata.strategy:
        return None
    timestamp = metadata.timestamp
    if not timestamp and "__" in metadata.run_name:
        timestamp = metadata.run_name.split("__", 1)[1]
    if not timestamp:
        return None
    return f"{metadata.dataset}_{metadata.strategy}_{timestamp}"


def build_run_selector_map(run_dirs: list[Path]) -> tuple[dict[str, Path], list[str]]:
    selector_map: dict[str, Path] = {}
    collisions: set[str] = set()

    for run_dir in run_dirs:
        aliases = {run_dir.name}
        legacy_alias = build_legacy_run_alias(run_dir)
        if legacy_alias:
            aliases.add(legacy_alias)

        for alias in aliases:
            if alias in collisions:
                continue
            existing = selector_map.get(alias)
            if existing is not None and existing != run_dir:
                selector_map.pop(alias, None)
                collisions.add(alias)
                continue
            selector_map[alias] = run_dir

    return selector_map, sorted(collisions)


def build_migration_manifest(runs_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    output_base_dir = runs_dir.parent

    for run_dir in find_run_dirs(runs_dir):
        rel_parts = run_dir.relative_to(runs_dir).parts
        metadata = read_run_metadata(run_dir)
        record = {
            "run_name": metadata.run_name,
            "old_path": str(run_dir),
            "dataset": metadata.dataset,
            "strategy": metadata.strategy,
            "run_mode": metadata.run_mode,
            "layout_depth": len(rel_parts),
        }

        if len(rel_parts) == 1:
            if not metadata.dataset or not metadata.strategy:
                record["status"] = "invalid_metadata"
                record["reason"] = "Unable to determine dataset/strategy from config or run name"
                issues.append(record)
                manifest.append(record)
                continue

            new_path = build_run_dir(output_base_dir, metadata.strategy, metadata.dataset, metadata.run_name)
            record["new_path"] = str(new_path)
            if new_path.exists():
                record["status"] = "conflict"
                record["reason"] = f"Target already exists: {new_path}"
                issues.append(record)
            else:
                record["status"] = "would_move"
            manifest.append(record)
            continue

        if len(rel_parts) == 3:
            record["new_path"] = str(run_dir)
            record["status"] = "already_structured"
            manifest.append(record)
            continue

        record["status"] = "unexpected_layout"
        record["reason"] = f"Unexpected run layout depth: {len(rel_parts)}"
        issues.append(record)
        manifest.append(record)

    return manifest, issues


def assert_unique_run_names(run_dirs: list[Path]) -> None:
    counts = Counter(path.name for path in run_dirs)
    duplicates = [name for name, count in counts.items() if count > 1]
    if duplicates:
        raise RuntimeError(f"Duplicate run names found; cannot migrate safely: {duplicates[:10]}")


def extract_summary_record_names(summary_payload: dict[str, Any]) -> list[str]:
    ordered_names: list[str] = []
    seen: set[str] = set()
    for record in summary_payload.get("records", []):
        run_name = record.get("run_name") or Path(record.get("run_dir", "")).name
        if run_name and run_name not in seen:
            ordered_names.append(run_name)
            seen.add(run_name)
    return ordered_names


def extract_summary_skipped_entries(summary_payload: dict[str, Any]) -> list[dict[str, Any]]:
    return list(summary_payload.get("skipped_runs", []))


def precheck_summaries(summary_dir: Path, current_run_map: dict[str, Path]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for summary_path in sorted(path for path in summary_dir.iterdir() if path.is_dir()):
        payload = load_json(summary_path / "summary.json")
        record_names = extract_summary_record_names(payload)
        skipped_entries = extract_summary_skipped_entries(payload)
        missing_record_names = [name for name in record_names if name not in current_run_map]
        missing_skipped_names = [
            Path(item.get("run_dir", "")).name
            for item in skipped_entries
            if Path(item.get("run_dir", "")).name not in current_run_map
        ]
        results.append(
            {
                "summary_dir": str(summary_path),
                "selection_label": payload.get("selection_label"),
                "selected_run_count": payload.get("selected_run_count"),
                "included_run_count": payload.get("included_run_count"),
                "skipped_run_count": payload.get("skipped_run_count"),
                "record_names": record_names,
                "missing_record_names": missing_record_names,
                "missing_skipped_names": missing_skipped_names,
            }
        )
    return results


def relocate_eval_summary(run_dir: Path) -> None:
    summary_path = run_dir / "eval" / "summary.json"
    if not summary_path.is_file():
        return

    payload = load_json(summary_path)
    payload["run_dir"] = str(run_dir)
    payload["config_file"] = str(run_dir / "config.yaml")
    payload["answers_file"] = str(run_dir / "answers.jsonl")
    payload["output_dir"] = str(run_dir / "eval")

    prediction_file = payload.get("prediction_file")
    if prediction_file:
        payload["prediction_file"] = str((run_dir / "eval") / Path(prediction_file).name)

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


def update_run_config(run_dir: Path, strategy: str, dataset: str, run_name: str) -> None:
    config_path = run_dir / "config.yaml"
    if not config_path.is_file():
        return

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    run_meta = config.setdefault("_run_meta", {})
    run_meta["dataset"] = dataset
    run_meta["strategy"] = strategy
    run_meta["run_name"] = run_name
    run_meta["run_rel_dir"] = build_run_rel_dir(run_dir.parents[3], strategy, dataset, run_name)
    run_meta["run_dir"] = str(run_dir)

    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, default_flow_style=False, allow_unicode=True, sort_keys=False)


def move_run(record: dict[str, Any]) -> dict[str, Any]:
    old_path = Path(record["old_path"])
    new_path = Path(record["new_path"])
    new_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(old_path), str(new_path))
    update_run_config(new_path, record["strategy"], record["dataset"], record["run_name"])
    relocate_eval_summary(new_path)

    moved_record = dict(record)
    moved_record["status"] = "moved"
    return moved_record


def archive_summaries(summary_dir: Path, summary_archive_root: Path) -> list[Path]:
    summary_archive_root.mkdir(parents=True, exist_ok=True)
    archived_dirs: list[Path] = []
    for current_dir in sorted(path for path in summary_dir.iterdir() if path.is_dir()):
        archived_path = summary_archive_root / current_dir.name
        shutil.move(str(current_dir), str(archived_path))
        archived_dirs.append(archived_path)
    return archived_dirs


def rebuild_summary(
    archived_summary_dir: Path,
    target_summary_dir: Path,
    current_run_map: dict[str, Path],
) -> dict[str, Any]:
    archived_summary = load_json(archived_summary_dir / "summary.json")
    record_names = extract_summary_record_names(archived_summary)
    missing_records = [name for name in record_names if name not in current_run_map]
    skipped_entries = extract_summary_skipped_entries(archived_summary)
    resolvable_skipped: list[str] = []
    carried_skipped: list[dict[str, Any]] = []
    for skipped in skipped_entries:
        run_name = Path(skipped.get("run_dir", "")).name
        if run_name in current_run_map:
            resolvable_skipped.append(run_name)
        else:
            carried_skipped.append(skipped)
    result = {
        "summary_name": archived_summary_dir.name,
        "selection_label": archived_summary.get("selection_label") or archived_summary_dir.name,
        "selected_run_count_before": archived_summary.get("selected_run_count"),
        "included_run_count_before": archived_summary.get("included_run_count"),
        "skipped_run_count_before": archived_summary.get("skipped_run_count"),
        "missing_record_names": missing_records,
        "carried_skipped_count": len(carried_skipped),
    }
    if missing_records:
        result["status"] = "failed_missing_records"
        return result

    selected_names = record_names + resolvable_skipped
    run_dirs = [str(current_run_map[name]) for name in selected_names]
    cmd = [
        sys.executable,
        str(LLAVA_ROOT / "entropy_exp" / "src" / "summarize_results.py"),
        "--selection-label",
        result["selection_label"],
        "--output-dir",
        str(target_summary_dir),
        "--run-dir",
        *run_dirs,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    result["stdout"] = proc.stdout
    result["stderr"] = proc.stderr
    if proc.returncode != 0:
        result["status"] = "failed_subprocess"
        result["returncode"] = proc.returncode
        return result

    regenerated_path = target_summary_dir / "summary.json"
    regenerated = load_json(regenerated_path)
    skipped_runs_path = target_summary_dir / "skipped_runs.json"
    if carried_skipped:
        regenerated["selected_run_count"] = regenerated.get("selected_run_count", 0) + len(carried_skipped)
        regenerated["skipped_run_count"] = regenerated.get("skipped_run_count", 0) + len(carried_skipped)
        regenerated.setdefault("skipped_runs", [])
        regenerated["skipped_runs"].extend(carried_skipped)
        write_json(regenerated_path, regenerated)
        write_json(skipped_runs_path, regenerated["skipped_runs"])

    result["status"] = "rebuilt"
    result["selected_run_count_after"] = regenerated.get("selected_run_count")
    result["included_run_count_after"] = regenerated.get("included_run_count")
    result["skipped_run_count_after"] = regenerated.get("skipped_run_count")
    return result


def main() -> int:
    args = parse_args()
    runs_dir = resolve_repo_path(args.runs_dir)
    summary_dir = resolve_repo_path(args.summary_dir)
    summary_archive_parent = resolve_repo_path(args.summary_archive_dir)
    logs_dir = resolve_repo_path(args.logs_dir)
    operation_stamp = timestamp_now()
    operation_dir = logs_dir / f"{operation_stamp}_runs_relayout"
    operation_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_process_check:
        active_processes = find_active_processes()
        tmux_sessions = find_tmux_sessions()
        if active_processes or tmux_sessions:
            payload = {"active_processes": active_processes, "tmux_sessions": tmux_sessions}
            write_json(operation_dir / "blocked_by_active_jobs.json", payload)
            print("Refusing to migrate while pruning jobs or tmux sessions are active.", file=sys.stderr)
            if active_processes:
                for line in active_processes:
                    print(f"  {line}", file=sys.stderr)
            if tmux_sessions:
                for line in tmux_sessions:
                    print(f"  {line}", file=sys.stderr)
            return 1

    before_run_dirs = find_run_dirs(runs_dir)
    assert_unique_run_names(before_run_dirs)
    current_run_map, selector_collisions_before = build_run_selector_map(before_run_dirs)

    manifest, issues = build_migration_manifest(runs_dir)
    summary_precheck = precheck_summaries(summary_dir, current_run_map)
    missing_summary_members = [item for item in summary_precheck if item["missing_record_names"]]
    if missing_summary_members:
        write_json(operation_dir / "summary_precheck.json", summary_precheck)
        print("Cannot migrate because some included summary records are missing from current runs.", file=sys.stderr)
        for item in missing_summary_members:
            print(f"  {item['summary_dir']}: {item['missing_record_names']}", file=sys.stderr)
        return 1

    dry_run_report = {
        "mode": "execute" if args.execute else "dry_run",
        "runs_before": len(before_run_dirs),
        "summary_dirs_before": len([path for path in summary_dir.iterdir() if path.is_dir()]),
        "inventory_before": build_run_inventory(before_run_dirs),
        "selector_collisions_before": selector_collisions_before,
        "manifest": manifest,
        "issues": issues,
        "summary_precheck": summary_precheck,
    }
    write_json(operation_dir / "dry_run_report.json", dry_run_report)
    write_json(operation_dir / "manifest.json", manifest)

    if issues:
        print("Migration planning found blocking issues; see manifest/report for details.", file=sys.stderr)
        return 1

    if not args.execute:
        print(f"Dry-run complete. Report written to {operation_dir / 'dry_run_report.json'}")
        print(f"Planned moves: {sum(1 for item in manifest if item['status'] == 'would_move')}")
        print(f"Already structured: {sum(1 for item in manifest if item['status'] == 'already_structured')}")
        return 0

    summary_archive_root = summary_archive_parent / f"{operation_stamp}_pre_runs_relayout"
    archived_summary_dirs = archive_summaries(summary_dir, summary_archive_root)

    executed_manifest: list[dict[str, Any]] = []
    for record in manifest:
        if record["status"] == "would_move":
            executed_manifest.append(move_run(record))
        else:
            executed_manifest.append(record)

    after_run_dirs = find_run_dirs(runs_dir)
    assert_unique_run_names(after_run_dirs)
    after_run_map, selector_collisions_after = build_run_selector_map(after_run_dirs)

    rebuild_results: list[dict[str, Any]] = []
    for archived_summary_dir in archived_summary_dirs:
        target_summary_dir = summary_dir / archived_summary_dir.name
        rebuild_results.append(rebuild_summary(archived_summary_dir, target_summary_dir, after_run_map))

    final_report = {
        "mode": "execute",
        "summary_archive_root": str(summary_archive_root),
        "runs_before": len(before_run_dirs),
        "runs_after": len(after_run_dirs),
        "inventory_before": build_run_inventory(before_run_dirs),
        "inventory_after": build_run_inventory(after_run_dirs),
        "selector_collisions_after": selector_collisions_after,
        "planned_move_count": sum(1 for item in manifest if item["status"] == "would_move"),
        "moved_count": sum(1 for item in executed_manifest if item["status"] == "moved"),
        "already_structured_count": sum(1 for item in executed_manifest if item["status"] == "already_structured"),
        "summary_rebuild_results": rebuild_results,
    }

    write_json(operation_dir / "manifest_executed.json", executed_manifest)
    write_json(operation_dir / "final_report.json", final_report)

    failed_rebuilds = [item for item in rebuild_results if item["status"] != "rebuilt"]
    if failed_rebuilds:
        print("Migration completed with summary rebuild failures; see final report.", file=sys.stderr)
        return 1

    print(f"Migration complete. Final report: {operation_dir / 'final_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
