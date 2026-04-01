"""Aggregate evaluated run summaries into CSV/JSON tables."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


LLAVA_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARY_BASE_DIR = LLAVA_ROOT / "entropy_exp" / "outputs" / "summary"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def sanitize_label(label: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", label.strip())
    return sanitized.strip("._") or "runs"


def resolve_output_dir(output_dir: str | None, selection_label: str) -> Path:
    if output_dir:
        output_path = Path(output_dir)
        if not output_path.is_absolute():
            output_path = LLAVA_ROOT / output_path
        return ensure_dir(output_path.resolve())

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dirname = f"{sanitize_label(selection_label)}_{timestamp}"
    return ensure_dir(DEFAULT_SUMMARY_BASE_DIR / dirname)


def json_string(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def csv_value(value: Any) -> str | int | float:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def primary_metric(dataset: str, metrics: dict[str, Any]) -> tuple[str, Any]:
    if dataset == "mme":
        return "overall_total_score", metrics.get("overall_total_score")
    if dataset == "pope":
        macro_f1 = metrics.get("macro_f1")
        if macro_f1 is not None:
            return "macro_f1", macro_f1
        weighted_average = metrics.get("weighted_average", {})
        return "macro_f1", weighted_average.get("f1_score")
    if dataset == "gqa":
        return "accuracy", metrics.get("accuracy")
    return "unknown", None


def extract_record(run_dir: Path) -> tuple[dict[str, Any] | None, dict[str, str] | None]:
    config_file = run_dir / "config.yaml"
    eval_summary_file = run_dir / "eval" / "summary.json"

    if not config_file.is_file():
        return None, {"run_dir": str(run_dir), "reason": f"Missing config file: {config_file}"}
    if not eval_summary_file.is_file():
        return None, {"run_dir": str(run_dir), "reason": f"Missing eval summary: {eval_summary_file}"}

    try:
        config = load_yaml(config_file)
        eval_summary = load_json(eval_summary_file)
    except Exception as exc:  # pragma: no cover - defensive parsing
        return None, {"run_dir": str(run_dir), "reason": f"Failed to load files: {exc}"}

    pruning = config.get("pruning", {}) or {}
    entropy_cfg = pruning.get("entropy", {}) or {}
    run_meta = config.get("_run_meta", {}) or {}
    metrics = eval_summary.get("metrics", {}) or {}
    dataset = eval_summary.get("dataset") or run_meta.get("dataset") or ""
    metric_name, metric_value = primary_metric(dataset, metrics)

    record = {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "config_file": str(config_file),
        "eval_summary_file": str(eval_summary_file),
        "dataset": dataset,
        "run_mode": run_meta.get("run_mode"),
        "timestamp": run_meta.get("timestamp"),
        "strategy": pruning.get("strategy"),
        "layer_selection": pruning.get("layer_selection"),
        "prune_layers": pruning.get("prune_layers"),
        "effective_prune_layers": pruning.get("effective_prune_layers"),
        "tail_start_layer": pruning.get("tail_start_layer"),
        "prune_ratio": pruning.get("prune_ratio"),
        "v_token_num": pruning.get("v_token_num"),
        "configured_max_samples": pruning.get("max_samples"),
        "run_max_samples": run_meta.get("max_samples"),
        "entropy_dynamic_ratio": entropy_cfg.get("dynamic_ratio"),
        "entropy_dynamic_scale": entropy_cfg.get("dynamic_scale"),
        "entropy_max_prune_ratio": entropy_cfg.get("max_prune_ratio"),
        "primary_metric_name": metric_name,
        "primary_metric_value": metric_value,
        "mme_overall_total_score": metrics.get("overall_total_score") if dataset == "mme" else None,
        "pope_macro_f1": (
            metrics.get("macro_f1", metrics.get("weighted_average", {}).get("f1_score"))
            if dataset == "pope"
            else None
        ),
        "gqa_accuracy": metrics.get("accuracy") if dataset == "gqa" else None,
        "pruning_config": pruning,
        "eval_metrics": metrics,
    }

    if metric_value is None:
        record["warning"] = f"Primary metric {metric_name} not found in eval summary"

    return record, None


def sort_key(record: dict[str, Any]) -> tuple[Any, ...]:
    score = record.get("primary_metric_value")
    sort_score = float(score) if isinstance(score, (int, float)) else float("-inf")
    return (
        record.get("dataset") or "",
        -sort_score,
        record.get("run_name") or "",
    )


def build_csv_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        rows.append(
            {
                "dataset": record["dataset"],
                "run_name": record["run_name"],
                "run_mode": record["run_mode"],
                "timestamp": record["timestamp"],
                "run_dir": record["run_dir"],
                "config_file": record["config_file"],
                "eval_summary_file": record["eval_summary_file"],
                "strategy": record["strategy"],
                "layer_selection": record["layer_selection"],
                "prune_layers": csv_value(record["prune_layers"]),
                "effective_prune_layers": csv_value(record["effective_prune_layers"]),
                "tail_start_layer": csv_value(record["tail_start_layer"]),
                "prune_ratio": csv_value(record["prune_ratio"]),
                "v_token_num": csv_value(record["v_token_num"]),
                "configured_max_samples": csv_value(record["configured_max_samples"]),
                "run_max_samples": csv_value(record["run_max_samples"]),
                "entropy_dynamic_ratio": csv_value(record["entropy_dynamic_ratio"]),
                "entropy_dynamic_scale": csv_value(record["entropy_dynamic_scale"]),
                "entropy_max_prune_ratio": csv_value(record["entropy_max_prune_ratio"]),
                "primary_metric_name": record["primary_metric_name"],
                "primary_metric_value": csv_value(record["primary_metric_value"]),
                "mme_overall_total_score": csv_value(record["mme_overall_total_score"]),
                "pope_macro_f1": csv_value(record["pope_macro_f1"]),
                "gqa_accuracy": csv_value(record["gqa_accuracy"]),
                "pruning_config_json": json_string(record["pruning_config"]),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "dataset",
        "run_name",
        "run_mode",
        "timestamp",
        "run_dir",
        "config_file",
        "eval_summary_file",
        "strategy",
        "layer_selection",
        "prune_layers",
        "effective_prune_layers",
        "tail_start_layer",
        "prune_ratio",
        "v_token_num",
        "configured_max_samples",
        "run_max_samples",
        "entropy_dynamic_ratio",
        "entropy_dynamic_scale",
        "entropy_max_prune_ratio",
        "primary_metric_name",
        "primary_metric_value",
        "mme_overall_total_score",
        "pope_macro_f1",
        "gqa_accuracy",
        "pruning_config_json",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate evaluated run summaries")
    parser.add_argument(
        "--run-dir",
        nargs="+",
        required=True,
        help="One or more run directories under entropy_exp/outputs/runs",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory to write aggregated summary artifacts",
    )
    parser.add_argument(
        "--selection-label",
        type=str,
        default="runs",
        help="Label used to name the default output directory",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    run_dirs = [Path(run_dir).resolve() for run_dir in args.run_dir]
    output_dir = resolve_output_dir(args.output_dir, args.selection_label)

    records: list[dict[str, Any]] = []
    skipped_runs: list[dict[str, str]] = []
    for run_dir in run_dirs:
        record, skipped = extract_record(run_dir)
        if skipped is not None:
            skipped_runs.append(skipped)
            continue
        records.append(record)

    records.sort(key=sort_key)
    csv_rows = build_csv_rows(records)

    summary_payload = {
        "selection_label": args.selection_label,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(output_dir),
        "selected_run_count": len(run_dirs),
        "included_run_count": len(records),
        "skipped_run_count": len(skipped_runs),
        "records": records,
        "skipped_runs": skipped_runs,
    }

    summary_json_path = output_dir / "summary.json"
    summary_csv_path = output_dir / "summary.csv"
    skipped_json_path = output_dir / "skipped_runs.json"

    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)
    write_csv(summary_csv_path, csv_rows)
    with skipped_json_path.open("w", encoding="utf-8") as f:
        json.dump(skipped_runs, f, indent=2, ensure_ascii=False)

    print(f"Writing summary artifacts to: {output_dir}")
    print(f"Included runs: {len(records)} / {len(run_dirs)}")
    if skipped_runs:
        print(f"Skipped runs:  {len(skipped_runs)}")
        for skipped in skipped_runs:
            print(f"  - {skipped['run_dir']}: {skipped['reason']}")

    print(f"Saved summary JSON: {summary_json_path}")
    print(f"Saved summary CSV:  {summary_csv_path}")
    print(f"Saved skipped log:  {skipped_json_path}")

    if not records:
        print("No evaluated runs were found in the selected inputs.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
