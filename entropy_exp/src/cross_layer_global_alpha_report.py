"""Build a compact report for the cross-layer global-alpha diagnostic matrix."""

from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


LLAVA_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARY = (
    LLAVA_ROOT
    / "entropy_exp"
    / "outputs"
    / "summary"
    / "cross_layer_global_alpha_diag"
    / "summary.csv"
)
DEFAULT_OUTPUT_DIR = LLAVA_ROOT / "entropy_exp" / "outputs" / "analysis" / "cross_layer_global_alpha_diag"
STRATEGY_ORDER = ["sparsevlm", "sparsevlm_entropy_alpha", "sparsevlm_entropy_alpha_global"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report cross-layer global-alpha diagnostic results")
    parser.add_argument("--summary-csv", default=str(DEFAULT_SUMMARY), help="Aggregated summary.csv to analyze")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for report artifacts")
    return parser.parse_args()


def parse_list(text: str) -> list[Any]:
    if not text:
        return []
    value = ast.literal_eval(text)
    return value if isinstance(value, list) else [value]


def parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def config_label(row: dict[str, str]) -> str:
    ratios = [round(float(item), 2) for item in parse_list(row.get("prune_ratio", ""))]
    if ratios == [0.21, 0.21, 0.21]:
        return "uniform_R0p5"
    if ratios == [0.08, 0.22, 0.38]:
        return "increasing_R0p5"
    if ratios == [0.33, 0.33, 0.33]:
        return "uniform_R0p7"
    if ratios == [0.12, 0.33, 0.52]:
        return "increasing_R0p7"
    return "r" + "-".join(str(ratio).replace(".", "p") for ratio in ratios)


def jaccard(a: list[int], b: list[int]) -> float:
    set_a = set(a)
    set_b = set(b)
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / max(1, len(set_a | set_b))


def _value_span(line: str, key: str) -> str | None:
    needle = f'"{key}": '
    start = line.find(needle)
    if start < 0:
        return None
    start += len(needle)
    if start >= len(line):
        return None
    if line[start] == "[":
        end = line.find("]", start)
        return line[start : end + 1] if end >= 0 else None
    end = line.find(",", start)
    if end < 0:
        end = line.find("}", start)
    return line[start:end] if end >= 0 else None


def extract_scalar(line: str, key: str) -> float | bool | None:
    value = _value_span(line, key)
    if value is None:
        return None
    value = value.strip()
    if value == "true":
        return True
    if value == "false":
        return False
    try:
        return float(value)
    except ValueError:
        return None


def extract_number_list(line: str, key: str) -> list[float] | None:
    value = _value_span(line, key)
    if value is None:
        return None
    value = value.strip()
    if not value.startswith("[") or not value.endswith("]"):
        return None
    body = value[1:-1].strip()
    if not body:
        return []
    return [float(item) for item in body.split(",")]


def extract_int_list(line: str, key: str) -> list[int] | None:
    values = extract_number_list(line, key)
    if values is None:
        return None
    return [int(value) for value in values]


def stats_summary(run_dir: Path) -> dict[str, Any]:
    stats_path = run_dir / "stats.jsonl"
    if not stats_path.is_file():
        return {}

    alpha_values: list[float] = []
    history_debt_values: list[float] = []
    combined_debt_values: list[float] = []
    jaccard_values: list[float] = []
    use_global_count = 0
    layer_count = 0
    sample_count = 0

    with stats_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            sample_count += 1
            layer_ids = (2, 8, 16)
            previous_keep = None
            for layer_idx in layer_ids:
                layer_count += 1
                alpha = parse_float(extract_scalar(line, f"layer_{layer_idx}_adaptive_alpha"))
                if alpha is not None:
                    alpha_values.append(alpha)
                if extract_scalar(line, f"layer_{layer_idx}_global_use_global") is True:
                    use_global_count += 1

                history_debt = extract_number_list(line, f"layer_{layer_idx}_stratum_history_debt")
                if history_debt is not None:
                    history_debt_values.append(mean(float(item) for item in history_debt) if history_debt else 0.0)
                combined_debt = extract_number_list(line, f"layer_{layer_idx}_stratum_combined_debt")
                if combined_debt is not None:
                    combined_debt_values.append(mean(float(item) for item in combined_debt) if combined_debt else 0.0)

                keep = extract_int_list(line, f"layer_{layer_idx}_keep_patch_indices")
                if previous_keep is not None and keep is not None:
                    jaccard_values.append(jaccard(previous_keep, keep))
                if keep is not None:
                    previous_keep = keep

    return {
        "stats_samples": sample_count,
        "stats_layers": layer_count,
        "mean_adaptive_alpha": mean(alpha_values) if alpha_values else None,
        "mean_history_debt": mean(history_debt_values) if history_debt_values else None,
        "mean_combined_debt": mean(combined_debt_values) if combined_debt_values else None,
        "mean_layer_keep_jaccard": mean(jaccard_values) if jaccard_values else None,
        "global_use_ratio": use_global_count / layer_count if layer_count else None,
    }


def load_rows(summary_csv: Path) -> list[dict[str, Any]]:
    with summary_csv.open(newline="", encoding="utf-8") as f:
        raw_rows = list(csv.DictReader(f))

    rows = []
    for row in raw_rows:
        run_dir = Path(row["run_dir"])
        enriched = {
            **row,
            "config_label": config_label(row),
            "metric": parse_float(row.get("primary_metric_value")),
            "run_dir_path": run_dir,
        }
        if row.get("strategy") == "sparsevlm_entropy_alpha_global":
            enriched.update(stats_summary(run_dir))
        rows.append(enriched)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def build_comparison(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(row["dataset"], row["config_label"], row["strategy"]): row for row in rows}
    comparisons = []
    for dataset in sorted({row["dataset"] for row in rows}):
        for cfg in sorted({row["config_label"] for row in rows if row["dataset"] == dataset}):
            sparse = by_key.get((dataset, cfg, "sparsevlm"))
            local = by_key.get((dataset, cfg, "sparsevlm_entropy_alpha"))
            global_row = by_key.get((dataset, cfg, "sparsevlm_entropy_alpha_global"))
            if not sparse or not local or not global_row:
                continue
            comparisons.append(
                {
                    "dataset": dataset,
                    "config": cfg,
                    "metric_name": global_row.get("primary_metric_name"),
                    "sparsevlm": sparse.get("metric"),
                    "entropy_alpha": local.get("metric"),
                    "entropy_alpha_global": global_row.get("metric"),
                    "global_minus_sparsevlm": (
                        global_row["metric"] - sparse["metric"]
                        if global_row.get("metric") is not None and sparse.get("metric") is not None
                        else None
                    ),
                    "global_minus_entropy_alpha": (
                        global_row["metric"] - local["metric"]
                        if global_row.get("metric") is not None and local.get("metric") is not None
                        else None
                    ),
                    "global_mean_alpha": global_row.get("mean_adaptive_alpha"),
                    "global_mean_history_debt": global_row.get("mean_history_debt"),
                    "global_mean_combined_debt": global_row.get("mean_combined_debt"),
                    "global_mean_keep_jaccard": global_row.get("mean_layer_keep_jaccard"),
                    "global_use_ratio": global_row.get("global_use_ratio"),
                }
            )
    return comparisons


def format_value(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_markdown(path: Path, comparisons: list[dict[str, Any]], detail_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Cross-Layer Global Alpha Diagnostic Report",
        "",
        "## Metric Comparison",
        "",
        "| dataset | config | metric | sparsevlm | entropy_alpha | entropy_alpha_global | global-sv2 | global-old |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        lines.append(
            "| {dataset} | {config} | {metric_name} | {sparsevlm} | {entropy_alpha} | "
            "{entropy_alpha_global} | {global_minus_sparsevlm} | {global_minus_entropy_alpha} |".format(
                dataset=row["dataset"],
                config=row["config"],
                metric_name=row["metric_name"],
                sparsevlm=format_value(row["sparsevlm"]),
                entropy_alpha=format_value(row["entropy_alpha"]),
                entropy_alpha_global=format_value(row["entropy_alpha_global"]),
                global_minus_sparsevlm=format_value(row["global_minus_sparsevlm"]),
                global_minus_entropy_alpha=format_value(row["global_minus_entropy_alpha"]),
            )
        )

    lines.extend(
        [
            "",
            "## Global Strategy Stats",
            "",
            "| dataset | config | mean alpha | mean history debt | mean combined debt | mean keep Jaccard | global use ratio |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparisons:
        lines.append(
            "| {dataset} | {config} | {global_mean_alpha} | {global_mean_history_debt} | "
            "{global_mean_combined_debt} | {global_mean_keep_jaccard} | {global_use_ratio} |".format(
                dataset=row["dataset"],
                config=row["config"],
                global_mean_alpha=format_value(row["global_mean_alpha"]),
                global_mean_history_debt=format_value(row["global_mean_history_debt"]),
                global_mean_combined_debt=format_value(row["global_mean_combined_debt"]),
                global_mean_keep_jaccard=format_value(row["global_mean_keep_jaccard"]),
                global_use_ratio=format_value(row["global_use_ratio"]),
            )
        )

    lines.extend(["", "## Coverage", ""])
    counts = defaultdict(int)
    for row in detail_rows:
        counts[(row["dataset"], row["strategy"])] += 1
    for dataset in sorted({row["dataset"] for row in detail_rows}):
        parts = [f"{strategy}={counts[(dataset, strategy)]}" for strategy in STRATEGY_ORDER]
        lines.append(f"- {dataset}: " + ", ".join(parts))

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    summary_csv = Path(args.summary_csv)
    if not summary_csv.is_absolute():
        summary_csv = LLAVA_ROOT / summary_csv
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = LLAVA_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(summary_csv)
    comparison_rows = build_comparison(rows)

    detail_fields = [
        "dataset",
        "config_label",
        "strategy",
        "primary_metric_name",
        "metric",
        "run_name",
        "run_dir",
        "mean_adaptive_alpha",
        "mean_history_debt",
        "mean_combined_debt",
        "mean_layer_keep_jaccard",
        "global_use_ratio",
        "stats_samples",
        "stats_layers",
    ]
    comparison_fields = [
        "dataset",
        "config",
        "metric_name",
        "sparsevlm",
        "entropy_alpha",
        "entropy_alpha_global",
        "global_minus_sparsevlm",
        "global_minus_entropy_alpha",
        "global_mean_alpha",
        "global_mean_history_debt",
        "global_mean_combined_debt",
        "global_mean_keep_jaccard",
        "global_use_ratio",
    ]

    write_csv(output_dir / "detail.csv", rows, detail_fields)
    write_csv(output_dir / "comparison.csv", comparison_rows, comparison_fields)
    write_markdown(output_dir / "report.md", comparison_rows, rows)
    print(f"Wrote report artifacts to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
