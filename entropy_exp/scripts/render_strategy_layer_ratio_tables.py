#!/usr/bin/env python3
"""Render one layer-ratio markdown table per pruning strategy from summary.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render strategy layer-ratio markdown tables from summary.json")
    parser.add_argument("--summary-json", required=True, help="Path to summarize_results summary.json")
    parser.add_argument("--output", required=True, help="Markdown output path")
    parser.add_argument("--dataset", default="pope", help="Dataset to include")
    parser.add_argument("--metric-name", help="Optional metric name override")
    return parser.parse_args()


def load_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_scalar_list(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    result: list[float] = []
    for item in value:
        if isinstance(item, (int, float)):
            result.append(float(item))
    return result


def format_ratio_label(ratio: float) -> str:
    return f"r{ratio:.1f}"


def format_metric(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.6f}"


def render_table(rows: list[list[str]]) -> list[str]:
    if not rows:
        return []
    header = rows[0]
    separators = ["---"] + ["---:" for _ in header[1:]]
    rendered = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(separators) + " |",
    ]
    for row in rows[1:]:
        rendered.append("| " + " | ".join(row) + " |")
    return rendered


def main() -> int:
    args = parse_args()
    summary_path = Path(args.summary_json).resolve()
    output_path = Path(args.output).resolve()
    payload = load_summary(summary_path)
    records = payload.get("records", [])

    selected_records = [row for row in records if row.get("dataset") == args.dataset]
    if not selected_records:
        raise RuntimeError(f"No records found for dataset={args.dataset}")

    metric_name = args.metric_name
    if metric_name is None:
        metric_names = {
            str(row.get("primary_metric_name"))
            for row in selected_records
            if row.get("primary_metric_name") is not None
        }
        metric_name = sorted(metric_names)[0] if metric_names else "primary_metric"

    baseline_records = [
        row
        for row in selected_records
        if row.get("run_mode") == "baseline" or row.get("strategy") == "baseline"
    ]
    strategy_records = [
        row
        for row in selected_records
        if row.get("run_mode") != "baseline" and row.get("strategy") != "baseline"
    ]

    ratios = sorted(
        {
            ratio
            for row in strategy_records
            for ratio in extract_scalar_list(row.get("prune_ratio"))
        }
    )
    layers = sorted(
        {
            int(layer)
            for row in strategy_records
            for layer in extract_scalar_list(row.get("prune_layers"))
        }
    )
    strategies = sorted({str(row.get("strategy")) for row in strategy_records})

    lines: list[str] = [
        f"# {args.dataset.upper()} Strategy Layer-Ratio Tables",
        "",
        f"Metric: `{metric_name}`",
        "",
    ]

    if baseline_records:
        baseline = max(
            baseline_records,
            key=lambda row: float(row.get("primary_metric_value") or float("-inf")),
        )
        lines.extend(
            [
                "## Baseline Reference",
                "",
                f"- score: `{format_metric(baseline.get('primary_metric_value'))}`",
                f"- run: `{baseline.get('run_name')}`",
                "",
            ]
        )

    for strategy in strategies:
        rows = [["layer"] + [format_ratio_label(ratio) for ratio in ratios]]
        selected_runs: list[list[str]] = [["layer", "ratio", "score", "selected_run"]]
        strategy_rows = [row for row in strategy_records if row.get("strategy") == strategy]

        by_pair: dict[tuple[int, float], dict[str, Any]] = {}
        for row in strategy_rows:
            prune_layers = extract_scalar_list(row.get("prune_layers"))
            prune_ratios = extract_scalar_list(row.get("prune_ratio"))
            if len(prune_layers) != 1 or len(prune_ratios) != 1:
                continue
            layer = int(prune_layers[0])
            ratio = float(prune_ratios[0])
            by_pair[(layer, ratio)] = row

        for layer in layers:
            current = [str(layer)]
            for ratio in ratios:
                record = by_pair.get((layer, ratio))
                current.append(format_metric(record.get("primary_metric_value")) if record else "-")
                if record:
                    selected_runs.append(
                        [
                            str(layer),
                            f"{ratio:.1f}",
                            format_metric(record.get("primary_metric_value")),
                            str(record.get("run_name")),
                        ]
                    )
            rows.append(current)

        lines.extend(
            [
                f"## {strategy}",
                "",
                *render_table(rows),
                "",
                "### Selected Runs",
                "",
                *render_table(selected_runs),
                "",
            ]
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
