"""Summarize layer-wise SV2/boost hybrid pruning results."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


CONFIG_ORDER = [
    "baseline_sv2",
    "baseline_ours",
    "hybrid_deep",
    "hybrid_mid",
    "hybrid_shallow",
    "hybrid_mid_only",
]
CONFIG_BY_MODES = {
    ("S", "S", "S"): "baseline_sv2",
    ("O", "O", "O"): "baseline_ours",
    ("O", "O", "S"): "hybrid_deep",
    ("O", "S", "S"): "hybrid_mid",
    ("S", "O", "O"): "hybrid_shallow",
    ("S", "O", "S"): "hybrid_mid_only",
}
TARGET_BY_RATIO = {
    (0.479, 0.333, 0.410): "target118",
    (0.587, 0.546, 0.444): "target60",
    (0.573, 0.780, 0.481): "target28",
}
TARGET_ORDER = ["target118", "target60", "target28"]
EXPECTED_DATASETS = ["gqa", "textvqa", "pope", "mme", "scienceqa"]
EXPECTED_ROWS = len(EXPECTED_DATASETS) * len(TARGET_ORDER) * len(CONFIG_ORDER)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_summary_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_pruning(row: dict[str, str]) -> dict[str, Any]:
    raw = row.get("pruning_config_json") or "{}"
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def ratio_key(value: Any) -> tuple[float, ...] | None:
    if not isinstance(value, list):
        return None
    try:
        return tuple(round(float(item), 3) for item in value)
    except (TypeError, ValueError):
        return None


def mode_key(value: Any) -> tuple[str, ...] | None:
    if not isinstance(value, list):
        return None
    return tuple(str(item).upper() for item in value)


def timestamp_key(row: dict[str, Any]) -> str:
    return str(row.get("timestamp") or row.get("run_name") or "")


def enrich_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        if row.get("strategy") != "sparsevlm_boost_hybrid":
            continue
        pruning = parse_pruning(row)
        hybrid_cfg = pruning.get("sparsevlm_boost_hybrid", {}) or {}
        target = TARGET_BY_RATIO.get(ratio_key(pruning.get("prune_ratio")))
        config = CONFIG_BY_MODES.get(mode_key(hybrid_cfg.get("layer_modes")))
        if target is None or config is None:
            continue
        value_raw = row.get("primary_metric_value")
        try:
            metric_value = float(value_raw) if value_raw not in (None, "") else None
        except ValueError:
            metric_value = None
        if metric_value is None:
            continue
        enriched.append(
            {
                **row,
                "target": target,
                "config": config,
                "metric_name": row.get("primary_metric_name") or "",
                "metric_value": metric_value,
                "layer_modes": list(mode_key(hybrid_cfg.get("layer_modes")) or ()),
            }
        )
    return enriched


def latest_unique(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in sorted(rows, key=timestamp_key):
        key = (row["dataset"], row["target"], row["config"])
        by_key[key] = row
    return list(by_key.values())


def matrix_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(row["dataset"], row["target"], row["config"]): row for row in rows}
    output: list[dict[str, Any]] = []
    for dataset in EXPECTED_DATASETS:
        for target in TARGET_ORDER:
            values = {
                config: by_key.get((dataset, target, config), {}).get("metric_value")
                for config in CONFIG_ORDER
            }
            available = {key: value for key, value in values.items() if value is not None}
            best_config = max(available, key=available.get) if available else ""
            metric_name = ""
            for config in CONFIG_ORDER:
                row = by_key.get((dataset, target, config))
                if row:
                    metric_name = row["metric_name"]
                    break
            output.append(
                {
                    "dataset": dataset,
                    "target": target,
                    "metric_name": metric_name,
                    **values,
                    "best_config": best_config,
                    "best_value": available.get(best_config),
                    "complete_configs": len(available),
                }
            )
    return output


def delta_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(row["dataset"], row["target"], row["config"]): row for row in rows}
    output: list[dict[str, Any]] = []
    for dataset in EXPECTED_DATASETS:
        for target in TARGET_ORDER:
            sv2 = by_key.get((dataset, target, "baseline_sv2"))
            ours = by_key.get((dataset, target, "baseline_ours"))
            sv2_value = sv2["metric_value"] if sv2 else None
            ours_value = ours["metric_value"] if ours else None
            metric_name = (sv2 or ours or {}).get("metric_name", "")
            for config in CONFIG_ORDER:
                row = by_key.get((dataset, target, config))
                value = row["metric_value"] if row else None
                output.append(
                    {
                        "dataset": dataset,
                        "target": target,
                        "metric_name": metric_name,
                        "config": config,
                        "baseline_sv2": sv2_value,
                        "baseline_ours": ours_value,
                        "value": value,
                        "delta_vs_sv2": None if value is None or sv2_value is None else value - sv2_value,
                        "delta_vs_ours": None if value is None or ours_value is None else value - ours_value,
                        "beats_sv2": bool(value is not None and sv2_value is not None and value > sv2_value),
                        "beats_ours": bool(value is not None and ours_value is not None and value > ours_value),
                        "beats_both": bool(
                            value is not None
                            and sv2_value is not None
                            and ours_value is not None
                            and value > sv2_value
                            and value > ours_value
                        ),
                    }
                )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def markdown_report(matrix: list[dict[str, Any]], deltas: list[dict[str, Any]], detail_rows: list[dict[str, Any]]) -> str:
    counts = Counter((row["dataset"], row["target"]) for row in detail_rows)
    complete_cells = sum(1 for row in matrix if int(row["complete_configs"]) == len(CONFIG_ORDER))
    hybrid_deep_rows = [row for row in deltas if row["config"] == "hybrid_deep" and row["value"] is not None]
    hybrid_win_both = sum(1 for row in hybrid_deep_rows if row["beats_both"])

    lines = [
        "# SV2/Ours Hybrid Boost Report",
        "",
        "## Coverage",
        "",
        f"- Included unique evaluated runs: {len(detail_rows)} / {EXPECTED_ROWS}",
        f"- Complete dataset-target cells: {complete_cells} / {len(EXPECTED_DATASETS) * len(TARGET_ORDER)}",
        f"- hybrid_deep beats both baselines: {hybrid_win_both} / {len(hybrid_deep_rows)}",
        "",
        "| dataset | target118 | target60 | target28 |",
        "|---|---:|---:|---:|",
    ]
    for dataset in EXPECTED_DATASETS:
        lines.append(
            "| {dataset} | {t118} | {t60} | {t28} |".format(
                dataset=dataset,
                t118=counts[(dataset, "target118")],
                t60=counts[(dataset, "target60")],
                t28=counts[(dataset, "target28")],
            )
        )

    lines.extend(
        [
            "",
            "## Best Config",
            "",
            "| dataset | target | metric | best config | best value |",
            "|---|---|---|---|---:|",
        ]
    )
    for row in matrix:
        lines.append(
            f"| {row['dataset']} | {row['target']} | {row['metric_name']} | "
            f"{row['best_config']} | {fmt(row['best_value'])} |"
        )

    lines.extend(
        [
            "",
            "## hybrid_deep Check",
            "",
            "| dataset | target | metric | sv2 | ours | hybrid_deep | deep-sv2 | deep-ours | beats both |",
            "|---|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in hybrid_deep_rows:
        lines.append(
            f"| {row['dataset']} | {row['target']} | {row['metric_name']} | "
            f"{fmt(row['baseline_sv2'])} | {fmt(row['baseline_ours'])} | {fmt(row['value'])} | "
            f"{fmt(row['delta_vs_sv2'])} | {fmt(row['delta_vs_ours'])} | {fmt(row['beats_both'])} |"
        )

    lines.extend(
        [
            "",
            "## Full Matrix",
            "",
            "| dataset | target | metric | baseline_sv2 | baseline_ours | hybrid_deep | hybrid_mid | hybrid_shallow | hybrid_mid_only |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in matrix:
        lines.append(
            f"| {row['dataset']} | {row['target']} | {row['metric_name']} | "
            f"{fmt(row['baseline_sv2'])} | {fmt(row['baseline_ours'])} | {fmt(row['hybrid_deep'])} | "
            f"{fmt(row['hybrid_mid'])} | {fmt(row['hybrid_shallow'])} | {fmt(row['hybrid_mid_only'])} |"
        )

    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize SV2/boost hybrid results")
    parser.add_argument("--summary-csv", required=True, help="summary.csv produced by summarize_results.py")
    parser.add_argument("--output-dir", required=True, help="Directory for hybrid report artifacts")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary_csv = Path(args.summary_csv)
    output_dir = ensure_dir(Path(args.output_dir))

    rows = latest_unique(enrich_rows(load_summary_rows(summary_csv)))
    matrix = matrix_rows(rows)
    deltas = delta_rows(rows)

    write_csv(output_dir / "hybrid_matrix.csv", matrix)
    write_csv(output_dir / "delta_report.csv", deltas)
    write_csv(output_dir / "detail.csv", rows)
    (output_dir / "report.md").write_text(markdown_report(matrix, deltas, rows), encoding="utf-8")

    print(f"Included unique evaluated runs: {len(rows)} / {EXPECTED_ROWS}")
    print(f"Saved matrix: {output_dir / 'hybrid_matrix.csv'}")
    print(f"Saved deltas: {output_dir / 'delta_report.csv'}")
    print(f"Saved report: {output_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
