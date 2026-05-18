"""Summarize sparsevlm_boost hybrid bw_max sweep results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


EXPECTED_DATASETS = ["gqa", "textvqa", "pope", "mme", "scienceqa"]
TARGET_BY_RATIO = {
    (0.479, 0.333, 0.410): "target118",
    (0.587, 0.546, 0.444): "target60",
    (0.573, 0.780, 0.481): "target28",
}
TARGET_ORDER = ["target118", "target60", "target28"]
EXPECTED_BW = ["0.05", "0.10", "0.15", "0.20", "0.25"]
CONFIG_BY_MODES = {
    ("S", "S", "S"): "baseline_sv2",
    ("O", "S", "S"): "hybrid_mid",
    ("O", "O", "S"): "hybrid_deep",
}
CONFIG_ORDER = ["baseline_sv2", "hybrid_mid", "hybrid_deep"]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_summary_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                row["_source_summary_csv"] = str(path)
                rows.append(row)
    return rows


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


def parse_metric(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def bw_key(value: Any) -> str | None:
    if value in (None, ""):
        return None
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return None


def timestamp_key(row: dict[str, Any]) -> str:
    return str(row.get("timestamp") or row.get("run_name") or "")


def enrich_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        if row.get("strategy") != "sparsevlm_boost_hybrid":
            continue
        pruning = parse_pruning(row)
        hybrid_cfg = pruning.get("sparsevlm_boost_hybrid", {}) or {}
        config = CONFIG_BY_MODES.get(mode_key(hybrid_cfg.get("layer_modes")))
        target = TARGET_BY_RATIO.get(ratio_key(pruning.get("prune_ratio")))
        metric_value = parse_metric(row.get("primary_metric_value"))
        if config is None or target is None or metric_value is None:
            continue

        boost_weight_max = "" if config == "baseline_sv2" else bw_key(hybrid_cfg.get("boost_weight_max"))
        if config != "baseline_sv2" and boost_weight_max not in EXPECTED_BW:
            continue

        enriched.append(
            {
                "dataset": row.get("dataset") or "",
                "target": target,
                "config": config,
                "bw_max": boost_weight_max,
                "metric_name": row.get("primary_metric_name") or "",
                "metric_value": metric_value,
                "run_name": row.get("run_name") or "",
                "run_dir": row.get("run_dir") or "",
                "timestamp": row.get("timestamp") or "",
                "source_summary_csv": row.get("_source_summary_csv") or "",
            }
        )
    return enriched


def latest_unique(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in sorted(rows, key=timestamp_key):
        if row["config"] == "baseline_sv2":
            key = (row["dataset"], row["target"], row["config"])
        else:
            key = (row["dataset"], row["target"], row["config"], row["bw_max"])
        by_key[key] = row
    return list(by_key.values())


def sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dataset_rank = {dataset: idx for idx, dataset in enumerate(EXPECTED_DATASETS)}
    target_rank = {target: idx for idx, target in enumerate(TARGET_ORDER)}
    config_rank = {config: idx for idx, config in enumerate(CONFIG_ORDER)}
    bw_rank = {bw: idx for idx, bw in enumerate(EXPECTED_BW)}
    return sorted(
        rows,
        key=lambda row: (
            dataset_rank.get(row["dataset"], 999),
            target_rank.get(row["target"], 999),
            config_rank.get(row["config"], 999),
            bw_rank.get(row["bw_max"], -1),
            timestamp_key(row),
        ),
    )


def build_indexes(rows: list[dict[str, Any]]) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[tuple[str, str, str, str], dict[str, Any]]]:
    baseline: dict[tuple[str, str], dict[str, Any]] = {}
    hybrids: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        if row["config"] == "baseline_sv2":
            baseline[(row["dataset"], row["target"])] = row
        else:
            hybrids[(row["dataset"], row["target"], row["config"], row["bw_max"])] = row
    return baseline, hybrids


def build_matrix(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline, hybrids = build_indexes(rows)
    matrix: list[dict[str, Any]] = []
    for dataset in EXPECTED_DATASETS:
        for target in TARGET_ORDER:
            base = baseline.get((dataset, target))
            for bw in EXPECTED_BW:
                mid = hybrids.get((dataset, target, "hybrid_mid", bw))
                deep = hybrids.get((dataset, target, "hybrid_deep", bw))
                base_value = base["metric_value"] if base else None
                mid_value = mid["metric_value"] if mid else None
                deep_value = deep["metric_value"] if deep else None
                hybrid_values = {
                    "hybrid_mid": mid_value,
                    "hybrid_deep": deep_value,
                }
                available_hybrids = {key: value for key, value in hybrid_values.items() if value is not None}
                best_hybrid = max(available_hybrids, key=available_hybrids.get) if available_hybrids else ""
                overall_values = {"baseline_sv2": base_value, **hybrid_values}
                available_overall = {key: value for key, value in overall_values.items() if value is not None}
                best_overall = max(available_overall, key=available_overall.get) if available_overall else ""
                metric_name = (base or mid or deep or {}).get("metric_name", "")
                matrix.append(
                    {
                        "dataset": dataset,
                        "target": target,
                        "bw_max": bw,
                        "metric_name": metric_name,
                        "baseline_sv2": base_value,
                        "hybrid_mid": mid_value,
                        "hybrid_deep": deep_value,
                        "hybrid_mid_delta_vs_sv2": None if base_value is None or mid_value is None else mid_value - base_value,
                        "hybrid_deep_delta_vs_sv2": None if base_value is None or deep_value is None else deep_value - base_value,
                        "hybrid_deep_delta_vs_mid": None if mid_value is None or deep_value is None else deep_value - mid_value,
                        "best_hybrid": best_hybrid,
                        "best_hybrid_value": available_hybrids.get(best_hybrid),
                        "best_overall": best_overall,
                        "best_overall_value": available_overall.get(best_overall),
                        "complete": base_value is not None and mid_value is not None and deep_value is not None,
                    }
                )
    return matrix


def build_delta_rows(matrix: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in matrix:
        for config in ["hybrid_mid", "hybrid_deep"]:
            value = row[config]
            delta_key = f"{config}_delta_vs_sv2"
            output.append(
                {
                    "dataset": row["dataset"],
                    "target": row["target"],
                    "bw_max": row["bw_max"],
                    "metric_name": row["metric_name"],
                    "config": config,
                    "baseline_sv2": row["baseline_sv2"],
                    "value": value,
                    "delta_vs_sv2": row[delta_key],
                    "other_hybrid_delta": row["hybrid_deep_delta_vs_mid"],
                    "beats_sv2": bool(value is not None and row["baseline_sv2"] is not None and value > row["baseline_sv2"]),
                    "best_hybrid": row["best_hybrid"],
                    "best_overall": row["best_overall"],
                }
            )
    return output


def expected_missing(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, str]]]:
    baseline, hybrids = build_indexes(rows)
    missing_baseline: list[dict[str, str]] = []
    missing_hybrid: list[dict[str, str]] = []
    for dataset in EXPECTED_DATASETS:
        for target in TARGET_ORDER:
            if (dataset, target) not in baseline:
                missing_baseline.append({"dataset": dataset, "target": target, "config": "baseline_sv2"})
            for bw in EXPECTED_BW:
                for config in ["hybrid_mid", "hybrid_deep"]:
                    if (dataset, target, config, bw) not in hybrids:
                        missing_hybrid.append(
                            {"dataset": dataset, "target": target, "config": config, "bw_max": bw}
                        )
    return {"baseline": missing_baseline, "hybrid": missing_hybrid}


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


def best_by_dataset_target(matrix: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for dataset in EXPECTED_DATASETS:
        for target in TARGET_ORDER:
            candidates = [row for row in matrix if row["dataset"] == dataset and row["target"] == target]
            available = [row for row in candidates if row["best_hybrid_value"] is not None]
            best = max(available, key=lambda row: row["best_hybrid_value"]) if available else None
            if best is None:
                output.append(
                    {
                        "dataset": dataset,
                        "target": target,
                        "metric_name": "",
                        "best_hybrid": "",
                        "best_bw_max": "",
                        "best_hybrid_value": None,
                        "baseline_sv2": None,
                        "delta_vs_sv2": None,
                    }
                )
                continue
            delta = (
                None
                if best["baseline_sv2"] is None or best["best_hybrid_value"] is None
                else best["best_hybrid_value"] - best["baseline_sv2"]
            )
            output.append(
                {
                    "dataset": dataset,
                    "target": target,
                    "metric_name": best["metric_name"],
                    "best_hybrid": best["best_hybrid"],
                    "best_bw_max": best["bw_max"],
                    "best_hybrid_value": best["best_hybrid_value"],
                    "baseline_sv2": best["baseline_sv2"],
                    "delta_vs_sv2": delta,
                }
            )
    return output


def markdown_report(matrix: list[dict[str, Any]], rows: list[dict[str, Any]], missing: dict[str, list[dict[str, str]]]) -> str:
    complete_rows = sum(1 for row in matrix if row["complete"])
    missing_count = len(missing["baseline"]) + len(missing["hybrid"])
    best_rows = best_by_dataset_target(matrix)
    lines = [
        "# Hybrid Boost bw_max Sweep Report",
        "",
        "## Coverage",
        "",
        f"- Unique evaluated runs included: {len(rows)} / 165",
        f"- Complete matrix rows: {complete_rows} / 75",
        f"- Missing expected keys: {missing_count}",
        "",
        "## Best Hybrid Per Dataset/Target",
        "",
        "| dataset | target | metric | best hybrid | bw_max | baseline SV2 | best value | delta vs SV2 |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for row in best_rows:
        lines.append(
            f"| {row['dataset']} | {row['target']} | {row['metric_name']} | {row['best_hybrid']} | "
            f"{row['best_bw_max']} | {fmt(row['baseline_sv2'])} | {fmt(row['best_hybrid_value'])} | "
            f"{fmt(row['delta_vs_sv2'])} |"
        )

    lines.extend(
        [
            "",
            "## Full Matrix",
            "",
            "| dataset | target | bw_max | metric | SV2 | O-S-S | O-O-S | O-S-S minus SV2 | O-O-S minus SV2 | O-O-S minus O-S-S | best overall |",
            "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in matrix:
        lines.append(
            f"| {row['dataset']} | {row['target']} | {row['bw_max']} | {row['metric_name']} | "
            f"{fmt(row['baseline_sv2'])} | {fmt(row['hybrid_mid'])} | {fmt(row['hybrid_deep'])} | "
            f"{fmt(row['hybrid_mid_delta_vs_sv2'])} | {fmt(row['hybrid_deep_delta_vs_sv2'])} | "
            f"{fmt(row['hybrid_deep_delta_vs_mid'])} | {row['best_overall']} |"
        )

    if missing_count:
        lines.extend(["", "## Missing Keys", ""])
        for item in missing["baseline"][:20]:
            lines.append(f"- baseline missing: {item['dataset']} {item['target']} {item['config']}")
        for item in missing["hybrid"][:40]:
            lines.append(
                f"- hybrid missing: {item['dataset']} {item['target']} {item['config']} bw={item['bw_max']}"
            )
        if missing_count > 60:
            lines.append(f"- ... {missing_count - 60} more missing keys in missing_keys.json")

    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize sparsevlm_boost hybrid bw_max sweep results")
    parser.add_argument("--summary-csv", nargs="+", required=True, help="One or more summary.csv files")
    parser.add_argument("--output-dir", required=True, help="Directory for report artifacts")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary_paths = [Path(path).resolve() for path in args.summary_csv]
    output_dir = ensure_dir(Path(args.output_dir).resolve())

    rows = sort_rows(latest_unique(enrich_rows(load_summary_rows(summary_paths))))
    matrix = build_matrix(rows)
    deltas = build_delta_rows(matrix)
    missing = expected_missing(rows)

    write_csv(output_dir / "bw_sweep_long.csv", rows)
    write_csv(output_dir / "bw_sweep_matrix.csv", matrix)
    write_csv(output_dir / "delta_report.csv", deltas)
    write_csv(output_dir / "best_by_dataset_target.csv", best_by_dataset_target(matrix))
    with (output_dir / "missing_keys.json").open("w", encoding="utf-8") as handle:
        json.dump(missing, handle, indent=2, ensure_ascii=False)
    (output_dir / "report.md").write_text(markdown_report(matrix, rows, missing), encoding="utf-8")

    print(f"Included unique evaluated runs: {len(rows)} / 165")
    print(f"Complete matrix rows: {sum(1 for row in matrix if row['complete'])} / 75")
    print(f"Missing expected keys: {len(missing['baseline']) + len(missing['hybrid'])}")
    print(f"Saved report: {output_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
