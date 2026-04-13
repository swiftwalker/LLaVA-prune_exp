"""Summarize adaptive hyperparameter sweeps against fixed-strategy references."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


LLAVA_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_BASE_DIR = LLAVA_ROOT / "entropy_exp" / "outputs" / "analysis" / "adaptive_hparams"
TARGET_DATASETS = {"gqa", "textvqa"}
TARGET_ADAPTIVE_STRATEGY = "sparsevlm_adaptive_stratified"
REFERENCE_STRATEGIES = {"baseline", "random", "sparsevlm"}
HIGH_RATIO_VALUES = {0.5, 0.6, 0.7}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def sanitize_label(label: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in label).strip("._-") or "adaptive_hparams"


def resolve_output_dir(output_dir: str | None, label: str) -> Path:
    if output_dir:
        path = Path(output_dir)
        if not path.is_absolute():
            path = LLAVA_ROOT / path
        return ensure_dir(path.resolve())
    return ensure_dir(DEFAULT_OUTPUT_BASE_DIR / sanitize_label(label))


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_yamlish(value: str | None) -> Any:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return yaml.safe_load(text)
    except Exception:
        return text


def parse_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_single_layer(value: Any) -> int | None:
    parsed = parse_yamlish(value) if isinstance(value, str) else value
    if isinstance(parsed, list):
        if len(parsed) == 1:
            return int(parsed[0])
        return None
    if isinstance(parsed, int):
        return int(parsed)
    return None


def parse_single_ratio(value: Any) -> float | None:
    parsed = parse_yamlish(value) if isinstance(value, str) else value
    if isinstance(parsed, list):
        if len(parsed) == 1:
            return float(parsed[0])
        if len(parsed) == 0:
            return None
        return None
    if isinstance(parsed, (int, float)):
        return float(parsed)
    return None


def normalize_strategy(row: dict[str, str]) -> str | None:
    strategy = (row.get("strategy") or "").strip()
    run_mode = (row.get("run_mode") or "").strip()
    run_name = (row.get("run_name") or "").strip()
    ratio = parse_single_ratio(row.get("prune_ratio"))
    if run_mode == "baseline" or strategy == "baseline" or "_baseline_" in run_name:
        return "baseline"
    if strategy in {"random", "sparsevlm", "sparsevlm_adaptive_stratified"}:
        return strategy
    if ratio == 0.0 and strategy == "attn_score":
        return "baseline"
    return None


def extract_adaptive_params(row: dict[str, str]) -> tuple[int | None, float | None]:
    grid_size = parse_float(row.get("adaptive_grid_size"))
    high_ratio = parse_float(row.get("adaptive_high_ratio"))
    if grid_size is not None and high_ratio is not None:
        return int(grid_size), float(high_ratio)

    pruning_cfg = parse_yamlish(row.get("pruning_config_json")) or {}
    adaptive_cfg = (pruning_cfg.get("sparsevlm_adaptive_stratified") or {}) if isinstance(pruning_cfg, dict) else {}
    grid_size = adaptive_cfg.get("grid_size")
    high_ratio = adaptive_cfg.get("high_ratio")
    return (
        int(grid_size) if grid_size is not None else None,
        float(high_ratio) if high_ratio is not None else None,
    )


def parse_summary_row(row: dict[str, str], *, source_label: str) -> dict[str, Any] | None:
    dataset = (row.get("dataset") or "").strip()
    if dataset not in TARGET_DATASETS:
        return None

    strategy = normalize_strategy(row)
    if strategy is None:
        return None

    metric = parse_float(row.get("primary_metric_value"))
    if metric is None:
        return None

    layer = parse_single_layer(row.get("effective_prune_layers"))
    if layer is None:
        layer = parse_single_layer(row.get("prune_layers"))
    ratio = parse_single_ratio(row.get("prune_ratio"))
    grid_size, high_ratio = extract_adaptive_params(row)

    return {
        "dataset": dataset,
        "strategy": strategy,
        "run_name": row.get("run_name") or "",
        "run_dir": row.get("run_dir") or "",
        "metric": metric,
        "layer": layer,
        "ratio": ratio,
        "grid_size": grid_size,
        "high_ratio": high_ratio,
        "source_label": source_label,
    }


def load_summary_entries(path: Path) -> list[dict[str, Any]]:
    return [
        parsed
        for row in load_csv_rows(path)
        if (parsed := parse_summary_row(row, source_label=path.stem)) is not None
    ]


def best_entry(entries: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        entries,
        key=lambda item: (
            item["metric"],
            item["ratio"] if item["ratio"] is not None else -1.0,
            -(item["high_ratio"] if item["high_ratio"] is not None else -1.0),
            -(item["grid_size"] if item["grid_size"] is not None else -1),
            -(item["layer"] if item["layer"] is not None else -1),
            item["run_name"],
        ),
    )


def build_reference_lookups(reference_entries: list[dict[str, Any]]) -> tuple[dict[str, float], dict[tuple[str, str, int, float], float]]:
    baseline_lookup: dict[str, float] = {}
    fixed_lookup: dict[tuple[str, str, int, float], float] = {}
    for entry in reference_entries:
        dataset = entry["dataset"]
        strategy = entry["strategy"]
        metric = entry["metric"]
        if strategy == "baseline":
            baseline_lookup[dataset] = max(metric, baseline_lookup.get(dataset, float("-inf")))
            continue
        if entry["layer"] is None or entry["ratio"] is None:
            continue
        key = (dataset, strategy, int(entry["layer"]), round(float(entry["ratio"]), 2))
        fixed_lookup[key] = max(metric, fixed_lookup.get(key, float("-inf")))
    return baseline_lookup, fixed_lookup


def enrich_adaptive_entries(
    adaptive_entries: list[dict[str, Any]],
    baseline_lookup: dict[str, float],
    fixed_lookup: dict[tuple[str, str, int, float], float],
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for entry in adaptive_entries:
        dataset = entry["dataset"]
        layer = int(entry["layer"])
        ratio = round(float(entry["ratio"]), 2)
        sparsevlm_metric = fixed_lookup.get((dataset, "sparsevlm", layer, ratio))
        random_metric = fixed_lookup.get((dataset, "random", layer, ratio))
        baseline_metric = baseline_lookup.get(dataset)
        enriched.append(
            {
                **entry,
                "sparsevlm_metric": sparsevlm_metric,
                "random_metric": random_metric,
                "baseline_metric": baseline_metric,
                "delta_vs_sparsevlm": (
                    entry["metric"] - sparsevlm_metric if sparsevlm_metric is not None else None
                ),
                "delta_vs_random": (
                    entry["metric"] - random_metric if random_metric is not None else None
                ),
                "delta_vs_baseline": (
                    entry["metric"] - baseline_metric if baseline_metric is not None else None
                ),
            }
        )
    return enriched


def require_reference_coverage(
    adaptive_entries: list[dict[str, Any]],
    baseline_lookup: dict[str, float],
    fixed_lookup: dict[tuple[str, str, int, float], float],
) -> None:
    missing: list[str] = []
    for entry in adaptive_entries:
        dataset = entry["dataset"]
        layer = int(entry["layer"])
        ratio = round(float(entry["ratio"]), 2)
        if dataset not in baseline_lookup:
            missing.append(f"baseline::{dataset}")
        for strategy in ("random", "sparsevlm"):
            key = (dataset, strategy, layer, ratio)
            if key not in fixed_lookup:
                missing.append(f"{strategy}::{dataset}::l{layer}::r{ratio}")
    if missing:
        preview = ", ".join(sorted(set(missing))[:10])
        raise ValueError(f"Reference summaries are missing required coverage: {preview}")


def rowify(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": entry["dataset"],
        "strategy": entry["strategy"],
        "layer": entry["layer"],
        "ratio": entry["ratio"],
        "grid_size": entry["grid_size"],
        "high_ratio": entry["high_ratio"],
        "metric": entry["metric"],
        "sparsevlm_metric": entry.get("sparsevlm_metric", ""),
        "random_metric": entry.get("random_metric", ""),
        "baseline_metric": entry.get("baseline_metric", ""),
        "delta_vs_sparsevlm": entry.get("delta_vs_sparsevlm", ""),
        "delta_vs_random": entry.get("delta_vs_random", ""),
        "delta_vs_baseline": entry.get("delta_vs_baseline", ""),
        "run_name": entry.get("run_name", ""),
        "run_dir": entry.get("run_dir", ""),
        "source_label": entry.get("source_label", ""),
    }


def build_best_overall_rows(adaptive_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in sorted({entry["dataset"] for entry in adaptive_entries}):
        dataset_entries = [entry for entry in adaptive_entries if entry["dataset"] == dataset]
        overall = best_entry(dataset_entries)
        high_ratio_entries = [entry for entry in dataset_entries if round(float(entry["ratio"]), 2) in HIGH_RATIO_VALUES]
        high_ratio_best = best_entry(high_ratio_entries) if high_ratio_entries else overall
        rows.append(
            {
                "dataset": dataset,
                "best_layer": overall["layer"],
                "best_ratio": overall["ratio"],
                "best_grid_size": overall["grid_size"],
                "best_high_ratio": overall["high_ratio"],
                "best_metric": overall["metric"],
                "best_delta_vs_sparsevlm": overall["delta_vs_sparsevlm"],
                "best_delta_vs_random": overall["delta_vs_random"],
                "best_delta_vs_baseline": overall["delta_vs_baseline"],
                "best_high_ratio_layer": high_ratio_best["layer"],
                "best_high_ratio_ratio": high_ratio_best["ratio"],
                "best_high_ratio_grid_size": high_ratio_best["grid_size"],
                "best_high_ratio_high_ratio": high_ratio_best["high_ratio"],
                "best_high_ratio_metric": high_ratio_best["metric"],
                "best_high_ratio_delta_vs_sparsevlm": high_ratio_best["delta_vs_sparsevlm"],
            }
        )
    return rows


def build_best_by_ratio_rows(adaptive_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, int, float], list[dict[str, Any]]] = {}
    for entry in adaptive_entries:
        key = (entry["dataset"], int(entry["layer"]), round(float(entry["ratio"]), 2))
        grouped.setdefault(key, []).append(entry)
    for (dataset, layer, ratio), entries in sorted(grouped.items()):
        best = best_entry(entries)
        rows.append(
            {
                "dataset": dataset,
                "layer": layer,
                "ratio": ratio,
                "best_grid_size": best["grid_size"],
                "best_high_ratio": best["high_ratio"],
                "best_metric": best["metric"],
                "best_delta_vs_sparsevlm": best["delta_vs_sparsevlm"],
                "best_delta_vs_random": best["delta_vs_random"],
                "best_delta_vs_baseline": best["delta_vs_baseline"],
                "run_name": best["run_name"],
            }
        )
    return rows


def build_high_ratio_rows(adaptive_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        rowify(entry)
        for entry in adaptive_entries
        if round(float(entry["ratio"]), 2) in HIGH_RATIO_VALUES
    ]
    rows.sort(key=lambda row: (row["dataset"], row["layer"], row["ratio"], row["grid_size"], row["high_ratio"]))
    return rows


def build_alpha_schedule_rows(adaptive_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for entry in adaptive_entries:
        if round(float(entry["ratio"]), 2) in HIGH_RATIO_VALUES:
            grouped.setdefault((entry["dataset"], int(entry["layer"])), []).append(entry)

    for (dataset, layer), entries in sorted(grouped.items()):
        best_by_ratio: dict[str, dict[str, Any]] = {}
        for ratio in sorted(HIGH_RATIO_VALUES):
            ratio_entries = [entry for entry in entries if round(float(entry["ratio"]), 2) == ratio]
            if ratio_entries:
                best = best_entry(ratio_entries)
                best_by_ratio[f"{ratio:.1f}"] = {
                    "grid_size": best["grid_size"],
                    "high_ratio": best["high_ratio"],
                    "metric": best["metric"],
                    "delta_vs_sparsevlm": best["delta_vs_sparsevlm"],
                }

        best_alphas = [payload["high_ratio"] for payload in best_by_ratio.values()]
        positive_gain_count = sum(
            1
            for payload in best_by_ratio.values()
            if payload["delta_vs_sparsevlm"] is not None and payload["delta_vs_sparsevlm"] > 0
        )
        recommend_schedule = len(set(best_alphas)) > 1 and positive_gain_count >= 2
        rows.append(
            {
                "dataset": dataset,
                "layer": layer,
                "best_params_by_ratio_json": json.dumps(best_by_ratio, ensure_ascii=False, sort_keys=True),
                "best_alpha_r0p5": best_by_ratio.get("0.5", {}).get("high_ratio", ""),
                "best_alpha_r0p6": best_by_ratio.get("0.6", {}).get("high_ratio", ""),
                "best_alpha_r0p7": best_by_ratio.get("0.7", {}).get("high_ratio", ""),
                "positive_gain_count_vs_sparsevlm": positive_gain_count,
                "recommend_ratio_conditioned_alpha_schedule": recommend_schedule,
            }
        )
    return rows


def build_report_markdown(
    adaptive_entries: list[dict[str, Any]],
    best_overall_rows: list[dict[str, Any]],
    best_by_ratio_rows: list[dict[str, Any]],
    alpha_schedule_rows: list[dict[str, Any]],
    reference_summary_paths: list[Path],
) -> str:
    lines = [
        "# Adaptive Hparam Round 1 Report",
        "",
        f"- Generated at: {datetime.now().isoformat(timespec='seconds')}",
        f"- Adaptive runs analyzed: {len(adaptive_entries)}",
        f"- Reference summaries: {', '.join(str(path) for path in reference_summary_paths)}",
        "",
        "## Best Overall By Dataset",
    ]
    for row in best_overall_rows:
        lines.append(
            f"- `{row['dataset']}`: best overall = layer {row['best_layer']}, ratio {row['best_ratio']}, "
            f"G={row['best_grid_size']}, alpha={row['best_high_ratio']}, metric={row['best_metric']:.4f}; "
            f"best high-ratio = layer {row['best_high_ratio_layer']}, ratio {row['best_high_ratio_ratio']}, "
            f"G={row['best_high_ratio_grid_size']}, alpha={row['best_high_ratio_high_ratio']}, "
            f"metric={row['best_high_ratio_metric']:.4f}"
        )

    lines.extend(["", "## Ratio Sensitivity"])
    for dataset in sorted({row["dataset"] for row in best_by_ratio_rows}):
        dataset_rows = [row for row in best_by_ratio_rows if row["dataset"] == dataset]
        lines.append(f"- `{dataset}`:")
        for layer in sorted({row["layer"] for row in dataset_rows}):
            layer_rows = [row for row in dataset_rows if row["layer"] == layer]
            params = [
                f"r={row['ratio']}: G={row['best_grid_size']}, alpha={row['best_high_ratio']}"
                for row in layer_rows
            ]
            lines.append(f"  layer {layer} -> " + "; ".join(params))

    lines.extend(["", "## Schedule Recommendation"])
    for row in alpha_schedule_rows:
        verdict = "recommend" if row["recommend_ratio_conditioned_alpha_schedule"] else "not recommended yet"
        lines.append(
            f"- `{row['dataset']}` layer {row['layer']}: {verdict} "
            f"(positive high-ratio wins vs sparsevlm = {row['positive_gain_count_vs_sparsevlm']})"
        )
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report adaptive hyperparameter sweep results")
    parser.add_argument("--adaptive-summary", required=True, help="Round 1 adaptive summary.csv")
    parser.add_argument(
        "--reference-summary",
        nargs="+",
        required=True,
        help="One or more reference summary.csv files containing baseline/random/sparsevlm results",
    )
    parser.add_argument("--output-dir", help="Directory to write CSV and Markdown reports")
    parser.add_argument("--label", default="adaptive_hparam_round1", help="Default output-dir label")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    adaptive_summary_path = Path(args.adaptive_summary)
    if not adaptive_summary_path.is_absolute():
        adaptive_summary_path = (LLAVA_ROOT / adaptive_summary_path).resolve()
    reference_summary_paths = []
    for raw_path in args.reference_summary:
        path = Path(raw_path)
        if not path.is_absolute():
            path = (LLAVA_ROOT / path).resolve()
        reference_summary_paths.append(path)

    output_dir = resolve_output_dir(args.output_dir, args.label)

    adaptive_entries = [
        entry
        for entry in load_summary_entries(adaptive_summary_path)
        if entry["strategy"] == TARGET_ADAPTIVE_STRATEGY
    ]
    if not adaptive_entries:
        raise ValueError(f"No adaptive rows found in summary: {adaptive_summary_path}")

    reference_entries: list[dict[str, Any]] = []
    for path in reference_summary_paths:
        reference_entries.extend(
            entry for entry in load_summary_entries(path) if entry["strategy"] in REFERENCE_STRATEGIES
        )
    baseline_lookup, fixed_lookup = build_reference_lookups(reference_entries)
    require_reference_coverage(adaptive_entries, baseline_lookup, fixed_lookup)
    adaptive_entries = enrich_adaptive_entries(adaptive_entries, baseline_lookup, fixed_lookup)

    best_overall_rows = build_best_overall_rows(adaptive_entries)
    best_by_ratio_rows = build_best_by_ratio_rows(adaptive_entries)
    high_ratio_rows = build_high_ratio_rows(adaptive_entries)
    alpha_schedule_rows = build_alpha_schedule_rows(adaptive_entries)

    write_csv(
        output_dir / "best_overall_by_dataset.csv",
        best_overall_rows,
        [
            "dataset",
            "best_layer",
            "best_ratio",
            "best_grid_size",
            "best_high_ratio",
            "best_metric",
            "best_delta_vs_sparsevlm",
            "best_delta_vs_random",
            "best_delta_vs_baseline",
            "best_high_ratio_layer",
            "best_high_ratio_ratio",
            "best_high_ratio_grid_size",
            "best_high_ratio_high_ratio",
            "best_high_ratio_metric",
            "best_high_ratio_delta_vs_sparsevlm",
        ],
    )
    write_csv(
        output_dir / "best_by_ratio.csv",
        best_by_ratio_rows,
        [
            "dataset",
            "layer",
            "ratio",
            "best_grid_size",
            "best_high_ratio",
            "best_metric",
            "best_delta_vs_sparsevlm",
            "best_delta_vs_random",
            "best_delta_vs_baseline",
            "run_name",
        ],
    )
    write_csv(
        output_dir / "high_ratio_slice_report.csv",
        high_ratio_rows,
        [
            "dataset",
            "strategy",
            "layer",
            "ratio",
            "grid_size",
            "high_ratio",
            "metric",
            "sparsevlm_metric",
            "random_metric",
            "baseline_metric",
            "delta_vs_sparsevlm",
            "delta_vs_random",
            "delta_vs_baseline",
            "run_name",
            "run_dir",
            "source_label",
        ],
    )
    write_csv(
        output_dir / "alpha_schedule_candidates.csv",
        alpha_schedule_rows,
        [
            "dataset",
            "layer",
            "best_params_by_ratio_json",
            "best_alpha_r0p5",
            "best_alpha_r0p6",
            "best_alpha_r0p7",
            "positive_gain_count_vs_sparsevlm",
            "recommend_ratio_conditioned_alpha_schedule",
        ],
    )
    (output_dir / "report.md").write_text(
        build_report_markdown(
            adaptive_entries,
            best_overall_rows,
            best_by_ratio_rows,
            alpha_schedule_rows,
            reference_summary_paths,
        ),
        encoding="utf-8",
    )

    print(f"Adaptive summary: {adaptive_summary_path}")
    print(f"Reference summaries: {len(reference_summary_paths)}")
    print(f"Output dir: {output_dir}")
    print(f"Adaptive rows analyzed: {len(adaptive_entries)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
