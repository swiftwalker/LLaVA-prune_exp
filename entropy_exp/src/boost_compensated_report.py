"""Build a compact report for the boost/compensated diagnostic matrix."""

from __future__ import annotations

import argparse
import ast
import csv
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
    / "boost_compensated_diag"
    / "summary.csv"
)
DEFAULT_OUTPUT_DIR = LLAVA_ROOT / "entropy_exp" / "outputs" / "analysis" / "boost_compensated_diag"
STRATEGY_ORDER = [
    "sparsevlm",
    "sparsevlm_entropy_alpha",
    "sparsevlm_entropy_alpha_global",
    "sparsevlm_boost",
    "sparsevlm_compensated",
]
LAYER_IDS = (2, 8, 16)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report boost/compensated diagnostic results")
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


def stats_summary(run_dir: Path, strategy: str) -> dict[str, Any]:
    stats_path = run_dir / "stats.jsonl"
    if not stats_path.is_file():
        return {}

    boost_values: list[float] = []
    beta_values: list[float] = []
    token_boost_means: list[float] = []
    sampling_fill_counts: list[float] = []
    shuffle_seeds: set[int] = set()
    jaccard_values: list[float] = []
    use_ema_count = 0
    layer_count = 0
    sample_count = 0

    with stats_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            sample_count += 1
            previous_keep = None
            for layer_idx in LAYER_IDS:
                layer_count += 1
                if extract_scalar(line, f"layer_{layer_idx}_global_use_ema") is True:
                    use_ema_count += 1

                boost = parse_float(extract_scalar(line, f"layer_{layer_idx}_boost_weight"))
                if boost is not None:
                    boost_values.append(boost)
                beta = parse_float(extract_scalar(line, f"layer_{layer_idx}_beta"))
                if beta is not None:
                    beta_values.append(beta)
                fill_count = parse_float(extract_scalar(line, f"layer_{layer_idx}_sampling_fill_count"))
                if fill_count is not None:
                    sampling_fill_counts.append(fill_count)
                shuffle_seed = extract_scalar(line, f"layer_{layer_idx}_shuffle_seed")
                if isinstance(shuffle_seed, float):
                    shuffle_seeds.add(int(shuffle_seed))

                token_boost = extract_number_list(line, f"layer_{layer_idx}_token_boost")
                if token_boost is not None:
                    token_boost_means.append(mean(token_boost) if token_boost else 0.0)

                keep = extract_int_list(line, f"layer_{layer_idx}_keep_patch_indices")
                if previous_keep is not None and keep is not None:
                    jaccard_values.append(jaccard(previous_keep, keep))
                if keep is not None:
                    previous_keep = keep

    return {
        "stats_samples": sample_count,
        "stats_layers": layer_count,
        "mean_boost_weight": mean(boost_values) if boost_values else None,
        "mean_beta": mean(beta_values) if beta_values else None,
        "mean_token_boost": mean(token_boost_means) if token_boost_means else None,
        "mean_sampling_fill_count": mean(sampling_fill_counts) if sampling_fill_counts else None,
        "unique_shuffle_seeds": len(shuffle_seeds) if shuffle_seeds else None,
        "mean_layer_keep_jaccard": mean(jaccard_values) if jaccard_values else None,
        "global_ema_use_ratio": use_ema_count / layer_count if layer_count else None,
        "stats_strategy": strategy,
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
        if row.get("strategy") in {"sparsevlm_boost", "sparsevlm_compensated"}:
            enriched.update(stats_summary(run_dir, row["strategy"]))
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
            boost = by_key.get((dataset, cfg, "sparsevlm_boost"))
            compensated = by_key.get((dataset, cfg, "sparsevlm_compensated"))
            if not sparse or not local or not global_row or not boost or not compensated:
                continue
            for candidate in (boost, compensated):
                metric = candidate.get("metric")
                comparisons.append(
                    {
                        "dataset": dataset,
                        "config": cfg,
                        "metric_name": candidate.get("primary_metric_name"),
                        "strategy": candidate["strategy"],
                        "sparsevlm": sparse.get("metric"),
                        "entropy_alpha": local.get("metric"),
                        "entropy_alpha_global": global_row.get("metric"),
                        "candidate": metric,
                        "candidate_minus_sparsevlm": (
                            metric - sparse["metric"] if metric is not None and sparse.get("metric") is not None else None
                        ),
                        "candidate_minus_entropy_alpha": (
                            metric - local["metric"] if metric is not None and local.get("metric") is not None else None
                        ),
                        "candidate_minus_entropy_alpha_global": (
                            metric - global_row["metric"] if metric is not None and global_row.get("metric") is not None else None
                        ),
                        "mean_boost_weight": candidate.get("mean_boost_weight"),
                        "mean_beta": candidate.get("mean_beta"),
                        "mean_token_boost": candidate.get("mean_token_boost"),
                        "mean_sampling_fill_count": candidate.get("mean_sampling_fill_count"),
                        "unique_shuffle_seeds": candidate.get("unique_shuffle_seeds"),
                        "mean_layer_keep_jaccard": candidate.get("mean_layer_keep_jaccard"),
                        "global_ema_use_ratio": candidate.get("global_ema_use_ratio"),
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
        "# Boost / Compensated Diagnostic Report",
        "",
        "## Metric Comparison",
        "",
        "| dataset | config | metric | strategy | sparsevlm | entropy_alpha | entropy_alpha_global | candidate | cand-sv2 | cand-old | cand-global |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        lines.append(
            "| {dataset} | {config} | {metric_name} | {strategy} | {sparsevlm} | {entropy_alpha} | "
            "{entropy_alpha_global} | {candidate} | {candidate_minus_sparsevlm} | "
            "{candidate_minus_entropy_alpha} | {candidate_minus_entropy_alpha_global} |".format(
                dataset=row["dataset"],
                config=row["config"],
                metric_name=row["metric_name"],
                strategy=row["strategy"],
                sparsevlm=format_value(row["sparsevlm"]),
                entropy_alpha=format_value(row["entropy_alpha"]),
                entropy_alpha_global=format_value(row["entropy_alpha_global"]),
                candidate=format_value(row["candidate"]),
                candidate_minus_sparsevlm=format_value(row["candidate_minus_sparsevlm"]),
                candidate_minus_entropy_alpha=format_value(row["candidate_minus_entropy_alpha"]),
                candidate_minus_entropy_alpha_global=format_value(row["candidate_minus_entropy_alpha_global"]),
            )
        )

    lines.extend(
        [
            "",
            "## A1/A2 Stats",
            "",
            "| dataset | config | strategy | mean boost | mean beta | mean token boost | fill count | unique seeds | keep Jaccard | EMA use |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparisons:
        lines.append(
            "| {dataset} | {config} | {strategy} | {mean_boost_weight} | {mean_beta} | "
            "{mean_token_boost} | {mean_sampling_fill_count} | {unique_shuffle_seeds} | "
            "{mean_layer_keep_jaccard} | {global_ema_use_ratio} |".format(
                dataset=row["dataset"],
                config=row["config"],
                strategy=row["strategy"],
                mean_boost_weight=format_value(row["mean_boost_weight"]),
                mean_beta=format_value(row["mean_beta"]),
                mean_token_boost=format_value(row["mean_token_boost"]),
                mean_sampling_fill_count=format_value(row["mean_sampling_fill_count"]),
                unique_shuffle_seeds=format_value(row["unique_shuffle_seeds"]),
                mean_layer_keep_jaccard=format_value(row["mean_layer_keep_jaccard"]),
                global_ema_use_ratio=format_value(row["global_ema_use_ratio"]),
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
        "mean_boost_weight",
        "mean_beta",
        "mean_token_boost",
        "mean_sampling_fill_count",
        "unique_shuffle_seeds",
        "mean_layer_keep_jaccard",
        "global_ema_use_ratio",
        "stats_samples",
        "stats_layers",
    ]
    comparison_fields = [
        "dataset",
        "config",
        "metric_name",
        "strategy",
        "sparsevlm",
        "entropy_alpha",
        "entropy_alpha_global",
        "candidate",
        "candidate_minus_sparsevlm",
        "candidate_minus_entropy_alpha",
        "candidate_minus_entropy_alpha_global",
        "mean_boost_weight",
        "mean_beta",
        "mean_token_boost",
        "mean_sampling_fill_count",
        "unique_shuffle_seeds",
        "mean_layer_keep_jaccard",
        "global_ema_use_ratio",
    ]

    write_csv(output_dir / "detail.csv", rows, detail_fields)
    write_csv(output_dir / "comparison.csv", comparison_rows, comparison_fields)
    write_markdown(output_dir / "report.md", comparison_rows, rows)
    print(f"Wrote report artifacts to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
