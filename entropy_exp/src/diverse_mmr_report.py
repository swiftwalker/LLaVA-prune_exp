"""Summarize SparseVLM diverse-MMR diagnostic runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


CONFIG_ORDER = [
    "S-S-S",
    "O-S-S",
    "D-S-S",
    "D-D-S",
    "A-S-S",
    "A-A-S",
    "C-B-B",
    "C-S-S",
    "C-B-S",
    "F-T-S",
    "F-S-S",
]
TARGET_BY_RATIO = {
    (0.4791667, 0.3333333, 0.45): "retain192",
    (0.4739583, 0.6369637, 0.6727273): "retain128",
    (0.8854167, 0.5454545, 0.4333333): "retain64",
}
TARGET_ORDER = ["retain192", "retain128", "retain64"]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_summary_rows(paths: Iterable[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def parse_json(value: str | None) -> Any:
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def ratio_key(value: Any) -> tuple[float, ...] | None:
    if not isinstance(value, list):
        return None
    try:
        return tuple(round(float(item), 7) for item in value)
    except (TypeError, ValueError):
        return None


def mode_tuple(value: Any) -> tuple[str, ...] | None:
    if not isinstance(value, list):
        return None
    return tuple(str(item).upper() for item in value)


def config_name(row: dict[str, str], pruning: dict[str, Any]) -> str | None:
    strategy = row.get("strategy")
    if strategy == "sparsevlm":
        return "S-S-S"
    if strategy == "sparsevlm_boost_hybrid":
        modes = mode_tuple((pruning.get("sparsevlm_boost_hybrid") or {}).get("layer_modes"))
        if modes == ("O", "S", "S"):
            return "O-S-S"
    if strategy == "sparsevlm_diverse_mmr":
        modes = mode_tuple((pruning.get("sparsevlm_diverse_mmr") or {}).get("layer_modes"))
        if modes == ("D", "S", "S"):
            return "D-S-S"
        if modes == ("D", "D", "S"):
            return "D-D-S"
    if strategy == "sparsevlm_adaptive_diverse_mmr":
        modes = mode_tuple((pruning.get("sparsevlm_adaptive_diverse_mmr") or {}).get("layer_modes"))
        if modes == ("A", "S", "S"):
            return "A-S-S"
        if modes == ("A", "A", "S"):
            return "A-A-S"
    if strategy == "sparsevlm_scnd":
        modes = mode_tuple((pruning.get("sparsevlm_scnd") or {}).get("layer_modes"))
        if modes == ("C", "B", "B"):
            return "C-B-B"
        if modes == ("C", "S", "S"):
            return "C-S-S"
        if modes == ("C", "B", "S"):
            return "C-B-S"
    if strategy == "sparsevlm_fast_scnd":
        modes = mode_tuple((pruning.get("sparsevlm_fast_scnd") or {}).get("layer_modes"))
        if modes is None:
            prune_layers = pruning.get("prune_layers") or []
            if isinstance(prune_layers, list) and len(prune_layers) >= 3:
                modes = ("F", "T") + tuple("S" for _ in prune_layers[2:])
        if modes == ("F", "T", "S"):
            return "F-T-S"
        if modes == ("F", "S", "S"):
            return "F-S-S"
    return None


def timestamp_key(row: dict[str, Any]) -> str:
    return str(row.get("timestamp") or row.get("run_name") or "")


def mean(values: list[float]) -> float | None:
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def spatial_entropy_norm(patch_indices: Any, grid_size: int = 6, patch_per_row: int = 24) -> float | None:
    if not isinstance(patch_indices, list) or not patch_indices:
        return None
    counts = [0 for _ in range(grid_size * grid_size)]
    cell_size = patch_per_row / grid_size
    for raw_idx in patch_indices:
        try:
            patch_idx = int(raw_idx)
        except (TypeError, ValueError):
            continue
        row = patch_idx // patch_per_row
        col = patch_idx % patch_per_row
        grid_row = min(grid_size - 1, int(row / cell_size))
        grid_col = min(grid_size - 1, int(col / cell_size))
        counts[grid_row * grid_size + grid_col] += 1
    total = sum(counts)
    if total <= 0:
        return None
    entropy = 0.0
    for count in counts:
        if count <= 0:
            continue
        prob = count / total
        entropy -= prob * math.log(prob)
    return entropy / math.log(grid_size * grid_size)


def jaccard(left: Any, right: Any) -> float | None:
    if not isinstance(left, list) or not isinstance(right, list):
        return None
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    if not union:
        return None
    return len(left_set & right_set) / len(union)


def stats_summary(run_dir: str, max_samples: int = 200) -> dict[str, Any]:
    stats_path = Path(run_dir) / "stats.jsonl"
    if not stats_path.is_file():
        return {}

    pairwise_distances: list[float] = []
    saliency_mass: list[float] = []
    spatial_entropies: list[float] = []
    layer_jaccards: list[float] = []
    lambda_values: list[float] = []
    diversity_ratios: list[float] = []
    anchor_ratios: list[float] = []
    seed_ratios: list[float] = []
    saliency_floor_etas: list[float] = []
    saliency_floor_gaps: list[float] = []
    boundary_counts: list[float] = []
    core_ratios: list[float] = []
    greedy_steps_used: list[float] = []
    tie_break_band_counts: list[float] = []
    selection_rules: list[str] = []

    samples_read = 0
    with stats_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            if max_samples > 0 and samples_read >= max_samples:
                break
            sample = json.loads(line)
            samples_read += 1
            keep_by_layer: dict[int, list[int]] = {}
            for key, value in sample.items():
                if not key.startswith("layer_"):
                    continue
                parts = key.split("_")
                if len(parts) < 3 or not parts[1].isdigit():
                    continue
                layer_idx = int(parts[1])
                field = "_".join(parts[2:])
                if field == "mean_selected_pairwise_distance" and value is not None:
                    pairwise_distances.append(float(value))
                elif field == "retained_saliency_mass_ratio" and value is not None:
                    saliency_mass.append(float(value))
                elif field == "lambda_div" and value is not None:
                    lambda_values.append(float(value))
                elif field == "diversity_ratio" and value is not None:
                    diversity_ratios.append(float(value))
                elif field == "anchor_ratio" and value is not None:
                    anchor_ratios.append(float(value))
                elif field == "seed_ratio" and value is not None:
                    seed_ratios.append(float(value))
                elif field == "saliency_floor_eta" and value is not None:
                    saliency_floor_etas.append(float(value))
                elif field == "saliency_mass_selected" and value is not None:
                    floor_value = sample.get(f"layer_{layer_idx}_saliency_mass_floor")
                    if floor_value is not None:
                        saliency_floor_gaps.append(float(value) - float(floor_value))
                elif field == "boundary_count" and value is not None:
                    boundary_counts.append(float(value))
                elif field == "core_ratio" and value is not None:
                    core_ratios.append(float(value))
                elif field == "greedy_steps_used" and value is not None:
                    greedy_steps_used.append(float(value))
                elif field == "tie_break_band_count" and value is not None:
                    tie_break_band_counts.append(float(value))
                elif field == "selection_rule" and value is not None:
                    selection_rules.append(str(value))
                elif field == "keep_patch_indices":
                    keep_by_layer[layer_idx] = value
                    entropy = spatial_entropy_norm(value)
                    if entropy is not None:
                        spatial_entropies.append(entropy)
            sorted_layers = sorted(keep_by_layer)
            for left, right in zip(sorted_layers, sorted_layers[1:]):
                value = jaccard(keep_by_layer[left], keep_by_layer[right])
                if value is not None:
                    layer_jaccards.append(value)

    return {
        "avg_mean_selected_pairwise_distance": mean(pairwise_distances),
        "avg_retained_saliency_mass_ratio": mean(saliency_mass),
        "avg_grid_spatial_entropy": mean(spatial_entropies),
        "avg_keep_patch_jaccard": mean(layer_jaccards),
        "avg_lambda_div": mean(lambda_values),
        "avg_diversity_ratio": mean(diversity_ratios),
        "avg_anchor_ratio": mean(anchor_ratios),
        "avg_seed_ratio": mean(seed_ratios),
        "avg_saliency_floor_eta": mean(saliency_floor_etas),
        "avg_saliency_floor_gap": mean(saliency_floor_gaps),
        "avg_boundary_count": mean(boundary_counts),
        "avg_core_ratio": mean(core_ratios),
        "avg_greedy_steps_used": mean(greedy_steps_used),
        "avg_tie_break_band_count": mean(tie_break_band_counts),
        "selection_rule": sorted(set(selection_rules))[0] if selection_rules else None,
        "stats_samples_read": samples_read,
    }


def enrich_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        pruning = parse_json(row.get("pruning_config_json")) or {}
        target = TARGET_BY_RATIO.get(ratio_key(pruning.get("prune_ratio")))
        config = config_name(row, pruning)
        if target is None or config is None:
            continue
        try:
            metric_value = float(row.get("primary_metric_value") or "")
        except ValueError:
            continue
        enriched.append(
            {
                **row,
                "target": target,
                "config": config,
                "metric_name": row.get("primary_metric_name") or "",
                "metric_value": metric_value,
            }
        )
    return enriched


def attach_stats(rows: list[dict[str, Any]], max_samples_per_run: int) -> None:
    for row in rows:
        row.update(stats_summary(row.get("run_dir") or "", max_samples=max_samples_per_run))


def latest_unique(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in sorted(rows, key=timestamp_key):
        by_key[(row["dataset"], row["target"], row["config"])] = row
    return list(by_key.values())


def matrix_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    datasets = sorted({row["dataset"] for row in rows})
    by_key = {(row["dataset"], row["target"], row["config"]): row for row in rows}
    output: list[dict[str, Any]] = []
    for dataset in datasets:
        for target in TARGET_ORDER:
            values = {config: by_key.get((dataset, target, config), {}).get("metric_value") for config in CONFIG_ORDER}
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
    for dataset, target in sorted({(row["dataset"], row["target"]) for row in rows}):
        sv = by_key.get((dataset, target, "S-S-S"))
        boost = by_key.get((dataset, target, "O-S-S"))
        sv_value = sv["metric_value"] if sv else None
        boost_value = boost["metric_value"] if boost else None
        for config in CONFIG_ORDER:
            row = by_key.get((dataset, target, config))
            value = row["metric_value"] if row else None
            output.append(
                {
                    "dataset": dataset,
                    "target": target,
                    "metric_name": (row or sv or boost or {}).get("metric_name", ""),
                    "config": config,
                    "value": value,
                    "delta_vs_S-S-S": None if value is None or sv_value is None else value - sv_value,
                    "delta_vs_O-S-S": None if value is None or boost_value is None else value - boost_value,
                    "avg_mean_selected_pairwise_distance": None if row is None else row.get("avg_mean_selected_pairwise_distance"),
                    "avg_retained_saliency_mass_ratio": None if row is None else row.get("avg_retained_saliency_mass_ratio"),
                    "avg_grid_spatial_entropy": None if row is None else row.get("avg_grid_spatial_entropy"),
                    "avg_keep_patch_jaccard": None if row is None else row.get("avg_keep_patch_jaccard"),
                    "avg_lambda_div": None if row is None else row.get("avg_lambda_div"),
                    "avg_diversity_ratio": None if row is None else row.get("avg_diversity_ratio"),
                    "avg_anchor_ratio": None if row is None else row.get("avg_anchor_ratio"),
                    "avg_seed_ratio": None if row is None else row.get("avg_seed_ratio"),
                    "avg_saliency_floor_eta": None if row is None else row.get("avg_saliency_floor_eta"),
                    "avg_saliency_floor_gap": None if row is None else row.get("avg_saliency_floor_gap"),
                    "avg_boundary_count": None if row is None else row.get("avg_boundary_count"),
                    "avg_core_ratio": None if row is None else row.get("avg_core_ratio"),
                    "avg_greedy_steps_used": None if row is None else row.get("avg_greedy_steps_used"),
                    "avg_tie_break_band_count": None if row is None else row.get("avg_tie_break_band_count"),
                    "selection_rule": None if row is None else row.get("selection_rule"),
                }
            )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def markdown_report(matrix: list[dict[str, Any]], deltas: list[dict[str, Any]]) -> str:
    complete = sum(1 for row in matrix if int(row["complete_configs"]) == len(CONFIG_ORDER))
    dss_rows = [row for row in deltas if row["config"] == "D-S-S" and row["value"] is not None]
    ass_rows = [row for row in deltas if row["config"] == "A-S-S" and row["value"] is not None]
    cbb_rows = [row for row in deltas if row["config"] == "C-B-B" and row["value"] is not None]
    fts_rows = [row for row in deltas if row["config"] == "F-T-S" and row["value"] is not None]
    dss_wins_vs_boost = sum(1 for row in dss_rows if row["delta_vs_O-S-S"] is not None and row["delta_vs_O-S-S"] > 0)
    ass_wins_vs_boost = sum(1 for row in ass_rows if row["delta_vs_O-S-S"] is not None and row["delta_vs_O-S-S"] > 0)
    cbb_wins_vs_boost = sum(1 for row in cbb_rows if row["delta_vs_O-S-S"] is not None and row["delta_vs_O-S-S"] > 0)
    fts_wins_vs_boost = sum(1 for row in fts_rows if row["delta_vs_O-S-S"] is not None and row["delta_vs_O-S-S"] > 0)
    lines = [
        "# Diverse MMR Diagnostic Report",
        "",
        "## Coverage",
        "",
        f"- Complete dataset-target cells: {complete} / {len(matrix)}",
        f"- D-S-S beats O-S-S: {dss_wins_vs_boost} / {len(dss_rows)}",
        f"- A-S-S beats O-S-S: {ass_wins_vs_boost} / {len(ass_rows)}",
        f"- C-B-B beats O-S-S: {cbb_wins_vs_boost} / {len(cbb_rows)}",
        f"- F-T-S beats O-S-S: {fts_wins_vs_boost} / {len(fts_rows)}",
        "",
        "## Metric Matrix",
        "",
        "| dataset | target | metric | S-S-S | O-S-S | D-S-S | D-D-S | A-S-S | A-A-S | C-B-B | C-S-S | C-B-S | F-T-S | F-S-S | best |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in matrix:
        lines.append(
            "| {dataset} | {target} | {metric_name} | {sss} | {oss} | {dss} | {dds} | {ass} | {aas} | {cbb} | {css} | {cbs} | {fts} | {fss} | {best} |".format(
                dataset=row["dataset"],
                target=row["target"],
                metric_name=row["metric_name"],
                sss=fmt(row.get("S-S-S")),
                oss=fmt(row.get("O-S-S")),
                dss=fmt(row.get("D-S-S")),
                dds=fmt(row.get("D-D-S")),
                ass=fmt(row.get("A-S-S")),
                aas=fmt(row.get("A-A-S")),
                cbb=fmt(row.get("C-B-B")),
                css=fmt(row.get("C-S-S")),
                cbs=fmt(row.get("C-B-S")),
                fts=fmt(row.get("F-T-S")),
                fss=fmt(row.get("F-S-S")),
                best=row.get("best_config") or "",
            )
        )
    lines.extend(
        [
            "",
            "## Diversity Diagnostics",
            "",
            "| config | dataset | target | value | delta vs S-S-S | delta vs O-S-S | pairwise dist | saliency mass | spatial entropy | layer Jaccard | div ratio | anchor ratio | seed ratio | core ratio | greedy steps | tie band | floor gap | boundary | rule/lambda |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in deltas:
        if row["config"] not in {"D-S-S", "A-S-S", "C-B-B", "C-S-S", "C-B-S", "F-T-S", "F-S-S"}:
            continue
        lines.append(
            "| {config} | {dataset} | {target} | {value} | {dsv} | {dbo} | {dist} | {mass} | {entropy} | {jaccard} | {dratio} | {aratio} | {seed} | {core} | {greedy} | {tie_band} | {floor_gap} | {boundary} | {rule} |".format(
                config=row["config"],
                dataset=row["dataset"],
                target=row["target"],
                value=fmt(row.get("value")),
                dsv=fmt(row.get("delta_vs_S-S-S")),
                dbo=fmt(row.get("delta_vs_O-S-S")),
                dist=fmt(row.get("avg_mean_selected_pairwise_distance")),
                mass=fmt(row.get("avg_retained_saliency_mass_ratio")),
                entropy=fmt(row.get("avg_grid_spatial_entropy")),
                jaccard=fmt(row.get("avg_keep_patch_jaccard")),
                dratio=fmt(row.get("avg_diversity_ratio")),
                aratio=fmt(row.get("avg_anchor_ratio")),
                seed=fmt(row.get("avg_seed_ratio")),
                core=fmt(row.get("avg_core_ratio")),
                greedy=fmt(row.get("avg_greedy_steps_used")),
                tie_band=fmt(row.get("avg_tie_break_band_count")),
                floor_gap=fmt(row.get("avg_saliency_floor_gap")),
                boundary=fmt(row.get("avg_boundary_count")),
                rule=(row.get("selection_rule") or fmt(row.get("avg_lambda_div"))),
            )
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", action="append", required=True, help="Input summary.csv; may be repeated")
    parser.add_argument("--output-dir", required=True, help="Directory for diverse MMR report artifacts")
    parser.add_argument(
        "--stats-max-samples-per-run",
        type=int,
        default=200,
        help="Maximum stats.jsonl samples to parse per run for diagnostics; 0 means all samples",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = ensure_dir(Path(args.output_dir))
    rows = enrich_rows(load_summary_rows([Path(path) for path in args.summary_csv]))
    unique = latest_unique(rows)
    attach_stats(unique, max_samples_per_run=args.stats_max_samples_per_run)
    matrix = matrix_rows(unique)
    deltas = delta_rows(unique)

    write_csv(output_dir / "diverse_mmr_long.csv", unique)
    write_csv(output_dir / "diverse_mmr_matrix.csv", matrix)
    write_csv(output_dir / "delta_report.csv", deltas)
    (output_dir / "report.md").write_text(markdown_report(matrix, deltas), encoding="utf-8")
    with (output_dir / "diverse_mmr_summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"long": unique, "matrix": matrix, "deltas": deltas}, handle, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
