"""Aggregate patch-retention distributions from pruning run stats."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - runtime dependency guard
    raise RuntimeError(
        "patch_distribution_report.py requires matplotlib with the Agg backend available"
    ) from exc


LLAVA_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_BASE_DIR = (
    LLAVA_ROOT / "entropy_exp" / "outputs" / "analysis" / "patch_distribution"
)
STRATEGY_ORDER = [
    "baseline",
    "random",
    "sparsevlm",
    "sparsevlm_adaptive_stratified",
]
LAYER_ORDER = [1, 2, 3]
RATIO_ORDER = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sanitize_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("._-") or "patch_distribution"


def resolve_output_dir(output_dir: str | None, label: str) -> Path:
    if output_dir:
        path = Path(output_dir)
        if not path.is_absolute():
            path = LLAVA_ROOT / path
        return ensure_dir(path.resolve())
    return ensure_dir(DEFAULT_OUTPUT_BASE_DIR / sanitize_label(label))


def iter_stats(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


def collect_completed_run_dirs_from_state_dir(state_dir: Path) -> list[Path]:
    attempts_dir = state_dir / "attempts"
    if not attempts_dir.is_dir():
        raise ValueError(f"Scheduler attempts directory not found: {attempts_dir}")

    run_dirs: list[Path] = []
    for path in sorted(attempts_dir.glob("*.json")):
        payload = load_json(path)
        if payload.get("status") != "completed":
            continue
        run_dir = payload.get("run_dir")
        if not run_dir:
            raise ValueError(f"Completed attempt is missing run_dir: {path}")
        run_dirs.append(Path(run_dir).resolve())
    if not run_dirs:
        raise ValueError(f"No completed run_dir values found under: {attempts_dir}")
    return run_dirs


def infer_patch_per_row(v_token_num: int) -> int:
    patch_per_row = int(round(math.sqrt(v_token_num)))
    if patch_per_row * patch_per_row != v_token_num:
        raise ValueError(f"v_token_num must form a square grid, got {v_token_num}")
    return patch_per_row


def create_aggregate_record(
    *,
    dataset: str,
    strategy: str,
    layer: int,
    ratio: float,
    run_dir: Path,
    run_name: str,
    patch_per_row: int,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "strategy": strategy,
        "layer": layer,
        "ratio": ratio,
        "run_dir": str(run_dir),
        "run_name": run_name,
        "patch_per_row": patch_per_row,
        "sample_count": 0,
        "sum_kept_tokens": 0.0,
        "sum_pruned_tokens": 0.0,
        "keep_count_map": np.zeros((patch_per_row, patch_per_row), dtype=np.float64),
        "pruned_count_map": np.zeros((patch_per_row, patch_per_row), dtype=np.float64),
        "high_keep_count_map": None,
        "low_keep_count_map": None,
        "stratum_quota_sum": None,
    }


def add_patch_indices(count_map: np.ndarray, patch_indices: list[int], patch_per_row: int) -> None:
    if not patch_indices:
        return
    indices = np.asarray(patch_indices, dtype=np.int64)
    if np.any(indices < 0) or np.any(indices >= patch_per_row * patch_per_row):
        raise ValueError(
            f"Patch indices out of range for {patch_per_row}x{patch_per_row} grid: {patch_indices[:8]}"
        )
    rows = indices // patch_per_row
    cols = indices % patch_per_row
    np.add.at(count_map, (rows, cols), 1.0)


def detect_layer_indices(sample_stats: dict[str, Any]) -> list[int]:
    layers = set()
    for key in sample_stats:
        match = re.fullmatch(r"layer_(\d+)_ratio", key)
        if match:
            layers.add(int(match.group(1)))
    return sorted(layers)


def get_patch_indices(sample_stats: dict[str, Any], layer_idx: int, patch_per_row: int) -> tuple[list[int], list[int]]:
    keep_patch = sample_stats.get(f"layer_{layer_idx}_keep_patch_indices")
    if keep_patch is None:
        keep_patch = sample_stats.get(f"layer_{layer_idx}_keep_indices")
    if keep_patch is None:
        raise ValueError(f"Missing keep indices for layer {layer_idx}")

    keep_patch = [int(value) for value in keep_patch]

    pruned_patch = sample_stats.get(f"layer_{layer_idx}_pruned_patch_indices")
    if pruned_patch is None:
        pruned_patch = sample_stats.get(f"layer_{layer_idx}_pruned_indices")
    if pruned_patch is None:
        all_indices = set(range(patch_per_row * patch_per_row))
        pruned_patch = sorted(all_indices - set(keep_patch))
    else:
        pruned_patch = [int(value) for value in pruned_patch]
    return keep_patch, pruned_patch


def aggregate_run_dir(run_dir: Path) -> list[dict[str, Any]]:
    config_path = run_dir / "config.yaml"
    stats_path = run_dir / "stats.jsonl"
    if not config_path.is_file():
        raise ValueError(f"Missing config file: {config_path}")
    if not stats_path.is_file():
        raise ValueError(f"Missing stats file: {stats_path}")

    config = load_yaml(config_path)
    pruning = config.get("pruning", {}) or {}
    run_meta = config.get("_run_meta", {}) or {}
    dataset = run_meta.get("dataset")
    strategy = run_meta.get("strategy") or pruning.get("strategy")
    if not dataset or not strategy:
        raise ValueError(f"Run metadata must include dataset and strategy: {run_dir}")

    patch_per_row = infer_patch_per_row(int(pruning.get("v_token_num", 576)))
    records: dict[tuple[int, float], dict[str, Any]] = {}

    for sample_stats in iter_stats(stats_path):
        for layer_idx in detect_layer_indices(sample_stats):
            ratio = round(float(sample_stats[f"layer_{layer_idx}_ratio"]), 1)
            key = (layer_idx, ratio)
            record = records.get(key)
            if record is None:
                record = create_aggregate_record(
                    dataset=dataset,
                    strategy=strategy,
                    layer=layer_idx,
                    ratio=ratio,
                    run_dir=run_dir,
                    run_name=run_dir.name,
                    patch_per_row=patch_per_row,
                )
                records[key] = record

            keep_patch, pruned_patch = get_patch_indices(sample_stats, layer_idx, patch_per_row)
            record["sample_count"] += 1
            record["sum_kept_tokens"] += float(sample_stats.get(f"layer_{layer_idx}_after", len(keep_patch)))
            record["sum_pruned_tokens"] += float(sample_stats.get(f"layer_{layer_idx}_pruned", len(pruned_patch)))
            add_patch_indices(record["keep_count_map"], keep_patch, patch_per_row)
            add_patch_indices(record["pruned_count_map"], pruned_patch, patch_per_row)

            high_keep_patch = sample_stats.get(f"layer_{layer_idx}_high_keep_patch_indices")
            if high_keep_patch is not None:
                if record["high_keep_count_map"] is None:
                    record["high_keep_count_map"] = np.zeros((patch_per_row, patch_per_row), dtype=np.float64)
                add_patch_indices(
                    record["high_keep_count_map"],
                    [int(value) for value in high_keep_patch],
                    patch_per_row,
                )

            low_keep_patch = sample_stats.get(f"layer_{layer_idx}_low_keep_patch_indices")
            if low_keep_patch is not None:
                if record["low_keep_count_map"] is None:
                    record["low_keep_count_map"] = np.zeros((patch_per_row, patch_per_row), dtype=np.float64)
                add_patch_indices(
                    record["low_keep_count_map"],
                    [int(value) for value in low_keep_patch],
                    patch_per_row,
                )

            stratum_quotas = sample_stats.get(f"layer_{layer_idx}_stratum_quotas")
            if stratum_quotas is not None:
                grid_size = int(round(math.sqrt(len(stratum_quotas))))
                if grid_size * grid_size != len(stratum_quotas):
                    raise ValueError(
                        f"stratum_quotas for layer {layer_idx} must form a square grid, got {len(stratum_quotas)}"
                    )
                if record["stratum_quota_sum"] is None:
                    record["stratum_quota_sum"] = np.zeros((grid_size, grid_size), dtype=np.float64)
                record["stratum_quota_sum"] += np.asarray(stratum_quotas, dtype=np.float64).reshape(grid_size, grid_size)

    return [finalize_record(record) for _, record in sorted(records.items())]


def build_region_masks(patch_per_row: int) -> dict[str, np.ndarray]:
    rows, cols = np.indices((patch_per_row, patch_per_row))
    mid = patch_per_row // 2
    quarter = patch_per_row // 4
    center_mask = (
        (rows >= quarter)
        & (rows < patch_per_row - quarter)
        & (cols >= quarter)
        & (cols < patch_per_row - quarter)
    )
    top_half = rows < mid
    bottom_half = rows >= mid
    left_half = cols < mid
    right_half = cols >= mid
    q1 = top_half & left_half
    q2 = top_half & right_half
    q3 = bottom_half & left_half
    q4 = bottom_half & right_half
    corner_extent = max(1, quarter)
    corner_mask = (
        ((rows < corner_extent) & (cols < corner_extent))
        | ((rows < corner_extent) & (cols >= patch_per_row - corner_extent))
        | ((rows >= patch_per_row - corner_extent) & (cols < corner_extent))
        | ((rows >= patch_per_row - corner_extent) & (cols >= patch_per_row - corner_extent))
    )
    return {
        "center": center_mask,
        "border": ~center_mask,
        "corner": corner_mask,
        "top_half": top_half,
        "bottom_half": bottom_half,
        "left_half": left_half,
        "right_half": right_half,
        "quadrant_q1": q1,
        "quadrant_q2": q2,
        "quadrant_q3": q3,
        "quadrant_q4": q4,
    }


def mask_mean(values: np.ndarray, mask: np.ndarray) -> float:
    masked = values[mask]
    return float(masked.mean()) if masked.size else 0.0


def finalize_record(record: dict[str, Any]) -> dict[str, Any]:
    sample_count = int(record["sample_count"])
    if sample_count <= 0:
        raise ValueError(f"Aggregated record has no samples: {record['run_dir']}")

    keep_rate_map = record["keep_count_map"] / sample_count
    prune_rate_map = record["pruned_count_map"] / sample_count
    high_keep_rate_map = (
        record["high_keep_count_map"] / sample_count if record["high_keep_count_map"] is not None else None
    )
    low_keep_rate_map = (
        record["low_keep_count_map"] / sample_count if record["low_keep_count_map"] is not None else None
    )
    stratum_quota_map = (
        record["stratum_quota_sum"] / sample_count if record["stratum_quota_sum"] is not None else None
    )

    masks = build_region_masks(int(record["patch_per_row"]))
    finalized = {
        **record,
        "keep_rate_map": keep_rate_map,
        "prune_rate_map": prune_rate_map,
        "high_keep_rate_map": high_keep_rate_map,
        "low_keep_rate_map": low_keep_rate_map,
        "stratum_quota_map": stratum_quota_map,
        "center_keep_rate": mask_mean(keep_rate_map, masks["center"]),
        "border_keep_rate": mask_mean(keep_rate_map, masks["border"]),
        "corner_keep_rate": mask_mean(keep_rate_map, masks["corner"]),
        "top_half_keep_rate": mask_mean(keep_rate_map, masks["top_half"]),
        "bottom_half_keep_rate": mask_mean(keep_rate_map, masks["bottom_half"]),
        "left_half_keep_rate": mask_mean(keep_rate_map, masks["left_half"]),
        "right_half_keep_rate": mask_mean(keep_rate_map, masks["right_half"]),
        "quadrant_keep_rate_q1": mask_mean(keep_rate_map, masks["quadrant_q1"]),
        "quadrant_keep_rate_q2": mask_mean(keep_rate_map, masks["quadrant_q2"]),
        "quadrant_keep_rate_q3": mask_mean(keep_rate_map, masks["quadrant_q3"]),
        "quadrant_keep_rate_q4": mask_mean(keep_rate_map, masks["quadrant_q4"]),
        "mean_kept_tokens": float(record["sum_kept_tokens"]) / sample_count,
        "mean_pruned_tokens": float(record["sum_pruned_tokens"]) / sample_count,
    }
    return finalized


def record_lookup(records: list[dict[str, Any]]) -> dict[tuple[str, str, int, float], dict[str, Any]]:
    lookup: dict[tuple[str, str, int, float], dict[str, Any]] = {}
    for record in records:
        key = (record["dataset"], record["strategy"], int(record["layer"]), round(float(record["ratio"]), 1))
        if key in lookup:
            raise ValueError(f"Duplicate aggregated record for key={key}")
        lookup[key] = record
    return lookup


def serialize_record(record: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, np.ndarray):
            payload[key] = value.tolist()
        else:
            payload[key] = value
    return payload


def write_summary_csv(records: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "dataset",
        "strategy",
        "layer",
        "ratio",
        "sample_count",
        "run_name",
        "run_dir",
        "center_keep_rate",
        "border_keep_rate",
        "corner_keep_rate",
        "top_half_keep_rate",
        "bottom_half_keep_rate",
        "left_half_keep_rate",
        "right_half_keep_rate",
        "quadrant_keep_rate_q1",
        "quadrant_keep_rate_q2",
        "quadrant_keep_rate_q3",
        "quadrant_keep_rate_q4",
        "mean_kept_tokens",
        "mean_pruned_tokens",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({name: record.get(name, "") for name in fieldnames})


def compute_limits(records: list[dict[str, Any]], field_name: str, fallback_vmax: float = 1.0) -> tuple[float, float]:
    maxima: list[float] = []
    for record in records:
        values = record.get(field_name)
        if isinstance(values, np.ndarray):
            maxima.append(float(values.max()))
    if not maxima:
        return 0.0, fallback_vmax
    return 0.0, max(maxima)


def plot_heatmap(ax, values: np.ndarray | None, title: str, *, cmap: str, vmin: float, vmax: float):
    if values is None:
        ax.axis("off")
        ax.set_title(f"{title}\nmissing", fontsize=9)
        return None
    image = ax.imshow(values, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    return image


def save_strategy_grid_figure(
    records_by_key: dict[tuple[str, str, int, float], dict[str, Any]],
    *,
    dataset: str,
    strategy: str,
    output_path: Path,
    field_name: str,
    title_prefix: str,
    cmap: str = "viridis",
    baseline_reference: bool = False,
) -> None:
    if baseline_reference:
        fig, axes = plt.subplots(
            1,
            len(LAYER_ORDER),
            figsize=(4 * len(LAYER_ORDER), 4),
            constrained_layout=True,
        )
        axes = np.atleast_1d(axes)
        candidate_records = [
            records_by_key.get((dataset, strategy, layer, 0.0)) for layer in LAYER_ORDER
        ]
    else:
        fig, axes = plt.subplots(
            len(LAYER_ORDER),
            len(RATIO_ORDER),
            figsize=(3 * len(RATIO_ORDER), 3 * len(LAYER_ORDER)),
            constrained_layout=True,
        )
        candidate_records = [
            records_by_key.get((dataset, strategy, layer, ratio))
            for layer in LAYER_ORDER
            for ratio in RATIO_ORDER
        ]

    non_missing = [record for record in candidate_records if record is not None]
    vmin, vmax = compute_limits(non_missing, field_name)
    if vmax <= vmin:
        vmax = 1.0

    image = None
    if baseline_reference:
        for axis, layer in zip(axes, LAYER_ORDER):
            record = records_by_key.get((dataset, strategy, layer, 0.0))
            image = plot_heatmap(
                axis,
                None if record is None else record[field_name],
                f"layer={layer}\nratio=0.0",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            ) or image
    else:
        for row_idx, layer in enumerate(LAYER_ORDER):
            for col_idx, ratio in enumerate(RATIO_ORDER):
                axis = axes[row_idx, col_idx]
                record = records_by_key.get((dataset, strategy, layer, ratio))
                image = plot_heatmap(
                    axis,
                    None if record is None else record[field_name],
                    f"l{layer} r{ratio}",
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                ) or image

    fig.suptitle(f"{dataset} · {strategy} · {title_prefix}", fontsize=14)
    if image is not None:
        fig.colorbar(image, ax=np.ravel(axes).tolist(), shrink=0.8)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_compare_figure(
    records_by_key: dict[tuple[str, str, int, float], dict[str, Any]],
    *,
    dataset: str,
    layer: int,
    ratio: float,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(
        1,
        len(STRATEGY_ORDER),
        figsize=(4 * len(STRATEGY_ORDER), 4),
        constrained_layout=True,
    )
    candidate_records = []
    for strategy in STRATEGY_ORDER:
        lookup_ratio = 0.0 if strategy == "baseline" else ratio
        candidate_records.append(records_by_key.get((dataset, strategy, layer, lookup_ratio)))
    non_missing = [record for record in candidate_records if record is not None]
    vmin, vmax = compute_limits(non_missing, "keep_rate_map")
    if vmax <= vmin:
        vmax = 1.0

    image = None
    for axis, strategy, record in zip(axes, STRATEGY_ORDER, candidate_records):
        image = plot_heatmap(
            axis,
            None if record is None else record["keep_rate_map"],
            f"{strategy}\nlayer={layer} ratio={0.0 if strategy == 'baseline' else ratio}",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        ) or image
    fig.suptitle(f"{dataset} · cross-strategy keep-rate compare · layer={layer} ratio={ratio}", fontsize=14)
    if image is not None:
        fig.colorbar(image, ax=axes.tolist(), shrink=0.8)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_trend_figure(records: list[dict[str, Any]], *, dataset: str, output_path: Path) -> None:
    fig, axes = plt.subplots(
        2,
        len(STRATEGY_ORDER),
        figsize=(4 * len(STRATEGY_ORDER), 7),
        sharex=True,
        constrained_layout=True,
    )
    metrics = [("center_keep_rate", "Center Keep Rate"), ("border_keep_rate", "Border Keep Rate")]
    x_values = np.asarray(RATIO_ORDER, dtype=float)

    for col_idx, strategy in enumerate(STRATEGY_ORDER):
        strategy_records = [record for record in records if record["dataset"] == dataset and record["strategy"] == strategy]
        for row_idx, (metric_name, title) in enumerate(metrics):
            axis = axes[row_idx, col_idx]
            axis.set_title(f"{strategy} · {title}")
            axis.set_xlabel("prune_ratio")
            axis.set_ylabel(title)
            axis.set_xticks(x_values)
            for layer in LAYER_ORDER:
                layer_records = [record for record in strategy_records if int(record["layer"]) == layer]
                if not layer_records:
                    continue
                layer_records = sorted(layer_records, key=lambda item: float(item["ratio"]))
                if strategy == "baseline":
                    y = np.repeat(layer_records[0][metric_name], len(x_values))
                    axis.plot(x_values, y, linestyle="--", label=f"layer {layer}")
                else:
                    xs = np.asarray([float(record["ratio"]) for record in layer_records], dtype=float)
                    ys = np.asarray([float(record[metric_name]) for record in layer_records], dtype=float)
                    axis.plot(xs, ys, marker="o", label=f"layer {layer}")
            axis.grid(alpha=0.2)
            handles, labels = axis.get_legend_handles_labels()
            if handles:
                axis.legend(fontsize=8)
    fig.suptitle(f"{dataset} · strategy-level patch-retention trends", fontsize=14)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_outputs(records: list[dict[str, Any]], run_dirs: list[Path], output_dir: Path) -> None:
    manifest_dir = ensure_dir(output_dir / "manifest")
    tables_dir = ensure_dir(output_dir / "tables")
    heatmaps_dir = ensure_dir(output_dir / "heatmaps")
    compare_dir = ensure_dir(output_dir / "compare")

    with (manifest_dir / "completed_run_dirs.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "run_count": len(run_dirs),
                "run_dirs": [str(path) for path in run_dirs],
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    serialized_records = [serialize_record(record) for record in records]
    with (tables_dir / "per_config_patch_summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"records": serialized_records}, handle, indent=2, ensure_ascii=False)
    write_summary_csv(records, tables_dir / "per_config_patch_summary.csv")

    by_key = record_lookup(records)
    for dataset in sorted({record["dataset"] for record in records}):
        dataset_heatmap_dir = ensure_dir(heatmaps_dir / dataset)
        dataset_compare_dir = ensure_dir(compare_dir / dataset)
        dataset_records = [record for record in records if record["dataset"] == dataset]

        for strategy in STRATEGY_ORDER:
            strategy_dir = ensure_dir(dataset_heatmap_dir / strategy)
            if strategy == "baseline":
                save_strategy_grid_figure(
                    by_key,
                    dataset=dataset,
                    strategy=strategy,
                    output_path=strategy_dir / "reference_grid.png",
                    field_name="keep_rate_map",
                    title_prefix="baseline reference keep-rate",
                    baseline_reference=True,
                )
                continue

            save_strategy_grid_figure(
                by_key,
                dataset=dataset,
                strategy=strategy,
                output_path=strategy_dir / "keep_rate_grid.png",
                field_name="keep_rate_map",
                title_prefix="keep-rate patch grid",
            )

        for layer in LAYER_ORDER:
            for ratio in RATIO_ORDER:
                save_compare_figure(
                    by_key,
                    dataset=dataset,
                    layer=layer,
                    ratio=ratio,
                    output_path=dataset_compare_dir / f"layer_{layer}_ratio_{str(ratio).replace('.', 'p')}_strategies.png",
                )

        save_trend_figure(
            dataset_records,
            dataset=dataset,
            output_path=dataset_compare_dir / "strategy_trends.png",
        )

        adaptive_dir = ensure_dir(dataset_heatmap_dir / "sparsevlm_adaptive_stratified")
        adaptive_records = [
            record
            for record in dataset_records
            if record["strategy"] == "sparsevlm_adaptive_stratified"
        ]
        if adaptive_records:
            save_strategy_grid_figure(
                by_key,
                dataset=dataset,
                strategy="sparsevlm_adaptive_stratified",
                output_path=adaptive_dir / "high_keep_rate_grid.png",
                field_name="high_keep_rate_map",
                title_prefix="adaptive high-keep patch grid",
            )
            save_strategy_grid_figure(
                by_key,
                dataset=dataset,
                strategy="sparsevlm_adaptive_stratified",
                output_path=adaptive_dir / "low_keep_rate_grid.png",
                field_name="low_keep_rate_map",
                title_prefix="adaptive low-keep patch grid",
            )
            save_strategy_grid_figure(
                by_key,
                dataset=dataset,
                strategy="sparsevlm_adaptive_stratified",
                output_path=adaptive_dir / "stratum_quota_grid.png",
                field_name="stratum_quota_map",
                title_prefix="adaptive mean stratum quota grid",
                cmap="magma",
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate patch-distribution reports from run stats")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--state-dir",
        type=str,
        help="Scheduler state directory; completed run_dir values are collected from attempts/*.json",
    )
    group.add_argument(
        "--run-dir",
        nargs="+",
        help="One or more completed run directories under entropy_exp/outputs/runs",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory to write manifest, tables, heatmaps, and compare figures",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.state_dir:
        state_dir = Path(args.state_dir)
        if not state_dir.is_absolute():
            state_dir = LLAVA_ROOT / state_dir
        state_dir = state_dir.resolve()
        run_dirs = collect_completed_run_dirs_from_state_dir(state_dir)
        output_dir = resolve_output_dir(args.output_dir, state_dir.name)
    else:
        run_dirs = [Path(run_dir).resolve() for run_dir in args.run_dir]
        output_dir = resolve_output_dir(args.output_dir, "patch_distribution")

    records: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        records.extend(aggregate_run_dir(run_dir))

    if not records:
        print("No patch-distribution records were produced.", file=sys.stderr)
        return 1

    records.sort(key=lambda item: (item["dataset"], item["strategy"], int(item["layer"]), float(item["ratio"])))
    write_outputs(records, run_dirs, output_dir)

    print(f"Writing patch-distribution artifacts to: {output_dir}")
    print(f"Completed runs: {len(run_dirs)}")
    print(f"Aggregated config-layer records: {len(records)}")
    print(f"Tables:   {output_dir / 'tables'}")
    print(f"Heatmaps: {output_dir / 'heatmaps'}")
    print(f"Compare:  {output_dir / 'compare'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
