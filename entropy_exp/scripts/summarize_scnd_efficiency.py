#!/usr/bin/env python3
"""Summarize SCND efficiency benchmark outputs."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Optional

import yaml


LLAVA_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = "entropy_exp/outputs/scnd_efficiency_benchmark_l2_6_16"

E2E_ROWS = [
    ("Full", "576", "eff_full_576"),
    ("SCND", "192", "eff_scnd_retain192"),
    ("SCND", "128", "eff_scnd_retain128"),
    ("SCND", "64", "eff_scnd_retain64"),
]

OVERHEAD_ROWS = [
    ("串行贪心（§3.4 定义）", ("eff_overhead_scnd_python_retain128",)),
    ("张量化实现（§3.6，默认）", ("eff_scnd_retain128", "eff_overhead_scnd_gpu_retain128")),
    ("Fast-SCND（低成本近似）", ("eff_overhead_fast_scnd_retain128",)),
    ("参照：saliency top-K", ("eff_overhead_sparsevlm_retain128",)),
]


@dataclass
class RunRecord:
    run_dir: Path
    suffix: str
    dataset: str
    timestamp: str
    strategy: str
    stats: list[dict[str, Any]]


def _fmt(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "N/A"
    return f"{float(value):.{digits}f}"


def _mean_numeric(values: Iterable[Any]) -> Optional[float]:
    numeric: list[float] = []
    for value in values:
        if value is None:
            continue
        try:
            numeric.append(float(value))
        except (TypeError, ValueError):
            continue
    if not numeric:
        return None
    return float(mean(numeric))


def _load_stats(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data if isinstance(data, dict) else {}


def _run_suffix(config: dict[str, Any], run_dir: Path) -> str:
    suffix = ((config.get("output") or {}).get("run_tag_suffix") or "").strip()
    if suffix:
        return suffix
    run_name = str(((config.get("_run_meta") or {}).get("run_name")) or run_dir.name)
    match = re.search(r"eff_[A-Za-z0-9_]+", run_name)
    return match.group(0) if match else run_name


def load_runs(root: Path, dataset: Optional[str]) -> dict[str, RunRecord]:
    by_suffix: dict[str, RunRecord] = {}
    for config_path in root.rglob("config.yaml"):
        run_dir = config_path.parent
        stats_path = run_dir / "stats.jsonl"
        if not stats_path.is_file():
            continue
        config = _load_config(config_path)
        run_meta = config.get("_run_meta") or {}
        run_dataset = str(run_meta.get("dataset") or "")
        if dataset and run_dataset != dataset:
            continue
        suffix = _run_suffix(config, run_dir)
        stats = _load_stats(stats_path)
        record = RunRecord(
            run_dir=run_dir,
            suffix=suffix,
            dataset=run_dataset,
            timestamp=str(run_meta.get("timestamp") or ""),
            strategy=str(run_meta.get("strategy") or (config.get("pruning") or {}).get("strategy") or ""),
            stats=stats,
        )
        existing = by_suffix.get(suffix)
        if existing is None or (record.timestamp, str(record.run_dir)) > (existing.timestamp, str(existing.run_dir)):
            by_suffix[suffix] = record
    return by_suffix


def timed_stats(record: RunRecord) -> list[dict[str, Any]]:
    rows = [row for row in record.stats if not bool(row.get("benchmark_is_warmup", False))]
    return rows or record.stats


def e2e_summary(record: RunRecord) -> dict[str, Any]:
    rows = timed_stats(record)
    return {
        "run_dir": str(record.run_dir),
        "samples": len(rows),
        "prefill_time_ms": _mean_numeric(row.get("prefill_time_ms") for row in rows),
        "decode_tokens_per_second": _mean_numeric(row.get("decode_tokens_per_second") for row in rows),
        "peak_memory_gb": _mean_numeric(row.get("peak_memory_gb") for row in rows),
        "end_to_end_time_ms": _mean_numeric(row.get("end_to_end_time_ms") for row in rows),
    }


def sample_selection_overhead_ms(row: dict[str, Any]) -> float:
    total = 0.0
    for key, value in row.items():
        if not re.fullmatch(r"layer_\d+_(distance_time_ms|selection_time_ms)", str(key)):
            continue
        if value is None:
            continue
        total += float(value)
    return total


def overhead_summary(record: RunRecord) -> dict[str, Any]:
    rows = timed_stats(record)
    overhead_values = [sample_selection_overhead_ms(row) for row in rows]
    overhead = _mean_numeric(overhead_values)
    prefill = _mean_numeric(row.get("prefill_time_ms") for row in rows)
    return {
        "run_dir": str(record.run_dir),
        "samples": len(rows),
        "selection_overhead_ms": overhead,
        "prefill_time_ms": prefill,
        "forward_ratio_percent": None if overhead is None or not prefill else 100.0 * overhead / prefill,
    }


def build_e2e_table(runs: dict[str, RunRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    full_time: Optional[float] = None
    summaries: dict[str, dict[str, Any]] = {}
    for _config, _retain, suffix in E2E_ROWS:
        record = runs.get(suffix)
        if record is not None:
            summaries[suffix] = e2e_summary(record)
    if "eff_full_576" in summaries:
        full_time = summaries["eff_full_576"].get("end_to_end_time_ms")
    for config, retain, suffix in E2E_ROWS:
        summary = summaries.get(suffix)
        if summary is None:
            rows.append({"config": config, "retain": retain, "missing_suffix": suffix})
            continue
        current_time = summary.get("end_to_end_time_ms")
        speedup = None
        if full_time and current_time:
            speedup = float(full_time) / float(current_time)
        rows.append({"config": config, "retain": retain, **summary, "speedup": speedup})
    return rows


def _first_existing_run(runs: dict[str, RunRecord], suffixes: tuple[str, ...]) -> tuple[Optional[str], Optional[RunRecord]]:
    for suffix in suffixes:
        record = runs.get(suffix)
        if record is not None:
            return suffix, record
    return None, None


def build_overhead_table(runs: dict[str, RunRecord]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    serial_overhead: Optional[float] = None
    for label, suffixes in OVERHEAD_ROWS:
        suffix, record = _first_existing_run(runs, suffixes)
        if record is None:
            summaries.append({"method": label, "missing_suffixes": list(suffixes)})
            continue
        summary = overhead_summary(record)
        if label.startswith("串行"):
            serial_overhead = summary.get("selection_overhead_ms")
        summaries.append({"method": label, "suffix": suffix, **summary})

    for row in summaries:
        overhead = row.get("selection_overhead_ms")
        if row.get("method", "").startswith("参照"):
            row["relative_serial_speedup"] = None
        elif serial_overhead and overhead:
            row["relative_serial_speedup"] = float(serial_overhead) / float(overhead)
        elif row.get("method", "").startswith("串行"):
            row["relative_serial_speedup"] = 1.0
        else:
            row["relative_serial_speedup"] = None
    return summaries


def write_markdown(e2e_rows: list[dict[str, Any]], overhead_rows: list[dict[str, Any]], output_dir: Path) -> None:
    e2e_lines = [
        "# SCND Efficiency E2E",
        "",
        "| 配置 | Retain | Prefill 时延 (ms) | 解码吞吐 (tok/s) | 峰值显存 (GB) | 端到端加速 (x) | samples |",
        "|:---|:---:|---:|---:|---:|---:|---:|",
    ]
    for row in e2e_rows:
        if "missing_suffix" in row:
            e2e_lines.append(f"| {row['config']} | {row['retain']} | missing `{row['missing_suffix']}` | N/A | N/A | N/A | 0 |")
            continue
        e2e_lines.append(
            "| {config} | {retain} | {prefill} | {decode} | {peak} | {speedup} | {samples} |".format(
                config=row["config"],
                retain=row["retain"],
                prefill=_fmt(row.get("prefill_time_ms")),
                decode=_fmt(row.get("decode_tokens_per_second")),
                peak=_fmt(row.get("peak_memory_gb")),
                speedup=_fmt(row.get("speedup")),
                samples=row.get("samples", 0),
            )
        )
    (output_dir / "efficiency_e2e_summary.md").write_text("\n".join(e2e_lines) + "\n", encoding="utf-8")

    overhead_lines = [
        "# SCND Efficiency Overhead",
        "",
        "| 实现 / 方法 | 选择耗时 (ms) | 占 prefill 比例 (%) | 相对串行加速 (x) | samples |",
        "|:---|---:|---:|---:|---:|",
    ]
    for row in overhead_rows:
        if "missing_suffixes" in row:
            overhead_lines.append(
                f"| {row['method']} | missing `{','.join(row['missing_suffixes'])}` | N/A | N/A | 0 |"
            )
            continue
        overhead_lines.append(
            "| {method} | {overhead} | {ratio} | {speedup} | {samples} |".format(
                method=row["method"],
                overhead=_fmt(row.get("selection_overhead_ms")),
                ratio=_fmt(row.get("forward_ratio_percent")),
                speedup=_fmt(row.get("relative_serial_speedup")),
                samples=row.get("samples", 0),
            )
        )
    (output_dir / "efficiency_overhead_summary.md").write_text(
        "\n".join(overhead_lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize SCND efficiency benchmark outputs")
    parser.add_argument("--root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_absolute():
        root = LLAVA_ROOT / root
    runs = load_runs(root, args.dataset)
    e2e_rows = build_e2e_table(runs)
    overhead_rows = build_overhead_table(runs)

    root.mkdir(parents=True, exist_ok=True)
    (root / "efficiency_e2e_summary.json").write_text(
        json.dumps(e2e_rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (root / "efficiency_overhead_summary.json").write_text(
        json.dumps(overhead_rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_markdown(e2e_rows, overhead_rows, root)
    print(f"[efficiency] summarized {len(runs)} run suffixes under {root}")
    print(f"[efficiency] wrote {root / 'efficiency_e2e_summary.md'}")
    print(f"[efficiency] wrote {root / 'efficiency_overhead_summary.md'}")


if __name__ == "__main__":
    main()
