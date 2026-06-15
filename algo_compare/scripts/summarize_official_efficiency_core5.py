#!/usr/bin/env python3
"""Summarize official-method E2E benchmark stats for the core five datasets."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / "algo_compare" / "analysis" / "official_efficiency_core5_sv1budget"
DEFAULT_FULL_ROOT = ROOT / "entropy_exp" / "outputs" / "scnd_efficiency_5datasets_full_triton_l2_6_16"
DEFAULT_Q8_ROOT = ROOT / "entropy_exp" / "outputs" / "scnd_efficiency_5datasets_full_triton_v3_q8_l2_6_16"
DATASETS = ("gqa", "textvqa", "pope", "mme", "scienceqa")
METHODS = ("fastv", "pdrop", "sparsevlm", "divprune", "visionzip")
RETAINS = ("retain192", "retain128", "retain64")
METHOD_LABELS = {
    "fastv": "FastV",
    "pdrop": "PDROP",
    "sparsevlm": "SparseVLM-v1",
    "divprune": "DivPrune",
    "visionzip": "VisionZip",
}
PRIMARY_METRICS = {
    "gqa": "accuracy",
    "textvqa": "accuracy",
    "pope": "macro_f1",
    "mme": "overall_total_score",
    "scienceqa": "accuracy",
}


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def mean(values: list[float]) -> float | None:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    if not finite:
        return None
    return sum(finite) / len(finite)


def metric_value_from_summary(run_dir: Path, dataset: str) -> tuple[str | None, float | None]:
    official_summary = load_json(run_dir / "official_summary.json")
    if official_summary.get("metric_name") is not None:
        value = official_summary.get("metric_value")
        return official_summary.get("metric_name"), float(value) if value is not None else None
    eval_summary = load_json(run_dir / "eval" / "summary.json")
    metrics = eval_summary.get("metrics", {}) or {}
    metric_name = PRIMARY_METRICS.get(dataset)
    value = metrics.get(metric_name) if metric_name else None
    return metric_name, float(value) if value is not None else None


def stats_summary(stats_file: Path) -> dict[str, Any]:
    rows = [row for row in load_jsonl(stats_file) if not row.get("benchmark_is_warmup", False)]
    def values(key: str) -> list[float]:
        out = []
        for row in rows:
            value = row.get(key)
            if value is None:
                continue
            out.append(float(value))
        return out

    return {
        "stats_file": str(stats_file) if stats_file.is_file() else None,
        "sample_count": len(rows),
        "prefill_time_ms": mean(values("prefill_time_ms")),
        "decode_time_ms": mean(values("decode_time_ms")),
        "end_to_end_time_ms": mean(values("end_to_end_time_ms") or values("total_time_ms")),
        "decode_tokens_per_second": mean(values("decode_tokens_per_second")),
        "peak_memory_gb": mean(values("peak_memory_gb")),
        "allocated_memory_gb": mean(values("allocated_memory_gb")),
    }


def weighted_mean(records: list[dict[str, Any]], key: str) -> float | None:
    numerator = 0.0
    denominator = 0
    for record in records:
        value = record.get(key)
        weight = int(record.get("sample_count") or 0)
        if value is None or weight <= 0:
            continue
        numerator += float(value) * weight
        denominator += weight
    if denominator == 0:
        return None
    return numerator / denominator


def collect_official_records(output_root: Path) -> list[dict[str, Any]]:
    records = []
    for dataset in DATASETS:
        for method in METHODS:
            for retain_label in RETAINS:
                run_dir = output_root / "runs" / dataset / method / retain_label
                metric_name, metric_value = metric_value_from_summary(run_dir, dataset)
                stats = stats_summary(run_dir / "benchmark_stats.jsonl")
                retain = retain_label.replace("retain", "")
                records.append(
                    {
                        "source": "official",
                        "method": METHOD_LABELS[method],
                        "method_key": method,
                        "retain": retain,
                        "dataset": dataset,
                        "run_dir": str(run_dir),
                        "answers_file": str(run_dir / "answers.jsonl") if (run_dir / "answers.jsonl").is_file() else None,
                        "eval_summary": str(run_dir / "eval" / "summary.json")
                        if (run_dir / "eval" / "summary.json").is_file()
                        else None,
                        "metric_name": metric_name,
                        "metric_value": metric_value,
                        "selection_overhead_ms": None,
                        "selection_overhead_note": "not isolated",
                        **stats,
                    }
                )
    return records


def collect_reused_scnd_rows(q8_root: Path, full_root: Path) -> tuple[list[dict[str, Any]], float | None]:
    summary_path = q8_root / "v3q8_vs_full_efficiency_summary.json"
    payload = load_json(summary_path)
    rows = []
    full_e2e = None
    if not payload:
        return rows, full_e2e
    for item in payload.get("summaries", []):
        label = item.get("label")
        retain = str(item.get("retain"))
        weighted = item.get("weighted", {}) or {}
        if label == "Full baseline":
            method = "Full"
            full_e2e = weighted.get("end_to_end_time_ms")
            mean_ret = 100.0
            overhead = 0.0
        elif label == "SCND v3 q8":
            method = "SCND q8"
            mean_ret = None
            overhead = weighted.get("selection_overhead_ms")
        else:
            continue
        rows.append(
            {
                "source": "reused",
                "method": method,
                "retain": retain,
                "mean_ret_pct": mean_ret,
                "prefill_time_ms": weighted.get("prefill_time_ms"),
                "decode_time_ms": None,
                "end_to_end_time_ms": weighted.get("end_to_end_time_ms"),
                "decode_tokens_per_second": weighted.get("decode_tokens_per_second"),
                "peak_memory_gb": weighted.get("peak_memory_gb"),
                "selection_overhead_ms": overhead,
                "selection_overhead_note": "isolated" if method == "SCND q8" else "0",
                "sample_count": weighted.get("samples"),
                "speedup_vs_full": weighted.get("speedup_vs_full"),
                "reuse_summary": str(summary_path),
                "full_root": str(full_root),
                "q8_root": str(q8_root),
            }
        )
    return rows, float(full_e2e) if full_e2e is not None else None


def aggregate_official(records: list[dict[str, Any]], full_e2e: float | None) -> list[dict[str, Any]]:
    rows = []
    for method in [METHOD_LABELS[key] for key in METHODS]:
        for retain in ("192", "128", "64"):
            group = [record for record in records if record["method"] == method and record["retain"] == retain]
            e2e = weighted_mean(group, "end_to_end_time_ms")
            rows.append(
                {
                    "source": "official",
                    "method": method,
                    "retain": retain,
                    "mean_ret_pct": None,
                    "prefill_time_ms": weighted_mean(group, "prefill_time_ms"),
                    "decode_time_ms": weighted_mean(group, "decode_time_ms"),
                    "end_to_end_time_ms": e2e,
                    "decode_tokens_per_second": weighted_mean(group, "decode_tokens_per_second"),
                    "peak_memory_gb": weighted_mean(group, "peak_memory_gb"),
                    "selection_overhead_ms": None,
                    "selection_overhead_note": "not isolated",
                    "sample_count": sum(int(record.get("sample_count") or 0) for record in group),
                    "speedup_vs_full": (full_e2e / e2e) if full_e2e and e2e else None,
                    "complete_stats": sum(1 for record in group if record.get("stats_file")) == len(DATASETS),
                    "complete_eval": sum(1 for record in group if record.get("eval_summary")) == len(DATASETS),
                }
            )
    return rows


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, str):
        return value
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Official External Method Efficiency",
        "",
        "| 方法 | retain | Mean Ret% | E2E (ms) | Speedup | 选择额外开销 (ms) |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        overhead = row.get("selection_overhead_ms")
        overhead_text = fmt(overhead) if overhead is not None else row.get("selection_overhead_note", "N/A")
        lines.append(
            "| {method} | {retain} | {mean_ret} | {e2e} | {speedup}x | {overhead} |".format(
                method=row.get("method"),
                retain=row.get("retain"),
                mean_ret=fmt(row.get("mean_ret_pct")),
                e2e=fmt(row.get("end_to_end_time_ms")),
                speedup=fmt(row.get("speedup_vs_full")),
                overhead=overhead_text,
            )
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- Official external methods report E2E timing around their official `generate()` path.",
            "- Internal selection overhead is marked `not isolated` for official methods.",
            "- Mean Ret% is left `N/A` unless a no-prune metric denominator is supplied later.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--full-root", default=str(DEFAULT_FULL_ROOT))
    parser.add_argument("--q8-root", default=str(DEFAULT_Q8_ROOT))
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    full_root = Path(args.full_root).resolve()
    q8_root = Path(args.q8_root).resolve()
    official_records = collect_official_records(output_root)
    reused_rows, full_e2e = collect_reused_scnd_rows(q8_root, full_root)
    summary_rows = reused_rows + aggregate_official(official_records, full_e2e)

    write_csv(output_root / "official_efficiency_core5_sv1budget_long.csv", official_records)
    write_csv(output_root / "official_efficiency_core5_sv1budget_summary.csv", summary_rows)
    with (output_root / "official_efficiency_core5_sv1budget_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "output_root": str(output_root),
                "full_root": str(full_root),
                "q8_root": str(q8_root),
                "full_e2e_ms": full_e2e,
                "summary": summary_rows,
                "official_long": official_records,
            },
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    write_markdown(output_root / "official_efficiency_core5_sv1budget_summary.md", summary_rows)
    print(f"Wrote {output_root / 'official_efficiency_core5_sv1budget_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
