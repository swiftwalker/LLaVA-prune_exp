#!/usr/bin/env python3
"""Print lightweight coverage summaries for manifest entries."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from algo_compare.manifest import load_manifest
from algo_compare.paths import resolve_repo_path


def summarize_csv(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    datasets = sorted({row.get("dataset", "") for row in rows if row.get("dataset")})
    strategies = sorted({row.get("strategy", "") for row in rows if row.get("strategy")})
    models = sorted({row.get("model_name", "") for row in rows if row.get("model_name")})
    metrics = sorted({row.get("primary_metric_name", "") for row in rows if row.get("primary_metric_name")})
    return {
        "rows": len(rows),
        "datasets": datasets,
        "strategies": strategies,
        "models": models,
        "metrics": metrics,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", help="Manifest YAML")
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest)
    output = []
    for run in manifest.get("runs", []):
        record = {
            "label": run.get("label"),
            "source_type": run.get("source_type"),
            "method": run.get("method"),
        }
        summary_csv = run.get("summary_csv")
        if summary_csv:
            path = resolve_repo_path(summary_csv)
            record["summary_csv"] = str(path)
            if path.is_file():
                record.update(summarize_csv(path))
            else:
                record["missing"] = True
        output.append(record)

    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
