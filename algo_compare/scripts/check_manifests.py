#!/usr/bin/env python3
"""Validate algo_compare manifests and check referenced paths."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from algo_compare.manifest import iter_manifest_paths, load_manifest, validate_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", nargs="+", help="Manifest YAML files")
    parser.add_argument(
        "--strict-paths",
        action="store_true",
        help="Return non-zero when referenced paths do not exist",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    failed = False
    for manifest_path in args.manifests:
        manifest = load_manifest(manifest_path)
        errors = validate_manifest(manifest)
        print(f"[manifest] {manifest_path}")
        if errors:
            failed = True
            for error in errors:
                print(f"  ERROR {error}")
        else:
            print("  schema: OK")

        missing = []
        for run in manifest.get("runs", []):
            label = run.get("label", "<unnamed>")
            for field, path in iter_manifest_paths(run):
                status = "OK" if path.exists() else "MISSING"
                print(f"  {label}: {field}: {status} {path}")
                if status == "MISSING":
                    missing.append((label, field, path))
        if missing and args.strict_paths:
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
