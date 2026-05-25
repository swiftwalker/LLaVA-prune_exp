#!/usr/bin/env python3
"""List official method families registered under algo_compare."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from algo_compare.manifest import load_yaml
from algo_compare.paths import method_dirs


def main() -> int:
    dirs = method_dirs()
    if not dirs:
        print("No methods found.")
        return 0

    for method_dir in dirs:
        data = load_yaml(method_dir / "method.yaml")
        name = data.get("name", method_dir.name)
        repo = data.get("official", {}).get("repo_url", "")
        commit = data.get("official", {}).get("target_commit", "")
        variants = ", ".join(v.get("name", "") for v in data.get("variants", []))
        print(f"{method_dir.name}\t{name}\t{repo}\t{commit}\t{variants}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
