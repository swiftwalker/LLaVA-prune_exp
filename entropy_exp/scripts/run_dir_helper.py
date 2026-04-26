#!/usr/bin/env python3
"""CLI helper for recursive experiment run discovery."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
LLAVA_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(LLAVA_ROOT))

from entropy_exp.src.run_layout import find_run_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List entropy_exp run directories")
    parser.add_argument(
        "command",
        choices=["list"],
        help="Helper action to perform",
    )
    parser.add_argument(
        "--runs-dir",
        default="entropy_exp/outputs/runs",
        help="Root directory containing run directories",
    )
    parser.add_argument(
        "--require-file",
        action="append",
        default=[],
        help="Only include run dirs containing this file; may be repeated",
    )
    parser.add_argument(
        "--prefix",
        help="Match run dirs whose leaf basename starts with this prefix",
    )
    parser.add_argument(
        "--exact-name",
        help="Match one run dir by exact leaf basename",
    )
    parser.add_argument(
        "--dataset",
        choices=["gqa", "mme", "pope", "textvqa", "scienceqa", "mmbench"],
        help="Filter runs by dataset metadata",
    )
    parser.add_argument(
        "--strategy",
        choices=[
            "baseline",
            "attn_score",
            "pre_attn_score",
            "masking_attn_score",
            "tail_masking_attn_score",
            "entropy",
            "random",
            "sparsevlm",
            "sparsevlm_adaptive_stratified",
            "sparsevlm_entropy_alpha",
        ],
        help="Filter runs by strategy branch",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command != "list":
        raise ValueError(f"Unsupported command: {args.command}")

    run_dirs = find_run_dirs(
        runs_dir=args.runs_dir,
        required_files=args.require_file,
        name_prefix=args.prefix,
        exact_name=args.exact_name,
        dataset=args.dataset,
        strategy=args.strategy,
    )
    for run_dir in run_dirs:
        print(os.fspath(run_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
