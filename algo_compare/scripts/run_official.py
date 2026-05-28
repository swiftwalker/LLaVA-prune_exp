#!/usr/bin/env python3
"""Run an official algorithm implementation against local models and datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from algo_compare.official_runner import (  # noqa: E402
    build_official_run,
    collect_command,
    execute_command,
    manifest_entry,
    print_dry_run,
    read_json,
    require_runnable,
    shell_join,
    write_run_files,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", default="sparsevlm", help="Method directory under algo_compare")
    parser.add_argument("--dataset", required=True, choices=["gqa", "mme", "pope", "textvqa", "scienceqa"])
    parser.add_argument("--variant", default=None, help="Official variant name from method.yaml")
    parser.add_argument("--retain-token", type=int, default=None, help="Official RETAIN_TOKN value")
    parser.add_argument("--use-version", default=None, help="Override official USE_VERSION")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--question-file", default=None)
    parser.add_argument("--image-folder", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--official-repo", default=None)
    parser.add_argument("--python-bin", default=None)
    parser.add_argument("--prune-config", default=None)
    parser.add_argument("--conv-mode", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--num-beams", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--num-chunks", type=int, default=None)
    parser.add_argument("--chunk-idx", type=int, default=None)
    parser.add_argument("--fastv-k", type=int, default=None, help="FastV pruning layer K")
    parser.add_argument("--fastv-r", type=float, default=None, help="FastV pruning ratio R")
    parser.add_argument("--fastv-attention-rank", type=int, default=None, help="FastV kept visual token rank")
    parser.add_argument("--fastv-image-token-length", type=int, default=None, help="FastV original image token count")
    parser.add_argument("--fastv-sys-length", type=int, default=None, help="FastV system/prompt token offset")
    parser.add_argument("--fastv-mode", default=None, help="FastV implementation mode")
    parser.add_argument(
        "--fastv-max-expanded-tokens",
        type=int,
        default=None,
        help="Disable FastV per sample when estimated expanded prompt length exceeds this value; <=0 disables fallback",
    )
    parser.add_argument("--pdrop-layer-list", default=None, help="PDROP/PyramidDrop pruning layer list, e.g. '[8,16,24]'")
    parser.add_argument(
        "--pdrop-image-token-ratio-list",
        default=None,
        help="PDROP/PyramidDrop image token ratio list, e.g. '[0.5,0.25,0.125]'",
    )
    parser.add_argument("--visionzip-dominant", type=int, default=None, help="VisionZip dominant visual token count")
    parser.add_argument("--visionzip-contextual", type=int, default=None, help="VisionZip contextual visual token count")
    parser.add_argument("--divprune-baseline", default=None, help="DivPrune BASELINE env value, usually OURS")
    parser.add_argument("--divprune-layer-index", type=int, default=None, help="DivPrune LAYER_INDEX env value")
    parser.add_argument("--divprune-subset-ratio", type=float, default=None, help="DivPrune SUBSET_RATIO env value")
    parser.add_argument(
        "--divprune-visual-token-count",
        type=int,
        default=None,
        help="Reference visual token count used for metadata retained-token calculation",
    )
    parser.add_argument("--eval", action="store_true", help="Run local entropy_exp evaluator after inference")
    parser.add_argument("--no-eval", action="store_false", dest="eval", help="Disable local eval")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved paths and command without running")
    parser.add_argument(
        "--emit-manifest-entry",
        action="store_true",
        help="Print a YAML snippet suitable for the method's manifests/official_runs.yaml",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run = build_official_run(args)

        if args.dry_run:
            print_dry_run(run)
            if args.emit_manifest_entry:
                print("\n# manifest entry")
                print(yaml.safe_dump({"runs": [manifest_entry(run)]}, sort_keys=False, allow_unicode=True).rstrip())
            return 0

        require_runnable(run)
        write_run_files(run)

        print(f"[official] cwd={run.cwd}")
        print(f"[official] env={run.env}")
        print(f"[official] command={shell_join(run.command)}")
        execute_command(run.command, cwd=run.cwd, env=run.env)

        if run.eval_command:
            print(f"[eval] command={shell_join(run.eval_command)}")
            execute_command(run.eval_command)

        command = collect_command(run)
        print(f"[collect] command={shell_join(command)}")
        execute_command(command)

        if args.emit_manifest_entry:
            summary = read_json(run.output_dir / "official_summary.json")
            print("\n# manifest entry")
            print(yaml.safe_dump({"runs": [manifest_entry(run, summary)]}, sort_keys=False, allow_unicode=True).rstrip())

        return 0
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
