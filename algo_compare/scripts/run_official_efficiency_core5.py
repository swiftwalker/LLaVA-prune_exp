#!/usr/bin/env python3
"""Launch official-method efficiency runs for the core five datasets."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUN_OFFICIAL = ROOT / "algo_compare" / "scripts" / "run_official.py"
DEFAULT_OUTPUT_ROOT = ROOT / "algo_compare" / "analysis" / "official_efficiency_core5_sv1budget"
LLAVA_PYTHON = Path("/home/liuyu/.conda/envs/llava/bin/python")
DEFAULT_PYTHON = LLAVA_PYTHON if LLAVA_PYTHON.exists() else Path(sys.executable)
DATASETS = ("gqa", "textvqa", "pope", "mme", "scienceqa")
RETAINS = ("retain192", "retain128", "retain64")
METHODS = ("fastv", "pdrop", "sparsevlm", "divprune", "visionzip")
DATASET_WEIGHTS = {
    "gqa": 12578,
    "pope": 8910,
    "textvqa": 5000,
    "scienceqa": 4241,
    "mme": 2374,
}


@dataclass(frozen=True)
class Case:
    method: str
    retain: str
    dataset: str
    args: tuple[str, ...]


METHOD_RETAIN_ARGS: dict[str, dict[str, tuple[str, ...]]] = {
    "sparsevlm": {
        "retain192": ("--variant", "sparsevlm_v1", "--retain-token", "192", "--use-version", "1_0"),
        "retain128": ("--variant", "sparsevlm_v1", "--retain-token", "128", "--use-version", "1_0"),
        "retain64": ("--variant", "sparsevlm_v1", "--retain-token", "64", "--use-version", "1_0"),
    },
    "fastv": {
        "retain192": (
            "--fastv-k",
            "2",
            "--fastv-r",
            "0.713542",
            "--fastv-attention-rank",
            "165",
        ),
        "retain128": (
            "--fastv-k",
            "2",
            "--fastv-r",
            "0.833333",
            "--fastv-attention-rank",
            "96",
        ),
        "retain64": (
            "--fastv-k",
            "2",
            "--fastv-r",
            "0.951389",
            "--fastv-attention-rank",
            "28",
        ),
    },
    "pdrop": {
        "retain192": (
            "--pdrop-layer-list",
            "[8,16,24]",
            "--pdrop-image-token-ratio-list",
            "[0.248264,0.062500,0.015625]",
        ),
        "retain128": (
            "--pdrop-layer-list",
            "[2,6,16]",
            "--pdrop-image-token-ratio-list",
            "[0.526043,0.190974,0.062502]",
        ),
        "retain64": (
            "--pdrop-layer-list",
            "[2,6,16]",
            "--pdrop-image-token-ratio-list",
            "[0.114585,0.052085,0.029516]",
        ),
    },
    "visionzip": {
        "retain192": ("--visionzip-dominant", "161", "--visionzip-contextual", "30"),
        "retain128": ("--visionzip-dominant", "106", "--visionzip-contextual", "20"),
        "retain64": ("--visionzip-dominant", "52", "--visionzip-contextual", "10"),
    },
    "divprune": {
        "retain192": ("--divprune-subset-ratio", "0.331598"),
        "retain128": ("--divprune-subset-ratio", "0.218750"),
        "retain64": ("--divprune-subset-ratio", "0.107639"),
    },
}


def build_cases(args: argparse.Namespace) -> list[Case]:
    methods = [args.method] if args.method else list(METHODS)
    datasets = [args.dataset] if args.dataset else list(DATASETS)
    retains = [args.retain] if args.retain else list(RETAINS)
    cases: list[Case] = []
    for dataset in datasets:
        for method in methods:
            for retain in retains:
                cases.append(Case(method=method, retain=retain, dataset=dataset, args=METHOD_RETAIN_ARGS[method][retain]))
    return cases


def run_dir_for_case(case: Case, args: argparse.Namespace) -> Path:
    output_root = Path(args.output_root).resolve()
    return output_root / "runs" / case.dataset / case.method / case.retain


def case_completed(case: Case, args: argparse.Namespace) -> bool:
    run_dir = run_dir_for_case(case, args)
    return all(
        path.is_file()
        for path in (
            run_dir / "answers.jsonl",
            run_dir / "benchmark_stats.jsonl",
            run_dir / "eval" / "summary.json",
            run_dir / "official_summary.json",
        )
    )


def select_shard(cases: list[Case], args: argparse.Namespace) -> tuple[list[Case], list[int]]:
    if args.num_shards is None:
        return cases, [sum(DATASET_WEIGHTS[case.dataset] for case in cases)]
    if args.shard_index is None:
        raise ValueError("--shard-index is required with --num-shards")
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")

    loads = [0 for _ in range(args.num_shards)]
    shards: list[list[Case]] = [[] for _ in range(args.num_shards)]
    weighted_cases = sorted(cases, key=lambda case: (-DATASET_WEIGHTS[case.dataset], case.dataset, case.method, case.retain))
    for case in weighted_cases:
        shard_idx = min(range(args.num_shards), key=lambda idx: (loads[idx], idx))
        shards[shard_idx].append(case)
        loads[shard_idx] += DATASET_WEIGHTS[case.dataset]
    return shards[args.shard_index], loads


def command_for_case(case: Case, args: argparse.Namespace) -> list[str]:
    run_dir = run_dir_for_case(case, args)
    command = [
        str(Path(args.launcher_python_bin or DEFAULT_PYTHON).resolve()),
        str(RUN_OFFICIAL),
        "--method",
        case.method,
        "--dataset",
        case.dataset,
        "--output-dir",
        str(run_dir),
        "--eval",
        "--benchmark",
        "--benchmark-warmup-samples",
        str(args.warmup_samples),
        "--benchmark-run-label",
        f"{case.method}_{case.retain}_{case.dataset}",
    ]
    command.extend(case.args)
    if args.max_samples is not None:
        command.extend(["--max-samples", str(args.max_samples)])
    command.extend(["--python-bin", str(Path(args.python_bin or DEFAULT_PYTHON).resolve())])
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--dataset", choices=DATASETS, default=None)
    parser.add_argument("--method", choices=METHODS, default=None)
    parser.add_argument("--retain", choices=RETAINS, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--warmup-samples", type=int, default=2)
    parser.add_argument("--launcher-python-bin", default=None, help="Python used to invoke run_official.py")
    parser.add_argument("--python-bin", default=None, help="Python used inside run_official.py for inference/eval")
    parser.add_argument("--gpu", default=None, help="Set CUDA_VISIBLE_DEVICES for executed commands")
    parser.add_argument("--skip-completed", action="store_true", help="Skip cases with answers/stats/eval/summary files")
    parser.add_argument("--num-shards", type=int, default=None, help="Split selected cases into LPT-balanced shards")
    parser.add_argument("--shard-index", type=int, default=None, help="Run only this shard index")
    parser.add_argument("--execute", action="store_true", help="Run commands sequentially instead of printing only")
    args = parser.parse_args()

    cases = build_cases(args)
    total_before_skip = len(cases)
    if args.skip_completed:
        cases = [case for case in cases if not case_completed(case, args)]
    cases, loads = select_shard(cases, args)
    commands = [command_for_case(case, args) for case in cases]
    shard_text = ""
    if args.num_shards is not None:
        shard_text = f" shard={args.shard_index}/{args.num_shards} shard_load={loads[args.shard_index]} all_loads={loads}"
    print(
        f"[official-efficiency] cases={len(commands)} selected_from={total_before_skip} "
        f"output_root={Path(args.output_root).resolve()}{shard_text}"
    )
    for idx, command in enumerate(commands, 1):
        print(f"[{idx:03d}/{len(commands):03d}] {shlex.join(command)}")

    if not args.execute:
        return 0

    env = os.environ.copy()
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    for idx, command in enumerate(commands, 1):
        print(f"\n[official-efficiency] running {idx}/{len(commands)}: {shlex.join(command)}", flush=True)
        subprocess.run(command, cwd=str(ROOT), env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
