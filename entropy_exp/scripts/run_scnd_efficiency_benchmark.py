#!/usr/bin/env python3
"""Run isolated SCND efficiency benchmark jobs.

This helper intentionally runs jobs sequentially on one visible GPU. Timing
benchmarks should not share a GPU with the high-concurrency experiment
scheduler.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


LLAVA_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = "entropy_exp/outputs/scnd_efficiency_benchmark_l2_6_16"
DEFAULT_DATASET = "pope"
PRUNE_LAYERS = "[2,6,16]"
RETAIN_RATIOS = {
    "retain192": "[0.4791666666666667,0.3333333333333333,0.45]",
    "retain128": "[0.4739583333333333,0.636963696369637,0.6727272727272727]",
    "retain64": "[0.8854166666666666,0.5454545454545454,0.43333333333333335]",
}


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    strategy: str
    retain: str
    suffix: str
    extra_sets: tuple[str, ...]


def _common_sets(output_root: str, warmup_samples: int) -> tuple[str, ...]:
    return (
        f"output.base_dir={output_root}",
        "benchmark.enabled=true",
        "benchmark.profile_timing=true",
        f"benchmark.warmup_samples={int(warmup_samples)}",
        "inference.seed=42",
        "capture.save_attention=false",
        "capture.save_importance_scores=false",
        "capture.save_keep_indices=false",
        f"pruning.prune_layers={PRUNE_LAYERS}",
    )


def _scnd_default_sets() -> tuple[str, ...]:
    return (
        'pruning.sparsevlm_scnd.layer_modes=["C","B","S"]',
        "pruning.sparsevlm_scnd.seed_ratio_min=0.15",
        "pruning.sparsevlm_scnd.seed_ratio_max=0.55",
        "pruning.sparsevlm_scnd.seed_pool_multiplier=2.0",
        "pruning.sparsevlm_scnd.saliency_floor_min=0.65",
        "pruning.sparsevlm_scnd.saliency_floor_max=0.90",
        "pruning.sparsevlm_scnd.boundary_ratio=0.25",
        "pruning.sparsevlm_scnd.saliency_repair=true",
        "pruning.sparsevlm_scnd.distance_metric=cosine",
        "pruning.sparsevlm_scnd.c_selection_rule=native",
        "pruning.sparsevlm_scnd.seed=42",
        "pruning.sparsevlm_scnd.use_score_memory=false",
        "pruning.sparsevlm_scnd.fallback_topk=4",
        "pruning.sparsevlm_scnd.exclude_special_tokens=true",
        "pruning.sparsevlm_scnd.min_visual_tokens_after_prune=16",
    )


def _fast_scnd_sets() -> tuple[str, ...]:
    return (
        'pruning.sparsevlm_fast_scnd.layer_modes=["F","T","S"]',
        "pruning.sparsevlm_fast_scnd.core_ratio_min=0.10",
        "pruning.sparsevlm_fast_scnd.core_ratio_max=0.35",
        "pruning.sparsevlm_fast_scnd.saliency_pool_multiplier=1.5",
        "pruning.sparsevlm_fast_scnd.reservoir_multiplier=0.5",
        "pruning.sparsevlm_fast_scnd.reservoir_rank_multiplier=4.0",
        "pruning.sparsevlm_fast_scnd.candidate_cap_multiplier=2.0",
        "pruning.sparsevlm_fast_scnd.projection_dim=64",
        "pruning.sparsevlm_fast_scnd.greedy_steps=16",
        "pruning.sparsevlm_fast_scnd.tie_break_band_ratio=0.10",
        "pruning.sparsevlm_fast_scnd.tie_break_band_max=32",
        "pruning.sparsevlm_fast_scnd.cached_diversity_weight=0.05",
        "pruning.sparsevlm_fast_scnd.distance_metric=cosine",
        "pruning.sparsevlm_fast_scnd.use_score_memory=false",
        "pruning.sparsevlm_fast_scnd.fallback_topk=4",
        "pruning.sparsevlm_fast_scnd.exclude_special_tokens=true",
        "pruning.sparsevlm_fast_scnd.min_visual_tokens_after_prune=16",
    )


def build_cases(group: str, output_root: str, warmup_samples: int) -> list[BenchmarkCase]:
    common = _common_sets(output_root, warmup_samples)
    cases: list[BenchmarkCase] = []
    include_e2e = group in {"all", "e2e"}
    include_overhead = group in {"all", "overhead"}

    if include_e2e:
        cases.append(
            BenchmarkCase(
                name="e2e_full_576",
                strategy="baseline",
                retain="576",
                suffix="eff_full_576",
                extra_sets=common
                + (
                    "benchmark.full_no_prune=true",
                    "pruning.prune_ratio=[0.0,0.0,0.0]",
                    "output.run_tag_suffix=eff_full_576",
                ),
            )
        )
        for retain in ("retain192", "retain128", "retain64"):
            cases.append(
                BenchmarkCase(
                    name=f"e2e_scnd_{retain}",
                    strategy="sparsevlm_scnd",
                    retain=retain,
                    suffix=f"eff_scnd_{retain}",
                    extra_sets=common
                    + _scnd_default_sets()
                    + (
                        "pruning.sparsevlm_scnd.selection_backend=gpu",
                        f"pruning.prune_ratio={RETAIN_RATIOS[retain]}",
                        f"output.run_tag_suffix=eff_scnd_{retain}",
                    ),
                )
            )

    if include_overhead:
        if not include_e2e:
            cases.append(
                BenchmarkCase(
                    name="overhead_scnd_gpu_retain128",
                    strategy="sparsevlm_scnd",
                    retain="retain128",
                    suffix="eff_overhead_scnd_gpu_retain128",
                    extra_sets=common
                    + _scnd_default_sets()
                    + (
                        "pruning.sparsevlm_scnd.selection_backend=gpu",
                        f"pruning.prune_ratio={RETAIN_RATIOS['retain128']}",
                        "output.run_tag_suffix=eff_overhead_scnd_gpu_retain128",
                    ),
                )
            )
        cases.extend(
            [
                BenchmarkCase(
                    name="overhead_scnd_python_retain128",
                    strategy="sparsevlm_scnd",
                    retain="retain128",
                    suffix="eff_overhead_scnd_python_retain128",
                    extra_sets=common
                    + _scnd_default_sets()
                    + (
                        "pruning.sparsevlm_scnd.selection_backend=python",
                        f"pruning.prune_ratio={RETAIN_RATIOS['retain128']}",
                        "output.run_tag_suffix=eff_overhead_scnd_python_retain128",
                    ),
                ),
                BenchmarkCase(
                    name="overhead_fast_scnd_retain128",
                    strategy="sparsevlm_fast_scnd",
                    retain="retain128",
                    suffix="eff_overhead_fast_scnd_retain128",
                    extra_sets=common
                    + _fast_scnd_sets()
                    + (
                        f"pruning.prune_ratio={RETAIN_RATIOS['retain128']}",
                        "output.run_tag_suffix=eff_overhead_fast_scnd_retain128",
                    ),
                ),
                BenchmarkCase(
                    name="overhead_sparsevlm_retain128",
                    strategy="sparsevlm",
                    retain="retain128",
                    suffix="eff_overhead_sparsevlm_retain128",
                    extra_sets=common
                    + (
                        "pruning.sparsevlm.fallback_topk=4",
                        "pruning.sparsevlm.exclude_special_tokens=true",
                        "pruning.sparsevlm.min_visual_tokens_after_prune=16",
                        f"pruning.prune_ratio={RETAIN_RATIOS['retain128']}",
                        "output.run_tag_suffix=eff_overhead_sparsevlm_retain128",
                    ),
                ),
            ]
        )

    return cases


def build_command(case: BenchmarkCase, dataset: str, total_samples: int) -> list[str]:
    command = [
        "bash",
        "entropy_exp/scripts/run_prune.sh",
        case.strategy,
        dataset,
        str(int(total_samples)),
        "--no-auto-gpu",
    ]
    for extra_set in case.extra_sets:
        command.extend(["--set", extra_set])
    return command


def command_text(command: Iterable[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run isolated SCND efficiency benchmarks")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--group", choices=["all", "e2e", "overhead"], default="all")
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Run only the named case; may be repeated. Names are printed by --dry-run.",
    )
    parser.add_argument("--gpu", default=None, help="Physical GPU id to expose through CUDA_VISIBLE_DEVICES")
    parser.add_argument("--timed-samples", type=int, default=10)
    parser.add_argument("--warmup-samples", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.timed_samples <= 0:
        raise SystemExit("--timed-samples must be positive")
    if args.warmup_samples < 0:
        raise SystemExit("--warmup-samples must be non-negative")

    total_samples = int(args.timed_samples) + int(args.warmup_samples)
    cases = build_cases(args.group, args.output_root, args.warmup_samples)
    if args.case:
        wanted = set(args.case)
        known = {case.name for case in cases}
        unknown = sorted(wanted - known)
        if unknown:
            raise SystemExit(f"Unknown --case value(s): {unknown}; known cases: {sorted(known)}")
        cases = [case for case in cases if case.name in wanted]
    env = os.environ.copy()
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("HF_DATASETS_OFFLINE", "1")
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    manifest = {
        "dataset": args.dataset,
        "output_root": args.output_root,
        "group": args.group,
        "timed_samples": args.timed_samples,
        "warmup_samples": args.warmup_samples,
        "total_samples": total_samples,
        "gpu": args.gpu,
        "cases": [],
    }

    for case in cases:
        command = build_command(case, args.dataset, total_samples)
        manifest["cases"].append(
            {
                "name": case.name,
                "strategy": case.strategy,
                "retain": case.retain,
                "suffix": case.suffix,
                "command": command,
            }
        )
        print(f"[efficiency] {case.name}: {command_text(command)}", flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=LLAVA_ROOT, env=env, check=True)

    output_dir = LLAVA_ROOT / args.output_root
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "efficiency_benchmark_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[efficiency] manifest: {manifest_path}")


if __name__ == "__main__":
    main()
