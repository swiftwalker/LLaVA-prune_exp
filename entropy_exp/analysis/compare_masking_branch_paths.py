#!/usr/bin/env python3
"""Compare masking_attn_score execution paths across keep/non-position-id branches.

This script has two modes:

1. Main mode:
   - prepares a shared 3-sample GQA question file
   - runs the same tiny masking_attn_score jobs on two repo roots
   - compares traces, run artifacts, and optional formal POPE summaries
   - writes compare/artifact_diff_summary.json and compare/conclusion.md

2. Worker mode:
   - prepends the target repo root to ``sys.path``
   - imports the target repo's pruning modules
   - installs runtime monkeypatch tracing
   - runs one tiny masking_attn_score configuration
   - evaluates the resulting run and writes one trace JSONL

The top-level module intentionally imports only stdlib modules so we do not
accidentally import pruning code from the current worktree before the worker
re-points ``sys.path``.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_KEEP_ROOT = Path("/storage/liuyu/LLaVA-prune_exp")
DEFAULT_NONPOS_ROOT = Path("/storage/liuyu/LLaVA-prune_exp__nonpos_compare")
DEFAULT_OUTPUT_DIR = DEFAULT_KEEP_ROOT / "entropy_exp" / "outputs" / "analysis" / "masking_branch_compare"
PREFERRED_PYTHON_CANDIDATES = (
    Path("/data_ssd/liuyu/miniconda3/envs/llava/bin/python"),
    Path("/home/liuyu/miniconda3/envs/llava/bin/python"),
    Path("/data/liuyu/anaconda3/envs/llava/bin/python"),
)
DEFAULT_DATASET = "gqa"
DEFAULT_GPU = "0"
DEFAULT_SEED = 42
DEFAULT_TINY_SAMPLE_COUNT = 3
TARGET_FILES = (
    "entropy_exp/src/prune_inference.py",
    "entropy_exp/src/pruner.py",
)
CONFIG_SPECS = {
    "l1_r0p2": {
        "label": "l1_r0p2",
        "prune_layers": [1],
        "prune_ratio": [0.2],
        "target_layer": 1,
    },
    "l3_r0p7": {
        "label": "l3_r0p7",
        "prune_layers": [3],
        "prune_ratio": [0.7],
        "target_layer": 3,
    },
}
BRANCH_SPECS = {
    "keep_position_ids": {
        "repo_arg": "keep_root",
        "display_name": "keep-position-ids",
    },
    "non_position_id": {
        "repo_arg": "nonpos_root",
        "display_name": "prune-exp-sync-non-position-id",
    },
}
FIRST_DIVERGENCE_ORDER = (
    "prepare/effective initial position_ids",
    "target-layer position_ids",
    "rotary effective span",
    "keep_indices",
    "target-layer post-mask attention",
    "answer",
    "eval summary",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare masking_attn_score paths across branches")
    parser.add_argument("--keep-root", type=Path, default=DEFAULT_KEEP_ROOT)
    parser.add_argument("--nonpos-root", type=Path, default=DEFAULT_NONPOS_ROOT)
    parser.add_argument("--dataset", choices=[DEFAULT_DATASET], default=DEFAULT_DATASET)
    parser.add_argument("--question-file", type=Path, default=None)
    parser.add_argument("--configs", type=str, default="l1_r0p2,l3_r0p7")
    parser.add_argument("--gpu", type=str, default=DEFAULT_GPU)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--repo-root", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--branch-label", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--config-label", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--tiny-question-file", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--trace-file", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--native-output-root", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--result-json", type=Path, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def safe_relpath(path: Path, start: Path) -> str:
    try:
        return str(path.resolve().relative_to(start.resolve()))
    except ValueError:
        return str(path.resolve())


def parse_config_labels(raw: str) -> list[str]:
    labels = [item.strip() for item in raw.split(",") if item.strip()]
    if not labels:
        raise ValueError("At least one config label is required.")
    unknown = [label for label in labels if label not in CONFIG_SPECS]
    if unknown:
        raise ValueError(f"Unknown config label(s): {', '.join(sorted(unknown))}")
    return labels


def read_first_jsonl_lines(src_path: Path, count: int) -> list[str]:
    lines: list[str] = []
    with src_path.open("r", encoding="utf-8") as handle:
        for _ in range(count):
            line = handle.readline()
            if not line:
                break
            lines.append(line.rstrip("\n"))
    return lines


def prepare_tiny_question_file(source_path: Path, output_dir: Path, dataset: str) -> Path:
    tiny_dir = ensure_dir(output_dir / "shared")
    tiny_path = tiny_dir / f"{dataset}_tiny_first{DEFAULT_TINY_SAMPLE_COUNT}.jsonl"
    lines = read_first_jsonl_lines(source_path, DEFAULT_TINY_SAMPLE_COUNT)
    if len(lines) != DEFAULT_TINY_SAMPLE_COUNT:
        raise RuntimeError(
            f"Expected {DEFAULT_TINY_SAMPLE_COUNT} samples in {source_path}, found {len(lines)}."
        )
    tiny_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tiny_path


def run_command(
    args: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        check=True,
        text=True,
        capture_output=capture_output,
    )


def resolve_existing_repo_relative_path(repo_root: Path, relative_path: str) -> Path:
    candidates = [
        repo_root / relative_path,
        DEFAULT_KEEP_ROOT / relative_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (repo_root / relative_path).resolve()


def interpreter_has_torch(python_bin: Path) -> bool:
    if not python_bin.is_file():
        return False
    result = subprocess.run(
        [str(python_bin), "-c", "import torch"],
        text=True,
        capture_output=True,
    )
    return result.returncode == 0


def resolve_runtime_python() -> Path:
    candidates = list(PREFERRED_PYTHON_CANDIDATES) + [Path(sys.executable)]
    for candidate in candidates:
        if interpreter_has_torch(candidate):
            return candidate
    return Path(sys.executable)


def build_diff_text(keep_root: Path) -> str:
    cmd = [
        "git",
        "-C",
        str(keep_root),
        "diff",
        "--unified=20",
        "origin/prune-exp-sync-non-position-id..keep-position-ids",
        "--",
        *TARGET_FILES,
    ]
    return run_command(cmd).stdout


def build_theory_diff_summary(keep_root: Path) -> dict[str, Any]:
    diff_text = build_diff_text(keep_root)
    stat_text = run_command(
        [
            "git",
            "-C",
            str(keep_root),
            "diff",
            "--stat",
            "origin/prune-exp-sync-non-position-id..keep-position-ids",
            "--",
            *TARGET_FILES,
        ]
    ).stdout

    return {
        "target_files": list(TARGET_FILES),
        "diff_stat": stat_text.strip(),
        "diff_text": diff_text,
        "theoretical_difference_points": [
            {
                "point": "keep-position-ids passes upstream position_ids into pruned_generate(), while prune-exp-sync-non-position-id does not.",
                "evidence_file": "entropy_exp/src/prune_inference.py",
            },
            {
                "point": "keep-position-ids calls enable_sparse_position_ids_compat(model) after model load; non-position-id branch does not.",
                "evidence_file": "entropy_exp/src/prune_inference.py",
            },
            {
                "point": "keep-position-ids adds _required_rotary_seq_len(position_ids, seq_len) and uses it in both pre-mask scoring and _forward_masked_layer rotary span selection.",
                "evidence_file": "entropy_exp/src/pruner.py",
            },
            {
                "point": "keep-position-ids changes _refresh_sequence_state() to preserve provided position_ids instead of rebuilding 0..L-1 after physical pruning.",
                "evidence_file": "entropy_exp/src/pruner.py",
            },
        ],
        "theoretical_no_effect_points": [
            {
                "point": "masking_attn_score does not physically prune the sequence, so _refresh_sequence_state() is not expected to run on the target masking path.",
                "reason": "Target layer stays full-length and final_seq_len should remain unchanged.",
            },
            {
                "point": "If effective position_ids are dense 0..L-1, _required_rotary_seq_len(position_ids, seq_len) collapses to seq_len.",
                "reason": "The keep-position-ids rotary compatibility branch becomes behaviorally equivalent to the non-position-id branch.",
            },
            {
                "point": "Sparse-position-id compatibility only matters when model attention receives sparse, non-reindexed position_ids.",
                "reason": "This analysis records whether the upstream multimodal path and masking target layer ever produce that state.",
            },
        ],
    }


def load_formal_summary(root: Path) -> tuple[Path, list[dict[str, str]]] | None:
    summary_path = root / "entropy_exp" / "outputs" / "summary" / "pope_masking_attn_score_eval" / "summary.csv"
    if not summary_path.is_file():
        return None
    with summary_path.open("r", encoding="utf-8") as handle:
        return summary_path, list(csv.DictReader(handle))


def compare_formal_summaries(keep_root: Path, nonpos_root: Path) -> dict[str, Any]:
    keep_payload = load_formal_summary(keep_root)
    nonpos_payload = load_formal_summary(nonpos_root)
    if keep_payload is None or nonpos_payload is None:
        return {
            "status": "unavailable",
            "missing": {
                "keep_position_ids": None if keep_payload is not None else str(
                    keep_root / "entropy_exp" / "outputs" / "summary" / "pope_masking_attn_score_eval" / "summary.csv"
                ),
                "non_position_id": None if nonpos_payload is not None else str(
                    nonpos_root / "entropy_exp" / "outputs" / "summary" / "pope_masking_attn_score_eval" / "summary.csv"
                ),
            },
        }

    keep_path, keep_rows = keep_payload
    nonpos_path, nonpos_rows = nonpos_payload
    keep_map = {(row["prune_layers"], row["prune_ratio"]): row for row in keep_rows}
    nonpos_map = {(row["prune_layers"], row["prune_ratio"]): row for row in nonpos_rows}
    all_keys = sorted(set(keep_map) | set(nonpos_map))
    comparisons: list[dict[str, Any]] = []
    all_equal = True
    for key in all_keys:
        keep_row = keep_map.get(key)
        nonpos_row = nonpos_map.get(key)
        keep_value = float(keep_row["primary_metric_value"]) if keep_row else None
        nonpos_value = float(nonpos_row["primary_metric_value"]) if nonpos_row else None
        same = keep_value == nonpos_value
        if not same:
            all_equal = False
        comparisons.append(
            {
                "prune_layers": key[0],
                "prune_ratio": key[1],
                "keep_position_ids": keep_value,
                "non_position_id": nonpos_value,
                "same": same,
                "abs_diff": None if keep_value is None or nonpos_value is None else abs(keep_value - nonpos_value),
            }
        )
    return {
        "status": "available",
        "keep_summary_csv": str(keep_path),
        "nonpos_summary_csv": str(nonpos_path),
        "comparison_count": len(comparisons),
        "all_equal": all_equal,
        "comparisons": comparisons,
    }


def replace_symlink(link_path: Path, target_path: Path) -> None:
    ensure_dir(link_path.parent)
    if link_path.is_symlink() or link_path.exists():
        if link_path.is_dir() and not link_path.is_symlink():
            shutil.rmtree(link_path)
        else:
            link_path.unlink()
    os.symlink(target_path, link_path)


def coerce_to_builtin(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): coerce_to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [coerce_to_builtin(item) for item in value]
    if hasattr(value, "tolist"):
        return coerce_to_builtin(value.tolist())
    if hasattr(value, "detach"):
        return coerce_to_builtin(value.detach().cpu().tolist())
    return value


def squeeze_position_ids(value: Any) -> list[int] | None:
    converted = coerce_to_builtin(value)
    if converted is None:
        return None
    if converted and isinstance(converted, list) and isinstance(converted[0], list):
        if len(converted) == 1:
            converted = converted[0]
    return [int(item) for item in converted]


def build_drop_indices(v_token_num: int, keep_indices: list[int]) -> list[int]:
    keep_set = set(int(item) for item in keep_indices)
    return [idx for idx in range(v_token_num) if idx not in keep_set]


def is_dense_contiguous(position_ids: list[int] | None) -> bool:
    if not position_ids:
        return False
    start = position_ids[0]
    return position_ids == list(range(start, start + len(position_ids)))


def position_ids_max_plus_one(position_ids: list[int] | None) -> int | None:
    if not position_ids:
        return None
    return int(max(position_ids) + 1)


def nested_shape(value: Any) -> list[int]:
    shape: list[int] = []
    current = coerce_to_builtin(value)
    while isinstance(current, list):
        shape.append(len(current))
        current = current[0] if current else []
    return shape


def flatten_nested(value: Any) -> list[float]:
    converted = coerce_to_builtin(value)
    if isinstance(converted, list):
        flattened: list[float] = []
        for item in converted:
            flattened.extend(flatten_nested(item))
        return flattened
    if converted is None:
        return []
    return [float(converted)]


def get_wrapper_arg(kwargs: dict[str, Any], args: tuple[Any, ...], index: int, name: str) -> Any:
    if name in kwargs:
        return kwargs[name]
    if index < len(args):
        return args[index]
    raise KeyError(f"Missing wrapper argument: {name}")


def bind_call_arguments(func: Any, self_obj: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
    signature = inspect.signature(func)
    bound = signature.bind_partial(self_obj, *args, **kwargs)
    return dict(bound.arguments)


def take_last_dim(value: Any, indices: list[int]) -> list[float]:
    converted = coerce_to_builtin(value)
    if not isinstance(converted, list):
        return [float(converted)]
    if not converted:
        return []
    if converted and isinstance(converted[0], list):
        flattened: list[float] = []
        for item in converted:
            flattened.extend(take_last_dim(item, indices))
        return flattened
    return [float(converted[index]) for index in indices]


def summarize_tv_attn(
    tv_attn: Any,
    keep_indices: list[int] | None,
    drop_indices: list[int] | None,
) -> dict[str, Any] | None:
    if tv_attn is None:
        return None
    flattened = flatten_nested(tv_attn)
    if not flattened:
        return None
    summary = {
        "shape": nested_shape(tv_attn),
        "mean": float(sum(flattened) / len(flattened)),
        "min": float(min(flattened)),
        "max": float(max(flattened)),
        "kept_mean": None,
        "dropped_mean": None,
        "dropped_max": None,
    }
    if keep_indices:
        kept_values = take_last_dim(tv_attn, keep_indices)
        summary["kept_mean"] = float(sum(kept_values) / len(kept_values))
    if drop_indices:
        dropped_values = take_last_dim(tv_attn, drop_indices)
        summary["dropped_mean"] = float(sum(dropped_values) / len(dropped_values))
        summary["dropped_max"] = float(max(dropped_values))
    return summary


def values_close(left: Any, right: Any, *, rtol: float = 1e-3, atol: float = 1e-4) -> bool:
    if left is None or right is None:
        return left is right
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return False
        return all(values_close(l_item, r_item, rtol=rtol, atol=atol) for l_item, r_item in zip(left, right))
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return False
        return all(values_close(left[key], right[key], rtol=rtol, atol=atol) for key in left)
    if isinstance(left, str) or isinstance(right, str):
        return left == right
    try:
        left_float = float(left)
        right_float = float(right)
    except (TypeError, ValueError):
        return left == right
    return abs(left_float - right_float) <= (atol + rtol * abs(right_float))


def tensors_allclose(left: Any, right: Any, *, rtol: float = 1e-3, atol: float = 1e-4) -> bool:
    if left is None or right is None:
        return left is right
    left_shape = nested_shape(left)
    right_shape = nested_shape(right)
    if left_shape != right_shape:
        return False
    left_values = flatten_nested(left)
    right_values = flatten_nested(right)
    if len(left_values) != len(right_values):
        return False
    return all(
        abs(left_value - right_value) <= (atol + rtol * abs(right_value))
        for left_value, right_value in zip(left_values, right_values)
    )


def normalize_eval_accuracy(summary: dict[str, Any] | None) -> float | None:
    if not summary:
        return None
    metrics = summary.get("metrics") or {}
    if "accuracy" not in metrics:
        return None
    return float(metrics["accuracy"])


def normalize_gqa_prediction(text: str) -> str:
    return text.strip().rstrip(".").lower()


def evaluate_gqa_tiny_run(repo_root: Path, run_dir: Path) -> dict[str, Any]:
    answers_path = run_dir / "answers.jsonl"
    config_path = run_dir / "config.yaml"
    eval_dir = ensure_dir(run_dir / "eval")
    questions_path = resolve_existing_repo_relative_path(
        repo_root,
        "entropy_exp/datasets/gqa/testdev_balanced_questions.json",
    )
    questions = load_json(questions_path)
    answers = read_jsonl(answers_path)

    comparisons: list[dict[str, Any]] = []
    correct = 0
    for row in answers:
        question_id = str(row["question_id"])
        prediction = normalize_gqa_prediction(str(row["text"]))
        gold = normalize_gqa_prediction(str(questions[question_id]["answer"]))
        is_correct = prediction == gold
        if is_correct:
            correct += 1
        comparisons.append(
            {
                "question_id": question_id,
                "prediction": prediction,
                "gold_answer": gold,
                "correct": is_correct,
            }
        )

    accuracy = float(correct / len(comparisons)) if comparisons else 0.0
    summary = {
        "dataset": "gqa",
        "answers_file": str(answers_path),
        "output_dir": str(eval_dir),
        "run_dir": str(run_dir),
        "config_file": str(config_path),
        "evaluation_mode": "analysis_tiny_gqa_exact_match",
        "metrics": {
            "accuracy": accuracy,
            "num_samples": len(comparisons),
            "num_correct": correct,
        },
        "samples": comparisons,
    }
    write_json(eval_dir / "summary.json", summary)
    (eval_dir / "stdout.txt").write_text(
        "analysis_tiny_gqa_exact_match\n"
        f"num_samples={len(comparisons)}\n"
        f"accuracy={accuracy:.6f}\n",
        encoding="utf-8",
    )
    return summary


def compare_worker_outputs(
    keep_result: dict[str, Any],
    nonpos_result: dict[str, Any],
) -> tuple[dict[str, Any], str | None]:
    keep_records = keep_result["records"]
    nonpos_records = nonpos_result["records"]
    keep_map = {str(record["question_id"]): record for record in keep_records}
    nonpos_map = {str(record["question_id"]): record for record in nonpos_records}
    all_question_ids = [str(record["question_id"]) for record in keep_records]

    sample_comparisons: list[dict[str, Any]] = []
    first_divergence: str | None = None

    for question_id in all_question_ids:
        keep_record = keep_map[question_id]
        nonpos_record = nonpos_map[question_id]
        prepare_match = keep_record.get("prepare_position_ids") == nonpos_record.get("prepare_position_ids")
        effective_match = keep_record.get("effective_initial_position_ids") == nonpos_record.get("effective_initial_position_ids")
        target_position_match = keep_record.get("target_layer_position_ids") == nonpos_record.get("target_layer_position_ids")
        dense_both_true = bool(keep_record.get("position_ids_dense_contiguous")) and bool(
            nonpos_record.get("position_ids_dense_contiguous")
        )
        maxeq_both_true = (
            keep_record.get("position_ids_max_plus_one") == keep_record.get("seq_len")
            and nonpos_record.get("position_ids_max_plus_one") == nonpos_record.get("seq_len")
        )
        rotary_match = keep_record.get("rotary_effective_span") == nonpos_record.get("rotary_effective_span")
        keep_indices_match = keep_record.get("target_layer_keep_indices") == nonpos_record.get("target_layer_keep_indices")
        pre_mask_allclose = tensors_allclose(
            keep_record.get("target_layer_pre_mask_tv_attn"),
            nonpos_record.get("target_layer_pre_mask_tv_attn"),
        )
        post_mask_allclose = tensors_allclose(
            keep_record.get("target_layer_post_mask_tv_attn"),
            nonpos_record.get("target_layer_post_mask_tv_attn"),
        )
        pre_mask_summary_match = values_close(
            keep_record.get("target_layer_pre_mask_tv_attn_summary"),
            nonpos_record.get("target_layer_pre_mask_tv_attn_summary"),
        )
        post_mask_summary_match = values_close(
            keep_record.get("target_layer_post_mask_tv_attn_summary"),
            nonpos_record.get("target_layer_post_mask_tv_attn_summary"),
        )
        answer_match = keep_record.get("answer") == nonpos_record.get("answer")

        sample_comparison = {
            "question_id": question_id,
            "keep_position_ids": keep_record,
            "non_position_id": nonpos_record,
            "checks": {
                "prepare_position_ids_match": prepare_match,
                "effective_initial_position_ids_match": effective_match,
                "target_layer_position_ids_match": target_position_match,
                "position_ids_dense_contiguous_both_true": dense_both_true,
                "position_ids_max_plus_one_eq_seq_len_both_true": maxeq_both_true,
                "rotary_effective_span_match": rotary_match,
                "keep_indices_match": keep_indices_match,
                "pre_mask_tv_attn_allclose": pre_mask_allclose,
                "post_mask_tv_attn_allclose": post_mask_allclose,
                "pre_mask_tv_attn_summary_match": pre_mask_summary_match,
                "post_mask_tv_attn_summary_match": post_mask_summary_match,
                "answer_match": answer_match,
                "final_answer_identical": answer_match,
            },
        }
        sample_comparisons.append(sample_comparison)

        if first_divergence is None:
            if not (prepare_match and effective_match):
                first_divergence = "prepare/effective initial position_ids"
            elif not (target_position_match and dense_both_true and maxeq_both_true):
                first_divergence = "target-layer position_ids"
            elif not rotary_match:
                first_divergence = "rotary effective span"
            elif not keep_indices_match:
                first_divergence = "keep_indices"
            elif not (post_mask_allclose and post_mask_summary_match):
                first_divergence = "target-layer post-mask attention"
            elif not answer_match:
                first_divergence = "answer"

    keep_accuracy = normalize_eval_accuracy(keep_result.get("eval_summary"))
    nonpos_accuracy = normalize_eval_accuracy(nonpos_result.get("eval_summary"))
    eval_match = keep_accuracy == nonpos_accuracy
    if first_divergence is None and not eval_match:
        first_divergence = "eval summary"

    config_summary = {
        "branch_runs": {
            "keep_position_ids": {
                "run_dir": keep_result["run_dir"],
                "trace_file": keep_result["trace_file"],
                "eval_summary": keep_result.get("eval_summary"),
                "compat_patch_enabled": keep_result.get("compat_patch_enabled"),
                "target_layer_compat_wrapped": keep_result.get("target_layer_compat_wrapped"),
            },
            "non_position_id": {
                "run_dir": nonpos_result["run_dir"],
                "trace_file": nonpos_result["trace_file"],
                "eval_summary": nonpos_result.get("eval_summary"),
                "compat_patch_enabled": nonpos_result.get("compat_patch_enabled"),
                "target_layer_compat_wrapped": nonpos_result.get("target_layer_compat_wrapped"),
            },
        },
        "sample_comparisons": sample_comparisons,
        "eval_summary_check": {
            "keep_position_ids_accuracy": keep_accuracy,
            "non_position_id_accuracy": nonpos_accuracy,
            "match": eval_match,
        },
    }
    return config_summary, first_divergence


def build_conclusion_markdown(summary: dict[str, Any]) -> str:
    theory = summary["theory_diff_summary"]
    formal = summary["formal_pope_compare"]
    lines = [
        "# Masking Branch Compare Conclusion",
        "",
        "## Theory Diff Summary",
    ]
    for item in theory["theoretical_difference_points"]:
        lines.append(f"- Difference: {item['point']}")
    for item in theory["theoretical_no_effect_points"]:
        lines.append(f"- No-effect hypothesis: {item['point']}")

    lines.extend(["", "## Formal Prior Evidence"])
    if formal["status"] == "available":
        lines.append(
            f"- Compared {formal['comparison_count']} POPE masking_attn_score tuples; all_equal={formal['all_equal']}."
        )
    else:
        lines.append("- Formal POPE comparison unavailable locally; tiny-sample compare proceeded without it.")

    lines.extend(["", "## Tiny-Sample Conclusion"])
    overall_no_runtime_divergence = True
    for config_label in CONFIG_SPECS:
        if config_label not in summary["tiny_sample_compare"]:
            continue
        first_divergence = summary["first_divergence"].get(config_label)
        config_result = summary["tiny_sample_compare"][config_label]
        eval_match = config_result["eval_summary_check"]["match"]
        sample_checks = [sample["checks"] for sample in config_result["sample_comparisons"]]
        keep_match = all(sample["keep_indices_match"] for sample in sample_checks)
        post_match = all(
            sample["post_mask_tv_attn_allclose"] and sample["post_mask_tv_attn_summary_match"]
            for sample in sample_checks
        )
        answers_match = all(sample["answer_match"] for sample in sample_checks)
        dense = all(
            sample["position_ids_dense_contiguous_both_true"]
            and sample["position_ids_max_plus_one_eq_seq_len_both_true"]
            for sample in sample_checks
        )

        if first_divergence is None and dense and keep_match and post_match and answers_match and eval_match:
            lines.append(
                f"- `{config_label}`: code differs, but masking path has no effective runtime divergence."
            )
        else:
            overall_no_runtime_divergence = False
            lines.append(
                f"- `{config_label}`: first divergence at `{first_divergence}`."
            )

    lines.extend(["", "## Runtime Flags"])
    for config_label in CONFIG_SPECS:
        if config_label not in summary["tiny_sample_compare"]:
            continue
        branch_runs = summary["tiny_sample_compare"][config_label]["branch_runs"]
        lines.append(
            f"- `{config_label}`: keep compat_enabled={branch_runs['keep_position_ids']['compat_patch_enabled']}, "
            f"keep target_layer_compat_wrapped={branch_runs['keep_position_ids']['target_layer_compat_wrapped']}, "
            f"nonpos compat_enabled={branch_runs['non_position_id']['compat_patch_enabled']}, "
            f"nonpos target_layer_compat_wrapped={branch_runs['non_position_id']['target_layer_compat_wrapped']}."
        )

    if overall_no_runtime_divergence:
        lines.extend(
            [
                "",
                "## Final Judgment",
                "- Across the configured tiny-sample runs, the masking path shows code-level differences without an observed effective runtime divergence.",
            ]
        )
    return "\n".join(lines) + "\n"


def launch_worker(
    *,
    python_bin: Path,
    script_path: Path,
    repo_root: Path,
    branch_label: str,
    config_label: str,
    tiny_question_file: Path,
    trace_file: Path,
    native_output_root: Path,
    result_json: Path,
    gpu: str,
) -> dict[str, Any]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    args = [
        str(python_bin),
        str(script_path),
        "--worker",
        "--repo-root",
        str(repo_root),
        "--branch-label",
        branch_label,
        "--config-label",
        config_label,
        "--tiny-question-file",
        str(tiny_question_file),
        "--trace-file",
        str(trace_file),
        "--native-output-root",
        str(native_output_root),
        "--result-json",
        str(result_json),
    ]
    run_command(args, cwd=repo_root, env=env)
    return load_json(result_json)


class WorkerTracer:
    def __init__(self, branch_label: str, config_label: str, target_layer: int) -> None:
        self.branch_label = branch_label
        self.config_label = config_label
        self.target_layer = int(target_layer)
        self.records: list[dict[str, Any]] = []
        self.current_sample_index: int | None = None
        self.current_masking_layer: int | None = None
        self.current_forward_context: dict[str, Any] | None = None
        self.model = None
        self.compat_patch_enabled = False
        self.target_layer_compat_wrapped = False

    def start_sample(self, prepare_position_ids: Any) -> dict[str, Any]:
        record = {
            "branch_label": self.branch_label,
            "config_label": self.config_label,
            "sample_idx": len(self.records),
            "prepare_position_ids": squeeze_position_ids(prepare_position_ids),
            "effective_initial_position_ids": None,
            "target_layer_position_ids": None,
            "seq_len": None,
            "position_ids_max_plus_one": None,
            "position_ids_dense_contiguous": None,
            "rotary_effective_span": None,
            "compat_patch_enabled": None,
            "target_layer_compat_wrapped": None,
            "target_layer_importance_scores": None,
            "target_layer_keep_indices": None,
            "target_layer_drop_indices": None,
            "target_layer_pre_mask_tv_attn": None,
            "target_layer_pre_mask_tv_attn_summary": None,
            "target_layer_post_mask_tv_attn": None,
            "target_layer_post_mask_tv_attn_summary": None,
            "answer": None,
            "question_id": None,
            "original_seq_len": None,
            "final_seq_len": None,
            "prefill_time": None,
            "decode_time": None,
        }
        self.records.append(record)
        self.current_sample_index = record["sample_idx"]
        return record

    def current_record(self) -> dict[str, Any]:
        if self.current_sample_index is None:
            raise RuntimeError("No active sample record.")
        return self.records[self.current_sample_index]

    def finalize_record(self) -> None:
        record = self.current_record()
        record["compat_patch_enabled"] = bool(self.compat_patch_enabled)
        record["target_layer_compat_wrapped"] = bool(self.target_layer_compat_wrapped)
        self.current_sample_index = None
        self.current_masking_layer = None
        self.current_forward_context = None


def worker_main(args: argparse.Namespace) -> None:
    if args.repo_root is None or args.branch_label is None or args.config_label is None:
        raise SystemExit("Worker mode requires --repo-root, --branch-label, and --config-label.")
    if args.tiny_question_file is None or args.trace_file is None or args.native_output_root is None or args.result_json is None:
        raise SystemExit("Worker mode requires --tiny-question-file, --trace-file, --native-output-root, and --result-json.")

    repo_root = args.repo_root.resolve()
    config_spec = CONFIG_SPECS[args.config_label]

    os.chdir(repo_root)
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "entropy_exp" / "src"))

    import strategies.base as strategies_base  # noqa: WPS433
    import prune_inference  # noqa: WPS433
    import pruner as pruner_module  # noqa: WPS433

    tracer = WorkerTracer(
        branch_label=args.branch_label,
        config_label=args.config_label,
        target_layer=config_spec["target_layer"],
    )

    original_load_pretrained_model = prune_inference.load_pretrained_model
    original_run_prune_inference = prune_inference.run_prune_inference
    original_build_run_rel_dir = prune_inference.build_run_rel_dir
    original_softmax = pruner_module.F.softmax
    original_pruned_generate = pruner_module.VisualTokenPruner.pruned_generate
    original_pruned_prefill = pruner_module.VisualTokenPruner._pruned_prefill
    original_run_masking_prune_layer = pruner_module.VisualTokenPruner._run_masking_prune_layer
    original_compute_pre_prune_scores = pruner_module.VisualTokenPruner._compute_pre_prune_scores
    original_forward_masked_layer = pruner_module.VisualTokenPruner._forward_masked_layer
    original_compute_keep_mask_from_importance = strategies_base.PruneStrategy.compute_keep_mask_from_importance
    original_enable_sparse_position_ids_compat = getattr(pruner_module, "enable_sparse_position_ids_compat", None)

    def safe_build_run_rel_dir(output_base_dir: str | os.PathLike[str] | Path, strategy: str, dataset: str, run_name: str) -> str:
        run_dir = Path(prune_inference.build_run_dir(output_base_dir, strategy, dataset, run_name))
        try:
            return str(run_dir.relative_to(repo_root))
        except ValueError:
            return str(run_dir)

    def load_pretrained_model_wrapper(*load_args: Any, **load_kwargs: Any) -> Any:
        tokenizer, model, image_processor, context_len = original_load_pretrained_model(*load_args, **load_kwargs)
        tracer.model = model

        original_prepare_inputs = model.prepare_inputs_labels_for_multimodal

        def prepare_inputs_wrapper(*prepare_args: Any, **prepare_kwargs: Any) -> Any:
            output = original_prepare_inputs(*prepare_args, **prepare_kwargs)
            tracer.start_sample(output[1])
            return output

        model.prepare_inputs_labels_for_multimodal = prepare_inputs_wrapper
        return tokenizer, model, image_processor, context_len

    def enable_sparse_position_ids_compat_wrapper(model: Any) -> bool:
        if original_enable_sparse_position_ids_compat is None:
            tracer.compat_patch_enabled = False
            return False
        result = bool(original_enable_sparse_position_ids_compat(model))
        tracer.compat_patch_enabled = result
        tracer.model = model
        return result

    def pruned_prefill_wrapper(self: Any, inputs_embeds: Any, initial_position_ids: Any, *prefill_args: Any, **prefill_kwargs: Any) -> Any:
        bound_args = bind_call_arguments(
            original_pruned_prefill,
            self,
            inputs_embeds,
            initial_position_ids,
            *prefill_args,
            **prefill_kwargs,
        )
        record = tracer.current_record()
        bound_inputs_embeds = bound_args["inputs_embeds"]
        bound_initial_position_ids = bound_args.get("initial_position_ids")
        if bound_initial_position_ids is None:
            import torch

            effective_initial_position_ids = torch.arange(
                bound_inputs_embeds.shape[1],
                device=bound_inputs_embeds.device,
                dtype=torch.long,
            ).unsqueeze(0)
        else:
            effective_initial_position_ids = bound_initial_position_ids.to(
                device=bound_inputs_embeds.device,
                dtype=bound_initial_position_ids.dtype,
            ).clone()
        record["effective_initial_position_ids"] = squeeze_position_ids(effective_initial_position_ids)
        return original_pruned_prefill(self, inputs_embeds, initial_position_ids, *prefill_args, **prefill_kwargs)

    def run_masking_prune_layer_wrapper(self: Any, *mask_args: Any, **mask_kwargs: Any) -> Any:
        bound_args = bind_call_arguments(original_run_masking_prune_layer, self, *mask_args, **mask_kwargs)
        tracer.current_masking_layer = int(bound_args["layer_idx"])
        try:
            return original_run_masking_prune_layer(self, *mask_args, **mask_kwargs)
        finally:
            tracer.current_masking_layer = None

    def compute_pre_prune_scores_wrapper(self: Any, *score_args: Any, **score_kwargs: Any) -> Any:
        tv_attn, importance_scores = original_compute_pre_prune_scores(self, *score_args, **score_kwargs)
        if tracer.current_masking_layer == tracer.target_layer:
            bound_args = bind_call_arguments(original_compute_pre_prune_scores, self, *score_args, **score_kwargs)
            record = tracer.current_record()
            position_ids = bound_args["position_ids"]
            hidden_states = bound_args["hidden_states"]
            seq_len = int(hidden_states.shape[1])
            position_id_list = squeeze_position_ids(position_ids)
            rotary_fn = getattr(pruner_module, "_required_rotary_seq_len", None)
            if rotary_fn is None:
                rotary_span = seq_len
            else:
                rotary_span = int(rotary_fn(position_ids, seq_len))
            record["target_layer_position_ids"] = position_id_list
            record["seq_len"] = seq_len
            record["position_ids_max_plus_one"] = position_ids_max_plus_one(position_id_list)
            record["position_ids_dense_contiguous"] = is_dense_contiguous(position_id_list)
            record["rotary_effective_span"] = rotary_span
            record["target_layer_pre_mask_tv_attn"] = coerce_to_builtin(tv_attn)
            record["target_layer_importance_scores"] = coerce_to_builtin(importance_scores)
        return tv_attn, importance_scores

    def compute_keep_mask_from_importance_wrapper(self: Any, importance_scores: Any, layer_idx: int, *keep_args: Any, **keep_kwargs: Any) -> Any:
        keep_indices, layer_info = original_compute_keep_mask_from_importance(
            self,
            importance_scores,
            layer_idx,
            *keep_args,
            **keep_kwargs,
        )
        if tracer.current_masking_layer == tracer.target_layer:
            record = tracer.current_record()
            keep_list = [int(item) for item in coerce_to_builtin(keep_indices)]
            drop_list = build_drop_indices(int(importance_scores.shape[0]), keep_list)
            record["target_layer_keep_indices"] = keep_list
            record["target_layer_drop_indices"] = drop_list
            record["target_layer_pre_mask_tv_attn_summary"] = summarize_tv_attn(
                record.get("target_layer_pre_mask_tv_attn"),
                keep_list,
                drop_list,
            )
        return keep_indices, layer_info

    def forward_masked_layer_wrapper(self: Any, *forward_args: Any, **forward_kwargs: Any) -> Any:
        bound_args = bind_call_arguments(original_forward_masked_layer, self, *forward_args, **forward_kwargs)
        keep_indices = coerce_to_builtin(bound_args["keep_indices"])
        keep_list = [int(item) for item in keep_indices]
        drop_list = build_drop_indices(int(bound_args["v_token_num"]), keep_list)
        layer = bound_args["layer"]
        tracer.current_forward_context = {
            "layer_idx": int(bound_args["layer_idx"]),
            "v_token_start": int(bound_args["v_token_start"]),
            "v_token_num": int(bound_args["v_token_num"]),
            "text_token_start": int(bound_args["text_token_start"]),
            "keep_indices": keep_list,
            "drop_indices": drop_list,
            "compat_wrapped": bool(getattr(getattr(layer, "self_attn", None), "_sparse_position_ids_compat_wrapped", False)),
        }
        tracer.target_layer_compat_wrapped = tracer.current_forward_context["compat_wrapped"]
        try:
            return original_forward_masked_layer(self, *forward_args, **forward_kwargs)
        finally:
            tracer.current_forward_context = None

    def softmax_wrapper(input_tensor: Any, *softmax_args: Any, **softmax_kwargs: Any) -> Any:
        output_tensor = original_softmax(input_tensor, *softmax_args, **softmax_kwargs)
        context = tracer.current_forward_context
        if context and context["layer_idx"] == tracer.target_layer and getattr(output_tensor, "dim", lambda: 0)() == 4:
            record = tracer.current_record()
            v_start = context["v_token_start"]
            v_end = v_start + context["v_token_num"]
            text_start = context["text_token_start"]
            post_mask_tv_attn = output_tensor[0, :, text_start:, v_start:v_end]
            record["target_layer_post_mask_tv_attn"] = coerce_to_builtin(post_mask_tv_attn)
            record["target_layer_post_mask_tv_attn_summary"] = summarize_tv_attn(
                post_mask_tv_attn,
                context["keep_indices"],
                context["drop_indices"],
            )
        return output_tensor

    def pruned_generate_wrapper(self: Any, *generate_args: Any, **generate_kwargs: Any) -> Any:
        generated_ids, prune_info = original_pruned_generate(self, *generate_args, **generate_kwargs)
        record = tracer.current_record()
        record["original_seq_len"] = int(prune_info["original_seq_len"])
        record["final_seq_len"] = int(prune_info["final_seq_len"])
        record["prefill_time"] = float(prune_info["prefill_time"])
        record["decode_time"] = float(prune_info["decode_time"])
        tracer.finalize_record()
        return generated_ids, prune_info

    prune_inference.build_run_rel_dir = safe_build_run_rel_dir
    prune_inference.load_pretrained_model = load_pretrained_model_wrapper
    pruner_module.F.softmax = softmax_wrapper
    pruner_module.VisualTokenPruner.pruned_generate = pruned_generate_wrapper
    pruner_module.VisualTokenPruner._pruned_prefill = pruned_prefill_wrapper
    pruner_module.VisualTokenPruner._run_masking_prune_layer = run_masking_prune_layer_wrapper
    pruner_module.VisualTokenPruner._compute_pre_prune_scores = compute_pre_prune_scores_wrapper
    pruner_module.VisualTokenPruner._forward_masked_layer = forward_masked_layer_wrapper
    strategies_base.PruneStrategy.compute_keep_mask_from_importance = compute_keep_mask_from_importance_wrapper
    if original_enable_sparse_position_ids_compat is not None:
        pruner_module.enable_sparse_position_ids_compat = enable_sparse_position_ids_compat_wrapper
        if hasattr(prune_inference, "enable_sparse_position_ids_compat"):
            prune_inference.enable_sparse_position_ids_compat = enable_sparse_position_ids_compat_wrapper

    try:
        config = prune_inference.load_config(str(repo_root / "entropy_exp" / "configs" / "prune.yaml"))
        overrides = [
            "pruning.strategy=masking_attn_score",
            f"pruning.prune_layers={json.dumps(config_spec['prune_layers'])}",
            f"pruning.prune_ratio={json.dumps(config_spec['prune_ratio'])}",
            "pruning.layer_selection=fixed",
            f"inference.seed={DEFAULT_SEED}",
            "capture.save_attention=true",
            "capture.capture_layers=all",
            "capture.save_importance_scores=true",
            "capture.save_keep_indices=true",
            f"model.path={resolve_existing_repo_relative_path(repo_root, 'entropy_exp/models/llava-v1.5-7b')}",
            f"datasets.gqa.question_file={args.tiny_question_file.resolve()}",
            f"datasets.gqa.image_folder={resolve_existing_repo_relative_path(repo_root, 'entropy_exp/datasets/gqa/images')}",
            f"output.base_dir={args.native_output_root.resolve()}",
        ]
        config = prune_inference.apply_overrides(config, overrides)

        run_dir, _answers_path, _stats_path = original_run_prune_inference(
            config=config,
            dataset_name=DEFAULT_DATASET,
            max_samples=DEFAULT_TINY_SAMPLE_COUNT,
            run_mode="prune",
        )

        eval_summary = evaluate_gqa_tiny_run(repo_root, Path(run_dir))
        eval_summary_path = Path(run_dir) / "eval" / "summary.json"

        stats_rows = read_jsonl(Path(run_dir) / "stats.jsonl")
        if len(stats_rows) != len(tracer.records):
            raise RuntimeError(
                f"Trace sample count mismatch for {args.branch_label}/{args.config_label}: "
                f"{len(tracer.records)} traced vs {len(stats_rows)} stats rows."
            )

        merged_records: list[dict[str, Any]] = []
        for record, stats_row in zip(tracer.records, stats_rows):
            merged = dict(record)
            merged["question_id"] = str(stats_row["question_id"])
            merged["answer"] = stats_row["answer"]
            merged["original_seq_len"] = int(stats_row["original_seq_len"])
            merged["final_seq_len"] = int(stats_row["final_seq_len"])
            merged["prefill_time"] = float(stats_row["prefill_time"])
            merged["decode_time"] = float(stats_row["decode_time"])
            merged["compat_patch_enabled"] = bool(tracer.compat_patch_enabled)
            merged["target_layer_compat_wrapped"] = bool(tracer.target_layer_compat_wrapped)
            merged_records.append(merged)

        write_jsonl(args.trace_file.resolve(), merged_records)

        result = {
            "branch_label": args.branch_label,
            "config_label": args.config_label,
            "repo_root": str(repo_root),
            "run_dir": str(Path(run_dir).resolve()),
            "trace_file": str(args.trace_file.resolve()),
            "eval_summary_path": str(eval_summary_path.resolve()),
            "eval_summary": eval_summary,
            "compat_patch_enabled": bool(tracer.compat_patch_enabled),
            "target_layer_compat_wrapped": bool(tracer.target_layer_compat_wrapped),
            "records": merged_records,
        }
        write_json(args.result_json.resolve(), result)
    finally:
        prune_inference.build_run_rel_dir = original_build_run_rel_dir
        prune_inference.load_pretrained_model = original_load_pretrained_model
        pruner_module.F.softmax = original_softmax
        pruner_module.VisualTokenPruner.pruned_generate = original_pruned_generate
        pruner_module.VisualTokenPruner._pruned_prefill = original_pruned_prefill
        pruner_module.VisualTokenPruner._run_masking_prune_layer = original_run_masking_prune_layer
        pruner_module.VisualTokenPruner._compute_pre_prune_scores = original_compute_pre_prune_scores
        pruner_module.VisualTokenPruner._forward_masked_layer = original_forward_masked_layer
        strategies_base.PruneStrategy.compute_keep_mask_from_importance = original_compute_keep_mask_from_importance
        if original_enable_sparse_position_ids_compat is not None:
            pruner_module.enable_sparse_position_ids_compat = original_enable_sparse_position_ids_compat
            if hasattr(prune_inference, "enable_sparse_position_ids_compat"):
                prune_inference.enable_sparse_position_ids_compat = original_enable_sparse_position_ids_compat


def main() -> None:
    args = parse_args()
    if args.worker:
        worker_main(args)
        return

    keep_root = args.keep_root.resolve()
    nonpos_root = args.nonpos_root.resolve()
    output_dir = args.output_dir.resolve()
    compare_dir = ensure_dir(output_dir / "compare")
    trace_dir = ensure_dir(output_dir / "trace")
    runs_dir = ensure_dir(output_dir / "runs")
    native_dir = ensure_dir(output_dir / "native")

    config_labels = parse_config_labels(args.configs)
    question_file = args.question_file.resolve() if args.question_file else (
        keep_root / "entropy_exp" / "eval_questions" / "gqa" / "llava_gqa_testdev_balanced.jsonl"
    )
    tiny_question_file = prepare_tiny_question_file(question_file, output_dir, args.dataset)

    summary: dict[str, Any] = {
        "keep_root": str(keep_root),
        "nonpos_root": str(nonpos_root),
        "output_dir": str(output_dir),
        "tiny_question_file": str(tiny_question_file),
        "runtime_python": str(resolve_runtime_python()),
        "theory_diff_summary": build_theory_diff_summary(keep_root),
        "formal_pope_compare": compare_formal_summaries(keep_root, nonpos_root),
        "tiny_sample_compare": {},
        "first_divergence": {},
    }

    script_path = Path(__file__).resolve()
    runtime_python = Path(summary["runtime_python"])
    worker_results: dict[str, dict[str, dict[str, Any]]] = {}
    for config_label in config_labels:
        worker_results[config_label] = {}
        for branch_label, branch_spec in BRANCH_SPECS.items():
            repo_root = keep_root if branch_label == "keep_position_ids" else nonpos_root
            trace_file = trace_dir / branch_label / f"{config_label}.jsonl"
            native_output_root = native_dir / branch_label / config_label
            result_json = compare_dir / "worker_results" / f"{branch_label}__{config_label}.json"
            ensure_dir(result_json.parent)

            result = launch_worker(
                python_bin=runtime_python,
                script_path=script_path,
                repo_root=repo_root,
                branch_label=branch_label,
                config_label=config_label,
                tiny_question_file=tiny_question_file,
                trace_file=trace_file,
                native_output_root=native_output_root,
                result_json=result_json,
                gpu=args.gpu,
            )
            worker_results[config_label][branch_label] = result

            alias_path = runs_dir / branch_label / config_label
            replace_symlink(alias_path, Path(result["run_dir"]))

        config_compare, first_divergence = compare_worker_outputs(
            worker_results[config_label]["keep_position_ids"],
            worker_results[config_label]["non_position_id"],
        )
        summary["tiny_sample_compare"][config_label] = config_compare
        summary["first_divergence"][config_label] = first_divergence

    artifact_summary_path = compare_dir / "artifact_diff_summary.json"
    write_json(artifact_summary_path, summary)

    conclusion_path = compare_dir / "conclusion.md"
    conclusion_path.write_text(build_conclusion_markdown(summary), encoding="utf-8")

    print(f"Saved artifact summary: {artifact_summary_path}")
    print(f"Saved conclusion: {conclusion_path}")


if __name__ == "__main__":
    main()
