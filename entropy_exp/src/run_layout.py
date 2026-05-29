"""Utilities for discovering and organizing pruning run directories."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml


LLAVA_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_BASE_DIR = LLAVA_ROOT / "entropy_exp" / "outputs"
DEFAULT_RUNS_DIR = DEFAULT_OUTPUT_BASE_DIR / "runs"
RUN_MARKER_FILES = ("config.yaml", "answers.jsonl")
KNOWN_DATASETS = ("gqa", "mme", "pope", "textvqa", "scienceqa", "mmbench", "mmvet", "ai2d")
KNOWN_STRATEGIES = (
    "tail_masking_attn_score",
    "masking_attn_score",
    "pre_attn_score",
    "attn_score",
    "entropy",
    "random",
    "sparsevlm",
    "sparsevlm_adaptive_stratified",
    "sparsevlm_boost",
    "sparsevlm_boost_hybrid",
    "sparsevlm_compensated",
    "sparsevlm_entropy_alpha",
    "sparsevlm_entropy_alpha_global",
    "baseline",
)


@dataclass(frozen=True)
class RunMetadata:
    run_dir: Path
    run_name: str
    dataset: str | None
    strategy: str | None
    run_mode: str | None
    timestamp: str | None
    run_rel_dir: str | None


def resolve_repo_path(path: str | os.PathLike[str] | Path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved.resolve()
    return (LLAVA_ROOT / resolved).resolve()


def resolve_output_base_dir(path: str | os.PathLike[str] | Path | None = None) -> Path:
    if path is None:
        return DEFAULT_OUTPUT_BASE_DIR
    return resolve_repo_path(path)


def resolve_runs_dir(path: str | os.PathLike[str] | Path | None = None) -> Path:
    if path is None:
        return DEFAULT_RUNS_DIR
    return resolve_repo_path(path)


def build_runs_base_dir(output_base_dir: str | os.PathLike[str] | Path | None = None) -> Path:
    return resolve_output_base_dir(output_base_dir) / "runs"


def build_run_dir(
    output_base_dir: str | os.PathLike[str] | Path,
    strategy: str,
    dataset: str,
    run_name: str,
) -> Path:
    return build_runs_base_dir(output_base_dir) / strategy / dataset / run_name


def build_run_rel_dir(
    output_base_dir: str | os.PathLike[str] | Path,
    strategy: str,
    dataset: str,
    run_name: str,
) -> str:
    run_dir = build_run_dir(output_base_dir, strategy, dataset, run_name)
    return str(run_dir.relative_to(LLAVA_ROOT))


def is_run_dir(path: Path, required_files: Sequence[str] = ()) -> bool:
    if not path.is_dir():
        return False
    if not any((path / marker).is_file() for marker in RUN_MARKER_FILES):
        return False
    return all((path / required).is_file() for required in required_files)


def iter_run_dirs(
    runs_dir: str | os.PathLike[str] | Path | None = None,
    required_files: Sequence[str] = (),
) -> list[Path]:
    base_dir = resolve_runs_dir(runs_dir)
    if not base_dir.exists():
        return []

    run_dirs: list[Path] = []
    for root, dirs, files in os.walk(base_dir, topdown=True):
        root_path = Path(root)
        if any(marker in files for marker in RUN_MARKER_FILES):
            if all((root_path / required).is_file() for required in required_files):
                run_dirs.append(root_path.resolve())
            # A run dir is terminal for our purposes; skip eval/analysis descendants.
            dirs[:] = []
            continue

    return sorted(run_dirs, key=lambda path: str(path))


def load_run_config(run_dir: str | os.PathLike[str] | Path) -> dict[str, Any]:
    config_path = Path(run_dir) / "config.yaml"
    if not config_path.is_file():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def infer_dataset_and_strategy_from_name(run_name: str) -> tuple[str | None, str | None]:
    for dataset in KNOWN_DATASETS:
        prefix = f"{dataset}_"
        if not run_name.startswith(prefix):
            continue

        suffix = run_name[len(prefix) :]
        for strategy in sorted(KNOWN_STRATEGIES, key=len, reverse=True):
            if suffix == strategy or suffix.startswith(f"{strategy}_") or suffix.startswith(f"{strategy}__"):
                return dataset, strategy
        return dataset, None

    return None, None


def read_run_metadata(run_dir: str | os.PathLike[str] | Path) -> RunMetadata:
    resolved_run_dir = Path(run_dir).resolve()
    config = load_run_config(resolved_run_dir)
    run_meta = config.get("_run_meta", {}) or {}
    pruning = config.get("pruning", {}) or {}

    dataset = run_meta.get("dataset")
    run_mode = run_meta.get("run_mode")
    strategy = run_meta.get("strategy")
    if not strategy:
        strategy = "baseline" if run_mode == "baseline" else pruning.get("strategy")

    fallback_dataset, fallback_strategy = infer_dataset_and_strategy_from_name(resolved_run_dir.name)
    dataset = dataset or fallback_dataset
    strategy = strategy or fallback_strategy

    run_rel_dir = run_meta.get("run_rel_dir")
    if not run_rel_dir:
        try:
            run_rel_dir = str(resolved_run_dir.relative_to(LLAVA_ROOT))
        except ValueError:
            run_rel_dir = None

    return RunMetadata(
        run_dir=resolved_run_dir,
        run_name=resolved_run_dir.name,
        dataset=dataset,
        strategy=strategy,
        run_mode=run_mode,
        timestamp=run_meta.get("timestamp"),
        run_rel_dir=run_rel_dir,
    )


def find_run_dirs(
    runs_dir: str | os.PathLike[str] | Path | None = None,
    required_files: Sequence[str] = (),
    name_prefix: str | None = None,
    exact_name: str | None = None,
    dataset: str | None = None,
    strategy: str | None = None,
) -> list[Path]:
    matches: list[Path] = []
    for run_dir in iter_run_dirs(runs_dir=runs_dir, required_files=required_files):
        run_name = run_dir.name
        if exact_name is not None and run_name != exact_name:
            continue
        if name_prefix is not None and not run_name.startswith(name_prefix):
            continue

        if dataset is not None or strategy is not None:
            metadata = read_run_metadata(run_dir)
            if dataset is not None and metadata.dataset != dataset:
                continue
            if strategy is not None and metadata.strategy != strategy:
                continue

        matches.append(run_dir)

    return matches


def map_run_dirs_by_name(run_dirs: Iterable[str | os.PathLike[str] | Path]) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for run_dir in run_dirs:
        path = Path(run_dir).resolve()
        mapping[path.name] = path
    return mapping
