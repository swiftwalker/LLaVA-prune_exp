"""Load and validate algo_compare YAML manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .paths import resolve_repo_path


VALID_SOURCE_TYPES = {"official", "local_reference"}


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest = load_yaml(path)
    if "runs" not in manifest:
        manifest["runs"] = []
    if not isinstance(manifest["runs"], list):
        raise ValueError(f"manifest runs must be a list: {path}")
    return manifest


def iter_manifest_paths(run: dict[str, Any]) -> list[tuple[str, Path]]:
    path_fields = [
        "output_root",
        "summary_csv",
        "summary_json",
        "analysis_dir",
        "result_path",
        "answers_file",
        "eval_summary",
    ]
    paths: list[tuple[str, Path]] = []
    for field in path_fields:
        value = run.get(field)
        if value:
            paths.append((field, resolve_repo_path(value)))
    return paths


def validate_manifest(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for idx, run in enumerate(manifest.get("runs", []), start=1):
        label = run.get("label", f"run_{idx}")
        source_type = run.get("source_type")
        if source_type not in VALID_SOURCE_TYPES:
            errors.append(
                f"{label}: source_type must be one of {sorted(VALID_SOURCE_TYPES)}, got {source_type!r}"
            )
        if source_type == "official" and not run.get("official_repo_commit"):
            errors.append(f"{label}: official runs must include official_repo_commit")
        if source_type == "local_reference":
            method = str(run.get("method", "")).lower()
            if any(name in method for name in ["entropy_alpha", "boost", "compensated", "global"]):
                errors.append(
                    f"{label}: local exploration method {run.get('method')!r} must not be marked as official"
                )
    return errors
