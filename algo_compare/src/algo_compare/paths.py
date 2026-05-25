"""Path helpers for the algo_compare workspace."""

from __future__ import annotations

from pathlib import Path


ALGO_COMPARE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = ALGO_COMPARE_ROOT.parent


def resolve_repo_path(path: str | Path) -> Path:
    """Resolve paths relative to the repository root."""
    value = Path(path)
    if value.is_absolute():
        return value
    return REPO_ROOT / value


def method_dirs() -> list[Path]:
    """Return top-level method directories that contain method.yaml."""
    ignored = {"scripts", "src", "__pycache__"}
    dirs: list[Path] = []
    for child in sorted(ALGO_COMPARE_ROOT.iterdir()):
        if not child.is_dir() or child.name in ignored:
            continue
        if (child / "method.yaml").is_file():
            dirs.append(child)
    return dirs
