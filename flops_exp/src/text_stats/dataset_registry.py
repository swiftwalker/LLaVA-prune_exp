import json
from pathlib import Path

from utils.io import load_yaml


THIS_DIR = Path(__file__).absolute().parent
FLOPS_ROOT = THIS_DIR.parent.parent
REPO_ROOT = FLOPS_ROOT.parent
DATASET_CONFIG_DIR = FLOPS_ROOT / "configs" / "datasets"


def resolve_repo_path(path: str | Path) -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    return REPO_ROOT / path_obj


def load_dataset_config(dataset_name: str | None = None, config_path: str | None = None) -> dict:
    if config_path:
        config = load_yaml(resolve_repo_path(config_path))
    elif dataset_name:
        config = load_yaml(DATASET_CONFIG_DIR / f"{dataset_name}.yaml")
    else:
        raise ValueError("Either dataset_name or config_path must be provided")
    return config


def load_dataset_records(config: dict, sample_limit: int | None = None) -> list[dict]:
    question_file = resolve_repo_path(config["question_file"])
    with question_file.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    if sample_limit is not None:
        rows = rows[:sample_limit]
    return rows
