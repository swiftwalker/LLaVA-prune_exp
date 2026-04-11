"""Dataset loading adapters for entropy_exp pruning inference."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


SUPPORTED_DATASETS = (
    "gqa",
    "mme",
    "pope",
    "textvqa",
    "scienceqa",
    "mmbench",
)

DIRECT_OPTION_ANSWER_PROMPT = "Answer with the option's letter from the given choices directly."
SCIENCEQA_DIRECT_ANSWER_PROMPT = DIRECT_OPTION_ANSWER_PROMPT
MMBENCH_DIRECT_ANSWER_PROMPT = DIRECT_OPTION_ANSWER_PROMPT
MMBENCH_OPTION_KEYS = ("A", "B", "C", "D")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            samples.append(
                {
                    "question_id": str(payload["question_id"]),
                    "text": str(payload["text"]).strip(),
                    "image_path": payload.get("image"),
                    "image_base64": None,
                    "has_image": bool(payload.get("image")),
                }
            )
    return samples


def _strip_leading_image_token(text: str) -> str:
    stripped = text.lstrip()
    if stripped.startswith("<image>"):
        stripped = stripped[len("<image>") :].lstrip("\n").strip()
    return stripped


def _append_direct_answer_prompt(text: str, prompt: str) -> str:
    normalized = text.strip()
    if normalized.endswith(prompt):
        return normalized
    return f"{normalized}\n{prompt}"


def _load_scienceqa(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"ScienceQA question file must be a JSON list: {path}")

    samples: list[dict[str, Any]] = []
    for item in payload:
        conversations = item.get("conversations") or []
        if not conversations:
            raise ValueError(f"ScienceQA item is missing conversations: {item!r}")
        human_turn = conversations[0]
        text = _strip_leading_image_token(str(human_turn.get("value", "")))
        image_path = item.get("image")
        samples.append(
            {
                "question_id": str(item["id"]),
                "text": _append_direct_answer_prompt(text, SCIENCEQA_DIRECT_ANSWER_PROMPT),
                "image_path": image_path,
                "image_base64": None,
                "has_image": bool(image_path),
            }
        )
    return samples


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    text = str(value).strip()
    if not text:
        return True
    return text.lower() in {"nan", "none"}


def _build_mmbench_prompt(row: dict[str, str]) -> str:
    parts: list[str] = []
    hint = row.get("hint")
    question = row.get("question")
    if not _is_missing_value(hint):
        parts.append(str(hint).strip())
    if not _is_missing_value(question):
        parts.append(str(question).strip())

    for option_key in MMBENCH_OPTION_KEYS:
        option_value = row.get(option_key)
        if _is_missing_value(option_value):
            continue
        parts.append(f"{option_key}. {str(option_value).strip()}")

    parts.append(MMBENCH_DIRECT_ANSWER_PROMPT)
    return "\n".join(parts)


def _load_mmbench(path: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            image_base64 = row.get("image")
            samples.append(
                {
                    "question_id": str(row["index"]),
                    "text": _build_mmbench_prompt(row),
                    "image_path": None,
                    "image_base64": image_base64,
                    "has_image": not _is_missing_value(image_base64),
                }
            )
    return samples


def load_dataset_samples(dataset_name: str, question_file: str | Path) -> list[dict[str, Any]]:
    path = Path(question_file)
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if dataset_name in {"gqa", "mme", "pope", "textvqa"}:
        return _load_jsonl(path)
    if dataset_name == "scienceqa":
        return _load_scienceqa(path)
    if dataset_name == "mmbench":
        return _load_mmbench(path)
    raise ValueError(f"Unhandled dataset: {dataset_name}")


def count_dataset_samples(dataset_name: str, question_file: str | Path) -> int:
    path = Path(question_file)
    if dataset_name in {"gqa", "mme", "pope", "textvqa"}:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    if dataset_name == "scienceqa":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list):
            raise ValueError(f"ScienceQA question file must be a JSON list: {path}")
        return len(payload)
    if dataset_name == "mmbench":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            return sum(1 for _ in reader)
    raise ValueError(f"Unsupported dataset: {dataset_name}")
