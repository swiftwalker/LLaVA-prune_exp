"""Shared local VQA dataset compatibility helpers for official wrappers."""

from __future__ import annotations

import base64
import csv
import json
import math
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image


SUPPORTED_DATASETS = ("gqa", "mme", "pope", "textvqa", "scienceqa", "mmbench", "mmvet", "ai2d")
DIRECT_OPTION_ANSWER_PROMPT = "Answer with the option's letter from the given choices directly."
MMBENCH_OPTION_KEYS = ("A", "B", "C", "D")


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    text = str(value).strip()
    return not text or text.lower() in {"nan", "none"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_mmbench_prompt(row: dict[str, str]) -> str:
    parts: list[str] = []
    if not is_missing(row.get("hint")):
        parts.append(str(row["hint"]).strip())
    if not is_missing(row.get("question")):
        parts.append(str(row["question"]).strip())
    for option_key in MMBENCH_OPTION_KEYS:
        option_value = row.get(option_key)
        if is_missing(option_value):
            continue
        parts.append(f"{option_key}. {str(option_value).strip()}")
    parts.append(DIRECT_OPTION_ANSWER_PROMPT)
    return "\n".join(parts)


def load_mmbench(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            image_base64 = row.get("image")
            rows.append(
                {
                    "question_id": str(row["index"]),
                    "image": None,
                    "image_base64": image_base64,
                    "text": build_mmbench_prompt(row),
                    "single_pred_prompt": False,
                }
            )
    return rows


def load_questions(dataset: str, path: Path) -> list[dict[str, Any]]:
    if dataset == "scienceqa":
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    if dataset == "mmbench":
        return load_mmbench(path)
    return load_jsonl(path)


def normalize_item(dataset: str, item: dict[str, Any]) -> dict[str, Any]:
    if dataset == "scienceqa":
        question = item["conversations"][0]
        return {
            "question_id": item["id"],
            "image": item.get("image"),
            "image_base64": None,
            "text": str(question["value"]).replace("<image>", "").strip(),
            "single_pred_prompt": True,
        }
    if dataset == "mmbench":
        return item
    return {
        "question_id": item["question_id"],
        "image": item.get("image"),
        "image_base64": item.get("image_base64"),
        "text": item["text"],
        "single_pred_prompt": False,
    }


def open_image(item: dict[str, Any], image_folder: Path) -> Image.Image | None:
    image_base64 = item.get("image_base64")
    if image_base64 and not is_missing(image_base64):
        payload = base64.b64decode(str(image_base64))
        return Image.open(BytesIO(payload)).convert("RGB")

    image_file = item.get("image")
    if image_file:
        return Image.open(image_folder / image_file).convert("RGB")
    return None
