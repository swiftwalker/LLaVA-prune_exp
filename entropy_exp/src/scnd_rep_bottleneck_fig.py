"""Mine and render SCND representation-bottleneck figure evidence.

The main use case is the retain64 comparison between saliency top-K (S-S-S)
and SCND (C-B-S).  Candidate mining is offline and reads existing stats.jsonl
files.  Final figure rendering consumes a small opt-in HDF5 hidden-state export
created by diagnostics.rep_bottleneck in prune_inference.py.
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import math
import mimetypes
import os
import re
import subprocess
import sys
from collections import deque
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont


SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parents[1]
sys.path.insert(0, os.fspath(SRC_DIR))

from dataset_adapters import load_dataset_samples  # noqa: E402


DEFAULT_DATASETS = ("gqa", "textvqa", "pope", "mme", "scienceqa")
DEFAULT_LAYER = 2
DEFAULT_TARGET_KEEP = 66
DEFAULT_PATCH_PER_ROW = 24
DEFAULT_COARSE_GRID = 6
DEFAULT_TOPK_ROOT = REPO_ROOT / "entropy_exp/outputs/scnd_gpu_sss_saliency_l2_6_16"
DEFAULT_SCND_ROOT = REPO_ROOT / "entropy_exp/outputs/scnd_ablation_bcd_l2_6_16"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "entropy_exp/outputs/analysis/scnd_rep_bottleneck_fig_retain64_l2"
DEFAULT_CONFIG = REPO_ROOT / "entropy_exp/configs/prune.yaml"
OUTCOME_DATASETS = ("textvqa", "pope")
DEFAULT_REP_FIG_SHORTLIST_RANKS = (21, 3, 31, 8, 11, 13, 15, 34, 39, 41, 43, 45, 49, 6, 46)
DEFAULT_REP_FIG_PREFERRED_RANKS = (21, 3)
DEFAULT_PAIRWISE_MAIN_RANKS = (46, 8)
TOPK_METHOD_COLOR = (96, 112, 128)  # steel gray
SCND_METHOD_COLOR = (209, 73, 91)   # #d1495b
RED_HEATMAP_COLOR = (209, 73, 91)
HEATMAP_LOW_COLOR = (250, 250, 252)
HEATMAP_HIGH_COLOR = RED_HEATMAP_COLOR

# Patch-grid ROIs for the manually reviewed TextVQA candidates. Coordinates are
# (row_min, col_min, row_max_exclusive, col_max_exclusive) on the 24x24 LLaVA grid.
REP_FIG_ROI_PATCH_BBOX_BY_RANK = {
    3: (7, 3, 16, 23),
    8: (5, 7, 12, 16),
    21: (3, 2, 13, 8),
    31: (0, 5, 6, 22),
    46: (1, 0, 8, 24),
}


def normalize_question_id(value: Any) -> str:
    return str(value)


def sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def resolve_repo_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return REPO_ROOT / resolved


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def patch_coordinates(indices: list[int], patch_per_row: int = DEFAULT_PATCH_PER_ROW) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for index in indices:
        patch = int(index)
        coords.append((patch // patch_per_row, patch % patch_per_row))
    return coords


def coarse_grid_entropy(
    indices: list[int],
    patch_per_row: int = DEFAULT_PATCH_PER_ROW,
    coarse_grid: int = DEFAULT_COARSE_GRID,
) -> float:
    if not indices:
        return 0.0
    cell_size = patch_per_row / coarse_grid
    counts = np.zeros(coarse_grid * coarse_grid, dtype=np.float64)
    for row, col in patch_coordinates(indices, patch_per_row):
        coarse_row = min(int(row / cell_size), coarse_grid - 1)
        coarse_col = min(int(col / cell_size), coarse_grid - 1)
        counts[coarse_row * coarse_grid + coarse_col] += 1.0
    probs = counts[counts > 0] / float(len(indices))
    entropy = -float(np.sum(probs * np.log(probs)))
    return entropy / math.log(coarse_grid * coarse_grid)


def bbox_area_ratio(indices: list[int], patch_per_row: int = DEFAULT_PATCH_PER_ROW) -> float:
    if not indices:
        return 0.0
    coords = patch_coordinates(indices, patch_per_row)
    rows = [row for row, _ in coords]
    cols = [col for _, col in coords]
    area = (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1)
    return float(area) / float(patch_per_row * patch_per_row)


def largest_component_ratio(indices: list[int], patch_per_row: int = DEFAULT_PATCH_PER_ROW) -> float:
    if not indices:
        return 0.0
    selected = set(patch_coordinates(indices, patch_per_row))
    largest = 0
    while selected:
        start = selected.pop()
        size = 1
        queue: deque[tuple[int, int]] = deque([start])
        while queue:
            row, col = queue.popleft()
            for nr, nc in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                if (nr, nc) in selected:
                    selected.remove((nr, nc))
                    queue.append((nr, nc))
                    size += 1
        largest = max(largest, size)
    return float(largest) / float(len(indices))


def mean_grid_distance(indices: list[int], patch_per_row: int = DEFAULT_PATCH_PER_ROW) -> float:
    if len(indices) < 2:
        return 0.0
    coords = np.asarray(patch_coordinates(indices, patch_per_row), dtype=np.float64)
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    upper = dist[np.triu_indices(len(indices), k=1)]
    normalizer = math.sqrt(2.0) * float(patch_per_row - 1)
    return float(np.mean(upper) / normalizer)


def spatial_metrics(indices: list[int], patch_per_row: int = DEFAULT_PATCH_PER_ROW) -> dict[str, float]:
    return {
        "spatial_entropy_6x6": coarse_grid_entropy(indices, patch_per_row=patch_per_row),
        "bbox_area_ratio": bbox_area_ratio(indices, patch_per_row=patch_per_row),
        "largest_component_ratio": largest_component_ratio(indices, patch_per_row=patch_per_row),
        "mean_grid_distance": mean_grid_distance(indices, patch_per_row=patch_per_row),
    }


def overlap_metrics(a: list[int], b: list[int]) -> dict[str, float | int]:
    set_a = set(map(int, a))
    set_b = set(map(int, b))
    union = set_a | set_b
    overlap = set_a & set_b
    return {
        "overlap_count": len(overlap),
        "jaccard": float(len(overlap)) / float(len(union)) if union else 0.0,
    }


def prefix_metrics(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}{key}": value for key, value in metrics.items()}


def find_stats_path(root: Path, dataset: str, pattern: str) -> Path:
    dataset_dir = root / "runs" / "sparsevlm_scnd" / dataset
    matches = sorted(dataset_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No stats.jsonl matching {pattern!r} under {dataset_dir}")
    return matches[-1]


def load_stats_by_qid(path: Path, layer: int, target_keep: int) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in load_stats_records(path, layer, target_keep):
        qid = normalize_question_id(row["question_id"])
        rows[qid] = row
    return rows


def load_stats_records(path: Path, layer: int, target_keep: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    keep_key = f"layer_{layer}_target_keep"
    patch_key = f"layer_{layer}_keep_patch_indices"
    for row in read_jsonl(path):
        if int(row.get(keep_key, -1)) != target_keep:
            continue
        if patch_key not in row:
            continue
        rows.append(row)
    return rows


def stats_ref(row: dict[str, Any]) -> tuple[str, int]:
    return normalize_question_id(row["question_id"]), int(row.get("sample_idx", -1))


def load_stats_by_ref(
    path: Path,
    layer: int,
    target_keep: int,
    question_id: str,
    sample_idx: int,
) -> dict[str, Any]:
    for row in load_stats_records(path, layer, target_keep):
        if stats_ref(row) == (normalize_question_id(question_id), int(sample_idx)):
            return row
    raise KeyError(f"Stats row not found for question_id={question_id}, sample_idx={sample_idx} in {path}")


def load_answers_by_ref(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    answers: dict[tuple[str, int], dict[str, Any]] = {}
    for sample_idx, row in enumerate(read_jsonl(path)):
        answers[(normalize_question_id(row["question_id"]), sample_idx)] = row
    return answers


def find_run_file(root: Path, dataset: str, pattern: str, filename: str) -> Path:
    stats_path = find_stats_path(root, dataset, pattern)
    path = stats_path.parent / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def normalize_pope_answer(text: str) -> str:
    first_sentence = str(text).split(".", 1)[0]
    first_sentence = first_sentence.replace(",", " ")
    words = {word.strip().lower() for word in first_sentence.split()}
    return "no" if ("no" in words or "not" in words) else "yes"


class OutcomeScorer:
    def __init__(self, dataset: str, correct_threshold: float = 0.0) -> None:
        self.dataset = dataset
        self.correct_threshold = correct_threshold
        if dataset == "textvqa":
            from eval_datasets import load_textvqa_evaluator, textvqa_prompt_processor

            self.textvqa_prompt_processor = textvqa_prompt_processor
            annotation_file = REPO_ROOT / "entropy_exp/eval_questions/textvqa/TextVQA_0.5.1_val.json"
            annotations = json.loads(annotation_file.read_text(encoding="utf-8"))["data"]
            self.annotation_map = {
                (normalize_question_id(item["image_id"]), str(item["question"]).lower()): item
                for item in annotations
            }
            self.evaluator = load_textvqa_evaluator()
        elif dataset == "pope":
            question_file = REPO_ROOT / "entropy_exp/eval_questions/pope/llava_pope_test.jsonl"
            annotation_dir = REPO_ROOT / "entropy_exp/datasets/pope/coco"
            labels_by_category: dict[str, list[dict[str, Any]]] = {}
            for label_path in sorted(annotation_dir.glob("coco_pope_*.json")):
                category = label_path.name[len("coco_pope_") : -len(".json")]
                labels_by_category[category] = read_jsonl(label_path)
            category_offsets = {category: 0 for category in labels_by_category}
            self.metadata: dict[str, dict[str, Any]] = {}
            for question in read_jsonl(question_file):
                category = str(question["category"])
                offset = category_offsets[category]
                category_offsets[category] += 1
                label_row = labels_by_category[category][offset]
                question_id = normalize_question_id(question["question_id"])
                self.metadata[question_id] = {
                    "label": str(label_row["label"]).lower(),
                    "category": category,
                    "question": str(question["text"]).split("\n", 1)[0],
                    "image_file": str(question["image"]),
                }
        else:
            raise ValueError(f"Outcome scorer is not available for dataset={dataset!r}")

    def evaluate(self, answer: dict[str, Any]) -> dict[str, Any]:
        question_id = normalize_question_id(answer["question_id"])
        if self.dataset == "textvqa":
            question = self.textvqa_prompt_processor(answer["prompt"])
            annotation = self.annotation_map[(question_id, question)]
            scores = self.evaluator._compute_answer_scores(annotation["answers"])
            parsed = self.evaluator.answer_processor(answer["text"])
            score = float(scores.get(parsed, 0.0))
            gt_answers = list(dict.fromkeys(str(value) for value in annotation["answers"]))
            return {
                "score": score,
                "correct": score > self.correct_threshold,
                "parsed_answer": parsed,
                "gt_answer": " | ".join(gt_answers[:5]),
                "question": str(annotation["question"]),
                "image_file": f"{annotation['image_id']}.jpg",
                "category": "",
            }
        metadata = self.metadata[question_id]
        parsed = normalize_pope_answer(str(answer["text"]))
        score = 1.0 if parsed == metadata["label"] else 0.0
        return {
            "score": score,
            "correct": bool(score),
            "parsed_answer": parsed,
            "gt_answer": metadata["label"],
            "question": metadata["question"],
            "image_file": metadata["image_file"],
            "category": metadata["category"],
        }


def classify_outcome(topk_eval: dict[str, Any], scnd_eval: dict[str, Any]) -> str:
    if topk_eval["correct"] and scnd_eval["correct"]:
        return "both_correct"
    if topk_eval["correct"] and not scnd_eval["correct"]:
        return "topk_correct_scnd_wrong"
    if not topk_eval["correct"] and scnd_eval["correct"]:
        return "scnd_correct_topk_wrong"
    return "both_wrong"


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_question_metadata(config: dict[str, Any], dataset: str) -> dict[str, dict[str, Any]]:
    ds_cfg = config["datasets"][dataset]
    question_file = resolve_repo_path(ds_cfg["question_file"])
    rows = load_dataset_samples(dataset, os.fspath(question_file))
    return {normalize_question_id(row["question_id"]): row for row in rows}


def load_question_rows(config: dict[str, Any], dataset: str) -> list[dict[str, Any]]:
    ds_cfg = config["datasets"][dataset]
    question_file = resolve_repo_path(ds_cfg["question_file"])
    return load_dataset_samples(dataset, os.fspath(question_file))


def resolve_image_path(config: dict[str, Any], dataset: str, image_file: str | None) -> Path | None:
    if not image_file or image_file.startswith("inline_base64:"):
        return None
    image_path = Path(image_file)
    if image_path.is_absolute():
        return image_path
    image_folder = config["datasets"][dataset].get("image_folder")
    if image_folder is None:
        return None
    return resolve_repo_path(image_folder) / image_path


def build_candidate_record(
    dataset: str,
    question_id: str,
    sample_idx: int,
    topk_row: dict[str, Any],
    scnd_row: dict[str, Any],
    question_row: dict[str, Any] | None,
    topk_stats_path: Path,
    scnd_stats_path: Path,
    layer: int,
) -> dict[str, Any]:
    patch_key = f"layer_{layer}_keep_patch_indices"
    topk_indices = [int(item) for item in topk_row[patch_key]]
    scnd_indices = [int(item) for item in scnd_row[patch_key]]

    topk_spatial = spatial_metrics(topk_indices)
    scnd_spatial = spatial_metrics(scnd_indices)
    overlap = overlap_metrics(topk_indices, scnd_indices)
    topk_answer = str(topk_row.get("answer", "")).strip()
    scnd_answer = str(scnd_row.get("answer", "")).strip()
    answer_agreement = topk_answer.lower() == scnd_answer.lower()
    scnd_hidden_distance = float(scnd_row.get(f"layer_{layer}_mean_selected_pairwise_distance", 0.0) or 0.0)

    entropy_gain = scnd_spatial["spatial_entropy_6x6"] - topk_spatial["spatial_entropy_6x6"]
    grid_distance_gain = scnd_spatial["mean_grid_distance"] - topk_spatial["mean_grid_distance"]
    bbox_gain = scnd_spatial["bbox_area_ratio"] - topk_spatial["bbox_area_ratio"]
    selection_score = (
        1.5 * entropy_gain
        + 1.2 * grid_distance_gain
        + 0.6 * bbox_gain
        + 0.8 * (1.0 - float(overlap["jaccard"]))
        + 0.4 * scnd_hidden_distance
        - 0.3 * topk_spatial["spatial_entropy_6x6"]
        + (0.2 if answer_agreement else 0.0)
    )

    question_text = ""
    image_file = topk_row.get("image_file") or scnd_row.get("image_file")
    if question_row is not None:
        question_text = str(question_row.get("text", ""))
        image_file = image_file or question_row.get("image_path")

    return {
        "dataset": dataset,
        "question_id": question_id,
        "sample_idx": int(sample_idx),
        "match_key": f"{dataset}:{sample_idx}:{question_id}",
        "image_file": image_file or "",
        "question": question_text,
        "topk_answer": topk_answer,
        "scnd_answer": scnd_answer,
        "answer_agreement": bool(answer_agreement),
        "selection_score": float(selection_score),
        "topk_keep_patch_indices": json.dumps(topk_indices),
        "scnd_keep_patch_indices": json.dumps(scnd_indices),
        **prefix_metrics("topk_", topk_spatial),
        **prefix_metrics("scnd_", scnd_spatial),
        "entropy_gain": float(entropy_gain),
        "grid_distance_gain": float(grid_distance_gain),
        "bbox_gain": float(bbox_gain),
        **overlap,
        "scnd_mean_selected_pairwise_distance": scnd_hidden_distance,
        "topk_run_dir": os.fspath(topk_stats_path.parent),
        "scnd_run_dir": os.fspath(scnd_stats_path.parent),
        "topk_stats_path": os.fspath(topk_stats_path),
        "scnd_stats_path": os.fspath(scnd_stats_path),
    }


def load_processed_square(image_path: Path | None, size: int = 336) -> Image.Image:
    if image_path is None or not image_path.is_file():
        return Image.new("RGB", (size, size), (238, 238, 238))
    image = Image.open(image_path).convert("RGB")
    side = max(image.size)
    canvas = Image.new("RGB", (side, side), (122, 116, 104))
    offset = ((side - image.size[0]) // 2, (side - image.size[1]) // 2)
    canvas.paste(image, offset)
    return canvas.resize((size, size), Image.BICUBIC)


def render_patch_overlay(
    image_path: Path | None,
    patch_indices: list[int],
    *,
    patch_per_row: int = DEFAULT_PATCH_PER_ROW,
    size: int = 336,
    color: tuple[int, int, int] = (220, 40, 40),
) -> Image.Image:
    base = load_processed_square(image_path, size=size).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    cell = size / float(patch_per_row)
    selected = set(map(int, patch_indices))
    for patch in selected:
        row = patch // patch_per_row
        col = patch % patch_per_row
        xy = (
            int(round(col * cell)),
            int(round(row * cell)),
            int(round((col + 1) * cell)),
            int(round((row + 1) * cell)),
        )
        draw.rectangle(xy, fill=(*color, 95), outline=(*color, 230), width=1)
    return Image.alpha_composite(base, overlay).convert("RGB")


def render_selection_grid(
    patch_indices: list[int],
    *,
    patch_per_row: int = DEFAULT_PATCH_PER_ROW,
    cell_size: int = 5,
    color: tuple[int, int, int] = (220, 40, 40),
) -> Image.Image:
    size = patch_per_row * cell_size
    image = Image.new("RGB", (size, size), (245, 245, 245))
    draw = ImageDraw.Draw(image)
    for idx in range(patch_per_row + 1):
        pos = idx * cell_size
        draw.line((pos, 0, pos, size), fill=(220, 220, 220))
        draw.line((0, pos, size, pos), fill=(220, 220, 220))
    for patch in map(int, patch_indices):
        row = patch // patch_per_row
        col = patch % patch_per_row
        xy = (
            col * cell_size + 1,
            row * cell_size + 1,
            (col + 1) * cell_size - 1,
            (row + 1) * cell_size - 1,
        )
        draw.rectangle(xy, fill=color)
    return image


def truncate_text(text: str, max_chars: int) -> str:
    text = " ".join(str(text).split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def parse_indices_field(value: Any) -> list[int]:
    if isinstance(value, list):
        return [int(item) for item in value]
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    return [int(item) for item in json.loads(text)]


def render_contact_sheet(
    rows: list[dict[str, Any]],
    output_path: Path,
    *,
    layer: int,
    max_rows: int,
) -> None:
    rows = rows[:max_rows]
    if not rows:
        return
    font = ImageFont.load_default()
    cols = 2
    cell_w = 650
    cell_h = 190
    sheet = Image.new("RGB", (cols * cell_w, math.ceil(len(rows) / cols) * cell_h), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)

    for rank, row in enumerate(rows, start=1):
        col = (rank - 1) % cols
        grid_row = (rank - 1) // cols
        x0 = col * cell_w
        y0 = grid_row * cell_h
        draw.rectangle((x0, y0, x0 + cell_w - 1, y0 + cell_h - 1), outline=(220, 220, 220))

        topk_grid = render_selection_grid(parse_indices_field(row["topk_keep_patch_indices"]), color=(220, 40, 40))
        scnd_grid = render_selection_grid(parse_indices_field(row["scnd_keep_patch_indices"]), color=(40, 110, 220))
        sheet.paste(topk_grid, (x0 + 10, y0 + 46))
        sheet.paste(scnd_grid, (x0 + 140, y0 + 46))

        lines = [
            f"#{rank} {row['dataset']} qid={row['question_id']} score={float(row['selection_score']):.3f}",
            f"topK ent={float(row['topk_spatial_entropy_6x6']):.3f} dist={float(row['topk_mean_grid_distance']):.3f}",
            f"SCND ent={float(row['scnd_spatial_entropy_6x6']):.3f} dist={float(row['scnd_mean_grid_distance']):.3f}",
            f"J={float(row['jaccard']):.3f} same_ans={row['answer_agreement']}",
            f"A: {truncate_text(row['topk_answer'], 42)} | {truncate_text(row['scnd_answer'], 42)}",
            truncate_text(row.get("question", ""), 84),
            truncate_text(row.get("image_file", ""), 84),
        ]
        text_x = x0 + 275
        text_y = y0 + 12
        for line in lines:
            draw.text((text_x, text_y), line, fill=(30, 30, 30), font=font)
            text_y += 18
    sheet.save(output_path)


def as_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def as_bool(row: dict[str, Any], key: str) -> bool:
    value = row.get(key)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def review_assessment(row: dict[str, Any], image_exists: bool = True) -> dict[str, Any]:
    entropy_gain = as_float(row, "entropy_gain")
    grid_gain = as_float(row, "grid_distance_gain")
    jaccard = as_float(row, "jaccard")
    topk_entropy = as_float(row, "topk_spatial_entropy_6x6")
    topk_component = as_float(row, "topk_largest_component_ratio")
    score = as_float(row, "selection_score")
    answer_agreement = as_bool(row, "answer_agreement")

    strengths: list[str] = []
    risks: list[str] = []

    if entropy_gain >= 0.15:
        strengths.append("strong entropy spread")
    elif entropy_gain < 0.08:
        risks.append("weak entropy gain")

    if grid_gain >= 0.06:
        strengths.append("strong spatial distance gain")
    elif grid_gain < 0.03:
        risks.append("weak spatial distance gain")

    if jaccard <= 0.30:
        strengths.append("low overlap")
    elif jaccard > 0.45:
        risks.append("high overlap")

    if topk_entropy <= 0.72 or topk_component >= 0.30:
        strengths.append("topK visibly concentrated")
    elif topk_entropy > 0.82:
        risks.append("topK not concentrated")

    if answer_agreement:
        strengths.append("answer agreement")
    else:
        risks.append("answer disagreement")

    if image_exists:
        strengths.append("image resolved")
    else:
        risks.append("missing image")

    if row.get("dataset") == "mme":
        risks.append("yes/no style; inspect semantic clarity")

    if (
        answer_agreement
        and image_exists
        and entropy_gain >= 0.14
        and grid_gain >= 0.06
        and jaccard <= 0.32
        and score >= 1.20
    ):
        tier = "A"
        verdict = "main-figure candidate"
    elif (
        image_exists
        and entropy_gain >= 0.10
        and grid_gain >= 0.04
        and jaccard <= 0.42
        and answer_agreement
    ):
        tier = "B"
        verdict = "appendix or backup"
    elif image_exists and entropy_gain >= 0.06 and grid_gain >= 0.02:
        tier = "C"
        verdict = "manual review"
    else:
        tier = "D"
        verdict = "low priority"

    return {
        "priority_tier": tier,
        "auto_verdict": verdict,
        "strengths": strengths,
        "risks": risks,
    }


def bar_html(value: float, max_value: float, label: str) -> str:
    width = 0.0 if max_value <= 0 else max(0.0, min(100.0, 100.0 * value / max_value))
    return (
        f'<div class="metric"><span>{html.escape(label)}</span>'
        f'<strong>{value:.3f}</strong>'
        f'<div class="bar"><i style="width:{width:.1f}%"></i></div></div>'
    )


def asset_rel(path: Path, root: Path) -> str:
    return os.fspath(path.relative_to(root)).replace(os.sep, "/")


def build_review_asset_stem(rank: int, row: dict[str, Any]) -> str:
    return (
        f"{rank:03d}_{row['dataset']}_s{int(row['sample_idx']):05d}_"
        f"{sanitize_filename(str(row['question_id']))}"
    )


def render_review_assets(
    row: dict[str, Any],
    rank: int,
    output_dir: Path,
    config: dict[str, Any],
) -> dict[str, str]:
    asset_dir = ensure_dir(output_dir / "review_assets")
    stem = build_review_asset_stem(rank, row)
    image_path = resolve_image_path(config, row["dataset"], row.get("image_file"))
    topk_indices = parse_indices_field(row.get("topk_keep_patch_indices"))
    scnd_indices = parse_indices_field(row.get("scnd_keep_patch_indices"))

    topk_overlay = asset_dir / f"{stem}_topk_overlay.png"
    scnd_overlay = asset_dir / f"{stem}_scnd_overlay.png"
    topk_grid = asset_dir / f"{stem}_topk_grid.png"
    scnd_grid = asset_dir / f"{stem}_scnd_grid.png"
    render_patch_overlay(image_path, topk_indices, size=252, color=(220, 40, 40)).save(topk_overlay)
    render_patch_overlay(image_path, scnd_indices, size=252, color=(40, 110, 220)).save(scnd_overlay)
    render_selection_grid(topk_indices, cell_size=6, color=(220, 40, 40)).save(topk_grid)
    render_selection_grid(scnd_indices, cell_size=6, color=(40, 110, 220)).save(scnd_grid)
    return {
        "topk_overlay": asset_rel(topk_overlay, output_dir),
        "scnd_overlay": asset_rel(scnd_overlay, output_dir),
        "topk_grid": asset_rel(topk_grid, output_dir),
        "scnd_grid": asset_rel(scnd_grid, output_dir),
        "image_exists": str(image_path is not None and image_path.is_file()).lower(),
    }


def review_card_html(
    row: dict[str, Any],
    rank: int,
    assets: dict[str, str],
    assessment: dict[str, Any],
    output_dir: Path,
) -> str:
    tier = assessment["priority_tier"]
    risks = assessment["risks"]
    strengths = assessment["strengths"]
    score = as_float(row, "selection_score")
    dataset = str(row["dataset"])
    question_id = str(row["question_id"])
    question = str(row.get("question", ""))
    image_file = str(row.get("image_file", ""))
    answer_pair = f"{row.get('topk_answer', '')} | {row.get('scnd_answer', '')}"
    search_blob = " ".join([dataset, question_id, question, image_file, answer_pair]).lower()
    hidden_links = ""
    stem = f"{dataset}_s{int(row['sample_idx']):05d}_{sanitize_filename(question_id)}"
    if (output_dir / "fig_panels" / f"{stem}_topk_similarity.png").is_file():
        hidden_links = (
            '<a class="mini-link" href="fig_panels/'
            f'{html.escape(stem)}_topk_similarity.png">TopK heatmap</a>'
            '<a class="mini-link" href="fig_panels/'
            f'{html.escape(stem)}_scnd_similarity.png">SCND heatmap</a>'
        )

    metric_block = "\n".join(
        [
            bar_html(as_float(row, "entropy_gain"), 0.30, "entropy gain"),
            bar_html(as_float(row, "grid_distance_gain"), 0.15, "grid-distance gain"),
            bar_html(1.0 - as_float(row, "jaccard"), 1.0, "non-overlap"),
            bar_html(as_float(row, "topk_largest_component_ratio"), 1.0, "topK component"),
        ]
    )
    strengths_html = "".join(f"<li>{html.escape(item)}</li>" for item in strengths) or "<li>none</li>"
    risks_html = "".join(f"<li>{html.escape(item)}</li>" for item in risks) or "<li>none</li>"
    return f"""
<article class="case-card" data-tier="{tier}" data-dataset="{html.escape(dataset)}" data-score="{score:.6f}" data-search="{html.escape(search_blob)}">
  <header>
    <div>
      <span class="rank">#{rank}</span>
      <span class="tier tier-{tier}">Tier {tier}</span>
      <span class="verdict">{html.escape(assessment['auto_verdict'])}</span>
    </div>
    <strong>{score:.3f}</strong>
  </header>
  <div class="meta">
    <code>{html.escape(dataset)}:{int(row['sample_idx'])}:{html.escape(question_id)}</code>
    <span>{html.escape(image_file)}</span>
  </div>
  <p class="question">{html.escape(question)}</p>
  <p class="answers"><b>TopK</b> {html.escape(str(row.get('topk_answer', '')))} <b>SCND</b> {html.escape(str(row.get('scnd_answer', '')))}</p>
  <div class="visuals">
    <figure><img src="{html.escape(assets['topk_overlay'])}" alt="TopK overlay"><figcaption>saliency top-K overlay</figcaption></figure>
    <figure><img src="{html.escape(assets['scnd_overlay'])}" alt="SCND overlay"><figcaption>SCND overlay</figcaption></figure>
    <figure><img src="{html.escape(assets['topk_grid'])}" alt="TopK grid"><figcaption>TopK 24x24 grid</figcaption></figure>
    <figure><img src="{html.escape(assets['scnd_grid'])}" alt="SCND grid"><figcaption>SCND 24x24 grid</figcaption></figure>
  </div>
  <div class="metrics">{metric_block}</div>
  <div class="checks">
    <div><b>Strengths</b><ul>{strengths_html}</ul></div>
    <div><b>Risks</b><ul>{risks_html}</ul></div>
  </div>
  <div class="links">
    {hidden_links}
    <button type="button" onclick="markCard(this, 'keep')">keep</button>
    <button type="button" onclick="markCard(this, 'maybe')">maybe</button>
    <button type="button" onclick="markCard(this, 'drop')">drop</button>
  </div>
</article>"""


def outcome_tier(row: dict[str, Any], image_exists: bool = True) -> tuple[str, str, list[str], list[str]]:
    case_type = str(row.get("case_type", ""))
    entropy_gain = as_float(row, "entropy_gain")
    grid_gain = as_float(row, "grid_distance_gain")
    jaccard = as_float(row, "jaccard")
    score_gain = as_float(row, "score_gain")
    strengths: list[str] = []
    risks: list[str] = []

    if case_type == "scnd_correct_topk_wrong":
        strengths.append("SCND correct / TopK wrong")
    else:
        risks.append(case_type.replace("_", " "))
    if score_gain > 0:
        strengths.append(f"score gain +{score_gain:.2f}")
    if entropy_gain >= 0.08:
        strengths.append("visible spread gain")
    else:
        risks.append("weak spread gain")
    if grid_gain >= 0.03:
        strengths.append("larger spatial coverage")
    else:
        risks.append("weak distance gain")
    if jaccard <= 0.45:
        strengths.append("selection differs")
    else:
        risks.append("high overlap")
    if not image_exists:
        risks.append("missing image")
    if row.get("dataset") == "pope":
        risks.append("yes/no task; inspect whether visual story is clear")

    if case_type == "scnd_correct_topk_wrong" and image_exists and entropy_gain >= 0.08 and grid_gain >= 0.03:
        return "A", "accuracy-benefit candidate", strengths, risks
    if case_type == "scnd_correct_topk_wrong" and image_exists:
        return "B", "benefit but weak visual contrast", strengths, risks
    if image_exists and entropy_gain >= 0.08:
        return "C", "mechanism backup", strengths, risks
    return "D", "low priority", strengths, risks


def outcome_card_html(row: dict[str, Any], rank: int, assets: dict[str, str], tier_info: tuple[str, str, list[str], list[str]]) -> str:
    tier, verdict, strengths, risks = tier_info
    dataset = str(row["dataset"])
    question_id = str(row["question_id"])
    score = as_float(row, "outcome_selection_score")
    strengths_html = "".join(f"<li>{html.escape(item)}</li>" for item in strengths) or "<li>none</li>"
    risks_html = "".join(f"<li>{html.escape(item)}</li>" for item in risks) or "<li>none</li>"
    metric_block = "\n".join(
        [
            bar_html(as_float(row, "score_gain"), 1.0, "score gain"),
            bar_html(as_float(row, "entropy_gain"), 0.30, "entropy gain"),
            bar_html(as_float(row, "grid_distance_gain"), 0.15, "grid-distance gain"),
            bar_html(1.0 - as_float(row, "jaccard"), 1.0, "non-overlap"),
        ]
    )
    search_blob = " ".join(
        [
            dataset,
            question_id,
            str(row.get("question", "")),
            str(row.get("gt_answer", "")),
            str(row.get("topk_answer", "")),
            str(row.get("scnd_answer", "")),
        ]
    ).lower()
    return f"""
<article class="case-card" data-tier="{tier}" data-dataset="{html.escape(dataset)}" data-score="{score:.6f}" data-search="{html.escape(search_blob)}">
  <header>
    <div>
      <span class="rank">#{rank}</span>
      <span class="tier tier-{tier}">Tier {tier}</span>
      <span class="verdict">{html.escape(verdict)}</span>
    </div>
    <strong>{score:.3f}</strong>
  </header>
  <div class="meta">
    <code>{html.escape(dataset)}:{int(row['sample_idx'])}:{html.escape(question_id)}</code>
    <span>{html.escape(str(row.get('image_file', '')))}</span>
    <span>{html.escape(str(row.get('case_type', '')))}</span>
  </div>
  <p class="question">{html.escape(str(row.get('question', '')))}</p>
  <p class="answers"><b>GT</b> {html.escape(str(row.get('gt_answer', '')))}</p>
  <p class="answers"><b>TopK</b> {html.escape(str(row.get('topk_answer', '')))} <span class="bad">wrong</span> <b>SCND</b> {html.escape(str(row.get('scnd_answer', '')))} <span class="good">correct</span></p>
  <div class="visuals">
    <figure><img src="{html.escape(assets['topk_overlay'])}" alt="TopK overlay"><figcaption>saliency top-K overlay</figcaption></figure>
    <figure><img src="{html.escape(assets['scnd_overlay'])}" alt="SCND overlay"><figcaption>SCND overlay</figcaption></figure>
    <figure><img src="{html.escape(assets['topk_grid'])}" alt="TopK grid"><figcaption>TopK 24x24 grid</figcaption></figure>
    <figure><img src="{html.escape(assets['scnd_grid'])}" alt="SCND grid"><figcaption>SCND 24x24 grid</figcaption></figure>
  </div>
  <div class="metrics">{metric_block}</div>
  <div class="checks">
    <div><b>Strengths</b><ul>{strengths_html}</ul></div>
    <div><b>Risks</b><ul>{risks_html}</ul></div>
  </div>
</article>"""


def review_dashboard_html(rows: list[dict[str, Any]], cards: list[str]) -> str:
    dataset_counts: dict[str, int] = {}
    tier_counts: dict[str, int] = {}
    for row in rows:
        dataset_counts[str(row["dataset"])] = dataset_counts.get(str(row["dataset"]), 0) + 1
        tier_counts[str(row["priority_tier"])] = tier_counts.get(str(row["priority_tier"]), 0) + 1
    dataset_options = "".join(
        f'<option value="{html.escape(dataset)}">{html.escape(dataset)} ({count})</option>'
        for dataset, count in sorted(dataset_counts.items())
    )
    tier_options = "".join(
        f'<option value="{html.escape(tier)}">Tier {html.escape(tier)} ({count})</option>'
        for tier, count in sorted(tier_counts.items())
    )
    cards_html = "\n".join(cards)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>SCND representation bottleneck review</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #f6f7f8; color: #1d2329; }}
    header.hero {{ padding: 28px 34px; background: #16202a; color: white; }}
    header.hero h1 {{ margin: 0 0 8px; font-size: 24px; letter-spacing: 0; }}
    header.hero p {{ margin: 4px 0; max-width: 1060px; line-height: 1.45; color: #d9e0e7; }}
    .toolbar {{ position: sticky; top: 0; z-index: 5; display: flex; gap: 12px; align-items: center; flex-wrap: wrap; padding: 14px 34px; background: white; border-bottom: 1px solid #d8dde3; }}
    .toolbar input, .toolbar select {{ height: 34px; border: 1px solid #b8c0c9; border-radius: 6px; padding: 0 10px; background: white; }}
    .guide {{ margin: 20px 34px; display: grid; grid-template-columns: repeat(4, minmax(180px, 1fr)); gap: 12px; }}
    .guide div {{ background: white; border: 1px solid #dce2e8; border-radius: 8px; padding: 12px; }}
    .guide b {{ display: block; margin-bottom: 6px; }}
    .cases {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(560px, 1fr)); gap: 18px; padding: 0 34px 34px; }}
    .case-card {{ background: white; border: 1px solid #dce2e8; border-radius: 8px; padding: 14px; box-shadow: 0 1px 2px rgba(0,0,0,.04); }}
    .case-card.mark-keep {{ outline: 3px solid #238636; }}
    .case-card.mark-maybe {{ outline: 3px solid #bf8700; }}
    .case-card.mark-drop {{ opacity: .56; }}
    .case-card header {{ display: flex; align-items: center; justify-content: space-between; gap: 12px; }}
    .rank {{ font-weight: 700; margin-right: 8px; }}
    .tier {{ display: inline-block; padding: 3px 8px; border-radius: 999px; color: white; font-size: 12px; font-weight: 700; }}
    .tier-A {{ background: #238636; }} .tier-B {{ background: #2f6fbb; }} .tier-C {{ background: #bf8700; }} .tier-D {{ background: #8c959f; }}
    .verdict {{ color: #4d5965; margin-left: 8px; }}
    .meta {{ display: flex; flex-direction: column; gap: 3px; margin-top: 8px; color: #57606a; font-size: 12px; }}
    .question {{ min-height: 42px; line-height: 1.35; }}
    .answers b {{ margin-right: 6px; }}
    .visuals {{ display: grid; grid-template-columns: 1fr 1fr 144px 144px; gap: 10px; align-items: start; }}
    figure {{ margin: 0; }}
    figure img {{ width: 100%; border-radius: 6px; border: 1px solid #d6dce2; background: #eee; }}
    figcaption {{ margin-top: 4px; color: #68717b; font-size: 12px; text-align: center; }}
    .metrics {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px 14px; margin-top: 12px; }}
    .metric {{ display: grid; grid-template-columns: 1fr auto; gap: 8px; align-items: center; font-size: 12px; }}
    .bar {{ grid-column: 1 / span 2; height: 6px; background: #e7ebef; border-radius: 999px; overflow: hidden; }}
    .bar i {{ display: block; height: 100%; background: #2f81f7; }}
    .checks {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 12px; font-size: 12px; }}
    .checks ul {{ margin: 6px 0 0 18px; padding: 0; }}
    .links {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-top: 12px; }}
    .mini-link, button {{ border: 1px solid #b8c0c9; border-radius: 6px; padding: 6px 9px; background: #fff; color: #1d2329; text-decoration: none; font-size: 12px; cursor: pointer; }}
    @media (max-width: 820px) {{ .guide {{ grid-template-columns: 1fr; }} .cases {{ grid-template-columns: 1fr; padding: 0 12px 20px; }} .visuals {{ grid-template-columns: 1fr 1fr; }} }}
  </style>
</head>
<body>
  <header class="hero">
    <h1>SCND representation bottleneck review</h1>
    <p>Use Tier A first for the main paper figure. A good case should show concentrated saliency top-K, visibly broader SCND coverage, low overlap, answer agreement, and a clean source image. Hidden-state heatmaps are optional until final selection.</p>
    <p>Manual rule of thumb: keep the sample only if the visual difference is obvious before reading the metrics.</p>
  </header>
  <section class="toolbar">
    <input id="search" placeholder="Search dataset, qid, question, answer" oninput="filterCards()">
    <select id="tier" onchange="filterCards()"><option value="">All tiers</option>{tier_options}</select>
    <select id="dataset" onchange="filterCards()"><option value="">All datasets</option>{dataset_options}</select>
    <select id="sort" onchange="sortCards()"><option value="score">Sort by score</option><option value="tier">Sort by tier</option></select>
    <span id="count"></span>
  </section>
  <section class="guide">
    <div><b>Priority</b>Tier A means strong spread gain, low Jaccard, same answer, and resolved image.</div>
    <div><b>Usability</b>Prefer images where TopK clustering is obvious and SCND expansion is interpretable.</div>
    <div><b>Risks</b>Flag answer disagreement, high overlap, weak spread gain, missing image, or ambiguous yes/no wording.</div>
    <div><b>Final proof</b>After selecting a case, export L2 hidden states and compare effective rank plus redundant cosine pairs.</div>
  </section>
  <main id="cases" class="cases">{cards_html}</main>
  <script>
    function filterCards() {{
      const q = document.getElementById('search').value.toLowerCase();
      const tier = document.getElementById('tier').value;
      const dataset = document.getElementById('dataset').value;
      let shown = 0;
      document.querySelectorAll('.case-card').forEach(card => {{
        const ok = (!q || card.dataset.search.includes(q)) && (!tier || card.dataset.tier === tier) && (!dataset || card.dataset.dataset === dataset);
        card.style.display = ok ? '' : 'none';
        if (ok) shown += 1;
      }});
      document.getElementById('count').textContent = shown + ' shown';
    }}
    function sortCards() {{
      const mode = document.getElementById('sort').value;
      const root = document.getElementById('cases');
      const cards = Array.from(root.children);
      cards.sort((a, b) => {{
        if (mode === 'tier') return a.dataset.tier.localeCompare(b.dataset.tier) || Number(b.dataset.score) - Number(a.dataset.score);
        return Number(b.dataset.score) - Number(a.dataset.score);
      }});
      cards.forEach(card => root.appendChild(card));
      filterCards();
    }}
    function markCard(button, label) {{
      const card = button.closest('.case-card');
      card.classList.remove('mark-keep', 'mark-maybe', 'mark-drop');
      card.classList.add('mark-' + label);
    }}
    filterCards();
  </script>
</body>
</html>
"""


def outcome_dashboard_html(rows: list[dict[str, Any]], cards: list[str]) -> str:
    html_text = review_dashboard_html(rows, cards)
    html_text = html_text.replace(
        "SCND representation bottleneck review",
        "SCND accuracy-benefit review",
    )
    html_text = html_text.replace(
        "Use Tier A first for the main paper figure. A good case should show concentrated saliency top-K, visibly broader SCND coverage, low overlap, answer agreement, and a clean source image. Hidden-state heatmaps are optional until final selection.",
        "Use Tier A first for an outcome-benefit figure. These cases are SCND correct / saliency top-K wrong, while still showing enough visual selection contrast to explain why complementary evidence matters.",
    )
    html_text = html_text.replace(
        "Manual rule of thumb: keep the sample only if the visual difference is obvious before reading the metrics.",
        "Manual rule of thumb: keep the sample only if the answer improvement and the visual token difference tell the same story.",
    )
    html_text = html_text.replace(
        ".mini-link, button {",
        ".good { color: #238636; font-weight: 700; margin-right: 8px; } .bad { color: #cf222e; font-weight: 700; margin-right: 8px; } .mini-link, button {",
    )
    return html_text


def embed_dashboard_images(html_text: str, output_dir: Path) -> str:
    def replace_src(match: re.Match[str]) -> str:
        src = match.group(1)
        if src.startswith("data:") or "://" in src:
            return match.group(0)
        asset_path = output_dir / src
        if not asset_path.is_file():
            return match.group(0)
        mime_type = mimetypes.guess_type(asset_path.name)[0] or "image/png"
        payload = base64.b64encode(asset_path.read_bytes()).decode("ascii")
        return f'<img src="data:{mime_type};base64,{payload}"'

    return re.sub(r'<img src="([^"]+)"', replace_src, html_text)


def write_review_guide(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "# SCND Representation Bottleneck Review Guide",
                "",
                "## Priority",
                "",
                "- Start with Tier A. These cases have strong SCND spread gain, low TopK/SCND overlap, answer agreement, and a resolved source image.",
                "- Tier B is useful for appendix or backup panels.",
                "- Tier C needs manual inspection before spending hidden-state export time.",
                "- Tier D is low priority unless it has a special qualitative story.",
                "",
                "## Main-Figure Usability Checklist",
                "",
                "- The TopK overlay should look visibly concentrated without reading numbers.",
                "- The SCND overlay should cover complementary regions that a reader can understand.",
                "- TopK and SCND answers should usually agree, so the figure explains representation coverage rather than an accuracy failure.",
                "- Prefer a non-cluttered image and a short, understandable question.",
                "- After choosing the case, run hidden export and require a better effective rank or fewer high-cosine redundant pairs for SCND.",
                "",
                "## Risk Flags",
                "",
                "- Answer disagreement: may be useful for failure analysis, but risky for the main mechanism figure.",
                "- High Jaccard: methods look too similar.",
                "- Weak entropy/grid-distance gain: contrast may not be visible.",
                "- MME yes/no samples: visually useful, but the wording can be less explanatory than open VQA.",
                "- Missing or padded image: avoid for the main figure.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def make_review(args: argparse.Namespace) -> None:
    output_dir = ensure_dir(args.output_dir.resolve())
    config = load_config(args.config.resolve())
    rows = read_csv_rows(args.candidate_metrics.resolve())[: args.top_n]
    review_rows: list[dict[str, Any]] = []
    cards: list[str] = []
    notes_rows: list[dict[str, Any]] = []

    for rank, row in enumerate(rows, start=1):
        image_path = resolve_image_path(config, row["dataset"], row.get("image_file"))
        image_exists = image_path is not None and image_path.is_file()
        assessment = review_assessment(row, image_exists=image_exists)
        assets = render_review_assets(row, rank, output_dir, config)
        cards.append(review_card_html(row, rank, assets, assessment, output_dir))
        review_row = {
            **row,
            "rank": rank,
            "priority_tier": assessment["priority_tier"],
            "auto_verdict": assessment["auto_verdict"],
            "strengths": "; ".join(assessment["strengths"]),
            "risks": "; ".join(assessment["risks"]),
            "image_exists": image_exists,
        }
        review_rows.append(review_row)
        notes_rows.append(
            {
                "rank": rank,
                "dataset": row["dataset"],
                "sample_idx": row["sample_idx"],
                "question_id": row["question_id"],
                "priority_tier": assessment["priority_tier"],
                "auto_verdict": assessment["auto_verdict"],
                "human_label": "",
                "selected_for": "",
                "notes": "",
            }
        )

    write_csv(output_dir / "candidate_review_table.csv", review_rows)
    write_csv(output_dir / "review_notes_template.csv", notes_rows)
    write_review_guide(output_dir / "review_guide.md")
    dashboard_html = review_dashboard_html(review_rows, cards)
    (output_dir / "review_dashboard.html").write_text(dashboard_html, encoding="utf-8")
    (output_dir / "review_dashboard_embedded.html").write_text(
        embed_dashboard_images(dashboard_html, output_dir),
        encoding="utf-8",
    )
    print(f"Wrote review dashboard to {output_dir / 'review_dashboard.html'}")
    print(f"Wrote embedded dashboard to {output_dir / 'review_dashboard_embedded.html'}")


def mine_outcome_candidates(args: argparse.Namespace) -> None:
    output_dir = ensure_dir(args.output_dir.resolve())
    config = load_config(args.config.resolve())
    rows: list[dict[str, Any]] = []
    manifest = {
        "task": "scnd_rep_bottleneck_outcome_candidate_mining",
        "commit": git_commit(),
        "layer": args.layer,
        "target_keep": args.target_keep,
        "case_type": args.case_type,
        "datasets": {},
    }

    for dataset in args.datasets:
        scorer = OutcomeScorer(dataset, correct_threshold=args.textvqa_correct_threshold)
        topk_stats_path = find_stats_path(
            args.topk_root.resolve(),
            dataset,
            f"*retain64_sss_saliency*/stats.jsonl",
        )
        scnd_stats_path = find_stats_path(
            args.scnd_root.resolve(),
            dataset,
            f"*retain64_scnd_cbs*/stats.jsonl",
        )
        topk_answers = load_answers_by_ref(topk_stats_path.parent / "answers.jsonl")
        scnd_answers = load_answers_by_ref(scnd_stats_path.parent / "answers.jsonl")
        topk_stats = {stats_ref(row): row for row in load_stats_records(topk_stats_path, args.layer, args.target_keep)}
        scnd_stats = {stats_ref(row): row for row in load_stats_records(scnd_stats_path, args.layer, args.target_keep)}
        common_refs = sorted(set(topk_answers) & set(scnd_answers) & set(topk_stats) & set(scnd_stats), key=lambda item: (item[1], item[0]))

        counts: dict[str, int] = {}
        for question_id, sample_idx in common_refs:
            ref = (question_id, sample_idx)
            topk_eval = scorer.evaluate(topk_answers[ref])
            scnd_eval = scorer.evaluate(scnd_answers[ref])
            case_type = classify_outcome(topk_eval, scnd_eval)
            counts[case_type] = counts.get(case_type, 0) + 1
            if args.case_type != "all" and case_type != args.case_type:
                continue

            base_row = build_candidate_record(
                dataset=dataset,
                question_id=question_id,
                sample_idx=sample_idx,
                topk_row=topk_stats[ref],
                scnd_row=scnd_stats[ref],
                question_row=None,
                topk_stats_path=topk_stats_path,
                scnd_stats_path=scnd_stats_path,
                layer=args.layer,
            )
            base_row.update(
                {
                    "question": scnd_eval["question"] or topk_eval["question"],
                    "image_file": scnd_eval["image_file"] or topk_eval["image_file"] or base_row.get("image_file", ""),
                    "gt_answer": scnd_eval["gt_answer"] or topk_eval["gt_answer"],
                    "category": scnd_eval["category"] or topk_eval["category"],
                    "topk_answer": str(topk_answers[ref].get("text", "")),
                    "scnd_answer": str(scnd_answers[ref].get("text", "")),
                    "topk_parsed_answer": topk_eval["parsed_answer"],
                    "scnd_parsed_answer": scnd_eval["parsed_answer"],
                    "topk_score": float(topk_eval["score"]),
                    "scnd_score": float(scnd_eval["score"]),
                    "score_gain": float(scnd_eval["score"]) - float(topk_eval["score"]),
                    "topk_correct": bool(topk_eval["correct"]),
                    "scnd_correct": bool(scnd_eval["correct"]),
                    "case_type": case_type,
                }
            )
            if args.outcome_profile == "anchored":
                jaccard = as_float(base_row, "jaccard")
                entropy_gain = as_float(base_row, "entropy_gain")
                grid_gain = as_float(base_row, "grid_distance_gain")
                if not (
                    args.min_jaccard <= jaccard <= args.max_jaccard
                    and entropy_gain >= args.min_entropy_gain
                    and grid_gain >= args.min_grid_distance_gain
                ):
                    continue
                base_row["outcome_selection_score"] = (
                    2.0 * max(0.0, float(base_row["score_gain"]))
                    + 1.0 * entropy_gain
                    + 0.9 * grid_gain
                    + 0.7 * jaccard
                    - 0.2 * abs(jaccard - 0.35)
                    + 0.2 * as_float(base_row, "scnd_mean_selected_pairwise_distance")
                )
            else:
                base_row["outcome_selection_score"] = (
                    2.0 * max(0.0, float(base_row["score_gain"]))
                    + 0.9 * as_float(base_row, "entropy_gain")
                    + 0.8 * as_float(base_row, "grid_distance_gain")
                    + 0.5 * (1.0 - as_float(base_row, "jaccard"))
                    + 0.2 * as_float(base_row, "scnd_mean_selected_pairwise_distance")
                )
            rows.append(base_row)

        manifest["datasets"][dataset] = {
            "topk_stats_path": os.fspath(topk_stats_path),
            "scnd_stats_path": os.fspath(scnd_stats_path),
            "common_refs": len(common_refs),
            "case_counts": counts,
            "outcome_profile": args.outcome_profile,
        }

    rows.sort(
        key=lambda row: (
            float(row["outcome_selection_score"]),
            float(row["score_gain"]),
            float(row["entropy_gain"]),
            float(row["grid_distance_gain"]),
            -float(row["jaccard"]),
            row["dataset"],
            int(row["sample_idx"]),
        ),
        reverse=True,
    )
    top_rows = rows[: args.top_n]

    write_csv(output_dir / "outcome_candidate_metrics.csv", top_rows)
    (output_dir / "outcome_candidate_metrics.json").write_text(
        json.dumps(top_rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "outcome_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "outcome_selected_cases.json").write_text(
        json.dumps(
            [
                {
                    "dataset": row["dataset"],
                    "question_id": row["question_id"],
                    "sample_idx": int(row["sample_idx"]),
                    "case_type": row["case_type"],
                    "outcome_selection_score": row["outcome_selection_score"],
                }
                for row in top_rows[: min(10, len(top_rows))]
            ],
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    cards: list[str] = []
    review_rows: list[dict[str, Any]] = []
    for rank, row in enumerate(top_rows, start=1):
        image_path = resolve_image_path(config, row["dataset"], row.get("image_file"))
        tier_info = outcome_tier(row, image_exists=image_path is not None and image_path.is_file())
        assets = render_review_assets(row, rank, output_dir, config)
        cards.append(outcome_card_html(row, rank, assets, tier_info))
        tier, verdict, strengths, risks = tier_info
        review_rows.append(
            {
                **row,
                "rank": rank,
                "priority_tier": tier,
                "auto_verdict": verdict,
                "strengths": "; ".join(strengths),
                "risks": "; ".join(risks),
            }
        )

    write_csv(output_dir / "outcome_review_table.csv", review_rows)
    dashboard_html = outcome_dashboard_html(review_rows, cards)
    (output_dir / "outcome_review_dashboard.html").write_text(dashboard_html, encoding="utf-8")
    (output_dir / "outcome_review_dashboard_embedded.html").write_text(
        embed_dashboard_images(dashboard_html, output_dir),
        encoding="utf-8",
    )
    print(f"Wrote {len(top_rows)} outcome candidates to {output_dir}")
    print(f"Wrote embedded outcome dashboard to {output_dir / 'outcome_review_dashboard_embedded.html'}")


def mine_candidates(args: argparse.Namespace) -> None:
    output_dir = ensure_dir(args.output_dir.resolve())
    config = load_config(args.config.resolve())
    all_rows: list[dict[str, Any]] = []
    manifest = {
        "task": "scnd_rep_bottleneck_candidate_mining",
        "commit": git_commit(),
        "layer": args.layer,
        "target_keep": args.target_keep,
        "topk_root": os.fspath(args.topk_root.resolve()),
        "scnd_root": os.fspath(args.scnd_root.resolve()),
        "datasets": {},
    }

    for dataset in args.datasets:
        topk_stats_path = find_stats_path(
            args.topk_root.resolve(),
            dataset,
            f"*retain64_sss_saliency*/stats.jsonl",
        )
        scnd_stats_path = find_stats_path(
            args.scnd_root.resolve(),
            dataset,
            f"*retain64_scnd_cbs*/stats.jsonl",
        )
        topk_records = load_stats_records(topk_stats_path, args.layer, args.target_keep)
        scnd_records = load_stats_records(scnd_stats_path, args.layer, args.target_keep)
        topk_rows = {stats_ref(row): row for row in topk_records}
        scnd_rows = {stats_ref(row): row for row in scnd_records}
        metadata_rows = load_question_rows(config, dataset)
        metadata_by_sample_idx = {idx: row for idx, row in enumerate(metadata_rows)}
        common_refs = sorted(set(topk_rows) & set(scnd_rows), key=lambda item: (item[1], item[0]))
        manifest["datasets"][dataset] = {
            "topk_stats_path": os.fspath(topk_stats_path),
            "scnd_stats_path": os.fspath(scnd_stats_path),
            "topk_visual_rows": len(topk_records),
            "scnd_visual_rows": len(scnd_records),
            "common_visual_rows": len(common_refs),
            "duplicate_question_id_safe_key": "question_id + sample_idx",
        }
        for question_id, sample_idx in common_refs:
            ref = (question_id, sample_idx)
            all_rows.append(
                build_candidate_record(
                    dataset=dataset,
                    question_id=question_id,
                    sample_idx=sample_idx,
                    topk_row=topk_rows[ref],
                    scnd_row=scnd_rows[ref],
                    question_row=metadata_by_sample_idx.get(sample_idx),
                    topk_stats_path=topk_stats_path,
                    scnd_stats_path=scnd_stats_path,
                    layer=args.layer,
                )
            )

    all_rows.sort(
        key=lambda row: (
            float(row["selection_score"]),
            float(row["entropy_gain"]),
            float(row["grid_distance_gain"]),
            -float(row["jaccard"]),
            row["dataset"],
            row["question_id"],
        ),
        reverse=True,
    )
    top_rows = all_rows[: args.top_n]
    write_csv(output_dir / "candidate_metrics.csv", top_rows)
    (output_dir / "candidate_metrics.json").write_text(
        json.dumps(top_rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    selected_suggestions = [
        {
            "dataset": row["dataset"],
            "question_id": row["question_id"],
            "sample_idx": row["sample_idx"],
            "image_file": row["image_file"],
            "selection_score": row["selection_score"],
        }
        for row in top_rows[: min(5, len(top_rows))]
    ]
    (output_dir / "selected_cases.json").write_text(
        json.dumps(selected_suggestions, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    manifest["total_common_visual_rows"] = len(all_rows)
    manifest["top_n"] = args.top_n
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    render_contact_sheet(top_rows, output_dir / "candidate_contact_sheet.png", layer=args.layer, max_rows=args.top_n)
    print(f"Wrote {len(top_rows)} candidates to {output_dir}")


def load_candidate_rows(path: Path) -> dict[tuple[str, str, int], dict[str, str]]:
    rows = read_csv_rows(path)
    return {
        (row["dataset"], normalize_question_id(row["question_id"]), int(row["sample_idx"])): row
        for row in rows
    }


def load_hidden_from_h5(
    path: Path,
    question_id: str,
    layer: int,
    sample_idx: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    with h5py.File(path, "r") as handle:
        for group_name in handle:
            group = handle[group_name]
            if str(group.attrs.get("question_id", "")) != str(question_id):
                continue
            if sample_idx is not None:
                attr_sample_idx = group.attrs.get("original_sample_idx")
                if attr_sample_idx is not None and int(attr_sample_idx) != int(sample_idx):
                    continue
            if bool(group.attrs.get("missing_visual_hidden", False)):
                raise KeyError(f"Hidden export for question_id={question_id} is marked missing")
            dataset = group[f"layer_{layer}/visual_hidden"]
            attrs = {key: group.attrs[key] for key in group.attrs}
            return np.asarray(dataset, dtype=np.float32), attrs
    raise KeyError(f"question_id={question_id} not found in {path}")


def selected_hidden_by_patch_order(
    hidden: np.ndarray,
    stats_row: dict[str, Any],
    layer: int,
) -> tuple[np.ndarray, list[int]]:
    patch_key = f"layer_{layer}_keep_patch_indices"
    current_key = f"layer_{layer}_current_patch_indices"
    patch_indices = [int(item) for item in stats_row[patch_key]]
    current_patch_indices = stats_row.get(current_key)
    if current_patch_indices:
        row_by_patch = {int(patch): idx for idx, patch in enumerate(current_patch_indices)}
    else:
        row_by_patch = {idx: idx for idx in range(hidden.shape[0])}
    ordered_patches = sorted(patch_indices)
    row_indices = [row_by_patch[patch] for patch in ordered_patches]
    return hidden[row_indices], ordered_patches


def cosine_similarity_matrix(hidden: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(hidden, axis=1, keepdims=True)
    denom = np.maximum(denom, 1e-12)
    normalized = hidden / denom
    sim = normalized @ normalized.T
    return np.clip(sim, -1.0, 1.0)


def effective_rank(hidden: np.ndarray) -> float:
    centered = hidden - hidden.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    total = float(np.sum(singular_values))
    if total <= 0:
        return 0.0
    probs = singular_values / total
    probs = probs[probs > 0]
    return float(np.exp(-np.sum(probs * np.log(probs))))


def similarity_summary(hidden: np.ndarray, sim: np.ndarray) -> dict[str, float]:
    off_diag = sim[~np.eye(sim.shape[0], dtype=bool)]
    return {
        "mean_offdiag_cosine": float(np.mean(off_diag)),
        "median_offdiag_cosine": float(np.median(off_diag)),
        "redundant_pair_ratio_cos_gt_0p90": float(np.mean(off_diag > 0.90)),
        "redundant_pair_ratio_cos_gt_0p95": float(np.mean(off_diag > 0.95)),
        "effective_rank": effective_rank(hidden),
    }


def pca_project_pair(topk_hidden: np.ndarray, scnd_hidden: np.ndarray) -> dict[str, Any]:
    """Project two selected-token sets with one deterministic PCA basis."""
    topk_hidden = np.asarray(topk_hidden, dtype=np.float64)
    scnd_hidden = np.asarray(scnd_hidden, dtype=np.float64)
    combined = np.concatenate([topk_hidden, scnd_hidden], axis=0)
    centered = combined - combined.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    if vt.shape[0] < 2:
        components = np.zeros((combined.shape[1], 2), dtype=np.float64)
        components[:, : vt.shape[0]] = vt.T
    else:
        components = vt[:2].T
    for dim in range(components.shape[1]):
        component = components[:, dim]
        anchor = int(np.argmax(np.abs(component)))
        if component[anchor] < 0:
            components[:, dim] *= -1.0
    projected = centered @ components
    total_var = float(np.sum(singular_values ** 2))
    explained = np.zeros(2, dtype=np.float64)
    if total_var > 0:
        explained[: min(2, singular_values.shape[0])] = (singular_values[:2] ** 2) / total_var
    split = topk_hidden.shape[0]
    return {
        "topk_xy": projected[:split].astype(np.float32),
        "scnd_xy": projected[split:].astype(np.float32),
        "explained_variance_ratio": explained.astype(np.float32),
    }


def pca_spread(xy: np.ndarray) -> float:
    xy = np.asarray(xy, dtype=np.float64)
    if xy.size == 0:
        return 0.0
    centered = xy - xy.mean(axis=0, keepdims=True)
    return float(np.sqrt(np.mean(np.sum(centered ** 2, axis=1))))


def pca_axis_limits(*arrays: np.ndarray, pad_ratio: float = 0.08) -> tuple[float, float, float, float]:
    combined = np.concatenate([np.asarray(array, dtype=np.float64) for array in arrays], axis=0)
    x_min, y_min = np.min(combined, axis=0)
    x_max, y_max = np.max(combined, axis=0)
    x_span = max(float(x_max - x_min), 1e-6)
    y_span = max(float(y_max - y_min), 1e-6)
    x_pad = x_span * pad_ratio
    y_pad = y_span * pad_ratio
    return float(x_min - x_pad), float(x_max + x_pad), float(y_min - y_pad), float(y_max + y_pad)


def parse_rank_list(value: str | None, default: tuple[int, ...]) -> list[int]:
    if value is None or not str(value).strip():
        return list(default)
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def candidate_rows_with_rank(path: Path) -> list[dict[str, str]]:
    rows = read_csv_rows(path)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = str(rank)
    return rows


def patch_bbox_to_square_pixels(
    bbox: tuple[int, int, int, int],
    *,
    side: int,
    patch_per_row: int = DEFAULT_PATCH_PER_ROW,
) -> tuple[float, float, float, float]:
    row_min, col_min, row_max, col_max = bbox
    cell = side / float(patch_per_row)
    return col_min * cell, row_min * cell, col_max * cell, row_max * cell


def patch_bbox_from_indices(indices: list[int], *, patch_per_row: int = DEFAULT_PATCH_PER_ROW) -> tuple[int, int, int, int]:
    coords = patch_coordinates(indices, patch_per_row=patch_per_row)
    if not coords:
        return (0, 0, patch_per_row, patch_per_row)
    rows = [row for row, _ in coords]
    cols = [col for _, col in coords]
    return min(rows), min(cols), max(rows) + 1, max(cols) + 1


def pad_patch_bbox(
    bbox: tuple[int, int, int, int],
    *,
    pad: int,
    patch_per_row: int = DEFAULT_PATCH_PER_ROW,
) -> tuple[int, int, int, int]:
    row_min, col_min, row_max, col_max = bbox
    return (
        max(0, row_min - pad),
        max(0, col_min - pad),
        min(patch_per_row, row_max + pad),
        min(patch_per_row, col_max + pad),
    )


def dashed_rectangle(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    *,
    fill: tuple[int, int, int],
    width: int = 4,
    dash: int = 18,
    gap: int = 10,
) -> None:
    x0, y0, x1, y1 = xy
    for x in range(x0, x1, dash + gap):
        draw.line((x, y0, min(x + dash, x1), y0), fill=fill, width=width)
        draw.line((x, y1, min(x + dash, x1), y1), fill=fill, width=width)
    for y in range(y0, y1, dash + gap):
        draw.line((x0, y, x0, min(y + dash, y1)), fill=fill, width=width)
        draw.line((x1, y, x1, min(y + dash, y1)), fill=fill, width=width)


def render_evidence_crop(
    image_path: Path | None,
    roi_bbox: tuple[int, int, int, int],
    *,
    patch_per_row: int = DEFAULT_PATCH_PER_ROW,
    output_size: int = 540,
    draw_roi: bool = True,
) -> Image.Image:
    if image_path is None or not image_path.is_file():
        return Image.new("RGB", (output_size, output_size), (245, 245, 245))
    image = Image.open(image_path).convert("RGB")
    side = max(image.size)
    offset = ((side - image.size[0]) // 2, (side - image.size[1]) // 2)
    square = Image.new("RGB", (side, side), (255, 255, 255))
    square.paste(image, offset)

    crop_patch_bbox = pad_patch_bbox(roi_bbox, pad=3, patch_per_row=patch_per_row)
    crop_xy = patch_bbox_to_square_pixels(crop_patch_bbox, side=side, patch_per_row=patch_per_row)
    roi_xy = patch_bbox_to_square_pixels(roi_bbox, side=side, patch_per_row=patch_per_row)
    content_xy = (offset[0], offset[1], offset[0] + image.size[0], offset[1] + image.size[1])
    crop_left = max(int(round(crop_xy[0])), content_xy[0])
    crop_top = max(int(round(crop_xy[1])), content_xy[1])
    crop_right = min(int(round(crop_xy[2])), content_xy[2])
    crop_bottom = min(int(round(crop_xy[3])), content_xy[3])
    if crop_right <= crop_left or crop_bottom <= crop_top:
        crop_left, crop_top, crop_right, crop_bottom = content_xy
    crop_width = crop_right - crop_left
    crop_height = crop_bottom - crop_top
    square_side = min(max(crop_width, crop_height), content_xy[2] - content_xy[0], content_xy[3] - content_xy[1])
    center_x = (crop_left + crop_right) / 2.0
    center_y = (crop_top + crop_bottom) / 2.0
    crop_left = int(round(center_x - square_side / 2.0))
    crop_top = int(round(center_y - square_side / 2.0))
    crop_right = crop_left + int(square_side)
    crop_bottom = crop_top + int(square_side)
    if crop_left < content_xy[0]:
        crop_right += content_xy[0] - crop_left
        crop_left = content_xy[0]
    if crop_top < content_xy[1]:
        crop_bottom += content_xy[1] - crop_top
        crop_top = content_xy[1]
    if crop_right > content_xy[2]:
        crop_left -= crop_right - content_xy[2]
        crop_right = content_xy[2]
    if crop_bottom > content_xy[3]:
        crop_top -= crop_bottom - content_xy[3]
        crop_bottom = content_xy[3]
    crop_left = max(crop_left, content_xy[0])
    crop_top = max(crop_top, content_xy[1])
    crop_right = min(crop_right, content_xy[2])
    crop_bottom = min(crop_bottom, content_xy[3])
    crop = square.crop((crop_left, crop_top, crop_right, crop_bottom)).convert("RGBA")
    if draw_roi:
        draw = ImageDraw.Draw(crop)
        rect = (
            max(0, int(round(roi_xy[0])) - crop_left),
            max(0, int(round(roi_xy[1])) - crop_top),
            min(crop.size[0] - 1, int(round(roi_xy[2])) - crop_left),
            min(crop.size[1] - 1, int(round(roi_xy[3])) - crop_top),
        )
        dashed_rectangle(draw, rect, fill=(30, 30, 30), width=max(3, output_size // 150))
    crop_rgb = crop.convert("RGB")
    crop_rgb.thumbnail((output_size, output_size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (output_size, output_size), (255, 255, 255))
    canvas.paste(crop_rgb, ((output_size - crop_rgb.size[0]) // 2, (output_size - crop_rgb.size[1]) // 2))
    return canvas


def load_figure_fonts() -> tuple[Any, Any, Any]:
    font_dir = Path("/usr/share/fonts/truetype/dejavu")
    try:
        return (
            ImageFont.truetype(os.fspath(font_dir / "DejaVuSans-Bold.ttf"), 30),
            ImageFont.truetype(os.fspath(font_dir / "DejaVuSans-Bold.ttf"), 24),
            ImageFont.truetype(os.fspath(font_dir / "DejaVuSans.ttf"), 22),
        )
    except Exception:
        return ImageFont.load_default(), ImageFont.load_default(), ImageFont.load_default()


def render_pca_panel(
    xy: np.ndarray,
    *,
    axis_limits: tuple[float, float, float, float],
    color: tuple[int, int, int],
    summary: dict[str, float],
    size: int = 540,
) -> Image.Image:
    _, font_label, font_small = load_figure_fonts()
    image = Image.new("RGBA", (size, size), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    margin = int(size * 0.11)
    x_min, x_max, y_min, y_max = axis_limits
    draw.rectangle((margin, margin, size - margin, size - margin), outline=(215, 220, 226), width=2)
    for point in np.asarray(xy, dtype=np.float64):
        x = margin + (point[0] - x_min) / max(x_max - x_min, 1e-12) * (size - 2 * margin)
        y = size - margin - (point[1] - y_min) / max(y_max - y_min, 1e-12) * (size - 2 * margin)
        radius = max(4, size // 95)
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=(*color, 120),
            outline=(*color, 230),
            width=2,
        )
    metric_text = f"mean cos {summary['mean_offdiag_cosine']:.2f}  eRank {summary['effective_rank']:.1f}"
    bbox = draw.textbbox((0, 0), metric_text, font=font_small)
    text_x = (size - (bbox[2] - bbox[0])) // 2
    draw.text((text_x, size - margin + 18), metric_text, fill=(35, 45, 55), font=font_small)
    return image.convert("RGB")


def render_rep_bottleneck_pca_figure(
    cases: list[dict[str, Any]],
    output_path: Path,
    *,
    title_labels: tuple[str, str, str] = ("Image evidence", "Saliency top-K", "SCND"),
) -> None:
    font_title, _, _ = load_figure_fonts()
    panel_size = 540
    gap = 28
    title_h = 60
    row_gap = 34
    cols = 3
    rows = len(cases)
    width = cols * panel_size + (cols - 1) * gap
    height = title_h + rows * panel_size + (rows - 1) * row_gap
    figure = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(figure)
    for col, label in enumerate(title_labels):
        bbox = draw.textbbox((0, 0), label, font=font_title)
        x = col * (panel_size + gap) + (panel_size - (bbox[2] - bbox[0])) // 2
        draw.text((x, 12), label, fill=(18, 28, 38), font=font_title)
    for row_idx, case in enumerate(cases):
        y = title_h + row_idx * (panel_size + row_gap)
        panels = [case["evidence_panel"], case["topk_pca_panel"], case["scnd_pca_panel"]]
        for col, panel in enumerate(panels):
            x = col * (panel_size + gap)
            figure.paste(panel, (x, y))
    figure.save(output_path, dpi=(300, 300))


def selected_hidden_by_saliency_order(
    hidden: np.ndarray,
    stats_row: dict[str, Any],
    layer: int,
) -> tuple[np.ndarray, list[int], list[float]]:
    patch_key = f"layer_{layer}_keep_patch_indices"
    current_key = f"layer_{layer}_current_patch_indices"
    score_key = f"layer_{layer}_current_rank_score"
    patch_indices = [int(item) for item in stats_row[patch_key]]
    current_patch_indices = [int(item) for item in stats_row.get(current_key, list(range(hidden.shape[0])))]
    current_scores = [float(item) for item in stats_row.get(score_key, [0.0] * len(current_patch_indices))]
    row_by_patch = {int(patch): idx for idx, patch in enumerate(current_patch_indices)}
    score_by_patch = {int(patch): current_scores[idx] for idx, patch in enumerate(current_patch_indices)}
    ordered_patches = sorted(patch_indices, key=lambda patch: (-score_by_patch.get(int(patch), 0.0), int(patch)))
    row_indices = [row_by_patch[int(patch)] for patch in ordered_patches]
    scores = [score_by_patch.get(int(patch), 0.0) for patch in ordered_patches]
    return hidden[row_indices], ordered_patches, scores


def offdiag_values(sim: np.ndarray) -> np.ndarray:
    return sim[~np.eye(sim.shape[0], dtype=bool)]


def render_pairwise_heatmap_panel(
    sim: np.ndarray,
    *,
    size: int = 360,
    vmin: float = 0.0,
    vmax: float = 1.0,
    footer_text: str | None = None,
    footer_color: tuple[int, int, int] = (35, 45, 55),
) -> Image.Image:
    _, _, font_small = load_figure_fonts()
    footer_h = 44 if footer_text else 0
    matrix_size = size - footer_h
    values = np.asarray(sim, dtype=np.float64)
    normalized = np.clip((values - vmin) / max(vmax - vmin, 1e-12), 0.0, 1.0)
    low = np.asarray(HEATMAP_LOW_COLOR, dtype=np.float64)
    high = np.asarray(HEATMAP_HIGH_COLOR, dtype=np.float64)
    rgb = (low[None, None, :] * (1.0 - normalized[..., None]) + high[None, None, :] * normalized[..., None])
    rgb = rgb.astype(np.uint8)
    diag = np.eye(values.shape[0], dtype=bool)
    rgb[diag] = np.asarray((226, 229, 234), dtype=np.uint8)
    image = Image.fromarray(rgb, mode="RGB").resize((matrix_size, matrix_size), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (size, size), "white")
    x0 = (size - matrix_size) // 2
    canvas.paste(image, (x0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((x0, 0, x0 + matrix_size - 1, matrix_size - 1), outline=(214, 220, 228), width=2)
    if footer_text:
        bbox = draw.textbbox((0, 0), footer_text, font=font_small)
        draw.text(((size - (bbox[2] - bbox[0])) // 2, matrix_size + 10), footer_text, fill=footer_color, font=font_small)
    return canvas


def render_similarity_colorbar(
    *,
    height: int,
    width: int = 58,
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> Image.Image:
    _, _, font_small = load_figure_fonts()
    image = Image.new("RGB", (width, height), "white")
    bar_w = 16
    bar_x = 10
    top = 12
    bottom = height - 38
    draw = ImageDraw.Draw(image)
    for y in range(top, bottom):
        value = 1.0 - (y - top) / max(bottom - top - 1, 1)
        low = np.asarray(HEATMAP_LOW_COLOR, dtype=np.float64)
        high = np.asarray(HEATMAP_HIGH_COLOR, dtype=np.float64)
        color = tuple((low * (1.0 - value) + high * value).astype(np.uint8).tolist())
        draw.line((bar_x, y, bar_x + bar_w, y), fill=color)
    draw.rectangle((bar_x, top, bar_x + bar_w, bottom), outline=(190, 196, 205), width=1)
    ticks = [(vmax, top), ((vmin + vmax) / 2.0, (top + bottom) // 2), (vmin, bottom)]
    for value, y in ticks:
        draw.line((bar_x + bar_w + 2, y, bar_x + bar_w + 8, y), fill=(80, 90, 100), width=1)
        draw.text((bar_x + bar_w + 10, y - 10), f"{value:.1f}", fill=(35, 45, 55), font=font_small)
    draw.text((0, bottom + 10), "cos", fill=(35, 45, 55), font=font_small)
    return image


def render_pairwise_distribution_panel(
    topk_values: np.ndarray,
    scnd_values: np.ndarray,
    *,
    size: tuple[int, int] = (430, 360),
    x_min: float = 0.0,
    x_max: float = 1.0,
    y_max: float | None = None,
    bins: int = 24,
    show_legend: bool = False,
) -> Image.Image:
    _, _, font_small = load_figure_fonts()
    width, height = size
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = 50, 26, width - 22, height - 54
    bin_edges = np.linspace(x_min, x_max, bins + 1)
    topk_hist, _ = np.histogram(np.clip(topk_values, x_min, x_max), bins=bin_edges, density=True)
    scnd_hist, _ = np.histogram(np.clip(scnd_values, x_min, x_max), bins=bin_edges, density=True)
    max_density = y_max if y_max is not None else max(float(topk_hist.max()), float(scnd_hist.max()), 1e-6)
    draw.rectangle((left, top, right, bottom), outline=(210, 216, 224), width=2)

    def to_xy(x_value: float, density: float) -> tuple[float, float]:
        x = left + (x_value - x_min) / max(x_max - x_min, 1e-12) * (right - left)
        y = bottom - density / max(max_density, 1e-12) * (bottom - top)
        return x, y

    def draw_step(hist: np.ndarray, color: tuple[int, int, int]) -> None:
        points: list[tuple[float, float]] = []
        points.append(to_xy(float(bin_edges[0]), 0.0))
        for idx, density in enumerate(hist):
            x0 = float(bin_edges[idx])
            x1 = float(bin_edges[idx + 1])
            points.append(to_xy(x0, float(density)))
            points.append(to_xy(x1, float(density)))
        points.append(to_xy(float(bin_edges[-1]), 0.0))
        draw.line(points, fill=color, width=4, joint="curve")

    draw_step(topk_hist, TOPK_METHOD_COLOR)
    draw_step(scnd_hist, SCND_METHOD_COLOR)
    topk_mean = float(np.mean(topk_values))
    scnd_mean = float(np.mean(scnd_values))
    for mean, color, text_y, label in (
        (topk_mean, TOPK_METHOD_COLOR, top + 10, f"Top-K {topk_mean:.2f}"),
        (scnd_mean, SCND_METHOD_COLOR, top + 35, f"SCND {scnd_mean:.2f}"),
    ):
        x, _ = to_xy(mean, 0.0)
        draw.line((x, top, x, bottom), fill=color, width=3)
        label_w = draw.textbbox((0, 0), label, font=font_small)[2]
        label_x = min(max(int(x) + 6, left), right - label_w - 4)
        draw.text((label_x, text_y), label, fill=color, font=font_small)
    for tick in (0.0, 0.5, 1.0):
        x, _ = to_xy(tick, 0.0)
        draw.line((x, bottom, x, bottom + 7), fill=(80, 90, 100), width=1)
        draw.text((x - 9, bottom + 12), f"{tick:g}", fill=(35, 45, 55), font=font_small)
    draw.text((left, bottom + 32), "pairwise cosine", fill=(35, 45, 55), font=font_small)
    if show_legend:
        legend_x = right - 130
        draw.line((legend_x, top + 12, legend_x + 28, top + 12), fill=TOPK_METHOD_COLOR, width=4)
        draw.text((legend_x + 36, top + 2), "Top-K", fill=(35, 45, 55), font=font_small)
        draw.line((legend_x, top + 40, legend_x + 28, top + 40), fill=SCND_METHOD_COLOR, width=4)
        draw.text((legend_x + 36, top + 30), "SCND", fill=(35, 45, 55), font=font_small)
    return image


def render_patch_evidence_overlay_crop(
    image_path: Path | None,
    roi_bbox: tuple[int, int, int, int],
    patch_indices: list[int],
    *,
    color: tuple[int, int, int],
    patch_per_row: int = DEFAULT_PATCH_PER_ROW,
    output_size: int = 172,
    draw_roi: bool = True,
    gaussian_sigma_patches: float = 0.85,
    max_alpha: int = 150,
) -> Image.Image:
    if image_path is None or not image_path.is_file():
        return Image.new("RGB", (output_size, output_size), (245, 245, 245))
    image = Image.open(image_path).convert("RGB")
    side = max(image.size)
    offset = ((side - image.size[0]) // 2, (side - image.size[1]) // 2)
    square = Image.new("RGB", (side, side), (255, 255, 255))
    square.paste(image, offset)
    content_xy = (offset[0], offset[1], offset[0] + image.size[0], offset[1] + image.size[1])
    crop_patch_bbox = pad_patch_bbox(roi_bbox, pad=3, patch_per_row=patch_per_row)
    crop_xy = patch_bbox_to_square_pixels(crop_patch_bbox, side=side, patch_per_row=patch_per_row)
    crop_left = max(int(round(crop_xy[0])), content_xy[0])
    crop_top = max(int(round(crop_xy[1])), content_xy[1])
    crop_right = min(int(round(crop_xy[2])), content_xy[2])
    crop_bottom = min(int(round(crop_xy[3])), content_xy[3])
    crop_width = crop_right - crop_left
    crop_height = crop_bottom - crop_top
    square_side = min(max(crop_width, crop_height), content_xy[2] - content_xy[0], content_xy[3] - content_xy[1])
    center_x = (crop_left + crop_right) / 2.0
    center_y = (crop_top + crop_bottom) / 2.0
    crop_left = int(round(center_x - square_side / 2.0))
    crop_top = int(round(center_y - square_side / 2.0))
    crop_right = crop_left + int(square_side)
    crop_bottom = crop_top + int(square_side)
    if crop_left < content_xy[0]:
        crop_right += content_xy[0] - crop_left
        crop_left = content_xy[0]
    if crop_top < content_xy[1]:
        crop_bottom += content_xy[1] - crop_top
        crop_top = content_xy[1]
    if crop_right > content_xy[2]:
        crop_left -= crop_right - content_xy[2]
        crop_right = content_xy[2]
    if crop_bottom > content_xy[3]:
        crop_top -= crop_bottom - content_xy[3]
        crop_bottom = content_xy[3]
    crop_left = max(crop_left, content_xy[0])
    crop_top = max(crop_top, content_xy[1])
    crop_right = min(crop_right, content_xy[2])
    crop_bottom = min(crop_bottom, content_xy[3])
    crop = square.crop((crop_left, crop_top, crop_right, crop_bottom)).convert("RGBA")
    overlay = Image.new("RGBA", crop.size, (0, 0, 0, 0))
    cell = side / float(patch_per_row)
    patches = [int(patch) for patch in patch_indices]
    if patches:
        yy, xx = np.mgrid[0 : crop.size[1], 0 : crop.size[0]]
        density = np.zeros((crop.size[1], crop.size[0]), dtype=np.float32)
        sigma = max(float(gaussian_sigma_patches) * cell, 1.0)
        radius = 3.0 * sigma
        for patch in patches:
            row = patch // patch_per_row
            col = patch % patch_per_row
            center_x = (col + 0.5) * cell - crop_left
            center_y = (row + 0.5) * cell - crop_top
            if center_x < -radius or center_y < -radius or center_x > crop.size[0] + radius or center_y > crop.size[1] + radius:
                continue
            density += np.exp(-((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2.0 * sigma * sigma))
        max_density = float(np.max(density))
        if max_density > 0:
            density = density / max_density
            color_arr = np.zeros((crop.size[1], crop.size[0], 4), dtype=np.uint8)
            color_arr[..., 0] = int(color[0])
            color_arr[..., 1] = int(color[1])
            color_arr[..., 2] = int(color[2])
            color_arr[..., 3] = np.clip(density ** 0.72 * max_alpha, 0, 255).astype(np.uint8)
            overlay = Image.fromarray(color_arr, mode="RGBA")
    if draw_roi:
        draw = ImageDraw.Draw(overlay)
        roi_xy = patch_bbox_to_square_pixels(roi_bbox, side=side, patch_per_row=patch_per_row)
        rect = (
            max(0, int(round(roi_xy[0])) - crop_left),
            max(0, int(round(roi_xy[1])) - crop_top),
            min(crop.size[0] - 1, int(round(roi_xy[2])) - crop_left),
            min(crop.size[1] - 1, int(round(roi_xy[3])) - crop_top),
        )
        dashed_rectangle(draw, rect, fill=(30, 30, 30), width=max(2, output_size // 120), dash=10, gap=7)
    merged = Image.alpha_composite(crop, overlay).convert("RGB")
    merged.thumbnail((output_size, output_size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (output_size, output_size), "white")
    canvas.paste(merged, ((output_size - merged.size[0]) // 2, (output_size - merged.size[1]) // 2))
    return canvas


def render_retained_patch_panel(
    image_path: Path | None,
    roi_bbox: tuple[int, int, int, int],
    topk_patches: list[int],
    scnd_patches: list[int],
    *,
    size: tuple[int, int] = (260, 360),
) -> Image.Image:
    _, _, font_small = load_figure_fonts()
    width, height = size
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    thumb = min(width - 62, (height - 54) // 2)
    topk = render_patch_evidence_overlay_crop(
        image_path,
        roi_bbox,
        topk_patches,
        color=RED_HEATMAP_COLOR,
        output_size=thumb,
        draw_roi=False,
    )
    scnd = render_patch_evidence_overlay_crop(
        image_path,
        roi_bbox,
        scnd_patches,
        color=RED_HEATMAP_COLOR,
        output_size=thumb,
        draw_roi=False,
    )
    label_w = 72
    y0 = 14
    y1 = y0 + thumb + 24
    draw.text((8, y0 + thumb // 2 - 10), "Top-K", fill=TOPK_METHOD_COLOR, font=font_small)
    draw.text((8, y1 + thumb // 2 - 10), "SCND", fill=SCND_METHOD_COLOR, font=font_small)
    image.paste(topk, (label_w, y0))
    image.paste(scnd, (label_w, y1))
    return image


def histogram_density_max(
    value_pairs: list[tuple[np.ndarray, np.ndarray]],
    *,
    x_min: float,
    x_max: float,
    bins: int,
) -> float:
    bin_edges = np.linspace(x_min, x_max, bins + 1)
    max_density = 0.0
    for topk_values, scnd_values in value_pairs:
        topk_hist, _ = np.histogram(np.clip(topk_values, x_min, x_max), bins=bin_edges, density=True)
        scnd_hist, _ = np.histogram(np.clip(scnd_values, x_min, x_max), bins=bin_edges, density=True)
        max_density = max(max_density, float(np.nanmax(topk_hist)), float(np.nanmax(scnd_hist)))
    return max(max_density, 1e-6)


def render_pairwise_main_figure(
    cases: list[dict[str, Any]],
    output_path: Path,
    *,
    vmin: float,
    vmax: float,
) -> None:
    font_title, font_label, font_small = load_figure_fonts()
    panel_h = 360
    evidence_w = 360
    retained_w = 270
    heatmap_w = 360
    colorbar_w = 68
    gap = 24
    title_h = 72
    row_gap = 34
    columns = [
        ("Image evidence", evidence_w),
        ("Retained patches", retained_w),
        ("Top-K pairwise cos", heatmap_w),
        ("SCND pairwise cos", heatmap_w),
        ("", colorbar_w),
    ]
    x_offsets: list[int] = []
    cursor = 0
    for _, width in columns:
        x_offsets.append(cursor)
        cursor += width + gap
    total_w = cursor - gap
    total_h = title_h + len(cases) * panel_h + (len(cases) - 1) * row_gap
    figure = Image.new("RGB", (total_w, total_h), "white")
    draw = ImageDraw.Draw(figure)

    for (label, width), x in zip(columns, x_offsets):
        if not label:
            continue
        bbox = draw.textbbox((0, 0), label, font=font_title)
        draw.text((x + (width - (bbox[2] - bbox[0])) // 2, 16), label, fill=(18, 28, 38), font=font_title)

    colorbar_h = panel_h - 44
    colorbar = render_similarity_colorbar(height=colorbar_h, width=colorbar_w, vmin=vmin, vmax=vmax)
    colorbar_y = title_h + max(0, (total_h - title_h - colorbar_h) // 2)
    figure.paste(colorbar, (x_offsets[4], colorbar_y))

    for row_idx, case in enumerate(cases):
        y = title_h + row_idx * (panel_h + row_gap)
        panels = [
            case["evidence_panel"],
            case["retained_patch_panel"],
            case["topk_heatmap_panel"],
            case["scnd_heatmap_panel"],
            None,
        ]
        for panel, x in zip(panels, x_offsets):
            if panel is None:
                continue
            figure.paste(panel, (x, y))
        case_label = f"#{case['rank']}  {case['question']}"
        bbox = draw.textbbox((0, 0), case_label, font=font_small)
        label_bg = (255, 255, 255)
        draw.rectangle((x_offsets[0], y - 4, x_offsets[0] + min(bbox[2] - bbox[0] + 18, evidence_w), y + 24), fill=label_bg)
        draw.text((x_offsets[0] + 6, y - 2), case_label[:52], fill=(55, 65, 75), font=font_small)

    figure.save(output_path, dpi=(300, 300))


def build_pairwise_main_figure(args: argparse.Namespace) -> None:
    output_dir = ensure_dir(args.output_dir.resolve())
    panel_dir = ensure_dir(output_dir / "pairwise_panels")
    matrix_dir = ensure_dir(output_dir / "pairwise_matrices")
    config = load_config(args.config.resolve())
    all_rows = candidate_rows_with_rank(args.candidate_metrics.resolve())
    rows_by_rank = {int(row["rank"]): row for row in all_rows}
    selected_ranks = parse_rank_list(args.selected_ranks, DEFAULT_PAIRWISE_MAIN_RANKS)
    stats_cache: dict[tuple[str, int], dict[tuple[str, int], dict[str, Any]]] = {}

    def stats_lookup(path_text: str, layer: int, target_keep: int) -> dict[tuple[str, int], dict[str, Any]]:
        key = (path_text, layer)
        if key not in stats_cache:
            stats_cache[key] = {
                stats_ref(row): row
                for row in load_stats_records(Path(path_text), layer, target_keep)
            }
        return stats_cache[key]

    rendered_cases: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    value_rows: list[dict[str, Any]] = []

    for rank in selected_ranks:
        if rank not in rows_by_rank:
            raise KeyError(f"rank={rank} is not present in {args.candidate_metrics}")
        row = rows_by_rank[rank]
        ref = (normalize_question_id(row["question_id"]), int(row["sample_idx"]))
        topk_stats = stats_lookup(row["topk_stats_path"], args.layer, args.target_keep)[ref]
        scnd_stats = stats_lookup(row["scnd_stats_path"], args.layer, args.target_keep)[ref]
        hidden, hidden_attrs = load_hidden_from_h5(args.hidden_h5.resolve(), ref[0], args.layer, sample_idx=ref[1])
        topk_hidden, topk_patches, topk_scores = selected_hidden_by_saliency_order(hidden, topk_stats, args.layer)
        scnd_hidden, scnd_patches, scnd_scores = selected_hidden_by_saliency_order(hidden, scnd_stats, args.layer)
        if topk_hidden.shape[0] != args.target_keep or scnd_hidden.shape[0] != args.target_keep:
            raise ValueError(
                f"rank={rank} keep mismatch: topk={topk_hidden.shape[0]}, "
                f"scnd={scnd_hidden.shape[0]}, expected={args.target_keep}"
            )
        topk_sim = cosine_similarity_matrix(topk_hidden)
        scnd_sim = cosine_similarity_matrix(scnd_hidden)
        topk_values = offdiag_values(topk_sim)
        scnd_values = offdiag_values(scnd_sim)
        topk_summary = similarity_summary(topk_hidden, topk_sim)
        scnd_summary = similarity_summary(scnd_hidden, scnd_sim)
        image_path = resolve_image_path(config, row["dataset"], row.get("image_file"))
        roi_bbox = REP_FIG_ROI_PATCH_BBOX_BY_RANK.get(
            rank,
            patch_bbox_from_indices(scnd_patches, patch_per_row=DEFAULT_PATCH_PER_ROW),
        )
        stem = f"rank{rank:02d}_{row['dataset']}_s{int(row['sample_idx']):05d}_{sanitize_filename(row['question_id'])}"
        evidence_panel = render_evidence_crop(image_path, roi_bbox, output_size=360, draw_roi=False)
        retained_patch_panel = render_retained_patch_panel(
            image_path,
            roi_bbox,
            topk_patches,
            scnd_patches,
            size=(270, 360),
        )
        topk_heatmap_panel = render_pairwise_heatmap_panel(
            topk_sim,
            size=360,
            vmin=args.vmin,
            vmax=args.vmax,
            footer_text=f"mean cos {topk_summary['mean_offdiag_cosine']:.2f}",
            footer_color=TOPK_METHOD_COLOR,
        )
        scnd_heatmap_panel = render_pairwise_heatmap_panel(
            scnd_sim,
            size=360,
            vmin=args.vmin,
            vmax=args.vmax,
            footer_text=f"mean cos {scnd_summary['mean_offdiag_cosine']:.2f}",
            footer_color=SCND_METHOD_COLOR,
        )
        evidence_panel.save(panel_dir / f"{stem}_image_evidence.png", dpi=(300, 300))
        retained_patch_panel.save(panel_dir / f"{stem}_retained_patches.png", dpi=(300, 300))
        topk_heatmap_panel.save(panel_dir / f"{stem}_topk_pairwise_cosine.png", dpi=(300, 300))
        scnd_heatmap_panel.save(panel_dir / f"{stem}_scnd_pairwise_cosine.png", dpi=(300, 300))
        np.save(matrix_dir / f"{stem}_topk_pairwise_cosine.npy", topk_sim)
        np.save(matrix_dir / f"{stem}_scnd_pairwise_cosine.npy", scnd_sim)
        save_similarity_csv(topk_sim, matrix_dir / f"{stem}_topk_pairwise_cosine.csv")
        save_similarity_csv(scnd_sim, matrix_dir / f"{stem}_scnd_pairwise_cosine.csv")

        rendered_cases.append(
            {
                "rank": rank,
                "row": row,
                "question": str(row.get("question", "")),
                "evidence_panel": evidence_panel,
                "retained_patch_panel": retained_patch_panel,
                "topk_heatmap_panel": topk_heatmap_panel,
                "scnd_heatmap_panel": scnd_heatmap_panel,
                "topk_values": topk_values,
                "scnd_values": scnd_values,
                "topk_summary": topk_summary,
                "scnd_summary": scnd_summary,
            }
        )
        metric_rows.append(
            {
                "rank": rank,
                "dataset": row["dataset"],
                "sample_idx": int(row["sample_idx"]),
                "question_id": row["question_id"],
                "question": row.get("question", ""),
                "topk_answer": row.get("topk_answer", ""),
                "scnd_answer": row.get("scnd_answer", ""),
                "gt_answer": row.get("gt_answer", ""),
                "hidden_shape": list(hidden.shape),
                "hidden_attrs": {key: str(value) for key, value in hidden_attrs.items()},
                "topk_keep_count": len(topk_patches),
                "scnd_keep_count": len(scnd_patches),
                "topk_mean_saliency_score": float(np.mean(topk_scores)),
                "scnd_mean_saliency_score": float(np.mean(scnd_scores)),
                "topk_mean_offdiag_cosine": topk_summary["mean_offdiag_cosine"],
                "scnd_mean_offdiag_cosine": scnd_summary["mean_offdiag_cosine"],
                "homogeneity_gain_topk_minus_scnd": (
                    topk_summary["mean_offdiag_cosine"] - scnd_summary["mean_offdiag_cosine"]
                ),
                "topk_median_offdiag_cosine": topk_summary["median_offdiag_cosine"],
                "scnd_median_offdiag_cosine": scnd_summary["median_offdiag_cosine"],
                "topk_redundant_pair_ratio_cos_gt_0p90": topk_summary["redundant_pair_ratio_cos_gt_0p90"],
                "scnd_redundant_pair_ratio_cos_gt_0p90": scnd_summary["redundant_pair_ratio_cos_gt_0p90"],
                "topk_redundant_pair_ratio_cos_gt_0p95": topk_summary["redundant_pair_ratio_cos_gt_0p95"],
                "scnd_redundant_pair_ratio_cos_gt_0p95": scnd_summary["redundant_pair_ratio_cos_gt_0p95"],
                "topk_effective_rank": topk_summary["effective_rank"],
                "scnd_effective_rank": scnd_summary["effective_rank"],
                "effective_rank_gain_scnd_minus_topk": (
                    scnd_summary["effective_rank"] - topk_summary["effective_rank"]
                ),
            }
        )
        for method, values in (("topk", topk_values), ("scnd", scnd_values)):
            for pair_idx, value in enumerate(values):
                value_rows.append(
                    {
                        "rank": rank,
                        "dataset": row["dataset"],
                        "sample_idx": int(row["sample_idx"]),
                        "question_id": row["question_id"],
                        "method": method,
                        "pair_idx": pair_idx,
                        "pairwise_cosine": float(value),
                    }
                )

    png_path = output_dir / "scnd_rep_bottleneck_pairwise_main.png"
    pdf_path = output_dir / "scnd_rep_bottleneck_pairwise_main.pdf"
    render_pairwise_main_figure(
        rendered_cases,
        png_path,
        vmin=args.vmin,
        vmax=args.vmax,
    )
    render_pairwise_main_figure(
        rendered_cases,
        pdf_path,
        vmin=args.vmin,
        vmax=args.vmax,
    )
    write_csv(output_dir / "pairwise_metric_summary.csv", metric_rows)
    write_csv(output_dir / "pairwise_cosine_values.csv", value_rows)
    caption = (
        "Representation bottleneck under saliency top-K. Each row is a TextVQA case at retain64. "
        "The first two columns show the answer region and retained-patch Gaussian heatmaps for "
        "saliency top-K versus SCND. Pairwise-cosine heatmaps compare the first-pruning-layer "
        "retained visual hidden states, ordered by pre-prune saliency rank; the diagonal is masked and "
        "all heatmaps use a shared red scale. The mean off-diagonal pairwise cosine is shown "
        "below each heatmap; lower values indicate less redundant retained representations.\n"
    )
    (output_dir / "scnd_rep_bottleneck_pairwise_caption.md").write_text(caption, encoding="utf-8")
    (output_dir / "pairwise_figure_manifest.json").write_text(
        json.dumps(
            {
                "task": "scnd_rep_bottleneck_pairwise_main_figure",
                "commit": git_commit(),
                "layer": args.layer,
                "target_keep": args.target_keep,
                "hidden_h5": os.fspath(args.hidden_h5.resolve()),
                "candidate_metrics": os.fspath(args.candidate_metrics.resolve()),
                "selected_ranks": selected_ranks,
                "heatmap_vmin": args.vmin,
                "heatmap_vmax": args.vmax,
                "heatmap_colormap": {
                    "low": HEATMAP_LOW_COLOR,
                    "high": HEATMAP_HIGH_COLOR,
                    "diagonal": (226, 229, 234),
                },
                "method_colors": {
                    "topk": TOPK_METHOD_COLOR,
                    "scnd": SCND_METHOD_COLOR,
                },
                "retained_patch_heatmap_color": RED_HEATMAP_COLOR,
                "heatmap_order": "selected tokens sorted by pre-prune saliency rank descending, then patch index ascending",
                "heatmap_footer": {
                    "metric": "mean off-diagonal pairwise cosine",
                    "lower_is_less_redundant": True,
                },
                "outputs": {
                    "figure_png": os.fspath(png_path),
                    "figure_pdf": os.fspath(pdf_path),
                    "metric_summary_csv": os.fspath(output_dir / "pairwise_metric_summary.csv"),
                    "cosine_values_csv": os.fspath(output_dir / "pairwise_cosine_values.csv"),
                },
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print("Built pairwise bottleneck figure for ranks: " + ", ".join(map(str, selected_ranks)))
    print(f"Wrote pairwise main figure to {png_path}")


def save_similarity_heatmap(sim: np.ndarray, path: Path, scale: int = 7) -> None:
    values = np.clip((sim + 1.0) / 2.0, 0.0, 1.0)
    red = np.full_like(values, 220.0)
    green_blue = 255.0 * (1.0 - values)
    rgb = np.stack([red, green_blue, green_blue], axis=-1).astype(np.uint8)
    image = Image.fromarray(rgb, mode="RGB").resize((sim.shape[1] * scale, sim.shape[0] * scale), Image.NEAREST)
    image.save(path)


def save_similarity_csv(sim: np.ndarray, path: Path) -> None:
    np.savetxt(path, sim, delimiter=",", fmt="%.6f")


def write_case_details(
    path: Path,
    row: dict[str, str],
    topk_summary: dict[str, float],
    scnd_summary: dict[str, float],
    topk_overlay: Path,
    scnd_overlay: Path,
    topk_heatmap: Path,
    scnd_heatmap: Path,
) -> None:
    lines = [
        f"# SCND Representation Bottleneck Case: {row['dataset']} / {row['question_id']}",
        "",
        f"- sample_idx: {row.get('sample_idx', '')}",
        f"- image: `{row.get('image_file', '')}`",
        f"- question: {row.get('question', '')}",
        f"- saliency top-K answer: {row.get('topk_answer', '')}",
        f"- SCND answer: {row.get('scnd_answer', '')}",
        f"- overlap/Jaccard: {row.get('overlap_count', '')} / {row.get('jaccard', '')}",
        f"- spatial entropy gain: {row.get('entropy_gain', '')}",
        f"- mean grid-distance gain: {row.get('grid_distance_gain', '')}",
        "",
        "## Hidden-State Similarity",
        "",
        "| method | mean offdiag cos | median cos | cos>0.90 | cos>0.95 | effective rank |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| saliency top-K | {topk_summary['mean_offdiag_cosine']:.4f} | "
            f"{topk_summary['median_offdiag_cosine']:.4f} | "
            f"{topk_summary['redundant_pair_ratio_cos_gt_0p90']:.4f} | "
            f"{topk_summary['redundant_pair_ratio_cos_gt_0p95']:.4f} | "
            f"{topk_summary['effective_rank']:.2f} |"
        ),
        (
            f"| SCND | {scnd_summary['mean_offdiag_cosine']:.4f} | "
            f"{scnd_summary['median_offdiag_cosine']:.4f} | "
            f"{scnd_summary['redundant_pair_ratio_cos_gt_0p90']:.4f} | "
            f"{scnd_summary['redundant_pair_ratio_cos_gt_0p95']:.4f} | "
            f"{scnd_summary['effective_rank']:.2f} |"
        ),
        "",
        "## Panels",
        "",
        f"- saliency top-K overlay: `{topk_overlay.name}`",
        f"- SCND overlay: `{scnd_overlay.name}`",
        f"- saliency top-K heatmap: `{topk_heatmap.name}`",
        f"- SCND heatmap: `{scnd_heatmap.name}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_figures(args: argparse.Namespace) -> None:
    output_dir = ensure_dir(args.output_dir.resolve())
    panel_dir = ensure_dir(output_dir / "fig_panels")
    matrix_dir = ensure_dir(output_dir / "similarity_matrices")
    details_dir = ensure_dir(output_dir / "case_details")
    config = load_config(args.config.resolve())
    candidates = load_candidate_rows(args.candidate_metrics.resolve())

    if args.case:
        selected = []
        for item in args.case:
            parts = item.split(":", 2)
            if len(parts) == 3:
                dataset, sample_idx_text, question_id = parts
                selected.append((dataset, normalize_question_id(question_id), int(sample_idx_text)))
            elif len(parts) == 2:
                dataset, question_id = parts
                matches = [
                    key for key in candidates
                    if key[0] == dataset and key[1] == normalize_question_id(question_id)
                ]
                if len(matches) != 1:
                    raise KeyError(
                        f"Case {item!r} matched {len(matches)} candidates; use "
                        "dataset:sample_idx:question_id to disambiguate."
                    )
                selected.append(matches[0])
            else:
                raise ValueError(f"Invalid --case value {item!r}; expected dataset:question_id or dataset:sample_idx:question_id")
    else:
        selected_payload = json.loads(args.selected_cases.read_text(encoding="utf-8"))
        selected = [
            (item["dataset"], normalize_question_id(item["question_id"]), int(item["sample_idx"]))
            for item in selected_payload
        ]

    report_rows: list[dict[str, Any]] = []
    for dataset, question_id, sample_idx in selected:
        key = (dataset, question_id, int(sample_idx))
        if key not in candidates:
            raise KeyError(f"Selected case {dataset}:{sample_idx}:{question_id} is not present in candidate metrics")
        row = candidates[key]
        topk_stats = load_stats_by_ref(Path(row["topk_stats_path"]), args.layer, args.target_keep, question_id, sample_idx)
        scnd_stats = load_stats_by_ref(Path(row["scnd_stats_path"]), args.layer, args.target_keep, question_id, sample_idx)
        hidden, hidden_attrs = load_hidden_from_h5(args.hidden_h5.resolve(), question_id, args.layer, sample_idx=sample_idx)
        topk_hidden, topk_patches = selected_hidden_by_patch_order(hidden, topk_stats, args.layer)
        scnd_hidden, scnd_patches = selected_hidden_by_patch_order(hidden, scnd_stats, args.layer)
        topk_sim = cosine_similarity_matrix(topk_hidden)
        scnd_sim = cosine_similarity_matrix(scnd_hidden)
        topk_summary = similarity_summary(topk_hidden, topk_sim)
        scnd_summary = similarity_summary(scnd_hidden, scnd_sim)

        stem = f"{dataset}_s{int(sample_idx):05d}_{sanitize_filename(question_id)}"
        image_path = resolve_image_path(config, dataset, row.get("image_file"))
        topk_overlay = panel_dir / f"{stem}_topk_overlay.png"
        scnd_overlay = panel_dir / f"{stem}_scnd_overlay.png"
        topk_heatmap = panel_dir / f"{stem}_topk_similarity.png"
        scnd_heatmap = panel_dir / f"{stem}_scnd_similarity.png"
        render_patch_overlay(image_path, topk_patches, color=(220, 40, 40)).save(topk_overlay)
        render_patch_overlay(image_path, scnd_patches, color=(40, 110, 220)).save(scnd_overlay)
        save_similarity_heatmap(topk_sim, topk_heatmap)
        save_similarity_heatmap(scnd_sim, scnd_heatmap)

        np.save(matrix_dir / f"{stem}_topk_similarity.npy", topk_sim)
        np.save(matrix_dir / f"{stem}_scnd_similarity.npy", scnd_sim)
        save_similarity_csv(topk_sim, matrix_dir / f"{stem}_topk_similarity.csv")
        save_similarity_csv(scnd_sim, matrix_dir / f"{stem}_scnd_similarity.csv")
        write_case_details(
            details_dir / f"{stem}_details.md",
            row,
            topk_summary,
            scnd_summary,
            topk_overlay,
            scnd_overlay,
            topk_heatmap,
            scnd_heatmap,
        )
        report_rows.append(
            {
                "dataset": dataset,
                "question_id": question_id,
                "sample_idx": int(sample_idx),
                "hidden_h5": os.fspath(args.hidden_h5.resolve()),
                "hidden_shape": list(hidden.shape),
                "hidden_attrs": {key: str(value) for key, value in hidden_attrs.items()},
                **prefix_metrics("topk_", topk_summary),
                **prefix_metrics("scnd_", scnd_summary),
                "mean_offdiag_cosine_delta_scnd_minus_topk": (
                    scnd_summary["mean_offdiag_cosine"] - topk_summary["mean_offdiag_cosine"]
                ),
                "effective_rank_delta_scnd_minus_topk": (
                    scnd_summary["effective_rank"] - topk_summary["effective_rank"]
                ),
            }
        )

    write_csv(output_dir / "case_hidden_similarity_summary.csv", report_rows)
    (output_dir / "figure_manifest.json").write_text(
        json.dumps(
            {
                "task": "scnd_rep_bottleneck_figure_build",
                "commit": git_commit(),
                "layer": args.layer,
                "target_keep": args.target_keep,
                "hidden_h5": os.fspath(args.hidden_h5.resolve()),
                "cases": report_rows,
                "patch_sort": "raster order by layer_2_keep_patch_indices",
                "image_preprocessing": "square pad then resize to 336 before drawing 24x24 patch grid",
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote figure materials for {len(report_rows)} cases to {output_dir}")


def build_rep_bottleneck_pca_figure(args: argparse.Namespace) -> None:
    output_dir = ensure_dir(args.output_dir.resolve())
    projection_dir = ensure_dir(output_dir / "rep_pca_projection")
    panel_dir = ensure_dir(output_dir / "rep_pca_panels")
    config = load_config(args.config.resolve())
    all_rows = candidate_rows_with_rank(args.candidate_metrics.resolve())
    rows_by_rank = {int(row["rank"]): row for row in all_rows}
    shortlist_ranks = parse_rank_list(args.shortlist_ranks, DEFAULT_REP_FIG_SHORTLIST_RANKS)
    preferred_ranks = parse_rank_list(args.preferred_ranks, DEFAULT_REP_FIG_PREFERRED_RANKS)

    stats_cache: dict[tuple[str, int], dict[tuple[str, int], dict[str, Any]]] = {}

    def stats_lookup(path_text: str, layer: int, target_keep: int) -> dict[tuple[str, int], dict[str, Any]]:
        key = (path_text, layer)
        if key not in stats_cache:
            stats_cache[key] = {
                stats_ref(row): row
                for row in load_stats_records(Path(path_text), layer, target_keep)
            }
        return stats_cache[key]

    scored_cases: list[dict[str, Any]] = []
    for rank in shortlist_ranks:
        if rank not in rows_by_rank:
            continue
        row = rows_by_rank[rank]
        ref = (normalize_question_id(row["question_id"]), int(row["sample_idx"]))
        topk_stats = stats_lookup(row["topk_stats_path"], args.layer, args.target_keep)[ref]
        scnd_stats = stats_lookup(row["scnd_stats_path"], args.layer, args.target_keep)[ref]
        hidden, hidden_attrs = load_hidden_from_h5(args.hidden_h5.resolve(), ref[0], args.layer, sample_idx=ref[1])
        topk_hidden, topk_patches = selected_hidden_by_patch_order(hidden, topk_stats, args.layer)
        scnd_hidden, scnd_patches = selected_hidden_by_patch_order(hidden, scnd_stats, args.layer)
        if topk_hidden.shape[0] != args.target_keep or scnd_hidden.shape[0] != args.target_keep:
            raise ValueError(
                f"rank={rank} keep mismatch: topk={topk_hidden.shape[0]}, "
                f"scnd={scnd_hidden.shape[0]}, expected={args.target_keep}"
            )
        topk_sim = cosine_similarity_matrix(topk_hidden)
        scnd_sim = cosine_similarity_matrix(scnd_hidden)
        topk_summary = similarity_summary(topk_hidden, topk_sim)
        scnd_summary = similarity_summary(scnd_hidden, scnd_sim)
        projection = pca_project_pair(topk_hidden, scnd_hidden)
        topk_xy = projection["topk_xy"]
        scnd_xy = projection["scnd_xy"]
        topk_summary["pca_spread"] = pca_spread(topk_xy)
        scnd_summary["pca_spread"] = pca_spread(scnd_xy)
        axis_limits = pca_axis_limits(topk_xy, scnd_xy)
        image_path = resolve_image_path(config, row["dataset"], row.get("image_file"))
        roi_bbox = REP_FIG_ROI_PATCH_BBOX_BY_RANK.get(
            rank,
            patch_bbox_from_indices(scnd_patches, patch_per_row=DEFAULT_PATCH_PER_ROW),
        )
        scored = {
            "rank": rank,
            "row": row,
            "hidden_shape": list(hidden.shape),
            "hidden_attrs": {key: str(value) for key, value in hidden_attrs.items()},
            "topk_summary": topk_summary,
            "scnd_summary": scnd_summary,
            "topk_xy": topk_xy,
            "scnd_xy": scnd_xy,
            "axis_limits": axis_limits,
            "image_path": image_path,
            "roi_bbox": roi_bbox,
            "homogeneity_gain": topk_summary["mean_offdiag_cosine"] - scnd_summary["mean_offdiag_cosine"],
            "effective_rank_gain": scnd_summary["effective_rank"] - topk_summary["effective_rank"],
            "redundant_0p90_gain": (
                topk_summary["redundant_pair_ratio_cos_gt_0p90"]
                - scnd_summary["redundant_pair_ratio_cos_gt_0p90"]
            ),
            "pca_spread_gain": scnd_summary["pca_spread"] - topk_summary["pca_spread"],
            "explained_variance_ratio": projection["explained_variance_ratio"],
        }
        scored_cases.append(scored)

    if not scored_cases:
        raise ValueError("No shortlist cases could be scored.")

    shortlist_position = {rank: idx for idx, rank in enumerate(shortlist_ranks)}
    preferred_position = {rank: idx for idx, rank in enumerate(preferred_ranks)}

    def is_supported(case: dict[str, Any]) -> bool:
        return (
            case["homogeneity_gain"] >= args.min_homogeneity_gain
            and (
                case["effective_rank_gain"] > 0.0
                or case["redundant_0p90_gain"] > 0.0
                or case["pca_spread_gain"] > 0.0
            )
        )

    preferred_supported = [
        case
        for case in scored_cases
        if case["rank"] in preferred_position and is_supported(case)
    ]
    preferred_supported.sort(key=lambda case: preferred_position[case["rank"]])
    selected_cases = preferred_supported[: args.num_cases]
    if len(selected_cases) < args.num_cases:
        selected_rank_set = {case["rank"] for case in selected_cases}
        remaining = [case for case in scored_cases if case["rank"] not in selected_rank_set and is_supported(case)]
        remaining.sort(
            key=lambda case: (
                case["homogeneity_gain"],
                case["effective_rank_gain"],
                case["redundant_0p90_gain"],
                -shortlist_position.get(case["rank"], 9999),
            ),
            reverse=True,
        )
        selected_cases.extend(remaining[: args.num_cases - len(selected_cases)])
    if len(selected_cases) < args.num_cases:
        selected_rank_set = {case["rank"] for case in selected_cases}
        fallback = [case for case in scored_cases if case["rank"] not in selected_rank_set]
        fallback.sort(
            key=lambda case: (
                case["homogeneity_gain"],
                case["effective_rank_gain"],
                -shortlist_position.get(case["rank"], 9999),
            ),
            reverse=True,
        )
        selected_cases.extend(fallback[: args.num_cases - len(selected_cases)])

    metric_rows: list[dict[str, Any]] = []
    projection_rows: list[dict[str, Any]] = []
    for case in scored_cases:
        row = case["row"]
        selected = case in selected_cases
        metric_rows.append(
            {
                "selected": selected,
                "rank": case["rank"],
                "dataset": row["dataset"],
                "sample_idx": int(row["sample_idx"]),
                "question_id": row["question_id"],
                "question": row.get("question", ""),
                "topk_answer": row.get("topk_answer", ""),
                "scnd_answer": row.get("scnd_answer", ""),
                "gt_answer": row.get("gt_answer", ""),
                "topk_mean_offdiag_cosine": case["topk_summary"]["mean_offdiag_cosine"],
                "scnd_mean_offdiag_cosine": case["scnd_summary"]["mean_offdiag_cosine"],
                "homogeneity_gain_topk_minus_scnd": case["homogeneity_gain"],
                "topk_median_offdiag_cosine": case["topk_summary"]["median_offdiag_cosine"],
                "scnd_median_offdiag_cosine": case["scnd_summary"]["median_offdiag_cosine"],
                "topk_redundant_pair_ratio_cos_gt_0p90": case["topk_summary"]["redundant_pair_ratio_cos_gt_0p90"],
                "scnd_redundant_pair_ratio_cos_gt_0p90": case["scnd_summary"]["redundant_pair_ratio_cos_gt_0p90"],
                "redundant_0p90_gain_topk_minus_scnd": case["redundant_0p90_gain"],
                "topk_effective_rank": case["topk_summary"]["effective_rank"],
                "scnd_effective_rank": case["scnd_summary"]["effective_rank"],
                "effective_rank_gain_scnd_minus_topk": case["effective_rank_gain"],
                "topk_pca_spread": case["topk_summary"]["pca_spread"],
                "scnd_pca_spread": case["scnd_summary"]["pca_spread"],
                "pca_spread_gain_scnd_minus_topk": case["pca_spread_gain"],
                "pca_explained_var_1": float(case["explained_variance_ratio"][0]),
                "pca_explained_var_2": float(case["explained_variance_ratio"][1]),
            }
        )
        for method, xy in (("topk", case["topk_xy"]), ("scnd", case["scnd_xy"])):
            for point_idx, (x, y) in enumerate(xy):
                projection_rows.append(
                    {
                        "selected": selected,
                        "rank": case["rank"],
                        "dataset": row["dataset"],
                        "sample_idx": int(row["sample_idx"]),
                        "question_id": row["question_id"],
                        "method": method,
                        "point_idx": point_idx,
                        "pc1": float(x),
                        "pc2": float(y),
                    }
                )

    write_csv(output_dir / "rep_bottleneck_pca_metric_summary.csv", metric_rows)
    write_csv(output_dir / "rep_bottleneck_pca_projection.csv", projection_rows)

    rendered_cases: list[dict[str, Any]] = []
    for case in selected_cases:
        rank = case["rank"]
        row = case["row"]
        stem = f"rank{rank:02d}_{row['dataset']}_s{int(row['sample_idx']):05d}_{sanitize_filename(row['question_id'])}"
        evidence_panel = render_evidence_crop(case["image_path"], case["roi_bbox"], output_size=540)
        topk_pca_panel = render_pca_panel(
            case["topk_xy"],
            axis_limits=case["axis_limits"],
            color=(220, 40, 40),
            summary=case["topk_summary"],
        )
        scnd_pca_panel = render_pca_panel(
            case["scnd_xy"],
            axis_limits=case["axis_limits"],
            color=(40, 110, 220),
            summary=case["scnd_summary"],
        )
        evidence_panel.save(panel_dir / f"{stem}_image_evidence.png", dpi=(300, 300))
        topk_pca_panel.save(panel_dir / f"{stem}_topk_pca.png", dpi=(300, 300))
        scnd_pca_panel.save(panel_dir / f"{stem}_scnd_pca.png", dpi=(300, 300))
        np.save(projection_dir / f"{stem}_topk_xy.npy", case["topk_xy"])
        np.save(projection_dir / f"{stem}_scnd_xy.npy", case["scnd_xy"])
        rendered_cases.append(
            {
                **case,
                "evidence_panel": evidence_panel,
                "topk_pca_panel": topk_pca_panel,
                "scnd_pca_panel": scnd_pca_panel,
            }
        )

    png_path = output_dir / "scnd_rep_bottleneck_hidden_pca_main.png"
    pdf_path = output_dir / "scnd_rep_bottleneck_hidden_pca_main.pdf"
    render_rep_bottleneck_pca_figure(rendered_cases, png_path)
    render_rep_bottleneck_pca_figure(rendered_cases, pdf_path)

    (output_dir / "rep_bottleneck_pca_figure_manifest.json").write_text(
        json.dumps(
            {
                "task": "scnd_rep_bottleneck_hidden_pca_main_figure",
                "commit": git_commit(),
                "layer": args.layer,
                "target_keep": args.target_keep,
                "hidden_h5": os.fspath(args.hidden_h5.resolve()),
                "candidate_metrics": os.fspath(args.candidate_metrics.resolve()),
                "shortlist_ranks": shortlist_ranks,
                "preferred_ranks": preferred_ranks,
                "selection_policy": (
                    "keep preferred ranks when they satisfy min_homogeneity_gain "
                    "and at least one auxiliary diversity metric; otherwise sort by hidden-space evidence"
                ),
                "min_homogeneity_gain": args.min_homogeneity_gain,
                "selected_ranks": [case["rank"] for case in selected_cases],
                "outputs": {
                    "figure_png": os.fspath(png_path),
                    "figure_pdf": os.fspath(pdf_path),
                    "metric_summary_csv": os.fspath(output_dir / "rep_bottleneck_pca_metric_summary.csv"),
                    "projection_csv": os.fspath(output_dir / "rep_bottleneck_pca_projection.csv"),
                },
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        "Selected ranks for hidden PCA figure: "
        + ", ".join(str(case["rank"]) for case in selected_cases)
    )
    print(f"Wrote hidden PCA figure to {png_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    mine = subparsers.add_parser("mine-candidates", help="Mine candidate cases from existing retain64 stats")
    mine.add_argument("--topk-root", type=Path, default=DEFAULT_TOPK_ROOT)
    mine.add_argument("--scnd-root", type=Path, default=DEFAULT_SCND_ROOT)
    mine.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    mine.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    mine.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS), choices=list(DEFAULT_DATASETS))
    mine.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    mine.add_argument("--target-keep", type=int, default=DEFAULT_TARGET_KEEP)
    mine.add_argument("--top-n", type=int, default=50)
    mine.set_defaults(func=mine_candidates)

    build = subparsers.add_parser("build-fig", help="Build overlays and similarity heatmaps for selected cases")
    build.add_argument("--hidden-h5", type=Path, required=True)
    build.add_argument("--candidate-metrics", type=Path, default=DEFAULT_OUTPUT_DIR / "candidate_metrics.csv")
    build.add_argument("--selected-cases", type=Path, default=DEFAULT_OUTPUT_DIR / "selected_cases.json")
    build.add_argument("--case", action="append", help="Selected case as dataset:question_id; may be repeated")
    build.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    build.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    build.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    build.add_argument("--target-keep", type=int, default=DEFAULT_TARGET_KEEP)
    build.set_defaults(func=build_figures)

    pca_fig = subparsers.add_parser(
        "build-pca-main-fig",
        help="Build the representation-space PCA main figure from exported hidden states",
    )
    pca_fig.add_argument("--hidden-h5", type=Path, required=True)
    pca_fig.add_argument(
        "--candidate-metrics",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "outcome_textvqa_only/outcome_candidate_metrics.csv",
    )
    pca_fig.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "pca_main_figure")
    pca_fig.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    pca_fig.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    pca_fig.add_argument("--target-keep", type=int, default=DEFAULT_TARGET_KEEP)
    pca_fig.add_argument(
        "--shortlist-ranks",
        default=",".join(str(item) for item in DEFAULT_REP_FIG_SHORTLIST_RANKS),
        help="Comma-separated 1-based ranks from candidate metrics to score",
    )
    pca_fig.add_argument(
        "--preferred-ranks",
        default=",".join(str(item) for item in DEFAULT_REP_FIG_PREFERRED_RANKS),
        help="Comma-separated preferred ranks to keep if hidden metrics support them",
    )
    pca_fig.add_argument("--num-cases", type=int, default=2)
    pca_fig.add_argument("--min-homogeneity-gain", type=float, default=0.01)
    pca_fig.set_defaults(func=build_rep_bottleneck_pca_figure)

    pairwise_fig = subparsers.add_parser(
        "build-pairwise-main-fig",
        help="Build the pairwise-cosine main figure with shared color scale and density plots",
    )
    pairwise_fig.add_argument("--hidden-h5", type=Path, required=True)
    pairwise_fig.add_argument(
        "--candidate-metrics",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "outcome_textvqa_only/outcome_candidate_metrics.csv",
    )
    pairwise_fig.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "outcome_textvqa_only/pairwise_main_figure",
    )
    pairwise_fig.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    pairwise_fig.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    pairwise_fig.add_argument("--target-keep", type=int, default=DEFAULT_TARGET_KEEP)
    pairwise_fig.add_argument(
        "--selected-ranks",
        default=",".join(str(item) for item in DEFAULT_PAIRWISE_MAIN_RANKS),
        help="Comma-separated 1-based ranks from candidate metrics to render",
    )
    pairwise_fig.add_argument("--vmin", type=float, default=0.0)
    pairwise_fig.add_argument("--vmax", type=float, default=1.0)
    pairwise_fig.set_defaults(func=build_pairwise_main_figure)

    review = subparsers.add_parser("make-review", help="Build a static HTML dashboard for human candidate triage")
    review.add_argument("--candidate-metrics", type=Path, default=DEFAULT_OUTPUT_DIR / "candidate_metrics.csv")
    review.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    review.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    review.add_argument("--top-n", type=int, default=50)
    review.set_defaults(func=make_review)

    outcome = subparsers.add_parser(
        "mine-outcome-candidates",
        help="Mine SCND-correct / saliency-topK-wrong cases from existing retain64 answers",
    )
    outcome.add_argument("--topk-root", type=Path, default=DEFAULT_TOPK_ROOT)
    outcome.add_argument("--scnd-root", type=Path, default=DEFAULT_SCND_ROOT)
    outcome.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    outcome.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    outcome.add_argument("--datasets", nargs="+", default=list(OUTCOME_DATASETS), choices=list(OUTCOME_DATASETS))
    outcome.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    outcome.add_argument("--target-keep", type=int, default=DEFAULT_TARGET_KEEP)
    outcome.add_argument("--top-n", type=int, default=50)
    outcome.add_argument(
        "--case-type",
        default="scnd_correct_topk_wrong",
        choices=["scnd_correct_topk_wrong", "topk_correct_scnd_wrong", "both_correct", "both_wrong", "all"],
    )
    outcome.add_argument("--textvqa-correct-threshold", type=float, default=0.0)
    outcome.add_argument("--outcome-profile", choices=["broad", "anchored"], default="broad")
    outcome.add_argument("--min-jaccard", type=float, default=0.25)
    outcome.add_argument("--max-jaccard", type=float, default=0.45)
    outcome.add_argument("--min-entropy-gain", type=float, default=0.07)
    outcome.add_argument("--min-grid-distance-gain", type=float, default=0.025)
    outcome.set_defaults(func=mine_outcome_candidates)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
