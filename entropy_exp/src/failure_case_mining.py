"""Mine SparseVLM-vs-entropy-alpha failure cases and visualize pruned patches."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
except Exception as exc:  # pragma: no cover - runtime dependency guard
    raise RuntimeError("failure_case_mining.py requires matplotlib with the Agg backend") from exc

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional image reader fallback
    Image = None


LLAVA_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eval_datasets import load_textvqa_evaluator, textvqa_prompt_processor  # noqa: E402


DEFAULT_MATRIX_PATH = (
    LLAVA_ROOT / "entropy_exp" / "outputs" / "summary" / "sv2_vs_progressive_5dataset_merged_20260429" / "matrix.csv"
)
DEFAULT_OUTPUT_DIR = (
    LLAVA_ROOT / "entropy_exp" / "outputs" / "analysis" / "sv2_vs_ours_failure_cases_textvqa_pope_20260429"
)
DEFAULT_RUNS_DIR = LLAVA_ROOT / "entropy_exp" / "outputs" / "runs"
DEFAULT_DATASETS = ("textvqa", "pope")
DEFAULT_KEEPS = ("118", "60", "28", "20")
PRUNE_LAYERS = (2, 6, 15)
PATCH_PER_ROW = 24
METRIC_COLUMNS = {
    "textvqa": "textvqa_accuracy",
    "pope": "pope_macro_f1_percent",
}
GT_JOIN_LIMIT = 5


@dataclass(frozen=True)
class RunPair:
    dataset: str
    keep: str
    sv2_family: str
    ours_family: str
    sv2_run_dir: Path
    ours_run_dir: Path


@dataclass(frozen=True)
class EvalResult:
    question_id: str
    score: float
    correct: bool
    pred_answer: str
    parsed_answer: str
    gt_answer: str
    question: str
    image_file: str
    category: str


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_question_id(value: Any) -> str:
    return str(value)


def load_answer_map(run_dir: Path) -> dict[str, dict[str, Any]]:
    answers_path = run_dir / "answers.jsonl"
    if not answers_path.is_file():
        raise FileNotFoundError(f"Missing answers file: {answers_path}")
    return {normalize_question_id(row["question_id"]): row for row in read_jsonl(answers_path)}


def load_summary_metrics(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "eval" / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing eval summary: {summary_path}")
    with summary_path.open("r", encoding="utf-8") as handle:
        return json.load(handle).get("metrics", {})


def load_matrix_rows(matrix_path: Path) -> dict[str, dict[str, str]]:
    with matrix_path.open("r", encoding="utf-8") as handle:
        return {row["family"]: row for row in csv.DictReader(handle)}


def find_latest_complete_run(runs_dir: Path, dataset: str, strategy: str, family: str) -> Path:
    dataset_dir = runs_dir / strategy / dataset
    pattern = f"{dataset}_{strategy}_{family}_*"
    candidates = sorted(path for path in dataset_dir.glob(pattern) if path.is_dir())
    complete = [
        path
        for path in candidates
        if (path / "answers.jsonl").is_file()
        and (path / "stats.jsonl").is_file()
        and (path / "eval" / "summary.json").is_file()
    ]
    if not complete:
        raise FileNotFoundError(
            f"No complete run found for dataset={dataset}, strategy={strategy}, family={family}, pattern={pattern}"
        )
    return complete[-1].resolve()


def build_run_pair(runs_dir: Path, dataset: str, keep: str) -> RunPair:
    sv2_family = f"sv2_avg_{keep}"
    ours_family = f"ours_sv2ratio_{keep}"
    return RunPair(
        dataset=dataset,
        keep=keep,
        sv2_family=sv2_family,
        ours_family=ours_family,
        sv2_run_dir=find_latest_complete_run(runs_dir, dataset, "sparsevlm", sv2_family),
        ours_run_dir=find_latest_complete_run(runs_dir, dataset, "sparsevlm_entropy_alpha", ours_family),
    )


class TextVQAScorer:
    def __init__(self, annotation_file: Path, correct_threshold: float = 0.0) -> None:
        with annotation_file.open("r", encoding="utf-8") as handle:
            annotations = json.load(handle)["data"]
        self.annotation_map = {
            (normalize_question_id(item["image_id"]), str(item["question"]).lower()): item
            for item in annotations
        }
        self.evaluator = load_textvqa_evaluator()
        self.correct_threshold = correct_threshold

    def evaluate(self, answer: dict[str, Any]) -> EvalResult:
        question_id = normalize_question_id(answer["question_id"])
        question = textvqa_prompt_processor(answer["prompt"])
        annotation = self.annotation_map[(question_id, question)]
        scores = self.evaluator._compute_answer_scores(annotation["answers"])
        parsed = self.evaluator.answer_processor(answer["text"])
        score = float(scores.get(parsed, 0.0))
        gt_answers = list(dict.fromkeys(str(value) for value in annotation["answers"]))
        return EvalResult(
            question_id=question_id,
            score=score,
            correct=score > self.correct_threshold,
            pred_answer=str(answer["text"]),
            parsed_answer=parsed,
            gt_answer=" | ".join(gt_answers[:GT_JOIN_LIMIT]),
            question=str(annotation["question"]),
            image_file=f"{annotation['image_id']}.jpg",
            category="",
        )


def normalize_pope_answer(text: str) -> str:
    first_sentence = str(text).split(".", 1)[0]
    first_sentence = first_sentence.replace(",", " ")
    words = {word.strip().lower() for word in first_sentence.split()}
    return "no" if ("no" in words or "not" in words) else "yes"


class PopeScorer:
    def __init__(self, question_file: Path, annotation_dir: Path) -> None:
        questions = read_jsonl(question_file)
        labels_by_category: dict[str, list[dict[str, Any]]] = {}
        for label_path in sorted(annotation_dir.glob("coco_pope_*.json")):
            category = label_path.name[len("coco_pope_") : -len(".json")]
            labels_by_category[category] = read_jsonl(label_path)

        category_offsets = {category: 0 for category in labels_by_category}
        self.metadata: dict[str, dict[str, Any]] = {}
        for question in questions:
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

    def evaluate(self, answer: dict[str, Any]) -> EvalResult:
        question_id = normalize_question_id(answer["question_id"])
        metadata = self.metadata[question_id]
        parsed = normalize_pope_answer(str(answer["text"]))
        score = 1.0 if parsed == metadata["label"] else 0.0
        return EvalResult(
            question_id=question_id,
            score=score,
            correct=bool(score),
            pred_answer=str(answer["text"]),
            parsed_answer=parsed,
            gt_answer=metadata["label"],
            question=metadata["question"],
            image_file=metadata["image_file"],
            category=metadata["category"],
        )


def build_scorer(dataset: str, correct_threshold: float) -> Any:
    if dataset == "textvqa":
        annotation_file = LLAVA_ROOT / "entropy_exp" / "eval_questions" / "textvqa" / "TextVQA_0.5.1_val.json"
        return TextVQAScorer(annotation_file, correct_threshold=correct_threshold)
    if dataset == "pope":
        return PopeScorer(
            LLAVA_ROOT / "entropy_exp" / "eval_questions" / "pope" / "llava_pope_test.jsonl",
            LLAVA_ROOT / "entropy_exp" / "datasets" / "pope" / "coco",
        )
    raise ValueError(f"Unsupported dataset: {dataset}")


def classify_case(sv2_eval: EvalResult, ours_eval: EvalResult) -> str:
    if sv2_eval.correct and ours_eval.correct:
        return "both_correct"
    if sv2_eval.correct and not ours_eval.correct:
        return "sv2_correct_ours_wrong"
    if not sv2_eval.correct and ours_eval.correct:
        return "ours_correct_sv2_wrong"
    return "both_wrong"


def iter_stats_for_qids(stats_path: Path, qids: set[str]) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    with stats_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            question_id = normalize_question_id(row.get("question_id"))
            if question_id in qids:
                found[question_id] = row
                if len(found) == len(qids):
                    break
    return found


def list_ints(value: Any) -> list[int]:
    if value is None:
        return []
    return [int(item) for item in value]


def _initial_patch_indices(row: dict[str, Any], patch_per_row: int = PATCH_PER_ROW) -> list[int]:
    first_layer = min(PRUNE_LAYERS)
    current_patch = list_ints(row.get(f"layer_{first_layer}_current_patch_indices"))
    if current_patch:
        return current_patch

    before = int(row.get(f"layer_{first_layer}_before", patch_per_row * patch_per_row))
    return list(range(before))


def canonicalize_patch_indices(
    row: dict[str, Any] | None,
    layers: tuple[int, ...] = PRUNE_LAYERS,
    patch_per_row: int = PATCH_PER_ROW,
) -> dict[str, Any] | None:
    """Return a stats row whose patch-index fields are original-image patch ids.

    Older SparseVLM stats only store layer-local keep/pruned indices. For a
    progressive prune run, layer-local index 0 at L6 means "the first token that
    survived L2", not original patch 0. This helper reconstructs the cumulative
    alive patch mapping so downstream visualizations always use original patch
    coordinates.
    """
    if row is None:
        return None

    canonical = dict(row)
    alive = _initial_patch_indices(row, patch_per_row=patch_per_row)
    max_requested_layer = max(layers) if layers else max(PRUNE_LAYERS)
    reconstruction_layers = tuple(layer for layer in PRUNE_LAYERS if layer <= max_requested_layer)

    for layer in reconstruction_layers:
        current_key = f"layer_{layer}_current_patch_indices"
        keep_key = f"layer_{layer}_keep_patch_indices"
        pruned_key = f"layer_{layer}_pruned_patch_indices"
        high_key = f"layer_{layer}_high_keep_patch_indices"
        low_key = f"layer_{layer}_low_keep_patch_indices"

        current_patch = list_ints(row.get(current_key))
        if current_patch:
            alive = current_patch
            canonical[current_key] = current_patch
        else:
            canonical[current_key] = list(alive)

        keep_local = list_ints(row.get(f"layer_{layer}_keep_indices"))
        pruned_local = list_ints(row.get(f"layer_{layer}_pruned_indices"))

        if keep_local:
            keep_patch = [alive[index] for index in keep_local if 0 <= index < len(alive)]
        else:
            keep_patch = list_ints(row.get(keep_key))
        if pruned_local:
            pruned_patch = [alive[index] for index in pruned_local if 0 <= index < len(alive)]
        else:
            pruned_patch = list_ints(row.get(pruned_key))

        canonical[keep_key] = keep_patch
        canonical[pruned_key] = pruned_patch

        high_local = list_ints(row.get(f"layer_{layer}_high_keep_indices"))
        low_local = list_ints(row.get(f"layer_{layer}_low_keep_indices"))
        if high_local:
            canonical[high_key] = [alive[index] for index in high_local if 0 <= index < len(alive)]
        elif high_key in row:
            canonical[high_key] = list_ints(row.get(high_key))
        if low_local:
            canonical[low_key] = [alive[index] for index in low_local if 0 <= index < len(alive)]
        elif low_key in row:
            canonical[low_key] = list_ints(row.get(low_key))

        if keep_patch:
            alive = keep_patch

    return canonical


def entropy_norm_from_scores(scores: list[float]) -> float:
    if len(scores) <= 1:
        return 0.0
    values = np.asarray(scores, dtype=np.float64)
    total = float(values.sum())
    if total <= 0.0:
        return 0.0
    probs = values / total
    positive = probs[probs > 0.0]
    entropy = float(-(positive * np.log(positive)).sum())
    return float(np.clip(entropy / math.log(len(values)), 0.0, 1.0))


def importance_mass_for_patches(row: dict[str, Any], layer: int, patch_indices: set[int]) -> float:
    scores = row.get(f"layer_{layer}_importance") or []
    current_patch_indices = list_ints(row.get(f"layer_{layer}_current_patch_indices"))
    if not scores or not current_patch_indices:
        return 0.0
    total = float(sum(float(score) for score in scores))
    if total <= 0.0:
        return 0.0
    mass = 0.0
    for patch_idx, score in zip(current_patch_indices, scores):
        if int(patch_idx) in patch_indices:
            mass += float(score)
    return mass / total


def compute_patch_diff_metrics(
    sv2_stats: dict[str, Any] | None,
    ours_stats: dict[str, Any] | None,
    layers: tuple[int, ...] = PRUNE_LAYERS,
) -> dict[str, Any]:
    sv2_stats = canonicalize_patch_indices(sv2_stats, layers=layers)
    ours_stats = canonicalize_patch_indices(ours_stats, layers=layers)
    if sv2_stats is None or ours_stats is None:
        return {
            "patch_diff_score": 0.0,
            "max_sv2_only_keep_ratio": 0.0,
            "max_ours_only_keep_ratio": 0.0,
            "max_ours_low_keep_ratio": 0.0,
            "max_sv2_only_saliency_mass": 0.0,
            "max_ours_entropy_norm": 0.0,
        }

    max_sv2_only_keep_ratio = 0.0
    max_ours_only_keep_ratio = 0.0
    max_ours_low_keep_ratio = 0.0
    max_sv2_only_saliency_mass = 0.0
    max_ours_entropy_norm = 0.0

    for layer in layers:
        sv2_keep = set(list_ints(sv2_stats.get(f"layer_{layer}_keep_patch_indices")))
        ours_keep = set(list_ints(ours_stats.get(f"layer_{layer}_keep_patch_indices")))
        if not sv2_keep or not ours_keep:
            continue
        sv2_only = sv2_keep - ours_keep
        ours_only = ours_keep - sv2_keep
        sv2_only_ratio = len(sv2_only) / max(1, len(sv2_keep))
        ours_only_ratio = len(ours_only) / max(1, len(ours_keep))
        low_keep = set(list_ints(ours_stats.get(f"layer_{layer}_low_keep_patch_indices")))
        low_ratio = len(low_keep) / max(1, len(ours_keep))
        saliency_mass = importance_mass_for_patches(ours_stats, layer, sv2_only)
        entropy_norm = entropy_norm_from_scores(ours_stats.get(f"layer_{layer}_importance") or [])

        max_sv2_only_keep_ratio = max(max_sv2_only_keep_ratio, sv2_only_ratio)
        max_ours_only_keep_ratio = max(max_ours_only_keep_ratio, ours_only_ratio)
        max_ours_low_keep_ratio = max(max_ours_low_keep_ratio, low_ratio)
        max_sv2_only_saliency_mass = max(max_sv2_only_saliency_mass, saliency_mass)
        max_ours_entropy_norm = max(max_ours_entropy_norm, entropy_norm)

    patch_diff_score = (
        max_sv2_only_keep_ratio
        + 0.5 * max_sv2_only_saliency_mass
        + 0.25 * max_ours_low_keep_ratio
    )
    return {
        "patch_diff_score": patch_diff_score,
        "max_sv2_only_keep_ratio": max_sv2_only_keep_ratio,
        "max_ours_only_keep_ratio": max_ours_only_keep_ratio,
        "max_ours_low_keep_ratio": max_ours_low_keep_ratio,
        "max_sv2_only_saliency_mass": max_sv2_only_saliency_mass,
        "max_ours_entropy_norm": max_ours_entropy_norm,
    }


def load_config(run_dir: Path) -> dict[str, Any]:
    with (run_dir / "config.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_image_path(run_dir: Path, dataset: str, image_file: str) -> Path | None:
    config = load_config(run_dir)
    image_folder = ((config.get("datasets") or {}).get(dataset) or {}).get("image_folder")
    candidates = []
    if image_folder:
        folder = Path(str(image_folder))
        if not folder.is_absolute():
            folder = LLAVA_ROOT / folder
        candidates.append(folder / image_file)
    candidates.extend(
        [
            LLAVA_ROOT / "entropy_exp" / "datasets" / "textvqa" / "train_images" / image_file,
            LLAVA_ROOT / "entropy_exp" / "datasets" / "pope" / "val2014" / image_file,
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def load_image(path: Path | None) -> np.ndarray | None:
    if path is None or not path.is_file():
        return None
    if Image is not None:
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"))
    return plt.imread(path)


def grid_from_indices(indices: Iterable[int], patch_per_row: int = PATCH_PER_ROW, value: int = 1) -> np.ndarray:
    grid = np.zeros((patch_per_row, patch_per_row), dtype=np.float64)
    for patch_idx in indices:
        patch_idx = int(patch_idx)
        if 0 <= patch_idx < patch_per_row * patch_per_row:
            grid[patch_idx // patch_per_row, patch_idx % patch_per_row] = value
    return grid


def saliency_grid(row: dict[str, Any], layer: int, patch_per_row: int = PATCH_PER_ROW) -> np.ndarray:
    grid = np.zeros((patch_per_row, patch_per_row), dtype=np.float64)
    scores = row.get(f"layer_{layer}_importance") or []
    patch_indices = list_ints(row.get(f"layer_{layer}_current_patch_indices"))
    for patch_idx, score in zip(patch_indices, scores):
        if 0 <= int(patch_idx) < patch_per_row * patch_per_row:
            grid[int(patch_idx) // patch_per_row, int(patch_idx) % patch_per_row] = float(score)
    max_value = float(grid.max())
    if max_value > 0:
        grid = grid / max_value
    return grid


def show_patch_overlay(
    ax: Any,
    image: np.ndarray | None,
    grid: np.ndarray,
    *,
    title: str,
    cmap: Any,
    norm: Any = None,
    alpha: float = 0.45,
) -> None:
    if image is not None:
        ax.imshow(image)
        height, width = image.shape[:2]
        extent = (0, width, height, 0)
    else:
        extent = None
    masked = np.ma.masked_where(grid == 0, grid)
    ax.imshow(masked, cmap=cmap, norm=norm, alpha=alpha, interpolation="nearest", extent=extent)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def show_heatmap(ax: Any, grid: np.ndarray, title: str) -> None:
    im = ax.imshow(grid, cmap="magma", interpolation="nearest", vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def truncate_text(text: str, max_len: int = 160) -> str:
    clean = " ".join(str(text).split())
    return clean if len(clean) <= max_len else clean[: max_len - 3] + "..."


def render_case_figure(
    case: dict[str, Any],
    sv2_stats: dict[str, Any],
    ours_stats: dict[str, Any],
    sv2_run_dir: Path,
    output_path: Path,
) -> None:
    sv2_stats = canonicalize_patch_indices(sv2_stats) or sv2_stats
    ours_stats = canonicalize_patch_indices(ours_stats) or ours_stats
    image_file = case.get("image_file") or ours_stats.get("image_file") or sv2_stats.get("image_file")
    image = load_image(resolve_image_path(sv2_run_dir, str(case["dataset"]), str(image_file)))

    keep_cmap = ListedColormap(["#2ca02c"])
    high_low_cmap = ListedColormap(["#ffbf00", "#17becf"])
    high_low_norm = BoundaryNorm([0.5, 1.5, 2.5], high_low_cmap.N)
    diff_cmap = ListedColormap(["#7f7f7f", "#d62728", "#1f77b4"])
    diff_norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5], diff_cmap.N)

    fig, axes = plt.subplots(len(PRUNE_LAYERS), 5, figsize=(18, 10), constrained_layout=True)
    if len(PRUNE_LAYERS) == 1:
        axes = np.asarray([axes])

    for row_idx, layer in enumerate(PRUNE_LAYERS):
        sv2_keep = set(list_ints(sv2_stats.get(f"layer_{layer}_keep_patch_indices")))
        ours_keep = set(list_ints(ours_stats.get(f"layer_{layer}_keep_patch_indices")))
        ours_high = set(list_ints(ours_stats.get(f"layer_{layer}_high_keep_patch_indices")))
        ours_low = set(list_ints(ours_stats.get(f"layer_{layer}_low_keep_patch_indices")))
        both_keep = sv2_keep & ours_keep
        sv2_only = sv2_keep - ours_keep
        ours_only = ours_keep - sv2_keep

        show_patch_overlay(
            axes[row_idx, 0],
            image,
            grid_from_indices(sv2_keep),
            title=f"L{layer} SV2 keep ({len(sv2_keep)})",
            cmap=keep_cmap,
        )
        show_patch_overlay(
            axes[row_idx, 1],
            image,
            grid_from_indices(ours_keep),
            title=f"L{layer} ours keep ({len(ours_keep)})",
            cmap=keep_cmap,
        )

        high_low_grid = grid_from_indices(ours_high, value=1) + grid_from_indices(ours_low, value=2)
        show_patch_overlay(
            axes[row_idx, 2],
            image,
            high_low_grid,
            title=f"L{layer} ours high/low ({len(ours_high)}/{len(ours_low)})",
            cmap=high_low_cmap,
            norm=high_low_norm,
            alpha=0.55,
        )

        diff_grid = (
            grid_from_indices(both_keep, value=1)
            + grid_from_indices(sv2_only, value=2)
            + grid_from_indices(ours_only, value=3)
        )
        show_patch_overlay(
            axes[row_idx, 3],
            image,
            diff_grid,
            title=f"L{layer} diff gray/shared red/SV2 blue/ours",
            cmap=diff_cmap,
            norm=diff_norm,
            alpha=0.55,
        )

        entropy_norm = entropy_norm_from_scores(ours_stats.get(f"layer_{layer}_importance") or [])
        show_heatmap(
            axes[row_idx, 4],
            saliency_grid(ours_stats, layer),
            title=f"L{layer} ours saliency Hnorm={entropy_norm:.3f}",
        )

    title = (
        f"{case['dataset']} keep={case['keep']} qid={case['question_id']} "
        f"type={case['case_type']} score_gap={case['score_gap']:.3f}\n"
        f"Q: {truncate_text(case['question'])}\n"
        f"GT: {truncate_text(case['gt_answer'], 100)} | SV2: {truncate_text(case['sv2_answer'], 100)} "
        f"| ours: {truncate_text(case['ours_answer'], 100)}"
    )
    fig.suptitle(title, fontsize=10)
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def build_case_records(
    pair: RunPair,
    scorer: Any,
    matrix_rows: dict[str, dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sv2_answers = load_answer_map(pair.sv2_run_dir)
    ours_answers = load_answer_map(pair.ours_run_dir)
    common_qids = sorted(set(sv2_answers) & set(ours_answers))
    metric_col = METRIC_COLUMNS[pair.dataset]
    sv2_metric = float(matrix_rows[pair.sv2_family][metric_col])
    ours_metric = float(matrix_rows[pair.ours_family][metric_col])

    cases: list[dict[str, Any]] = []
    counts = {
        "both_correct": 0,
        "both_wrong": 0,
        "sv2_correct_ours_wrong": 0,
        "ours_correct_sv2_wrong": 0,
    }
    for question_id in common_qids:
        sv2_eval = scorer.evaluate(sv2_answers[question_id])
        ours_eval = scorer.evaluate(ours_answers[question_id])
        case_type = classify_case(sv2_eval, ours_eval)
        counts[case_type] += 1
        cases.append(
            {
                "dataset": pair.dataset,
                "keep": pair.keep,
                "question_id": question_id,
                "image_file": sv2_eval.image_file or ours_eval.image_file,
                "question": sv2_eval.question or ours_eval.question,
                "gt_answer": sv2_eval.gt_answer or ours_eval.gt_answer,
                "category": sv2_eval.category or ours_eval.category,
                "sv2_answer": sv2_eval.pred_answer,
                "ours_answer": ours_eval.pred_answer,
                "sv2_parsed_answer": sv2_eval.parsed_answer,
                "ours_parsed_answer": ours_eval.parsed_answer,
                "sv2_score": sv2_eval.score,
                "ours_score": ours_eval.score,
                "score_gap": sv2_eval.score - ours_eval.score,
                "sv2_correct": sv2_eval.correct,
                "ours_correct": ours_eval.correct,
                "case_type": case_type,
                "sv2_family": pair.sv2_family,
                "ours_family": pair.ours_family,
                "sv2_run_dir": str(pair.sv2_run_dir),
                "ours_run_dir": str(pair.ours_run_dir),
                "sv2_metric": sv2_metric,
                "ours_metric": ours_metric,
                "metric_delta": ours_metric - sv2_metric,
            }
        )

    gap_record = {
        "dataset": pair.dataset,
        "keep": pair.keep,
        "sv2_family": pair.sv2_family,
        "ours_family": pair.ours_family,
        "sv2_run_dir": str(pair.sv2_run_dir),
        "ours_run_dir": str(pair.ours_run_dir),
        "metric_name": metric_col,
        "sv2_metric": sv2_metric,
        "ours_metric": ours_metric,
        "metric_delta": ours_metric - sv2_metric,
        "metric_delta_relative_percent": ((ours_metric - sv2_metric) / sv2_metric * 100.0) if sv2_metric else 0.0,
        "sample_count": len(common_qids),
        **counts,
        "net_loss": counts["sv2_correct_ours_wrong"] - counts["ours_correct_sv2_wrong"],
    }
    return cases, gap_record


def enrich_and_select_cases(
    cases: list[dict[str, Any]],
    pair: RunPair,
    max_cases_per_pair: int,
) -> list[dict[str, Any]]:
    failure_cases = [case for case in cases if case["case_type"] == "sv2_correct_ours_wrong"]
    qids = {case["question_id"] for case in failure_cases}
    if not qids:
        return []

    sv2_stats = iter_stats_for_qids(pair.sv2_run_dir / "stats.jsonl", qids)
    ours_stats = iter_stats_for_qids(pair.ours_run_dir / "stats.jsonl", qids)
    enriched: list[dict[str, Any]] = []
    for case in failure_cases:
        question_id = case["question_id"]
        diff_metrics = compute_patch_diff_metrics(sv2_stats.get(question_id), ours_stats.get(question_id))
        enriched_case = {**case, **diff_metrics}
        enriched_case["selection_score"] = float(enriched_case["score_gap"]) + float(diff_metrics["patch_diff_score"])
        enriched.append(enriched_case)

    enriched.sort(
        key=lambda item: (
            float(item["selection_score"]),
            float(item["patch_diff_score"]),
            float(item["score_gap"]),
            item["question_id"],
        ),
        reverse=True,
    )
    return enriched[:max_cases_per_pair]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def render_selected_figures(
    selected_cases: list[dict[str, Any]],
    output_dir: Path,
    runs_dir: Path,
) -> list[dict[str, Any]]:
    selected_by_pair: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for case in selected_cases:
        selected_by_pair.setdefault((case["dataset"], case["keep"]), []).append(case)

    rendered: list[dict[str, Any]] = []
    for (dataset, keep), cases in selected_by_pair.items():
        if not cases:
            continue
        pair = build_run_pair(runs_dir, dataset, keep)
        qids = {case["question_id"] for case in cases}
        sv2_stats = iter_stats_for_qids(pair.sv2_run_dir / "stats.jsonl", qids)
        ours_stats = iter_stats_for_qids(pair.ours_run_dir / "stats.jsonl", qids)
        for rank, case in enumerate(cases, start=1):
            question_id = case["question_id"]
            figure_path = (
                output_dir
                / "figures"
                / dataset
                / f"keep_{keep}"
                / f"{rank:02d}_{re.sub(r'[^A-Za-z0-9._-]+', '_', question_id)}.png"
            )
            sv2_row = sv2_stats.get(question_id)
            ours_row = ours_stats.get(question_id)
            if sv2_row is None or ours_row is None:
                case = {**case, "figure_path": "", "figure_error": "missing_stats"}
            else:
                render_case_figure(case, sv2_row, ours_row, pair.sv2_run_dir, figure_path)
                case = {
                    **case,
                    "figure_path": str(figure_path.relative_to(output_dir)),
                    "figure_error": "",
                }
            rendered.append(case)
    return rendered


def write_index(output_dir: Path, gap_rows: list[dict[str, Any]], selected_cases: list[dict[str, Any]]) -> None:
    lines = [
        "# TextVQA / POPE SV2-vs-Ours Failure Cases",
        "",
        "This report compares `sv2_avg_*` against `ours_sv2ratio_*` for keep levels 118/60/28/20.",
        "",
        "## Gap Rank",
        "",
        "| dataset | keep | metric delta | sv2_correct_ours_wrong | ours_correct_sv2_wrong | net_loss |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(gap_rows, key=lambda item: (item["net_loss"], -item["metric_delta"]), reverse=True):
        lines.append(
            f"| {row['dataset']} | {row['keep']} | {row['metric_delta']:.4f} | "
            f"{row['sv2_correct_ours_wrong']} | {row['ours_correct_sv2_wrong']} | {row['net_loss']} |"
        )

    lines.extend(["", "## Selected Cases", ""])
    for dataset in DEFAULT_DATASETS:
        dataset_cases = [case for case in selected_cases if case["dataset"] == dataset]
        if not dataset_cases:
            continue
        lines.extend([f"### {dataset}", ""])
        for keep in DEFAULT_KEEPS:
            keep_cases = [case for case in dataset_cases if case["keep"] == keep]
            if not keep_cases:
                continue
            lines.extend([f"#### keep {keep}", ""])
            for case in keep_cases:
                figure_path = case.get("figure_path") or ""
                link = f"[figure]({figure_path})" if figure_path else "figure unavailable"
                lines.append(
                    f"- `{case['question_id']}` score_gap={case['score_gap']:.3f}, "
                    f"patch_score={case['patch_diff_score']:.3f}, {link}; "
                    f"Q: {truncate_text(case['question'], 100)}"
                )
            lines.append("")

    (output_dir / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mine TextVQA/POPE cases where SparseVLM is correct and entropy-alpha is wrong."
    )
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX_PATH)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS), choices=list(DEFAULT_DATASETS))
    parser.add_argument("--keeps", nargs="+", default=list(DEFAULT_KEEPS), choices=list(DEFAULT_KEEPS))
    parser.add_argument("--max-cases-per-pair", type=int, default=30)
    parser.add_argument("--textvqa-correct-threshold", type=float, default=0.0)
    parser.add_argument("--skip-figures", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir.resolve())
    matrix_rows = load_matrix_rows(args.matrix.resolve())

    all_cases: list[dict[str, Any]] = []
    gap_rows: list[dict[str, Any]] = []
    selected_cases: list[dict[str, Any]] = []

    scorers = {
        dataset: build_scorer(dataset, correct_threshold=args.textvqa_correct_threshold)
        for dataset in args.datasets
    }

    for dataset in args.datasets:
        for keep in args.keeps:
            pair = build_run_pair(args.runs_dir.resolve(), dataset, keep)
            cases, gap_record = build_case_records(pair, scorers[dataset], matrix_rows)
            all_cases.extend(cases)
            gap_rows.append(gap_record)
            selected_cases.extend(enrich_and_select_cases(cases, pair, args.max_cases_per_pair))

    gap_rows.sort(key=lambda item: (item["net_loss"], -item["metric_delta"]), reverse=True)
    selected_cases.sort(
        key=lambda item: (
            item["dataset"],
            int(item["keep"]),
            -float(item["selection_score"]),
        )
    )

    if not args.skip_figures:
        selected_cases = render_selected_figures(selected_cases, output_dir, args.runs_dir.resolve())

    write_csv(output_dir / "gap_rank.csv", gap_rows)
    write_jsonl(output_dir / "all_cases.jsonl", all_cases)
    write_jsonl(output_dir / "selected_cases.jsonl", selected_cases)
    write_csv(output_dir / "case_summary.csv", selected_cases)
    write_index(output_dir, gap_rows, selected_cases)

    summary = {
        "matrix": str(args.matrix.resolve()),
        "runs_dir": str(args.runs_dir.resolve()),
        "output_dir": str(output_dir),
        "datasets": args.datasets,
        "keeps": args.keeps,
        "max_cases_per_pair": args.max_cases_per_pair,
        "gap_rows": len(gap_rows),
        "all_case_count": len(all_cases),
        "selected_case_count": len(selected_cases),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, sort_keys=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
