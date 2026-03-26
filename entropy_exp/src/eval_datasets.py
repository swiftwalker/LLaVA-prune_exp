"""Evaluation integration for legacy answers files and run-local outputs."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import OrderedDict

LLAVA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_POPE_QUESTION_FILE = os.path.join(
    LLAVA_ROOT, "entropy_exp", "eval_questions", "pope", "llava_pope_test.jsonl"
)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def write_json(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def emit_process_output(result: subprocess.CompletedProcess, output_path: str) -> None:
    stdout = result.stdout or ""
    stderr = result.stderr or ""

    if stdout:
        print(stdout, end="" if stdout.endswith("\n") else "\n")
    if stderr:
        print(stderr, file=sys.stderr, end="" if stderr.endswith("\n") else "\n")

    combined = stdout
    if stderr:
        if combined and not combined.endswith("\n"):
            combined += "\n"
        combined += "[stderr]\n" + stderr
    write_text(output_path, combined)

    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode,
            result.args,
            output=result.stdout,
            stderr=result.stderr,
        )


def parse_gqa_metrics(stdout: str) -> dict:
    metrics = {}
    wanted = {
        "binary",
        "open",
        "accuracy",
        "consistency",
        "validity",
        "plausibility",
        "grounding",
        "distribution",
    }
    pattern = re.compile(r"^([A-Za-z]+):\s+(-?\d+(?:\.\d+)?)")
    for line in stdout.splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        key = match.group(1).lower()
        if key in wanted:
            metrics[key] = float(match.group(2))
    return metrics


def parse_mme_metrics(stdout: str) -> dict:
    metrics = {}
    current_section = None
    section_pattern = re.compile(r"^=+\s*([A-Za-z]+)\s*=+$")
    total_pattern = re.compile(r"^total score:\s*(-?\d+(?:\.\d+)?)")
    overall_pattern = re.compile(r"^overall total score:\s*(-?\d+(?:\.\d+)?)$")
    task_pattern = re.compile(r"^([A-Za-z_]+)\s+score:\s*(-?\d+(?:\.\d+)?)$")

    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        overall_match = overall_pattern.match(line)
        if overall_match:
            metrics["overall_total_score"] = float(overall_match.group(1))
            continue
        section_match = section_pattern.match(line)
        if section_match:
            current_section = section_match.group(1)
            metrics[current_section] = {"tasks": {}}
            continue
        if current_section is None:
            continue
        total_match = total_pattern.match(line)
        if total_match:
            metrics[current_section]["total_score"] = float(total_match.group(1))
            continue
        task_match = task_pattern.match(line)
        if task_match:
            metrics[current_section]["tasks"][task_match.group(1)] = float(task_match.group(2))
    return metrics


def parse_pope_metrics(stdout: str) -> dict:
    metrics = {}
    current_category = None
    category_pattern = re.compile(r"^Category:\s*([^,]+),\s*# samples:\s*(\d+)")
    value_pattern = re.compile(r"^(Accuracy|Precision|Recall|F1 score|Yes ratio):\s*(-?\d+(?:\.\d+)?)$")

    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        category_match = category_pattern.match(line)
        if category_match:
            current_category = category_match.group(1)
            metrics[current_category] = {"samples": int(category_match.group(2))}
            continue
        if current_category is None:
            continue
        value_match = value_pattern.match(line)
        if value_match:
            key = value_match.group(1).lower().replace(" ", "_")
            metrics[current_category][key] = float(value_match.group(2))
    return metrics


def add_pope_macro_f1(metrics: dict) -> dict:
    category_metrics = {
        key: value
        for key, value in metrics.items()
        if isinstance(value, dict) and "samples" in value
    }
    if not category_metrics:
        return metrics

    f1_scores = [item["f1_score"] for item in category_metrics.values() if "f1_score" in item]
    if not f1_scores:
        return metrics

    metrics["macro_f1"] = sum(f1_scores) / len(f1_scores)
    return metrics


def format_pope_macro_f1(macro_f1: float) -> str:
    return f"Macro-F1: {macro_f1:.6f}"


def load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def normalize_pope_answer(text: str) -> str:
    if text.find(".") != -1:
        text = text.split(".")[0]
    text = text.replace(",", "")
    words = text.split(" ")
    if "No" in words or "not" in words or "no" in words:
        return "no"
    return "yes"


def normalize_pope_question_text(text: str) -> str:
    return text.split("\n", 1)[0].strip()


def make_pope_question_key(image: str | None, text: str | None) -> tuple[str, str]:
    return (str(image or ""), normalize_pope_question_text(str(text or "")))


def evaluate_pope_category(
    answers: list[dict],
    question_by_id: dict[int, dict],
    label_by_question_id: dict[int, int],
    label_by_question_key: dict[tuple[str, str], int],
) -> dict | None:
    pred_list: list[int] = []
    label_list: list[int] = []

    for answer in answers:
        question_id = int(answer.get("question_id"))
        question = question_by_id.get(question_id)
        label = label_by_question_id.get(question_id)
        if label is None and question is not None:
            label = label_by_question_key.get(make_pope_question_key(question.get("image"), question.get("text")))
        if label is None:
            continue
        pred_list.append(0 if normalize_pope_answer(str(answer.get("text", ""))) == "no" else 1)
        label_list.append(label)

    if not pred_list:
        return None

    yes_ratio = pred_list.count(1) / len(pred_list)
    pos = 1
    neg = 0
    tp = tn = fp = fn = 0
    for pred, label in zip(pred_list, label_list):
        if pred == pos and label == pos:
            tp += 1
        elif pred == pos and label == neg:
            fp += 1
        elif pred == neg and label == neg:
            tn += 1
        elif pred == neg and label == pos:
            fn += 1

    precision = float(tp) / float(tp + fp) if (tp + fp) else 0.0
    recall = float(tp) / float(tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0

    return {
        "samples": len(pred_list),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "yes_ratio": yes_ratio,
    }


def evaluate_pope_answers(answers: list[dict], questions: list[dict], annotation_dir: str) -> tuple[str, dict]:
    categories_in_order: list[str] = []
    question_category_by_id: OrderedDict[int, str] = OrderedDict()
    question_by_id: dict[int, dict] = {}
    for question in questions:
        category = question["category"]
        question_id = int(question["question_id"])
        question_category_by_id[question_id] = category
        question_by_id[question_id] = question
        if category not in categories_in_order:
            categories_in_order.append(category)

    answers_by_category: dict[str, list[dict]] = {category: [] for category in categories_in_order}
    for answer in answers:
        category = question_category_by_id.get(answer.get("question_id"))
        if category is None:
            continue
        answers_by_category.setdefault(category, []).append(answer)

    metrics: dict = {}
    stdout_lines: list[str] = []
    for category in categories_in_order:
        label_file = os.path.join(annotation_dir, f"coco_pope_{category}.json")
        if not os.path.exists(label_file):
            raise FileNotFoundError(f"POPE annotation file not found: {label_file}")

        label_by_question_id = {}
        label_by_question_key = {}
        for row in load_jsonl(label_file):
            label = 1 if row["label"] != "no" else 0
            label_by_question_id[int(row["question_id"])] = label
            label_by_question_key[make_pope_question_key(row.get("image"), row.get("text"))] = label

        category_metrics = evaluate_pope_category(
            answers_by_category.get(category, []),
            question_by_id,
            label_by_question_id,
            label_by_question_key,
        )
        if category_metrics is None:
            continue

        metrics[category] = {
            "samples": category_metrics["samples"],
            "accuracy": category_metrics["accuracy"],
            "precision": category_metrics["precision"],
            "recall": category_metrics["recall"],
            "f1_score": category_metrics["f1_score"],
            "yes_ratio": category_metrics["yes_ratio"],
        }
        stdout_lines.extend(
            [
                f"Category: {category}, # samples: {category_metrics['samples']}",
                "TP\tFP\tTN\tFN\t",
                f"{category_metrics['tp']}\t{category_metrics['fp']}\t{category_metrics['tn']}\t{category_metrics['fn']}",
                f"Accuracy: {category_metrics['accuracy']}",
                f"Precision: {category_metrics['precision']}",
                f"Recall: {category_metrics['recall']}",
                f"F1 score: {category_metrics['f1_score']}",
                f"Yes ratio: {category_metrics['yes_ratio']}",
                (
                    f"{category_metrics['f1_score']:.3f}, {category_metrics['accuracy']:.3f}, "
                    f"{category_metrics['precision']:.3f}, {category_metrics['recall']:.3f}, "
                    f"{category_metrics['yes_ratio']:.3f}"
                ),
                "====================================",
            ]
        )

    metrics = add_pope_macro_f1(metrics)
    return "\n".join(stdout_lines).rstrip(), metrics


def infer_config_value(config_file: str, path: tuple[str, ...]) -> str | None:
    stack: list[tuple[int, str]] = []
    with open(config_file, "r", encoding="utf-8") as f:
        for raw_line in f:
            if not raw_line.strip() or raw_line.lstrip().startswith("#"):
                continue

            stripped = raw_line.strip()
            indent = len(raw_line) - len(raw_line.lstrip(" "))
            while stack and indent <= stack[-1][0]:
                stack.pop()

            if ":" not in stripped:
                continue

            key, value = stripped.split(":", 1)
            key = key.strip()
            value = value.strip()

            current_path = tuple(item[1] for item in stack) + (key,)
            if value:
                if current_path == path:
                    return value.strip("'\"")
                continue

            stack.append((indent, key))

    return None


def infer_dataset_from_config(config_file: str) -> str | None:
    return infer_config_value(config_file, ("_run_meta", "dataset")) or infer_config_value(
        config_file, ("dataset",)
    )


def resolve_config_path(path: str | None) -> str | None:
    if path is None or path.lower() == "null":
        return None
    if os.path.isabs(path):
        return os.path.abspath(path)

    root_relative = os.path.abspath(os.path.join(LLAVA_ROOT, path))
    if os.path.exists(root_relative):
        return root_relative
    return os.path.abspath(path)


def infer_dataset_question_file(config_file: str, dataset: str) -> str | None:
    return resolve_config_path(infer_config_value(config_file, ("datasets", dataset, "question_file")))


def resolve_pope_question_file(config_file: str | None) -> str:
    if config_file is not None:
        question_file = infer_dataset_question_file(config_file, "pope")
        if question_file is not None:
            if not os.path.exists(question_file):
                raise FileNotFoundError(f"POPE question file not found from run config: {question_file}")
            return question_file
    return DEFAULT_POPE_QUESTION_FILE


def infer_run_metadata(run_dir: str) -> tuple[str, str, str, str]:
    run_dir = os.path.abspath(run_dir)
    answers_file = os.path.join(run_dir, "answers.jsonl")
    config_file = os.path.join(run_dir, "config.yaml")

    if not os.path.isfile(answers_file):
        raise FileNotFoundError(f"Run answers file not found: {answers_file}")
    if not os.path.isfile(config_file):
        raise FileNotFoundError(f"Run config not found: {config_file}")

    dataset = infer_dataset_from_config(config_file)
    if dataset not in {"gqa", "mme", "pope"}:
        raise ValueError(f"Unable to infer dataset from run config: {config_file}")

    output_dir = os.path.join(run_dir, "eval")
    return dataset, answers_file, output_dir, config_file


def eval_gqa(answers_file: str, output_dir: str) -> dict:
    data_dir = os.path.join(LLAVA_ROOT, "entropy_exp", "datasets", "gqa")
    convert_script = os.path.join(LLAVA_ROOT, "scripts", "convert_gqa_for_eval.py")
    eval_script = os.path.join(data_dir, "eval.py")

    if not os.path.exists(convert_script):
        raise FileNotFoundError(f"GQA convert script not found: {convert_script}")
    if not os.path.exists(eval_script):
        raise FileNotFoundError(f"GQA eval script not found: {eval_script}")

    pred_file = os.path.join(output_dir, "testdev_balanced_predictions.json")
    print(f"Converting: {answers_file} -> {pred_file}")
    subprocess.run(
        [
            sys.executable,
            convert_script,
            "--src",
            answers_file,
            "--dst",
            pred_file,
        ],
        check=True,
    )

    print("Running GQA evaluation...")
    result = subprocess.run(
        [
            sys.executable,
            eval_script,
            "--tier",
            "testdev_balanced",
            "--predictions",
            pred_file,
        ],
        cwd=data_dir,
        capture_output=True,
        text=True,
    )
    emit_process_output(result, os.path.join(output_dir, "stdout.txt"))

    return {
        "prediction_file": pred_file,
        "metrics": parse_gqa_metrics(result.stdout or ""),
    }


def eval_mme(answers_file: str, output_dir: str, mme_data_path: str | None = None) -> dict:
    eval_dir = os.path.join(LLAVA_ROOT, "entropy_exp", "eval_questions", "MME")
    convert_script = os.path.join(eval_dir, "convert_answer_to_mme.py")
    calc_script = os.path.join(eval_dir, "eval_tool", "calculation.py")

    if not os.path.exists(convert_script):
        raise FileNotFoundError(f"MME convert script not found: {convert_script}")
    if not os.path.exists(calc_script):
        raise FileNotFoundError(f"MME calculation script not found: {calc_script}")

    result_dir = ensure_dir(os.path.join(output_dir, "results"))

    print("Converting MME answers...")
    convert_cmd = [
        sys.executable,
        convert_script,
        "--experiment",
        os.path.basename(output_dir),
        "--answer-file",
        answers_file,
        "--result-dir",
        result_dir,
    ]
    supports_data_path = False
    with open(convert_script, "r", encoding="utf-8") as f:
        supports_data_path = "--data-path" in f.read()

    if mme_data_path and supports_data_path:
        convert_cmd.extend(["--data-path", mme_data_path])
    subprocess.run(convert_cmd, cwd=eval_dir, check=True)

    print("Running MME evaluation...")
    result = subprocess.run(
        [
            sys.executable,
            calc_script,
            "--results_dir",
            result_dir,
        ],
        cwd=os.path.join(eval_dir, "eval_tool"),
        capture_output=True,
        text=True,
    )
    emit_process_output(result, os.path.join(output_dir, "stdout.txt"))

    summary = {
        "results_dir": result_dir,
        "metrics": parse_mme_metrics(result.stdout or ""),
    }
    if mme_data_path:
        summary["mme_data_path"] = mme_data_path
    return summary


def eval_pope(answers_file: str, output_dir: str, question_file: str | None = None) -> dict:
    question_file = question_file or DEFAULT_POPE_QUESTION_FILE
    annotation_dir = os.path.join(LLAVA_ROOT, "entropy_exp", "datasets", "pope", "coco")

    if not os.path.exists(question_file):
        raise FileNotFoundError(f"POPE question file not found: {question_file}")
    if not os.path.isdir(annotation_dir):
        raise FileNotFoundError(f"POPE annotation directory not found: {annotation_dir}")

    print("Running POPE evaluation...")
    stdout, metrics = evaluate_pope_answers(
        answers=load_jsonl(answers_file),
        questions=load_jsonl(question_file),
        annotation_dir=annotation_dir,
    )

    stdout_path = os.path.join(output_dir, "stdout.txt")
    if stdout:
        print(stdout)
    macro_f1 = metrics.get("macro_f1")
    if macro_f1 is not None:
        macro_f1_output = format_pope_macro_f1(macro_f1)
        print(macro_f1_output)
        stdout = f"{stdout}\n{macro_f1_output}" if stdout else macro_f1_output

    write_text(stdout_path, stdout + ("\n" if stdout else ""))

    return {
        "metrics": metrics,
        "question_file": question_file,
        "annotation_dir": annotation_dir,
    }


EVAL_FUNCTIONS = {
    "gqa": eval_gqa,
    "mme": eval_mme,
    "pope": eval_pope,
}


def resolve_legacy_output_dir(dataset: str, answers_file: str, output_dir: str | None) -> str:
    if output_dir:
        return os.path.abspath(output_dir)
    answers_stem = os.path.splitext(os.path.basename(answers_file))[0]
    return os.path.join(LLAVA_ROOT, "entropy_exp", "outputs", "eval", dataset, answers_stem)


def main():
    parser = argparse.ArgumentParser(description="Run evaluation on inference outputs")
    parser.add_argument("--dataset", type=str, choices=["gqa", "mme", "pope"])
    parser.add_argument("--answers-file", type=str,
                        help="Path to the JSONL answers file from inference")
    parser.add_argument("--output-dir", type=str,
                        help="Directory to write evaluation artifacts")
    parser.add_argument("--mme-data-path", type=str,
                        help="Optional MME benchmark root directory override")
    parser.add_argument("--run-dir", type=str,
                        help="Run directory under entropy_exp/outputs/runs")
    args = parser.parse_args()

    if args.run_dir:
        if args.dataset or args.answers_file:
            parser.error("--run-dir cannot be combined with --dataset or --answers-file")
        dataset, answers_file, output_dir, config_file = infer_run_metadata(args.run_dir)
        run_dir = os.path.abspath(args.run_dir)
    else:
        if not args.dataset or not args.answers_file:
            parser.error("Either provide --run-dir, or provide both --dataset and --answers-file")
        dataset = args.dataset
        answers_file = os.path.abspath(args.answers_file)
        output_dir = resolve_legacy_output_dir(dataset, answers_file, args.output_dir)
        config_file = None
        run_dir = None

    if not os.path.exists(answers_file):
        print(f"Error: answers file not found: {answers_file}")
        sys.exit(1)

    output_dir = ensure_dir(os.path.abspath(output_dir))
    print(f"Writing evaluation artifacts to: {output_dir}")

    mme_data_path = None
    if dataset == "mme":
        mme_data_path = resolve_config_path(args.mme_data_path)
        if mme_data_path is None and config_file is not None:
            mme_data_path = resolve_config_path(
                infer_config_value(config_file, ("datasets", "mme", "image_folder"))
            )
        summary = eval_mme(answers_file, output_dir, mme_data_path=mme_data_path)
    elif dataset == "pope":
        if args.mme_data_path:
            parser.error("--mme-data-path can only be used with --dataset mme or an MME run directory")
        summary = eval_pope(
            answers_file,
            output_dir,
            question_file=resolve_pope_question_file(config_file),
        )
    else:
        if args.mme_data_path:
            parser.error("--mme-data-path can only be used with --dataset mme or an MME run directory")
        summary = EVAL_FUNCTIONS[dataset](answers_file, output_dir)
    summary.update(
        {
            "dataset": dataset,
            "answers_file": answers_file,
            "output_dir": output_dir,
        }
    )
    if run_dir is not None:
        summary["run_dir"] = run_dir
    if config_file is not None:
        summary["config_file"] = config_file

    summary_path = os.path.join(output_dir, "summary.json")
    write_json(summary_path, summary)
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
