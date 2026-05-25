"""Evaluation integration for legacy answers files and run-local outputs."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys

LLAVA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOCAL_EVAL_DATASETS = {"gqa", "mme", "pope", "textvqa", "scienceqa"}
INFERENCE_ONLY_DATASETS = {"mmbench"}


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


def parse_textvqa_metrics(stdout: str) -> dict:
    match = re.search(r"Accuracy:\s*(-?\d+(?:\.\d+)?)%", stdout)
    if not match:
        return {}
    return {"accuracy": float(match.group(1))}


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


def load_scienceqa_metrics(result_file: str) -> dict:
    with open(result_file, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return {
        "accuracy": payload.get("acc"),
        "correct": payload.get("correct"),
        "count": payload.get("count"),
    }


def unsupported_local_metric_message(dataset: str) -> str:
    return (
        f"Dataset '{dataset}' currently supports inference input compatibility only. "
        "This repo does not provide a local final-metric evaluation path for it."
    )


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize_open_answer(text: str) -> str:
    return str(text).strip().lower().rstrip(".")


def is_subset_answers_file(answers_file: str, full_count: int) -> bool:
    return len(load_jsonl(answers_file)) < int(full_count)


def load_textvqa_evaluator():
    evaluator_path = os.path.join(LLAVA_ROOT, "llava", "eval", "m4c_evaluator.py")
    spec = importlib.util.spec_from_file_location("textvqa_evaluator_local", evaluator_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load TextVQA evaluator from: {evaluator_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TextVQAAccuracyEvaluator()


def eval_gqa_subset(answers_file: str, output_dir: str) -> dict:
    question_file = os.path.join(LLAVA_ROOT, "entropy_exp", "datasets", "gqa", "testdev_balanced_questions.json")
    questions = load_json(question_file)
    answers = load_jsonl(answers_file)

    details = []
    correct = 0
    for answer in answers:
        question_id = str(answer["question_id"])
        question = questions[question_id]
        pred_text = normalize_open_answer(answer.get("text", ""))
        gt_text = normalize_open_answer(question.get("answer", ""))
        is_correct = pred_text == gt_text
        correct += int(is_correct)
        details.append(
            {
                "question_id": question_id,
                "prediction": pred_text,
                "answer": gt_text,
                "correct": is_correct,
            }
        )

    total = len(details)
    accuracy = (correct / total * 100.0) if total else 0.0
    stdout_text = (
        f"Subset GQA evaluation\n"
        f"Samples: {total}\n"
        f"Correct: {correct}\n"
        f"Accuracy: {accuracy:.2f}\n"
    )
    print(stdout_text, end="")
    write_text(os.path.join(output_dir, "stdout.txt"), stdout_text)
    analysis_file = os.path.join(output_dir, "subset_analysis.json")
    write_json(analysis_file, {"results": details})
    return {
        "question_file": question_file,
        "analysis_file": analysis_file,
        "subset_evaluation": True,
        "metrics": {
            "accuracy": accuracy,
            "correct": correct,
            "count": total,
        },
    }


def normalize_pope_answer(text: str) -> int:
    first_sentence = str(text).split(".")[0].replace(",", "")
    words = first_sentence.split()
    return 0 if ("No" in words or "not" in words or "no" in words) else 1


def compute_binary_metrics(pred_list: list[int], label_list: list[int]) -> dict:
    tp = fp = tn = fn = 0
    for pred, label in zip(pred_list, label_list):
        if pred == 1 and label == 1:
            tp += 1
        elif pred == 1 and label == 0:
            fp += 1
        elif pred == 0 and label == 0:
            tn += 1
        elif pred == 0 and label == 1:
            fn += 1
    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / total if total else 0.0
    yes_ratio = pred_list.count(1) / len(pred_list) if pred_list else 0.0
    return {
        "samples": total,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "yes_ratio": yes_ratio,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def eval_pope_subset(answers_file: str, output_dir: str) -> dict:
    question_file = os.path.join(LLAVA_ROOT, "entropy_exp", "eval_questions", "pope", "llava_pope_test.jsonl")
    annotation_dir = os.path.join(LLAVA_ROOT, "entropy_exp", "datasets", "pope", "coco")
    questions = {str(item["question_id"]): item for item in load_jsonl(question_file)}
    answers = load_jsonl(answers_file)

    labels_by_category: dict[str, dict[str, int]] = {}
    for filename in os.listdir(annotation_dir):
        if not filename.startswith("coco_pope_") or not filename.endswith(".json"):
            continue
        category = filename[10:-5]
        labels_by_category[category] = {
            str(item["question_id"]): 1 if item["label"] == "yes" else 0
            for item in load_jsonl(os.path.join(annotation_dir, filename))
        }

    grouped_predictions: dict[str, list[tuple[int, int]]] = {}
    for answer in answers:
        question_id = str(answer["question_id"])
        category = questions[question_id]["category"]
        label = labels_by_category[category][question_id]
        pred = normalize_pope_answer(answer.get("text", ""))
        grouped_predictions.setdefault(category, []).append((pred, label))

    metrics = {}
    stdout_parts = ["Subset POPE evaluation"]
    for category in sorted(grouped_predictions):
        pairs = grouped_predictions[category]
        pred_list = [pred for pred, _label in pairs]
        label_list = [label for _pred, label in pairs]
        metrics[category] = compute_binary_metrics(pred_list, label_list)
        stdout_parts.append(f"Category: {category}, # samples: {metrics[category]['samples']}")
        stdout_parts.append(f"Accuracy: {metrics[category]['accuracy']}")
        stdout_parts.append(f"Precision: {metrics[category]['precision']}")
        stdout_parts.append(f"Recall: {metrics[category]['recall']}")
        stdout_parts.append(f"F1 score: {metrics[category]['f1_score']}")
        stdout_parts.append(f"Yes ratio: {metrics[category]['yes_ratio']}")
        stdout_parts.append("====================================")
    metrics = add_pope_macro_f1(metrics)
    if "macro_f1" in metrics:
        stdout_parts.append(format_pope_macro_f1(metrics["macro_f1"]))
    stdout_text = "\n".join(stdout_parts) + "\n"
    print(stdout_text, end="")
    write_text(os.path.join(output_dir, "stdout.txt"), stdout_text)
    return {
        "question_file": question_file,
        "annotation_dir": annotation_dir,
        "subset_evaluation": True,
        "metrics": metrics,
    }


def textvqa_prompt_processor(prompt: str) -> str:
    if prompt.startswith("OCR tokens: "):
        pattern = r"Question: (.*?) Short answer:"
        match = re.search(pattern, prompt, re.DOTALL)
        if match is None:
            raise ValueError(f"Unsupported TextVQA prompt: {prompt!r}")
        question = match.group(1)
    elif "Reference OCR token: " in prompt and len(prompt.split("\n")) == 3:
        if prompt.startswith("Reference OCR token:"):
            question = prompt.split("\n")[1]
        else:
            question = prompt.split("\n")[0]
    elif len(prompt.split("\n")) == 2:
        question = prompt.split("\n")[0]
    else:
        raise ValueError(f"Unsupported TextVQA prompt: {prompt!r}")
    return question.lower()


def parse_scienceqa_answer(pred_text: str, options: list[str]) -> str:
    if pred_text in options:
        return pred_text
    if len(pred_text) >= 3 and pred_text[0] in options and pred_text[1:3] == ". ":
        return pred_text[0]
    pattern = re.compile(r"The answer is ([A-Z]).")
    matches = pattern.findall(pred_text)
    if len(matches) == 1:
        return matches[0]
    return "FAILED"


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


def infer_run_max_samples(config_file: str | None) -> int | None:
    if config_file is None:
        return None
    raw_value = infer_config_value(config_file, ("_run_meta", "max_samples"))
    if raw_value is None:
        raw_value = infer_config_value(config_file, ("pruning", "max_samples"))
    if raw_value is None or raw_value.lower() in {"null", "none", "~"}:
        return None
    return int(raw_value)


def resolve_config_path(path: str | None) -> str | None:
    if path is None or path.lower() == "null":
        return None
    if os.path.isabs(path):
        return os.path.abspath(path)

    root_relative = os.path.abspath(os.path.join(LLAVA_ROOT, path))
    if os.path.exists(root_relative):
        return root_relative
    return os.path.abspath(path)


def infer_run_metadata(run_dir: str) -> tuple[str, str, str, str]:
    run_dir = os.path.abspath(run_dir)
    answers_file = os.path.join(run_dir, "answers.jsonl")
    config_file = os.path.join(run_dir, "config.yaml")

    if not os.path.isfile(answers_file):
        raise FileNotFoundError(f"Run answers file not found: {answers_file}")
    if not os.path.isfile(config_file):
        raise FileNotFoundError(f"Run config not found: {config_file}")

    dataset = infer_dataset_from_config(config_file)
    if dataset not in LOCAL_EVAL_DATASETS | INFERENCE_ONLY_DATASETS:
        raise ValueError(f"Unable to infer dataset from run config: {config_file}")

    output_dir = os.path.join(run_dir, "eval")
    return dataset, answers_file, output_dir, config_file


def eval_gqa(answers_file: str, output_dir: str) -> dict:
    data_dir = os.path.join(LLAVA_ROOT, "entropy_exp", "datasets", "gqa")
    convert_script = os.path.join(LLAVA_ROOT, "scripts", "convert_gqa_for_eval.py")
    eval_script = os.path.join(data_dir, "eval.py")
    question_file = os.path.join(data_dir, "testdev_balanced_questions.json")

    if not os.path.exists(convert_script):
        raise FileNotFoundError(f"GQA convert script not found: {convert_script}")
    if not os.path.exists(eval_script):
        raise FileNotFoundError(f"GQA eval script not found: {eval_script}")
    if is_subset_answers_file(answers_file, len(load_json(question_file))):
        return eval_gqa_subset(answers_file, output_dir)

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


def eval_pope(answers_file: str, output_dir: str) -> dict:
    eval_script = os.path.join(LLAVA_ROOT, "llava", "eval", "eval_pope.py")
    question_file = os.path.join(LLAVA_ROOT, "entropy_exp", "eval_questions", "pope", "llava_pope_test.jsonl")
    annotation_dir = os.path.join(LLAVA_ROOT, "entropy_exp", "datasets", "pope", "coco")

    if not os.path.exists(eval_script):
        raise FileNotFoundError(f"POPE eval script not found: {eval_script}")
    if is_subset_answers_file(answers_file, len(load_jsonl(question_file))):
        return eval_pope_subset(answers_file, output_dir)

    print("Running POPE evaluation...")
    result = subprocess.run(
        [
            sys.executable,
            eval_script,
            "--annotation-dir",
            annotation_dir,
            "--question-file",
            question_file,
            "--result-file",
            answers_file,
        ],
        capture_output=True,
        text=True,
    )
    emit_process_output(result, os.path.join(output_dir, "stdout.txt"))

    metrics = add_pope_macro_f1(parse_pope_metrics(result.stdout or ""))
    macro_f1 = metrics.get("macro_f1")
    if macro_f1 is not None:
        macro_f1_output = format_pope_macro_f1(macro_f1)
        print(macro_f1_output)
        stdout_path = os.path.join(output_dir, "stdout.txt")
        with open(stdout_path, "a", encoding="utf-8") as f:
            if result.stdout and not result.stdout.endswith("\n"):
                f.write("\n")
            f.write(macro_f1_output)
            f.write("\n")

    return {
        "metrics": metrics,
    }


def eval_textvqa(answers_file: str, output_dir: str) -> dict:
    annotation_file = os.path.join(
        LLAVA_ROOT,
        "entropy_exp",
        "eval_questions",
        "textvqa",
        "TextVQA_0.5.1_val.json",
    )
    annotations = load_json(annotation_file)["data"]
    annotation_map = {
        (annotation["image_id"], annotation["question"].lower()): annotation
        for annotation in annotations
    }
    results = load_jsonl(answers_file)
    pred_list = []
    for result in results:
        annotation = annotation_map[(result["question_id"], textvqa_prompt_processor(result["prompt"]))]
        pred_list.append(
            {
                "pred_answer": result["text"],
                "gt_answers": annotation["answers"],
            }
        )

    evaluator = load_textvqa_evaluator()
    accuracy = 100.0 * evaluator.eval_pred_list(pred_list)
    stdout_text = f"Samples: {len(pred_list)}\nAccuracy: {accuracy:.2f}%\n"
    print(stdout_text, end="")
    write_text(os.path.join(output_dir, "stdout.txt"), stdout_text)
    return {
        "annotation_file": annotation_file,
        "metrics": {"accuracy": accuracy},
    }


def eval_scienceqa(answers_file: str, output_dir: str, max_samples: int | None = None) -> dict:
    base_dir = os.path.join(LLAVA_ROOT, "entropy_exp", "eval_questions", "scienceqa")
    analysis_file = os.path.join(output_dir, "analysis.json")
    result_file = os.path.join(output_dir, "result.json")
    pid_splits = load_json(os.path.join(base_dir, "pid_splits.json"))
    problems = load_json(os.path.join(base_dir, "problems.json"))
    predictions = {str(pred["question_id"]): pred for pred in load_jsonl(answers_file)}
    split_indices = [str(idx) for idx in pid_splits["test"]]
    if max_samples is not None:
        split_indices = split_indices[:max_samples]
    options = ["A", "B", "C", "D", "E"]

    results = {"correct": [], "incorrect": []}
    result_payload = {
        "acc": None,
        "correct": None,
        "count": None,
        "results": {},
        "outputs": {},
    }

    for prob_id in split_indices:
        prob = problems[prob_id]
        pred = predictions.get(prob_id, {"text": "FAILED", "prompt": "Unknown"})
        pred_text = pred["text"]
        answer = parse_scienceqa_answer(pred_text, options)
        pred_idx = options.index(answer) if answer in options[: len(prob["choices"])] else -1

        analysis = {
            "question_id": prob_id,
            "parsed_ans": answer,
            "ground_truth": options[prob["answer"]],
            "question": pred["prompt"],
            "pred": pred_text,
        }
        result_payload["results"][prob_id] = pred_idx
        result_payload["outputs"][prob_id] = pred_text
        if pred_idx == prob["answer"]:
            results["correct"].append(analysis)
        else:
            results["incorrect"].append(analysis)

    correct = len(results["correct"])
    total = correct + len(results["incorrect"])
    accuracy = (correct / total * 100.0) if total else 0.0
    result_payload["acc"] = accuracy
    result_payload["correct"] = correct
    result_payload["count"] = total

    write_json(analysis_file, results)
    write_json(result_file, result_payload)
    stdout_text = f"Total: {total}, Correct: {correct}, Accuracy: {accuracy:.2f}%\n"
    print(stdout_text, end="")
    write_text(os.path.join(output_dir, "stdout.txt"), stdout_text)
    return {
        "base_dir": base_dir,
        "analysis_file": analysis_file,
        "result_file": result_file,
        "metrics": load_scienceqa_metrics(result_file),
    }


EVAL_FUNCTIONS = {
    "gqa": eval_gqa,
    "mme": eval_mme,
    "pope": eval_pope,
    "textvqa": eval_textvqa,
    "scienceqa": eval_scienceqa,
}


def resolve_legacy_output_dir(dataset: str, answers_file: str, output_dir: str | None) -> str:
    if output_dir:
        return os.path.abspath(output_dir)
    answers_stem = os.path.splitext(os.path.basename(answers_file))[0]
    return os.path.join(LLAVA_ROOT, "entropy_exp", "outputs", "eval", dataset, answers_stem)


def main():
    parser = argparse.ArgumentParser(description="Run evaluation on inference outputs")
    parser.add_argument("--dataset", type=str, choices=sorted(LOCAL_EVAL_DATASETS | INFERENCE_ONLY_DATASETS))
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

    if dataset in INFERENCE_ONLY_DATASETS:
        print(unsupported_local_metric_message(dataset))
        return 0

    run_max_samples = infer_run_max_samples(config_file)
    mme_data_path = None
    if dataset == "mme":
        mme_data_path = resolve_config_path(args.mme_data_path)
        if mme_data_path is None and config_file is not None:
            mme_data_path = resolve_config_path(
                infer_config_value(config_file, ("datasets", "mme", "image_folder"))
            )
        summary = eval_mme(answers_file, output_dir, mme_data_path=mme_data_path)
    else:
        if args.mme_data_path:
            parser.error("--mme-data-path can only be used with --dataset mme or an MME run directory")
        if dataset == "scienceqa":
            summary = eval_scienceqa(answers_file, output_dir, max_samples=run_max_samples)
        else:
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
