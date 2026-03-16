"""Evaluation integration for legacy answers files and run-local outputs."""

import argparse
import json
import os
import re
import subprocess
import sys

LLAVA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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


def add_pope_weighted_average(metrics: dict) -> dict:
    category_metrics = {
        key: value
        for key, value in metrics.items()
        if isinstance(value, dict) and "samples" in value
    }
    if not category_metrics:
        return metrics

    metric_keys = ("accuracy", "precision", "recall", "f1_score", "yes_ratio")
    total_samples = sum(item["samples"] for item in category_metrics.values())
    if total_samples <= 0:
        return metrics

    weighted_average = {"samples": total_samples}
    for metric_key in metric_keys:
        weighted_sum = 0.0
        has_value = False
        for item in category_metrics.values():
            if metric_key not in item:
                continue
            weighted_sum += item[metric_key] * item["samples"]
            has_value = True
        if has_value:
            weighted_average[metric_key] = weighted_sum / total_samples

    metrics["weighted_average"] = weighted_average
    return metrics


def format_pope_weighted_average(weighted_average: dict) -> str:
    return "\n".join(
        [
            f"Category: weighted_average, # samples: {weighted_average['samples']}",
            f"Accuracy: {weighted_average['accuracy']}",
            f"Precision: {weighted_average['precision']}",
            f"Recall: {weighted_average['recall']}",
            f"F1 score: {weighted_average['f1_score']}",
            f"Yes ratio: {weighted_average['yes_ratio']}",
            "====================================",
        ]
    )


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
    if mme_data_path:
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

    metrics = add_pope_weighted_average(parse_pope_metrics(result.stdout or ""))
    weighted_average = metrics.get("weighted_average")
    if weighted_average:
        weighted_output = format_pope_weighted_average(weighted_average)
        print(weighted_output)
        stdout_path = os.path.join(output_dir, "stdout.txt")
        with open(stdout_path, "a", encoding="utf-8") as f:
            if result.stdout and not result.stdout.endswith("\n"):
                f.write("\n")
            f.write(weighted_output)
            f.write("\n")

    return {
        "metrics": metrics,
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
