"""
Evaluation integration: run existing eval scripts on inference outputs.
"""

import argparse
import os
import sys
import subprocess
import json

LLAVA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def eval_gqa(answers_file: str):
    """Run GQA evaluation using existing eval pipeline."""
    data_dir = os.path.join(LLAVA_ROOT, "entropy_exp", "datasets", "gqa")
    convert_script = os.path.join(LLAVA_ROOT, "scripts", "convert_gqa_for_eval.py")

    # Convert answers to GQA format
    pred_file = os.path.join(data_dir, "testdev_balanced_predictions.json")
    print(f"Converting: {answers_file} → {pred_file}")
    subprocess.run([
        sys.executable, convert_script,
        "--src", answers_file,
        "--dst", pred_file
    ], check=True)

    # Run GQA eval
    eval_script = os.path.join(data_dir, "eval", "eval.py")
    if os.path.exists(eval_script):
        print("Running GQA evaluation...")
        result = subprocess.run(
            [sys.executable, eval_script, "--tier", "testdev_balanced"],
            cwd=data_dir,
            capture_output=True, text=True
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
    else:
        print(f"  Warning: eval script not found at {eval_script}")


def eval_mme(answers_file: str):
    """Run MME evaluation using existing eval pipeline."""
    eval_dir = os.path.join(LLAVA_ROOT, "entropy_exp", "eval_questions", "MME")
    convert_script = os.path.join(eval_dir, "convert_answer_to_mme.py")

    if not os.path.exists(convert_script):
        print(f"  Warning: MME convert script not found at {convert_script}")
        return

    # Create temporary experiment directory for MME
    exp_name = "entropy_exp"
    answers_dir = os.path.join(eval_dir, "answers", exp_name)
    os.makedirs(answers_dir, exist_ok=True)

    # Copy answers to expected location
    import shutil
    dst = os.path.join(answers_dir, os.path.basename(answers_file))
    shutil.copy2(answers_file, dst)

    print("Converting MME answers...")
    subprocess.run(
        [sys.executable, convert_script, "--experiment", exp_name],
        cwd=eval_dir,
        check=True
    )

    # Run MME calculation
    calc_script = os.path.join(eval_dir, "eval_tool", "calculation.py")
    if os.path.exists(calc_script):
        print("Running MME evaluation...")
        result = subprocess.run(
            [sys.executable, calc_script, "--results_dir", f"answers/{exp_name}"],
            cwd=os.path.join(eval_dir, "eval_tool"),
            capture_output=True, text=True
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr)


def eval_pope(answers_file: str):
    """Run POPE evaluation."""
    eval_script = os.path.join(LLAVA_ROOT, "llava", "eval", "eval_pope.py")
    question_file = os.path.join(LLAVA_ROOT, "entropy_exp", "eval_questions", "pope", "llava_pope_test.jsonl")
    annotation_dir = os.path.join(LLAVA_ROOT, "entropy_exp", "datasets", "pope", "coco")

    if not os.path.exists(eval_script):
        print(f"  Warning: POPE eval script not found at {eval_script}")
        return

    print("Running POPE evaluation...")
    result = subprocess.run(
        [sys.executable, eval_script,
         "--annotation-dir", annotation_dir,
         "--question-file", question_file,
         "--result-file", answers_file],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)


EVAL_FUNCTIONS = {
    "gqa": eval_gqa,
    "mme": eval_mme,
    "pope": eval_pope,
}


def main():
    parser = argparse.ArgumentParser(description="Run evaluation on inference outputs")
    parser.add_argument("--dataset", type=str, required=True, choices=["gqa", "mme", "pope"])
    parser.add_argument("--answers-file", type=str, required=True,
                        help="Path to the JSONL answers file from inference")
    args = parser.parse_args()

    if not os.path.exists(args.answers_file):
        print(f"Error: answers file not found: {args.answers_file}")
        sys.exit(1)

    EVAL_FUNCTIONS[args.dataset](args.answers_file)


if __name__ == "__main__":
    main()
