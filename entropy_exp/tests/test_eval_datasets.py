import os
import sys
import tempfile
import unittest
import json


SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC_DIR)

from eval_datasets import (
    DEFAULT_POPE_QUESTION_FILE,
    add_pope_macro_f1,
    evaluate_pope_answers,
    format_pope_macro_f1,
    parse_pope_metrics,
    resolve_pope_question_file,
)
from summarize_results import primary_metric


class PopeEvalDatasetsTests(unittest.TestCase):
    def test_parse_pope_metrics_extracts_category_metrics(self):
        stdout = """
Category: popular, # samples: 3000
Accuracy: 0.86
Precision: 0.95
Recall: 0.78
F1 score: 0.85
Yes ratio: 0.41
====================================
Category: adversarial, # samples: 3000
Accuracy: 0.84
Precision: 0.90
Recall: 0.77
F1 score: 0.83
Yes ratio: 0.43
====================================
""".strip()

        metrics = parse_pope_metrics(stdout)
        self.assertEqual(metrics["popular"]["samples"], 3000)
        self.assertAlmostEqual(metrics["popular"]["f1_score"], 0.85)
        self.assertEqual(metrics["adversarial"]["samples"], 3000)
        self.assertAlmostEqual(metrics["adversarial"]["precision"], 0.90)

    def test_add_pope_macro_f1_uses_simple_mean_over_categories(self):
        metrics = {
            "popular": {"samples": 3000, "f1_score": 0.85},
            "adversarial": {"samples": 3000, "f1_score": 0.83},
            "random": {"samples": 2910, "f1_score": 0.87},
        }

        enriched = add_pope_macro_f1(metrics)
        self.assertAlmostEqual(enriched["macro_f1"], (0.85 + 0.83 + 0.87) / 3.0)
        self.assertNotIn("weighted_average", enriched)

    def test_add_pope_macro_f1_is_category_count_agnostic(self):
        metrics = {
            "cat_a": {"samples": 10, "f1_score": 0.20},
            "cat_b": {"samples": 20, "f1_score": 0.40},
        }
        enriched = add_pope_macro_f1(metrics)
        self.assertAlmostEqual(enriched["macro_f1"], 0.30)

    def test_format_pope_macro_f1(self):
        self.assertEqual(format_pope_macro_f1(0.8530153837), "Macro-F1: 0.853015")

    def test_primary_metric_prefers_macro_f1_for_pope(self):
        name, value = primary_metric("pope", {"macro_f1": 0.8123})
        self.assertEqual(name, "macro_f1")
        self.assertAlmostEqual(value, 0.8123)

    def test_primary_metric_falls_back_to_legacy_weighted_average(self):
        name, value = primary_metric("pope", {"weighted_average": {"f1_score": 0.799}})
        self.assertEqual(name, "macro_f1")
        self.assertAlmostEqual(value, 0.799)

    def test_resolve_pope_question_file_prefers_run_local_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.yaml")
            with open(config_path, "w", encoding="utf-8") as handle:
                handle.write(
                    "datasets:\n"
                    "  pope:\n"
                    "    question_file: entropy_exp/eval_questions/pope/llava_pope_smoke_9.jsonl\n"
                )

            resolved = resolve_pope_question_file(config_path)
            self.assertTrue(resolved.endswith("entropy_exp/eval_questions/pope/llava_pope_smoke_9.jsonl"))

    def test_resolve_pope_question_file_falls_back_to_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.yaml")
            with open(config_path, "w", encoding="utf-8") as handle:
                handle.write("datasets:\n  pope:\n    image_folder: entropy_exp/datasets/pope/val2014\n")

            self.assertEqual(resolve_pope_question_file(config_path), DEFAULT_POPE_QUESTION_FILE)

    def test_resolve_pope_question_file_raises_for_missing_run_local_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.yaml")
            with open(config_path, "w", encoding="utf-8") as handle:
                handle.write("datasets:\n  pope:\n    question_file: entropy_exp/eval_questions/pope/missing.jsonl\n")

            with self.assertRaises(FileNotFoundError):
                resolve_pope_question_file(config_path)

    def test_evaluate_pope_answers_supports_single_category_subset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            annotation_dir = os.path.join(tmpdir, "coco")
            os.makedirs(annotation_dir, exist_ok=True)
            label_path = os.path.join(annotation_dir, "coco_pope_adversarial.json")
            with open(label_path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps({"question_id": 1, "label": "yes"}) + "\n")
                handle.write(json.dumps({"question_id": 2, "label": "no"}) + "\n")

            questions = [
                {"question_id": 1, "category": "adversarial"},
                {"question_id": 2, "category": "adversarial"},
            ]
            answers = [
                {"question_id": 1, "text": "Yes."},
                {"question_id": 2, "text": "No"},
            ]

            stdout, metrics = evaluate_pope_answers(answers, questions, annotation_dir)
            self.assertIn("Category: adversarial, # samples: 2", stdout)
            self.assertAlmostEqual(metrics["adversarial"]["f1_score"], 1.0)
            self.assertAlmostEqual(metrics["macro_f1"], 1.0)

    def test_evaluate_pope_answers_supports_category_local_annotation_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            annotation_dir = os.path.join(tmpdir, "coco")
            os.makedirs(annotation_dir, exist_ok=True)
            for category in ("adversarial", "random"):
                label_path = os.path.join(annotation_dir, f"coco_pope_{category}.json")
                with open(label_path, "w", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "question_id": 1,
                                "image": "img.jpg",
                                "text": "Is there a cat in the image?",
                                "label": "yes",
                            }
                        )
                        + "\n"
                    )

            questions = [
                {
                    "question_id": 1,
                    "image": "img.jpg",
                    "text": "Is there a cat in the image?\nAnswer the question using a single word or phrase.",
                    "category": "adversarial",
                },
                {
                    "question_id": 10000001,
                    "image": "img.jpg",
                    "text": "Is there a cat in the image?\nAnswer the question using a single word or phrase.",
                    "category": "random",
                },
            ]
            answers = [
                {"question_id": 1, "text": "Yes"},
                {"question_id": 10000001, "text": "Yes"},
            ]

            stdout, metrics = evaluate_pope_answers(answers, questions, annotation_dir)
            self.assertIn("Category: adversarial, # samples: 1", stdout)
            self.assertIn("Category: random, # samples: 1", stdout)
            self.assertAlmostEqual(metrics["adversarial"]["f1_score"], 1.0)
            self.assertAlmostEqual(metrics["random"]["f1_score"], 1.0)
            self.assertAlmostEqual(metrics["macro_f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
