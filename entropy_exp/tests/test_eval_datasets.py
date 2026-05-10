import json
import importlib.util
import os
import sys
import tempfile
import unittest


ENTROPY_EXP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LLAVA_ROOT = os.path.dirname(ENTROPY_EXP_DIR)
SRC_DIR = os.path.join(ENTROPY_EXP_DIR, "src")
sys.path.insert(0, SRC_DIR)

from eval_datasets import (
    add_pope_macro_f1,
    format_pope_macro_f1,
    load_scienceqa_metrics,
    parse_pope_metrics,
    parse_textvqa_metrics,
    unsupported_local_metric_message,
)
from summarize_results import primary_metric


class PopeEvalDatasetsTests(unittest.TestCase):
    def load_llava_pope_eval_module(self):
        module_path = os.path.join(LLAVA_ROOT, "llava", "eval", "eval_pope.py")
        spec = importlib.util.spec_from_file_location("llava_eval_pope_for_tests", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

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

    def test_llava_pope_question_lookup_accepts_int_and_string_ids(self):
        eval_pope_module = self.load_llava_pope_eval_module()
        question_lookup = eval_pope_module.build_question_lookup(
            [
                {"question_id": 1, "category": "popular"},
                {"question_id": "2", "category": "adversarial"},
            ]
        )

        self.assertEqual(eval_pope_module.get_question_category(question_lookup, "1"), "popular")
        self.assertEqual(eval_pope_module.get_question_category(question_lookup, 2), "adversarial")


class NewDatasetEvalTests(unittest.TestCase):
    def test_parse_textvqa_metrics_extracts_accuracy_percent(self):
        stdout = "demo-model\nSamples: 5000\nAccuracy: 63.42%\n"
        metrics = parse_textvqa_metrics(stdout)
        self.assertEqual(metrics, {"accuracy": 63.42})

    def test_load_scienceqa_metrics_prefers_result_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result_file = os.path.join(temp_dir, "scienceqa_result.json")
            with open(result_file, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "acc": 78.5,
                        "correct": 1570,
                        "count": 2000,
                        "results": {},
                        "outputs": {},
                    },
                    handle,
                )

            metrics = load_scienceqa_metrics(result_file)
            self.assertEqual(metrics["accuracy"], 78.5)
            self.assertEqual(metrics["correct"], 1570)
            self.assertEqual(metrics["count"], 2000)

    def test_unsupported_local_metric_message_for_mmbench_is_clear(self):
        message = unsupported_local_metric_message("mmbench")
        self.assertIn("mmbench", message)
        self.assertIn("inference input compatibility", message)
        self.assertIn("local final-metric evaluation", message)


if __name__ == "__main__":
    unittest.main()
