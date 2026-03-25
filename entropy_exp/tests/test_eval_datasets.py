import os
import sys
import unittest


SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC_DIR)

from eval_datasets import add_pope_macro_f1, format_pope_macro_f1, parse_pope_metrics
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


if __name__ == "__main__":
    unittest.main()
