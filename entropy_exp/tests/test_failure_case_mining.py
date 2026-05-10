import os
import sys
import unittest


SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC_DIR)

from failure_case_mining import (  # noqa: E402
    EvalResult,
    canonicalize_patch_indices,
    classify_case,
    compute_patch_diff_metrics,
    entropy_norm_from_scores,
    normalize_pope_answer,
)


class FailureCaseMiningTests(unittest.TestCase):
    def test_normalize_pope_answer_matches_yes_no_rule(self):
        self.assertEqual(normalize_pope_answer("No, there is not."), "no")
        self.assertEqual(normalize_pope_answer("There is not a backpack."), "no")
        self.assertEqual(normalize_pope_answer("Yes, it is visible."), "yes")
        self.assertEqual(normalize_pope_answer("A snowboard"), "yes")

    def test_classify_case_identifies_sv2_only_success(self):
        sv2 = EvalResult("1", 1.0, True, "yes", "yes", "yes", "Q", "img.jpg", "popular")
        ours = EvalResult("1", 0.0, False, "no", "no", "yes", "Q", "img.jpg", "popular")
        self.assertEqual(classify_case(sv2, ours), "sv2_correct_ours_wrong")

    def test_entropy_norm_edges_and_uniform(self):
        self.assertAlmostEqual(entropy_norm_from_scores([1.0, 1.0, 1.0, 1.0]), 1.0)
        self.assertAlmostEqual(entropy_norm_from_scores([1.0, 0.0, 0.0, 0.0]), 0.0)
        self.assertAlmostEqual(entropy_norm_from_scores([0.0, 0.0, 0.0, 0.0]), 0.0)

    def test_compute_patch_diff_metrics_uses_keep_and_low_compensation(self):
        sv2_stats = {
            "layer_2_keep_patch_indices": [0, 1, 2, 3],
            "layer_2_importance": [1.0, 1.0, 1.0, 1.0],
            "layer_2_current_patch_indices": [0, 1, 2, 3],
        }
        ours_stats = {
            "layer_2_keep_patch_indices": [0, 1, 4, 5],
            "layer_2_low_keep_patch_indices": [4, 5],
            "layer_2_importance": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "layer_2_current_patch_indices": [0, 1, 2, 3, 4, 5],
        }
        metrics = compute_patch_diff_metrics(sv2_stats, ours_stats, layers=(2,))
        self.assertAlmostEqual(metrics["max_sv2_only_keep_ratio"], 0.5)
        self.assertAlmostEqual(metrics["max_ours_only_keep_ratio"], 0.5)
        self.assertAlmostEqual(metrics["max_ours_low_keep_ratio"], 0.5)
        self.assertGreater(metrics["max_sv2_only_saliency_mass"], 0.0)
        self.assertGreater(metrics["patch_diff_score"], metrics["max_sv2_only_keep_ratio"])

    def test_canonicalize_patch_indices_reconstructs_sv2_local_indices(self):
        sv2_stats = {
            "layer_2_before": 8,
            "layer_2_keep_indices": [2, 4, 6],
            "layer_2_pruned_indices": [0, 1, 3, 5, 7],
            "layer_2_keep_patch_indices": [2, 4, 6],
            "layer_6_keep_indices": [0, 2],
            "layer_6_pruned_indices": [1],
            "layer_6_keep_patch_indices": [0, 2],
            "layer_15_keep_indices": [1],
            "layer_15_pruned_indices": [0],
            "layer_15_keep_patch_indices": [1],
        }

        canonical = canonicalize_patch_indices(sv2_stats, layers=(2, 6, 15), patch_per_row=3)

        self.assertEqual(canonical["layer_2_current_patch_indices"], list(range(8)))
        self.assertEqual(canonical["layer_2_keep_patch_indices"], [2, 4, 6])
        self.assertEqual(canonical["layer_6_current_patch_indices"], [2, 4, 6])
        self.assertEqual(canonical["layer_6_keep_patch_indices"], [2, 6])
        self.assertEqual(canonical["layer_15_current_patch_indices"], [2, 6])
        self.assertEqual(canonical["layer_15_keep_patch_indices"], [6])
        self.assertTrue(
            set(canonical["layer_15_keep_patch_indices"])
            <= set(canonical["layer_6_keep_patch_indices"])
            <= set(canonical["layer_2_keep_patch_indices"])
        )

    def test_compute_patch_diff_metrics_canonicalizes_sv2_before_comparison(self):
        sv2_stats = {
            "layer_2_before": 8,
            "layer_2_keep_indices": [2, 4, 6],
            "layer_6_keep_indices": [0, 2],
            "layer_6_keep_patch_indices": [0, 2],
        }
        ours_stats = {
            "layer_2_current_patch_indices": list(range(8)),
            "layer_2_keep_indices": [2, 4, 6],
            "layer_2_keep_patch_indices": [2, 4, 6],
            "layer_6_current_patch_indices": [2, 4, 6],
            "layer_6_keep_indices": [0, 1],
            "layer_6_keep_patch_indices": [2, 4],
            "layer_6_low_keep_patch_indices": [4],
            "layer_6_importance": [1.0, 2.0, 3.0],
        }

        metrics = compute_patch_diff_metrics(sv2_stats, ours_stats, layers=(6,))

        self.assertAlmostEqual(metrics["max_sv2_only_keep_ratio"], 0.5)
        self.assertGreater(metrics["max_sv2_only_saliency_mass"], 0.0)


if __name__ == "__main__":
    unittest.main()
