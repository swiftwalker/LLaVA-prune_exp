import os
import sys
import unittest


SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC_DIR)

from prune_inference import append_layer_stats_fields  # noqa: E402


class PruneInferenceStatsTests(unittest.TestCase):
    def test_append_layer_stats_fields_records_baseline_reference_keep_indices(self):
        sample_stats = {}
        full_keep = list(range(576))

        for layer_idx in (1, 2, 3):
            append_layer_stats_fields(
                sample_stats,
                layer_idx,
                {
                    "prune_ratio": 0.0,
                    "num_visual_before": 576,
                    "num_visual_after": 576,
                    "num_pruned": 0,
                    "keep_indices": full_keep,
                    "pruned_indices": [],
                },
                save_importance=False,
                save_indices=True,
            )

        for layer_idx in (1, 2, 3):
            keep_key = f"layer_{layer_idx}_keep_indices"
            pruned_key = f"layer_{layer_idx}_pruned_indices"
            self.assertEqual(len(sample_stats[keep_key]), 576)
            self.assertEqual(sample_stats[keep_key][0], 0)
            self.assertEqual(sample_stats[keep_key][-1], 575)
            self.assertEqual(sample_stats[pruned_key], [])

    def test_append_layer_stats_fields_records_adaptive_patch_metadata(self):
        sample_stats = {}
        append_layer_stats_fields(
            sample_stats,
            2,
            {
                "prune_ratio": 0.4,
                "num_visual_before": 576,
                "num_visual_after": 346,
                "num_pruned": 230,
                "keep_indices": [0, 1, 2, 3],
                "pruned_indices": [4, 5],
                "keep_patch_indices": [0, 1, 2, 3],
                "pruned_patch_indices": [4, 5],
                "high_keep_patch_indices": [0, 1],
                "low_keep_patch_indices": [2, 3],
                "target_keep": 346,
                "strategy_keep_high": 242,
                "strategy_keep_low": 104,
                "grid_size": 6,
                "patch_per_row": 24,
                "high_ratio": 0.7,
                "stratum_quotas": [1] * 36,
            },
            save_importance=False,
            save_indices=True,
        )

        self.assertEqual(sample_stats["layer_2_keep_patch_indices"], [0, 1, 2, 3])
        self.assertEqual(sample_stats["layer_2_pruned_patch_indices"], [4, 5])
        self.assertEqual(sample_stats["layer_2_high_keep_patch_indices"], [0, 1])
        self.assertEqual(sample_stats["layer_2_low_keep_patch_indices"], [2, 3])
        self.assertEqual(sample_stats["layer_2_target_keep"], 346)
        self.assertEqual(sample_stats["layer_2_strategy_keep_high"], 242)
        self.assertEqual(sample_stats["layer_2_strategy_keep_low"], 104)
        self.assertEqual(sample_stats["layer_2_grid_size"], 6)
        self.assertEqual(sample_stats["layer_2_patch_per_row"], 24)
        self.assertEqual(sample_stats["layer_2_high_ratio"], 0.7)
        self.assertEqual(sample_stats["layer_2_stratum_quotas"], [1] * 36)

    def test_append_layer_stats_fields_records_efficiency_timing_fields(self):
        sample_stats = {}
        append_layer_stats_fields(
            sample_stats,
            2,
            {
                "prune_ratio": 0.5,
                "num_visual_before": 576,
                "num_visual_after": 128,
                "num_pruned": 448,
                "selection_rule": "saliency_topk",
                "layer_strategy_effective": "sparsevlm",
                "selection_time_ms": 0.12,
                "distance_time_ms": 0.0,
                "distance_cost_proxy": 0,
            },
            save_importance=False,
            save_indices=False,
        )

        self.assertEqual(sample_stats["layer_2_selection_rule"], "saliency_topk")
        self.assertEqual(sample_stats["layer_2_layer_strategy_effective"], "sparsevlm")
        self.assertEqual(sample_stats["layer_2_selection_time_ms"], 0.12)
        self.assertEqual(sample_stats["layer_2_distance_time_ms"], 0.0)
        self.assertEqual(sample_stats["layer_2_distance_cost_proxy"], 0)


if __name__ == "__main__":
    unittest.main()
