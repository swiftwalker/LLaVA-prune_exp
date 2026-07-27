import os
import sys
import unittest

import torch


SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC_DIR)

from prune_inference import append_layer_stats_fields, compute_visual_token_role_stats  # noqa: E402


class PruneInferenceStatsTests(unittest.TestCase):
    def test_scalar_stats_are_recorded_without_index_exports(self):
        sample_stats = {}
        append_layer_stats_fields(
            sample_stats,
            2,
            {
                "prune_ratio": 0.5,
                "num_visual_before": 8,
                "num_visual_after": 4,
                "num_pruned": 4,
                "saliency_entropy_norm": 0.72,
                "saliency_entropy_norm_control": 0.44,
                "entropy_calibration_mode": "quantile_affine",
                "entropy_calibration_applied": True,
            },
            save_importance=False,
            save_indices=False,
        )

        self.assertEqual(sample_stats["layer_2_saliency_entropy_norm"], 0.72)
        self.assertEqual(sample_stats["layer_2_saliency_entropy_norm_control"], 0.44)
        self.assertTrue(sample_stats["layer_2_entropy_calibration_applied"])
        self.assertNotIn("layer_2_keep_indices", sample_stats)

    def test_anyres_role_stats_and_patch_only_entropy(self):
        layout = {
            "base_token_count": 4,
            "local_patch_token_count": 6,
            "newline_token_count": 2,
            "local_grid_width": 3,
            "total_token_count": 12,
        }
        layer_info = {
            "current_patch_indices": list(range(12)),
            "keep_patch_indices": [0, 2, 4, 7, 8, 11],
            "importance_scores": torch.tensor(
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100.0, 1.0, 1.0, 1.0, 100.0]
            ),
        }

        stats = compute_visual_token_role_stats(layer_info, layout)

        self.assertEqual(stats["base_token_count_before"], 4)
        self.assertEqual(stats["base_token_count_after"], 2)
        self.assertEqual(stats["local_patch_token_count_before"], 6)
        self.assertEqual(stats["local_patch_token_count_after"], 2)
        self.assertEqual(stats["newline_token_count_before"], 2)
        self.assertEqual(stats["newline_token_count_after"], 2)
        self.assertEqual(stats["patch_only_saliency_token_count"], 10)
        self.assertAlmostEqual(stats["patch_only_saliency_entropy_norm"], 1.0, places=7)
        self.assertGreater(stats["newline_saliency_mass_ratio"], 0.9)
        self.assertEqual(stats["topk_base_token_count"], 4)
        self.assertEqual(stats["topk_local_patch_token_count"], 0)
        self.assertEqual(stats["topk_newline_token_count"], 2)
        self.assertAlmostEqual(
            stats["base_token_saliency_mass_ratio"]
            + stats["local_patch_token_saliency_mass_ratio"]
            + stats["newline_saliency_mass_ratio"],
            1.0,
        )

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


if __name__ == "__main__":
    unittest.main()
