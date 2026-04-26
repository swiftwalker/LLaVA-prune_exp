import math
import os
import sys
import unittest

import torch

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC_DIR)

from strategies import get_strategy
from strategies.sparsevlm_entropy_alpha import SparseVLMEntropyAlphaStrategy


def _full_attention(seq_len, text_rows, visual_cols, values):
    attn = torch.zeros((1, 1, seq_len, seq_len), dtype=torch.float32)
    for row, row_values in zip(text_rows, values):
        attn[0, 0, row, visual_cols] = torch.tensor(row_values, dtype=torch.float32)
    return attn


class SparseVLMEntropyAlphaStrategyTests(unittest.TestCase):
    def _make_strategy(self, **overrides):
        config = {
            "prune_ratio": 0.5,
            "fallback_topk": 4,
            "exclude_special_tokens": True,
            "min_visual_tokens_after_prune": 1,
            "grid_size": 2,
            "patch_per_row": 2,
            "intra_stratum_mode": "random",
            "alpha_min": 0.4,
            "alpha_max": 0.95,
        }
        config.update(overrides)
        return SparseVLMEntropyAlphaStrategy(config)

    def _make_inputs(self):
        return torch.tensor(
            [
                [
                    [9.0, 9.0],
                    [1.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                ]
            ],
            dtype=torch.float32,
        )

    def test_entropy_alpha_uniform_scores(self):
        strategy = self._make_strategy()

        alpha, entropy_raw, entropy_norm = strategy._compute_adaptive_alpha(torch.ones(4, dtype=torch.float32))

        self.assertAlmostEqual(entropy_raw, math.log(4.0), places=6)
        self.assertAlmostEqual(entropy_norm, 1.0, places=6)
        self.assertAlmostEqual(alpha, 0.4, places=6)

    def test_entropy_alpha_concentrated_scores(self):
        strategy = self._make_strategy()

        alpha, entropy_raw, entropy_norm = strategy._compute_adaptive_alpha(
            torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        )

        self.assertAlmostEqual(entropy_raw, 0.0, places=6)
        self.assertAlmostEqual(entropy_norm, 0.0, places=6)
        self.assertAlmostEqual(alpha, 0.95, places=6)

    def test_entropy_alpha_range(self):
        strategy = self._make_strategy(alpha_min=0.3, alpha_max=0.8)

        for scores in (
            torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32),
            torch.tensor([3.0, 1.0, 1.0, 1.0], dtype=torch.float32),
            torch.tensor([5.0, 0.5, 0.25, 0.25], dtype=torch.float32),
            torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
        ):
            alpha, _, _ = strategy._compute_adaptive_alpha(scores)
            self.assertGreaterEqual(alpha, 0.3)
            self.assertLessEqual(alpha, 0.8)

    def test_entropy_alpha_edge_cases(self):
        strategy = self._make_strategy()

        alpha_zero, entropy_zero, entropy_zero_norm = strategy._compute_adaptive_alpha(torch.zeros(4))
        alpha_single, entropy_single, entropy_single_norm = strategy._compute_adaptive_alpha(torch.tensor([2.0]))
        alpha_equal, _, entropy_equal_norm = strategy._compute_adaptive_alpha(torch.tensor([2.0, 2.0, 2.0, 2.0]))

        self.assertEqual((alpha_zero, entropy_zero, entropy_zero_norm), (0.95, 0.0, 0.0))
        self.assertEqual((alpha_single, entropy_single, entropy_single_norm), (0.95, 0.0, 0.0))
        self.assertAlmostEqual(alpha_equal, 0.4, places=6)
        self.assertAlmostEqual(entropy_equal_norm, 1.0, places=6)

    def test_keep_mask_output_format(self):
        strategy = self._make_strategy()
        inputs = self._make_inputs()
        attn = _full_attention(
            seq_len=7,
            text_rows=[5, 6],
            visual_cols=[1, 2, 3, 4],
            values=[[0.9, 0.2, 0.1, 0.05], [0.3, 0.2, 0.2, 0.1]],
        )

        sample_info = strategy.prepare_sample(
            inputs_embeds=inputs,
            v_token_start=1,
            v_token_num=4,
            text_token_start=5,
            text_token_ids=torch.tensor([11, 12]),
            text_special_token_mask=torch.tensor([False, False]),
        )
        self.assertIn("rater_indices", sample_info)
        self.assertNotIn("high_ratio", strategy.config)

        keep_indices, info = strategy.compute_keep_mask(
            attn_weights=attn,
            v_token_start=1,
            v_token_num=4,
            text_token_start=5,
            layer_idx=0,
        )

        self.assertEqual(len(keep_indices), info["target_keep"])
        self.assertEqual(info["alpha_mode"], "entropy")
        self.assertAlmostEqual(info["adaptive_alpha"], info["high_ratio"], places=6)
        self.assertEqual(info["strategy_keep_high"] + info["strategy_keep_low"], info["target_keep"])
        for key in (
            "adaptive_alpha",
            "saliency_entropy",
            "saliency_entropy_norm",
            "alpha_min",
            "alpha_max",
        ):
            self.assertIn(key, info)

    def test_budget_consistency(self):
        strategy = self._make_strategy(prune_ratio=0.75, min_visual_tokens_after_prune=2)
        inputs = self._make_inputs()
        attn = _full_attention(
            seq_len=7,
            text_rows=[5, 6],
            visual_cols=[1, 2, 3, 4],
            values=[[0.8, 0.3, 0.2, 0.1], [0.4, 0.2, 0.1, 0.1]],
        )

        strategy.prepare_sample(
            inputs_embeds=inputs,
            v_token_start=1,
            v_token_num=4,
            text_token_start=5,
            text_token_ids=torch.tensor([11, 12]),
            text_special_token_mask=torch.tensor([False, False]),
        )
        keep_indices, info = strategy.compute_keep_mask(
            attn_weights=attn,
            v_token_start=1,
            v_token_num=4,
            text_token_start=5,
            layer_idx=0,
        )

        self.assertEqual(int(keep_indices.numel()), 2)
        self.assertEqual(int(keep_indices.numel()), info["target_keep"])
        self.assertEqual(info["num_visual_after"], info["target_keep"])

    def test_registry_instantiates_new_strategy(self):
        strategy = get_strategy(
            "sparsevlm_entropy_alpha",
            {
                "prune_ratio": 0.5,
                "grid_size": 6,
                "patch_per_row": 24,
                "intra_stratum_mode": "random",
                "alpha_min": 0.4,
                "alpha_max": 0.95,
            },
        )

        self.assertIsInstance(strategy, SparseVLMEntropyAlphaStrategy)


if __name__ == "__main__":
    unittest.main()
