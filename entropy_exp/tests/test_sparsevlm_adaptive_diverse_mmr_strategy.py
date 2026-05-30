import os
import sys
import unittest

import torch

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC_DIR)

from strategies import get_strategy
from strategies.sparsevlm_adaptive_diverse_mmr import SparseVLMAdaptiveDiverseMMRStrategy


def _full_attention(seq_len, text_rows, visual_cols, values):
    attn = torch.zeros((1, 1, seq_len, seq_len), dtype=torch.float32)
    for row, row_values in zip(text_rows, values):
        attn[0, 0, row, visual_cols] = torch.tensor(row_values, dtype=torch.float32)
    return attn


def _inputs(num_visual_tokens, dim=2):
    return torch.zeros((1, 1 + num_visual_tokens + 2, dim), dtype=torch.float32)


class SparseVLMAdaptiveDiverseMMRTests(unittest.TestCase):
    def _prepare(self, strategy, num_visual_tokens):
        strategy.prepare_sample(
            inputs_embeds=_inputs(num_visual_tokens),
            v_token_start=1,
            v_token_num=num_visual_tokens,
            text_token_start=1 + num_visual_tokens,
            text_token_ids=torch.tensor([11, 12]),
            text_special_token_mask=torch.tensor([False, False]),
        )

    def _strategy(self, **overrides):
        config = {
            "prune_layers": [0],
            "layer_modes": ["A"],
            "prune_ratio": 0.5,
            "fallback_topk": 4,
            "exclude_special_tokens": True,
            "min_visual_tokens_after_prune": 1,
            "grid_size": 1,
            "patch_per_row": 2,
            "grid_balance_weight": 0.25,
            "min_candidate_multiplier": 1.2,
            "max_candidate_multiplier": 2.5,
            "distance_metric": "cosine",
            "use_score_memory": False,
        }
        config.update(overrides)
        return SparseVLMAdaptiveDiverseMMRStrategy(config)

    def test_registry_instantiates_strategy(self):
        self.assertIsInstance(
            get_strategy("sparsevlm_adaptive_diverse_mmr", {}),
            SparseVLMAdaptiveDiverseMMRStrategy,
        )

    def test_low_prune_ratio_keeps_saliency_anchors(self):
        strategy = self._strategy(prune_ratio=0.25, min_candidate_multiplier=1.0, max_candidate_multiplier=1.0)
        self._prepare(strategy, 4)
        attn = _full_attention(
            seq_len=7,
            text_rows=[5, 6],
            visual_cols=[1, 2, 3, 4],
            values=[[0.9, 0.8, 0.7, 0.1], [0.9, 0.8, 0.7, 0.1]],
        )
        embeds = torch.tensor([[1.0, 0.0], [0.99, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        keep, info = strategy.compute_keep_mask(attn, 1, 4, 5, 0, current_visual_embeds=embeds)

        self.assertEqual(keep.tolist(), [0, 1, 2])
        self.assertEqual(info["selection_rule"], "adaptive_grid_saliency_diversity")
        self.assertAlmostEqual(info["diversity_ratio"], 0.25)
        self.assertAlmostEqual(info["anchor_ratio"], 0.75)
        self.assertEqual(info["grid_anchor_indices"][0], [0, 1, 2])
        self.assertEqual(info["grid_diverse_fill_indices"][0], [])

    def test_high_prune_ratio_increases_diverse_fill_share(self):
        strategy = self._strategy(
            prune_ratio=0.5,
            min_candidate_multiplier=2.0,
            max_candidate_multiplier=2.0,
        )
        self._prepare(strategy, 4)
        attn = _full_attention(
            seq_len=7,
            text_rows=[5, 6],
            visual_cols=[1, 2, 3, 4],
            values=[[0.9, 0.8, 0.7, 0.1], [0.9, 0.8, 0.7, 0.1]],
        )
        embeds = torch.tensor([[1.0, 0.0], [0.99, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        keep, info = strategy.compute_keep_mask(attn, 1, 4, 5, 0, current_visual_embeds=embeds)

        self.assertEqual(keep.tolist(), [0, 3])
        self.assertEqual(info["grid_anchor_indices"][0], [0])
        self.assertEqual(info["grid_diverse_fill_indices"][0], [3])
        self.assertEqual(len(info["grid_diverse_fill_indices"][0]), 1)

    def test_candidate_pool_blocks_low_saliency_far_token(self):
        strategy = self._strategy(prune_ratio=0.5, min_candidate_multiplier=1.0, max_candidate_multiplier=1.0)
        self._prepare(strategy, 4)
        attn = _full_attention(
            seq_len=7,
            text_rows=[5, 6],
            visual_cols=[1, 2, 3, 4],
            values=[[0.9, 0.8, 0.2, 0.1], [0.9, 0.8, 0.2, 0.1]],
        )
        embeds = torch.tensor([[1.0, 0.0], [0.99, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        keep, info = strategy.compute_keep_mask(attn, 1, 4, 5, 0, current_visual_embeds=embeds)

        self.assertEqual(keep.tolist(), [0, 1])
        self.assertEqual(info["candidate_pool_indices"].tolist(), [0, 1])
        self.assertNotIn(3, keep.tolist())

    def test_diverse_fill_uses_distance_only_inside_candidate_pool(self):
        strategy = self._strategy(prune_ratio=0.5, min_candidate_multiplier=2.0, max_candidate_multiplier=2.0)
        self._prepare(strategy, 4)
        attn = _full_attention(
            seq_len=7,
            text_rows=[5, 6],
            visual_cols=[1, 2, 3, 4],
            values=[[0.9, 0.8, 0.7, 0.1], [0.9, 0.8, 0.7, 0.1]],
        )
        embeds = torch.tensor([[1.0, 0.0], [0.99, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        keep, info = strategy.compute_keep_mask(attn, 1, 4, 5, 0, current_visual_embeds=embeds)

        self.assertEqual(keep.tolist(), [0, 3])
        self.assertEqual(info["grid_diverse_fill_indices"][0], [3])
        self.assertLess(float(info["current_rank_score"][3]), float(info["current_rank_score"][1]))

    def test_candidate_multiplier_widens_with_prune_ratio(self):
        low = self._strategy(prune_ratio=0.25, min_candidate_multiplier=1.2, max_candidate_multiplier=2.4)
        high = self._strategy(prune_ratio=0.75, min_candidate_multiplier=1.2, max_candidate_multiplier=2.4)
        self._prepare(low, 4)
        self._prepare(high, 4)
        attn = _full_attention(
            seq_len=7,
            text_rows=[5, 6],
            visual_cols=[1, 2, 3, 4],
            values=[[0.9, 0.8, 0.7, 0.1], [0.9, 0.8, 0.7, 0.1]],
        )
        embeds = torch.eye(4, dtype=torch.float32)

        _keep_low, info_low = low.compute_keep_mask(attn, 1, 4, 5, 0, current_visual_embeds=embeds)
        _keep_high, info_high = high.compute_keep_mask(attn, 1, 4, 5, 0, current_visual_embeds=embeds)

        self.assertLess(
            info_low["candidate_pool_multiplier_effective"],
            info_high["candidate_pool_multiplier_effective"],
        )

    def test_layer_modes_switch_to_sparsevlm_branch(self):
        strategy = SparseVLMAdaptiveDiverseMMRStrategy(
            {
                "prune_layers": [2, 6],
                "layer_modes": ["A", "S"],
                "prune_ratio_map": {2: 0.25, 6: 0.333},
                "fallback_topk": 4,
                "exclude_special_tokens": True,
                "min_visual_tokens_after_prune": 1,
                "grid_size": 2,
                "patch_per_row": 2,
                "grid_balance_weight": 0.25,
                "min_candidate_multiplier": 1.2,
                "max_candidate_multiplier": 2.5,
                "distance_metric": "cosine",
                "use_score_memory": False,
            }
        )
        self._prepare(strategy, 4)
        layer2_attn = _full_attention(
            seq_len=7,
            text_rows=[5, 6],
            visual_cols=[1, 2, 3, 4],
            values=[[0.9, 0.6, 0.4, 0.1], [0.8, 0.6, 0.4, 0.1]],
        )
        embeds = torch.eye(4, dtype=torch.float32)
        keep2, info2 = strategy.compute_keep_mask(layer2_attn, 1, 4, 5, 2, current_visual_embeds=embeds)
        strategy.update_after_prune(keep2, 2)

        layer6_attn = _full_attention(
            seq_len=6,
            text_rows=[4, 5],
            visual_cols=[1, 2, 3],
            values=[[0.1, 0.9, 0.2], [0.1, 0.8, 0.2]],
        )
        _keep6, info6 = strategy.compute_keep_mask(layer6_attn, 1, 3, 4, 6)

        self.assertEqual(info2["layer_mode"], "A")
        self.assertEqual(info2["selection_rule"], "adaptive_grid_saliency_diversity")
        self.assertEqual(info6["layer_mode"], "S")
        self.assertEqual(info6["layer_strategy_effective"], "sparsevlm")
        self.assertNotIn("selection_rule", info6)
        self.assertEqual(info6["current_patch_indices"].tolist(), keep2.tolist())

    def test_final_keep_count_matches_target(self):
        strategy = self._strategy(grid_size=2, patch_per_row=2, prune_ratio=0.5)
        self._prepare(strategy, 4)
        attn = _full_attention(
            seq_len=7,
            text_rows=[5, 6],
            visual_cols=[1, 2, 3, 4],
            values=[[0.9, 0.8, 0.2, 0.1], [0.9, 0.8, 0.2, 0.1]],
        )
        embeds = torch.eye(4, dtype=torch.float32)
        keep, info = strategy.compute_keep_mask(attn, 1, 4, 5, 0, current_visual_embeds=embeds)

        self.assertEqual(int(keep.numel()), info["target_keep"])
        self.assertEqual(sum(info["grid_quota"]), info["target_keep"])


if __name__ == "__main__":
    unittest.main()
