import os
import sys
import unittest

import torch

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC_DIR)

from strategies import get_strategy
from strategies.sparsevlm_diverse_mmr import SparseVLMDiverseMMRStrategy


def _full_attention(seq_len, text_rows, visual_cols, values):
    attn = torch.zeros((1, 1, seq_len, seq_len), dtype=torch.float32)
    for row, row_values in zip(text_rows, values):
        attn[0, 0, row, visual_cols] = torch.tensor(row_values, dtype=torch.float32)
    return attn


def _inputs(num_visual_tokens, dim=2):
    return torch.zeros((1, 1 + num_visual_tokens + 2, dim), dtype=torch.float32)


class SparseVLMDiverseMMRTests(unittest.TestCase):
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
            "layer_modes": ["D"],
            "prune_ratio": 0.5,
            "fallback_topk": 4,
            "exclude_special_tokens": True,
            "min_visual_tokens_after_prune": 1,
            "grid_size": 1,
            "patch_per_row": 2,
            "grid_balance_weight": 0.25,
            "core_keep_ratio": 0.5,
            "candidate_pool_multiplier": 2.0,
            "distance_metric": "cosine",
            "use_score_memory": False,
        }
        config.update(overrides)
        return SparseVLMDiverseMMRStrategy(config)

    def test_registry_instantiates_strategy(self):
        self.assertIsInstance(get_strategy("sparsevlm_diverse_mmr", {}), SparseVLMDiverseMMRStrategy)

    def test_grid_quota_sums_to_target_keep_and_keeps_anchors(self):
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

        self.assertEqual(keep.numel(), info["target_keep"])
        self.assertEqual(sum(info["grid_quota"]), info["target_keep"])
        for sid, quota in enumerate(info["grid_quota"]):
            if quota > 0:
                self.assertGreaterEqual(len(info["grid_anchor_indices"][sid]), 1)
        self.assertEqual(info["selection_rule"], "grid_local_pure_distance")
        self.assertNotIn("lambda_div", info)

    def test_candidate_pool_blocks_low_saliency_far_token(self):
        strategy = self._strategy(candidate_pool_multiplier=1.0, core_keep_ratio=0.5)
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

    def test_grid_local_fill_uses_distance_only(self):
        strategy = self._strategy(candidate_pool_multiplier=2.0, core_keep_ratio=0.5)
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
        self.assertEqual(info["grid_fill_indices"][0], [3])

    def test_diversity_weight_config_does_not_change_keep_indices(self):
        base = {
            "candidate_pool_multiplier": 2.0,
            "core_keep_ratio": 0.5,
            "diversity_weight_max": 0.0,
            "diversity_decay_per_step": 0.0,
        }
        alt = dict(base)
        alt.update({"diversity_weight_max": 1.0, "diversity_decay_per_step": 1.0})
        strategy_a = self._strategy(**base)
        strategy_b = self._strategy(**alt)
        self._prepare(strategy_a, 4)
        self._prepare(strategy_b, 4)
        attn = _full_attention(
            seq_len=7,
            text_rows=[5, 6],
            visual_cols=[1, 2, 3, 4],
            values=[[0.9, 0.8, 0.7, 0.1], [0.9, 0.8, 0.7, 0.1]],
        )
        embeds = torch.tensor([[1.0, 0.0], [0.99, 0.0], [0.0, 1.0], [-1.0, 0.0]])

        keep_a, _info_a = strategy_a.compute_keep_mask(attn, 1, 4, 5, 0, current_visual_embeds=embeds)
        keep_b, _info_b = strategy_b.compute_keep_mask(attn, 1, 4, 5, 0, current_visual_embeds=embeds)

        self.assertEqual(keep_a.tolist(), keep_b.tolist())

    def test_grid_balance_zero_core_one_degenerates_to_saliency_topk(self):
        strategy = self._strategy(
            grid_size=2,
            patch_per_row=2,
            grid_balance_weight=0.0,
            core_keep_ratio=1.0,
            candidate_pool_multiplier=1.0,
        )
        self._prepare(strategy, 4)
        attn = _full_attention(
            seq_len=7,
            text_rows=[5, 6],
            visual_cols=[1, 2, 3, 4],
            values=[[0.8, 0.3, 0.9, 0.2], [0.8, 0.3, 0.9, 0.2]],
        )
        embeds = torch.eye(4, dtype=torch.float32)
        keep, info = strategy.compute_keep_mask(attn, 1, 4, 5, 0, current_visual_embeds=embeds)

        self.assertEqual(keep.tolist(), [0, 2])
        self.assertEqual(info["grid_saliency_topk_counts"], info["grid_quota"])

    def test_layer_modes_switch_to_sparsevlm_branch(self):
        strategy = SparseVLMDiverseMMRStrategy(
            {
                "prune_layers": [2, 6],
                "layer_modes": ["D", "S"],
                "prune_ratio_map": {2: 0.25, 6: 0.333},
                "fallback_topk": 4,
                "exclude_special_tokens": True,
                "min_visual_tokens_after_prune": 1,
                "grid_size": 2,
                "patch_per_row": 2,
                "grid_balance_weight": 0.25,
                "core_keep_ratio": 0.5,
                "candidate_pool_multiplier": 2.0,
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

        self.assertEqual(info2["layer_mode"], "D")
        self.assertEqual(info2["selection_rule"], "grid_local_pure_distance")
        self.assertNotIn("lambda_div", info2)
        self.assertEqual(info6["layer_mode"], "S")
        self.assertEqual(info6["layer_strategy_effective"], "sparsevlm")
        self.assertNotIn("selection_rule", info6)
        self.assertEqual(info6["current_patch_indices"].tolist(), keep2.tolist())


if __name__ == "__main__":
    unittest.main()
