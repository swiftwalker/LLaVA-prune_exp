import os
import sys
import unittest

import torch

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC_DIR)

from strategies import get_strategy
from strategies.sparsevlm_scnd import SparseVLMSCNDStrategy


def _full_attention(seq_len, text_rows, visual_cols, values):
    attn = torch.zeros((1, 1, seq_len, seq_len), dtype=torch.float32)
    for row, row_values in zip(text_rows, values):
        attn[0, 0, row, visual_cols] = torch.tensor(row_values, dtype=torch.float32)
    return attn


def _inputs(num_visual_tokens, dim=2):
    return torch.zeros((1, 1 + num_visual_tokens + 2, dim), dtype=torch.float32)


class SparseVLMSCNDTests(unittest.TestCase):
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
            "layer_modes": ["C"],
            "prune_ratio": 0.5,
            "fallback_topk": 4,
            "exclude_special_tokens": True,
            "min_visual_tokens_after_prune": 1,
            "seed_ratio_min": 0.5,
            "seed_ratio_max": 0.5,
            "seed_pool_multiplier": 2.0,
            "saliency_floor_min": 0.65,
            "saliency_floor_max": 0.65,
            "boundary_ratio": 0.25,
            "saliency_repair": True,
            "distance_metric": "cosine",
            "use_score_memory": False,
        }
        config.update(overrides)
        return SparseVLMSCNDStrategy(config)

    def test_registry_instantiates_strategy(self):
        self.assertIsInstance(get_strategy("sparsevlm_scnd", {}), SparseVLMSCNDStrategy)

    def test_layer_modes_length_mismatch_raises(self):
        strategy = self._strategy(prune_layers=[0, 1], layer_modes=["C"])
        with self.assertRaises(ValueError):
            strategy._layer_mode_map()

    def test_c_layer_keep_count_and_native_rule_without_grid_quota(self):
        strategy = self._strategy(prune_ratio=0.5)
        self._prepare(strategy, 4)
        attn = _full_attention(
            seq_len=7,
            text_rows=[5, 6],
            visual_cols=[1, 2, 3, 4],
            values=[[0.9, 0.8, 0.7, 0.1], [0.9, 0.8, 0.7, 0.1]],
        )
        embeds = torch.eye(4, dtype=torch.float32)
        keep, info = strategy.compute_keep_mask(attn, 1, 4, 5, 0, current_visual_embeds=embeds)

        self.assertEqual(keep.numel(), info["target_keep"])
        self.assertEqual(info["selection_rule"], "saliency_constrained_native_divprune")
        self.assertNotIn("grid_quota", info)
        self.assertEqual(info["layer_strategy_effective"], "sparsevlm_scnd")

    def test_seed_pool_uses_native_diversity_not_top_saliency_anchors(self):
        strategy = self._strategy(
            prune_ratio=0.5,
            seed_ratio_min=1.0,
            seed_ratio_max=1.0,
            seed_pool_multiplier=2.0,
            saliency_floor_min=0.0,
            saliency_floor_max=0.0,
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
        self.assertEqual(info["seed_indices"].tolist(), [0, 3])
        self.assertNotEqual(info["seed_indices"].tolist(), [0, 1])

    def test_saliency_mass_feasibility_blocks_low_saliency_outlier(self):
        strategy = self._strategy(
            prune_ratio=0.4,
            seed_ratio_min=0.34,
            seed_ratio_max=0.34,
            seed_pool_multiplier=1.0,
            saliency_floor_min=1.0,
            saliency_floor_max=1.0,
            saliency_repair=False,
        )
        self._prepare(strategy, 5)
        attn = _full_attention(
            seq_len=8,
            text_rows=[6, 7],
            visual_cols=[1, 2, 3, 4, 5],
            values=[[0.9, 0.8, 0.7, 0.2, 0.1], [0.9, 0.8, 0.7, 0.2, 0.1]],
        )
        embeds = torch.tensor([[1.0, 0.0], [0.9, 0.0], [0.8, 0.0], [0.7, 0.0], [-1.0, 0.0]])
        keep, info = strategy.compute_keep_mask(attn, 1, 5, 6, 0, current_visual_embeds=embeds)

        self.assertEqual(keep.tolist(), [0, 1, 2])
        self.assertNotIn(4, keep.tolist())
        self.assertGreaterEqual(info["saliency_mass_selected"] + 1e-8, info["saliency_mass_floor"])

    def test_repair_can_raise_saliency_mass(self):
        strategy = self._strategy()
        selected, replacements = strategy._repair_saliency_mass(
            selected=[0, 3],
            seed_indices=[],
            saliency_score=torch.tensor([1.0, 0.8, 0.4, 0.0]),
            distance=torch.eye(4, dtype=torch.float32),
            saliency_mass_floor=1.8,
        )

        self.assertEqual(selected, [0, 1])
        self.assertEqual(replacements[0]["out"], 3)
        self.assertEqual(replacements[0]["in"], 1)

    def test_b_layer_large_margin_degenerates_to_sparsevlm_topk(self):
        strategy = self._strategy(layer_modes=["B"], prune_ratio=0.5, boundary_ratio=1.0)
        self._prepare(strategy, 4)
        attn = _full_attention(
            seq_len=7,
            text_rows=[5, 6],
            visual_cols=[1, 2, 3, 4],
            values=[[10.0, 9.0, 1.0, 0.0], [10.0, 9.0, 1.0, 0.0]],
        )
        embeds = torch.eye(4, dtype=torch.float32)
        keep, info = strategy.compute_keep_mask(attn, 1, 4, 5, 0, current_visual_embeds=embeds)

        self.assertEqual(keep.tolist(), [0, 1])
        self.assertEqual(info["boundary_count"], 0)
        self.assertEqual(info["selection_rule"], "boundary_only_diversity_refinement")

    def test_b_layer_boundary_uses_distance_without_challenging_core(self):
        strategy = self._strategy(layer_modes=["B"], prune_ratio=0.5, boundary_ratio=0.5)
        self._prepare(strategy, 4)
        attn = _full_attention(
            seq_len=7,
            text_rows=[5, 6],
            visual_cols=[1, 2, 3, 4],
            values=[[0.90, 0.89, 0.88, 0.87], [0.90, 0.89, 0.88, 0.87]],
        )
        embeds = torch.tensor([[1.0, 0.0], [0.99, 0.0], [-1.0, 0.0], [0.0, 1.0]])
        keep, info = strategy.compute_keep_mask(attn, 1, 4, 5, 0, current_visual_embeds=embeds)

        self.assertEqual(info["boundary_core_indices"].tolist(), [0])
        self.assertEqual(info["boundary_fill_indices"].tolist(), [2])
        self.assertEqual(keep.tolist(), [0, 2])

    def test_s_layer_matches_sparsevlm_branch(self):
        strategy = SparseVLMSCNDStrategy(
            {
                "prune_layers": [2, 6],
                "layer_modes": ["C", "S"],
                "prune_ratio_map": {2: 0.25, 6: 0.34},
                "fallback_topk": 4,
                "exclude_special_tokens": True,
                "min_visual_tokens_after_prune": 1,
                "seed_ratio_min": 0.5,
                "seed_ratio_max": 0.5,
                "seed_pool_multiplier": 2.0,
                "saliency_floor_min": 0.65,
                "saliency_floor_max": 0.65,
                "boundary_ratio": 0.25,
                "saliency_repair": True,
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
        keep2, _info2 = strategy.compute_keep_mask(layer2_attn, 1, 4, 5, 2, current_visual_embeds=torch.eye(4))
        strategy.update_after_prune(keep2, 2)

        layer6_attn = _full_attention(
            seq_len=6,
            text_rows=[4, 5],
            visual_cols=[1, 2, 3],
            values=[[0.1, 0.9, 0.2], [0.1, 0.8, 0.2]],
        )
        keep6, info6 = strategy.compute_keep_mask(layer6_attn, 1, 3, 4, 6)

        self.assertEqual(keep6.tolist(), [1, 2])
        self.assertEqual(info6["layer_mode"], "S")
        self.assertEqual(info6["layer_strategy_effective"], "sparsevlm")
        self.assertNotIn("selection_rule", info6)


if __name__ == "__main__":
    unittest.main()
