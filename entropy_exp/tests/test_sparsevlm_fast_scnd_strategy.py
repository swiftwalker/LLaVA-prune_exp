import os
import sys
import unittest

import torch

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC_DIR)

from strategies import get_strategy
from strategies.sparsevlm_fast_scnd import SparseVLMFastSCNDStrategy


def _full_attention(seq_len, text_rows, visual_cols, values):
    attn = torch.zeros((1, 1, seq_len, seq_len), dtype=torch.float32)
    for row, row_values in zip(text_rows, values):
        attn[0, 0, row, visual_cols] = torch.tensor(row_values, dtype=torch.float32)
    return attn


def _inputs(num_visual_tokens, dim=4):
    return torch.zeros((1, 1 + num_visual_tokens + 2, dim), dtype=torch.float32)


class SparseVLMFastSCNDTests(unittest.TestCase):
    def _prepare(self, strategy, num_visual_tokens, dim=4):
        strategy.prepare_sample(
            inputs_embeds=_inputs(num_visual_tokens, dim=dim),
            v_token_start=1,
            v_token_num=num_visual_tokens,
            text_token_start=1 + num_visual_tokens,
            text_token_ids=torch.tensor([11, 12]),
            text_special_token_mask=torch.tensor([False, False]),
        )

    def _strategy(self, **overrides):
        config = {
            "prune_layers": [0],
            "layer_modes": ["F"],
            "prune_ratio": 0.5,
            "fallback_topk": 4,
            "exclude_special_tokens": True,
            "min_visual_tokens_after_prune": 1,
            "core_ratio_min": 0.10,
            "core_ratio_max": 0.10,
            "saliency_pool_multiplier": 1.5,
            "reservoir_multiplier": 0.5,
            "reservoir_rank_multiplier": 4.0,
            "candidate_cap_multiplier": 2.0,
            "projection_dim": 4,
            "greedy_steps": 16,
            "tie_break_band_ratio": 0.10,
            "tie_break_band_max": 32,
            "cached_diversity_weight": 0.05,
            "distance_metric": "cosine",
            "use_score_memory": False,
        }
        config.update(overrides)
        return SparseVLMFastSCNDStrategy(config)

    def test_registry_instantiates_strategy(self):
        self.assertIsInstance(get_strategy("sparsevlm_fast_scnd", {}), SparseVLMFastSCNDStrategy)

    def test_layer_modes_length_mismatch_raises(self):
        strategy = self._strategy(prune_layers=[0, 1], layer_modes=["F"])
        with self.assertRaises(ValueError):
            strategy._layer_mode_map()

    def test_f_layer_keep_count_and_fast_rule_without_scnd_repair_fields(self):
        strategy = self._strategy(prune_ratio=0.5)
        self._prepare(strategy, 4)
        attn = _full_attention(
            seq_len=7,
            text_rows=[5, 6],
            visual_cols=[1, 2, 3, 4],
            values=[[0.9, 0.8, 0.7, 0.1], [0.9, 0.8, 0.7, 0.1]],
        )
        keep, info = strategy.compute_keep_mask(attn, 1, 4, 5, 0, current_visual_embeds=torch.eye(4))

        self.assertEqual(int(keep.numel()), info["target_keep"])
        self.assertEqual(info["selection_rule"], "fast_scnd_micro_greedy")
        self.assertEqual(info["layer_strategy_effective"], "sparsevlm_fast_scnd")
        self.assertNotIn("grid_quota", info)
        self.assertNotIn("saliency_mass_floor", info)
        self.assertNotIn("repair_replacements", info)

    def test_candidate_pool_is_capped_and_reservoir_comes_from_top_saliency_window(self):
        strategy = self._strategy(prune_ratio=0.8, greedy_steps=4)
        num_tokens = 20
        self._prepare(strategy, num_tokens)
        scores = [float(num_tokens - idx) for idx in range(num_tokens)]
        attn = _full_attention(
            seq_len=1 + num_tokens + 2,
            text_rows=[1 + num_tokens, 2 + num_tokens],
            visual_cols=list(range(1, 1 + num_tokens)),
            values=[scores, scores],
        )
        generator = torch.Generator()
        generator.manual_seed(7)
        embeds = torch.randn(num_tokens, 8, generator=generator)
        _keep, info = strategy.compute_keep_mask(attn, 1, num_tokens, 1 + num_tokens, 0, current_visual_embeds=embeds)

        target_keep = int(info["target_keep"])
        self.assertLessEqual(len(info["candidate_pool_indices"]), 2 * target_keep)
        self.assertLessEqual(len(info["reservoir_indices"]), int(info["reservoir_source_count"]))
        self.assertTrue(all(int(idx) < int(info["reservoir_source_count"]) for idx in info["reservoir_indices"]))

    def test_greedy_steps_zero_degenerates_to_saliency_core_plus_saliency_fill(self):
        strategy = self._strategy(prune_ratio=0.5, greedy_steps=0)
        self._prepare(strategy, 4)
        attn = _full_attention(
            seq_len=7,
            text_rows=[5, 6],
            visual_cols=[1, 2, 3, 4],
            values=[[0.9, 0.8, 0.7, 0.1], [0.9, 0.8, 0.7, 0.1]],
        )
        keep, info = strategy.compute_keep_mask(attn, 1, 4, 5, 0, current_visual_embeds=torch.eye(4))

        self.assertEqual(keep.tolist(), [0, 1])
        self.assertEqual(info["greedy_steps_used"], 0)
        self.assertEqual(info["core_indices"].tolist(), [0])

    def test_greedy_steps_limit_is_respected_then_saliency_fills(self):
        strategy = self._strategy(prune_ratio=0.25, greedy_steps=1)
        self._prepare(strategy, 4)
        attn = _full_attention(
            seq_len=7,
            text_rows=[5, 6],
            visual_cols=[1, 2, 3, 4],
            values=[[0.9, 0.8, 0.7, 0.1], [0.9, 0.8, 0.7, 0.1]],
        )
        embeds = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.99, 0.0, 0.0, 0.0], [0.98, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]]
        )
        keep, info = strategy.compute_keep_mask(attn, 1, 4, 5, 0, current_visual_embeds=embeds)

        self.assertEqual(int(info["greedy_steps_used"]), 1)
        self.assertEqual(int(keep.numel()), 3)
        self.assertIn(3, keep.tolist())

    def test_t_layer_uses_cached_gain_for_boundary_tie_break(self):
        strategy = self._strategy(
            prune_layers=[2, 6],
            layer_modes=["F", "T"],
            prune_ratio_map={2: 0.25, 6: 0.34},
            greedy_steps=1,
            tie_break_band_ratio=0.5,
            tie_break_band_max=1,
            cached_diversity_weight=1.0,
        )
        self._prepare(strategy, 4)
        layer2_attn = _full_attention(
            seq_len=7,
            text_rows=[5, 6],
            visual_cols=[1, 2, 3, 4],
            values=[[0.9, 0.8, 0.7, 0.1], [0.9, 0.8, 0.7, 0.1]],
        )
        embeds = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.99, 0.0, 0.0, 0.0], [0.98, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]]
        )
        keep2, info2 = strategy.compute_keep_mask(layer2_attn, 1, 4, 5, 2, current_visual_embeds=embeds)
        strategy.update_after_prune(keep2, 2)

        layer6_attn = _full_attention(
            seq_len=6,
            text_rows=[4, 5],
            visual_cols=[1, 2, 3],
            values=[[0.9, 0.89, 0.88], [0.9, 0.89, 0.88]],
        )
        keep6, info6 = strategy.compute_keep_mask(
            layer6_attn,
            1,
            3,
            4,
            6,
            current_visual_embeds=embeds.index_select(0, keep2),
        )

        self.assertEqual(info2["selection_rule"], "fast_scnd_micro_greedy")
        self.assertEqual(info6["selection_rule"], "fast_scnd_cached_tie_break")
        self.assertEqual(info6["tie_break_band_count"], 1)
        self.assertEqual(info6["boundary_fill_indices"].tolist(), [2])
        self.assertEqual(keep6.tolist(), [0, 2])
        self.assertIn("cached_diversity_gain", info6)
        self.assertNotIn("reservoir_indices", info6)

    def test_s_layer_matches_sparsevlm_branch(self):
        strategy = self._strategy(
            prune_layers=[2, 6],
            layer_modes=["F", "S"],
            prune_ratio_map={2: 0.25, 6: 0.34},
            greedy_steps=0,
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

    def test_cached_gain_tracks_current_patch_mapping_after_prune(self):
        strategy = self._strategy(
            prune_layers=[2, 6],
            layer_modes=["F", "T"],
            prune_ratio_map={2: 0.25, 6: 0.34},
            greedy_steps=1,
        )
        self._prepare(strategy, 4)
        layer2_attn = _full_attention(
            seq_len=7,
            text_rows=[5, 6],
            visual_cols=[1, 2, 3, 4],
            values=[[0.9, 0.8, 0.7, 0.1], [0.9, 0.8, 0.7, 0.1]],
        )
        embeds = torch.eye(4)
        keep2, _info2 = strategy.compute_keep_mask(layer2_attn, 1, 4, 5, 2, current_visual_embeds=embeds)
        strategy.update_after_prune(keep2, 2)

        layer6_attn = _full_attention(
            seq_len=6,
            text_rows=[4, 5],
            visual_cols=[1, 2, 3],
            values=[[0.9, 0.8, 0.7], [0.9, 0.8, 0.7]],
        )
        _keep6, info6 = strategy.compute_keep_mask(
            layer6_attn,
            1,
            3,
            4,
            6,
            current_visual_embeds=embeds.index_select(0, keep2),
        )

        self.assertEqual(info6["current_patch_indices"].tolist(), keep2.tolist())
        self.assertEqual(len(info6["cached_diversity_gain"]), int(keep2.numel()))


if __name__ == "__main__":
    unittest.main()
