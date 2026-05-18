import os
import sys
import unittest

import torch

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC_DIR)

from strategies import get_strategy
from strategies.sparsevlm_boost import SparseVLMBoostStrategy
from strategies.sparsevlm_boost_hybrid import SparseVLMBoostHybridStrategy
from strategies.sparsevlm_compensated import SparseVLMCompensatedStrategy
from strategies.sparsevlm import SparseVLMStrategy
from strategies.sparsevlm_score_memory import rank_normalize_scores


def _full_attention(seq_len, text_rows, visual_cols, values):
    attn = torch.zeros((1, 1, seq_len, seq_len), dtype=torch.float32)
    for row, row_values in zip(text_rows, values):
        attn[0, 0, row, visual_cols] = torch.tensor(row_values, dtype=torch.float32)
    return attn


def _make_grid_inputs(num_visual_tokens):
    visual_tokens = [[1.0, 0.0] for _ in range(num_visual_tokens)]
    return torch.tensor(
        [
            [
                [9.0, 9.0],
                *visual_tokens,
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        ],
        dtype=torch.float32,
    )


class SparseVLMBoostCompensatedTests(unittest.TestCase):
    def test_registry_instantiates_new_strategies(self):
        self.assertIsInstance(get_strategy("sparsevlm_boost", {}), SparseVLMBoostStrategy)
        self.assertIsInstance(get_strategy("sparsevlm_boost_hybrid", {}), SparseVLMBoostHybridStrategy)
        self.assertIsInstance(get_strategy("sparsevlm_compensated", {}), SparseVLMCompensatedStrategy)

    def test_boost_zero_weight_degenerates_to_sparsevlm_topk(self):
        layer0_attn = _full_attention(
            seq_len=7,
            text_rows=[5, 6],
            visual_cols=[1, 2, 3, 4],
            values=[[0.1, 0.8, 0.4, 0.2], [0.1, 0.7, 0.3, 0.2]],
        )
        strategy = SparseVLMBoostStrategy(
            {
                "prune_ratio": 0.5,
                "fallback_topk": 4,
                "exclude_special_tokens": True,
                "min_visual_tokens_after_prune": 1,
                "boost_weight_min": 0.0,
                "boost_weight_max": 0.0,
                "grid_size": 2,
                "patch_per_row": 2,
            }
        )
        strategy.prepare_sample(
            inputs_embeds=_make_grid_inputs(4),
            v_token_start=1,
            v_token_num=4,
            text_token_start=5,
            text_token_ids=torch.tensor([11, 12]),
            text_special_token_mask=torch.tensor([False, False]),
        )
        keep_indices, info = strategy.compute_keep_mask(layer0_attn, 1, 4, 5, 0)

        self.assertEqual(keep_indices.tolist(), [1, 2])
        self.assertEqual(info["token_boost"].tolist(), [0.0, 0.0, 0.0, 0.0])
        self.assertEqual(info["adjusted_scores"].tolist(), info["mixed_score"].tolist())

    def test_boost_does_not_change_order_inside_same_stratum(self):
        layer0_attn = _full_attention(
            seq_len=7,
            text_rows=[5, 6],
            visual_cols=[1, 2, 3, 4],
            values=[[0.9, 0.6, 0.05, 0.04], [0.8, 0.5, 0.05, 0.04]],
        )
        strategy = SparseVLMBoostStrategy(
            {
                "prune_ratio": 0.5,
                "fallback_topk": 4,
                "exclude_special_tokens": True,
                "min_visual_tokens_after_prune": 1,
                "boost_weight_min": 0.15,
                "boost_weight_max": 0.15,
                "grid_size": 1,
                "patch_per_row": 2,
            }
        )
        strategy.prepare_sample(
            inputs_embeds=_make_grid_inputs(4),
            v_token_start=1,
            v_token_num=4,
            text_token_start=5,
            text_token_ids=torch.tensor([11, 12]),
            text_special_token_mask=torch.tensor([False, False]),
        )
        _keep_indices, info = strategy.compute_keep_mask(layer0_attn, 1, 4, 5, 0)

        adjusted_order = torch.argsort(torch.tensor(info["adjusted_scores"]), descending=True).tolist()
        rank_order = torch.argsort(torch.tensor(info["current_rank_score"]), descending=True).tolist()
        self.assertEqual(adjusted_order, rank_order)

    def test_boost_second_layer_uses_ema_mixed_score(self):
        strategy = SparseVLMBoostStrategy(
            {
                "prune_ratio": 0.5,
                "prune_ratio_map": {0: 0.25, 1: 0.5},
                "fallback_topk": 4,
                "exclude_special_tokens": True,
                "min_visual_tokens_after_prune": 1,
                "boost_weight_min": 0.0,
                "boost_weight_max": 0.0,
                "grid_size": 2,
                "patch_per_row": 2,
                "global_current_weight": 0.5,
                "global_ema_decay": 0.6,
            }
        )
        strategy.prepare_sample(
            inputs_embeds=_make_grid_inputs(4),
            v_token_start=1,
            v_token_num=4,
            text_token_start=5,
            text_token_ids=torch.tensor([11, 12]),
            text_special_token_mask=torch.tensor([False, False]),
        )
        layer0_attn = _full_attention(
            seq_len=7,
            text_rows=[5, 6],
            visual_cols=[1, 2, 3, 4],
            values=[[0.9, 0.6, 0.4, 0.1], [0.8, 0.6, 0.4, 0.1]],
        )
        keep0, info0 = strategy.compute_keep_mask(layer0_attn, 1, 4, 5, 0)
        strategy.update_after_prune(keep0, 0)

        layer1_attn = _full_attention(
            seq_len=6,
            text_rows=[4, 5],
            visual_cols=[1, 2, 3],
            values=[[0.1, 0.9, 0.2], [0.1, 0.8, 0.2]],
        )
        _keep1, info1 = strategy.compute_keep_mask(layer1_attn, 1, 3, 4, 1)
        current = torch.tensor(info1["current_rank_score"])
        ema = torch.tensor(info1["global_saliency_ema"])
        expected = 0.5 * current + 0.5 * ema

        self.assertFalse(info0["global_use_ema"])
        self.assertTrue(info1["global_use_ema"])
        self.assertTrue(torch.allclose(torch.tensor(info1["mixed_score"]), expected))

    def test_boost_can_disable_score_memory(self):
        strategy = SparseVLMBoostStrategy(
            {
                "prune_ratio": 0.5,
                "prune_ratio_map": {0: 0.25, 1: 0.5},
                "fallback_topk": 4,
                "exclude_special_tokens": True,
                "min_visual_tokens_after_prune": 1,
                "boost_weight_min": 0.0,
                "boost_weight_max": 0.0,
                "grid_size": 2,
                "patch_per_row": 2,
                "global_current_weight": 0.0,
                "global_ema_decay": 0.6,
                "use_score_memory": False,
            }
        )
        strategy.prepare_sample(
            inputs_embeds=_make_grid_inputs(4),
            v_token_start=1,
            v_token_num=4,
            text_token_start=5,
            text_token_ids=torch.tensor([11, 12]),
            text_special_token_mask=torch.tensor([False, False]),
        )
        layer0_attn = _full_attention(
            seq_len=7,
            text_rows=[5, 6],
            visual_cols=[1, 2, 3, 4],
            values=[[0.9, 0.6, 0.4, 0.1], [0.8, 0.6, 0.4, 0.1]],
        )
        keep0, _info0 = strategy.compute_keep_mask(layer0_attn, 1, 4, 5, 0)
        strategy.update_after_prune(keep0, 0)

        layer1_attn = _full_attention(
            seq_len=6,
            text_rows=[4, 5],
            visual_cols=[1, 2, 3],
            values=[[0.1, 0.9, 0.2], [0.1, 0.8, 0.2]],
        )
        _keep1, info1 = strategy.compute_keep_mask(layer1_attn, 1, 3, 4, 1)

        self.assertFalse(info1["global_use_ema"])
        self.assertFalse(info1["use_score_memory"])
        self.assertTrue(torch.allclose(torch.tensor(info1["mixed_score"]), torch.tensor(info1["current_rank_score"])))

    def test_hybrid_rejects_layer_mode_length_mismatch(self):
        strategy = SparseVLMBoostHybridStrategy(
            {
                "prune_layers": [0, 1],
                "layer_modes": ["S"],
                "prune_ratio": [0.5, 0.5],
            }
        )

        with self.assertRaisesRegex(ValueError, "same length"):
            strategy._layer_mode_map()

    def test_hybrid_all_s_matches_sparsevlm_topk(self):
        attn = _full_attention(
            seq_len=7,
            text_rows=[5, 6],
            visual_cols=[1, 2, 3, 4],
            values=[[0.1, 0.8, 0.4, 0.2], [0.1, 0.7, 0.3, 0.2]],
        )
        base = SparseVLMStrategy(
            {
                "prune_ratio": 0.5,
                "fallback_topk": 4,
                "exclude_special_tokens": True,
                "min_visual_tokens_after_prune": 1,
            }
        )
        hybrid = SparseVLMBoostHybridStrategy(
            {
                "prune_layers": [0],
                "layer_modes": ["S"],
                "prune_ratio": 0.5,
                "fallback_topk": 4,
                "exclude_special_tokens": True,
                "min_visual_tokens_after_prune": 1,
                "grid_size": 2,
                "patch_per_row": 2,
                "use_score_memory": False,
            }
        )
        for strategy in (base, hybrid):
            strategy.prepare_sample(
                inputs_embeds=_make_grid_inputs(4),
                v_token_start=1,
                v_token_num=4,
                text_token_start=5,
                text_token_ids=torch.tensor([11, 12]),
                text_special_token_mask=torch.tensor([False, False]),
            )

        base_keep, _base_info = base.compute_keep_mask(
            attn,
            1,
            4,
            5,
            0,
            current_visual_embeds=_make_grid_inputs(4)[0, 1:5],
        )
        hybrid_keep, hybrid_info = hybrid.compute_keep_mask(attn, 1, 4, 5, 0)

        self.assertEqual(hybrid_keep.tolist(), base_keep.tolist())
        self.assertEqual(hybrid_info["layer_mode"], "S")
        self.assertEqual(hybrid_info["layer_strategy_effective"], "sparsevlm")

    def test_hybrid_all_o_uses_current_layer_without_memory(self):
        strategy = SparseVLMBoostHybridStrategy(
            {
                "prune_layers": [0, 1],
                "layer_modes": ["O", "O"],
                "prune_ratio_map": {0: 0.25, 1: 0.5},
                "fallback_topk": 4,
                "exclude_special_tokens": True,
                "min_visual_tokens_after_prune": 1,
                "boost_weight_min": 0.0,
                "boost_weight_max": 0.0,
                "grid_size": 2,
                "patch_per_row": 2,
                "global_current_weight": 0.0,
                "use_score_memory": False,
            }
        )
        strategy.prepare_sample(
            inputs_embeds=_make_grid_inputs(4),
            v_token_start=1,
            v_token_num=4,
            text_token_start=5,
            text_token_ids=torch.tensor([11, 12]),
            text_special_token_mask=torch.tensor([False, False]),
        )
        layer0_attn = _full_attention(
            seq_len=7,
            text_rows=[5, 6],
            visual_cols=[1, 2, 3, 4],
            values=[[0.9, 0.6, 0.4, 0.1], [0.8, 0.6, 0.4, 0.1]],
        )
        keep0, _info0 = strategy.compute_keep_mask(layer0_attn, 1, 4, 5, 0)
        strategy.update_after_prune(keep0, 0)
        layer1_attn = _full_attention(
            seq_len=6,
            text_rows=[4, 5],
            visual_cols=[1, 2, 3],
            values=[[0.1, 0.9, 0.2], [0.1, 0.8, 0.2]],
        )
        _keep1, info1 = strategy.compute_keep_mask(layer1_attn, 1, 3, 4, 1)

        self.assertEqual(info1["layer_mode"], "O")
        self.assertEqual(info1["layer_strategy_effective"], "sparsevlm_boost")
        self.assertFalse(info1["global_use_ema"])
        self.assertFalse(info1["use_score_memory"])
        self.assertTrue(torch.allclose(torch.tensor(info1["mixed_score"]), torch.tensor(info1["current_rank_score"])))

    def test_hybrid_oos_marks_third_layer_as_sparsevlm(self):
        attn = _full_attention(
            seq_len=7,
            text_rows=[5, 6],
            visual_cols=[1, 2, 3, 4],
            values=[[0.1, 0.8, 0.4, 0.2], [0.1, 0.7, 0.3, 0.2]],
        )
        strategy = SparseVLMBoostHybridStrategy(
            {
                "prune_layers": [2, 6, 15],
                "layer_modes": ["O", "O", "S"],
                "prune_ratio": 0.5,
                "fallback_topk": 4,
                "exclude_special_tokens": True,
                "min_visual_tokens_after_prune": 1,
                "grid_size": 2,
                "patch_per_row": 2,
                "use_score_memory": False,
            }
        )
        strategy.prepare_sample(
            inputs_embeds=_make_grid_inputs(4),
            v_token_start=1,
            v_token_num=4,
            text_token_start=5,
            text_token_ids=torch.tensor([11, 12]),
            text_special_token_mask=torch.tensor([False, False]),
        )
        _keep, info = strategy.compute_keep_mask(attn, 1, 4, 5, 15)

        self.assertEqual(info["layer_mode"], "S")
        self.assertEqual(info["layer_strategy_effective"], "sparsevlm")
        self.assertNotIn("boost_weight", info)

    def test_compensated_fixed_seed_is_reproducible(self):
        layer0_attn = _full_attention(
            seq_len=7,
            text_rows=[5, 6],
            visual_cols=[1, 2, 3, 4],
            values=[[0.1, 0.8, 0.4, 0.2], [0.1, 0.7, 0.3, 0.2]],
        )
        config = {
            "prune_ratio": 0.5,
            "fallback_topk": 4,
            "exclude_special_tokens": True,
            "min_visual_tokens_after_prune": 1,
            "beta_min": 0.3,
            "beta_max": 1.0,
            "seed": 123,
        }
        outputs = []
        for _ in range(2):
            strategy = SparseVLMCompensatedStrategy(dict(config))
            strategy.prepare_sample(
                inputs_embeds=_make_grid_inputs(4),
                v_token_start=1,
                v_token_num=4,
                text_token_start=5,
                text_token_ids=torch.tensor([11, 12]),
                text_special_token_mask=torch.tensor([False, False]),
            )
            keep_indices, info = strategy.compute_keep_mask(layer0_attn, 1, 4, 5, 0)
            outputs.append((keep_indices.tolist(), info["shuffle_seed"]))

        self.assertEqual(outputs[0], outputs[1])

    def test_compensated_beta_extremes_affect_weights(self):
        mixed_score = torch.tensor([1.0, 0.5, 0.0, 0.25])
        uniformish = SparseVLMCompensatedStrategy._sampling_weights(mixed_score, 0.0)
        saliency_weighted = SparseVLMCompensatedStrategy._sampling_weights(mixed_score, 1.0)

        self.assertEqual(uniformish.tolist(), [1.0, 1.0, 0.0, 1.0])
        self.assertTrue(torch.allclose(saliency_weighted, mixed_score))

    def test_compensated_does_not_require_grid_or_positions_config(self):
        layer0_attn = _full_attention(
            seq_len=7,
            text_rows=[5, 6],
            visual_cols=[1, 2, 3, 4],
            values=[[0.1, 0.8, 0.4, 0.2], [0.1, 0.7, 0.3, 0.2]],
        )
        strategy = SparseVLMCompensatedStrategy(
            {
                "prune_ratio": 0.5,
                "fallback_topk": 4,
                "exclude_special_tokens": True,
                "min_visual_tokens_after_prune": 1,
                "beta_min": 0.3,
                "beta_max": 1.0,
                "seed": 42,
            }
        )
        strategy.prepare_sample(
            inputs_embeds=_make_grid_inputs(4),
            v_token_start=1,
            v_token_num=4,
            text_token_start=5,
            text_token_ids=torch.tensor([11, 12]),
            text_special_token_mask=torch.tensor([False, False]),
        )
        keep_indices, info = strategy.compute_keep_mask(layer0_attn, 1, 4, 5, 0)

        self.assertEqual(len(keep_indices), 2)
        self.assertNotIn("grid_size", info)
        self.assertNotIn("stratum_deficit", info)
        self.assertIn("sampling_weights", info)

    def test_update_after_prune_updates_patch_mapping_and_ema(self):
        strategy = SparseVLMCompensatedStrategy(
            {
                "prune_ratio": 0.5,
                "fallback_topk": 4,
                "exclude_special_tokens": True,
                "min_visual_tokens_after_prune": 1,
                "beta_min": 0.3,
                "beta_max": 1.0,
                "seed": 42,
            }
        )
        strategy.prepare_sample(
            inputs_embeds=_make_grid_inputs(4),
            v_token_start=1,
            v_token_num=4,
            text_token_start=5,
            text_token_ids=torch.tensor([11, 12]),
            text_special_token_mask=torch.tensor([False, False]),
        )
        layer0_attn = _full_attention(
            seq_len=7,
            text_rows=[5, 6],
            visual_cols=[1, 2, 3, 4],
            values=[[0.1, 0.8, 0.4, 0.2], [0.1, 0.7, 0.3, 0.2]],
        )
        keep_indices, _info = strategy.compute_keep_mask(layer0_attn, 1, 4, 5, 0)
        strategy.update_after_prune(keep_indices, 0)
        context = strategy._require_sample_context()

        self.assertEqual(context["current_patch_indices"].tolist(), keep_indices.tolist())
        self.assertEqual(int(context["score_prune_step"]), 1)
        self.assertTrue(context["score_memory_observed"].all().item())

    def test_rank_normalize_scores_is_deterministic(self):
        scores = torch.tensor([0.2, 0.7, 0.7, 0.1])
        normalized = rank_normalize_scores(scores)
        expected = torch.tensor([1 / 3, 1.0, 2 / 3, 0.0], dtype=torch.float32)
        self.assertTrue(torch.allclose(normalized, expected))


if __name__ == "__main__":
    unittest.main()
