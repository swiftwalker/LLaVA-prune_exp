import os
import sys
import unittest

import torch

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC_DIR)

from strategies import get_strategy
from strategies.sparsevlm_entropy_alpha import SparseVLMEntropyAlphaStrategy
from strategies.sparsevlm_entropy_alpha_global import (
    SparseVLMEntropyAlphaGlobalStrategy,
    allocate_debt_aware_stratified_quotas,
)


def _full_attention(seq_len, text_rows, visual_cols, values):
    attn = torch.zeros((1, 1, seq_len, seq_len), dtype=torch.float32)
    for row, row_values in zip(text_rows, values):
        attn[0, 0, row, visual_cols] = torch.tensor(row_values, dtype=torch.float32)
    return attn


class SparseVLMEntropyAlphaGlobalStrategyTests(unittest.TestCase):
    def _make_config(self, **overrides):
        config = {
            "prune_ratio": 0.5,
            "fallback_topk": 4,
            "exclude_special_tokens": True,
            "min_visual_tokens_after_prune": 1,
            "grid_size": 2,
            "patch_per_row": 2,
            "intra_stratum_mode": "random",
            "alpha_min": 0.5,
            "alpha_max": 1.0,
            "global_current_weight": 0.5,
            "global_ema_decay": 0.6,
            "global_debt_weight": 0.5,
            "global_first_layer_fallback": True,
        }
        config.update(overrides)
        return config

    def _make_strategy(self, **overrides):
        return SparseVLMEntropyAlphaGlobalStrategy(self._make_config(**overrides))

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

    def test_first_layer_fallback_matches_entropy_alpha(self):
        config = self._make_config()
        global_strategy = SparseVLMEntropyAlphaGlobalStrategy(config.copy())
        local_strategy = SparseVLMEntropyAlphaStrategy(config.copy())
        inputs = self._make_inputs()
        attn = _full_attention(
            seq_len=7,
            text_rows=[5, 6],
            visual_cols=[1, 2, 3, 4],
            values=[[0.9, 0.2, 0.1, 0.05], [0.3, 0.2, 0.2, 0.1]],
        )

        for strategy in (global_strategy, local_strategy):
            strategy.prepare_sample(
                inputs_embeds=inputs,
                v_token_start=1,
                v_token_num=4,
                text_token_start=5,
                text_token_ids=torch.tensor([11, 12]),
                text_special_token_mask=torch.tensor([False, False]),
            )

        global_keep, global_info = global_strategy.compute_keep_mask(
            attn_weights=attn,
            v_token_start=1,
            v_token_num=4,
            text_token_start=5,
            layer_idx=0,
        )
        local_keep, local_info = local_strategy.compute_keep_mask(
            attn_weights=attn,
            v_token_start=1,
            v_token_num=4,
            text_token_start=5,
            layer_idx=0,
        )

        self.assertEqual(global_keep.tolist(), local_keep.tolist())
        self.assertEqual(global_info["high_keep_indices"].tolist(), local_info["high_keep_indices"].tolist())
        self.assertEqual(global_info["low_keep_indices"].tolist(), local_info["low_keep_indices"].tolist())
        self.assertFalse(global_info["global_use_global"])

    def test_second_layer_uses_current_and_global_selection_score(self):
        strategy = self._make_strategy(
            prune_ratio=0.5,
            alpha_min=1.0,
            alpha_max=1.0,
            global_current_weight=0.2,
            global_ema_decay=1.0,
        )
        strategy.prepare_sample(
            inputs_embeds=self._make_inputs(),
            v_token_start=1,
            v_token_num=4,
            text_token_start=5,
            text_token_ids=torch.tensor([11, 12]),
            text_special_token_mask=torch.tensor([False, False]),
        )
        context = strategy._require_sample_context()
        context["global_prune_step"] = 1
        context["current_patch_indices"] = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        context["global_saliency_ema"] = torch.tensor([1.0, 0.9, 0.0, 0.0], dtype=torch.float32)
        context["global_saliency_observed"] = torch.tensor([True, True, True, True])
        attn = _full_attention(
            seq_len=7,
            text_rows=[5, 6],
            visual_cols=[1, 2, 3, 4],
            values=[[0.1, 0.2, 0.9, 0.8], [0.1, 0.2, 0.9, 0.8]],
        )

        keep_indices, info = strategy.compute_keep_mask(
            attn_weights=attn,
            v_token_start=1,
            v_token_num=4,
            text_token_start=5,
            layer_idx=1,
        )

        self.assertTrue(info["global_use_global"])
        self.assertEqual(info["global_prune_step"], 1)
        self.assertEqual(int(info["strategy_keep_high"]), 2)
        # Current scores favor tokens 2/3, but cross-layer EMA lifts token 0 into the high set.
        self.assertIn(0, info["high_keep_indices"].tolist())
        self.assertEqual(len(keep_indices), info["target_keep"])
        self.assertIn("global_selection_score", info)
        self.assertIn("global_saliency_ema", info)
        self.assertIn("current_saliency_norm", info)

    def test_historical_debt_allocation_prioritizes_undercovered_stratum(self):
        quotas = allocate_debt_aware_stratified_quotas(
            debts={0: 0.0, 1: 5.0, 2: 0.0, 3: 1.0},
            candidate_counts={0: 2, 1: 2, 2: 2, 3: 2},
            n_low=2,
            grid_size=2,
        )

        self.assertEqual(quotas[1], 2)
        self.assertEqual(sum(quotas.values()), 2)

    def test_update_after_prune_updates_global_state(self):
        strategy = self._make_strategy()
        strategy.prepare_sample(
            inputs_embeds=self._make_inputs(),
            v_token_start=1,
            v_token_num=4,
            text_token_start=5,
            text_token_ids=torch.tensor([11, 12]),
            text_special_token_mask=torch.tensor([False, False]),
        )
        context = strategy._require_sample_context()
        context["global_pending_decision"] = {
            "layer_idx": 0,
            "current_patch_indices": torch.tensor([0, 1, 2, 3], dtype=torch.long),
            "grid_size": 2,
            "patch_per_row": 2,
        }

        strategy.update_after_prune(keep_indices=torch.tensor([0, 2]), layer_idx=0)

        self.assertEqual(context["current_patch_indices"].tolist(), [0, 2])
        self.assertEqual(context["global_prune_step"], 1)
        self.assertEqual(context["global_drop_layer"].tolist(), [-1, 0, -1, 0])
        self.assertEqual(float(context["global_stratum_keep_exposure"].sum().item()), 2.0)
        self.assertIsNone(context["global_pending_decision"])

    def test_registry_instantiates_global_strategy(self):
        strategy = get_strategy(
            "sparsevlm_entropy_alpha_global",
            {
                "prune_ratio": 0.5,
                "grid_size": 6,
                "patch_per_row": 24,
                "intra_stratum_mode": "random",
            },
        )

        self.assertIsInstance(strategy, SparseVLMEntropyAlphaGlobalStrategy)


if __name__ == "__main__":
    unittest.main()
