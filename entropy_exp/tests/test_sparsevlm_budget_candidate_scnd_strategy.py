import os
import sys
import unittest

import torch

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC_DIR)

from strategies import get_strategy
from strategies.sparsevlm_budget_candidate_scnd import SparseVLMBudgetCandidateSCNDStrategy


def _full_attention(seq_len, text_rows, visual_cols, values):
    attn = torch.zeros((1, 1, seq_len, seq_len), dtype=torch.float32)
    for row, row_values in zip(text_rows, values):
        attn[0, 0, row, visual_cols] = torch.tensor(row_values, dtype=torch.float32)
    return attn


def _inputs(num_visual_tokens, dim=4):
    return torch.zeros((1, 1 + num_visual_tokens + 2, dim), dtype=torch.float32)


class SparseVLMBudgetCandidateSCNDTests(unittest.TestCase):
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
            "layer_modes": ["C"],
            "prune_ratio": 0.5,
            "fallback_topk": 4,
            "exclude_special_tokens": True,
            "min_visual_tokens_after_prune": 1,
            "budget_target": "retain128",
            "layer_policy": "auto",
            "seed_ratio_min": 0.15,
            "seed_ratio_max": 0.55,
            "seed_pool_multiplier": 2.0,
            "saliency_repair": True,
            "distance_metric": "cosine",
            "use_score_memory": False,
        }
        config.update(overrides)
        return SparseVLMBudgetCandidateSCNDStrategy(config)

    def test_registry_instantiates_strategy(self):
        self.assertIsInstance(
            get_strategy("sparsevlm_budget_candidate_scnd", {}),
            SparseVLMBudgetCandidateSCNDStrategy,
        )

    def test_auto_budget_target_inference_and_unknown_ratio(self):
        strategy = self._strategy(
            budget_target="auto",
            prune_layers=[2, 6, 16],
            layer_modes=None,
            prune_ratio=[0.4791666666666667, 0.3333333333333333, 0.45],
        )
        self.assertEqual(strategy._resolve_budget_target(), "retain192")
        self.assertEqual(strategy._layer_mode_map(), {2: "C", 6: "B", 16: "S"})

        bad = self._strategy(budget_target="auto", prune_ratio=[0.1, 0.2, 0.3])
        with self.assertRaises(ValueError):
            bad._resolve_budget_target()

    def test_auto_layer_policy_for_three_targets(self):
        ratios = {
            "retain192": [0.4791666666666667, 0.3333333333333333, 0.45],
            "retain128": [0.4739583333333333, 0.636963696369637, 0.6727272727272727],
            "retain64": [0.8854166666666666, 0.5454545454545454, 0.43333333333333335],
        }
        expected = {
            "retain192": {2: "C", 6: "B", 16: "S"},
            "retain128": {2: "C", 6: "B", 16: "B"},
            "retain64": {2: "C", 6: "B", 16: "B"},
        }
        for target, ratio in ratios.items():
            strategy = self._strategy(
                budget_target="auto",
                prune_layers=[2, 6, 16],
                layer_modes=None,
                prune_ratio=ratio,
            )
            self.assertEqual(strategy._resolve_budget_target(), target)
            self.assertEqual(strategy._layer_mode_map(), expected[target])

    def test_candidate_c_keep_count_pool_cap_and_selection_source(self):
        strategy = self._strategy(budget_target="retain192", prune_ratio=0.8, saliency_repair=True)
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
        generator.manual_seed(13)
        embeds = torch.randn(num_tokens, 8, generator=generator)
        keep, info = strategy.compute_keep_mask(attn, 1, num_tokens, 1 + num_tokens, 0, current_visual_embeds=embeds)

        target_keep = int(info["target_keep"])
        self.assertEqual(int(keep.numel()), target_keep)
        self.assertEqual(info["selection_rule"], "budget_candidate_scnd")
        self.assertEqual(info["profile_version"], "v3")
        self.assertLessEqual(int(info["candidate_pool_size"]), int(info["candidate_cap_count"]))
        self.assertGreater(int(info["global_reservoir_count"]), 0)
        self.assertTrue(set(keep.tolist()).issubset(set(info["candidate_pool_indices"].tolist())))
        self.assertTrue(all(int(idx) < int(info["reservoir_source_count"]) for idx in info["reservoir_indices"]))
        global_only = set(info["global_reservoir_indices"].tolist()) - set(info["reservoir_source_indices"].tolist())
        self.assertTrue(global_only)

    def test_retain128_candidate_pool_expands_without_saturating_and_reserves_global_reservoir(self):
        strategy = self._strategy(budget_target="retain128")
        profile = strategy._profile("retain128")
        generator = torch.Generator()
        generator.manual_seed(7)
        mixed = torch.linspace(1.0, 0.0, steps=576)
        projected = torch.nn.functional.normalize(torch.randn(576, 32, generator=generator), dim=-1)

        pool = strategy._build_candidate_pool(
            mixed_score=mixed,
            projected=projected,
            target_keep=100,
            profile=profile,
        )

        self.assertLess(len(pool["candidate_pool_indices"]), 576)
        self.assertGreaterEqual(len(pool["candidate_pool_indices"]), int(0.90 * 576))
        self.assertLessEqual(len(pool["candidate_pool_indices"]), int(pool["candidate_cap"]))
        reservoir_only = set(pool["reservoir_indices"]) - set(pool["saliency_pool_indices"])
        if reservoir_only:
            self.assertTrue(reservoir_only & set(pool["candidate_pool_indices"]))
            self.assertGreater(int(pool["candidate_pool_reservoir_only_count"]), 0)
        global_outside_source = set(pool["global_reservoir_indices"]) - set(pool["reservoir_source_indices"])
        self.assertTrue(global_outside_source)
        self.assertTrue(global_outside_source & set(pool["candidate_pool_indices"]))
        self.assertGreater(int(pool["candidate_pool_global_reservoir_only_count"]), 0)

    def test_diversity_steps_are_target_aware_not_fixed_sixteen(self):
        strategy = self._strategy()
        params = strategy._params()
        mixed = torch.linspace(1.0, 0.0, steps=24)
        projected = torch.nn.functional.normalize(torch.eye(24, 24), dim=-1)
        budgets = []
        for target in ("retain192", "retain128", "retain64"):
            profile = strategy._profile(target)
            selection = strategy._candidate_select(
                mixed_score=mixed,
                projected=projected,
                target_keep=10,
                entropy_norm=1.0,
                margin_confidence=0.0,
                profile=profile,
                params=params,
            )
            budgets.append(int(selection["diversity_steps_budget"]))
        self.assertEqual(budgets, [7, 8, 9])
        self.assertGreater(budgets[2], budgets[1])
        self.assertNotIn(16, budgets)

    def test_saliency_core_is_always_kept_in_candidate_c(self):
        strategy = self._strategy(budget_target="retain192", prune_ratio=0.5)
        num_tokens = 12
        self._prepare(strategy, num_tokens)
        scores = [float(num_tokens - idx) for idx in range(num_tokens)]
        attn = _full_attention(
            seq_len=1 + num_tokens + 2,
            text_rows=[1 + num_tokens, 2 + num_tokens],
            visual_cols=list(range(1, 1 + num_tokens)),
            values=[scores, scores],
        )
        embeds = torch.nn.functional.normalize(torch.eye(num_tokens, num_tokens), dim=-1)
        keep, info = strategy.compute_keep_mask(attn, 1, num_tokens, 1 + num_tokens, 0, current_visual_embeds=embeds)

        core = set(info["saliency_core_indices"].tolist())
        self.assertGreater(len(core), 0)
        self.assertTrue(core.issubset(set(keep.tolist())))
        self.assertEqual(int(info["saliency_core_count"]), len(core))

    def test_b_lite_uses_cached_gain_without_visual_embeds(self):
        strategy = self._strategy(
            layer_modes=["B"],
            prune_ratio=0.5,
            boundary_ratio=0.5,
            boundary_gate_min=0.0,
            cached_diversity_weight=1.0,
        )
        self._prepare(strategy, 4)
        context = strategy._require_sample_context()
        context["budget_candidate_scnd_diversity_gain"] = torch.tensor([0.0, 0.0, 1.0, 0.0])
        attn = _full_attention(
            seq_len=7,
            text_rows=[5, 6],
            visual_cols=[1, 2, 3, 4],
            values=[[0.90, 0.89, 0.88, 0.87], [0.90, 0.89, 0.88, 0.87]],
        )
        keep, info = strategy.compute_keep_mask(attn, 1, 4, 5, 0)

        self.assertEqual(info["selection_rule"], "budget_candidate_scnd_blite")
        self.assertEqual(info["blite_mode"], "cached_boundary")
        self.assertEqual(int(info["boundary_count"]), 1)
        self.assertTrue(info["blite_enabled"])
        self.assertEqual(info["boundary_fill_indices"].tolist(), [2])
        self.assertEqual(keep.tolist(), [0, 2])
        self.assertIn("cached_diversity_gain", info)
        self.assertNotIn("mean_selected_pairwise_distance", info)

    def test_b_native_boundary_uses_visual_embeds_without_cached_gain(self):
        strategy = self._strategy(
            layer_modes=["B"],
            prune_ratio=0.5,
            boundary_ratio=0.5,
            boundary_gate_min=0.0,
        )
        self._prepare(strategy, 4, dim=2)
        attn = _full_attention(
            seq_len=7,
            text_rows=[5, 6],
            visual_cols=[1, 2, 3, 4],
            values=[[0.90, 0.89, 0.88, 0.87], [0.90, 0.89, 0.88, 0.87]],
        )
        embeds = torch.tensor(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [-1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        keep, info = strategy.compute_keep_mask(attn, 1, 4, 5, 0, current_visual_embeds=embeds)

        self.assertEqual(info["selection_rule"], "budget_candidate_scnd_blite")
        self.assertEqual(info["blite_mode"], "native_boundary")
        self.assertTrue(info["native_boundary_used"])
        self.assertTrue(info["blite_enabled"])
        self.assertEqual(info["boundary_fill_indices"].tolist(), [2])
        self.assertEqual(keep.tolist(), [0, 2])
        self.assertGreater(int(info["distance_cost_proxy"]), 0)

    def test_b_lite_disables_when_cached_gain_is_uninformative(self):
        strategy = self._strategy(
            layer_modes=["B"],
            prune_ratio=0.5,
            boundary_ratio=0.5,
            boundary_gate_min=0.0,
            cached_diversity_weight=1.0,
        )
        self._prepare(strategy, 4)
        attn = _full_attention(
            seq_len=7,
            text_rows=[5, 6],
            visual_cols=[1, 2, 3, 4],
            values=[[0.90, 0.89, 0.88, 0.87], [0.90, 0.89, 0.88, 0.87]],
        )
        keep, info = strategy.compute_keep_mask(attn, 1, 4, 5, 0)

        self.assertEqual(info["selection_rule"], "budget_candidate_scnd_blite")
        self.assertFalse(info["blite_enabled"])
        self.assertEqual(info["blite_disabled_reason"], "no_cached_diversity_gain")
        self.assertEqual(info["blite_mode"], "none")
        self.assertEqual(keep.tolist(), [0, 1])

    def test_s_layer_matches_sparsevlm_branch(self):
        strategy = self._strategy(
            prune_layers=[2, 6],
            layer_modes=["C", "S"],
            prune_ratio_map={2: 0.25, 6: 0.34},
            budget_target="retain128",
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
        self.assertEqual(info6["budget_target"], "retain128")


if __name__ == "__main__":
    unittest.main()
