import os
import sys
import unittest
from unittest import mock

import torch


SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC_DIR)

from pruner import _make_causal_mask, enable_sparse_position_ids_compat  # noqa: E402
from strategies.scnd_counterfactual import (  # noqa: E402
    NextLayerRoutingContext,
    build_next_layer_preview,
    compute_visual_effect_errors,
    global_maxmin_candidate_exact,
    numerical_tolerance,
    pareto_dominates_legacy,
    select_candidate_deterministically,
)
from strategies.sparsevlm_score_memory import deterministic_topk_indices  # noqa: E402
from strategies.sparsevlm_scnd import SparseVLMSCNDStrategy  # noqa: E402


def _tiny_decoder(family: str, num_key_value_heads: int):
    if family == "llama":
        from transformers import LlamaConfig, LlamaForCausalLM

        config_cls = LlamaConfig
        model_cls = LlamaForCausalLM
    elif family == "mistral":
        from transformers import MistralConfig, MistralForCausalLM

        config_cls = MistralConfig
        model_cls = MistralForCausalLM
    else:
        raise ValueError(f"Unsupported test family: {family}")
    config = config_cls(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=num_key_value_heads,
        max_position_embeddings=128,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    model = model_cls(config).eval()
    if not enable_sparse_position_ids_compat(model):
        raise AssertionError(f"Could not enable sparse position IDs for {family}")
    return model


def _physical_query_output(context: NextLayerRoutingContext, visual_keep: torch.Tensor) -> torch.Tensor:
    seq_len = int(context.hidden_states.shape[1])
    full_keep = torch.ones(seq_len, device=context.hidden_states.device, dtype=torch.bool)
    visual_end = int(context.visual_start + context.visual_count)
    full_keep[context.visual_start:visual_end] = False
    if int(visual_keep.numel()) > 0:
        full_keep[context.visual_start + visual_keep] = True

    hidden = context.hidden_states[:, full_keep]
    position_ids = context.position_ids[:, full_keep]
    mask = _make_causal_mask(int(hidden.shape[1]), hidden.dtype, hidden.device)
    output = context.next_layer(
        hidden,
        attention_mask=mask,
        position_ids=position_ids,
        use_cache=False,
        output_attentions=False,
    )[0]

    original_queries = torch.arange(
        context.text_start,
        seq_len,
        device=context.hidden_states.device,
        dtype=torch.long,
    )
    if context.text_special_token_mask is not None:
        original_queries = original_queries[~context.text_special_token_mask]
    mapped_queries = full_keep.to(dtype=torch.long).cumsum(dim=0).index_select(
        0,
        original_queries,
    ) - 1
    return output.index_select(1, mapped_queries)


class NextLayerPreviewTests(unittest.TestCase):
    def _context(self, family: str, num_key_value_heads: int, *, device="cpu", dtype=torch.float32):
        torch.manual_seed(13)
        model = _tiny_decoder(family, num_key_value_heads).to(device=device, dtype=dtype)
        hidden = torch.randn(1, 12, 32, device=device, dtype=dtype)
        position_ids = torch.tensor(
            [[0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13]],
            device=device,
            dtype=torch.long,
        )
        return NextLayerRoutingContext(
            current_layer_idx=0,
            next_layer=model.model.layers[1],
            hidden_states=hidden,
            position_ids=position_ids,
            causal_mask=_make_causal_mask(12, dtype, torch.device(device)),
            visual_start=2,
            visual_count=6,
            text_start=8,
            text_special_token_mask=torch.tensor(
                [True, False, False, False],
                device=device,
                dtype=torch.bool,
            ),
        )

    def test_preview_matches_real_pruned_layer_for_llama_and_mistral_mha_gqa(self):
        for family in ("llama", "mistral"):
            for num_key_value_heads in (4, 2):
                with self.subTest(family=family, num_key_value_heads=num_key_value_heads):
                    context = self._context(family, num_key_value_heads)
                    preview = build_next_layer_preview(context)
                    self.assertIsNotNone(preview)
                    for keep in (
                        torch.arange(6),
                        torch.tensor([0, 2, 5]),
                        torch.empty(0, dtype=torch.long),
                    ):
                        with torch.no_grad():
                            actual = preview.evaluate(keep)
                            expected = _physical_query_output(context, keep)
                        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_full_reference_has_zero_error_and_joint_deletion_is_nonlinear(self):
        context = self._context("llama", 2)
        preview = build_next_layer_preview(context)
        full = preview.evaluate(torch.arange(6))
        empty = preview.evaluate(torch.empty(0, dtype=torch.long))
        full_error = compute_visual_effect_errors(
            phi_keep=full,
            phi_empty=empty,
            phi_full=full,
        )
        self.assertEqual(full_error["error_sum"], 0.0)
        self.assertEqual(full_error["error_max"], 0.0)

        without_zero = preview.evaluate(torch.tensor([1, 2, 3, 4, 5]))
        without_one = preview.evaluate(torch.tensor([0, 2, 3, 4, 5]))
        without_both = preview.evaluate(torch.tensor([2, 3, 4, 5]))
        additive_prediction = without_zero + without_one - full
        self.assertGreater(float((without_both - additive_prediction).abs().max().item()), 0.0)

    def test_all_special_text_queries_return_no_preview(self):
        context = self._context("llama", 2)
        context = NextLayerRoutingContext(
            **{
                **context.__dict__,
                "text_special_token_mask": torch.ones(4, dtype=torch.bool),
            }
        )
        self.assertIsNone(build_next_layer_preview(context))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for mixed-precision preview tests")
    def test_cuda_fp16_and_bf16_match_real_pruned_layer(self):
        for dtype, tolerance in ((torch.float16, 2e-3), (torch.bfloat16, 2e-2)):
            with self.subTest(dtype=dtype):
                context = self._context("llama", 2, device="cuda:0", dtype=dtype)
                preview = build_next_layer_preview(context)
                keep = torch.tensor([0, 2, 5], device="cuda:0")
                with torch.no_grad():
                    actual = preview.evaluate(keep)
                    expected = _physical_query_output(context, keep)
                torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


class CounterfactualDecisionTests(unittest.TestCase):
    def test_host_global_maxmin_respects_gain_saliency_index_ties(self):
        distance = torch.tensor(
            [
                [0.0, 0.5, 0.9, 0.9],
                [0.5, 0.0, 0.7, 0.7],
                [0.9, 0.7, 0.0, 0.2],
                [0.9, 0.7, 0.2, 0.0],
            ],
            dtype=torch.float32,
        )
        saliency = torch.tensor([0.1, 1.0, 0.8, 0.8], dtype=torch.float32)
        selected = global_maxmin_candidate_exact(distance, saliency, 4)
        self.assertEqual(selected.tolist(), [1, 2, 0, 3])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for GPU selector comparison")
    def test_host_global_maxmin_matches_gpu_selector(self):
        torch.manual_seed(23)
        features = torch.randn(128, 32, device="cuda:0")
        features = torch.nn.functional.normalize(features, dim=-1)
        distance = (1.0 - features @ features.T) / 2.0
        saliency = torch.rand(128, device="cuda:0")
        strategy = CounterfactualStrategyIntegrationTests._strategy()
        expected, _ = strategy._max_min_select_from_pool_gpu(
            distance,
            saliency,
            torch.arange(128, device="cuda:0"),
            32,
        )
        actual = global_maxmin_candidate_exact(distance, saliency, 32)
        torch.testing.assert_close(actual, expected)

    def test_numerical_tolerance_and_pareto_rule(self):
        legacy = {"error_sum": 0.40, "error_max": 0.50}
        self.assertGreater(numerical_tolerance(0.4, 0.5), 0.0)
        self.assertTrue(
            pareto_dominates_legacy(
                {"error_sum": 0.39, "error_max": 0.50},
                legacy,
            )
        )
        self.assertFalse(pareto_dominates_legacy(dict(legacy), legacy))
        self.assertFalse(
            pareto_dominates_legacy(
                {"error_sum": 0.39, "error_max": 0.51},
                legacy,
            )
        )

    def test_candidate_choice_uses_error_max_then_sum_then_fixed_order(self):
        errors = {
            "legacy_scnd": {"error_sum": 0.5, "error_max": 0.5},
            "sparsevlm_topk": {"error_sum": 0.4, "error_max": 0.4},
            "native_maxmin": {"error_sum": 0.3, "error_max": 0.4},
        }
        selected, admissible = select_candidate_deterministically(
            errors,
            ("legacy_scnd", "sparsevlm_topk", "native_maxmin"),
        )
        self.assertEqual(selected, "native_maxmin")
        self.assertEqual(admissible, ("sparsevlm_topk", "native_maxmin"))

        tied = dict(errors)
        tied["native_maxmin"] = dict(tied["sparsevlm_topk"])
        selected, _ = select_candidate_deterministically(
            tied,
            ("legacy_scnd", "sparsevlm_topk", "native_maxmin"),
        )
        self.assertEqual(selected, "sparsevlm_topk")


class CounterfactualStrategyIntegrationTests(unittest.TestCase):
    @staticmethod
    def _strategy(*, prune_ratio=0.5, counterfactual=None, **overrides):
        config = {
            "prune_layers": [0],
            "layer_modes": ["C"],
            "prune_ratio": prune_ratio,
            "fallback_topk": 2,
            "exclude_special_tokens": True,
            "min_visual_tokens_after_prune": 1,
            "seed_ratio_min": 0.25,
            "seed_ratio_max": 0.25,
            "seed_pool_multiplier": 2.0,
            "saliency_floor_min": 0.65,
            "saliency_floor_max": 0.65,
            "boundary_ratio": 0.25,
            "saliency_repair": True,
            "distance_metric": "cosine",
            "selection_backend": "auto",
            "c_selection_rule": "native",
            "seed": 42,
            "use_score_memory": False,
            "entropy_calibration": {
                "mode": "identity",
                "budget_keep_fraction_low": 0.125,
                "budget_keep_fraction_high": 0.5,
            },
            "evidence_reconciliation": {"mode": "none"},
        }
        if counterfactual is not None:
            config["next_layer_counterfactual"] = counterfactual
        config.update(overrides)
        return SparseVLMSCNDStrategy(config)

    @staticmethod
    def _prepare(strategy, num_visual=8):
        strategy.prepare_sample(
            inputs_embeds=torch.zeros(1, num_visual + 3, 8),
            v_token_start=1,
            v_token_num=num_visual,
            text_token_start=num_visual + 1,
            text_token_ids=torch.tensor([11, 12]),
            text_special_token_mask=torch.tensor([False, False]),
        )
        attention = torch.zeros(1, 1, num_visual + 3, num_visual + 3)
        scores = torch.linspace(1.0, 0.1, num_visual)
        attention[0, 0, -2:, 1 : num_visual + 1] = scores
        embeds = torch.eye(num_visual)
        return attention, embeds

    def test_parameter_validation_and_mutual_exclusion(self):
        valid = self._strategy(
            counterfactual={
                "mode": "audit_only",
                "candidate_sets": ["legacy_scnd", "sparsevlm_topk", "native_maxmin"],
            }
        )._get_scnd_params()
        self.assertEqual(valid["counterfactual_mode"], "audit_only")

        invalid_configs = (
            {"mode": "bogus"},
            {"mode": "route", "require_pareto_dominance": False},
            {"mode": "audit_only", "candidate_sets": ["legacy_scnd", "legacy_scnd"]},
            {"mode": "audit_only", "candidate_sets": ["sparsevlm_topk"]},
            {"mode": "force_candidate", "forced_candidate": "legacy_scnd"},
            {"mode": "audit_only", "query_scope": "last_text"},
        )
        for config in invalid_configs:
            with self.subTest(config=config), self.assertRaises(ValueError):
                self._strategy(counterfactual=config)._get_scnd_params()

        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            self._strategy(
                counterfactual={"mode": "audit_only"},
                evidence_reconciliation={"mode": "herc_v3"},
            )._get_scnd_params()
        with self.assertRaisesRegex(ValueError, "c_selection_rule=native"):
            self._strategy(
                counterfactual={"mode": "audit_only"},
                c_selection_rule="random_feasible",
            )._get_scnd_params()

    def test_high_profile_bypasses_without_candidate_state(self):
        baseline = self._strategy(prune_ratio=0.5)
        routed = self._strategy(
            prune_ratio=0.5,
            counterfactual={"mode": "route"},
        )
        baseline_attention, baseline_embeds = self._prepare(baseline)
        routed_attention, routed_embeds = self._prepare(routed)
        baseline_keep, baseline_info = baseline.compute_keep_mask(
            baseline_attention,
            1,
            8,
            9,
            0,
            current_visual_embeds=baseline_embeds,
        )
        routed_keep, routed_info = routed.compute_keep_mask(
            routed_attention,
            1,
            8,
            9,
            0,
            current_visual_embeds=routed_embeds,
        )
        torch.testing.assert_close(routed_keep, baseline_keep, rtol=0, atol=0)
        self.assertEqual(routed_info["selection_rule"], baseline_info["selection_rule"])
        self.assertTrue(routed_info["cf_bypassed"])
        self.assertEqual(routed_info["cf_bypass_reason"], "high_budget_profile")
        self.assertFalse(routed.wants_post_selection_routing(0, routed_info))
        self.assertIsNone(routed.sample_context["counterfactual_pending"])

    def test_force_candidate_replaces_complete_set_without_preview(self):
        strategy = self._strategy(
            prune_ratio=0.75,
            counterfactual={
                "mode": "force_candidate",
                "forced_candidate": "sparsevlm_topk",
                "candidate_sets": ["legacy_scnd", "sparsevlm_topk", "native_maxmin"],
            },
        )
        attention, embeds = self._prepare(strategy)
        legacy_keep, info = strategy.compute_keep_mask(
            attention,
            1,
            8,
            9,
            0,
            current_visual_embeds=embeds,
        )
        pending = strategy.sample_context["counterfactual_pending"]
        self.assertIsNotNone(pending)
        expected = deterministic_topk_indices(pending["visual_scores"], int(legacy_keep.numel()))
        selected, routing_info = strategy.route_keep_indices(
            legacy_keep_indices=legacy_keep,
            layer_idx=0,
            layer_info=info,
            routing_context=None,
        )
        torch.testing.assert_close(selected, expected.sort().values, rtol=0, atol=0)
        self.assertEqual(routing_info["cf_selected_candidate"], "sparsevlm_topk")
        self.assertTrue(routing_info["cf_switched_from_legacy"])
        self.assertEqual(routing_info["cf_query_count"], 0)
        self.assertEqual(routing_info["num_visual_after"], int(legacy_keep.numel()))
        self.assertIsNone(strategy.sample_context["counterfactual_pending"])

    def test_audit_keeps_legacy_while_route_applies_pareto_candidate(self):
        class FakePreview:
            projection_time_ms = 0.25
            query_count = 1

            def __init__(self, topk_indices):
                self.topk_indices = tuple(sorted(topk_indices))

            def evaluate(self, keep_indices):
                keep_tuple = tuple(sorted(torch.as_tensor(keep_indices).tolist()))
                if len(keep_tuple) == 0:
                    value = 0.0
                elif len(keep_tuple) == 8:
                    value = 1.0
                elif keep_tuple == self.topk_indices:
                    value = 0.9
                else:
                    value = 0.2
                return torch.tensor([[[value, 0.0]]], dtype=torch.float32)

        results = {}
        for mode in ("audit_only", "route"):
            strategy = self._strategy(
                prune_ratio=0.75,
                saliency_floor_min=0.0,
                saliency_floor_max=0.0,
                counterfactual={"mode": mode},
            )
            attention, _ = self._prepare(strategy)
            embeds = torch.tensor(
                [
                    [1.0, 0.0],
                    [0.99, 0.01],
                    [-1.0, 0.0],
                    [0.0, 1.0],
                    [0.0, -1.0],
                    [0.7, 0.7],
                    [-0.7, 0.7],
                    [-0.7, -0.7],
                ],
                dtype=torch.float32,
            )
            legacy_keep, info = strategy.compute_keep_mask(
                attention,
                1,
                8,
                9,
                0,
                current_visual_embeds=embeds,
            )
            pending = strategy.sample_context["counterfactual_pending"]
            candidates = strategy._build_counterfactual_candidates(pending)
            self.assertFalse(torch.equal(candidates["legacy_scnd"], candidates["sparsevlm_topk"]))
            fake_preview = FakePreview(candidates["sparsevlm_topk"].tolist())
            with mock.patch(
                "strategies.sparsevlm_scnd.build_next_layer_preview",
                return_value=fake_preview,
            ):
                selected, routing_info = strategy.route_keep_indices(
                    legacy_keep_indices=legacy_keep,
                    layer_idx=0,
                    layer_info=info,
                    routing_context=None,
                )
            results[mode] = (legacy_keep, selected, candidates, routing_info)

        audit_legacy, audit_selected, _, audit_info = results["audit_only"]
        torch.testing.assert_close(audit_selected, audit_legacy, rtol=0, atol=0)
        self.assertEqual(audit_info["cf_routed_candidate"], "sparsevlm_topk")
        self.assertEqual(audit_info["cf_selected_candidate"], "legacy_scnd")
        self.assertFalse(audit_info["cf_switched_from_legacy"])

        route_legacy, route_selected, route_candidates, route_info = results["route"]
        self.assertFalse(torch.equal(route_selected, route_legacy))
        torch.testing.assert_close(
            route_selected,
            route_candidates["sparsevlm_topk"],
            rtol=0,
            atol=0,
        )
        self.assertEqual(route_info["cf_selected_candidate"], "sparsevlm_topk")
        self.assertTrue(route_info["cf_switched_from_legacy"])
        self.assertGreater(route_info["cf_error_sum_margin"], 0.0)


if __name__ == "__main__":
    unittest.main()
