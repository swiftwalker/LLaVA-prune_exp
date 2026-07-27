import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC_DIR)

from strategies import get_strategy
from strategies.scnd_entropy_calibration import ENTROPY_DEFINITION
from strategies.sparsevlm_scnd import SparseVLMSCNDStrategy, _scnd_distance_matrix


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
            "selection_backend": "auto",
            "c_selection_rule": "native",
            "seed": 42,
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
        self.assertEqual(info["selection_backend_requested"], "auto")
        self.assertIn(info["selection_backend_effective"], {"python", "gpu"})
        self.assertEqual(info["distance_metric"], "cosine")
        self.assertIn("selection_time_ms", info)
        self.assertIn("distance_time_ms", info)

    def test_selection_backend_validation_and_cpu_auto_fallback(self):
        strategy = self._strategy(selection_backend="auto")
        params = strategy._get_scnd_params()
        self.assertEqual(params["selection_backend"], "auto")

        invalid = self._strategy(selection_backend="bogus")
        with self.assertRaises(ValueError):
            invalid._get_scnd_params()

        for metric in ("cosine", "euclidean", "dot"):
            self.assertEqual(self._strategy(distance_metric=metric)._get_scnd_params()["distance_metric"], metric)

        invalid_metric = self._strategy(distance_metric="manhattan")
        with self.assertRaises(ValueError):
            invalid_metric._get_scnd_params()

        b_non_cosine = self._strategy(layer_modes=["B"], distance_metric="euclidean")
        with self.assertRaises(ValueError):
            b_non_cosine._get_scnd_params()

        random_rule = self._strategy(c_selection_rule="random_feasible")
        self.assertEqual(random_rule._get_scnd_params()["c_selection_rule"], "random_feasible")

        invalid_rule = self._strategy(c_selection_rule="bogus")
        with self.assertRaises(ValueError):
            invalid_rule._get_scnd_params()

        invalid_calibration = self._strategy(entropy_calibration={"mode": "bogus"})
        with self.assertRaises(ValueError):
            invalid_calibration._get_scnd_params()

        missing_artifact = self._strategy(entropy_calibration={"mode": "quantile_affine"})
        with self.assertRaisesRegex(ValueError, "artifact_path"):
            missing_artifact._get_scnd_params()

        invalid_budget_bounds = self._strategy(
            entropy_calibration={
                "mode": "identity",
                "budget_keep_fraction_low": 0.5,
                "budget_keep_fraction_high": 0.5,
            }
        )
        with self.assertRaisesRegex(ValueError, "budget bounds"):
            invalid_budget_bounds._get_scnd_params()

        role_constraint = self._strategy(
            visual_role_constraint={"mode": "budget_adaptive_local_floor", "local_floor_ratio": 0.9}
        )
        self.assertEqual(
            role_constraint._get_scnd_params()["visual_role_constraint_mode"],
            "budget_adaptive_local_floor",
        )
        gated_role_constraint = self._strategy(
            visual_role_constraint={
                "mode": "budget_adaptive_saliency_gated_local_floor",
                "max_visual_tokens_before": 2200,
                "min_local_saliency_mass_ratio": 0.70,
            }
        )
        gated_params = gated_role_constraint._get_scnd_params()
        self.assertEqual(
            gated_params["visual_role_constraint_mode"],
            "budget_adaptive_saliency_gated_local_floor",
        )
        self.assertEqual(gated_params["visual_role_max_visual_tokens_before"], 2200)
        self.assertEqual(gated_params["visual_role_min_local_saliency_mass_ratio"], 0.70)
        invalid_role_mode = self._strategy(visual_role_constraint={"mode": "bogus"})
        with self.assertRaisesRegex(ValueError, "visual_role_constraint.mode"):
            invalid_role_mode._get_scnd_params()
        invalid_role_ratio = self._strategy(
            visual_role_constraint={"mode": "local_floor", "local_floor_ratio": 1.1}
        )
        with self.assertRaisesRegex(ValueError, "local_floor_ratio"):
            invalid_role_ratio._get_scnd_params()
        invalid_role_max_tokens = self._strategy(
            visual_role_constraint={"mode": "none", "max_visual_tokens_before": 0}
        )
        with self.assertRaisesRegex(ValueError, "max_visual_tokens_before"):
            invalid_role_max_tokens._get_scnd_params()
        invalid_role_saliency_mass = self._strategy(
            visual_role_constraint={"mode": "none", "min_local_saliency_mass_ratio": 1.1}
        )
        with self.assertRaisesRegex(ValueError, "min_local_saliency_mass_ratio"):
            invalid_role_saliency_mass._get_scnd_params()
        random_role_constraint = self._strategy(
            c_selection_rule="random_feasible",
            visual_role_constraint={"mode": "local_floor"},
        )
        with self.assertRaisesRegex(ValueError, "c_selection_rule=native"):
            random_role_constraint._get_scnd_params()

        self._prepare(strategy, 4)
        attn = _full_attention(
            seq_len=7,
            text_rows=[5, 6],
            visual_cols=[1, 2, 3, 4],
            values=[[0.9, 0.8, 0.7, 0.1], [0.9, 0.8, 0.7, 0.1]],
        )
        keep, info = strategy.compute_keep_mask(attn, 1, 4, 5, 0, current_visual_embeds=torch.eye(4))

        self.assertEqual(keep.numel(), info["target_keep"])
        self.assertEqual(info["selection_backend_effective"], "python")

    def test_budget_adaptive_visual_role_constraint_only_activates_under_pressure(self):
        high = self._strategy()._visual_role_constraint_control(
            "budget_adaptive_local_floor", 1.0, 0.5, 0.125, 0.5
        )
        ultra = self._strategy()._visual_role_constraint_control(
            "budget_adaptive_local_floor", 1.0, 0.125, 0.125, 0.5
        )
        middle = self._strategy()._visual_role_constraint_control(
            "budget_adaptive_local_floor", 0.9, 0.3125, 0.125, 0.5
        )

        self.assertEqual(high["budget_pressure"], 0.0)
        self.assertEqual(high["local_floor_ratio_effective"], 0.0)
        self.assertEqual(ultra["budget_pressure"], 1.0)
        self.assertEqual(ultra["local_floor_ratio_effective"], 1.0)
        self.assertAlmostEqual(middle["local_floor_ratio_effective"], 0.45)

    def test_saliency_gated_visual_role_constraint_requires_both_gates(self):
        kwargs = {
            "mode": "budget_adaptive_saliency_gated_local_floor",
            "local_floor_ratio": 1.0,
            "keep_fraction": 0.125,
            "keep_fraction_low": 0.125,
            "keep_fraction_high": 0.5,
            "max_visual_tokens_before": 2200,
            "min_local_saliency_mass_ratio": 0.70,
        }
        passed = self._strategy()._visual_role_constraint_control(
            num_visual_tokens_before=2184,
            local_saliency_mass_ratio=0.75,
            **kwargs,
        )
        layout_failed = self._strategy()._visual_role_constraint_control(
            num_visual_tokens_before=2242,
            local_saliency_mass_ratio=0.75,
            **kwargs,
        )
        saliency_failed = self._strategy()._visual_role_constraint_control(
            num_visual_tokens_before=2184,
            local_saliency_mass_ratio=0.69,
            **kwargs,
        )

        self.assertTrue(passed["gate_passed"])
        self.assertEqual(passed["local_floor_ratio_effective"], 1.0)
        self.assertFalse(layout_failed["layout_gate_passed"])
        self.assertEqual(layout_failed["local_floor_ratio_effective"], 0.0)
        self.assertFalse(saliency_failed["saliency_gate_passed"])
        self.assertEqual(saliency_failed["local_floor_ratio_effective"], 0.0)

    def test_saliency_gated_visual_role_constraint_uses_raw_positive_mass(self):
        strategy = self._strategy(
            prune_ratio=0.5,
            seed_ratio_min=0.5,
            seed_ratio_max=0.5,
            saliency_floor_min=0.0,
            saliency_floor_max=0.0,
            visual_role_constraint={
                "mode": "budget_adaptive_saliency_gated_local_floor",
                "local_floor_ratio": 1.0,
                "budget_keep_fraction_low": 0.125,
                "budget_keep_fraction_high": 0.75,
                "max_visual_tokens_before": 4,
                "min_local_saliency_mass_ratio": 0.80,
            },
        )
        self._prepare(strategy, 4)
        strategy.sample_context["visual_layout"] = {
            "base_token_count": 2,
            "local_patch_token_count": 2,
            "newline_token_count": 0,
            "local_grid_width": 2,
            "total_token_count": 4,
        }
        attn = _full_attention(
            seq_len=7,
            text_rows=[5, 6],
            visual_cols=[1, 2, 3, 4],
            values=[[0.2, 0.1, 0.8, 0.7], [0.2, 0.1, 0.8, 0.7]],
        )
        embeds = torch.eye(4, dtype=torch.float32)

        keep, info = strategy.compute_keep_mask(attn, 1, 4, 5, 0, current_visual_embeds=embeds)

        self.assertEqual(keep.numel(), 2)
        self.assertTrue(info["visual_role_gate_passed"])
        self.assertAlmostEqual(info["visual_role_local_saliency_mass_ratio"], 1.5 / 1.8)
        self.assertEqual(
            info["selection_rule"],
            "saliency_constrained_native_divprune_anyres_saliency_gated_local_floor",
        )

    def test_anyres_local_floor_preserves_saliency_topk_local_capacity(self):
        strategy = self._strategy(
            prune_ratio=0.5,
            seed_ratio_min=0.5,
            seed_ratio_max=0.5,
            saliency_floor_min=0.0,
            saliency_floor_max=0.0,
            visual_role_constraint={"mode": "local_floor", "local_floor_ratio": 1.0},
        )
        self._prepare(strategy, 8)
        strategy.sample_context["visual_layout"] = {
            "base_token_count": 4,
            "local_patch_token_count": 3,
            "newline_token_count": 1,
            "local_grid_width": 3,
            "total_token_count": 8,
        }
        attn = _full_attention(
            seq_len=11,
            text_rows=[9, 10],
            visual_cols=list(range(1, 9)),
            values=[
                [0.7, 0.6, 0.5, 0.4, 1.0, 0.95, 0.9, 0.1],
                [0.7, 0.6, 0.5, 0.4, 1.0, 0.95, 0.9, 0.1],
            ],
        )
        embeds = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.99],
                [0.0, 0.0, 0.98],
                [0.0, 0.0, -1.0],
            ]
        )

        keep, info = strategy.compute_keep_mask(attn, 1, 8, 9, 0, current_visual_embeds=embeds)

        self.assertEqual(keep.numel(), 4)
        self.assertEqual(info["topk_local_patch_count"], 3)
        self.assertEqual(info["local_patch_floor_count"], 3)
        self.assertEqual(info["selected_local_patch_count"], 3)
        self.assertEqual(info["selection_rule"], "saliency_constrained_native_divprune_anyres_local_floor")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for GPU role-constraint parity")
    def test_anyres_local_floor_gpu_matches_python_selection(self):
        saliency = torch.tensor([0.7, 0.6, 0.5, 0.4, 1.0, 0.95, 0.9, 0.1])
        embeds = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.99],
                [0.0, 0.0, 0.98],
                [0.0, 0.0, -1.0],
            ]
        )
        role_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 2])
        distance = _scnd_distance_matrix(embeds, "cosine")
        kwargs = {
            "target_keep": 4,
            "entropy_norm": 0.5,
            "seed_ratio_min": 0.5,
            "seed_ratio_max": 0.5,
            "seed_pool_multiplier": 2.0,
            "saliency_floor_min": 0.0,
            "saliency_floor_max": 0.0,
            "saliency_repair": True,
            "local_floor_ratio": 1.0,
        }
        strategy = self._strategy()

        expected = strategy._saliency_constrained_native_divprune_select(
            saliency_score=saliency,
            distance=distance,
            visual_role_ids=role_ids,
            **kwargs,
        )
        actual = strategy._saliency_constrained_native_divprune_select_gpu(
            saliency_score=saliency.cuda(),
            distance=distance.cuda(),
            visual_role_ids=role_ids.cuda(),
            **kwargs,
        )

        self.assertEqual(actual["keep_indices"].cpu().tolist(), expected["keep_indices"].tolist())
        self.assertEqual(actual["local_patch_floor_count"], 3)
        self.assertEqual(actual["selected_local_patch_count"], 3)

    def test_entropy_calibration_identity_and_quantile_control(self):
        identity = self._strategy(entropy_calibration={"mode": "identity"})
        identity_params = identity._get_scnd_params()
        identity_result = identity._entropy_control(0.72, 0, "C", identity_params)
        self.assertEqual(identity_result["control_value"], 0.72)

        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_path = Path(tmp_dir) / "calibration.json"
            artifact_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "mode": "quantile_affine",
                        "entropy_definition": ENTROPY_DEFINITION,
                        "model_name": "toy-model",
                        "model_config_fingerprint": "toy-fingerprint",
                        "capture_layer": 0,
                        "quantiles": {
                            "low_probability": 0.05,
                            "high_probability": 0.95,
                            "low_value": 0.60,
                            "high_value": 0.80,
                        },
                    }
                ),
                encoding="utf-8",
            )
            calibrated = self._strategy(
                entropy_calibration={"mode": "quantile_affine", "artifact_path": str(artifact_path)},
                _model_name="toy-model",
                _model_config_fingerprint="toy-fingerprint",
            )
            calibrated_result = calibrated._entropy_control(0.70, 0, "C", calibrated._get_scnd_params())
            self.assertAlmostEqual(calibrated_result["control_value"], 0.5)
            with self.assertRaisesRegex(ValueError, "bound to layer"):
                calibrated._entropy_control(0.70, 1, "C", calibrated._get_scnd_params())

            budget_adaptive = self._strategy(
                entropy_calibration={
                    "mode": "budget_adaptive_quantile",
                    "artifact_path": str(artifact_path),
                    "budget_keep_fraction_low": 0.125,
                    "budget_keep_fraction_high": 0.5,
                    "diversity_tail_gain": 1.0,
                },
                _model_name="toy-model",
                _model_config_fingerprint="toy-fingerprint",
            )
            high_result = budget_adaptive._entropy_control(
                0.65,
                0,
                "C",
                budget_adaptive._get_scnd_params(),
                keep_fraction=0.5,
            )
            ultra_result = budget_adaptive._entropy_control(
                0.65,
                0,
                "C",
                budget_adaptive._get_scnd_params(),
                keep_fraction=0.125,
            )
            self.assertEqual(high_result["budget_pressure"], 0.0)
            self.assertEqual(ultra_result["budget_pressure"], 1.0)
            self.assertLess(high_result["control_value"], ultra_result["control_value"])

            tail_adaptive = self._strategy(
                entropy_calibration={
                    "mode": "budget_adaptive_tail_quantile",
                    "artifact_path": str(artifact_path),
                    "budget_keep_fraction_low": 0.125,
                    "budget_keep_fraction_high": 0.5,
                    "diversity_tail_gain": 1.0,
                },
                _model_name="toy-model",
                _model_config_fingerprint="toy-fingerprint",
            )
            tail_result = tail_adaptive._entropy_control(
                0.70,
                0,
                "C",
                tail_adaptive._get_scnd_params(),
                keep_fraction=0.125,
            )
            self.assertAlmostEqual(tail_result["control_value"], 0.75)

    def test_distance_metric_matrices_are_larger_is_more_diverse(self):
        embeds = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)

        for metric in ("cosine", "euclidean", "dot"):
            distance = _scnd_distance_matrix(embeds, metric)
            self.assertEqual(tuple(distance.shape), (3, 3))
            self.assertEqual(distance.device, embeds.device)
            self.assertEqual(distance.dtype, torch.float32)
            self.assertTrue(torch.isfinite(distance).all())
            self.assertTrue(torch.equal(torch.diag(distance), torch.zeros(3)))
            self.assertGreater(float(distance[0, 1].item()), float(distance[0, 2].item()))

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

    def test_non_cosine_metrics_keep_larger_is_more_diverse_selection(self):
        for metric in ("euclidean", "dot"):
            strategy = self._strategy(
                prune_ratio=0.5,
                distance_metric=metric,
                seed_ratio_min=1.0,
                seed_ratio_max=1.0,
                seed_pool_multiplier=2.0,
                saliency_floor_min=0.65,
                saliency_floor_max=0.65,
            )
            self._prepare(strategy, 4)
            attn = _full_attention(
                seq_len=7,
                text_rows=[5, 6],
                visual_cols=[1, 2, 3, 4],
                values=[[0.9, 0.8, 0.7, 0.1], [0.9, 0.8, 0.7, 0.1]],
            )
            embeds = torch.tensor([[1.0, 0.0], [0.99, 0.0], [-1.0, 0.0], [0.0, 1.0]])
            keep, info = strategy.compute_keep_mask(attn, 1, 4, 5, 0, current_visual_embeds=embeds)

            self.assertEqual(keep.tolist(), [0, 2])
            self.assertEqual(info["seed_indices"].tolist(), [0, 2])
            self.assertEqual(info["distance_metric"], metric)
            self.assertGreaterEqual(info["saliency_mass_selected"] + 1e-8, info["saliency_mass_floor"])

    def test_random_feasible_rule_is_reproducible_and_respects_saliency_floor(self):
        keeps = []
        orders = []
        infos = []
        for _ in range(2):
            strategy = self._strategy(
                prune_ratio=0.5,
                c_selection_rule="random_feasible",
                selection_backend="python",
                seed=7,
                seed_ratio_min=0.34,
                seed_ratio_max=0.34,
                seed_pool_multiplier=2.0,
                saliency_floor_min=0.8,
                saliency_floor_max=0.8,
                saliency_repair=True,
            )
            self._prepare(strategy, 6)
            attn = _full_attention(
                seq_len=9,
                text_rows=[7, 8],
                visual_cols=[1, 2, 3, 4, 5, 6],
                values=[[1.0, 0.9, 0.8, 0.2, 0.1, 0.05], [1.0, 0.9, 0.8, 0.2, 0.1, 0.05]],
            )
            embeds = torch.tensor(
                [
                    [1.0, 0.0],
                    [0.9, 0.0],
                    [0.0, 1.0],
                    [-1.0, 0.0],
                    [0.0, -1.0],
                    [0.5, 0.5],
                ],
                dtype=torch.float32,
            )
            keep, info = strategy.compute_keep_mask(attn, 1, 6, 7, 0, current_visual_embeds=embeds)
            keeps.append(keep.tolist())
            orders.append(info["mmr_selected_order"].tolist())
            infos.append(info)

        self.assertEqual(keeps[0], keeps[1])
        self.assertEqual(orders[0], orders[1])
        self.assertEqual(infos[0]["selection_rule"], "saliency_constrained_random_feasible")
        self.assertEqual(infos[0]["c_selection_rule"], "random_feasible")
        self.assertEqual(len(keeps[0]), infos[0]["target_keep"])
        self.assertGreaterEqual(infos[0]["saliency_mass_selected"] + 1e-8, infos[0]["saliency_mass_floor"])

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

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for explicit GPU backend equivalence")
    def test_gpu_backend_matches_python_backend_for_c_and_b_layers(self):
        cases = [
            (
                {"layer_modes": ["C"], "prune_ratio": 0.5, "selection_backend": "python"},
                {"layer_modes": ["C"], "prune_ratio": 0.5, "selection_backend": "gpu"},
                [[0.9, 0.8, 0.7, 0.1], [0.9, 0.8, 0.7, 0.1]],
                torch.tensor([[1.0, 0.0], [0.99, 0.0], [0.0, 1.0], [-1.0, 0.0]], device="cuda"),
            ),
            (
                {
                    "layer_modes": ["C"],
                    "prune_ratio": 0.5,
                    "selection_backend": "python",
                    "distance_metric": "euclidean",
                },
                {
                    "layer_modes": ["C"],
                    "prune_ratio": 0.5,
                    "selection_backend": "gpu",
                    "distance_metric": "euclidean",
                },
                [[0.9, 0.8, 0.7, 0.1], [0.9, 0.8, 0.7, 0.1]],
                torch.tensor([[1.0, 0.0], [0.99, 0.0], [0.0, 1.0], [-1.0, 0.0]], device="cuda"),
            ),
            (
                {
                    "layer_modes": ["C"],
                    "prune_ratio": 0.5,
                    "selection_backend": "python",
                    "distance_metric": "dot",
                },
                {
                    "layer_modes": ["C"],
                    "prune_ratio": 0.5,
                    "selection_backend": "gpu",
                    "distance_metric": "dot",
                },
                [[0.9, 0.8, 0.7, 0.1], [0.9, 0.8, 0.7, 0.1]],
                torch.tensor([[1.0, 0.0], [0.99, 0.0], [0.0, 1.0], [-1.0, 0.0]], device="cuda"),
            ),
            (
                {
                    "layer_modes": ["C"],
                    "prune_ratio": 0.5,
                    "selection_backend": "python",
                    "c_selection_rule": "random_feasible",
                    "seed": 13,
                },
                {
                    "layer_modes": ["C"],
                    "prune_ratio": 0.5,
                    "selection_backend": "gpu",
                    "c_selection_rule": "random_feasible",
                    "seed": 13,
                },
                [[0.9, 0.8, 0.7, 0.1], [0.9, 0.8, 0.7, 0.1]],
                torch.tensor([[1.0, 0.0], [0.99, 0.0], [0.0, 1.0], [-1.0, 0.0]], device="cuda"),
            ),
            (
                {"layer_modes": ["B"], "prune_ratio": 0.5, "boundary_ratio": 0.5, "selection_backend": "python"},
                {"layer_modes": ["B"], "prune_ratio": 0.5, "boundary_ratio": 0.5, "selection_backend": "gpu"},
                [[0.90, 0.89, 0.88, 0.87], [0.90, 0.89, 0.88, 0.87]],
                torch.tensor([[1.0, 0.0], [0.99, 0.0], [-1.0, 0.0], [0.0, 1.0]], device="cuda"),
            ),
        ]

        for python_overrides, gpu_overrides, values, embeds in cases:
            keeps = []
            infos = []
            for overrides in (python_overrides, gpu_overrides):
                strategy = self._strategy(**overrides)
                strategy.prepare_sample(
                    inputs_embeds=_inputs(4).to("cuda"),
                    v_token_start=1,
                    v_token_num=4,
                    text_token_start=5,
                    text_token_ids=torch.tensor([11, 12], device="cuda"),
                    text_special_token_mask=torch.tensor([False, False], device="cuda"),
                )
                attn = _full_attention(
                    seq_len=7,
                    text_rows=[5, 6],
                    visual_cols=[1, 2, 3, 4],
                    values=values,
                ).to("cuda")
                keep, info = strategy.compute_keep_mask(attn, 1, 4, 5, 0, current_visual_embeds=embeds)
                keeps.append(keep.detach().cpu().tolist())
                infos.append(info)

            self.assertEqual(keeps[0], keeps[1])
            self.assertEqual(infos[1]["selection_backend_effective"], "gpu")
            self.assertGreaterEqual(infos[1]["selection_time_ms"], 0.0)
            self.assertEqual(infos[0]["distance_metric"], infos[1]["distance_metric"])
            if infos[1]["layer_mode"] == "C":
                self.assertGreaterEqual(infos[1]["saliency_mass_selected"] + 1e-8, infos[1]["saliency_mass_floor"])

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
