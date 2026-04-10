import os
import sys
import types
import unittest

import torch

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC_DIR)


class DynamicCache:
    def __init__(self):
        self.key_cache = []
        self.value_cache = []
        self._seen_tokens = 0

    def get_seq_length(self):
        if not self.key_cache:
            return 0
        return self.key_cache[0].shape[2]


sys.modules.setdefault("transformers", types.SimpleNamespace(DynamicCache=DynamicCache))

from pruner import VisualTokenPruner
from strategies.sparsevlm import SparseVLMStrategy
from strategies.sparsevlm_adaptive_stratified import (
    SparseVLMAdaptiveStratifiedStrategy,
    patch_index_to_stratum_id,
)


class DummyAttentionLayer:
    def __init__(self, layer_idx, attn_by_seq_len):
        self.layer_idx = layer_idx
        self.attn_by_seq_len = attn_by_seq_len
        self.output_attentions_history = []

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        use_cache=True,
        output_attentions=False,
    ):
        self.output_attentions_history.append(output_attentions)
        seq_len = hidden_states.shape[1]
        key = hidden_states.new_zeros((1, 1, seq_len, 1))
        value = hidden_states.new_zeros((1, 1, seq_len, 1))

        if len(past_key_value.key_cache) <= self.layer_idx:
            past_key_value.key_cache.append(key)
            past_key_value.value_cache.append(value)
        else:
            past_key_value.key_cache[self.layer_idx] = key
            past_key_value.value_cache[self.layer_idx] = value
        past_key_value._seen_tokens = seq_len

        hidden_out = hidden_states + 1
        if output_attentions:
            attn = self.attn_by_seq_len[seq_len].to(hidden_states.device, hidden_states.dtype)
            return (hidden_out, attn)
        return (hidden_out, past_key_value)


class DummyBackbone:
    def __init__(self, layers):
        self.layers = layers
        self.norm = torch.nn.Identity()


class DummyModel:
    def __init__(self, layers):
        self.model = DummyBackbone(layers)


def _full_attention(seq_len, text_rows, visual_cols, values):
    attn = torch.zeros((1, 1, seq_len, seq_len), dtype=torch.float32)
    for row, row_values in zip(text_rows, values):
        attn[0, 0, row, visual_cols] = torch.tensor(row_values, dtype=torch.float32)
    return attn


class SparseVLMPrunerTests(unittest.TestCase):
    def _make_inputs(self):
        return torch.tensor(
            [
                [
                    [9.0, 9.0],  # prefix
                    [1.0, 0.0],  # visual 0
                    [0.0, 1.0],  # visual 1
                    [1.0, 1.0],  # visual 2
                    [1.0, 0.0],  # text 0
                    [0.0, 1.0],  # text 1
                ]
            ],
            dtype=torch.float32,
        )

    def _make_grid_inputs(self, num_visual_tokens):
        visual_tokens = [[1.0, 0.0] for _ in range(num_visual_tokens)]
        return torch.tensor(
            [
                [
                    [9.0, 9.0],  # prefix
                    *visual_tokens,
                    [1.0, 0.0],  # text 0
                    [0.0, 1.0],  # text 1
                ]
            ],
            dtype=torch.float32,
        )

    def test_sparsevlm_prunes_one_layer_without_touching_text_tokens(self):
        layer0_attn = _full_attention(
            seq_len=6,
            text_rows=[4, 5],
            visual_cols=[1, 2, 3],
            values=[[0.1, 0.7, 0.2], [0.2, 0.6, 0.2]],
        )
        model = DummyModel([DummyAttentionLayer(0, {6: layer0_attn})])
        strategy = SparseVLMStrategy(
            {
                "prune_ratio": [1.0 / 3.0],
                "fallback_topk": 4,
                "exclude_special_tokens": True,
                "min_visual_tokens_after_prune": 1,
            }
        )
        pruner = VisualTokenPruner(
            model,
            strategy,
            {
                "layer_selection": "fixed",
                "prune_layers": [0],
                "prune_ratio": [1.0 / 3.0],
            },
        )

        hidden_states, _, prune_info = pruner._pruned_prefill(
            inputs_embeds=self._make_inputs(),
            initial_position_ids=None,
            v_token_start=1,
            v_token_num=3,
            text_token_start=4,
            text_token_ids=torch.tensor([11, 12]),
            text_special_token_mask=torch.tensor([False, False]),
            save_tv_attn=False,
            capture_layers=None,
        )

        self.assertEqual(model.model.layers[0].output_attentions_history, [True])
        self.assertEqual(hidden_states.shape[1], 5)
        self.assertEqual(prune_info["layers"][0]["num_visual_after"], 2)
        self.assertEqual(prune_info["layers"][0]["keep_indices"].tolist(), [1, 2])
        self.assertEqual(prune_info["layers"][0]["pruned_indices"].tolist(), [0])
        self.assertEqual(prune_info["layers"][0]["rater_indices"].tolist(), [0, 1])
        self.assertEqual(hidden_states.shape[1] - prune_info["layers"][0]["num_visual_after"], 3)

    def test_sparsevlm_updates_visual_positions_across_multiple_layers(self):
        layer0_attn = _full_attention(
            seq_len=6,
            text_rows=[4, 5],
            visual_cols=[1, 2, 3],
            values=[[0.1, 0.7, 0.2], [0.2, 0.6, 0.2]],
        )
        layer1_attn = _full_attention(
            seq_len=5,
            text_rows=[3, 4],
            visual_cols=[1, 2],
            values=[[0.9, 0.1], [0.7, 0.3]],
        )
        model = DummyModel(
            [
                DummyAttentionLayer(0, {6: layer0_attn}),
                DummyAttentionLayer(1, {5: layer1_attn}),
            ]
        )
        strategy = SparseVLMStrategy(
            {
                "prune_ratio": [1.0 / 3.0, 0.5],
                "fallback_topk": 4,
                "exclude_special_tokens": True,
                "min_visual_tokens_after_prune": 1,
            }
        )
        pruner = VisualTokenPruner(
            model,
            strategy,
            {
                "layer_selection": "fixed",
                "prune_layers": [0, 1],
                "prune_ratio": [1.0 / 3.0, 0.5],
            },
        )

        hidden_states, _, prune_info = pruner._pruned_prefill(
            inputs_embeds=self._make_inputs(),
            initial_position_ids=None,
            v_token_start=1,
            v_token_num=3,
            text_token_start=4,
            text_token_ids=torch.tensor([11, 12]),
            text_special_token_mask=torch.tensor([False, False]),
            save_tv_attn=False,
            capture_layers=None,
        )

        self.assertEqual(hidden_states.shape[1], 4)
        self.assertEqual(prune_info["layers"][0]["num_visual_after"], 2)
        self.assertEqual(prune_info["layers"][1]["num_visual_after"], 1)
        self.assertEqual(prune_info["layers"][1]["pruned_indices"].tolist(), [1])
        self.assertEqual(model.model.layers[0].output_attentions_history, [True])
        self.assertEqual(model.model.layers[1].output_attentions_history, [True])

    def test_sparsevlm_raises_on_invalid_attention_shape(self):
        invalid_attn = torch.zeros((6, 6), dtype=torch.float32)
        model = DummyModel([DummyAttentionLayer(0, {6: invalid_attn})])
        strategy = SparseVLMStrategy(
            {
                "prune_ratio": [0.5],
                "fallback_topk": 4,
                "exclude_special_tokens": True,
                "min_visual_tokens_after_prune": 1,
            }
        )
        pruner = VisualTokenPruner(
            model,
            strategy,
            {
                "layer_selection": "fixed",
                "prune_layers": [0],
                "prune_ratio": [0.5],
            },
        )

        with self.assertRaises(ValueError):
            pruner._pruned_prefill(
                inputs_embeds=self._make_inputs(),
                initial_position_ids=None,
                v_token_start=1,
                v_token_num=3,
                text_token_start=4,
                text_token_ids=torch.tensor([11, 12]),
                text_special_token_mask=torch.tensor([False, False]),
                save_tv_attn=False,
                capture_layers=None,
            )

    def test_adaptive_stratified_adds_spatial_compensation_in_single_layer(self):
        layer0_attn = _full_attention(
            seq_len=7,
            text_rows=[5, 6],
            visual_cols=[1, 2, 3, 4],
            values=[[0.9, 0.2, 0.1, 0.05], [0.8, 0.2, 0.1, 0.05]],
        )
        model = DummyModel([DummyAttentionLayer(0, {7: layer0_attn})])
        strategy = SparseVLMAdaptiveStratifiedStrategy(
            {
                "prune_ratio": [0.5],
                "fallback_topk": 4,
                "exclude_special_tokens": True,
                "min_visual_tokens_after_prune": 1,
                "high_ratio": 0.5,
                "grid_size": 2,
                "patch_per_row": 2,
                "intra_stratum_mode": "random",
            }
        )
        pruner = VisualTokenPruner(
            model,
            strategy,
            {
                "layer_selection": "fixed",
                "prune_layers": [0],
                "prune_ratio": [0.5],
            },
        )

        hidden_states, _, prune_info = pruner._pruned_prefill(
            inputs_embeds=self._make_grid_inputs(num_visual_tokens=4),
            initial_position_ids=None,
            v_token_start=1,
            v_token_num=4,
            text_token_start=5,
            text_token_ids=torch.tensor([11, 12]),
            text_special_token_mask=torch.tensor([False, False]),
            save_tv_attn=False,
            capture_layers=None,
        )

        self.assertEqual(hidden_states.shape[1], 5)
        self.assertEqual(prune_info["layers"][0]["high_keep_indices"].tolist(), [0])
        self.assertEqual(prune_info["layers"][0]["low_keep_indices"].tolist(), [1])
        self.assertEqual(prune_info["layers"][0]["keep_patch_indices"].tolist(), [0, 1])
        self.assertEqual(prune_info["layers"][0]["stratum_quotas"], [0, 1, 0, 0])

    def test_adaptive_stratified_uses_original_patch_coordinates_after_prior_pruning(self):
        layer0_values = [
            [0.9 if pos in {1, 4} else 0.6 if pos in {13, 16} else 0.05 for pos in range(1, 17)],
            [0.85 if pos in {1, 4} else 0.55 if pos in {13, 16} else 0.05 for pos in range(1, 17)],
        ]
        layer0_attn = _full_attention(
            seq_len=19,
            text_rows=[17, 18],
            visual_cols=list(range(1, 17)),
            values=layer0_values,
        )
        layer1_attn = _full_attention(
            seq_len=7,
            text_rows=[5, 6],
            visual_cols=[1, 2, 3, 4],
            values=[[0.9, 0.4, 0.3, 0.2], [0.8, 0.4, 0.3, 0.2]],
        )
        model = DummyModel(
            [
                DummyAttentionLayer(0, {19: layer0_attn}),
                DummyAttentionLayer(1, {7: layer1_attn}),
            ]
        )
        strategy = SparseVLMAdaptiveStratifiedStrategy(
            {
                "prune_ratio": [0.75, 0.5],
                "fallback_topk": 4,
                "exclude_special_tokens": True,
                "min_visual_tokens_after_prune": 1,
                "high_ratio": 0.5,
                "grid_size": 2,
                "patch_per_row": 4,
                "intra_stratum_mode": "farthest",
            }
        )
        pruner = VisualTokenPruner(
            model,
            strategy,
            {
                "layer_selection": "fixed",
                "prune_layers": [0, 1],
                "prune_ratio": [0.75, 0.5],
            },
        )

        hidden_states, _, prune_info = pruner._pruned_prefill(
            inputs_embeds=self._make_grid_inputs(num_visual_tokens=16),
            initial_position_ids=None,
            v_token_start=1,
            v_token_num=16,
            text_token_start=17,
            text_token_ids=torch.tensor([11, 12]),
            text_special_token_mask=torch.tensor([False, False]),
            save_tv_attn=False,
            capture_layers=None,
        )

        self.assertEqual(hidden_states.shape[1], 5)
        keep_patch_indices = prune_info["layers"][0]["keep_patch_indices"].tolist()
        self.assertEqual(
            sorted(patch_index_to_stratum_id(idx, grid_size=2, patch_per_row=4) for idx in keep_patch_indices),
            [0, 1, 2, 3],
        )
        self.assertEqual(prune_info["layers"][1]["current_patch_indices"].tolist(), keep_patch_indices)
        self.assertEqual(prune_info["layers"][1]["stratum_candidate_counts"], [0, 1, 1, 1])

    def test_adaptive_stratified_respects_minimum_visual_token_floor(self):
        layer0_attn = _full_attention(
            seq_len=7,
            text_rows=[5, 6],
            visual_cols=[1, 2, 3, 4],
            values=[[0.9, 0.2, 0.1, 0.05], [0.8, 0.2, 0.1, 0.05]],
        )
        model = DummyModel([DummyAttentionLayer(0, {7: layer0_attn})])
        strategy = SparseVLMAdaptiveStratifiedStrategy(
            {
                "prune_ratio": [0.9],
                "fallback_topk": 4,
                "exclude_special_tokens": True,
                "min_visual_tokens_after_prune": 4,
                "high_ratio": 0.5,
                "grid_size": 2,
                "patch_per_row": 2,
                "intra_stratum_mode": "random",
            }
        )
        pruner = VisualTokenPruner(
            model,
            strategy,
            {
                "layer_selection": "fixed",
                "prune_layers": [0],
                "prune_ratio": [0.9],
            },
        )

        hidden_states, _, prune_info = pruner._pruned_prefill(
            inputs_embeds=self._make_grid_inputs(num_visual_tokens=4),
            initial_position_ids=None,
            v_token_start=1,
            v_token_num=4,
            text_token_start=5,
            text_token_ids=torch.tensor([11, 12]),
            text_special_token_mask=torch.tensor([False, False]),
            save_tv_attn=False,
            capture_layers=None,
        )

        self.assertEqual(hidden_states.shape[1], 7)
        self.assertEqual(prune_info["layers"][0]["num_visual_after"], 4)
        self.assertEqual(prune_info["layers"][0]["pruned_indices"].tolist(), [])


if __name__ == "__main__":
    unittest.main()
