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

from pruner import VisualTokenPruner, _apply_rotary_pos_emb, _repeat_kv  # noqa: E402
from strategies.masking_attn_score import MaskingAttnScoreStrategy  # noqa: E402


class IdentityRotary:
    def __call__(self, value_states, seq_len=None):
        seq_len = int(seq_len or value_states.shape[-2])
        head_dim = int(value_states.shape[-1])
        cos = torch.ones((seq_len, head_dim), dtype=value_states.dtype, device=value_states.device)
        sin = torch.zeros((seq_len, head_dim), dtype=value_states.dtype, device=value_states.device)
        return cos, sin


class ZeroMLP(torch.nn.Module):
    def forward(self, hidden_states):
        return torch.zeros_like(hidden_states)


class DummySelfAttention:
    def __init__(self, hidden_size: int):
        self.q_proj = torch.nn.Identity()
        self.k_proj = torch.nn.Identity()
        self.v_proj = torch.nn.Identity()
        self.o_proj = torch.nn.Identity()
        self.rotary_emb = IdentityRotary()
        self.num_heads = 1
        self.num_key_value_heads = 1
        self.num_key_value_groups = 1
        self.head_dim = hidden_size


class DummyDecoderLayer:
    def __init__(self, layer_idx: int, hidden_size: int):
        self.layer_idx = layer_idx
        self.input_layernorm = torch.nn.Identity()
        self.post_attention_layernorm = torch.nn.Identity()
        self.mlp = ZeroMLP()
        self.self_attn = DummySelfAttention(hidden_size)
        self.output_attentions_history = []
        self.seq_lens = []

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
        self.seq_lens.append(hidden_states.shape[1])

        residual = hidden_states
        batch_size, seq_len, hidden_size = hidden_states.shape
        num_heads = int(self.self_attn.num_heads)
        head_dim = int(self.self_attn.head_dim)
        num_key_value_heads = int(self.self_attn.num_key_value_heads)
        num_key_value_groups = int(self.self_attn.num_key_value_groups)

        query_states = self.self_attn.q_proj(hidden_states).view(
            batch_size, seq_len, num_heads, head_dim
        ).transpose(1, 2)
        key_states = self.self_attn.k_proj(hidden_states).view(
            batch_size, seq_len, num_key_value_heads, head_dim
        ).transpose(1, 2)
        value_states = self.self_attn.v_proj(hidden_states).view(
            batch_size, seq_len, num_key_value_heads, head_dim
        ).transpose(1, 2)

        cos, sin = self.self_attn.rotary_emb(value_states, seq_len=seq_len)
        query_states, key_states = _apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if len(past_key_value.key_cache) <= self.layer_idx:
            past_key_value.key_cache.append(key_states)
            past_key_value.value_cache.append(value_states)
        else:
            past_key_value.key_cache[self.layer_idx] = key_states
            past_key_value.value_cache[self.layer_idx] = value_states
        past_key_value._seen_tokens = seq_len

        key_states = _repeat_kv(key_states, num_key_value_groups)
        value_states = _repeat_kv(value_states, num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / (head_dim ** 0.5)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask.to(attn_weights.dtype)
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        hidden_out = residual + self.self_attn.o_proj(attn_output)

        if output_attentions:
            return (hidden_out, attn_weights)
        return (hidden_out, past_key_value)


class DummyBackbone:
    def __init__(self, layers):
        self.layers = layers
        self.norm = torch.nn.Identity()


class DummyModel:
    def __init__(self, layers):
        self.model = DummyBackbone(layers)


class MaskingAttnScorePrunerTests(unittest.TestCase):
    def _make_inputs(self):
        return torch.tensor(
            [
                [
                    [1.0, 0.0],  # visual 0
                    [0.0, 1.0],  # visual 1
                    [1.0, 1.0],  # visual 2
                    [1.0, 0.0],  # text 0
                    [0.0, 1.0],  # text 1
                ]
            ],
            dtype=torch.float32,
        )

    def test_masking_attn_score_masks_current_layer_without_changing_sequence_or_kv_length(self):
        model = DummyModel([DummyDecoderLayer(0, hidden_size=2)])
        strategy = MaskingAttnScoreStrategy({"prune_ratio": [1.0 / 3.0]})
        pruner = VisualTokenPruner(
            model,
            strategy,
            {
                "layer_selection": "fixed",
                "prune_layers": [0],
                "prune_ratio": [1.0 / 3.0],
            },
        )

        inputs = self._make_inputs()
        hidden_states, past_kv, prune_info = pruner._pruned_prefill(
            inputs_embeds=inputs,
            v_token_start=0,
            v_token_num=3,
            text_token_start=3,
            save_tv_attn=True,
            capture_layers={0},
        )

        baseline_cache = DynamicCache()
        baseline_output = model.model.layers[0](
            inputs,
            attention_mask=pruner._refresh_sequence_state(5, inputs.dtype, inputs.device)[1],
            position_ids=torch.arange(5, dtype=torch.long).unsqueeze(0),
            past_key_value=baseline_cache,
            use_cache=True,
            output_attentions=False,
        )[0]

        self.assertEqual(prune_info["prune_stage"], "masking")
        self.assertEqual(hidden_states.shape[1], 5)
        self.assertEqual(prune_info["original_seq_len"], 5)
        self.assertEqual(prune_info["final_seq_len"], 5)
        self.assertEqual(past_kv.get_seq_length(), 5)
        self.assertEqual(model.model.layers[0].seq_lens, [5])
        self.assertFalse(torch.allclose(hidden_states, baseline_output))

        layer_info = prune_info["layers"][0]
        self.assertEqual(layer_info["prune_stage"], "masking")
        self.assertEqual(layer_info["keep_indices"].tolist(), [0, 2])
        self.assertEqual(layer_info["num_visual_before"], 3)
        self.assertEqual(layer_info["num_visual_after"], 2)
        self.assertEqual(layer_info["num_pruned"], 1)
        self.assertEqual(tuple(layer_info["tv_attn"].shape), (1, 2, 3))
        self.assertTrue(torch.allclose(
            torch.tensor(layer_info["importance_scores"]),
            torch.tensor([0.2050, 0.1960, 0.2686]),
            atol=1e-4,
        ))

    def test_masking_attn_score_only_affects_target_layer_and_keeps_full_kv_for_following_layers(self):
        model = DummyModel(
            [
                DummyDecoderLayer(0, hidden_size=2),
                DummyDecoderLayer(1, hidden_size=2),
                DummyDecoderLayer(2, hidden_size=2),
            ]
        )
        strategy = MaskingAttnScoreStrategy({"prune_ratio": [1.0 / 3.0]})
        pruner = VisualTokenPruner(
            model,
            strategy,
            {
                "layer_selection": "fixed",
                "prune_layers": [1],
                "prune_ratio": [1.0 / 3.0],
            },
        )

        hidden_states, past_kv, prune_info = pruner._pruned_prefill(
            inputs_embeds=self._make_inputs(),
            v_token_start=0,
            v_token_num=3,
            text_token_start=3,
            save_tv_attn=False,
            capture_layers=None,
        )

        self.assertEqual(hidden_states.shape[1], 5)
        self.assertEqual(prune_info["final_seq_len"], 5)
        self.assertEqual(model.model.layers[0].seq_lens, [5])
        self.assertEqual(model.model.layers[1].seq_lens, [])
        self.assertEqual(model.model.layers[2].seq_lens, [5])
        self.assertEqual(past_kv.get_seq_length(), 5)
        self.assertEqual(past_kv.key_cache[0].shape[2], 5)
        self.assertEqual(past_kv.key_cache[1].shape[2], 5)
        self.assertEqual(past_kv.key_cache[2].shape[2], 5)
        self.assertEqual(prune_info["layers"][1]["keep_indices"].tolist(), [0, 2])


if __name__ == "__main__":
    unittest.main()
