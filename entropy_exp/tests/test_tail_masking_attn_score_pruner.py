import json
import os
import sys
import tempfile
import unittest

import torch

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC_DIR)

from prune_inference import derive_effective_prune_config  # noqa: E402
from pruner import VisualTokenPruner, _apply_rotary_pos_emb, _repeat_kv  # noqa: E402
from strategies.tail_masking_attn_score import TailMaskingAttnScoreStrategy  # noqa: E402


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


class TailMaskingConfigExpansionTests(unittest.TestCase):
    def test_derive_effective_prune_config_expands_tail_layers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = os.path.join(tmpdir, "model")
            os.makedirs(model_dir, exist_ok=True)
            with open(os.path.join(model_dir, "config.json"), "w", encoding="utf-8") as f:
                json.dump({"num_hidden_layers": 32}, f)

            effective_cfg, effective_layers, tail_start_layer = derive_effective_prune_config(
                {
                    "strategy": "tail_masking_attn_score",
                    "layer_selection": "fixed",
                    "prune_layers": [3],
                    "prune_ratio": [0.2],
                },
                model_dir,
            )
            self.assertEqual(effective_layers, list(range(3, 32)))
            self.assertEqual(effective_cfg["prune_layers"], list(range(3, 32)))
            self.assertEqual(effective_cfg["prune_ratio"], 0.2)
            self.assertEqual(tail_start_layer, 3)

            scalar_cfg, scalar_layers, scalar_start = derive_effective_prune_config(
                {
                    "strategy": "tail_masking_attn_score",
                    "layer_selection": "fixed",
                    "prune_layers": 3,
                    "prune_ratio": 0.2,
                },
                model_dir,
            )
            self.assertEqual(scalar_layers, list(range(3, 32)))
            self.assertEqual(scalar_cfg["prune_ratio"], 0.2)
            self.assertEqual(scalar_start, 3)

    def test_derive_effective_prune_config_rejects_invalid_tail_configs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = os.path.join(tmpdir, "model")
            os.makedirs(model_dir, exist_ok=True)
            with open(os.path.join(model_dir, "config.json"), "w", encoding="utf-8") as f:
                json.dump({"num_hidden_layers": 32}, f)

            invalid_configs = [
                {
                    "strategy": "tail_masking_attn_score",
                    "layer_selection": "fixed",
                    "prune_layers": [1, 3],
                    "prune_ratio": [0.2],
                },
                {
                    "strategy": "tail_masking_attn_score",
                    "layer_selection": "fixed",
                    "prune_layers": [3],
                    "prune_ratio": [0.2, 0.3],
                },
                {
                    "strategy": "tail_masking_attn_score",
                    "layer_selection": "dynamic",
                    "prune_layers": [3],
                    "prune_ratio": [0.2],
                },
                {
                    "strategy": "tail_masking_attn_score",
                    "layer_selection": "fixed",
                    "prune_layers": [32],
                    "prune_ratio": [0.2],
                },
            ]

            for cfg in invalid_configs:
                with self.assertRaises(ValueError):
                    derive_effective_prune_config(cfg, model_dir)


class TailMaskingAttnScorePrunerTests(unittest.TestCase):
    def _make_inputs(self):
        return torch.tensor(
            [
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                ]
            ],
            dtype=torch.float32,
        )

    def test_tail_masking_applies_to_every_effective_tail_layer(self):
        model = DummyModel(
            [
                DummyDecoderLayer(0, hidden_size=2),
                DummyDecoderLayer(1, hidden_size=2),
                DummyDecoderLayer(2, hidden_size=2),
            ]
        )
        strategy = TailMaskingAttnScoreStrategy({"prune_ratio": 1.0 / 3.0})
        pruner = VisualTokenPruner(
            model,
            strategy,
            {
                "layer_selection": "fixed",
                "prune_layers": [1, 2],
                "prune_ratio": 1.0 / 3.0,
            },
        )

        hidden_states, past_kv, prune_info = pruner._pruned_prefill(
            inputs_embeds=self._make_inputs(),
            initial_position_ids=None,
            v_token_start=0,
            v_token_num=3,
            text_token_start=3,
            save_tv_attn=False,
            capture_layers=None,
        )

        self.assertEqual(hidden_states.shape[1], 5)
        self.assertEqual(prune_info["original_seq_len"], 5)
        self.assertEqual(prune_info["final_seq_len"], 5)
        self.assertEqual(sorted(prune_info["layers"].keys()), [1, 2])
        self.assertEqual(prune_info["layers"][1]["prune_stage"], "masking")
        self.assertEqual(prune_info["layers"][2]["prune_stage"], "masking")
        self.assertEqual(len(prune_info["layers"][1]["keep_indices"]), 2)
        self.assertEqual(len(prune_info["layers"][2]["keep_indices"]), 2)
        self.assertEqual(model.model.layers[0].seq_lens, [5])
        self.assertEqual(model.model.layers[1].seq_lens, [])
        self.assertEqual(model.model.layers[2].seq_lens, [])
        self.assertEqual(past_kv.get_seq_length(), 5)
        self.assertEqual(past_kv.key_cache[0].shape[2], 5)
        self.assertEqual(past_kv.key_cache[1].shape[2], 5)
        self.assertEqual(past_kv.key_cache[2].shape[2], 5)


if __name__ == "__main__":
    unittest.main()
