import os
import types
import sys
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

from pruner import VisualTokenPruner, _required_rotary_seq_len
from strategies.base import PruneStrategy
from strategies.random import RandomStrategy


class DummyLayer:
    def __init__(self, layer_idx: int):
        self.layer_idx = layer_idx
        self.output_attentions_history = []
        self.position_ids_history = []

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
        if position_ids is not None:
            self.position_ids_history.append(position_ids.detach().clone())
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
            attn = hidden_states.new_full((1, 1, seq_len, seq_len), 1.0 / max(seq_len, 1))
            return (hidden_out, attn)
        return (hidden_out, past_key_value)


class DummyBackbone:
    def __init__(self, num_layers: int):
        self.layers = [DummyLayer(i) for i in range(num_layers)]
        self.norm = torch.nn.Identity()
        self.embed_tokens = torch.nn.Embedding(8, 4)


class DummyModel:
    def __init__(self, num_layers: int):
        self.model = DummyBackbone(num_layers)
        self.lm_head = torch.nn.Linear(4, 8, bias=False)
        with torch.no_grad():
            self.lm_head.weight.zero_()


class FixedPostPruneStrategy(PruneStrategy):
    def requires_attention(self) -> bool:
        return False

    def compute_importance(
        self,
        attn_weights,
        v_token_start,
        v_token_num,
        text_token_start,
        layer_idx,
        device=None,
    ):
        if device is None:
            device = torch.device("cpu")
        return torch.arange(v_token_num, 0, -1, device=device, dtype=torch.float32)


class PrunerAttentionRequirementTests(unittest.TestCase):
    def _build_pruner(self):
        model = DummyModel(num_layers=2)
        strategy = RandomStrategy({"prune_ratio": [0.5]})
        prune_config = {
            "layer_selection": "fixed",
            "prune_layers": [0],
            "prune_ratio": [0.5],
        }
        return model, VisualTokenPruner(model, strategy, prune_config)

    def test_random_strategy_can_prune_without_attention(self):
        model, pruner = self._build_pruner()
        torch.manual_seed(0)

        hidden_states, past_kv, prune_info = pruner._pruned_prefill(
            inputs_embeds=torch.zeros((1, 6, 4)),
            initial_position_ids=None,
            v_token_start=1,
            v_token_num=4,
            text_token_start=5,
            save_tv_attn=False,
            capture_layers=None,
        )

        self.assertEqual(model.model.layers[0].output_attentions_history, [False])
        self.assertEqual(model.model.layers[1].output_attentions_history, [False])
        self.assertEqual(prune_info["layers"][0]["num_visual_after"], 2)
        self.assertEqual(hidden_states.shape[1], 4)
        self.assertTrue(hasattr(past_kv, "key_cache"))
        self.assertTrue(hasattr(past_kv, "value_cache"))

    def test_capture_enabled_still_collects_attention(self):
        model, pruner = self._build_pruner()
        torch.manual_seed(0)

        _, _, prune_info = pruner._pruned_prefill(
            inputs_embeds=torch.zeros((1, 6, 4)),
            initial_position_ids=None,
            v_token_start=1,
            v_token_num=4,
            text_token_start=5,
            save_tv_attn=True,
            capture_layers={0},
        )

        self.assertEqual(model.model.layers[0].output_attentions_history, [True])
        self.assertIn("tv_attn", prune_info["layers"][0])
        self.assertEqual(tuple(prune_info["layers"][0]["tv_attn"].shape), (1, 1, 4))

    def test_required_rotary_seq_len_uses_max_position_id(self):
        position_ids = torch.tensor([[10, 11, 12, 15]], dtype=torch.long)
        self.assertEqual(_required_rotary_seq_len(position_ids, 4), 16)
        self.assertEqual(_required_rotary_seq_len(None, 4), 4)

    def test_physical_pruning_preserves_position_ids_and_decode_advances_from_last_original_index(self):
        model = DummyModel(num_layers=2)
        strategy = FixedPostPruneStrategy({"prune_ratio": [0.5]})
        prune_config = {
            "layer_selection": "fixed",
            "prune_layers": [0],
            "prune_ratio": [0.5],
        }
        pruner = VisualTokenPruner(model, strategy, prune_config)

        _, prune_info = pruner.pruned_generate(
            inputs_embeds=torch.zeros((1, 6, 4)),
            attention_mask=None,
            position_ids=torch.tensor([[10, 11, 12, 13, 14, 15]], dtype=torch.long),
            v_token_start=1,
            v_token_num=4,
            text_token_start=5,
            max_new_tokens=2,
            eos_token_id=7,
        )

        self.assertEqual(prune_info["final_position_ids"], [10, 11, 12, 15])
        self.assertEqual(model.model.layers[0].position_ids_history[0].tolist(), [[10, 11, 12, 13, 14, 15]])
        self.assertEqual(model.model.layers[1].position_ids_history[0].tolist(), [[10, 11, 12, 15]])
        self.assertEqual(model.model.layers[0].position_ids_history[1].tolist(), [[16]])
        self.assertEqual(model.model.layers[1].position_ids_history[1].tolist(), [[16]])


if __name__ == "__main__":
    unittest.main()
