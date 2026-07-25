import os
import sys
import types
import unittest

import torch

from llava.mm_utils import get_visual_token_layout
from llava.model.llava_arch import unpad_image


SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC_DIR)

from prune_inference import (  # noqa: E402
    normalize_configured_v_token_num,
    validate_loaded_model_runtime,
)
from pruner import (  # noqa: E402
    _make_causal_mask,
    enable_sparse_position_ids_compat,
)


class VisualTokenLayoutTests(unittest.TestCase):
    def setUp(self):
        self.anyres_config = types.SimpleNamespace(
            image_aspect_ratio="anyres",
            image_grid_pinpoints=[
                [336, 672],
                [672, 336],
                [672, 672],
                [1008, 336],
                [336, 1008],
            ],
            mm_patch_merge_type="spatial_unpad",
        )

    def test_anyres_spatial_unpad_matches_merged_sequence_length(self):
        layout = get_visual_token_layout(
            (500, 280),
            self.anyres_config,
            patches_per_side=24,
            vision_image_size=336,
            num_image_crops=3,
        )

        self.assertEqual(layout["base_token_count"], 576)
        self.assertEqual(layout["crop_grid_width"], 2)
        self.assertEqual(layout["crop_grid_height"], 1)
        self.assertEqual(layout["local_grid_height"], 24)
        self.assertEqual(layout["local_grid_width"], 42)
        self.assertEqual(layout["local_patch_token_count"], 1008)
        self.assertEqual(layout["newline_token_count"], 24)
        self.assertEqual(layout["total_token_count"], 1608)
        self.assertEqual(layout["visual_index_semantics"], "merged_visual_sequence")

        feature_map = torch.zeros((4, 24, 48))
        self.assertEqual(tuple(unpad_image(feature_map, (500, 280)).shape), (4, 24, 42))

    def test_anyres_rejects_crop_count_mismatch(self):
        with self.assertRaisesRegex(ValueError, "crop count"):
            get_visual_token_layout(
                (500, 280),
                self.anyres_config,
                patches_per_side=24,
                vision_image_size=336,
                num_image_crops=2,
            )

    def test_fixed_square_layout_remains_576(self):
        config = types.SimpleNamespace(
            image_aspect_ratio="square",
            mm_patch_merge_type="flat",
        )
        layout = get_visual_token_layout(
            (640, 480),
            config,
            patches_per_side=24,
            vision_image_size=336,
            num_image_crops=1,
        )
        self.assertEqual(layout["total_token_count"], 576)
        self.assertEqual(layout["layout_kind"], "square")


class _DummyVisionTower:
    num_patches = 576
    num_patches_per_side = 24


class _DummyRuntimeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(32)])
        self._vision_tower = _DummyVisionTower()

    def get_vision_tower(self):
        return self._vision_tower


class DynamicVisualTokenConfigTests(unittest.TestCase):
    def test_v_token_num_normalization(self):
        self.assertEqual(normalize_configured_v_token_num("auto"), "auto")
        self.assertEqual(normalize_configured_v_token_num("576"), 576)
        self.assertEqual(normalize_configured_v_token_num(576), 576)
        for invalid in (0, -1, True, "dynamic"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    normalize_configured_v_token_num(invalid)

    def test_anyres_runtime_requires_auto_count(self):
        model = _DummyRuntimeModel()
        metadata = {
            "num_hidden_layers": 32,
            "image_aspect_ratio": "anyres",
            "mm_patch_merge_type": "spatial_unpad",
        }
        runtime = validate_loaded_model_runtime(model, metadata, "auto")
        self.assertEqual(runtime["visual_token_count_mode"], "auto")
        self.assertEqual(runtime["vision_token_count"], 576)

        with self.assertRaisesRegex(RuntimeError, "require pruning.v_token_num=auto"):
            validate_loaded_model_runtime(model, metadata, 576)

    def test_fixed_layout_keeps_strict_vision_count_check(self):
        model = _DummyRuntimeModel()
        metadata = {
            "num_hidden_layers": 32,
            "image_aspect_ratio": "square",
            "mm_patch_merge_type": "flat",
        }
        validate_loaded_model_runtime(model, metadata, 576)
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            validate_loaded_model_runtime(model, metadata, 575)


class MistralSparsePositionCompatibilityTests(unittest.TestCase):
    def test_mistral_patch_preserves_dense_output_and_supports_sparse_cache(self):
        from transformers import MistralConfig, MistralForCausalLM
        from transformers.cache_utils import DynamicCache

        torch.manual_seed(7)
        config = MistralConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
            attention_dropout=0.0,
        )
        config._attn_implementation = "eager"
        model = MistralForCausalLM(config).eval()
        hidden = torch.randn(1, 6, 32)
        dense_positions = torch.arange(6, dtype=torch.long).unsqueeze(0)
        dense_mask = _make_causal_mask(6, hidden.dtype, hidden.device)

        with torch.no_grad():
            native = model.model.layers[0](
                hidden,
                attention_mask=dense_mask,
                position_ids=dense_positions,
                use_cache=False,
                output_attentions=True,
            )

        self.assertTrue(enable_sparse_position_ids_compat(model))
        self.assertEqual(model._sparse_position_ids_compat_family, "mistral")
        with torch.no_grad():
            patched = model.model.layers[0](
                hidden,
                attention_mask=dense_mask,
                position_ids=dense_positions,
                use_cache=False,
                output_attentions=True,
            )
        torch.testing.assert_close(patched[0], native[0], rtol=0, atol=0)
        torch.testing.assert_close(patched[1], native[1], rtol=0, atol=0)

        sparse_hidden = hidden[:, :4]
        sparse_positions = torch.tensor([[0, 1, 3, 6]], dtype=torch.long)
        sparse_mask = _make_causal_mask(4, hidden.dtype, hidden.device)
        cache = DynamicCache()
        with torch.no_grad():
            current = sparse_hidden
            for layer in model.model.layers:
                current = layer(
                    current,
                    attention_mask=sparse_mask,
                    position_ids=sparse_positions,
                    past_key_value=cache,
                    use_cache=True,
                    output_attentions=False,
                )[0]

            decode_hidden = torch.randn(1, 1, 32)
            decode_mask = _make_causal_mask(
                1,
                decode_hidden.dtype,
                decode_hidden.device,
                past_kv_len=cache.get_seq_length(),
            )
            for layer in model.model.layers:
                decode_hidden = layer(
                    decode_hidden,
                    attention_mask=decode_mask,
                    position_ids=torch.tensor([[7]], dtype=torch.long),
                    past_key_value=cache,
                    use_cache=True,
                    output_attentions=False,
                )[0]

        self.assertEqual(tuple(current.shape), (1, 4, 32))
        self.assertEqual(tuple(decode_hidden.shape), (1, 1, 32))
        self.assertEqual(cache.get_seq_length(), 5)


class LlamaSparsePositionCompatibilityTests(unittest.TestCase):
    def test_llama_patch_preserves_dense_output_and_supports_sparse_cache(self):
        from transformers import LlamaConfig, LlamaForCausalLM
        from transformers.cache_utils import DynamicCache

        torch.manual_seed(11)
        config = LlamaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
            attention_dropout=0.0,
        )
        config._attn_implementation = "eager"
        model = LlamaForCausalLM(config).eval()
        hidden = torch.randn(1, 6, 32)
        dense_positions = torch.arange(6, dtype=torch.long).unsqueeze(0)
        dense_mask = _make_causal_mask(6, hidden.dtype, hidden.device)

        with torch.no_grad():
            native = model.model.layers[0](
                hidden,
                attention_mask=dense_mask,
                position_ids=dense_positions,
                use_cache=False,
                output_attentions=True,
            )

        self.assertTrue(enable_sparse_position_ids_compat(model))
        self.assertEqual(model._sparse_position_ids_compat_family, "llama")
        with torch.no_grad():
            patched = model.model.layers[0](
                hidden,
                attention_mask=dense_mask,
                position_ids=dense_positions,
                use_cache=False,
                output_attentions=True,
            )
        torch.testing.assert_close(patched[0], native[0], rtol=0, atol=0)
        torch.testing.assert_close(patched[1], native[1], rtol=0, atol=0)

        sparse_hidden = hidden[:, :4]
        sparse_positions = torch.tensor([[0, 1, 3, 6]], dtype=torch.long)
        sparse_mask = _make_causal_mask(4, hidden.dtype, hidden.device)
        cache = DynamicCache()
        with torch.no_grad():
            current = sparse_hidden
            for layer in model.model.layers:
                current = layer(
                    current,
                    attention_mask=sparse_mask,
                    position_ids=sparse_positions,
                    past_key_value=cache,
                    use_cache=True,
                    output_attentions=False,
                )[0]

            decode_hidden = torch.randn(1, 1, 32)
            decode_mask = _make_causal_mask(
                1,
                decode_hidden.dtype,
                decode_hidden.device,
                past_kv_len=cache.get_seq_length(),
            )
            for layer in model.model.layers:
                decode_hidden = layer(
                    decode_hidden,
                    attention_mask=decode_mask,
                    position_ids=torch.tensor([[7]], dtype=torch.long),
                    past_key_value=cache,
                    use_cache=True,
                    output_attentions=False,
                )[0]

        self.assertEqual(tuple(current.shape), (1, 4, 32))
        self.assertEqual(tuple(decode_hidden.shape), (1, 1, 32))
        self.assertEqual(cache.get_seq_length(), 5)


if __name__ == "__main__":
    unittest.main()
