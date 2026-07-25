from __future__ import annotations

import os
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "algo_compare" / "src" / "algo_compare" / "llava_next_official.py"
SPEC = importlib.util.spec_from_file_location("llava_next_official", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
compat = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = compat
SPEC.loader.exec_module(compat)

apply_divprune_to_prepared_inputs = compat.apply_divprune_to_prepared_inputs
canonical_anyres_best_resolution = compat.canonical_anyres_best_resolution
configure_cdpruner_model = compat.configure_cdpruner_model
generate_with_cdpruner = compat.generate_with_cdpruner
image_crop_count = compat.image_crop_count
install_cdpruner_next_image_adapter = compat.install_cdpruner_next_image_adapter
install_divprune_next_adapter = compat.install_divprune_next_adapter
is_llava_next_config = compat.is_llava_next_config


class FakeDivPruneModel:
    def DivPrune(self, visual_tokens, image_feature_length, cosine_matrix=None, threshold_ratio=0.5):
        del visual_tokens, cosine_matrix
        count = round(image_feature_length * threshold_ratio)
        return torch.arange(0, count * 2, 2), None


class FakeWrappedDivPruneModel(FakeDivPruneModel):
    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        images,
        image_sizes=None,
    ):
        del input_ids, position_ids, attention_mask, labels, images, image_sizes
        embeds = torch.arange(8 * 2, dtype=torch.float32).reshape(1, 8, 2)
        return None, None, None, past_key_values, embeds, None


class FakeGenerationBase:
    def generate(self, **kwargs):
        self.parent_generate_kwargs = kwargs
        return torch.tensor([[7, 8]])


class FakeCDPrunerMistral(FakeGenerationBase):
    def __init__(self):
        self.config = SimpleNamespace(model_type="llava_mistral")
        self.model = SimpleNamespace()
        self._vision_tower = SimpleNamespace(num_patches=576)

    def get_vision_tower(self):
        return self._vision_tower

    def prepare_inputs_labels_for_multimodal(self, *args, **kwargs):
        self.prepare_args = args
        self.prepare_kwargs = kwargs
        return None, None, None, None, torch.ones(1, 6, 4), None, 126

    def generate(self, *args, **kwargs):
        raise AssertionError("The unadapted Mistral generate path must be bypassed for images")


class LlavaNextOfficialCompatTest(unittest.TestCase):
    def test_next_config_detection(self):
        self.assertTrue(is_llava_next_config(SimpleNamespace(model_type="llava_mistral")))
        self.assertTrue(is_llava_next_config(SimpleNamespace(image_aspect_ratio="anyres")))
        self.assertFalse(is_llava_next_config(SimpleNamespace(model_type="llava_llama")))

    def test_image_crop_count(self):
        self.assertEqual(image_crop_count(torch.zeros(1, 5, 3, 4, 4)), 5)
        self.assertEqual(image_crop_count(torch.zeros(3, 4, 4)), 1)
        self.assertEqual(image_crop_count(None), 0)

    def test_divprune_dynamic_visual_span(self):
        model = FakeDivPruneModel()
        input_ids = torch.tensor([[11, -200, 12, 13]])
        prepared = (None, None, None, None, torch.arange(8 * 2).reshape(1, 8, 2), None)
        pruned, stats = apply_divprune_to_prepared_inputs(
            model,
            input_ids=input_ids,
            input_attention_mask=None,
            prepared=prepared,
            image_token_index=-200,
            subset_ratio=0.4,
        )
        self.assertEqual(stats["visual_start"], 1)
        self.assertEqual(stats["tokens_before"], 5)
        self.assertEqual(stats["tokens_after"], 2)
        self.assertEqual(tuple(pruned[4].shape), (1, 5, 2))

    def test_divprune_wrapper_suppresses_upstream_fixed_offset(self):
        model = FakeWrappedDivPruneModel()
        install_divprune_next_adapter(model, image_token_index=-200, subset_ratio=0.4)
        os.environ["LAYER_INDEX"] = "0"
        try:
            result = model.prepare_inputs_labels_for_multimodal(
                torch.tensor([[11, -200, 12, 13]]),
                None,
                None,
                None,
                None,
                torch.zeros(1, 5, 3, 4, 4),
            )
        finally:
            os.environ.pop("LAYER_INDEX", None)
        self.assertEqual(tuple(result[4].shape), (1, 5, 2))
        self.assertEqual(model._divprune_last_stats["crop_count"], 5)

    def test_cdpruner_mistral_generation_bridge(self):
        model = FakeCDPrunerMistral()
        configure_cdpruner_model(model, 126)
        output, effective = generate_with_cdpruner(
            model,
            torch.tensor([[1, -200, 2]]),
            images=torch.zeros(1, 5, 3, 4, 4),
            image_sizes=[(8, 8)],
            texts="what is shown?",
            max_new_tokens=4,
        )
        self.assertEqual(effective, 126)
        self.assertTrue(torch.equal(output, torch.tensor([[7, 8]])))
        self.assertEqual(model.visual_token_num, 126)
        self.assertEqual(model.model.visual_token_num, 126)
        self.assertEqual(model.prepare_kwargs["texts"], "what is shown?")
        self.assertEqual(model.parent_generate_kwargs["max_new_tokens"], 4)

    def test_cdpruner_budget_validation(self):
        model = FakeCDPrunerMistral()
        with self.assertRaises(ValueError):
            configure_cdpruner_model(model, 577)

    def test_cdpruner_restores_canonical_anyres_resolution(self):
        resolutions = [(336, 672), (672, 336), (672, 672)]
        expected = canonical_anyres_best_resolution((1000, 500), resolutions)
        module = SimpleNamespace(
            select_best_resolution=lambda original_size, candidates: (672, 672)
        )
        install_cdpruner_next_image_adapter(module)
        self.assertEqual(module.select_best_resolution((1000, 500), resolutions), expected)
        self.assertEqual(expected, (672, 336))
        self.assertTrue(module._cdpruner_next_image_adapter_installed)

    def test_canonical_anyres_rejects_empty_candidates(self):
        with self.assertRaises(ValueError):
            canonical_anyres_best_resolution((640, 480), [])


if __name__ == "__main__":
    unittest.main()
