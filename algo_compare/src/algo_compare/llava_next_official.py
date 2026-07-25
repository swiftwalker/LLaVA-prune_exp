"""Compatibility helpers for official pruning methods on LLaVA-NeXT.

The helpers keep third-party checkouts pristine. They only repair model-family
interface assumptions around the official selection implementations.
"""

from __future__ import annotations

import os
import types
from typing import Any

import torch


def is_llava_next_config(config: Any) -> bool:
    """Return whether a model config uses the LLaVA-NeXT multimodal path."""

    model_type = str(getattr(config, "model_type", "")).lower()
    aspect_ratio = str(getattr(config, "image_aspect_ratio", "")).lower()
    patch_merge = str(getattr(config, "mm_patch_merge_type", "")).lower()
    return model_type == "llava_mistral" or aspect_ratio == "anyres" or patch_merge.startswith("spatial")


def llava_language_backbone(model_or_config: Any) -> str:
    """Return the decoder family used by a LLaVA model.

    Original LLaVA-NeXT Vicuna checkpoints use ``model_type=llava`` while the
    official pruning forks instantiate ``LlavaLlamaForCausalLM``.  Looking at
    both the config and runtime class keeps that path distinct from Mistral.
    """

    config = getattr(model_or_config, "config", model_or_config)
    text_config = getattr(config, "text_config", None)
    candidates = (
        str(getattr(config, "model_type", "")).lower(),
        str(getattr(text_config, "model_type", "")).lower(),
        type(model_or_config).__name__.lower(),
    )
    if any("mistral" in value for value in candidates):
        return "mistral"
    if any("llama" in value or value == "llava" for value in candidates):
        return "llama"
    return "unknown"


def image_crop_count(images: Any) -> int:
    """Count CLIP crops in the batch-size-one image representation."""

    if images is None:
        return 0
    if isinstance(images, (list, tuple)):
        return sum(image_crop_count(image) for image in images)
    if not isinstance(images, torch.Tensor):
        return 0
    if images.ndim == 5:
        return int(images.shape[1])
    if images.ndim == 4:
        return int(images.shape[0])
    return 1


def canonical_anyres_best_resolution(
    original_size: tuple[int, int],
    possible_resolutions: list[tuple[int, int]],
) -> tuple[int, int]:
    """Select the canonical LLaVA-NeXT anyres canvas.

    CDPruner's official fork overrides the candidate list with ``[(672, 672)]``.
    That changes both the image evidence and compute budget. The compatibility
    wrapper restores the upstream LLaVA-NeXT policy without editing the checkout.
    """

    if not possible_resolutions:
        raise ValueError("possible_resolutions must not be empty")
    original_width, original_height = original_size
    best_fit = possible_resolutions[0]
    max_effective_resolution = -1
    min_wasted_resolution = float("inf")
    for width, height in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width = int(original_width * scale)
        downscaled_height = int(original_height * scale)
        effective_resolution = min(
            downscaled_width * downscaled_height,
            original_width * original_height,
        )
        wasted_resolution = width * height - effective_resolution
        if effective_resolution > max_effective_resolution or (
            effective_resolution == max_effective_resolution
            and wasted_resolution < min_wasted_resolution
        ):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)
    return best_fit


def install_cdpruner_next_image_adapter(mm_utils_module: Any) -> None:
    """Restore canonical anyres crop selection in the official CDPruner fork."""

    if getattr(mm_utils_module, "_cdpruner_next_image_adapter_installed", False):
        return
    mm_utils_module._cdpruner_original_select_best_resolution = (
        mm_utils_module.select_best_resolution
    )
    mm_utils_module.select_best_resolution = canonical_anyres_best_resolution
    mm_utils_module._cdpruner_next_image_adapter_installed = True


def _valid_input_ids(input_ids: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("The LLaVA-NeXT official adapters currently require batch_size=1")
    if attention_mask is None:
        return input_ids[0]
    return input_ids[0][attention_mask[0].to(dtype=torch.bool)]


def _index_sequence_tensor(value: torch.Tensor | None, indices: torch.Tensor) -> torch.Tensor | None:
    if value is None:
        return None
    if value.ndim < 2:
        raise ValueError(f"Expected a sequence tensor with at least 2 dimensions, got {tuple(value.shape)}")
    return value.index_select(1, indices.to(device=value.device))


def apply_divprune_to_prepared_inputs(
    model: Any,
    *,
    input_ids: torch.Tensor,
    input_attention_mask: torch.Tensor | None,
    prepared: tuple[Any, ...],
    image_token_index: int,
    subset_ratio: float,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Prune the merged visual span returned by official LLaVA preparation."""

    if len(prepared) != 6:
        raise ValueError(f"Expected six official multimodal outputs, got {len(prepared)}")
    if not 0.0 < subset_ratio <= 1.0:
        raise ValueError(f"DivPrune subset_ratio must be in (0, 1], got {subset_ratio}")

    _, position_ids, attention_mask, past_key_values, inputs_embeds, labels = prepared
    if inputs_embeds is None:
        return prepared, {"enabled": False, "reason": "no_visual_embeddings"}
    if inputs_embeds.ndim != 3 or inputs_embeds.shape[0] != 1:
        raise ValueError("The LLaVA-NeXT DivPrune adapter requires one prepared sample")

    valid_ids = _valid_input_ids(input_ids, input_attention_mask)
    image_positions = torch.where(valid_ids == image_token_index)[0]
    if image_positions.numel() != 1:
        raise ValueError(
            "The LLaVA-NeXT DivPrune adapter currently supports exactly one image token; "
            f"found {int(image_positions.numel())}"
        )

    prepared_length = int(inputs_embeds.shape[1])
    if attention_mask is not None:
        prepared_length = int(attention_mask[0].to(dtype=torch.long).sum().item())
    text_token_count = int(valid_ids.numel()) - 1
    visual_token_count = prepared_length - text_token_count
    visual_start = int(image_positions[0].item())
    visual_end = visual_start + visual_token_count
    if visual_token_count < 2 or visual_end > prepared_length:
        raise ValueError(
            "Could not infer a valid merged visual span: "
            f"start={visual_start}, visual={visual_token_count}, prepared={prepared_length}"
        )

    visual_tokens = inputs_embeds[0, visual_start:visual_end]
    selected, _ = model.DivPrune(
        visual_tokens,
        visual_token_count,
        cosine_matrix=None,
        threshold_ratio=subset_ratio,
    )
    selected = torch.sort(selected.to(dtype=torch.long)).values
    retained_count = int(selected.numel())
    expected_count = int(round(subset_ratio * visual_token_count))
    if retained_count != expected_count:
        raise RuntimeError(f"Official DivPrune selected {retained_count} tokens; expected {expected_count}")

    sequence_device = inputs_embeds.device
    keep_indices = torch.cat(
        (
            torch.arange(visual_start, device=sequence_device),
            selected.to(device=sequence_device) + visual_start,
            torch.arange(visual_end, prepared_length, device=sequence_device),
        )
    )
    pruned = (
        None,
        _index_sequence_tensor(position_ids, keep_indices),
        _index_sequence_tensor(attention_mask, keep_indices),
        past_key_values,
        _index_sequence_tensor(inputs_embeds, keep_indices),
        _index_sequence_tensor(labels, keep_indices),
    )
    stats = {
        "enabled": True,
        "adapter": "llava_next_dynamic_visual_span",
        "visual_start": visual_start,
        "tokens_before": visual_token_count,
        "tokens_after": retained_count,
        "effective_ratio": retained_count / visual_token_count,
        "selected_indices": selected.detach().cpu().tolist(),
    }
    return pruned, stats


def install_divprune_next_adapter(
    model: Any,
    *,
    image_token_index: int,
    subset_ratio: float,
    layer_index: int = 0,
) -> None:
    """Wrap official multimodal preparation without modifying its checkout."""

    if layer_index != 0:
        raise ValueError("The LLaVA-NeXT DivPrune adapter only supports the official pre-LLM layer_index=0")
    if getattr(model, "_divprune_next_adapter_installed", False):
        return

    original_prepare = model.prepare_inputs_labels_for_multimodal

    def _prepare(
        self: Any,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        past_key_values: Any,
        labels: torch.Tensor | None,
        images: Any,
        image_sizes: Any = None,
    ) -> tuple[Any, ...]:
        # Prevent the upstream SYS_TOKEN_LEN=35 block from running. The official
        # DivPrune selector is called below on the correctly inferred span.
        saved_layer_index = os.environ.pop("LAYER_INDEX", None)
        try:
            prepared = original_prepare(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes,
            )
        finally:
            if saved_layer_index is not None:
                os.environ["LAYER_INDEX"] = saved_layer_index

        if images is None or input_ids.shape[1] == 1 or os.environ.get("BASELINE", "OURS") != "OURS":
            return prepared
        pruned, stats = apply_divprune_to_prepared_inputs(
            self,
            input_ids=input_ids,
            input_attention_mask=attention_mask,
            prepared=tuple(prepared),
            image_token_index=image_token_index,
            subset_ratio=subset_ratio,
        )
        stats["crop_count"] = image_crop_count(images)
        self._divprune_last_stats = stats
        return pruned

    model.prepare_inputs_labels_for_multimodal = types.MethodType(_prepare, model)
    model._divprune_next_adapter_installed = True
    model._divprune_last_stats = {"enabled": False, "reason": "not_run"}


def configure_cdpruner_model(model: Any, visual_token_num: int) -> None:
    """Attach the official per-crop CDPruner budget to a loaded model."""

    if visual_token_num <= 0:
        raise ValueError(f"visual_token_num must be positive, got {visual_token_num}")
    vision_tower = model.get_vision_tower()
    max_tokens = int(getattr(vision_tower, "num_patches", 0) or 0)
    if max_tokens and visual_token_num > max_tokens:
        raise ValueError(f"visual_token_num={visual_token_num} exceeds the per-crop token count {max_tokens}")
    model.visual_token_num = int(visual_token_num)
    if hasattr(model, "model"):
        model.model.visual_token_num = int(visual_token_num)


def is_cdpruner_mistral(model: Any) -> bool:
    return llava_language_backbone(model) == "mistral"


@torch.no_grad()
def generate_with_cdpruner(
    model: Any,
    inputs: torch.Tensor,
    *,
    images: Any = None,
    image_sizes: Any = None,
    texts: str | None = None,
    **kwargs: Any,
) -> tuple[Any, int]:
    """Run official CDPruner through its native Llama or bridged Mistral path."""

    if not is_cdpruner_mistral(model):
        output = model.generate(inputs, images=images, image_sizes=image_sizes, texts=texts, **kwargs)
        if isinstance(output, tuple) and len(output) == 2:
            return output[0], int(output[1])
        return output, 0 if images is None else int(getattr(model, "visual_token_num", 0))

    if images is None:
        return model.generate(inputs, **kwargs), 0

    position_ids = kwargs.pop("position_ids", None)
    attention_mask = kwargs.pop("attention_mask", None)
    if "inputs_embeds" in kwargs:
        raise NotImplementedError("inputs_embeds is managed by the CDPruner Mistral adapter")

    prepared = model.prepare_inputs_labels_for_multimodal(
        inputs,
        position_ids,
        attention_mask,
        None,
        None,
        images,
        image_sizes=image_sizes,
        texts=texts,
    )
    if len(prepared) != 7:
        raise ValueError(f"Expected seven CDPruner multimodal outputs, got {len(prepared)}")
    _, position_ids, attention_mask, _, inputs_embeds, _, effective_visual_tokens = prepared
    parent_generate = super(type(model), model).generate
    output = parent_generate(
        position_ids=position_ids,
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
        **kwargs,
    )
    return output, int(effective_visual_tokens)
