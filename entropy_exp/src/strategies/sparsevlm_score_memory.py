"""Shared helpers for SparseVLM score-reweighting pruning variants."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from .sparsevlm import SparseVLMStrategy


def deterministic_descending_indices(scores: torch.Tensor) -> torch.Tensor:
    """Return score-descending indices with token-index tie breaking."""
    if scores.ndim != 1:
        raise ValueError(f"scores must be 1D, got {tuple(scores.shape)}")
    score_values = scores.detach().cpu().tolist()
    order = sorted(
        range(int(scores.numel())),
        key=lambda idx: (-float(score_values[idx]), idx),
    )
    return torch.tensor(order, device=scores.device, dtype=torch.long)


def deterministic_topk_indices(scores: torch.Tensor, keep_count: int) -> torch.Tensor:
    """Select top-k by score and return indices in sequence order."""
    if keep_count < 0:
        raise ValueError(f"keep_count must be >= 0, got {keep_count}")
    if keep_count >= int(scores.numel()):
        return torch.arange(scores.numel(), device=scores.device, dtype=torch.long)
    if keep_count == 0:
        return torch.empty(0, device=scores.device, dtype=torch.long)
    return deterministic_descending_indices(scores)[:keep_count].sort().values


def rank_normalize_scores(scores: torch.Tensor) -> torch.Tensor:
    """Map scores to [0, 1] by descending rank; highest score receives 1."""
    if scores.ndim != 1:
        raise ValueError(f"scores must be 1D, got {tuple(scores.shape)}")
    num_scores = int(scores.numel())
    if num_scores == 0:
        return scores.to(dtype=torch.float32)
    if num_scores == 1:
        return torch.ones_like(scores, dtype=torch.float32)

    order = deterministic_descending_indices(scores)
    rank_values = torch.linspace(1.0, 0.0, steps=num_scores, device=scores.device, dtype=torch.float32)
    normalized = torch.empty(num_scores, device=scores.device, dtype=torch.float32)
    normalized[order] = rank_values
    return normalized


def compute_entropy_stats(scores: torch.Tensor) -> Tuple[float, float]:
    """Compute saliency entropy using the same normalized entropy口径 as entropy-alpha."""
    if scores.ndim != 1:
        raise ValueError(f"scores must be 1D, got {tuple(scores.shape)}")
    num_tokens = int(scores.numel())
    if num_tokens <= 1:
        return 0.0, 0.0

    positive_scores = scores.to(dtype=torch.float64).clamp_min(0.0)
    score_sum = positive_scores.sum()
    if float(score_sum.item()) <= 0.0:
        return 0.0, 0.0

    probs = positive_scores / score_sum
    positive_mask = probs > 0
    entropy_raw = -(probs[positive_mask] * torch.log(probs[positive_mask])).sum()
    entropy_denom = torch.log(positive_scores.new_tensor(float(num_tokens)))
    entropy_norm = torch.clamp(entropy_raw / entropy_denom, min=0.0, max=1.0)
    return float(entropy_raw.item()), float(entropy_norm.item())


class SparseVLMScoreMemoryStrategy(SparseVLMStrategy):
    """SparseVLM base with sample-local original-patch mapping and EMA memory."""

    def __init__(self, config: dict):
        super().__init__(config)
        self._sample_counter = 0

    def prepare_sample(
        self,
        inputs_embeds: torch.Tensor,
        v_token_start: int,
        v_token_num: int,
        text_token_start: int,
        text_token_ids: Optional[torch.Tensor] = None,
        text_special_token_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        sample_info = super().prepare_sample(
            inputs_embeds=inputs_embeds,
            v_token_start=v_token_start,
            v_token_num=v_token_num,
            text_token_start=text_token_start,
            text_token_ids=text_token_ids,
            text_special_token_mask=text_special_token_mask,
        )
        context = self._require_sample_context()
        device = inputs_embeds.device
        context["current_patch_indices"] = torch.arange(v_token_num, device=device, dtype=torch.long)
        context["initial_v_token_num"] = int(v_token_num)
        context["score_memory_ema"] = torch.zeros(v_token_num, device=device, dtype=torch.float32)
        context["score_memory_observed"] = torch.zeros(v_token_num, device=device, dtype=torch.bool)
        context["score_prune_step"] = 0
        context["score_pending_decision"] = None
        context["score_sample_index"] = int(self._sample_counter)
        self._sample_counter += 1
        return sample_info

    def _get_memory_params(self) -> Tuple[float, float]:
        current_weight = float(self.config.get("global_current_weight", 0.5))
        ema_decay = float(self.config.get("global_ema_decay", 0.6))
        if not 0.0 <= current_weight <= 1.0:
            raise ValueError(f"global_current_weight must be in [0, 1], got {current_weight}")
        if not 0.0 <= ema_decay <= 1.0:
            raise ValueError(f"global_ema_decay must be in [0, 1], got {ema_decay}")
        return current_weight, ema_decay

    def _require_current_patch_indices(self, context: Dict[str, Any], v_token_num: int) -> torch.Tensor:
        current_patch_indices = context["current_patch_indices"]
        if int(current_patch_indices.numel()) != int(v_token_num):
            raise ValueError(
                "current_patch_indices length must match the current visual token count, "
                f"got {current_patch_indices.numel()} vs {v_token_num}"
            )
        return current_patch_indices

    def _alive_ema(self, context: Dict[str, Any], current_patch_indices: torch.Tensor, device: torch.device) -> torch.Tensor:
        ema = context["score_memory_ema"]
        patch_indices = current_patch_indices.to(device=ema.device, dtype=torch.long)
        return ema.index_select(0, patch_indices).to(device=device, dtype=torch.float32)

    def _update_ema(
        self,
        context: Dict[str, Any],
        current_patch_indices: torch.Tensor,
        current_rank_score: torch.Tensor,
        ema_decay: float,
    ) -> None:
        ema = context["score_memory_ema"]
        observed = context["score_memory_observed"]
        patch_indices = current_patch_indices.to(device=ema.device, dtype=torch.long)
        current_values = current_rank_score.to(device=ema.device, dtype=torch.float32)

        previous = ema.index_select(0, patch_indices)
        was_observed = observed.index_select(0, patch_indices)
        updated = torch.where(
            was_observed,
            previous * ema_decay + current_values * (1.0 - ema_decay),
            current_values,
        )
        ema[patch_indices] = updated.detach()
        observed[patch_indices] = True

    def _score_memory_inputs(
        self,
        attn_weights: Optional[torch.Tensor],
        v_token_start: int,
        v_token_num: int,
        text_token_start: int,
        layer_idx: int,
        device: Optional[torch.device],
    ) -> Dict[str, Any]:
        if attn_weights is None:
            raise ValueError(f"{self.__class__.__name__} requires attention weights")

        context = self._require_sample_context()
        current_patch_indices = self._require_current_patch_indices(context, v_token_num)
        current_weight, ema_decay = self._get_memory_params()

        visual_scores = self.compute_importance(
            attn_weights=attn_weights,
            v_token_start=v_token_start,
            v_token_num=v_token_num,
            text_token_start=text_token_start,
            layer_idx=layer_idx,
            device=device,
        )
        current_rank_score = rank_normalize_scores(visual_scores)
        global_prune_step = int(context.get("score_prune_step", 0))
        global_use_ema = global_prune_step > 0
        alive_ema = self._alive_ema(context, current_patch_indices, device=current_rank_score.device)
        if global_use_ema:
            mixed_score = current_rank_score * current_weight + alive_ema * (1.0 - current_weight)
        else:
            mixed_score = current_rank_score

        self._update_ema(
            context=context,
            current_patch_indices=current_patch_indices,
            current_rank_score=current_rank_score,
            ema_decay=ema_decay,
        )

        return {
            "context": context,
            "current_patch_indices": current_patch_indices,
            "visual_scores": visual_scores,
            "current_rank_score": current_rank_score,
            "global_saliency_ema": alive_ema,
            "mixed_score": mixed_score.to(dtype=torch.float32),
            "global_prune_step": global_prune_step,
            "global_use_ema": global_use_ema,
            "global_current_weight": current_weight,
            "global_ema_decay": ema_decay,
        }

    def _set_pending_decision(self, context: Dict[str, Any], current_patch_indices: torch.Tensor, layer_idx: int) -> None:
        context["score_pending_decision"] = {
            "layer_idx": int(layer_idx),
            "current_patch_indices": current_patch_indices.detach(),
        }

    def update_after_prune(self, keep_indices: torch.Tensor, layer_idx: int) -> None:
        context = self._require_sample_context()
        pending = context.get("score_pending_decision")
        if pending is not None:
            if int(pending["layer_idx"]) != int(layer_idx):
                raise ValueError(f"Pending score-memory decision is for layer {pending['layer_idx']}, got layer {layer_idx}")
            current_patch_indices = pending["current_patch_indices"]
        else:
            current_patch_indices = context["current_patch_indices"]

        keep_indices = keep_indices.to(device=current_patch_indices.device, dtype=torch.long)
        context["current_patch_indices"] = current_patch_indices.index_select(0, keep_indices).detach()
        context["score_prune_step"] = int(context.get("score_prune_step", 0)) + 1
        context["score_pending_decision"] = None

    def _patch_index_info(
        self,
        current_patch_indices: torch.Tensor,
        keep_indices: torch.Tensor,
        pruned_indices: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        keep_patch_indices = current_patch_indices.index_select(
            0, keep_indices.to(device=current_patch_indices.device)
        )
        pruned_patch_indices = current_patch_indices.index_select(
            0, pruned_indices.to(device=current_patch_indices.device)
        )
        return {
            "keep_patch_indices": keep_patch_indices,
            "pruned_patch_indices": pruned_patch_indices,
        }
