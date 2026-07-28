"""
SparseVLM-style text-rater visual token pruning.

This strategy first selects a small set of image-relevant text raters from the
prepared multimodal input, then uses current-layer text->vision attention from
those raters to score visual tokens at each prune layer.
"""

import math
from typing import Any, Dict, Optional, Tuple

import torch

from .base import PruneStrategy


def select_text_raters(
    H_v: torch.Tensor,
    H_q: torch.Tensor,
    special_token_mask: Optional[torch.Tensor] = None,
    fallback_topk: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Select text raters using mean-thresholded visual-text relevance.

    Args:
        H_v: [Lv, D] visual token embeddings
        H_q: [Lt, D] text token embeddings
        special_token_mask: optional [Lt] bool mask; True means exclude
        fallback_topk: fallback top-k when thresholding keeps zero raters

    Returns:
        rater_indices: [R] text-block-relative indices into H_q
        text_relevance_scores: [Lt] mean relevance score per text token
    """
    if H_v.ndim != 2 or H_q.ndim != 2:
        raise ValueError(f"Expected H_v/H_q to be 2D, got {tuple(H_v.shape)} and {tuple(H_q.shape)}")
    if H_v.shape[1] != H_q.shape[1]:
        raise ValueError(f"H_v and H_q hidden sizes must match, got {H_v.shape[1]} and {H_q.shape[1]}")
    if H_q.shape[0] == 0:
        raise ValueError("Cannot select text raters from an empty text block")

    logits = H_v.float() @ H_q.float().transpose(0, 1)  # [Lv, Lt]
    relevance = torch.softmax(logits, dim=-1)
    text_relevance_scores = relevance.mean(dim=0)  # [Lt]

    candidate_mask = torch.ones(H_q.shape[0], dtype=torch.bool, device=H_q.device)
    if special_token_mask is not None:
        if special_token_mask.shape != (H_q.shape[0],):
            raise ValueError(
                f"special_token_mask must have shape {(H_q.shape[0],)}, got {tuple(special_token_mask.shape)}"
            )
        candidate_mask = ~special_token_mask.to(device=H_q.device, dtype=torch.bool)

    candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).flatten()
    if candidate_indices.numel() == 0:
        all_indices = torch.arange(H_q.shape[0], device=H_q.device, dtype=torch.long)
        k = max(1, min(int(fallback_topk), all_indices.numel()))
        topk_relative = torch.topk(text_relevance_scores, k=k, largest=True).indices
        return all_indices.index_select(0, topk_relative).sort().values, text_relevance_scores

    candidate_scores = text_relevance_scores.index_select(0, candidate_indices)
    threshold = candidate_scores.mean()
    selected_mask = candidate_scores >= threshold
    rater_indices = candidate_indices[selected_mask]

    if rater_indices.numel() == 0:
        k = max(1, min(int(fallback_topk), candidate_indices.numel()))
        topk_relative = torch.topk(candidate_scores, k=k, largest=True).indices
        rater_indices = candidate_indices.index_select(0, topk_relative)

    rater_indices = rater_indices.sort().values
    return rater_indices, text_relevance_scores


def compute_rater_visual_scores_from_attention(
    attn_weights: torch.Tensor,
    rater_indices: torch.Tensor,
    text_positions: torch.Tensor,
    visual_positions: torch.Tensor,
) -> torch.Tensor:
    """
    Compute per-rater visual-token scores from current-layer attention.

    Args:
        attn_weights: [B, H, L, L] or [H, L, L]
        rater_indices: [R] text-block-relative indices into text_positions
        text_positions: [Lt_current] absolute decoder positions for current text tokens
        visual_positions: [Lv_current] absolute decoder positions for current visual tokens

    Returns:
        rater_visual_scores: [R, Lv_current]
    """
    if attn_weights.ndim == 4:
        if attn_weights.shape[0] != 1:
            raise ValueError(f"Only batch size 1 is supported, got attention shape {tuple(attn_weights.shape)}")
        attn = attn_weights[0]
    elif attn_weights.ndim == 3:
        attn = attn_weights
    else:
        raise ValueError(f"Unsupported attention shape {tuple(attn_weights.shape)}")

    if attn.ndim != 3:
        raise ValueError(f"Expected [H, L, L] attention after batch squeeze, got {tuple(attn.shape)}")

    device = attn.device
    text_positions = torch.as_tensor(text_positions, device=device, dtype=torch.long)
    visual_positions = torch.as_tensor(visual_positions, device=device, dtype=torch.long)
    rater_indices = torch.as_tensor(rater_indices, device=device, dtype=torch.long)

    if text_positions.ndim != 1 or visual_positions.ndim != 1 or rater_indices.ndim != 1:
        raise ValueError("rater_indices, text_positions, and visual_positions must be 1D")
    if text_positions.numel() == 0 or visual_positions.numel() == 0:
        raise ValueError("Text and visual positions must both be non-empty")
    if rater_indices.numel() == 0:
        raise ValueError("rater_indices must be non-empty")
    if rater_indices.min().item() < 0 or rater_indices.max().item() >= text_positions.numel():
        raise ValueError(
            f"rater_indices out of range for current text positions: {rater_indices.tolist()} vs {text_positions.numel()}"
        )

    seq_len = attn.shape[-1]
    if text_positions.min().item() < 0 or text_positions.max().item() >= seq_len:
        raise ValueError(f"text_positions out of range for seq_len={seq_len}: {text_positions.tolist()}")
    if visual_positions.min().item() < 0 or visual_positions.max().item() >= seq_len:
        raise ValueError(f"visual_positions out of range for seq_len={seq_len}: {visual_positions.tolist()}")

    attn_mean = attn.mean(dim=0)  # [L, L]
    selected_text_positions = text_positions.index_select(0, rater_indices)
    tv = attn_mean.index_select(0, selected_text_positions).index_select(1, visual_positions)  # [R, Lv]
    return tv


def compute_visual_scores_from_attention(
    attn_weights: torch.Tensor,
    rater_indices: torch.Tensor,
    text_positions: torch.Tensor,
    visual_positions: torch.Tensor,
) -> torch.Tensor:
    """Compute the legacy mean-rater visual score without changing semantics."""
    return compute_rater_visual_scores_from_attention(
        attn_weights=attn_weights,
        rater_indices=rater_indices,
        text_positions=text_positions,
        visual_positions=visual_positions,
    ).mean(dim=0)


def prune_visual_tokens(
    current_visual_embeds: torch.Tensor,
    visual_scores: torch.Tensor,
    prune_ratio: float,
    min_visual_tokens_after_prune: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prune the lowest-score visual tokens subject to a minimum kept-token floor.

    Args:
        current_visual_embeds: [Nv, D]
        visual_scores: [Nv]
        prune_ratio: removal ratio in [0, 1]
        min_visual_tokens_after_prune: minimum visual tokens that must remain

    Returns:
        pruned_visual_embeds: [Nv_after, D]
        keep_indices: [Nv_after] relative indices into current_visual_embeds
        pruned_indices: [Nv_pruned] relative indices removed from current_visual_embeds
    """
    if current_visual_embeds.ndim != 2:
        raise ValueError(f"current_visual_embeds must be 2D, got {tuple(current_visual_embeds.shape)}")
    if visual_scores.ndim != 1:
        raise ValueError(f"visual_scores must be 1D, got {tuple(visual_scores.shape)}")
    if current_visual_embeds.shape[0] != visual_scores.shape[0]:
        raise ValueError(
            f"visual_scores length {visual_scores.shape[0]} does not match visual token count {current_visual_embeds.shape[0]}"
        )
    if not 0.0 <= float(prune_ratio) <= 1.0:
        raise ValueError(f"prune_ratio must be in [0, 1], got {prune_ratio}")
    if min_visual_tokens_after_prune < 0:
        raise ValueError(f"min_visual_tokens_after_prune must be >= 0, got {min_visual_tokens_after_prune}")

    num_visual = current_visual_embeds.shape[0]
    raw_num_pruned = math.floor(num_visual * float(prune_ratio))
    max_prunable = max(num_visual - int(min_visual_tokens_after_prune), 0)
    num_pruned = min(max(raw_num_pruned, 0), max_prunable)

    if num_pruned <= 0:
        keep_indices = torch.arange(num_visual, device=current_visual_embeds.device, dtype=torch.long)
        pruned_indices = torch.empty(0, device=current_visual_embeds.device, dtype=torch.long)
        return current_visual_embeds, keep_indices, pruned_indices

    sorted_indices = torch.argsort(visual_scores, dim=0, descending=False)
    pruned_indices = sorted_indices[:num_pruned].sort().values
    keep_indices = sorted_indices[num_pruned:].sort().values
    pruned_visual_embeds = current_visual_embeds.index_select(0, keep_indices)
    return pruned_visual_embeds, keep_indices, pruned_indices


class SparseVLMStrategy(PruneStrategy):
    """Text-rater-based visual token pruning using current-layer decoder attention."""

    def prepare_sample(
        self,
        inputs_embeds: torch.Tensor,
        v_token_start: int,
        v_token_num: int,
        text_token_start: int,
        text_token_ids: Optional[torch.Tensor] = None,
        text_special_token_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        super().prepare_sample(
            inputs_embeds=inputs_embeds,
            v_token_start=v_token_start,
            v_token_num=v_token_num,
            text_token_start=text_token_start,
            text_token_ids=text_token_ids,
            text_special_token_mask=text_special_token_mask,
        )
        H_v = inputs_embeds[0, v_token_start:text_token_start]  # [Lv, D]
        H_q = inputs_embeds[0, text_token_start:]  # [Lt, D]
        if H_q.shape[0] == 0:
            raise ValueError("SparseVLM requires at least one text token after the visual block")

        use_special_mask = self.config.get("exclude_special_tokens", True)
        special_mask = text_special_token_mask if use_special_mask else None
        rater_indices, text_relevance_scores = select_text_raters(
            H_v=H_v,
            H_q=H_q,
            special_token_mask=special_mask,
            fallback_topk=self.config.get("fallback_topk", 4),
        )
        self.sample_context = {
            "rater_indices": rater_indices.detach(),
            "text_relevance_scores": text_relevance_scores.detach(),
            "text_length": int(H_q.shape[0]),
            "text_token_ids": (
                None if text_token_ids is None else text_token_ids.detach()
            ),
            "text_special_token_mask": (
                None
                if text_special_token_mask is None
                else text_special_token_mask.detach().to(dtype=torch.bool)
            ),
        }
        return {
            "rater_indices": rater_indices.detach().cpu().numpy(),
            "text_relevance_scores": text_relevance_scores.detach().cpu().numpy(),
        }

    def _require_sample_context(self) -> Dict[str, Any]:
        if self.sample_context is None:
            raise ValueError("SparseVLMStrategy sample context is missing. Call prepare_sample() first.")
        return self.sample_context

    def compute_rater_importance(
        self,
        attn_weights: Optional[torch.Tensor],
        v_token_start: int,
        v_token_num: int,
        text_token_start: int,
        layer_idx: int,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if attn_weights is None:
            raise ValueError("SparseVLMStrategy requires attention weights")
        context = self._require_sample_context()
        seq_len = attn_weights.shape[-1]
        text_length = context["text_length"]

        text_positions = torch.arange(text_token_start, seq_len, device=attn_weights.device, dtype=torch.long)
        if text_positions.numel() != text_length:
            raise ValueError(
                f"SparseVLM text position mismatch at layer {layer_idx}: expected {text_length}, got {text_positions.numel()}"
            )
        visual_positions = torch.arange(
            v_token_start, v_token_start + v_token_num, device=attn_weights.device, dtype=torch.long
        )
        return compute_rater_visual_scores_from_attention(
            attn_weights=attn_weights,
            rater_indices=context["rater_indices"],
            text_positions=text_positions,
            visual_positions=visual_positions,
        )

    def compute_importance(
        self,
        attn_weights: Optional[torch.Tensor],
        v_token_start: int,
        v_token_num: int,
        text_token_start: int,
        layer_idx: int,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        return self.compute_rater_importance(
            attn_weights=attn_weights,
            v_token_start=v_token_start,
            v_token_num=v_token_num,
            text_token_start=text_token_start,
            layer_idx=layer_idx,
            device=device,
        ).mean(dim=0)

    def compute_keep_mask(
        self,
        attn_weights: Optional[torch.Tensor],
        v_token_start: int,
        v_token_num: int,
        text_token_start: int,
        layer_idx: int,
        device: Optional[torch.device] = None,
        current_visual_embeds: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if attn_weights is None:
            raise ValueError("SparseVLMStrategy requires attention weights")
        if current_visual_embeds is None:
            raise ValueError("SparseVLMStrategy requires current_visual_embeds for pruning")

        context = self._require_sample_context()
        visual_scores = self.compute_importance(
            attn_weights=attn_weights,
            v_token_start=v_token_start,
            v_token_num=v_token_num,
            text_token_start=text_token_start,
            layer_idx=layer_idx,
            device=device,
        )
        prune_ratio = self.get_prune_ratio(layer_idx, visual_scores)
        _, keep_indices, pruned_indices = prune_visual_tokens(
            current_visual_embeds=current_visual_embeds,
            visual_scores=visual_scores,
            prune_ratio=prune_ratio,
            min_visual_tokens_after_prune=int(self.config.get("min_visual_tokens_after_prune", 16)),
        )

        info = {
            "prune_ratio": prune_ratio,
            "num_visual_before": v_token_num,
            "num_visual_after": int(keep_indices.numel()),
            "num_pruned": int(pruned_indices.numel()),
            "importance_scores": visual_scores.detach().cpu().numpy(),
            "visual_scores": visual_scores.detach().cpu().numpy(),
            "keep_indices": keep_indices.detach().cpu().numpy(),
            "pruned_indices": pruned_indices.detach().cpu().numpy(),
            "rater_indices": context["rater_indices"].detach().cpu().numpy(),
            "text_relevance_scores": context["text_relevance_scores"].detach().cpu().numpy(),
        }
        return keep_indices, info
