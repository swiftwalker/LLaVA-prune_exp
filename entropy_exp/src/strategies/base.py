"""
Abstract base class for visual token pruning strategies.

A strategy defines:
  1. How to compute per-visual-token importance scores from attention weights.
  2. How to determine the prune ratio (fixed or dynamic).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple

import torch
import numpy as np


class PruneStrategy(ABC):
    """Base class for visual token pruning strategies."""

    def __init__(self, config: dict):
        self.config = config
        self.sample_context: Optional[Dict[str, Any]] = None

    def prepare_sample(
        self,
        inputs_embeds: torch.Tensor,
        v_token_start: int,
        v_token_num: int,
        text_token_start: int,
        text_token_ids: Optional[torch.Tensor] = None,
        text_special_token_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Prepare any per-sample state needed before layer-by-layer prefill."""
        self.sample_context = None
        return {}

    def clear_sample(self):
        """Clear any sample-local state cached on the strategy."""
        self.sample_context = None

    def requires_attention(self) -> bool:
        """Whether this strategy needs attention weights to decide pruning."""
        return True

    def prune_stage(self) -> str:
        """Whether pruning happens before or after the target layer forward."""
        return "post"

    @abstractmethod
    def compute_importance(
        self,
        attn_weights: Optional[torch.Tensor],
        v_token_start: int,
        v_token_num: int,
        text_token_start: int,
        layer_idx: int,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Compute per-visual-token importance scores.

        Args:
            attn_weights: [B, H, L, L] post-softmax attention weights at the prune layer,
                or None for strategies that do not require them
            v_token_start: start position of visual tokens in the sequence
            v_token_num: number of visual tokens (e.g. 576)
            text_token_start: start position of text tokens (= v_token_start + v_token_num)
            layer_idx: current layer index
            device: device for creating new tensors when attention is unavailable

        Returns:
            torch.Tensor of shape [v_token_num] — importance score per visual token
        """

    def get_prune_ratio(self, layer_idx: int, importance_scores: torch.Tensor) -> float:
        """
        Get the fraction of visual tokens to REMOVE.

        Looks up the per-layer ratio map first (populated by the Pruner),
        then falls back to a scalar default.
        Override for dynamic ratio computation.
        """
        ratio_map = self.config.get("prune_ratio_map")
        if ratio_map and layer_idx in ratio_map:
            return ratio_map[layer_idx]
        return self.config.get("prune_ratio", 0.5)

    def compute_keep_mask_from_importance(
        self,
        importance_scores: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Select kept visual tokens from externally computed importance scores."""
        v_token_num = int(importance_scores.shape[0])
        prune_ratio = self.get_prune_ratio(layer_idx, importance_scores)

        num_prune = int(v_token_num * prune_ratio)
        num_keep = v_token_num - num_prune

        # Keep the top-num_keep tokens by importance (preserve spatial order)
        _, sorted_indices = importance_scores.sort(descending=True)
        keep_indices = sorted_indices[:num_keep].sort().values

        info = {
            "prune_ratio": prune_ratio,
            "num_visual_before": v_token_num,
            "num_visual_after": num_keep,
            "num_pruned": num_prune,
            "importance_scores": importance_scores.detach().cpu().numpy(),
            "keep_indices": keep_indices.detach().cpu().numpy(),
        }
        return keep_indices, info

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
        """
        Compute which visual tokens to keep.

        Returns:
            keep_indices: 1D tensor of indices (relative to visual token block) to keep,
                          sorted in ascending order to preserve spatial layout.
            info: dict with pruning statistics for logging/analysis.
        """
        if self.requires_attention() and attn_weights is None:
            raise ValueError(
                f"{self.__class__.__name__} requires attention weights, but none were provided"
            )

        importance = self.compute_importance(
            attn_weights, v_token_start, v_token_num, text_token_start, layer_idx, device=device
        )
        return self.compute_keep_mask_from_importance(importance, layer_idx)
