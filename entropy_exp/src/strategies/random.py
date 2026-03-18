"""
Random visual-token pruning strategy.

Importance is assigned by per-token random scores so the existing top-k pruning
path, logging, and capture outputs remain schema-compatible with other
strategies.
"""

from typing import Optional

import torch

from .base import PruneStrategy


class RandomStrategy(PruneStrategy):
    """Uniform random pruning controlled by the configured prune ratio."""

    def requires_attention(self) -> bool:
        return False

    def compute_importance(
        self,
        attn_weights: Optional[torch.Tensor],
        v_token_start: int,
        v_token_num: int,
        text_token_start: int,
        layer_idx: int,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if device is None:
            if attn_weights is not None:
                device = attn_weights.device
            else:
                device = torch.device("cpu")
        return torch.rand(v_token_num, device=device)
