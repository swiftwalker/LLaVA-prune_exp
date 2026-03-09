"""
Attention-score-based pruning strategy (SparseVLM style).

Importance = mean over attention heads → mean over text query positions
→ per visual token importance score.
"""

import torch
from .base import PruneStrategy


class AttnScoreStrategy(PruneStrategy):
    """
    SparseVLM-style attention score pruning.

    For each visual token, importance = average attention it receives from all
    text query positions across all attention heads at the prune layer.
    """

    def compute_importance(
        self,
        attn_weights: torch.Tensor,
        v_token_start: int,
        v_token_num: int,
        text_token_start: int,
        layer_idx: int,
    ) -> torch.Tensor:
        # attn_weights: [B, H, L, L]  (post-softmax)
        v_end = v_token_start + v_token_num
        # text queries (rows) attending to visual keys (cols): [H, L_t, L_v]
        tv = attn_weights[0, :, text_token_start:, v_token_start:v_end]
        # Mean over heads → mean over text positions → [L_v]
        scores = tv.mean(dim=0).mean(dim=0)
        return scores
