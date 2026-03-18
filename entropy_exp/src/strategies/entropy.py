"""
Entropy-based pruning strategy.

Uses per-head entropy of the attention distribution over visual tokens to weight
heads when computing importance. Heads with lower entropy (more concentrated
attention) are considered more decisive and receive higher weight.

Optionally supports dynamic prune ratio based on the overall concentration
of importance scores.
"""

from typing import Optional

import torch

from .base import PruneStrategy


class EntropyStrategy(PruneStrategy):
    """
    Entropy-weighted attention pruning.

    Unlike the plain AttnScore strategy which treats all heads equally,
    this strategy weights each head by the inverse of its entropy over
    the visual token dimension. Lower entropy → more confident head
    → higher weight in the importance aggregation.
    """

    def compute_importance(
        self,
        attn_weights: Optional[torch.Tensor],
        v_token_start: int,
        v_token_num: int,
        text_token_start: int,
        layer_idx: int,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if attn_weights is None:
            raise ValueError("EntropyStrategy requires attention weights")
        v_end = v_token_start + v_token_num
        # [H, L_t, L_v]
        tv = attn_weights[0, :, text_token_start:, v_token_start:v_end]

        # Normalise each (head, text_pos) row to a distribution over visual tokens
        tv_norm = tv / tv.sum(dim=-1, keepdim=True).clamp(min=1e-12)

        # Per-head entropy averaged over text positions: [H]
        log_p = torch.log2(tv_norm.clamp(min=1e-12))
        entropy_per_pos = -(tv_norm * log_p).sum(dim=-1)  # [H, L_t]
        head_entropy = entropy_per_pos.mean(dim=1)         # [H]

        # Lower entropy → higher weight  (softmax of negative entropy)
        head_weights = torch.softmax(-head_entropy, dim=0)  # [H]

        # Weighted average over heads → [L_t, L_v]
        weighted_tv = (tv * head_weights[:, None, None]).sum(dim=0)

        # Mean over text positions → [L_v]
        scores = weighted_tv.mean(dim=0)
        return scores

    # ------------------------------------------------------------------
    # Dynamic prune ratio (optional)
    # ------------------------------------------------------------------
    def get_prune_ratio(self, layer_idx: int, importance_scores: torch.Tensor) -> float:
        # Per-layer base ratio from the ratio map, or scalar fallback
        ratio_map = self.config.get("prune_ratio_map")
        if ratio_map and layer_idx in ratio_map:
            base_ratio = ratio_map[layer_idx]
        else:
            base_ratio = self.config.get("prune_ratio", 0.5)
        if not self.config.get("dynamic_ratio", False):
            return base_ratio

        # Use normalised entropy of importance distribution to modulate ratio.
        # Concentrated importance → can prune more aggressively.
        scores = importance_scores
        probs = scores / scores.sum().clamp(min=1e-12)
        entropy = -(probs * torch.log2(probs.clamp(min=1e-12))).sum()
        max_entropy = torch.log2(torch.tensor(float(len(scores)), device=scores.device))
        norm_entropy = (entropy / max_entropy).item()  # 0 = concentrated, 1 = uniform

        scale = self.config.get("dynamic_scale", 0.5)
        ratio = base_ratio * (1.0 + (1.0 - norm_entropy) * scale)
        ratio = min(ratio, self.config.get("max_prune_ratio", 0.9))
        return ratio
