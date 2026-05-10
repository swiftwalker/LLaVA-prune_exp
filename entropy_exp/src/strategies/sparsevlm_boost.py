"""SparseVLM score-boosting pruning.

This variant replaces hard spatial compensation quotas with deterministic
additive score boosts for under-scored image strata, then uses a single top-k
selection over the adjusted scores.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from .sparsevlm_adaptive_stratified import build_stratum_index, compute_target_keep_count
from .sparsevlm_score_memory import (
    SparseVLMScoreMemoryStrategy,
    compute_entropy_stats,
    deterministic_topk_indices,
)


class SparseVLMBoostStrategy(SparseVLMScoreMemoryStrategy):
    """SparseVLM saliency top-k with entropy-adaptive spatial score boosting."""

    def _get_boost_params(self) -> Tuple[float, float, int, int]:
        boost_weight_min = float(self.config.get("boost_weight_min", 0.0))
        boost_weight_max = float(self.config.get("boost_weight_max", 0.15))
        grid_size = int(self.config.get("grid_size", 6))
        patch_per_row = int(self.config.get("patch_per_row", 24))
        initial_v_token_num = int(self._require_sample_context()["initial_v_token_num"])

        if not 0.0 <= boost_weight_min <= boost_weight_max:
            raise ValueError(
                "boost_weight bounds must satisfy 0 <= boost_weight_min <= boost_weight_max, "
                f"got {boost_weight_min}, {boost_weight_max}"
            )
        if grid_size <= 0:
            raise ValueError(f"grid_size must be > 0, got {grid_size}")
        if patch_per_row <= 0:
            raise ValueError(f"patch_per_row must be > 0, got {patch_per_row}")
        if patch_per_row * patch_per_row != initial_v_token_num:
            raise ValueError(
                "patch_per_row * patch_per_row must equal the initial visual token count, "
                f"got {patch_per_row}^2 vs {initial_v_token_num}"
            )
        if patch_per_row % grid_size != 0:
            raise ValueError(
                f"grid_size must divide patch_per_row, got grid_size={grid_size}, patch_per_row={patch_per_row}"
            )
        return boost_weight_min, boost_weight_max, grid_size, patch_per_row

    @staticmethod
    def _compute_stratum_deficit(
        mixed_score: torch.Tensor,
        stratum_index: Dict[int, list[int]],
        grid_size: int,
    ) -> torch.Tensor:
        global_mean = mixed_score.mean()
        deficits = torch.zeros(grid_size * grid_size, device=mixed_score.device, dtype=torch.float32)
        for sid in range(grid_size * grid_size):
            token_indices = stratum_index.get(sid, [])
            if not token_indices:
                continue
            token_tensor = torch.tensor(token_indices, device=mixed_score.device, dtype=torch.long)
            stratum_mean = mixed_score.index_select(0, token_tensor).mean()
            deficits[sid] = torch.clamp(global_mean - stratum_mean, min=0.0)
        max_deficit = deficits.max()
        if float(max_deficit.item()) > 0.0:
            deficits = deficits / max_deficit
        return deficits

    def compute_keep_mask(
        self,
        attn_weights: Optional[torch.Tensor],
        v_token_start: int,
        v_token_num: int,
        text_token_start: int,
        layer_idx: int,
        device: Optional[torch.device] = None,
        current_visual_embeds: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        del current_visual_embeds
        memory = self._score_memory_inputs(
            attn_weights=attn_weights,
            v_token_start=v_token_start,
            v_token_num=v_token_num,
            text_token_start=text_token_start,
            layer_idx=layer_idx,
            device=device,
        )
        context = memory["context"]
        current_patch_indices = memory["current_patch_indices"]
        visual_scores = memory["visual_scores"]
        mixed_score = memory["mixed_score"]

        boost_min, boost_max, grid_size, patch_per_row = self._get_boost_params()
        entropy_raw, entropy_norm = compute_entropy_stats(visual_scores)
        boost_weight = boost_min + entropy_norm * (boost_max - boost_min)

        prune_ratio = self.get_prune_ratio(layer_idx, visual_scores)
        min_visual_tokens_after_prune = int(self.config.get("min_visual_tokens_after_prune", 16))
        target_keep = compute_target_keep_count(
            num_visual=v_token_num,
            prune_ratio=prune_ratio,
            min_visual_tokens_after_prune=min_visual_tokens_after_prune,
        )

        stratum_index = build_stratum_index(
            current_patch_indices=current_patch_indices,
            grid_size=grid_size,
            patch_per_row=patch_per_row,
        )
        stratum_deficit = self._compute_stratum_deficit(
            mixed_score=mixed_score,
            stratum_index=stratum_index,
            grid_size=grid_size,
        )
        patch_strata = torch.zeros(v_token_num, device=mixed_score.device, dtype=torch.long)
        for sid, token_indices in stratum_index.items():
            if token_indices:
                patch_strata[torch.tensor(token_indices, device=mixed_score.device, dtype=torch.long)] = int(sid)
        token_boost = mixed_score.new_tensor(float(boost_weight)) * stratum_deficit.index_select(0, patch_strata)
        adjusted_scores = mixed_score + token_boost

        keep_indices = deterministic_topk_indices(adjusted_scores, target_keep)
        pruned_mask = torch.ones(v_token_num, device=visual_scores.device, dtype=torch.bool)
        pruned_mask[keep_indices] = False
        pruned_indices = torch.nonzero(pruned_mask, as_tuple=False).flatten()
        patch_info = self._patch_index_info(current_patch_indices, keep_indices, pruned_indices)

        self._set_pending_decision(context, current_patch_indices, layer_idx)

        info = {
            "prune_ratio": prune_ratio,
            "num_visual_before": v_token_num,
            "num_visual_after": int(keep_indices.numel()),
            "num_pruned": int(pruned_indices.numel()),
            "importance_scores": visual_scores.detach().cpu().numpy(),
            "visual_scores": visual_scores.detach().cpu().numpy(),
            "current_rank_score": memory["current_rank_score"].detach().cpu().numpy(),
            "mixed_score": mixed_score.detach().cpu().numpy(),
            "boost_weight": float(boost_weight),
            "boost_weight_min": boost_min,
            "boost_weight_max": boost_max,
            "token_boost": token_boost.detach().cpu().numpy(),
            "adjusted_scores": adjusted_scores.detach().cpu().numpy(),
            "stratum_deficit": stratum_deficit.detach().cpu().numpy(),
            "keep_indices": keep_indices.detach().cpu().numpy(),
            "pruned_indices": pruned_indices.detach().cpu().numpy(),
            "rater_indices": context["rater_indices"].detach().cpu().numpy(),
            "text_relevance_scores": context["text_relevance_scores"].detach().cpu().numpy(),
            "current_patch_indices": current_patch_indices.detach().cpu().numpy(),
            "keep_patch_indices": patch_info["keep_patch_indices"].detach().cpu().numpy(),
            "pruned_patch_indices": patch_info["pruned_patch_indices"].detach().cpu().numpy(),
            "high_keep_indices": keep_indices.detach().cpu().numpy(),
            "low_keep_indices": torch.empty(0, device=keep_indices.device, dtype=torch.long).cpu().numpy(),
            "high_keep_patch_indices": patch_info["keep_patch_indices"].detach().cpu().numpy(),
            "low_keep_patch_indices": torch.empty(0, device=current_patch_indices.device, dtype=torch.long).cpu().numpy(),
            "target_keep": target_keep,
            "strategy_keep_high": int(keep_indices.numel()),
            "strategy_keep_low": 0,
            "grid_size": grid_size,
            "patch_per_row": patch_per_row,
            "saliency_entropy": entropy_raw,
            "saliency_entropy_norm": entropy_norm,
            "global_prune_step": memory["global_prune_step"],
            "global_current_weight": memory["global_current_weight"],
            "global_ema_decay": memory["global_ema_decay"],
            "global_use_ema": memory["global_use_ema"],
            "global_saliency_ema": memory["global_saliency_ema"].detach().cpu().numpy(),
        }
        return keep_indices, info
