"""
Entropy-adaptive SparseVLM stratified pruning.

This strategy preserves SparseVLM adaptive stratified compensation while
replacing the fixed high-score keep ratio with an instance-level ratio derived
from the spatial entropy of visual saliency scores.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch

from .sparsevlm_adaptive_stratified import (
    SparseVLMAdaptiveStratifiedStrategy,
    allocate_adaptive_stratified_quotas,
    build_stratum_index,
    compute_target_keep_count,
    randomly_select_token_indices,
    select_farthest_token_indices,
    validate_adaptive_stratified_config,
)


class SparseVLMEntropyAlphaStrategy(SparseVLMAdaptiveStratifiedStrategy):
    """Adaptive stratified SparseVLM with entropy-driven alpha."""

    _VALIDATION_HIGH_RATIO_PLACEHOLDER = 0.5

    def _get_alpha_bounds(self) -> Tuple[float, float]:
        alpha_min = float(self.config.get("alpha_min", 0.4))
        alpha_max = float(self.config.get("alpha_max", 0.95))
        if not 0.0 <= alpha_min <= alpha_max <= 1.0:
            raise ValueError(
                f"alpha bounds must satisfy 0 <= alpha_min <= alpha_max <= 1, got {alpha_min}, {alpha_max}"
            )
        return alpha_min, alpha_max

    def _compute_adaptive_alpha(self, visual_scores: torch.Tensor) -> Tuple[float, float, float]:
        if visual_scores.ndim != 1:
            raise ValueError(f"visual_scores must be 1D, got {tuple(visual_scores.shape)}")

        alpha_min, alpha_max = self._get_alpha_bounds()
        num_tokens = int(visual_scores.numel())
        if num_tokens <= 1:
            return alpha_max, 0.0, 0.0

        scores = visual_scores.to(dtype=torch.float64)
        score_sum = scores.sum()
        if float(score_sum.item()) <= 0.0:
            return alpha_max, 0.0, 0.0

        probs = scores / score_sum
        positive_mask = probs > 0
        entropy_raw = -(probs[positive_mask] * torch.log(probs[positive_mask])).sum()
        entropy_denom = torch.log(scores.new_tensor(float(num_tokens)))
        entropy_norm = torch.clamp(entropy_raw / entropy_denom, min=0.0, max=1.0)
        alpha = scores.new_tensor(alpha_min) + (1.0 - entropy_norm) * (alpha_max - alpha_min)
        alpha = torch.clamp(alpha, min=alpha_min, max=alpha_max)
        return float(alpha.item()), float(entropy_raw.item()), float(entropy_norm.item())

    def prepare_sample(
        self,
        inputs_embeds: torch.Tensor,
        v_token_start: int,
        v_token_num: int,
        text_token_start: int,
        text_token_ids: Optional[torch.Tensor] = None,
        text_special_token_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, object]:
        if "high_ratio" in self.config:
            return super().prepare_sample(
                inputs_embeds=inputs_embeds,
                v_token_start=v_token_start,
                v_token_num=v_token_num,
                text_token_start=text_token_start,
                text_token_ids=text_token_ids,
                text_special_token_mask=text_special_token_mask,
            )

        original_config = self.config
        self.config = {**self.config, "high_ratio": self._VALIDATION_HIGH_RATIO_PLACEHOLDER}
        try:
            return super().prepare_sample(
                inputs_embeds=inputs_embeds,
                v_token_start=v_token_start,
                v_token_num=v_token_num,
                text_token_start=text_token_start,
                text_token_ids=text_token_ids,
                text_special_token_mask=text_special_token_mask,
            )
        finally:
            self.config = original_config

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
        if attn_weights is None:
            raise ValueError("SparseVLMEntropyAlphaStrategy requires attention weights")

        context = self._require_sample_context()
        current_patch_indices = context["current_patch_indices"]
        if int(current_patch_indices.numel()) != int(v_token_num):
            raise ValueError(
                "current_patch_indices length must match the current visual token count, "
                f"got {current_patch_indices.numel()} vs {v_token_num}"
            )

        grid_size = int(self.config.get("grid_size", 6))
        patch_per_row = int(self.config.get("patch_per_row", 24))
        intra_stratum_mode = str(self.config.get("intra_stratum_mode", "random"))
        alpha_min, alpha_max = self._get_alpha_bounds()

        visual_scores = self.compute_importance(
            attn_weights=attn_weights,
            v_token_start=v_token_start,
            v_token_num=v_token_num,
            text_token_start=text_token_start,
            layer_idx=layer_idx,
            device=device,
        )
        adaptive_alpha, entropy_raw, entropy_norm = self._compute_adaptive_alpha(visual_scores)
        validate_adaptive_stratified_config(
            initial_v_token_num=int(context["initial_v_token_num"]),
            patch_per_row=patch_per_row,
            grid_size=grid_size,
            high_ratio=adaptive_alpha,
            intra_stratum_mode=intra_stratum_mode,
        )

        prune_ratio = self.get_prune_ratio(layer_idx, visual_scores)
        min_visual_tokens_after_prune = int(self.config.get("min_visual_tokens_after_prune", 16))
        target_keep = compute_target_keep_count(
            num_visual=v_token_num,
            prune_ratio=prune_ratio,
            min_visual_tokens_after_prune=min_visual_tokens_after_prune,
        )
        n_high = math.floor(target_keep * adaptive_alpha)
        n_low = target_keep - n_high

        sorted_desc = torch.argsort(visual_scores, descending=True)
        high_keep_indices = sorted_desc[:n_high].sort().values
        high_keep_set = set(high_keep_indices.detach().cpu().tolist())

        stratum_index = build_stratum_index(
            current_patch_indices=current_patch_indices,
            grid_size=grid_size,
            patch_per_row=patch_per_row,
        )
        selected_counts = {}
        candidate_counts = {}
        for sid in range(grid_size * grid_size):
            tokens_in_stratum = stratum_index.get(sid, [])
            selected_counts[sid] = sum(1 for token_idx in tokens_in_stratum if token_idx in high_keep_set)
            candidate_counts[sid] = sum(1 for token_idx in tokens_in_stratum if token_idx not in high_keep_set)

        quotas, deficits = allocate_adaptive_stratified_quotas(
            selected_counts=selected_counts,
            candidate_counts=candidate_counts,
            target_keep=target_keep,
            n_low=n_low,
            grid_size=grid_size,
        )

        low_keep_indices: List[int] = []
        high_keep_patch_indices = current_patch_indices.index_select(
            0, high_keep_indices.to(device=current_patch_indices.device)
        )
        selected_patch_indices = high_keep_patch_indices.detach().cpu().tolist()

        for sid in range(grid_size * grid_size):
            quota = quotas.get(sid, 0)
            if quota <= 0:
                continue

            candidates = [token_idx for token_idx in stratum_index.get(sid, []) if token_idx not in high_keep_set]
            if intra_stratum_mode == "random":
                chosen = randomly_select_token_indices(candidates, quota)
            else:
                chosen = select_farthest_token_indices(
                    candidate_token_indices=candidates,
                    current_patch_indices=current_patch_indices,
                    selected_patch_indices=selected_patch_indices,
                    n_select=quota,
                    patch_per_row=patch_per_row,
                )
            low_keep_indices.extend(chosen)

        low_keep_tensor = (
            torch.tensor(low_keep_indices, device=visual_scores.device, dtype=torch.long)
            if low_keep_indices
            else torch.empty(0, device=visual_scores.device, dtype=torch.long)
        )
        combined_keep = torch.cat([high_keep_indices.to(device=visual_scores.device), low_keep_tensor], dim=0)
        if combined_keep.numel() != target_keep:
            raise ValueError(
                f"Adaptive stratified keep budget mismatch: expected {target_keep}, got {combined_keep.numel()}"
            )
        keep_indices = combined_keep.sort().values

        pruned_mask = torch.ones(v_token_num, device=visual_scores.device, dtype=torch.bool)
        pruned_mask[keep_indices] = False
        pruned_indices = torch.nonzero(pruned_mask, as_tuple=False).flatten()

        low_keep_patch_indices = (
            current_patch_indices.index_select(0, low_keep_tensor.to(device=current_patch_indices.device))
            if low_keep_tensor.numel() > 0
            else torch.empty(0, device=current_patch_indices.device, dtype=torch.long)
        )
        keep_patch_indices = current_patch_indices.index_select(
            0, keep_indices.to(device=current_patch_indices.device)
        )
        pruned_patch_indices = current_patch_indices.index_select(
            0, pruned_indices.to(device=current_patch_indices.device)
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
            "current_patch_indices": current_patch_indices.detach().cpu().numpy(),
            "high_keep_indices": high_keep_indices.detach().cpu().numpy(),
            "low_keep_indices": low_keep_tensor.detach().cpu().numpy(),
            "keep_patch_indices": keep_patch_indices.detach().cpu().numpy(),
            "pruned_patch_indices": pruned_patch_indices.detach().cpu().numpy(),
            "high_keep_patch_indices": high_keep_patch_indices.detach().cpu().numpy(),
            "low_keep_patch_indices": low_keep_patch_indices.detach().cpu().numpy(),
            "target_keep": target_keep,
            "strategy_keep_high": int(high_keep_indices.numel()),
            "strategy_keep_low": int(low_keep_tensor.numel()),
            "grid_size": grid_size,
            "patch_per_row": patch_per_row,
            "high_ratio": adaptive_alpha,
            "intra_stratum_mode": intra_stratum_mode,
            "stratum_selected_counts": [selected_counts[sid] for sid in range(grid_size * grid_size)],
            "stratum_candidate_counts": [candidate_counts[sid] for sid in range(grid_size * grid_size)],
            "stratum_deficits": [deficits[sid] for sid in range(grid_size * grid_size)],
            "stratum_quotas": [quotas[sid] for sid in range(grid_size * grid_size)],
            "alpha_mode": "entropy",
            "adaptive_alpha": adaptive_alpha,
            "saliency_entropy": entropy_raw,
            "saliency_entropy_norm": entropy_norm,
            "alpha_min": alpha_min,
            "alpha_max": alpha_max,
        }
        return keep_indices, info
