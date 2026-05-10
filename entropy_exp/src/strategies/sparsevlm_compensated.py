"""SparseVLM compensated sampling pruning.

This variant removes explicit spatial grids.  It compresses the saliency
distribution with an entropy-adaptive power, then performs fixed-seed
sequential sampling without replacement.
"""

from __future__ import annotations

import random
from typing import Dict, Optional, Tuple

import torch

from .sparsevlm_adaptive_stratified import compute_target_keep_count
from .sparsevlm_score_memory import (
    SparseVLMScoreMemoryStrategy,
    compute_entropy_stats,
    deterministic_descending_indices,
)


class SparseVLMCompensatedStrategy(SparseVLMScoreMemoryStrategy):
    """SparseVLM saliency pruning via entropy-adaptive compensated sampling."""

    def _get_sampling_params(self) -> Tuple[float, float, int]:
        beta_min = float(self.config.get("beta_min", 0.3))
        beta_max = float(self.config.get("beta_max", 1.0))
        seed = int(self.config.get("seed", 42))
        if not 0.0 <= beta_min <= beta_max:
            raise ValueError(f"beta bounds must satisfy 0 <= beta_min <= beta_max, got {beta_min}, {beta_max}")
        return beta_min, beta_max, seed

    @staticmethod
    def _sampling_weights(mixed_score: torch.Tensor, beta: float) -> torch.Tensor:
        positive = mixed_score.to(dtype=torch.float32).clamp_min(0.0)
        if beta == 0.0:
            return torch.where(positive > 0, torch.ones_like(positive), torch.zeros_like(positive))
        return torch.where(positive > 0, torch.pow(positive, beta), torch.zeros_like(positive))

    @staticmethod
    def _sequential_sample(weights: torch.Tensor, keep_count: int, seed: int) -> Tuple[list[int], int]:
        if keep_count <= 0:
            return [], 0
        num_tokens = int(weights.numel())
        if keep_count >= num_tokens:
            return list(range(num_tokens)), 0

        rng = random.Random(int(seed))
        order = list(range(num_tokens))
        rng.shuffle(order)
        weight_values = [float(weights.detach().cpu()[idx].item()) for idx in range(num_tokens)]

        selected: list[int] = []
        selected_set: set[int] = set()
        remaining_budget = int(keep_count)
        for pos, idx in enumerate(order):
            if remaining_budget <= 0:
                break
            remaining_order = order[pos:]
            selectable_remaining = [candidate for candidate in remaining_order if weight_values[candidate] > 0.0]
            if len(selectable_remaining) <= remaining_budget:
                for candidate in selectable_remaining:
                    if candidate not in selected_set:
                        selected.append(candidate)
                        selected_set.add(candidate)
                remaining_budget = keep_count - len(selected)
                break

            weight = weight_values[idx]
            if weight <= 0.0:
                continue
            remaining_weight = sum(weight_values[candidate] for candidate in remaining_order)
            if remaining_weight <= 0.0:
                break
            hazard = min(max(remaining_budget * weight / remaining_weight, 0.0), 1.0)
            if rng.random() < hazard:
                selected.append(idx)
                selected_set.add(idx)
                remaining_budget -= 1

        fill_count = 0
        if len(selected) < keep_count:
            fill_count = keep_count - len(selected)
            fallback_order = deterministic_descending_indices(weights)
            for idx in fallback_order.detach().cpu().tolist():
                if idx in selected_set:
                    continue
                selected.append(int(idx))
                selected_set.add(int(idx))
                if len(selected) == keep_count:
                    break

        if len(selected) != keep_count:
            raise ValueError(f"Compensated sampling keep budget mismatch: expected {keep_count}, got {len(selected)}")
        return selected, fill_count

    def _derive_shuffle_seed(self, context: Dict[str, object], layer_idx: int, global_prune_step: int) -> int:
        _beta_min, _beta_max, base_seed = self._get_sampling_params()
        sample_index = int(context.get("score_sample_index", 0))
        return int(base_seed + sample_index * 1_000_003 + int(layer_idx) * 9_176 + int(global_prune_step))

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

        beta_min, beta_max, _base_seed = self._get_sampling_params()
        entropy_raw, entropy_norm = compute_entropy_stats(visual_scores)
        beta = beta_min + (1.0 - entropy_norm) * (beta_max - beta_min)

        prune_ratio = self.get_prune_ratio(layer_idx, visual_scores)
        min_visual_tokens_after_prune = int(self.config.get("min_visual_tokens_after_prune", 16))
        target_keep = compute_target_keep_count(
            num_visual=v_token_num,
            prune_ratio=prune_ratio,
            min_visual_tokens_after_prune=min_visual_tokens_after_prune,
        )

        sampling_weights = self._sampling_weights(mixed_score, beta)
        shuffle_seed = self._derive_shuffle_seed(context, layer_idx, int(memory["global_prune_step"]))
        selected, sampling_fill_count = self._sequential_sample(
            weights=sampling_weights,
            keep_count=target_keep,
            seed=shuffle_seed,
        )
        keep_indices = torch.tensor(selected, device=visual_scores.device, dtype=torch.long).sort().values
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
            "beta": float(beta),
            "beta_min": beta_min,
            "beta_max": beta_max,
            "sampling_weights": sampling_weights.detach().cpu().numpy(),
            "sampling_fill_count": int(sampling_fill_count),
            "shuffle_seed": int(shuffle_seed),
            "keep_indices": keep_indices.detach().cpu().numpy(),
            "pruned_indices": pruned_indices.detach().cpu().numpy(),
            "rater_indices": context["rater_indices"].detach().cpu().numpy(),
            "text_relevance_scores": context["text_relevance_scores"].detach().cpu().numpy(),
            "current_patch_indices": current_patch_indices.detach().cpu().numpy(),
            "keep_patch_indices": patch_info["keep_patch_indices"].detach().cpu().numpy(),
            "pruned_patch_indices": patch_info["pruned_patch_indices"].detach().cpu().numpy(),
            "target_keep": target_keep,
            "saliency_entropy": entropy_raw,
            "saliency_entropy_norm": entropy_norm,
            "global_prune_step": memory["global_prune_step"],
            "global_current_weight": memory["global_current_weight"],
            "global_ema_decay": memory["global_ema_decay"],
            "global_use_ema": memory["global_use_ema"],
            "global_saliency_ema": memory["global_saliency_ema"].detach().cpu().numpy(),
        }
        return keep_indices, info
