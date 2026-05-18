"""Layer-wise SparseVLM / score-boosting hybrid pruning."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from .sparsevlm_adaptive_stratified import build_stratum_index, compute_target_keep_count
from .sparsevlm_boost import SparseVLMBoostStrategy
from .sparsevlm_score_memory import (
    compute_entropy_stats,
    deterministic_topk_indices,
    rank_normalize_scores,
)


VALID_LAYER_MODES = {"S", "O"}


class SparseVLMBoostHybridStrategy(SparseVLMBoostStrategy):
    """Switch between pure SparseVLM and boost selection at configured prune layers."""

    def _layer_mode_map(self) -> Dict[int, str]:
        prune_ratio_map = self.config.get("prune_ratio_map")
        prune_layers = list(prune_ratio_map.keys()) if prune_ratio_map else self.config.get("prune_layers", [])
        if isinstance(prune_layers, int):
            prune_layers = [prune_layers]
        prune_layers = [int(layer) for layer in prune_layers]

        layer_modes = self.config.get("layer_modes")
        if layer_modes is None:
            layer_modes = ["O"] * len(prune_layers)
        if isinstance(layer_modes, str):
            layer_modes = [mode.strip() for mode in layer_modes.split(",") if mode.strip()]
        layer_modes = [str(mode).upper() for mode in layer_modes]

        if len(layer_modes) != len(prune_layers):
            raise ValueError(
                "sparsevlm_boost_hybrid.layer_modes must have the same length as prune_layers, "
                f"got modes={layer_modes} and prune_layers={prune_layers}"
            )
        invalid = [mode for mode in layer_modes if mode not in VALID_LAYER_MODES]
        if invalid:
            raise ValueError(
                f"sparsevlm_boost_hybrid.layer_modes only supports {sorted(VALID_LAYER_MODES)}, got {invalid}"
            )
        return dict(zip(prune_layers, layer_modes))

    def _layer_mode(self, layer_idx: int) -> str:
        mode_map = self._layer_mode_map()
        if int(layer_idx) not in mode_map:
            raise ValueError(f"Layer {layer_idx} is missing from sparsevlm_boost_hybrid.layer_modes")
        return mode_map[int(layer_idx)]

    def _sv2_keep_mask(
        self,
        visual_scores: torch.Tensor,
        current_patch_indices: torch.Tensor,
        layer_idx: int,
        v_token_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
        current_rank_score = rank_normalize_scores(visual_scores)
        prune_ratio = self.get_prune_ratio(layer_idx, visual_scores)
        min_visual_tokens_after_prune = int(self.config.get("min_visual_tokens_after_prune", 16))
        target_keep = compute_target_keep_count(
            num_visual=v_token_num,
            prune_ratio=prune_ratio,
            min_visual_tokens_after_prune=min_visual_tokens_after_prune,
        )

        keep_indices = deterministic_topk_indices(current_rank_score, target_keep)
        pruned_mask = torch.ones(v_token_num, device=visual_scores.device, dtype=torch.bool)
        pruned_mask[keep_indices] = False
        pruned_indices = torch.nonzero(pruned_mask, as_tuple=False).flatten()
        patch_info = self._patch_index_info(current_patch_indices, keep_indices, pruned_indices)

        return keep_indices, pruned_indices, {
            "prune_ratio": prune_ratio,
            "num_visual_before": v_token_num,
            "num_visual_after": int(keep_indices.numel()),
            "num_pruned": int(pruned_indices.numel()),
            "importance_scores": visual_scores.detach().cpu().numpy(),
            "visual_scores": visual_scores.detach().cpu().numpy(),
            "current_rank_score": current_rank_score.detach().cpu().numpy(),
            "mixed_score": current_rank_score.detach().cpu().numpy(),
            "keep_indices": keep_indices.detach().cpu().numpy(),
            "pruned_indices": pruned_indices.detach().cpu().numpy(),
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
        }

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
            raise ValueError("SparseVLMBoostHybridStrategy requires attention weights")

        context = self._require_sample_context()
        current_patch_indices = self._require_current_patch_indices(context, v_token_num)
        visual_scores = self.compute_importance(
            attn_weights=attn_weights,
            v_token_start=v_token_start,
            v_token_num=v_token_num,
            text_token_start=text_token_start,
            layer_idx=layer_idx,
            device=device,
        )
        mode = self._layer_mode(layer_idx)

        if mode == "S":
            keep_indices, _pruned_indices, info = self._sv2_keep_mask(
                visual_scores=visual_scores,
                current_patch_indices=current_patch_indices,
                layer_idx=layer_idx,
                v_token_num=v_token_num,
            )
            info.update(
                {
                    "layer_mode": mode,
                    "layer_strategy_effective": "sparsevlm",
                    "rater_indices": context["rater_indices"].detach().cpu().numpy(),
                    "text_relevance_scores": context["text_relevance_scores"].detach().cpu().numpy(),
                    "global_prune_step": int(context.get("score_prune_step", 0)),
                    "global_use_ema": False,
                    "use_score_memory": False,
                }
            )
            self._set_pending_decision(context, current_patch_indices, layer_idx)
            return keep_indices, info

        current_rank_score = rank_normalize_scores(visual_scores)
        mixed_score = current_rank_score
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

        return keep_indices, {
            "prune_ratio": prune_ratio,
            "num_visual_before": v_token_num,
            "num_visual_after": int(keep_indices.numel()),
            "num_pruned": int(pruned_indices.numel()),
            "importance_scores": visual_scores.detach().cpu().numpy(),
            "visual_scores": visual_scores.detach().cpu().numpy(),
            "current_rank_score": current_rank_score.detach().cpu().numpy(),
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
            "global_prune_step": int(context.get("score_prune_step", 0)),
            "global_use_ema": False,
            "use_score_memory": False,
            "global_saliency_ema": torch.zeros_like(current_rank_score).detach().cpu().numpy(),
            "layer_mode": mode,
            "layer_strategy_effective": "sparsevlm_boost",
        }
