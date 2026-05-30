"""SparseVLM adaptive saliency-diversity pruning.

This strategy keeps SparseVLM's text-conditioned saliency as the gate for grid
quotas and candidate pools. Within each grid, low pruning ratios keep mostly
saliency anchors, while high pruning ratios allocate more of the quota to
max-min visual diversity fills.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch

from .sparsevlm_adaptive_stratified import build_stratum_index, compute_target_keep_count
from .sparsevlm_diverse_mmr import (
    SparseVLMDiverseMMRStrategy,
    _cosine_distance_matrix,
    _saliency_mass_ratio,
    _selected_mean_pairwise_distance,
)
from .sparsevlm_score_memory import compute_entropy_stats, rank_normalize_scores


VALID_LAYER_MODES = {"A", "S"}


def _clamp01(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


class SparseVLMAdaptiveDiverseMMRStrategy(SparseVLMDiverseMMRStrategy):
    """SparseVLM with pruning-ratio-adaptive saliency anchors and diversity fills."""

    def _layer_mode_map(self) -> Dict[int, str]:
        prune_ratio_map = self.config.get("prune_ratio_map")
        prune_layers = list(prune_ratio_map.keys()) if prune_ratio_map else self.config.get("prune_layers", [])
        if isinstance(prune_layers, int):
            prune_layers = [prune_layers]
        prune_layers = [int(layer) for layer in prune_layers]

        layer_modes = self.config.get("layer_modes")
        if layer_modes is None:
            layer_modes = ["A"] * len(prune_layers)
        if isinstance(layer_modes, str):
            layer_modes = [mode.strip() for mode in layer_modes.split(",") if mode.strip()]
        layer_modes = [str(mode).upper() for mode in layer_modes]

        if len(layer_modes) != len(prune_layers):
            raise ValueError(
                "sparsevlm_adaptive_diverse_mmr.layer_modes must have the same length as prune_layers, "
                f"got modes={layer_modes} and prune_layers={prune_layers}"
            )
        invalid = [mode for mode in layer_modes if mode not in VALID_LAYER_MODES]
        if invalid:
            raise ValueError(
                f"sparsevlm_adaptive_diverse_mmr.layer_modes only supports {sorted(VALID_LAYER_MODES)}, got {invalid}"
            )
        return dict(zip(prune_layers, layer_modes))

    def _layer_mode(self, layer_idx: int) -> str:
        mode_map = self._layer_mode_map()
        if int(layer_idx) not in mode_map:
            raise ValueError(f"Layer {layer_idx} is missing from sparsevlm_adaptive_diverse_mmr.layer_modes")
        return mode_map[int(layer_idx)]

    def _get_adaptive_params(self, initial_v_token_num: int) -> Tuple[int, int, float, float, float]:
        inferred_patch_per_row = int(round(math.sqrt(float(initial_v_token_num))))
        if inferred_patch_per_row * inferred_patch_per_row != int(initial_v_token_num):
            inferred_patch_per_row = 24

        grid_size = int(self.config.get("grid_size", 6))
        patch_per_row = int(self.config.get("patch_per_row", inferred_patch_per_row))
        grid_balance_weight = float(self.config.get("grid_balance_weight", 0.25))
        min_candidate_multiplier = float(self.config.get("min_candidate_multiplier", 1.2))
        max_candidate_multiplier = float(self.config.get("max_candidate_multiplier", 2.5))
        distance_metric = str(self.config.get("distance_metric", "cosine")).lower()

        if grid_size <= 0:
            raise ValueError(f"grid_size must be > 0, got {grid_size}")
        if patch_per_row <= 0:
            raise ValueError(f"patch_per_row must be > 0, got {patch_per_row}")
        if patch_per_row * patch_per_row != int(initial_v_token_num):
            raise ValueError(
                "patch_per_row * patch_per_row must equal the initial visual token count, "
                f"got {patch_per_row}^2 vs {initial_v_token_num}"
            )
        if patch_per_row % grid_size != 0:
            raise ValueError(
                f"grid_size must divide patch_per_row, got grid_size={grid_size}, patch_per_row={patch_per_row}"
            )
        if not 0.0 <= grid_balance_weight <= 1.0:
            raise ValueError(f"grid_balance_weight must be in [0, 1], got {grid_balance_weight}")
        if min_candidate_multiplier < 1.0:
            raise ValueError(f"min_candidate_multiplier must be >= 1, got {min_candidate_multiplier}")
        if max_candidate_multiplier < min_candidate_multiplier:
            raise ValueError(
                "max_candidate_multiplier must be >= min_candidate_multiplier, "
                f"got {max_candidate_multiplier} < {min_candidate_multiplier}"
            )
        if distance_metric != "cosine":
            raise ValueError(f"Only distance_metric=cosine is supported, got {distance_metric!r}")
        return grid_size, patch_per_row, grid_balance_weight, min_candidate_multiplier, max_candidate_multiplier

    @staticmethod
    def _candidate_multiplier(
        diversity_ratio: float,
        min_candidate_multiplier: float,
        max_candidate_multiplier: float,
    ) -> float:
        return min_candidate_multiplier + diversity_ratio * (max_candidate_multiplier - min_candidate_multiplier)

    def _grid_local_adaptive_select(
        self,
        mixed_score: torch.Tensor,
        distance: torch.Tensor,
        stratum_index: Dict[int, List[int]],
        quotas: List[int],
        target_keep: int,
        anchor_ratio: float,
        candidate_pool_multiplier: float,
    ) -> Dict[str, object]:
        num_visual = int(mixed_score.numel())
        num_strata = len(quotas)
        selected_mask = torch.zeros(num_visual, device=mixed_score.device, dtype=torch.bool)
        grid_anchor_indices: List[List[int]] = [[] for _ in range(num_strata)]
        grid_diverse_fill_indices: List[List[int]] = [[] for _ in range(num_strata)]
        grid_distance_selected_order: List[List[int]] = [[] for _ in range(num_strata)]
        grid_candidate_indices: List[List[int]] = [[] for _ in range(num_strata)]
        grid_candidate_counts = [0 for _ in range(num_strata)]
        flat_candidate_pool: List[int] = []
        flat_anchors: List[int] = []
        flat_order: List[int] = []
        flat_gains: List[float] = []

        for sid in range(num_strata):
            quota = int(quotas[sid])
            members = self._ordered_subset(mixed_score, [int(idx) for idx in stratum_index.get(sid, [])])
            if quota <= 0 or not members:
                continue

            raw_anchor_count = int(math.ceil(float(quota) * anchor_ratio))
            anchor_count = min(len(members), quota, max(1, raw_anchor_count))
            candidate_count = min(
                len(members),
                max(quota, int(math.ceil(float(quota) * candidate_pool_multiplier))),
            )
            anchors = members[:anchor_count]
            candidates = members[:candidate_count]
            grid_candidate_indices[sid] = candidates
            grid_candidate_counts[sid] = len(candidates)
            flat_candidate_pool.extend(candidates)

            selected_in_grid: List[int] = []
            for token_idx in anchors:
                if selected_mask[token_idx]:
                    continue
                selected_mask[token_idx] = True
                selected_in_grid.append(token_idx)
                grid_anchor_indices[sid].append(token_idx)
                grid_distance_selected_order[sid].append(token_idx)
                flat_anchors.append(token_idx)
                flat_order.append(token_idx)
                flat_gains.append(0.0)

            while len(selected_in_grid) < quota:
                remaining = [idx for idx in candidates if not bool(selected_mask[idx].item())]
                if not remaining:
                    break
                selected_tensor = torch.tensor(selected_in_grid, device=mixed_score.device, dtype=torch.long)
                remaining_tensor = torch.tensor(remaining, device=mixed_score.device, dtype=torch.long)
                if selected_tensor.numel() == 0:
                    gains = torch.zeros(remaining_tensor.numel(), device=mixed_score.device, dtype=torch.float32)
                else:
                    gains = distance.index_select(0, remaining_tensor).index_select(1, selected_tensor).min(dim=1).values
                gain_values = gains.detach().cpu().tolist()
                best_position = min(
                    range(len(remaining)),
                    key=lambda pos: (-float(gain_values[pos]), int(remaining[pos])),
                )
                best_index = int(remaining[best_position])
                best_gain = float(gain_values[best_position])
                selected_mask[best_index] = True
                selected_in_grid.append(best_index)
                grid_diverse_fill_indices[sid].append(best_index)
                grid_distance_selected_order[sid].append(best_index)
                flat_order.append(best_index)
                flat_gains.append(best_gain)

        global_fallback_indices: List[int] = []
        selected_count = int(selected_mask.sum().item())
        if selected_count < target_keep:
            for token_idx in self._ordered_subset(mixed_score, list(range(num_visual))):
                if bool(selected_mask[int(token_idx)].item()):
                    continue
                selected_mask[int(token_idx)] = True
                global_fallback_indices.append(int(token_idx))
                flat_order.append(int(token_idx))
                flat_gains.append(0.0)
                selected_count += 1
                if selected_count == target_keep:
                    break

        keep_indices = torch.nonzero(selected_mask, as_tuple=False).flatten().sort().values
        grid_selected_counts = [0 for _ in range(num_strata)]
        for sid in range(num_strata):
            grid_selected_counts[sid] = sum(1 for idx in stratum_index.get(sid, []) if bool(selected_mask[int(idx)].item()))

        return {
            "keep_indices": keep_indices,
            "core_indices": torch.tensor(sorted(flat_anchors), device=mixed_score.device, dtype=torch.long),
            "candidate_pool_indices": torch.tensor(sorted(set(flat_candidate_pool)), device=mixed_score.device, dtype=torch.long),
            "mmr_selected_order": torch.tensor(flat_order, device=mixed_score.device, dtype=torch.long),
            "diversity_gain": torch.tensor(flat_gains, device=mixed_score.device, dtype=torch.float32),
            "grid_selected_counts": grid_selected_counts,
            "grid_candidate_counts": grid_candidate_counts,
            "grid_anchor_indices": grid_anchor_indices,
            "grid_diverse_fill_indices": grid_diverse_fill_indices,
            "grid_distance_selected_order": grid_distance_selected_order,
            "grid_candidate_indices": grid_candidate_indices,
            "grid_global_fallback_indices": global_fallback_indices,
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
        if attn_weights is None:
            raise ValueError("SparseVLMAdaptiveDiverseMMRStrategy requires attention weights")

        context = self._require_sample_context()
        current_patch_indices = self._require_current_patch_indices(context, v_token_num)
        mode = self._layer_mode(layer_idx)

        if mode == "S":
            visual_scores = self.compute_importance(
                attn_weights=attn_weights,
                v_token_start=v_token_start,
                v_token_num=v_token_num,
                text_token_start=text_token_start,
                layer_idx=layer_idx,
                device=device,
            )
            current_rank_score = rank_normalize_scores(visual_scores)
            keep_indices, _pruned_indices, info = self._sv_keep_mask(
                visual_scores=visual_scores,
                current_rank_score=current_rank_score,
                current_patch_indices=current_patch_indices,
                layer_idx=layer_idx,
                v_token_num=v_token_num,
            )
            info.update(
                {
                    "rater_indices": context["rater_indices"].detach().cpu().numpy(),
                    "text_relevance_scores": context["text_relevance_scores"].detach().cpu().numpy(),
                    "global_prune_step": int(context.get("score_prune_step", 0)),
                }
            )
            self._set_pending_decision(context, current_patch_indices, layer_idx)
            return keep_indices, info

        if current_visual_embeds is None:
            raise ValueError("SparseVLMAdaptiveDiverseMMRStrategy requires current_visual_embeds for A layers")
        if int(current_visual_embeds.shape[0]) != int(v_token_num):
            raise ValueError(
                "current_visual_embeds length must match current visual token count, "
                f"got {current_visual_embeds.shape[0]} vs {v_token_num}"
            )

        memory = self._score_memory_inputs(
            attn_weights=attn_weights,
            v_token_start=v_token_start,
            v_token_num=v_token_num,
            text_token_start=text_token_start,
            layer_idx=layer_idx,
            device=device,
        )
        visual_scores = memory["visual_scores"]
        current_rank_score = memory["current_rank_score"]
        mixed_score = memory["mixed_score"]
        entropy_raw, entropy_norm = compute_entropy_stats(visual_scores)
        (
            grid_size,
            patch_per_row,
            grid_balance_weight,
            min_candidate_multiplier,
            max_candidate_multiplier,
        ) = self._get_adaptive_params(int(context["initial_v_token_num"]))

        prune_ratio = self.get_prune_ratio(layer_idx, visual_scores)
        diversity_ratio = _clamp01(float(prune_ratio))
        anchor_ratio = 1.0 - diversity_ratio
        candidate_pool_multiplier_effective = self._candidate_multiplier(
            diversity_ratio=diversity_ratio,
            min_candidate_multiplier=min_candidate_multiplier,
            max_candidate_multiplier=max_candidate_multiplier,
        )
        min_visual_tokens_after_prune = int(self.config.get("min_visual_tokens_after_prune", 16))
        target_keep = compute_target_keep_count(
            num_visual=v_token_num,
            prune_ratio=prune_ratio,
            min_visual_tokens_after_prune=min_visual_tokens_after_prune,
        )

        distance = _cosine_distance_matrix(current_visual_embeds).to(device=mixed_score.device)
        stratum_index = build_stratum_index(
            current_patch_indices=current_patch_indices,
            grid_size=grid_size,
            patch_per_row=patch_per_row,
        )
        grid_quota, grid_saliency_topk_counts = self._compute_grid_quotas(
            mixed_score=mixed_score,
            stratum_index=stratum_index,
            target_keep=target_keep,
            grid_size=grid_size,
            grid_balance_weight=grid_balance_weight,
        )
        selection = self._grid_local_adaptive_select(
            mixed_score=mixed_score,
            distance=distance,
            stratum_index=stratum_index,
            quotas=grid_quota,
            target_keep=target_keep,
            anchor_ratio=anchor_ratio,
            candidate_pool_multiplier=candidate_pool_multiplier_effective,
        )
        keep_indices = selection["keep_indices"]
        if int(keep_indices.numel()) != int(target_keep):
            raise AssertionError(
                "sparsevlm_adaptive_diverse_mmr selection produced the wrong keep count, "
                f"got {keep_indices.numel()} vs {target_keep}"
            )
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
            "current_rank_score": current_rank_score.detach().cpu().numpy(),
            "mixed_score": mixed_score.detach().cpu().numpy(),
            "keep_indices": keep_indices.detach().cpu().numpy(),
            "pruned_indices": pruned_indices.detach().cpu().numpy(),
            "rater_indices": context["rater_indices"].detach().cpu().numpy(),
            "text_relevance_scores": context["text_relevance_scores"].detach().cpu().numpy(),
            "current_patch_indices": current_patch_indices.detach().cpu().numpy(),
            "keep_patch_indices": patch_info["keep_patch_indices"].detach().cpu().numpy(),
            "pruned_patch_indices": patch_info["pruned_patch_indices"].detach().cpu().numpy(),
            "target_keep": target_keep,
            "strategy_keep_high": int(keep_indices.numel()),
            "strategy_keep_low": 0,
            "layer_mode": mode,
            "layer_strategy_effective": "sparsevlm_adaptive_diverse_mmr",
            "selection_rule": "adaptive_grid_saliency_diversity",
            "grid_size": grid_size,
            "patch_per_row": patch_per_row,
            "grid_balance_weight": grid_balance_weight,
            "diversity_ratio": diversity_ratio,
            "anchor_ratio": anchor_ratio,
            "candidate_pool_multiplier": candidate_pool_multiplier_effective,
            "candidate_pool_multiplier_effective": candidate_pool_multiplier_effective,
            "min_candidate_multiplier": min_candidate_multiplier,
            "max_candidate_multiplier": max_candidate_multiplier,
            "distance_metric": "cosine",
            "saliency_entropy": entropy_raw,
            "saliency_entropy_norm": entropy_norm,
            "grid_saliency_topk_counts": grid_saliency_topk_counts,
            "grid_quota": grid_quota,
            "grid_selected_counts": selection["grid_selected_counts"],
            "grid_candidate_counts": selection["grid_candidate_counts"],
            "grid_anchor_indices": selection["grid_anchor_indices"],
            "grid_diverse_fill_indices": selection["grid_diverse_fill_indices"],
            "grid_fill_indices": selection["grid_diverse_fill_indices"],
            "grid_distance_selected_order": selection["grid_distance_selected_order"],
            "grid_candidate_indices": selection["grid_candidate_indices"],
            "grid_global_fallback_indices": selection["grid_global_fallback_indices"],
            "core_indices": selection["core_indices"].detach().cpu().numpy(),
            "candidate_pool_indices": selection["candidate_pool_indices"].detach().cpu().numpy(),
            "mmr_selected_order": selection["mmr_selected_order"].detach().cpu().numpy(),
            "diversity_gain": selection["diversity_gain"].detach().cpu().numpy(),
            "mean_selected_pairwise_distance": _selected_mean_pairwise_distance(distance, keep_indices),
            "retained_saliency_mass_ratio": _saliency_mass_ratio(visual_scores, keep_indices),
            "global_prune_step": memory["global_prune_step"],
            "global_current_weight": memory["global_current_weight"],
            "global_ema_decay": memory["global_ema_decay"],
            "global_use_ema": memory["global_use_ema"],
            "use_score_memory": memory["use_score_memory"],
            "global_saliency_ema": memory["global_saliency_ema"].detach().cpu().numpy(),
        }
        return keep_indices, info
