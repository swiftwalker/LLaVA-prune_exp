"""
Cross-layer global-memory variant of entropy-adaptive SparseVLM pruning.

The strategy keeps the existing online pruning contract: tokens are physically
removed at each configured layer and cannot be revived.  The difference is that
later pruning layers can consult sample-local history collected from earlier
layers: rank-normalized saliency EMA and spatial keep-exposure debt.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch

from .sparsevlm_adaptive_stratified import (
    allocate_adaptive_stratified_quotas,
    build_stratum_index,
    compute_target_keep_count,
    patch_index_to_stratum_id,
    randomly_select_token_indices,
    select_farthest_token_indices,
    validate_adaptive_stratified_config,
)
from .sparsevlm_entropy_alpha import SparseVLMEntropyAlphaStrategy


def _rank_normalize_scores(scores: torch.Tensor) -> torch.Tensor:
    """Map scores to [0, 1] by descending rank; highest score receives 1."""
    if scores.ndim != 1:
        raise ValueError(f"scores must be 1D, got {tuple(scores.shape)}")
    num_scores = int(scores.numel())
    if num_scores == 0:
        return scores.to(dtype=torch.float32)
    if num_scores == 1:
        return torch.ones_like(scores, dtype=torch.float32)

    order = torch.argsort(scores, descending=True)
    rank_values = torch.linspace(1.0, 0.0, steps=num_scores, device=scores.device, dtype=torch.float32)
    normalized = torch.empty(num_scores, device=scores.device, dtype=torch.float32)
    normalized[order] = rank_values
    return normalized


def allocate_debt_aware_stratified_quotas(
    debts: Dict[int, float],
    candidate_counts: Dict[int, int],
    n_low: int,
    grid_size: int,
) -> Dict[int, int]:
    """Allocate low-score compensation quotas using externally supplied debt."""
    num_strata = grid_size * grid_size
    quotas = {sid: 0 for sid in range(num_strata)}
    if n_low <= 0:
        return quotas

    capacities = {sid: max(int(candidate_counts.get(sid, 0)), 0) for sid in range(num_strata)}
    total_capacity = sum(capacities.values())
    if total_capacity < n_low:
        raise ValueError(f"Not enough candidate capacity to allocate {n_low} low-score tokens, capacity={total_capacity}")

    clamped_debts = {sid: max(float(debts.get(sid, 0.0)), 0.0) for sid in range(num_strata)}
    total_debt = sum(clamped_debts.values())
    if total_debt > 0:
        for sid in range(num_strata):
            raw_quota = math.floor(n_low * clamped_debts[sid] / total_debt)
            quotas[sid] = min(raw_quota, capacities[sid])
        fill_order = sorted(range(num_strata), key=lambda sid: (-clamped_debts[sid], sid))
    else:
        uniform_base = n_low // num_strata
        for sid in range(num_strata):
            quotas[sid] = min(uniform_base, capacities[sid])
        fill_order = list(range(num_strata))

    remaining = n_low - sum(quotas.values())
    while remaining > 0:
        progressed = False
        for sid in fill_order:
            if quotas[sid] >= capacities[sid]:
                continue
            quotas[sid] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            raise ValueError("Unable to finish debt-aware quota allocation despite sufficient total capacity")

    return quotas


class SparseVLMEntropyAlphaGlobalStrategy(SparseVLMEntropyAlphaStrategy):
    """Entropy-alpha SparseVLM with online cross-layer global memory."""

    def _get_alpha_bounds(self) -> Tuple[float, float]:
        alpha_min = float(self.config.get("alpha_min", 0.5))
        alpha_max = float(self.config.get("alpha_max", 1.0))
        if not 0.0 <= alpha_min <= alpha_max <= 1.0:
            raise ValueError(
                f"alpha bounds must satisfy 0 <= alpha_min <= alpha_max <= 1, got {alpha_min}, {alpha_max}"
            )
        return alpha_min, alpha_max

    def _get_global_params(self) -> Tuple[float, float, float, bool]:
        current_weight = float(self.config.get("global_current_weight", 0.5))
        ema_decay = float(self.config.get("global_ema_decay", 0.6))
        debt_weight = float(self.config.get("global_debt_weight", 0.5))
        first_layer_fallback = bool(self.config.get("global_first_layer_fallback", True))
        if not 0.0 <= current_weight <= 1.0:
            raise ValueError(f"global_current_weight must be in [0, 1], got {current_weight}")
        if not 0.0 <= ema_decay <= 1.0:
            raise ValueError(f"global_ema_decay must be in [0, 1], got {ema_decay}")
        if not 0.0 <= debt_weight <= 1.0:
            raise ValueError(f"global_debt_weight must be in [0, 1], got {debt_weight}")
        return current_weight, ema_decay, debt_weight, first_layer_fallback

    def prepare_sample(
        self,
        inputs_embeds: torch.Tensor,
        v_token_start: int,
        v_token_num: int,
        text_token_start: int,
        text_token_ids: Optional[torch.Tensor] = None,
        text_special_token_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, object]:
        sample_info = super().prepare_sample(
            inputs_embeds=inputs_embeds,
            v_token_start=v_token_start,
            v_token_num=v_token_num,
            text_token_start=text_token_start,
            text_token_ids=text_token_ids,
            text_special_token_mask=text_special_token_mask,
        )
        context = self._require_sample_context()
        grid_size = int(self.config.get("grid_size", 6))
        device = inputs_embeds.device
        context["global_saliency_ema"] = torch.zeros(v_token_num, device=device, dtype=torch.float32)
        context["global_saliency_observed"] = torch.zeros(v_token_num, device=device, dtype=torch.bool)
        context["global_drop_layer"] = torch.full((v_token_num,), -1, device=device, dtype=torch.long)
        context["global_stratum_keep_exposure"] = torch.zeros(grid_size * grid_size, device=device, dtype=torch.float32)
        context["global_prune_step"] = 0
        context["global_pending_decision"] = None
        return sample_info

    def _update_global_saliency(
        self,
        context: Dict[str, object],
        current_patch_indices: torch.Tensor,
        current_saliency_norm: torch.Tensor,
        ema_decay: float,
    ) -> torch.Tensor:
        ema = context["global_saliency_ema"]
        observed = context["global_saliency_observed"]
        patch_indices = current_patch_indices.to(device=ema.device, dtype=torch.long)
        current_values = current_saliency_norm.to(device=ema.device, dtype=torch.float32)

        previous = ema.index_select(0, patch_indices)
        was_observed = observed.index_select(0, patch_indices)
        updated = torch.where(
            was_observed,
            previous * ema_decay + current_values * (1.0 - ema_decay),
            current_values,
        )
        ema[patch_indices] = updated.detach()
        observed[patch_indices] = True
        return updated.to(device=current_saliency_norm.device)

    def _historical_stratum_debt(
        self,
        context: Dict[str, object],
        target_keep: int,
        grid_size: int,
    ) -> Dict[int, float]:
        num_strata = grid_size * grid_size
        if int(context.get("global_prune_step", 0)) <= 0:
            return {sid: 0.0 for sid in range(num_strata)}

        exposure = context["global_stratum_keep_exposure"].detach().to(dtype=torch.float64).cpu()
        total_exposure = float(exposure.sum().item())
        if total_exposure <= 0.0:
            return {sid: 0.0 for sid in range(num_strata)}

        uniform_share = 1.0 / float(num_strata)
        scale = float(target_keep)
        debts = {}
        for sid in range(num_strata):
            share = float(exposure[sid].item()) / total_exposure
            debts[sid] = max(0.0, uniform_share - share) * scale
        return debts

    def update_after_prune(self, keep_indices: torch.Tensor, layer_idx: int) -> None:
        context = self._require_sample_context()
        pending = context.get("global_pending_decision")
        if pending is None:
            super().update_after_prune(keep_indices=keep_indices, layer_idx=layer_idx)
            return
        if int(pending["layer_idx"]) != int(layer_idx):
            raise ValueError(f"Pending global decision is for layer {pending['layer_idx']}, got layer {layer_idx}")

        current_patch_indices = pending["current_patch_indices"]
        keep_indices = keep_indices.to(device=current_patch_indices.device, dtype=torch.long)
        pruned_mask = torch.ones(current_patch_indices.numel(), device=current_patch_indices.device, dtype=torch.bool)
        pruned_mask[keep_indices] = False
        pruned_patch_indices = current_patch_indices[pruned_mask]
        keep_patch_indices = current_patch_indices.index_select(0, keep_indices)

        drop_layer = context["global_drop_layer"]
        if pruned_patch_indices.numel() > 0:
            drop_layer[pruned_patch_indices.to(device=drop_layer.device, dtype=torch.long)] = int(layer_idx)

        grid_size = int(pending["grid_size"])
        patch_per_row = int(pending["patch_per_row"])
        exposure = context["global_stratum_keep_exposure"]
        for patch_idx in keep_patch_indices.detach().cpu().tolist():
            sid = patch_index_to_stratum_id(int(patch_idx), grid_size=grid_size, patch_per_row=patch_per_row)
            exposure[sid] += 1.0

        context["current_patch_indices"] = keep_patch_indices.detach()
        context["global_prune_step"] = int(context.get("global_prune_step", 0)) + 1
        context["global_pending_decision"] = None

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
            raise ValueError("SparseVLMEntropyAlphaGlobalStrategy requires attention weights")

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
        current_weight, ema_decay, debt_weight, first_layer_fallback = self._get_global_params()
        global_prune_step = int(context.get("global_prune_step", 0))

        visual_scores = self.compute_importance(
            attn_weights=attn_weights,
            v_token_start=v_token_start,
            v_token_num=v_token_num,
            text_token_start=text_token_start,
            layer_idx=layer_idx,
            device=device,
        )
        current_saliency_norm = _rank_normalize_scores(visual_scores)
        global_saliency_ema = self._update_global_saliency(
            context=context,
            current_patch_indices=current_patch_indices,
            current_saliency_norm=current_saliency_norm,
            ema_decay=ema_decay,
        )

        use_global = not (first_layer_fallback and global_prune_step == 0)
        if use_global:
            selection_score = (
                current_saliency_norm.to(dtype=torch.float32) * current_weight
                + global_saliency_ema.to(device=current_saliency_norm.device, dtype=torch.float32) * (1.0 - current_weight)
            )
            alpha_scores = selection_score
        else:
            selection_score = visual_scores.to(dtype=torch.float32)
            alpha_scores = visual_scores

        adaptive_alpha, entropy_raw, entropy_norm = self._compute_adaptive_alpha(alpha_scores)
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

        sorted_desc = torch.argsort(selection_score, descending=True)
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

        if use_global:
            num_strata = grid_size * grid_size
            current_deficits = {
                sid: max(0.0, float(target_keep) / num_strata - int(selected_counts.get(sid, 0)))
                for sid in range(num_strata)
            }
            history_debt = self._historical_stratum_debt(context, target_keep=target_keep, grid_size=grid_size)
            combined_debt = {
                sid: (1.0 - debt_weight) * current_deficits[sid] + debt_weight * history_debt[sid]
                for sid in range(num_strata)
            }
            quotas = allocate_debt_aware_stratified_quotas(
                debts=combined_debt,
                candidate_counts=candidate_counts,
                n_low=n_low,
                grid_size=grid_size,
            )
            deficits = current_deficits
        else:
            quotas, deficits = allocate_adaptive_stratified_quotas(
                selected_counts=selected_counts,
                candidate_counts=candidate_counts,
                target_keep=target_keep,
                n_low=n_low,
                grid_size=grid_size,
            )
            history_debt = {sid: 0.0 for sid in range(grid_size * grid_size)}
            combined_debt = deficits

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
                f"Global entropy-alpha keep budget mismatch: expected {target_keep}, got {combined_keep.numel()}"
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

        context["global_pending_decision"] = {
            "layer_idx": int(layer_idx),
            "current_patch_indices": current_patch_indices.detach(),
            "grid_size": grid_size,
            "patch_per_row": patch_per_row,
        }

        info = {
            "prune_ratio": prune_ratio,
            "num_visual_before": v_token_num,
            "num_visual_after": int(keep_indices.numel()),
            "num_pruned": int(pruned_indices.numel()),
            "importance_scores": visual_scores.detach().cpu().numpy(),
            "visual_scores": visual_scores.detach().cpu().numpy(),
            "current_saliency_norm": current_saliency_norm.detach().cpu().numpy(),
            "global_selection_score": selection_score.detach().cpu().numpy(),
            "global_saliency_ema": global_saliency_ema.detach().cpu().numpy(),
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
            "stratum_history_debt": [history_debt[sid] for sid in range(grid_size * grid_size)],
            "stratum_combined_debt": [combined_debt[sid] for sid in range(grid_size * grid_size)],
            "alpha_mode": "entropy_global",
            "adaptive_alpha": adaptive_alpha,
            "saliency_entropy": entropy_raw,
            "saliency_entropy_norm": entropy_norm,
            "alpha_min": alpha_min,
            "alpha_max": alpha_max,
            "global_prune_step": global_prune_step,
            "global_current_weight": current_weight,
            "global_ema_decay": ema_decay,
            "global_debt_weight": debt_weight,
            "global_first_layer_fallback": first_layer_fallback,
            "global_use_global": use_global,
            **self._get_alpha_mapping_info(),
        }
        return keep_indices, info
