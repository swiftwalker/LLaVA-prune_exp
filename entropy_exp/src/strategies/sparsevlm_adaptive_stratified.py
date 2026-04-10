"""
Adaptive stratified SparseVLM pruning.

This strategy reuses SparseVLM's text-rater selection and text->vision
attention scoring, then fills the keep budget with spatially adaptive
compensation across image strata.
"""

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from .sparsevlm import SparseVLMStrategy


VALID_INTRA_STRATUM_MODES = {"random", "farthest"}


def validate_adaptive_stratified_config(
    initial_v_token_num: int,
    patch_per_row: int,
    grid_size: int,
    high_ratio: float,
    intra_stratum_mode: str,
) -> None:
    """Validate configuration shared by adaptive stratified pruning helpers."""
    if initial_v_token_num <= 0:
        raise ValueError(f"initial_v_token_num must be > 0, got {initial_v_token_num}")
    if patch_per_row <= 0:
        raise ValueError(f"patch_per_row must be > 0, got {patch_per_row}")
    if grid_size <= 0:
        raise ValueError(f"grid_size must be > 0, got {grid_size}")
    if patch_per_row * patch_per_row != initial_v_token_num:
        raise ValueError(
            "patch_per_row * patch_per_row must equal the initial visual token count, "
            f"got {patch_per_row}^2 vs {initial_v_token_num}"
        )
    if patch_per_row % grid_size != 0:
        raise ValueError(
            f"grid_size must divide patch_per_row, got grid_size={grid_size}, patch_per_row={patch_per_row}"
        )
    if not 0.0 <= float(high_ratio) <= 1.0:
        raise ValueError(f"high_ratio must be in [0, 1], got {high_ratio}")
    if intra_stratum_mode not in VALID_INTRA_STRATUM_MODES:
        raise ValueError(
            f"intra_stratum_mode must be one of {sorted(VALID_INTRA_STRATUM_MODES)}, got {intra_stratum_mode}"
        )


def patch_index_to_coords(patch_index: int, patch_per_row: int) -> Tuple[int, int]:
    """Map a flattened patch index back to 2D grid coordinates."""
    return patch_index // patch_per_row, patch_index % patch_per_row


def patch_index_to_stratum_id(
    patch_index: int,
    grid_size: int,
    patch_per_row: int,
) -> int:
    """Map an original patch index to its stratum id."""
    y, x = patch_index_to_coords(int(patch_index), patch_per_row)
    sy = y * grid_size // patch_per_row
    sx = x * grid_size // patch_per_row
    return sy * grid_size + sx


def build_stratum_index(
    current_patch_indices: Sequence[int] | torch.Tensor,
    grid_size: int,
    patch_per_row: int,
) -> Dict[int, List[int]]:
    """Group current visual-token indices by stratum id."""
    if isinstance(current_patch_indices, torch.Tensor):
        patch_indices = current_patch_indices.detach().cpu().tolist()
    else:
        patch_indices = [int(value) for value in current_patch_indices]

    stratum_index: Dict[int, List[int]] = {sid: [] for sid in range(grid_size * grid_size)}
    for token_idx, patch_idx in enumerate(patch_indices):
        sid = patch_index_to_stratum_id(patch_idx, grid_size=grid_size, patch_per_row=patch_per_row)
        stratum_index[sid].append(token_idx)
    return stratum_index


def compute_target_keep_count(
    num_visual: int,
    prune_ratio: float,
    min_visual_tokens_after_prune: int,
) -> int:
    """Compute the exact number of tokens to keep after applying the prune budget."""
    if num_visual < 0:
        raise ValueError(f"num_visual must be >= 0, got {num_visual}")
    if not 0.0 <= float(prune_ratio) <= 1.0:
        raise ValueError(f"prune_ratio must be in [0, 1], got {prune_ratio}")
    if min_visual_tokens_after_prune < 0:
        raise ValueError(
            f"min_visual_tokens_after_prune must be >= 0, got {min_visual_tokens_after_prune}"
        )

    raw_num_pruned = math.floor(num_visual * float(prune_ratio))
    max_prunable = max(num_visual - int(min_visual_tokens_after_prune), 0)
    target_prune = min(max(raw_num_pruned, 0), max_prunable)
    return num_visual - target_prune


def allocate_adaptive_stratified_quotas(
    selected_counts: Dict[int, int],
    candidate_counts: Dict[int, int],
    target_keep: int,
    n_low: int,
    grid_size: int,
) -> Tuple[Dict[int, int], Dict[int, float]]:
    """Allocate low-score compensation quotas across strata."""
    num_strata = grid_size * grid_size
    deficits = {
        sid: max(0.0, float(target_keep) / num_strata - int(selected_counts.get(sid, 0)))
        for sid in range(num_strata)
    }
    quotas = {sid: 0 for sid in range(num_strata)}

    if n_low <= 0:
        return quotas, deficits

    capacities = {sid: max(int(candidate_counts.get(sid, 0)), 0) for sid in range(num_strata)}
    total_capacity = sum(capacities.values())
    if total_capacity < n_low:
        raise ValueError(
            f"Not enough candidate capacity to allocate {n_low} low-score tokens, capacity={total_capacity}"
        )

    total_deficit = sum(deficits.values())
    if total_deficit > 0:
        for sid in range(num_strata):
            raw_quota = math.floor(n_low * deficits[sid] / total_deficit)
            quotas[sid] = min(raw_quota, capacities[sid])
    else:
        uniform_base = n_low // num_strata
        for sid in range(num_strata):
            quotas[sid] = min(uniform_base, capacities[sid])

    remaining = n_low - sum(quotas.values())
    if remaining <= 0:
        return quotas, deficits

    if total_deficit > 0:
        fill_order = sorted(range(num_strata), key=lambda sid: (-deficits[sid], sid))
    else:
        fill_order = list(range(num_strata))

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
            raise ValueError("Unable to finish quota allocation despite sufficient total capacity")

    return quotas, deficits


def randomly_select_token_indices(candidates: Sequence[int], n_select: int) -> List[int]:
    """Randomly choose token indices without replacement."""
    if n_select <= 0 or not candidates:
        return []

    candidate_tensor = torch.tensor(list(candidates), dtype=torch.long)
    if n_select >= int(candidate_tensor.numel()):
        return candidate_tensor.tolist()

    order = torch.randperm(candidate_tensor.numel())[:n_select]
    return candidate_tensor.index_select(0, order).tolist()


def _euclidean_patch_distance(
    patch_idx_a: int,
    patch_idx_b: int,
    patch_per_row: int,
) -> float:
    ay, ax = patch_index_to_coords(patch_idx_a, patch_per_row)
    by, bx = patch_index_to_coords(patch_idx_b, patch_per_row)
    return math.sqrt((ay - by) ** 2 + (ax - bx) ** 2)


def select_farthest_token_indices(
    candidate_token_indices: Sequence[int],
    current_patch_indices: Sequence[int] | torch.Tensor,
    selected_patch_indices: Iterable[int],
    n_select: int,
    patch_per_row: int,
) -> List[int]:
    """Greedily choose candidates farthest from the already selected patches."""
    if n_select <= 0 or not candidate_token_indices:
        return []

    if isinstance(current_patch_indices, torch.Tensor):
        patch_indices = current_patch_indices.detach().cpu().tolist()
    else:
        patch_indices = [int(value) for value in current_patch_indices]

    chosen: List[int] = []
    remaining = list(candidate_token_indices)
    reference_patches = [int(patch_idx) for patch_idx in selected_patch_indices]

    while remaining and len(chosen) < n_select:
        if not reference_patches:
            best_token_idx = min(remaining)
        else:
            best_token_idx = max(
                remaining,
                key=lambda token_idx: (
                    min(
                        _euclidean_patch_distance(
                            patch_indices[token_idx],
                            ref_patch_idx,
                            patch_per_row=patch_per_row,
                        )
                        for ref_patch_idx in reference_patches
                    ),
                    -token_idx,
                ),
            )
        chosen.append(best_token_idx)
        reference_patches.append(patch_indices[best_token_idx])
        remaining.remove(best_token_idx)

    return chosen


class SparseVLMAdaptiveStratifiedStrategy(SparseVLMStrategy):
    """SparseVLM scoring with adaptive stratified spatial compensation."""

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

        patch_per_row = int(self.config.get("patch_per_row", 24))
        grid_size = int(self.config.get("grid_size", 6))
        high_ratio = float(self.config.get("high_ratio", 0.7))
        intra_stratum_mode = str(self.config.get("intra_stratum_mode", "random"))
        validate_adaptive_stratified_config(
            initial_v_token_num=int(v_token_num),
            patch_per_row=patch_per_row,
            grid_size=grid_size,
            high_ratio=high_ratio,
            intra_stratum_mode=intra_stratum_mode,
        )

        context["current_patch_indices"] = torch.arange(
            v_token_num,
            device=inputs_embeds.device,
            dtype=torch.long,
        )
        context["initial_v_token_num"] = int(v_token_num)
        return sample_info

    def update_after_prune(self, keep_indices: torch.Tensor, layer_idx: int) -> None:
        del layer_idx
        context = self._require_sample_context()
        current_patch_indices = context["current_patch_indices"]
        keep_indices = keep_indices.to(device=current_patch_indices.device, dtype=torch.long)
        context["current_patch_indices"] = current_patch_indices.index_select(0, keep_indices).detach()

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
            raise ValueError("SparseVLMAdaptiveStratifiedStrategy requires attention weights")

        context = self._require_sample_context()
        current_patch_indices = context["current_patch_indices"]
        if int(current_patch_indices.numel()) != int(v_token_num):
            raise ValueError(
                "current_patch_indices length must match the current visual token count, "
                f"got {current_patch_indices.numel()} vs {v_token_num}"
            )

        grid_size = int(self.config.get("grid_size", 6))
        patch_per_row = int(self.config.get("patch_per_row", 24))
        high_ratio = float(self.config.get("high_ratio", 0.7))
        intra_stratum_mode = str(self.config.get("intra_stratum_mode", "random"))
        validate_adaptive_stratified_config(
            initial_v_token_num=int(context["initial_v_token_num"]),
            patch_per_row=patch_per_row,
            grid_size=grid_size,
            high_ratio=high_ratio,
            intra_stratum_mode=intra_stratum_mode,
        )

        visual_scores = self.compute_importance(
            attn_weights=attn_weights,
            v_token_start=v_token_start,
            v_token_num=v_token_num,
            text_token_start=text_token_start,
            layer_idx=layer_idx,
            device=device,
        )
        prune_ratio = self.get_prune_ratio(layer_idx, visual_scores)
        min_visual_tokens_after_prune = int(self.config.get("min_visual_tokens_after_prune", 16))
        target_keep = compute_target_keep_count(
            num_visual=v_token_num,
            prune_ratio=prune_ratio,
            min_visual_tokens_after_prune=min_visual_tokens_after_prune,
        )
        n_high = math.floor(target_keep * high_ratio)
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
        high_keep_patch_indices = current_patch_indices.index_select(0, high_keep_indices)
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

        low_keep_tensor = torch.tensor(
            low_keep_indices,
            device=visual_scores.device,
            dtype=torch.long,
        ) if low_keep_indices else torch.empty(0, device=visual_scores.device, dtype=torch.long)
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
        pruned_patch_indices = current_patch_indices.index_select(0, pruned_indices.to(device=current_patch_indices.device))

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
            "high_ratio": high_ratio,
            "intra_stratum_mode": intra_stratum_mode,
            "stratum_selected_counts": [selected_counts[sid] for sid in range(grid_size * grid_size)],
            "stratum_candidate_counts": [candidate_counts[sid] for sid in range(grid_size * grid_size)],
            "stratum_deficits": [deficits[sid] for sid in range(grid_size * grid_size)],
            "stratum_quotas": [quotas[sid] for sid in range(grid_size * grid_size)],
        }
        return keep_indices, info
