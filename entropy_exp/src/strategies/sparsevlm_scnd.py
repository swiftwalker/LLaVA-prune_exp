"""SparseVLM SCND: saliency-constrained native DivPrune.

SCND keeps SparseVLM's text-conditioned saliency as the task gate, then uses
global max-min visual diversity inside that saliency constraint. The first
pruning layer can run native DivPrune-style selection (C), later layers can use
boundary-only diversity refinement (B), and any layer can fall back to pure
SparseVLM saliency top-k (S).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch

from .sparsevlm_adaptive_stratified import compute_target_keep_count
from .sparsevlm_diverse_mmr import (
    SparseVLMDiverseMMRStrategy,
    _cosine_distance_matrix,
    _saliency_mass_ratio,
    _selected_mean_pairwise_distance,
)
from .sparsevlm_score_memory import compute_entropy_stats, deterministic_descending_indices, rank_normalize_scores


VALID_LAYER_MODES = {"C", "B", "S"}


def _clamp01(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


class SparseVLMSCNDStrategy(SparseVLMDiverseMMRStrategy):
    """Saliency-constrained native DivPrune strategy."""

    def _layer_mode_map(self) -> Dict[int, str]:
        prune_ratio_map = self.config.get("prune_ratio_map")
        prune_layers = list(prune_ratio_map.keys()) if prune_ratio_map else self.config.get("prune_layers", [])
        if isinstance(prune_layers, int):
            prune_layers = [prune_layers]
        prune_layers = [int(layer) for layer in prune_layers]

        layer_modes = self.config.get("layer_modes")
        if layer_modes is None:
            layer_modes = ["C"] + ["B"] * max(len(prune_layers) - 1, 0)
        if isinstance(layer_modes, str):
            layer_modes = [mode.strip() for mode in layer_modes.split(",") if mode.strip()]
        layer_modes = [str(mode).upper() for mode in layer_modes]

        if len(layer_modes) != len(prune_layers):
            raise ValueError(
                "sparsevlm_scnd.layer_modes must have the same length as prune_layers, "
                f"got modes={layer_modes} and prune_layers={prune_layers}"
            )
        invalid = [mode for mode in layer_modes if mode not in VALID_LAYER_MODES]
        if invalid:
            raise ValueError(f"sparsevlm_scnd.layer_modes only supports {sorted(VALID_LAYER_MODES)}, got {invalid}")
        return dict(zip(prune_layers, layer_modes))

    def _layer_mode(self, layer_idx: int) -> str:
        mode_map = self._layer_mode_map()
        if int(layer_idx) not in mode_map:
            raise ValueError(f"Layer {layer_idx} is missing from sparsevlm_scnd.layer_modes")
        return mode_map[int(layer_idx)]

    def _get_scnd_params(self) -> Dict[str, float | bool | str]:
        seed_ratio_min = float(self.config.get("seed_ratio_min", 0.15))
        seed_ratio_max = float(self.config.get("seed_ratio_max", 0.55))
        seed_pool_multiplier = float(self.config.get("seed_pool_multiplier", 2.0))
        saliency_floor_min = float(self.config.get("saliency_floor_min", 0.65))
        saliency_floor_max = float(self.config.get("saliency_floor_max", 0.90))
        boundary_ratio = float(self.config.get("boundary_ratio", 0.25))
        distance_metric = str(self.config.get("distance_metric", "cosine")).lower()
        saliency_repair = bool(self.config.get("saliency_repair", True))

        if not 0.0 <= seed_ratio_min <= seed_ratio_max <= 1.0:
            raise ValueError(
                "seed_ratio_min and seed_ratio_max must satisfy 0 <= min <= max <= 1, "
                f"got {seed_ratio_min}, {seed_ratio_max}"
            )
        if seed_pool_multiplier < 1.0:
            raise ValueError(f"seed_pool_multiplier must be >= 1, got {seed_pool_multiplier}")
        if not 0.0 <= saliency_floor_min <= saliency_floor_max <= 1.0:
            raise ValueError(
                "saliency_floor_min and saliency_floor_max must satisfy 0 <= min <= max <= 1, "
                f"got {saliency_floor_min}, {saliency_floor_max}"
            )
        if not 0.0 <= boundary_ratio <= 1.0:
            raise ValueError(f"boundary_ratio must be in [0, 1], got {boundary_ratio}")
        if distance_metric != "cosine":
            raise ValueError(f"Only distance_metric=cosine is supported, got {distance_metric!r}")

        return {
            "seed_ratio_min": seed_ratio_min,
            "seed_ratio_max": seed_ratio_max,
            "seed_pool_multiplier": seed_pool_multiplier,
            "saliency_floor_min": saliency_floor_min,
            "saliency_floor_max": saliency_floor_max,
            "boundary_ratio": boundary_ratio,
            "distance_metric": distance_metric,
            "saliency_repair": saliency_repair,
        }

    @staticmethod
    def _score_list(scores: torch.Tensor) -> List[float]:
        return [float(value) for value in scores.detach().cpu().tolist()]

    @staticmethod
    def _ordered_by_distance_then_saliency(
        candidates: List[int],
        gains: List[float],
        saliency_values: List[float],
    ) -> int:
        return min(
            range(len(candidates)),
            key=lambda pos: (-float(gains[pos]), -float(saliency_values[int(candidates[pos])]), int(candidates[pos])),
        )

    def _max_min_select_from_pool(
        self,
        distance: torch.Tensor,
        saliency_score: torch.Tensor,
        pool_indices: List[int],
        select_count: int,
    ) -> Tuple[List[int], List[float]]:
        if select_count <= 0 or not pool_indices:
            return [], []

        saliency_values = self._score_list(saliency_score)
        selected: List[int] = []
        gains: List[float] = []
        ordered_pool = sorted(set(int(idx) for idx in pool_indices), key=lambda idx: (-saliency_values[idx], idx))

        first = ordered_pool[0]
        selected.append(first)
        gains.append(0.0)

        while len(selected) < min(select_count, len(ordered_pool)):
            remaining = [idx for idx in ordered_pool if idx not in selected]
            selected_tensor = torch.tensor(selected, device=distance.device, dtype=torch.long)
            remaining_tensor = torch.tensor(remaining, device=distance.device, dtype=torch.long)
            dist_gains = distance.index_select(0, remaining_tensor).index_select(1, selected_tensor).min(dim=1).values
            gain_values = [float(value) for value in dist_gains.detach().cpu().tolist()]
            best_position = self._ordered_by_distance_then_saliency(remaining, gain_values, saliency_values)
            best_idx = int(remaining[best_position])
            selected.append(best_idx)
            gains.append(float(gain_values[best_position]))

        return selected, gains

    @staticmethod
    def _saliency_sum(saliency_values: List[float], indices: List[int]) -> float:
        return float(sum(max(float(saliency_values[int(idx)]), 0.0) for idx in indices))

    def _max_possible_remaining_saliency(
        self,
        saliency_desc: List[int],
        saliency_values: List[float],
        selected_mask: List[bool],
        extra_idx: int,
        slots: int,
    ) -> float:
        if slots <= 0:
            return 0.0
        total = 0.0
        used = 0
        for idx in saliency_desc:
            if selected_mask[int(idx)] or int(idx) == int(extra_idx):
                continue
            total += max(float(saliency_values[int(idx)]), 0.0)
            used += 1
            if used == slots:
                break
        return total

    def _diversity_contributions(self, distance: torch.Tensor, selected: List[int]) -> Dict[int, float]:
        if len(selected) <= 1:
            return {int(idx): 0.0 for idx in selected}
        selected_tensor = torch.tensor(selected, device=distance.device, dtype=torch.long)
        contributions: Dict[int, float] = {}
        for idx in selected:
            other = selected_tensor[selected_tensor != int(idx)]
            value = distance.index_select(0, torch.tensor([idx], device=distance.device)).index_select(1, other).min()
            contributions[int(idx)] = float(value.item())
        return contributions

    def _repair_saliency_mass(
        self,
        selected: List[int],
        seed_indices: List[int],
        saliency_score: torch.Tensor,
        distance: torch.Tensor,
        saliency_mass_floor: float,
    ) -> Tuple[List[int], List[Dict[str, float | int]]]:
        saliency_values = self._score_list(saliency_score)
        selected_set = set(int(idx) for idx in selected)
        seed_set = set(int(idx) for idx in seed_indices)
        repair_replacements: List[Dict[str, float | int]] = []

        def selected_mass() -> float:
            return self._saliency_sum(saliency_values, list(selected_set))

        while selected_mass() + 1e-8 < float(saliency_mass_floor):
            unselected = [idx for idx in range(len(saliency_values)) if idx not in selected_set]
            if not unselected:
                break
            incoming = max(unselected, key=lambda idx: (saliency_values[idx], -idx))
            contributions = self._diversity_contributions(distance, sorted(selected_set))
            replaceable = [idx for idx in selected_set if idx not in seed_set]
            if not replaceable:
                replaceable = list(selected_set)
            outgoing = min(
                replaceable,
                key=lambda idx: (saliency_values[idx], contributions.get(int(idx), 0.0), -int(idx)),
            )
            if saliency_values[incoming] <= saliency_values[outgoing] + 1e-8:
                break
            selected_set.remove(int(outgoing))
            selected_set.add(int(incoming))
            repair_replacements.append(
                {
                    "out": int(outgoing),
                    "in": int(incoming),
                    "mass_gain": float(saliency_values[incoming] - saliency_values[outgoing]),
                }
            )

        return sorted(selected_set), repair_replacements

    def _saliency_constrained_native_divprune_select(
        self,
        saliency_score: torch.Tensor,
        distance: torch.Tensor,
        target_keep: int,
        entropy_norm: float,
        *,
        seed_ratio_min: float,
        seed_ratio_max: float,
        seed_pool_multiplier: float,
        saliency_floor_min: float,
        saliency_floor_max: float,
        saliency_repair: bool,
    ) -> Dict[str, object]:
        num_visual = int(saliency_score.numel())
        if target_keep <= 0:
            return {
                "keep_indices": torch.empty(0, device=saliency_score.device, dtype=torch.long),
                "seed_ratio": 0.0,
                "seed_count": 0,
                "seed_pool_indices": torch.empty(0, device=saliency_score.device, dtype=torch.long),
                "seed_indices": torch.empty(0, device=saliency_score.device, dtype=torch.long),
                "mmr_selected_order": torch.empty(0, device=saliency_score.device, dtype=torch.long),
                "diversity_gain": torch.empty(0, device=saliency_score.device, dtype=torch.float32),
                "saliency_floor_eta": 0.0,
                "saliency_mass_floor": 0.0,
                "saliency_mass_selected": 0.0,
                "saliency_mass_topk": 0.0,
                "feasible_candidate_count": 0,
                "repair_replacements": [],
            }

        keep_count = min(int(target_keep), num_visual)
        saliency_values = self._score_list(saliency_score)
        saliency_desc = deterministic_descending_indices(saliency_score).detach().cpu().tolist()
        topk_indices = [int(idx) for idx in saliency_desc[:keep_count]]
        saliency_mass_topk = self._saliency_sum(saliency_values, topk_indices)

        entropy_norm = _clamp01(entropy_norm)
        seed_ratio = seed_ratio_min + (seed_ratio_max - seed_ratio_min) * (1.0 - entropy_norm)
        seed_count = min(keep_count, int(math.ceil(float(keep_count) * seed_ratio)))
        if keep_count > 0 and seed_ratio > 0.0:
            seed_count = max(1, seed_count)
        seed_pool_count = min(num_visual, max(seed_count, int(math.ceil(float(seed_count) * seed_pool_multiplier))))
        seed_pool_indices = [int(idx) for idx in saliency_desc[:seed_pool_count]]
        seed_indices, seed_gains = self._max_min_select_from_pool(
            distance=distance,
            saliency_score=saliency_score,
            pool_indices=seed_pool_indices,
            select_count=seed_count,
        )

        eta = saliency_floor_min + (saliency_floor_max - saliency_floor_min) * (1.0 - entropy_norm)
        saliency_mass_floor = float(eta * saliency_mass_topk)
        selected = list(seed_indices)
        selected_mask = [False for _ in range(num_visual)]
        for idx in selected:
            selected_mask[int(idx)] = True
        selected_mass = self._saliency_sum(saliency_values, selected)
        selected_order = list(selected)
        diversity_gains = list(seed_gains)
        feasible_candidate_count = 0

        while len(selected) < keep_count:
            remaining = [idx for idx in range(num_visual) if not selected_mask[idx]]
            if not remaining:
                break
            slots_after_candidate = keep_count - len(selected) - 1
            feasible: List[int] = []
            for idx in remaining:
                possible_mass = (
                    selected_mass
                    + max(saliency_values[int(idx)], 0.0)
                    + self._max_possible_remaining_saliency(
                        saliency_desc=saliency_desc,
                        saliency_values=saliency_values,
                        selected_mask=selected_mask,
                        extra_idx=int(idx),
                        slots=slots_after_candidate,
                    )
                )
                if possible_mass + 1e-8 >= saliency_mass_floor:
                    feasible.append(int(idx))
            feasible_candidate_count += len(feasible)
            candidate_pool = feasible if feasible else remaining

            candidate_tensor = torch.tensor(candidate_pool, device=distance.device, dtype=torch.long)
            if selected:
                selected_tensor = torch.tensor(selected, device=distance.device, dtype=torch.long)
                gains_tensor = distance.index_select(0, candidate_tensor).index_select(1, selected_tensor).min(dim=1).values
                gain_values = [float(value) for value in gains_tensor.detach().cpu().tolist()]
            else:
                gain_values = [0.0 for _ in candidate_pool]
            best_position = self._ordered_by_distance_then_saliency(candidate_pool, gain_values, saliency_values)
            best_idx = int(candidate_pool[best_position])
            selected.append(best_idx)
            selected_mask[best_idx] = True
            selected_mass += max(saliency_values[best_idx], 0.0)
            selected_order.append(best_idx)
            diversity_gains.append(float(gain_values[best_position]))

        repair_replacements: List[Dict[str, float | int]] = []
        if saliency_repair and selected:
            selected, repair_replacements = self._repair_saliency_mass(
                selected=selected,
                seed_indices=seed_indices,
                saliency_score=saliency_score,
                distance=distance,
                saliency_mass_floor=saliency_mass_floor,
            )
            selected_mass = self._saliency_sum(saliency_values, selected)

        keep_indices = torch.tensor(sorted(selected), device=saliency_score.device, dtype=torch.long)
        return {
            "keep_indices": keep_indices,
            "seed_ratio": float(seed_ratio),
            "seed_count": int(seed_count),
            "seed_pool_indices": torch.tensor(seed_pool_indices, device=saliency_score.device, dtype=torch.long),
            "seed_indices": torch.tensor(sorted(seed_indices), device=saliency_score.device, dtype=torch.long),
            "mmr_selected_order": torch.tensor(selected_order, device=saliency_score.device, dtype=torch.long),
            "diversity_gain": torch.tensor(diversity_gains, device=saliency_score.device, dtype=torch.float32),
            "saliency_floor_eta": float(eta),
            "saliency_mass_floor": float(saliency_mass_floor),
            "saliency_mass_selected": float(selected_mass),
            "saliency_mass_topk": float(saliency_mass_topk),
            "feasible_candidate_count": int(feasible_candidate_count),
            "repair_replacements": repair_replacements,
        }

    def _margin_confidence(self, saliency_score: torch.Tensor, target_keep: int) -> float:
        num_visual = int(saliency_score.numel())
        if target_keep <= 0 or target_keep >= num_visual:
            return 1.0
        order = deterministic_descending_indices(saliency_score)
        sorted_scores = saliency_score.index_select(0, order)
        gap = float((sorted_scores[target_keep - 1] - sorted_scores[target_keep]).clamp_min(0.0).item())
        std = float(saliency_score.to(dtype=torch.float32).std(unbiased=False).item())
        if std <= 1e-8:
            return 1.0
        return _clamp01(gap / (std + 1e-8))

    def _boundary_refinement_select(
        self,
        saliency_score: torch.Tensor,
        margin_score: torch.Tensor,
        distance: torch.Tensor,
        target_keep: int,
        entropy_norm: float,
        boundary_ratio: float,
    ) -> Dict[str, object]:
        num_visual = int(saliency_score.numel())
        keep_count = min(max(int(target_keep), 0), num_visual)
        if keep_count <= 0:
            empty = torch.empty(0, device=saliency_score.device, dtype=torch.long)
            return {
                "keep_indices": empty,
                "boundary_count": 0,
                "boundary_core_indices": empty,
                "boundary_pool_indices": empty,
                "boundary_fill_indices": empty,
                "margin_confidence": 1.0,
                "mmr_selected_order": empty,
                "diversity_gain": torch.empty(0, device=saliency_score.device, dtype=torch.float32),
            }

        saliency_values = self._score_list(saliency_score)
        order = [int(idx) for idx in deterministic_descending_indices(saliency_score).detach().cpu().tolist()]
        margin_confidence = self._margin_confidence(margin_score, keep_count)
        raw_boundary = float(boundary_ratio) * float(keep_count) * _clamp01(entropy_norm) * (1.0 - margin_confidence)
        boundary_count = min(keep_count, int(math.ceil(raw_boundary)))

        if boundary_count <= 0:
            keep = torch.tensor(sorted(order[:keep_count]), device=saliency_score.device, dtype=torch.long)
            return {
                "keep_indices": keep,
                "boundary_count": 0,
                "boundary_core_indices": keep,
                "boundary_pool_indices": torch.empty(0, device=saliency_score.device, dtype=torch.long),
                "boundary_fill_indices": torch.empty(0, device=saliency_score.device, dtype=torch.long),
                "margin_confidence": float(margin_confidence),
                "mmr_selected_order": keep,
                "diversity_gain": torch.zeros(int(keep.numel()), device=saliency_score.device, dtype=torch.float32),
            }

        core_count = keep_count - boundary_count
        core = order[:core_count]
        pool_count = min(num_visual - core_count, max(boundary_count, boundary_count * 2))
        boundary_pool = order[core_count:core_count + pool_count]
        selected = list(core)
        selected_set = set(selected)
        fills: List[int] = []
        fill_gains: List[float] = []

        while len(fills) < boundary_count and boundary_pool:
            candidates = [idx for idx in boundary_pool if idx not in selected_set]
            if not candidates:
                break
            candidate_tensor = torch.tensor(candidates, device=distance.device, dtype=torch.long)
            if selected:
                selected_tensor = torch.tensor(selected, device=distance.device, dtype=torch.long)
                gains_tensor = distance.index_select(0, candidate_tensor).index_select(1, selected_tensor).min(dim=1).values
                gain_values = [float(value) for value in gains_tensor.detach().cpu().tolist()]
            else:
                gain_values = [0.0 for _ in candidates]
            best_position = self._ordered_by_distance_then_saliency(candidates, gain_values, saliency_values)
            best_idx = int(candidates[best_position])
            selected.append(best_idx)
            selected_set.add(best_idx)
            fills.append(best_idx)
            fill_gains.append(float(gain_values[best_position]))

        if len(selected) < keep_count:
            for idx in order:
                if idx in selected_set:
                    continue
                selected.append(idx)
                selected_set.add(idx)
                if len(selected) == keep_count:
                    break

        keep = torch.tensor(sorted(selected), device=saliency_score.device, dtype=torch.long)
        return {
            "keep_indices": keep,
            "boundary_count": int(boundary_count),
            "boundary_core_indices": torch.tensor(sorted(core), device=saliency_score.device, dtype=torch.long),
            "boundary_pool_indices": torch.tensor(boundary_pool, device=saliency_score.device, dtype=torch.long),
            "boundary_fill_indices": torch.tensor(sorted(fills), device=saliency_score.device, dtype=torch.long),
            "margin_confidence": float(margin_confidence),
            "mmr_selected_order": torch.tensor(selected, device=saliency_score.device, dtype=torch.long),
            "diversity_gain": torch.tensor([0.0] * len(core) + fill_gains, device=saliency_score.device, dtype=torch.float32),
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
            raise ValueError("SparseVLMSCNDStrategy requires attention weights")

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
            raise ValueError("SparseVLMSCNDStrategy requires current_visual_embeds for C/B layers")
        if int(current_visual_embeds.shape[0]) != int(v_token_num):
            raise ValueError(
                "current_visual_embeds length must match current visual token count, "
                f"got {current_visual_embeds.shape[0]} vs {v_token_num}"
            )

        params = self._get_scnd_params()
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
        prune_ratio = self.get_prune_ratio(layer_idx, visual_scores)
        min_visual_tokens_after_prune = int(self.config.get("min_visual_tokens_after_prune", 16))
        target_keep = compute_target_keep_count(
            num_visual=v_token_num,
            prune_ratio=prune_ratio,
            min_visual_tokens_after_prune=min_visual_tokens_after_prune,
        )
        distance = _cosine_distance_matrix(current_visual_embeds).to(device=mixed_score.device)

        if mode == "C":
            selection = self._saliency_constrained_native_divprune_select(
                saliency_score=mixed_score,
                distance=distance,
                target_keep=target_keep,
                entropy_norm=entropy_norm,
                seed_ratio_min=float(params["seed_ratio_min"]),
                seed_ratio_max=float(params["seed_ratio_max"]),
                seed_pool_multiplier=float(params["seed_pool_multiplier"]),
                saliency_floor_min=float(params["saliency_floor_min"]),
                saliency_floor_max=float(params["saliency_floor_max"]),
                saliency_repair=bool(params["saliency_repair"]),
            )
            selection_rule = "saliency_constrained_native_divprune"
        elif mode == "B":
            selection = self._boundary_refinement_select(
                saliency_score=mixed_score,
                margin_score=visual_scores,
                distance=distance,
                target_keep=target_keep,
                entropy_norm=entropy_norm,
                boundary_ratio=float(params["boundary_ratio"]),
            )
            selection_rule = "boundary_only_diversity_refinement"
        else:  # defensive; mode validation happens in _layer_mode.
            raise ValueError(f"Unsupported sparsevlm_scnd layer mode: {mode}")

        keep_indices = selection["keep_indices"]
        if int(keep_indices.numel()) != int(target_keep):
            raise AssertionError(
                "sparsevlm_scnd selection produced the wrong keep count, "
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
            "layer_strategy_effective": "sparsevlm_scnd",
            "selection_rule": selection_rule,
            "distance_metric": "cosine",
            "saliency_entropy": entropy_raw,
            "saliency_entropy_norm": entropy_norm,
            "seed_ratio_min": params["seed_ratio_min"],
            "seed_ratio_max": params["seed_ratio_max"],
            "seed_pool_multiplier": params["seed_pool_multiplier"],
            "saliency_floor_min": params["saliency_floor_min"],
            "saliency_floor_max": params["saliency_floor_max"],
            "boundary_ratio": params["boundary_ratio"],
            "saliency_repair": params["saliency_repair"],
            "mean_selected_pairwise_distance": _selected_mean_pairwise_distance(distance, keep_indices),
            "retained_saliency_mass_ratio": _saliency_mass_ratio(visual_scores, keep_indices),
            "global_prune_step": memory["global_prune_step"],
            "global_current_weight": memory["global_current_weight"],
            "global_ema_decay": memory["global_ema_decay"],
            "global_use_ema": memory["global_use_ema"],
            "use_score_memory": memory["use_score_memory"],
            "global_saliency_ema": memory["global_saliency_ema"].detach().cpu().numpy(),
        }

        for key, value in selection.items():
            if key == "keep_indices":
                continue
            if torch.is_tensor(value):
                info[key] = value.detach().cpu().numpy()
            else:
                info[key] = value

        return keep_indices, info
