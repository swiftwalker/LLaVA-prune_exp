"""Fast-SCND: low-cost saliency-constrained native diversity pruning."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch

from .sparsevlm_adaptive_stratified import compute_target_keep_count
from .sparsevlm_score_memory import (
    SparseVLMScoreMemoryStrategy,
    compute_entropy_stats,
    deterministic_descending_indices,
    deterministic_topk_indices,
    rank_normalize_scores,
)


VALID_LAYER_MODES = {"F", "T", "S"}


def _clamp01(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def _as_float_list(values: torch.Tensor) -> List[float]:
    return [float(item) for item in values.detach().cpu().tolist()]


def _safe_rank_normalize(values: torch.Tensor) -> torch.Tensor:
    if int(values.numel()) == 0:
        return values.to(dtype=torch.float32)
    values_f = values.to(dtype=torch.float32)
    if float((values_f.max() - values_f.min()).item()) <= 1e-8:
        return torch.zeros_like(values_f)
    return rank_normalize_scores(values_f)


def _projected_mean_pairwise_distance(projected: torch.Tensor, keep_indices: torch.Tensor) -> float:
    selected_count = int(keep_indices.numel())
    if selected_count <= 1:
        return 0.0
    selected = projected.index_select(0, keep_indices.to(device=projected.device))
    distance = torch.clamp((1.0 - selected @ selected.transpose(0, 1)) / 2.0, min=0.0, max=1.0)
    return float((distance.sum() / (selected_count * (selected_count - 1))).item())


def _saliency_mass_ratio(scores: torch.Tensor, keep_indices: torch.Tensor) -> float:
    positive = scores.to(dtype=torch.float64).clamp_min(0.0)
    total = positive.sum()
    if float(total.item()) <= 0.0:
        return 0.0
    kept = positive.index_select(0, keep_indices.to(device=positive.device)).sum()
    return float((kept / total).item())


class SparseVLMFastSCNDStrategy(SparseVLMScoreMemoryStrategy):
    """Fast saliency-constrained native DivPrune with cached tie-break refinement."""

    def __init__(self, config: dict):
        super().__init__(config)
        self._projection_cache: Dict[Tuple[int, int], torch.Tensor] = {}

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
        context["fast_scnd_diversity_gain"] = torch.zeros(v_token_num, device=inputs_embeds.device, dtype=torch.float32)
        return sample_info

    def _layer_mode_map(self) -> Dict[int, str]:
        prune_ratio_map = self.config.get("prune_ratio_map")
        prune_layers = list(prune_ratio_map.keys()) if prune_ratio_map else self.config.get("prune_layers", [])
        if isinstance(prune_layers, int):
            prune_layers = [prune_layers]
        prune_layers = [int(layer) for layer in prune_layers]

        layer_modes = self.config.get("layer_modes")
        if layer_modes is None:
            if len(prune_layers) == 0:
                layer_modes = []
            elif len(prune_layers) == 1:
                layer_modes = ["F"]
            elif len(prune_layers) == 2:
                layer_modes = ["F", "T"]
            else:
                layer_modes = ["F", "T"] + ["S"] * (len(prune_layers) - 2)
        if isinstance(layer_modes, str):
            layer_modes = [mode.strip() for mode in layer_modes.split(",") if mode.strip()]
        layer_modes = [str(mode).upper() for mode in layer_modes]

        if len(layer_modes) != len(prune_layers):
            raise ValueError(
                "sparsevlm_fast_scnd.layer_modes must have the same length as prune_layers, "
                f"got modes={layer_modes} and prune_layers={prune_layers}"
            )
        invalid = [mode for mode in layer_modes if mode not in VALID_LAYER_MODES]
        if invalid:
            raise ValueError(
                f"sparsevlm_fast_scnd.layer_modes only supports {sorted(VALID_LAYER_MODES)}, got {invalid}"
            )
        return dict(zip(prune_layers, layer_modes))

    def _layer_mode(self, layer_idx: int) -> str:
        mode_map = self._layer_mode_map()
        if int(layer_idx) not in mode_map:
            raise ValueError(f"Layer {layer_idx} is missing from sparsevlm_fast_scnd.layer_modes")
        return mode_map[int(layer_idx)]

    def _params(self) -> Dict[str, float | int | str]:
        core_ratio_min = float(self.config.get("core_ratio_min", 0.10))
        core_ratio_max = float(self.config.get("core_ratio_max", 0.35))
        saliency_pool_multiplier = float(self.config.get("saliency_pool_multiplier", 1.5))
        reservoir_multiplier = float(self.config.get("reservoir_multiplier", 0.5))
        reservoir_rank_multiplier = float(self.config.get("reservoir_rank_multiplier", 4.0))
        candidate_cap_multiplier = float(self.config.get("candidate_cap_multiplier", 2.0))
        projection_dim = int(self.config.get("projection_dim", 64))
        greedy_steps = int(self.config.get("greedy_steps", 16))
        tie_break_band_ratio = float(self.config.get("tie_break_band_ratio", 0.10))
        tie_break_band_max = int(self.config.get("tie_break_band_max", 32))
        cached_diversity_weight = float(self.config.get("cached_diversity_weight", 0.05))
        distance_metric = str(self.config.get("distance_metric", "cosine")).lower()

        if not 0.0 <= core_ratio_min <= core_ratio_max <= 1.0:
            raise ValueError(
                "core_ratio_min/core_ratio_max must satisfy 0 <= min <= max <= 1, "
                f"got {core_ratio_min}, {core_ratio_max}"
            )
        for name, value in (
            ("saliency_pool_multiplier", saliency_pool_multiplier),
            ("reservoir_multiplier", reservoir_multiplier),
            ("reservoir_rank_multiplier", reservoir_rank_multiplier),
            ("candidate_cap_multiplier", candidate_cap_multiplier),
        ):
            if value < 0.0:
                raise ValueError(f"{name} must be >= 0, got {value}")
        if saliency_pool_multiplier < 1.0:
            raise ValueError(f"saliency_pool_multiplier must be >= 1, got {saliency_pool_multiplier}")
        if candidate_cap_multiplier < 1.0:
            raise ValueError(f"candidate_cap_multiplier must be >= 1, got {candidate_cap_multiplier}")
        if projection_dim <= 0:
            raise ValueError(f"projection_dim must be > 0, got {projection_dim}")
        if greedy_steps < 0:
            raise ValueError(f"greedy_steps must be >= 0, got {greedy_steps}")
        if not 0.0 <= tie_break_band_ratio <= 1.0:
            raise ValueError(f"tie_break_band_ratio must be in [0, 1], got {tie_break_band_ratio}")
        if tie_break_band_max < 0:
            raise ValueError(f"tie_break_band_max must be >= 0, got {tie_break_band_max}")
        if cached_diversity_weight < 0.0:
            raise ValueError(f"cached_diversity_weight must be >= 0, got {cached_diversity_weight}")
        if distance_metric != "cosine":
            raise ValueError(f"Only distance_metric=cosine is supported, got {distance_metric!r}")

        return {
            "core_ratio_min": core_ratio_min,
            "core_ratio_max": core_ratio_max,
            "saliency_pool_multiplier": saliency_pool_multiplier,
            "reservoir_multiplier": reservoir_multiplier,
            "reservoir_rank_multiplier": reservoir_rank_multiplier,
            "candidate_cap_multiplier": candidate_cap_multiplier,
            "projection_dim": projection_dim,
            "greedy_steps": greedy_steps,
            "tie_break_band_ratio": tie_break_band_ratio,
            "tie_break_band_max": tie_break_band_max,
            "cached_diversity_weight": cached_diversity_weight,
            "distance_metric": distance_metric,
        }

    def _sv_keep_mask(
        self,
        visual_scores: torch.Tensor,
        current_rank_score: torch.Tensor,
        current_patch_indices: torch.Tensor,
        layer_idx: int,
        v_token_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
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
        info = {
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
            "target_keep": target_keep,
            "strategy_keep_high": int(keep_indices.numel()),
            "strategy_keep_low": 0,
            "layer_mode": "S",
            "layer_strategy_effective": "sparsevlm",
            "global_use_ema": False,
            "use_score_memory": False,
        }
        return keep_indices, pruned_indices, info

    def _projection_matrix(self, input_dim: int, projection_dim: int, device: torch.device) -> torch.Tensor:
        key = (int(input_dim), int(projection_dim))
        if key not in self._projection_cache:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(20260531 + input_dim * 17 + projection_dim)
            matrix = torch.randn(input_dim, projection_dim, generator=generator, dtype=torch.float32)
            matrix = matrix / math.sqrt(float(projection_dim))
            self._projection_cache[key] = matrix
        return self._projection_cache[key].to(device=device, dtype=torch.float32)

    def _project_visual_embeds(self, current_visual_embeds: torch.Tensor, projection_dim: int) -> torch.Tensor:
        embeds = current_visual_embeds.to(dtype=torch.float32)
        matrix = self._projection_matrix(int(embeds.shape[-1]), int(projection_dim), embeds.device)
        projected = embeds @ matrix
        return torch.nn.functional.normalize(projected, dim=-1, eps=1e-6)

    @staticmethod
    def _ordered_subset(scores: torch.Tensor, indices: List[int]) -> List[int]:
        score_values = _as_float_list(scores)
        return sorted(set(int(idx) for idx in indices), key=lambda idx: (-float(score_values[int(idx)]), int(idx)))

    @staticmethod
    def _distance_to_vector(projected: torch.Tensor, indices: List[int], vector: torch.Tensor) -> torch.Tensor:
        if not indices:
            return torch.empty(0, device=projected.device, dtype=torch.float32)
        idx_tensor = torch.tensor(indices, device=projected.device, dtype=torch.long)
        values = projected.index_select(0, idx_tensor) @ vector.to(device=projected.device, dtype=torch.float32)
        return torch.clamp((1.0 - values) / 2.0, min=0.0, max=1.0)

    def _build_candidate_pool(
        self,
        mixed_score: torch.Tensor,
        projected: torch.Tensor,
        core_indices: List[int],
        target_keep: int,
        params: Dict[str, float | int | str],
    ) -> Dict[str, object]:
        num_visual = int(mixed_score.numel())
        saliency_order = [int(idx) for idx in deterministic_descending_indices(mixed_score).detach().cpu().tolist()]
        saliency_pool_count = min(
            num_visual,
            max(target_keep, int(math.ceil(float(target_keep) * float(params["saliency_pool_multiplier"])))),
        )
        saliency_pool = saliency_order[:saliency_pool_count]

        reservoir_rank_count = min(
            num_visual,
            max(target_keep, int(math.ceil(float(target_keep) * float(params["reservoir_rank_multiplier"])))),
        )
        reservoir_count = min(
            reservoir_rank_count,
            int(math.ceil(float(target_keep) * float(params["reservoir_multiplier"]))),
        )
        reservoir_source = saliency_order[:reservoir_rank_count]
        if core_indices:
            core_tensor = torch.tensor(core_indices, device=projected.device, dtype=torch.long)
            core_center = projected.index_select(0, core_tensor).mean(dim=0)
            core_center = torch.nn.functional.normalize(core_center, dim=0, eps=1e-6)
            distances = self._distance_to_vector(projected, reservoir_source, core_center)
            dist_values = _as_float_list(distances)
            score_values = _as_float_list(mixed_score)
            reservoir_order = sorted(
                range(len(reservoir_source)),
                key=lambda pos: (-float(dist_values[pos]), -float(score_values[reservoir_source[pos]]), reservoir_source[pos]),
            )
            reservoir = [int(reservoir_source[pos]) for pos in reservoir_order[:reservoir_count]]
        else:
            reservoir = []

        candidate_cap = min(
            num_visual,
            max(target_keep, int(math.ceil(float(target_keep) * float(params["candidate_cap_multiplier"])))),
        )
        candidate_pool: List[int] = []
        seen: set[int] = set()
        for group in (core_indices, saliency_pool, reservoir):
            for idx in group:
                idx = int(idx)
                if idx in seen:
                    continue
                candidate_pool.append(idx)
                seen.add(idx)
                if len(candidate_pool) >= candidate_cap:
                    break
            if len(candidate_pool) >= candidate_cap:
                break

        return {
            "saliency_order": saliency_order,
            "saliency_pool_indices": saliency_pool,
            "reservoir_indices": reservoir,
            "candidate_pool_indices": candidate_pool,
            "reservoir_source_count": reservoir_rank_count,
        }

    def _fast_select(
        self,
        mixed_score: torch.Tensor,
        projected: torch.Tensor,
        target_keep: int,
        entropy_norm: float,
        params: Dict[str, float | int | str],
    ) -> Dict[str, object]:
        num_visual = int(mixed_score.numel())
        keep_count = min(max(int(target_keep), 0), num_visual)
        if keep_count <= 0:
            empty = torch.empty(0, device=mixed_score.device, dtype=torch.long)
            return {
                "keep_indices": empty,
                "core_ratio": 0.0,
                "core_indices": empty,
                "saliency_pool_indices": empty,
                "reservoir_indices": empty,
                "candidate_pool_indices": empty,
                "mmr_selected_order": empty,
                "diversity_gain": torch.empty(0, device=mixed_score.device, dtype=torch.float32),
                "cached_diversity_gain": torch.zeros(num_visual, device=mixed_score.device, dtype=torch.float32),
                "greedy_steps_used": 0,
                "reservoir_source_count": 0,
            }

        entropy_norm = _clamp01(entropy_norm)
        core_ratio = float(params["core_ratio_min"]) + (
            float(params["core_ratio_max"]) - float(params["core_ratio_min"])
        ) * (1.0 - entropy_norm)
        core_count = min(keep_count, max(1, int(math.ceil(float(keep_count) * core_ratio))))
        saliency_order = [int(idx) for idx in deterministic_descending_indices(mixed_score).detach().cpu().tolist()]
        core_indices = saliency_order[:core_count]
        pool = self._build_candidate_pool(
            mixed_score=mixed_score,
            projected=projected,
            core_indices=core_indices,
            target_keep=keep_count,
            params=params,
        )
        candidate_pool = [int(idx) for idx in pool["candidate_pool_indices"]]
        score_values = _as_float_list(mixed_score)
        selected: List[int] = list(core_indices)
        selected_set = set(selected)
        selected_order = list(selected)
        selected_gains: List[float] = [0.0 for _ in selected]
        cached_gain = torch.zeros(num_visual, device=mixed_score.device, dtype=torch.float32)

        candidate_remaining = [idx for idx in candidate_pool if idx not in selected_set]
        greedy_limit = min(int(params["greedy_steps"]), keep_count - len(selected))
        greedy_steps_used = 0

        if candidate_remaining and selected and greedy_limit > 0:
            candidate_tensor = torch.tensor(candidate_remaining, device=projected.device, dtype=torch.long)
            selected_tensor = torch.tensor(selected, device=projected.device, dtype=torch.long)
            min_distance = torch.clamp(
                (1.0 - projected.index_select(0, candidate_tensor) @ projected.index_select(0, selected_tensor).T) / 2.0,
                min=0.0,
                max=1.0,
            ).min(dim=1).values
            active_candidates = list(candidate_remaining)

            while greedy_steps_used < greedy_limit and active_candidates and len(selected) < keep_count:
                gains = _as_float_list(min_distance)
                best_pos = min(
                    range(len(active_candidates)),
                    key=lambda pos: (-float(gains[pos]), -float(score_values[active_candidates[pos]]), active_candidates[pos]),
                )
                best_idx = int(active_candidates[best_pos])
                best_gain = float(gains[best_pos])
                selected.append(best_idx)
                selected_set.add(best_idx)
                selected_order.append(best_idx)
                selected_gains.append(best_gain)
                cached_gain[best_idx] = best_gain
                greedy_steps_used += 1

                del active_candidates[best_pos]
                if active_candidates:
                    keep_mask = torch.ones(int(min_distance.numel()), device=min_distance.device, dtype=torch.bool)
                    keep_mask[best_pos] = False
                    min_distance = min_distance[keep_mask]
                    active_tensor = torch.tensor(active_candidates, device=projected.device, dtype=torch.long)
                    best_vec = projected.index_select(0, torch.tensor([best_idx], device=projected.device)).T
                    dist_to_best = torch.clamp(
                        (1.0 - projected.index_select(0, active_tensor) @ best_vec).flatten() / 2.0,
                        min=0.0,
                        max=1.0,
                    )
                    min_distance = torch.minimum(min_distance, dist_to_best)

        if len(selected) < keep_count:
            for idx in saliency_order:
                if idx in selected_set:
                    continue
                selected.append(idx)
                selected_set.add(idx)
                selected_order.append(idx)
                selected_gains.append(float(cached_gain[idx].item()))
                if len(selected) == keep_count:
                    break

        keep_indices = torch.tensor(sorted(selected), device=mixed_score.device, dtype=torch.long)
        selected_order_tensor = torch.tensor(selected_order, device=mixed_score.device, dtype=torch.long)
        gain_tensor = torch.tensor(selected_gains, device=mixed_score.device, dtype=torch.float32)
        return {
            "keep_indices": keep_indices,
            "core_ratio": float(core_ratio),
            "core_indices": torch.tensor(sorted(core_indices), device=mixed_score.device, dtype=torch.long),
            "saliency_pool_indices": torch.tensor(pool["saliency_pool_indices"], device=mixed_score.device, dtype=torch.long),
            "reservoir_indices": torch.tensor(pool["reservoir_indices"], device=mixed_score.device, dtype=torch.long),
            "candidate_pool_indices": torch.tensor(candidate_pool, device=mixed_score.device, dtype=torch.long),
            "mmr_selected_order": selected_order_tensor,
            "diversity_gain": gain_tensor,
            "cached_diversity_gain": cached_gain,
            "greedy_steps_used": int(greedy_steps_used),
            "reservoir_source_count": int(pool["reservoir_source_count"]),
        }

    def _alive_cached_gain(
        self,
        context: Dict[str, object],
        current_patch_indices: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        cache = context.get("fast_scnd_diversity_gain")
        if cache is None:
            return torch.zeros(int(current_patch_indices.numel()), device=device, dtype=torch.float32)
        patch_indices = current_patch_indices.to(device=cache.device, dtype=torch.long)
        return cache.index_select(0, patch_indices).to(device=device, dtype=torch.float32)

    def _store_cached_gain(
        self,
        context: Dict[str, object],
        current_patch_indices: torch.Tensor,
        selection: Dict[str, object],
    ) -> None:
        cache = context.get("fast_scnd_diversity_gain")
        if cache is None:
            return
        order = selection.get("mmr_selected_order")
        gains = selection.get("diversity_gain")
        if not torch.is_tensor(order) or not torch.is_tensor(gains) or int(order.numel()) == 0:
            return
        patch_indices = current_patch_indices.to(device=cache.device, dtype=torch.long)
        order_cpu = order.to(device=patch_indices.device, dtype=torch.long)
        selected_patch_indices = patch_indices.index_select(0, order_cpu)
        cache[selected_patch_indices] = gains.to(device=cache.device, dtype=torch.float32)

    def _tie_break_select(
        self,
        mixed_score: torch.Tensor,
        cached_gain: torch.Tensor,
        target_keep: int,
        params: Dict[str, float | int | str],
    ) -> Dict[str, object]:
        num_visual = int(mixed_score.numel())
        keep_count = min(max(int(target_keep), 0), num_visual)
        if keep_count <= 0:
            empty = torch.empty(0, device=mixed_score.device, dtype=torch.long)
            return {
                "keep_indices": empty,
                "core_indices": empty,
                "boundary_pool_indices": empty,
                "boundary_fill_indices": empty,
                "candidate_pool_indices": empty,
                "cached_diversity_gain": cached_gain,
                "tie_break_band_count": 0,
            }

        saliency_order = [int(idx) for idx in deterministic_descending_indices(mixed_score).detach().cpu().tolist()]
        band_count = min(
            keep_count,
            int(params["tie_break_band_max"]),
            int(math.ceil(float(keep_count) * float(params["tie_break_band_ratio"]))),
        )
        if band_count <= 0:
            keep = torch.tensor(sorted(saliency_order[:keep_count]), device=mixed_score.device, dtype=torch.long)
            return {
                "keep_indices": keep,
                "core_indices": keep,
                "boundary_pool_indices": torch.empty(0, device=mixed_score.device, dtype=torch.long),
                "boundary_fill_indices": torch.empty(0, device=mixed_score.device, dtype=torch.long),
                "candidate_pool_indices": torch.empty(0, device=mixed_score.device, dtype=torch.long),
                "cached_diversity_gain": cached_gain,
                "tie_break_band_count": 0,
            }

        core_count = keep_count - band_count
        core = saliency_order[:core_count]
        boundary_pool_count = min(num_visual - core_count, max(band_count, band_count * 2))
        boundary_pool = saliency_order[core_count:core_count + boundary_pool_count]
        gain_rank = _safe_rank_normalize(cached_gain)
        tie_score = mixed_score.to(dtype=torch.float32) + float(params["cached_diversity_weight"]) * gain_rank
        fill = self._ordered_subset(tie_score, boundary_pool)[:band_count]

        selected_set = set(core)
        for idx in fill:
            selected_set.add(int(idx))
        if len(selected_set) < keep_count:
            for idx in saliency_order:
                selected_set.add(int(idx))
                if len(selected_set) == keep_count:
                    break

        keep = torch.tensor(sorted(selected_set), device=mixed_score.device, dtype=torch.long)
        return {
            "keep_indices": keep,
            "core_indices": torch.tensor(sorted(core), device=mixed_score.device, dtype=torch.long),
            "boundary_pool_indices": torch.tensor(boundary_pool, device=mixed_score.device, dtype=torch.long),
            "boundary_fill_indices": torch.tensor(sorted(fill), device=mixed_score.device, dtype=torch.long),
            "candidate_pool_indices": torch.tensor(boundary_pool, device=mixed_score.device, dtype=torch.long),
            "cached_diversity_gain": cached_gain,
            "tie_break_band_count": int(band_count),
            "tie_break_scores": tie_score,
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
            raise ValueError("SparseVLMFastSCNDStrategy requires attention weights")

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
            raise ValueError("SparseVLMFastSCNDStrategy requires current_visual_embeds for F/T layers")
        if int(current_visual_embeds.shape[0]) != int(v_token_num):
            raise ValueError(
                "current_visual_embeds length must match current visual token count, "
                f"got {current_visual_embeds.shape[0]} vs {v_token_num}"
            )

        params = self._params()
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
        projected = self._project_visual_embeds(current_visual_embeds, int(params["projection_dim"]))

        if mode == "F":
            selection = self._fast_select(
                mixed_score=mixed_score,
                projected=projected,
                target_keep=target_keep,
                entropy_norm=entropy_norm,
                params=params,
            )
            self._store_cached_gain(context, current_patch_indices, selection)
            selection_rule = "fast_scnd_micro_greedy"
        elif mode == "T":
            cached_gain = self._alive_cached_gain(context, current_patch_indices, mixed_score.device)
            selection = self._tie_break_select(
                mixed_score=mixed_score,
                cached_gain=cached_gain,
                target_keep=target_keep,
                params=params,
            )
            selection_rule = "fast_scnd_cached_tie_break"
        else:
            raise ValueError(f"Unsupported sparsevlm_fast_scnd layer mode: {mode}")

        keep_indices = selection["keep_indices"]
        if int(keep_indices.numel()) != int(target_keep):
            raise AssertionError(
                "sparsevlm_fast_scnd selection produced the wrong keep count, "
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
            "layer_strategy_effective": "sparsevlm_fast_scnd",
            "selection_rule": selection_rule,
            "distance_metric": "cosine",
            "saliency_entropy": entropy_raw,
            "saliency_entropy_norm": entropy_norm,
            "core_ratio_min": params["core_ratio_min"],
            "core_ratio_max": params["core_ratio_max"],
            "saliency_pool_multiplier": params["saliency_pool_multiplier"],
            "reservoir_multiplier": params["reservoir_multiplier"],
            "reservoir_rank_multiplier": params["reservoir_rank_multiplier"],
            "candidate_cap_multiplier": params["candidate_cap_multiplier"],
            "projection_dim": params["projection_dim"],
            "greedy_steps": params["greedy_steps"],
            "tie_break_band_ratio": params["tie_break_band_ratio"],
            "tie_break_band_max": params["tie_break_band_max"],
            "cached_diversity_weight": params["cached_diversity_weight"],
            "mean_selected_pairwise_distance": _projected_mean_pairwise_distance(projected, keep_indices),
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
