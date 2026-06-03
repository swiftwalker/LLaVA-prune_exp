"""Budget-aware Candidate-SCND pruning.

This strategy keeps SCND's saliency-constrained native diversity objective, but
restricts the expensive distance computation to a target-aware saliency
candidate pool. The first pruning layer can run Candidate-C (C), later layers
can run cached boundary refinement (B), and any layer can fall back to pure
SparseVLM saliency (S).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from .sparsevlm_adaptive_stratified import compute_target_keep_count
from .sparsevlm_fast_scnd import (
    SparseVLMFastSCNDStrategy,
    _as_float_list,
    _projected_mean_pairwise_distance,
    _safe_rank_normalize,
    _saliency_mass_ratio,
)
from .sparsevlm_score_memory import compute_entropy_stats, deterministic_descending_indices, rank_normalize_scores


VALID_LAYER_MODES = {"C", "B", "S"}
MODE_ALIASES = {
    "C": "C",
    "CAND": "C",
    "CANDIDATE": "C",
    "B": "B",
    "BLITE": "B",
    "B-LITE": "B",
    "S": "S",
}
KNOWN_TARGET_RATIOS = {
    (0.4791667, 0.3333333, 0.45): "retain192",
    (0.4739583, 0.6369637, 0.6727273): "retain128",
    (0.8854167, 0.5454545, 0.4333333): "retain64",
}
AUTO_LAYER_POLICY = {
    "retain192": ["C", "B", "S"],
    "retain128": ["C", "B", "B"],
    "retain64": ["C", "B", "B"],
}
PROFILE_VERSION = "v3"


@dataclass(frozen=True)
class BudgetProfile:
    alpha: float
    beta: float
    rho: float
    candidate_cap: float
    candidate_cap_fraction: Optional[float]
    projection_dim: int
    step_ratio: float
    gate_floor: float
    saliency_floor_min: float
    saliency_floor_max: float
    boundary_ratio: float
    cached_diversity_weight: float
    reservoir_min_fraction: float
    global_reservoir_fraction: float
    saliency_core_ratio: float
    boundary_pool_multiplier: float
    blite_gate_min: float


TARGET_PROFILES = {
    "retain192": BudgetProfile(
        alpha=1.25,
        beta=0.50,
        rho=4.0,
        candidate_cap=1.50,
        candidate_cap_fraction=0.90,
        projection_dim=128,
        step_ratio=0.35,
        gate_floor=0.25,
        saliency_floor_min=0.68,
        saliency_floor_max=0.82,
        boundary_ratio=0.14,
        cached_diversity_weight=0.03,
        reservoir_min_fraction=0.12,
        global_reservoir_fraction=0.20,
        saliency_core_ratio=0.30,
        boundary_pool_multiplier=2.5,
        blite_gate_min=0.05,
    ),
    "retain128": BudgetProfile(
        alpha=1.50,
        beta=0.75,
        rho=5.0,
        candidate_cap=1.75,
        candidate_cap_fraction=0.95,
        projection_dim=128,
        step_ratio=0.45,
        gate_floor=0.25,
        saliency_floor_min=0.65,
        saliency_floor_max=0.80,
        boundary_ratio=0.20,
        cached_diversity_weight=0.04,
        reservoir_min_fraction=0.18,
        global_reservoir_fraction=0.25,
        saliency_core_ratio=0.25,
        boundary_pool_multiplier=2.5,
        blite_gate_min=0.05,
    ),
    "retain64": BudgetProfile(
        alpha=1.75,
        beta=1.25,
        rho=6.0,
        candidate_cap=3.0,
        candidate_cap_fraction=None,
        projection_dim=128,
        step_ratio=0.85,
        gate_floor=0.35,
        saliency_floor_min=0.50,
        saliency_floor_max=0.72,
        boundary_ratio=0.25,
        cached_diversity_weight=0.05,
        reservoir_min_fraction=0.30,
        global_reservoir_fraction=0.0,
        saliency_core_ratio=0.0,
        boundary_pool_multiplier=2.0,
        blite_gate_min=0.05,
    ),
}


def _clamp01(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def _empty_long(device: torch.device) -> torch.Tensor:
    return torch.empty(0, device=device, dtype=torch.long)


def _ratio_key(values: object) -> tuple[float, ...] | None:
    if isinstance(values, (int, float)):
        return (round(float(values), 7),)
    if not isinstance(values, (list, tuple)):
        return None
    try:
        return tuple(round(float(item), 7) for item in values)
    except (TypeError, ValueError):
        return None


class SparseVLMBudgetCandidateSCNDStrategy(SparseVLMFastSCNDStrategy):
    """Budget-aware saliency-constrained native diversity strategy."""

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
        context["budget_candidate_scnd_diversity_gain"] = torch.zeros(
            v_token_num, device=inputs_embeds.device, dtype=torch.float32
        )
        return sample_info

    def _prune_layers(self) -> List[int]:
        prune_ratio_map = self.config.get("prune_ratio_map")
        prune_layers = list(prune_ratio_map.keys()) if prune_ratio_map else self.config.get("prune_layers", [])
        if isinstance(prune_layers, int):
            prune_layers = [prune_layers]
        return [int(layer) for layer in prune_layers]

    def _configured_ratio_key(self) -> tuple[float, ...] | None:
        prune_ratio_map = self.config.get("prune_ratio_map")
        if prune_ratio_map:
            values = [float(prune_ratio_map[layer]) for layer in sorted(prune_ratio_map)]
            return _ratio_key(values)
        return _ratio_key(self.config.get("prune_ratio"))

    def _resolve_budget_target(self) -> str:
        configured = str(self.config.get("budget_target", "auto")).strip().lower()
        if configured != "auto":
            if configured not in TARGET_PROFILES:
                raise ValueError(
                    "sparsevlm_budget_candidate_scnd.budget_target must be auto, retain192, "
                    f"retain128, or retain64; got {configured!r}"
                )
            return configured
        ratio = self._configured_ratio_key()
        target = KNOWN_TARGET_RATIOS.get(ratio)
        if target is None:
            raise ValueError(
                "sparsevlm_budget_candidate_scnd could not infer budget_target from prune_ratio. "
                f"Known ratios are {sorted(KNOWN_TARGET_RATIOS)}, got {ratio}. "
                "Set pruning.sparsevlm_budget_candidate_scnd.budget_target explicitly."
            )
        return target

    @staticmethod
    def _normalize_mode(value: object) -> str:
        mode = str(value).strip().upper()
        if mode not in MODE_ALIASES:
            raise ValueError(
                f"sparsevlm_budget_candidate_scnd.layer_modes supports C/CAND, B/BLITE, S; got {value!r}"
            )
        return MODE_ALIASES[mode]

    def _layer_mode_map(self) -> Dict[int, str]:
        prune_layers = self._prune_layers()
        layer_modes = self.config.get("layer_modes")
        layer_policy = str(self.config.get("layer_policy", "auto")).strip().lower()

        if layer_modes is None:
            if layer_policy != "auto":
                raise ValueError(
                    "sparsevlm_budget_candidate_scnd currently supports layer_policy=auto "
                    "unless layer_modes is explicitly set"
                )
            target = self._resolve_budget_target()
            layer_modes = AUTO_LAYER_POLICY[target][: len(prune_layers)]
        elif isinstance(layer_modes, str):
            layer_modes = [mode.strip() for mode in layer_modes.split(",") if mode.strip()]

        layer_modes = [self._normalize_mode(mode) for mode in layer_modes]
        if len(layer_modes) != len(prune_layers):
            raise ValueError(
                "sparsevlm_budget_candidate_scnd.layer_modes must have the same length as prune_layers, "
                f"got modes={layer_modes} and prune_layers={prune_layers}"
            )
        invalid = [mode for mode in layer_modes if mode not in VALID_LAYER_MODES]
        if invalid:
            raise ValueError(
                f"sparsevlm_budget_candidate_scnd.layer_modes only supports {sorted(VALID_LAYER_MODES)}, got {invalid}"
            )
        return dict(zip(prune_layers, layer_modes))

    def _layer_mode(self, layer_idx: int) -> str:
        mode_map = self._layer_mode_map()
        if int(layer_idx) not in mode_map:
            raise ValueError(f"Layer {layer_idx} is missing from sparsevlm_budget_candidate_scnd.layer_modes")
        return mode_map[int(layer_idx)]

    def _config_float(self, key: str, default: float) -> float:
        raw_value = self.config.get(key)
        if raw_value is None:
            return float(default)
        return float(raw_value)

    def _params(self, profile: BudgetProfile | None = None) -> Dict[str, float | bool | str]:
        if profile is None:
            profile = self._profile(self._resolve_budget_target())
        seed_ratio_min = float(self.config.get("seed_ratio_min", 0.15))
        seed_ratio_max = float(self.config.get("seed_ratio_max", 0.55))
        seed_pool_multiplier = float(self.config.get("seed_pool_multiplier", 2.0))
        saliency_floor_min = self._config_float("saliency_floor_min", profile.saliency_floor_min)
        saliency_floor_max = self._config_float("saliency_floor_max", profile.saliency_floor_max)
        boundary_ratio = self._config_float("boundary_ratio", profile.boundary_ratio)
        boundary_pool_multiplier = self._config_float("boundary_pool_multiplier", profile.boundary_pool_multiplier)
        cached_diversity_weight = self._config_float("cached_diversity_weight", profile.cached_diversity_weight)
        boundary_gate_min = self._config_float("boundary_gate_min", profile.blite_gate_min)
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
        if boundary_pool_multiplier < 1.0:
            raise ValueError(f"boundary_pool_multiplier must be >= 1, got {boundary_pool_multiplier}")
        if cached_diversity_weight < 0.0:
            raise ValueError(f"cached_diversity_weight must be >= 0, got {cached_diversity_weight}")
        if not 0.0 <= boundary_gate_min <= 1.0:
            raise ValueError(f"boundary_gate_min must be in [0, 1], got {boundary_gate_min}")
        if distance_metric != "cosine":
            raise ValueError(f"Only distance_metric=cosine is supported, got {distance_metric!r}")

        return {
            "seed_ratio_min": seed_ratio_min,
            "seed_ratio_max": seed_ratio_max,
            "seed_pool_multiplier": seed_pool_multiplier,
            "saliency_floor_min": saliency_floor_min,
            "saliency_floor_max": saliency_floor_max,
            "boundary_ratio": boundary_ratio,
            "boundary_pool_multiplier": boundary_pool_multiplier,
            "cached_diversity_weight": cached_diversity_weight,
            "boundary_gate_min": boundary_gate_min,
            "distance_metric": distance_metric,
            "saliency_repair": saliency_repair,
        }

    def _profile(self, budget_target: str) -> BudgetProfile:
        try:
            return TARGET_PROFILES[budget_target]
        except KeyError as exc:
            raise ValueError(f"Unsupported budget target: {budget_target}") from exc

    @staticmethod
    def _score_values(scores: torch.Tensor) -> List[float]:
        return [float(value) for value in scores.detach().cpu().tolist()]

    @staticmethod
    def _saliency_sum(values: List[float], indices: List[int]) -> float:
        return float(sum(max(float(values[int(idx)]), 0.0) for idx in indices))

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

    def _distance_to_center(self, projected: torch.Tensor, indices: List[int], center_indices: List[int]) -> torch.Tensor:
        if not indices:
            return torch.empty(0, device=projected.device, dtype=torch.float32)
        if center_indices:
            center_tensor = torch.tensor(center_indices, device=projected.device, dtype=torch.long)
            center = projected.index_select(0, center_tensor).mean(dim=0)
        else:
            center = projected.mean(dim=0)
        center = torch.nn.functional.normalize(center, dim=0, eps=1e-6)
        idx_tensor = torch.tensor(indices, device=projected.device, dtype=torch.long)
        similarity = projected.index_select(0, idx_tensor) @ center
        return torch.clamp((1.0 - similarity) / 2.0, min=0.0, max=1.0)

    def _build_candidate_pool(
        self,
        mixed_score: torch.Tensor,
        projected: torch.Tensor,
        target_keep: int,
        profile: BudgetProfile,
    ) -> Dict[str, object]:
        num_visual = int(mixed_score.numel())
        keep_count = min(max(int(target_keep), 0), num_visual)
        saliency_order = [int(idx) for idx in deterministic_descending_indices(mixed_score).detach().cpu().tolist()]

        saliency_pool_count = min(
            num_visual,
            max(keep_count, int(math.ceil(float(keep_count) * profile.alpha))),
        )
        reservoir_source_count = min(
            num_visual,
            max(keep_count, int(math.ceil(float(keep_count) * profile.rho))),
        )
        reservoir_count = min(
            reservoir_source_count,
            int(math.ceil(float(keep_count) * profile.beta)),
        )
        cap_by_multiplier = int(math.ceil(float(keep_count) * profile.candidate_cap))
        cap_by_fraction = 0
        if profile.candidate_cap_fraction is not None:
            cap_by_fraction = int(math.ceil(float(num_visual) * float(profile.candidate_cap_fraction)))
        candidate_cap = min(num_visual, max(keep_count, cap_by_multiplier, cap_by_fraction))

        saliency_pool = saliency_order[:saliency_pool_count]
        reservoir_source = saliency_order[:reservoir_source_count]
        topk_center = saliency_order[:keep_count]
        saliency_values = self._score_values(mixed_score)
        reservoir: List[int] = []
        if reservoir_count > 0 and reservoir_source:
            distances = self._distance_to_center(projected, reservoir_source, topk_center)
            dist_values = _as_float_list(distances)
            reservoir_order = sorted(
                range(len(reservoir_source)),
                key=lambda pos: (-float(dist_values[pos]), -float(saliency_values[reservoir_source[pos]]), reservoir_source[pos]),
            )
            reservoir = [int(reservoir_source[pos]) for pos in reservoir_order[:reservoir_count]]

        global_reservoir_count = min(
            num_visual,
            int(math.ceil(float(candidate_cap) * float(profile.global_reservoir_fraction))),
        )
        global_reservoir: List[int] = []
        if global_reservoir_count > 0:
            all_indices = list(range(num_visual))
            global_distances = self._distance_to_center(projected, all_indices, topk_center)
            global_dist_values = _as_float_list(global_distances)
            saliency_pool_set = set(int(idx) for idx in saliency_pool)
            reservoir_source_set = set(int(idx) for idx in reservoir_source)
            global_order = sorted(
                all_indices,
                key=lambda idx: (
                    int(idx) in saliency_pool_set,
                    int(idx) in reservoir_source_set,
                    -float(global_dist_values[int(idx)]),
                    -float(saliency_values[int(idx)]),
                    int(idx),
                ),
            )
            global_reservoir = [int(idx) for idx in global_order[:global_reservoir_count]]

        saliency_set = set(int(idx) for idx in saliency_pool)
        unique_reservoir: List[int] = []
        reservoir_seen: set[int] = set()
        for raw_idx in reservoir:
            idx = int(raw_idx)
            if idx in reservoir_seen:
                continue
            reservoir_seen.add(idx)
            unique_reservoir.append(idx)
        unique_global_reservoir: List[int] = []
        global_seen: set[int] = set()
        for raw_idx in global_reservoir:
            idx = int(raw_idx)
            if idx in global_seen:
                continue
            global_seen.add(idx)
            unique_global_reservoir.append(idx)
        reservoir_priority = [idx for idx in unique_reservoir if idx not in saliency_set]
        reservoir_priority.extend(idx for idx in unique_reservoir if idx in saliency_set)
        reservoir_reserved_count = min(
            len(reservoir_priority),
            int(math.ceil(float(candidate_cap) * float(profile.reservoir_min_fraction))),
        )
        reserved_reservoir = reservoir_priority[:reservoir_reserved_count]
        global_priority = [idx for idx in unique_global_reservoir if idx not in saliency_set and idx not in reservoir_seen]
        global_priority.extend(
            idx for idx in unique_global_reservoir if idx not in global_priority and idx not in reservoir_seen
        )
        global_reservoir_reserved_count = min(len(global_priority), global_reservoir_count)
        reserved_global_reservoir = global_priority[:global_reservoir_reserved_count]
        saliency_slot_count = max(0, candidate_cap - reservoir_reserved_count - global_reservoir_reserved_count)

        candidate_pool: List[int] = []
        seen: set[int] = set()
        for raw_idx in saliency_pool[:saliency_slot_count]:
            idx = int(raw_idx)
            if idx in seen:
                continue
            candidate_pool.append(idx)
            seen.add(idx)

        for raw_idx in reserved_reservoir:
            idx = int(raw_idx)
            if idx in seen:
                continue
            candidate_pool.append(idx)
            seen.add(idx)
            if len(candidate_pool) >= candidate_cap:
                break

        for raw_idx in reserved_global_reservoir:
            idx = int(raw_idx)
            if idx in seen:
                continue
            candidate_pool.append(idx)
            seen.add(idx)
            if len(candidate_pool) >= candidate_cap:
                break

        for group in (saliency_pool[saliency_slot_count:], reservoir, global_reservoir, saliency_order):
            for raw_idx in group:
                idx = int(raw_idx)
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
            "reservoir_source_indices": reservoir_source,
            "reservoir_indices": reservoir,
            "reservoir_reserved_indices": reserved_reservoir,
            "global_reservoir_indices": global_reservoir,
            "global_reservoir_reserved_indices": reserved_global_reservoir,
            "candidate_pool_indices": candidate_pool,
            "saliency_pool_count": saliency_pool_count,
            "reservoir_source_count": reservoir_source_count,
            "reservoir_reserved_count": reservoir_reserved_count,
            "global_reservoir_count": len(global_reservoir),
            "global_reservoir_reserved_count": len(reserved_global_reservoir),
            "candidate_pool_saliency_count": sum(1 for idx in candidate_pool if idx in saliency_set),
            "candidate_pool_reservoir_count": sum(1 for idx in candidate_pool if idx in set(unique_reservoir)),
            "candidate_pool_reservoir_only_count": sum(
                1 for idx in candidate_pool if idx in set(unique_reservoir) and idx not in saliency_set
            ),
            "candidate_pool_global_reservoir_count": sum(
                1 for idx in candidate_pool if idx in set(unique_global_reservoir)
            ),
            "candidate_pool_global_reservoir_only_count": sum(
                1 for idx in candidate_pool if idx in set(unique_global_reservoir) and idx not in saliency_set
            ),
            "candidate_cap_fraction_count": cap_by_fraction,
            "candidate_cap": candidate_cap,
        }

    @staticmethod
    def _candidate_distance(projected: torch.Tensor, candidate_pool: List[int]) -> torch.Tensor:
        if not candidate_pool:
            return torch.empty((0, 0), device=projected.device, dtype=torch.float32)
        idx_tensor = torch.tensor(candidate_pool, device=projected.device, dtype=torch.long)
        candidate_projected = projected.index_select(0, idx_tensor)
        return torch.clamp(
            (1.0 - candidate_projected @ candidate_projected.transpose(0, 1)) / 2.0,
            min=0.0,
            max=1.0,
        )

    def _top_remaining_saliency(
        self,
        saliency_order: List[int],
        saliency_values: List[float],
        selected: set[int],
        extra_idx: int,
        slots: int,
        allowed: set[int],
    ) -> float:
        if slots <= 0:
            return 0.0
        total = 0.0
        used = 0
        for idx in saliency_order:
            idx = int(idx)
            if idx not in allowed or idx in selected or idx == int(extra_idx):
                continue
            total += max(float(saliency_values[idx]), 0.0)
            used += 1
            if used == slots:
                break
        return total

    def _repair_saliency_mass_within_pool(
        self,
        selected: List[int],
        seed_indices: List[int],
        saliency_score: torch.Tensor,
        distance: torch.Tensor,
        candidate_pool: List[int],
        saliency_mass_floor: float,
    ) -> Tuple[List[int], List[Dict[str, float | int]]]:
        saliency_values = self._score_values(saliency_score)
        selected_set = set(int(idx) for idx in selected)
        seed_set = set(int(idx) for idx in seed_indices)
        candidate_set = set(int(idx) for idx in candidate_pool)
        local_pos = {int(idx): pos for pos, idx in enumerate(candidate_pool)}
        replacements: List[Dict[str, float | int]] = []

        def selected_mass() -> float:
            return self._saliency_sum(saliency_values, list(selected_set))

        while selected_mass() + 1e-8 < float(saliency_mass_floor):
            unselected = [idx for idx in candidate_set if idx not in selected_set]
            if not unselected:
                break
            incoming = max(unselected, key=lambda idx: (saliency_values[idx], -idx))
            replaceable = [idx for idx in selected_set if idx not in seed_set]
            if not replaceable:
                replaceable = list(selected_set)
            if len(selected_set) > 1:
                selected_local = torch.tensor(
                    [local_pos[idx] for idx in selected_set], device=distance.device, dtype=torch.long
                )
                contributions: Dict[int, float] = {}
                for idx in selected_set:
                    pos = local_pos[idx]
                    other = selected_local[selected_local != pos]
                    if int(other.numel()) == 0:
                        contributions[idx] = 0.0
                    else:
                        value = distance.index_select(0, torch.tensor([pos], device=distance.device)).index_select(1, other).min()
                        contributions[idx] = float(value.item())
            else:
                contributions = {idx: 0.0 for idx in selected_set}
            outgoing = min(
                replaceable,
                key=lambda idx: (saliency_values[idx], contributions.get(int(idx), 0.0), -int(idx)),
            )
            if saliency_values[incoming] <= saliency_values[outgoing] + 1e-8:
                break
            selected_set.remove(int(outgoing))
            selected_set.add(int(incoming))
            replacements.append(
                {
                    "out": int(outgoing),
                    "in": int(incoming),
                    "mass_gain": float(saliency_values[incoming] - saliency_values[outgoing]),
                }
            )

        return sorted(selected_set), replacements

    def _max_min_seed(
        self,
        distance: torch.Tensor,
        saliency_score: torch.Tensor,
        candidate_pool: List[int],
        seed_pool_global: List[int],
        seed_count: int,
    ) -> Tuple[List[int], List[float]]:
        if seed_count <= 0 or not seed_pool_global:
            return [], []
        saliency_values = self._score_values(saliency_score)
        local_pos = {int(idx): pos for pos, idx in enumerate(candidate_pool)}
        ordered_pool = sorted(
            [int(idx) for idx in seed_pool_global if int(idx) in local_pos],
            key=lambda idx: (-float(saliency_values[idx]), idx),
        )
        if not ordered_pool:
            return [], []

        selected = [ordered_pool[0]]
        gains = [0.0]
        while len(selected) < min(seed_count, len(ordered_pool)):
            remaining = [idx for idx in ordered_pool if idx not in selected]
            remaining_local = torch.tensor([local_pos[idx] for idx in remaining], device=distance.device, dtype=torch.long)
            selected_local = torch.tensor([local_pos[idx] for idx in selected], device=distance.device, dtype=torch.long)
            dist_gain = distance.index_select(0, remaining_local).index_select(1, selected_local).min(dim=1).values
            gain_values = _as_float_list(dist_gain)
            best_pos = min(
                range(len(remaining)),
                key=lambda pos: (-float(gain_values[pos]), -float(saliency_values[remaining[pos]]), remaining[pos]),
            )
            selected.append(int(remaining[best_pos]))
            gains.append(float(gain_values[best_pos]))
        return selected, gains

    def _candidate_select(
        self,
        mixed_score: torch.Tensor,
        projected: torch.Tensor,
        target_keep: int,
        entropy_norm: float,
        margin_confidence: float,
        profile: BudgetProfile,
        params: Dict[str, float | bool | str],
    ) -> Dict[str, object]:
        start_time = time.perf_counter()
        num_visual = int(mixed_score.numel())
        keep_count = min(max(int(target_keep), 0), num_visual)
        if keep_count <= 0:
            empty = _empty_long(mixed_score.device)
            return {
                "keep_indices": empty,
                "candidate_pool_indices": empty,
                "saliency_pool_indices": empty,
                "reservoir_source_indices": empty,
                "reservoir_indices": empty,
                "global_reservoir_indices": empty,
                "seed_pool_indices": empty,
                "seed_indices": empty,
                "saliency_core_indices": empty,
                "mmr_selected_order": empty,
                "diversity_gain": torch.empty(0, device=mixed_score.device, dtype=torch.float32),
                "cached_diversity_gain": torch.zeros(num_visual, device=mixed_score.device, dtype=torch.float32),
                "diversity_steps_budget": 0,
                "diversity_steps_used": 0,
                "diversity_gate": 0.0,
                "diversity_gate_raw": 0.0,
                "selection_time_ms": 0.0,
                "distance_cost_proxy": 0,
            }

        pool = self._build_candidate_pool(
            mixed_score=mixed_score,
            projected=projected,
            target_keep=keep_count,
            profile=profile,
        )
        saliency_order = [int(idx) for idx in pool["saliency_order"]]
        candidate_pool = [int(idx) for idx in pool["candidate_pool_indices"]]
        candidate_set = set(candidate_pool)
        saliency_values = self._score_values(mixed_score)
        distance = self._candidate_distance(projected, candidate_pool)
        local_pos = {int(idx): pos for pos, idx in enumerate(candidate_pool)}
        entropy_norm = _clamp01(entropy_norm)
        margin_confidence = _clamp01(margin_confidence)
        raw_gate = entropy_norm * (1.0 - margin_confidence)
        gate = max(float(profile.gate_floor), raw_gate)
        diversity_fill_budget = min(
            keep_count,
            int(math.ceil(float(keep_count) * profile.step_ratio * gate)),
        )

        topk = saliency_order[:keep_count]
        saliency_mass_topk = self._saliency_sum(saliency_values, topk)
        eta = float(params["saliency_floor_min"]) + (
            float(params["saliency_floor_max"]) - float(params["saliency_floor_min"])
        ) * (1.0 - entropy_norm)
        saliency_mass_floor = float(eta * saliency_mass_topk)

        saliency_core_count = min(
            keep_count,
            int(math.ceil(float(keep_count) * float(profile.saliency_core_ratio))),
        )
        saliency_core_indices: List[int] = []
        if saliency_core_count > 0:
            for idx in saliency_order:
                if int(idx) not in candidate_set:
                    continue
                saliency_core_indices.append(int(idx))
                if len(saliency_core_indices) == saliency_core_count:
                    break
        diversity_steps_budget = min(keep_count, len(saliency_core_indices) + diversity_fill_budget)

        seed_ratio = float(params["seed_ratio_min"]) + (
            float(params["seed_ratio_max"]) - float(params["seed_ratio_min"])
        ) * (1.0 - entropy_norm)
        seed_pool_indices: List[int] = []
        if saliency_core_indices:
            selected = list(saliency_core_indices)
            selected_gains = [0.0 for _ in selected]
            seed_count = len(selected)
            seed_pool_indices = list(selected)
        else:
            seed_count = min(diversity_steps_budget, int(math.ceil(float(keep_count) * seed_ratio)))
            if diversity_steps_budget > 0 and seed_ratio > 0.0:
                seed_count = max(1, seed_count)
            seed_pool_count = min(
                len(candidate_pool),
                max(seed_count, int(math.ceil(float(max(seed_count, 1)) * float(params["seed_pool_multiplier"])))),
            )

            def add_seed_candidate(raw_idx: int) -> None:
                idx = int(raw_idx)
                if idx not in candidate_set or idx in seed_pool_indices:
                    return
                if len(seed_pool_indices) < seed_pool_count:
                    seed_pool_indices.append(idx)

            saliency_seed_quota = min(seed_pool_count, max(seed_count, int(math.ceil(float(seed_pool_count) * 0.5))))
            for idx in saliency_order:
                if len(seed_pool_indices) >= saliency_seed_quota:
                    break
                add_seed_candidate(idx)
            for idx in pool.get("reservoir_reserved_indices", []):
                add_seed_candidate(int(idx))
            for idx in pool.get("global_reservoir_reserved_indices", []):
                add_seed_candidate(int(idx))
            for idx in pool.get("reservoir_indices", []):
                add_seed_candidate(int(idx))
            for idx in pool.get("global_reservoir_indices", []):
                add_seed_candidate(int(idx))
            for idx in candidate_pool:
                add_seed_candidate(idx)
            selected, selected_gains = self._max_min_seed(
                distance=distance,
                saliency_score=mixed_score,
                candidate_pool=candidate_pool,
                seed_pool_global=seed_pool_indices,
                seed_count=seed_count,
            )
        selected_set = set(selected)
        selected_order = list(selected)
        cached_gain = torch.zeros(num_visual, device=mixed_score.device, dtype=torch.float32)
        for idx, gain in zip(selected, selected_gains):
            cached_gain[int(idx)] = float(gain)

        feasible_candidate_count = 0
        diversity_steps_used = len(selected)
        while len(selected) < min(diversity_steps_budget, keep_count):
            remaining = [idx for idx in candidate_pool if idx not in selected_set]
            if not remaining:
                break
            if not selected:
                best_idx = max(remaining, key=lambda idx: (float(saliency_values[int(idx)]), -int(idx)))
                best_gain = 0.0
                selected.append(int(best_idx))
                selected_set.add(int(best_idx))
                selected_order.append(int(best_idx))
                selected_gains.append(best_gain)
                cached_gain[int(best_idx)] = best_gain
                diversity_steps_used += 1
                continue
            selected_mass = self._saliency_sum(saliency_values, list(selected_set))
            slots_after = keep_count - len(selected) - 1
            feasible: List[int] = []
            for idx in remaining:
                possible_mass = (
                    selected_mass
                    + max(saliency_values[int(idx)], 0.0)
                    + self._top_remaining_saliency(
                        saliency_order=saliency_order,
                        saliency_values=saliency_values,
                        selected=selected_set,
                        extra_idx=int(idx),
                        slots=slots_after,
                        allowed=candidate_set,
                    )
                )
                if possible_mass + 1e-8 >= saliency_mass_floor:
                    feasible.append(int(idx))
            feasible_candidate_count += len(feasible)
            candidate = feasible if feasible else remaining
            selected_local = torch.tensor([local_pos[idx] for idx in selected], device=distance.device, dtype=torch.long)
            candidate_local = torch.tensor([local_pos[idx] for idx in candidate], device=distance.device, dtype=torch.long)
            gains = distance.index_select(0, candidate_local).index_select(1, selected_local).min(dim=1).values
            gain_values = _as_float_list(gains)
            best_pos = min(
                range(len(candidate)),
                key=lambda pos: (-float(gain_values[pos]), -float(saliency_values[candidate[pos]]), candidate[pos]),
            )
            best_idx = int(candidate[best_pos])
            best_gain = float(gain_values[best_pos])
            selected.append(best_idx)
            selected_set.add(best_idx)
            selected_order.append(best_idx)
            selected_gains.append(best_gain)
            cached_gain[best_idx] = best_gain
            diversity_steps_used += 1

        diversity_steps_used = min(diversity_steps_used, diversity_steps_budget)
        if len(selected) < keep_count:
            for idx in saliency_order:
                if idx not in candidate_set or idx in selected_set:
                    continue
                selected.append(idx)
                selected_set.add(idx)
                selected_order.append(idx)
                selected_gains.append(float(cached_gain[idx].item()))
                if len(selected) == keep_count:
                    break

        repair_replacements: List[Dict[str, float | int]] = []
        if bool(params["saliency_repair"]) and selected:
            selected, repair_replacements = self._repair_saliency_mass_within_pool(
                selected=selected,
                seed_indices=selected_order[:seed_count],
                saliency_score=mixed_score,
                distance=distance,
                candidate_pool=candidate_pool,
                saliency_mass_floor=saliency_mass_floor,
            )

        if len(selected) < keep_count:
            selected_set = set(selected)
            for idx in saliency_order:
                if idx not in candidate_set or idx in selected_set:
                    continue
                selected.append(idx)
                selected_set.add(idx)
                if len(selected) == keep_count:
                    break

        keep_indices = torch.tensor(sorted(selected[:keep_count]), device=mixed_score.device, dtype=torch.long)
        saliency_mass_selected = self._saliency_sum(saliency_values, [int(idx) for idx in keep_indices.detach().cpu().tolist()])
        selection_time_ms = (time.perf_counter() - start_time) * 1000.0
        return {
            "keep_indices": keep_indices,
            "candidate_pool_indices": torch.tensor(candidate_pool, device=mixed_score.device, dtype=torch.long),
            "saliency_pool_indices": torch.tensor(pool["saliency_pool_indices"], device=mixed_score.device, dtype=torch.long),
            "reservoir_source_indices": torch.tensor(pool["reservoir_source_indices"], device=mixed_score.device, dtype=torch.long),
            "reservoir_indices": torch.tensor(pool["reservoir_indices"], device=mixed_score.device, dtype=torch.long),
            "reservoir_reserved_indices": torch.tensor(
                pool["reservoir_reserved_indices"], device=mixed_score.device, dtype=torch.long
            ),
            "global_reservoir_indices": torch.tensor(
                pool["global_reservoir_indices"], device=mixed_score.device, dtype=torch.long
            ),
            "global_reservoir_reserved_indices": torch.tensor(
                pool["global_reservoir_reserved_indices"], device=mixed_score.device, dtype=torch.long
            ),
            "seed_pool_indices": torch.tensor(seed_pool_indices, device=mixed_score.device, dtype=torch.long),
            "seed_indices": torch.tensor(sorted(selected_order[:seed_count]), device=mixed_score.device, dtype=torch.long),
            "saliency_core_indices": torch.tensor(saliency_core_indices, device=mixed_score.device, dtype=torch.long),
            "mmr_selected_order": torch.tensor(selected_order, device=mixed_score.device, dtype=torch.long),
            "diversity_gain": torch.tensor(selected_gains, device=mixed_score.device, dtype=torch.float32),
            "cached_diversity_gain": cached_gain,
            "saliency_floor_eta": float(eta),
            "saliency_mass_floor": float(saliency_mass_floor),
            "saliency_mass_selected": float(saliency_mass_selected),
            "saliency_mass_topk": float(saliency_mass_topk),
            "feasible_candidate_count": int(feasible_candidate_count),
            "repair_replacements": repair_replacements,
            "seed_ratio": float(seed_ratio),
            "seed_count": int(seed_count),
            "candidate_pool_size": int(len(candidate_pool)),
            "reservoir_size": int(len(pool["reservoir_indices"])),
            "reservoir_source_count": int(pool["reservoir_source_count"]),
            "reservoir_reserved_count": int(pool["reservoir_reserved_count"]),
            "global_reservoir_size": int(len(pool["global_reservoir_indices"])),
            "global_reservoir_count": int(pool["global_reservoir_count"]),
            "global_reservoir_reserved_count": int(pool["global_reservoir_reserved_count"]),
            "saliency_pool_count": int(pool["saliency_pool_count"]),
            "candidate_pool_saliency_count": int(pool["candidate_pool_saliency_count"]),
            "candidate_pool_reservoir_count": int(pool["candidate_pool_reservoir_count"]),
            "candidate_pool_reservoir_only_count": int(pool["candidate_pool_reservoir_only_count"]),
            "candidate_pool_global_reservoir_count": int(pool["candidate_pool_global_reservoir_count"]),
            "candidate_pool_global_reservoir_only_count": int(pool["candidate_pool_global_reservoir_only_count"]),
            "candidate_cap_fraction_count": int(pool["candidate_cap_fraction_count"]),
            "candidate_cap_count": int(pool["candidate_cap"]),
            "saliency_core_count": int(len(saliency_core_indices)),
            "saliency_core_ratio": float(profile.saliency_core_ratio),
            "diversity_fill_budget": int(diversity_fill_budget),
            "diversity_steps_budget": int(diversity_steps_budget),
            "diversity_steps_used": int(diversity_steps_used),
            "diversity_gate": float(gate),
            "diversity_gate_raw": float(raw_gate),
            "selection_time_ms": float(selection_time_ms),
            "distance_cost_proxy": int(len(candidate_pool) * len(candidate_pool)),
        }

    def _alive_cached_gain(
        self,
        context: Dict[str, object],
        current_patch_indices: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        cache = context.get("budget_candidate_scnd_diversity_gain")
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
        cache = context.get("budget_candidate_scnd_diversity_gain")
        if cache is None:
            return
        full_gain = selection.get("cached_diversity_gain")
        if not torch.is_tensor(full_gain) or int(full_gain.numel()) == 0:
            return
        patch_indices = current_patch_indices.to(device=cache.device, dtype=torch.long)
        cache[patch_indices] = full_gain.to(device=cache.device, dtype=torch.float32)

    def _blite_select(
        self,
        mixed_score: torch.Tensor,
        cached_gain: torch.Tensor,
        target_keep: int,
        entropy_norm: float,
        margin_confidence: float,
        params: Dict[str, float | bool | str],
        projected: Optional[torch.Tensor] = None,
    ) -> Dict[str, object]:
        start_time = time.perf_counter()
        num_visual = int(mixed_score.numel())
        keep_count = min(max(int(target_keep), 0), num_visual)
        if keep_count <= 0:
            empty = _empty_long(mixed_score.device)
            return {
                "keep_indices": empty,
                "core_indices": empty,
                "boundary_core_indices": empty,
                "boundary_pool_indices": empty,
                "boundary_fill_indices": empty,
                "candidate_pool_indices": empty,
                "cached_diversity_gain": cached_gain,
                "boundary_count": 0,
                "blite_enabled": False,
                "blite_disabled_reason": "empty_keep",
                "blite_mode": "none",
                "native_boundary_used": False,
                "selection_time_ms": 0.0,
                "distance_cost_proxy": 0,
            }

        order = [int(idx) for idx in deterministic_descending_indices(mixed_score).detach().cpu().tolist()]
        gate = _clamp01(entropy_norm) * (1.0 - _clamp01(margin_confidence))
        raw_boundary = float(params["boundary_ratio"]) * float(keep_count) * gate
        boundary_count = min(keep_count, int(math.ceil(raw_boundary)))

        gain_float = cached_gain.to(dtype=torch.float32)
        gain_max = float(gain_float.abs().max().item()) if int(gain_float.numel()) else 0.0
        gain_std = float(gain_float.std(unbiased=False).item()) if int(gain_float.numel()) else 0.0
        disabled_reason = ""
        if boundary_count <= 0:
            disabled_reason = "empty_boundary"
        elif gate < float(params["boundary_gate_min"]):
            disabled_reason = "low_entropy_or_high_margin"
        elif projected is None and (gain_max <= 1e-8 or gain_std <= 1e-8):
            disabled_reason = "no_cached_diversity_gain"

        if disabled_reason:
            keep = torch.tensor(sorted(order[:keep_count]), device=mixed_score.device, dtype=torch.long)
            return {
                "keep_indices": keep,
                "core_indices": keep,
                "boundary_core_indices": keep,
                "boundary_pool_indices": _empty_long(mixed_score.device),
                "boundary_fill_indices": _empty_long(mixed_score.device),
                "candidate_pool_indices": _empty_long(mixed_score.device),
                "cached_diversity_gain": cached_gain,
                "boundary_count": 0,
                "blite_enabled": False,
                "blite_disabled_reason": disabled_reason,
                "blite_mode": "none",
                "native_boundary_used": False,
                "selection_time_ms": float((time.perf_counter() - start_time) * 1000.0),
                "distance_cost_proxy": 0,
            }

        core_count = keep_count - boundary_count
        core = order[:core_count]
        pool_count = min(
            num_visual - core_count,
            max(boundary_count, int(math.ceil(float(boundary_count) * float(params["boundary_pool_multiplier"])))),
        )
        boundary_pool = order[core_count:core_count + pool_count]
        selected = set(int(idx) for idx in core)
        score_values = self._score_values(mixed_score)
        tie_score = None
        native_boundary_used = projected is not None
        blite_mode = "native_boundary" if native_boundary_used else "cached_boundary"
        boundary_distance_cost = 0

        if native_boundary_used:
            fill: List[int] = []
            remaining = [int(idx) for idx in boundary_pool]
            while len(fill) < boundary_count and remaining:
                candidate_tensor = torch.tensor(remaining, device=projected.device, dtype=torch.long)
                if selected:
                    selected_tensor = torch.tensor(sorted(selected), device=projected.device, dtype=torch.long)
                    candidate_embeds = projected.index_select(0, candidate_tensor)
                    selected_embeds = projected.index_select(0, selected_tensor)
                    distances = torch.clamp(
                        (1.0 - candidate_embeds @ selected_embeds.transpose(0, 1)) / 2.0,
                        min=0.0,
                        max=1.0,
                    ).min(dim=1).values
                    gain_values = _as_float_list(distances)
                    boundary_distance_cost += int(candidate_embeds.shape[0] * selected_embeds.shape[0])
                else:
                    gain_values = [0.0 for _ in remaining]
                best_pos = min(
                    range(len(remaining)),
                    key=lambda pos: (-float(gain_values[pos]), -float(score_values[remaining[pos]]), remaining[pos]),
                )
                best_idx = int(remaining.pop(best_pos))
                fill.append(best_idx)
                selected.add(best_idx)
        else:
            gain_rank = _safe_rank_normalize(cached_gain)
            tie_score = mixed_score.to(dtype=torch.float32) + float(params["cached_diversity_weight"]) * gain_rank
            tie_score_values = self._score_values(tie_score)
            fill = sorted(
                set(int(idx) for idx in boundary_pool),
                key=lambda idx: (-float(tie_score_values[idx]), idx),
            )[:boundary_count]
            for idx in fill:
                selected.add(int(idx))

        if len(selected) < keep_count:
            for idx in order:
                selected.add(int(idx))
                if len(selected) == keep_count:
                    break
        keep = torch.tensor(sorted(selected), device=mixed_score.device, dtype=torch.long)
        return {
            "keep_indices": keep,
            "core_indices": torch.tensor(sorted(core), device=mixed_score.device, dtype=torch.long),
            "boundary_core_indices": torch.tensor(sorted(core), device=mixed_score.device, dtype=torch.long),
            "boundary_pool_indices": torch.tensor(boundary_pool, device=mixed_score.device, dtype=torch.long),
            "boundary_fill_indices": torch.tensor(sorted(fill), device=mixed_score.device, dtype=torch.long),
            "candidate_pool_indices": torch.tensor(boundary_pool, device=mixed_score.device, dtype=torch.long),
            "cached_diversity_gain": cached_gain,
            "boundary_count": int(boundary_count),
            "blite_enabled": True,
            "blite_disabled_reason": "",
            "blite_mode": blite_mode,
            "native_boundary_used": bool(native_boundary_used),
            "tie_break_scores": tie_score,
            "selection_time_ms": float((time.perf_counter() - start_time) * 1000.0),
            "distance_cost_proxy": int(boundary_distance_cost),
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
            raise ValueError("SparseVLMBudgetCandidateSCNDStrategy requires attention weights")

        context = self._require_sample_context()
        current_patch_indices = self._require_current_patch_indices(context, v_token_num)
        budget_target = self._resolve_budget_target()
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
                    "budget_target": budget_target,
                    "layer_policy": str(self.config.get("layer_policy", "auto")),
                }
            )
            self._set_pending_decision(context, current_patch_indices, layer_idx)
            return keep_indices, info

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
        margin_confidence = self._margin_confidence(visual_scores, target_keep)
        profile = self._profile(budget_target)
        params = self._params(profile)

        if mode == "C":
            if current_visual_embeds is None:
                raise ValueError("SparseVLMBudgetCandidateSCNDStrategy requires current_visual_embeds for C layers")
            if int(current_visual_embeds.shape[0]) != int(v_token_num):
                raise ValueError(
                    "current_visual_embeds length must match current visual token count, "
                    f"got {current_visual_embeds.shape[0]} vs {v_token_num}"
                )
            projected = self._project_visual_embeds(current_visual_embeds, int(profile.projection_dim))
            selection = self._candidate_select(
                mixed_score=mixed_score,
                projected=projected,
                target_keep=target_keep,
                entropy_norm=entropy_norm,
                margin_confidence=margin_confidence,
                profile=profile,
                params=params,
            )
            self._store_cached_gain(context, current_patch_indices, selection)
            selection_rule = "budget_candidate_scnd"
            mean_pairwise_distance = _projected_mean_pairwise_distance(projected, selection["keep_indices"])
        elif mode == "B":
            cached_gain = self._alive_cached_gain(context, current_patch_indices, mixed_score.device)
            projected = None
            if current_visual_embeds is not None:
                if int(current_visual_embeds.shape[0]) != int(v_token_num):
                    raise ValueError(
                        "current_visual_embeds length must match current visual token count, "
                        f"got {current_visual_embeds.shape[0]} vs {v_token_num}"
                    )
                projected = self._project_visual_embeds(current_visual_embeds, int(profile.projection_dim))
            selection = self._blite_select(
                mixed_score=mixed_score,
                cached_gain=cached_gain,
                target_keep=target_keep,
                entropy_norm=entropy_norm,
                margin_confidence=margin_confidence,
                params=params,
                projected=projected,
            )
            selection_rule = "budget_candidate_scnd_blite"
            if projected is not None and bool(selection.get("native_boundary_used", False)):
                mean_pairwise_distance = _projected_mean_pairwise_distance(projected, selection["keep_indices"])
            else:
                mean_pairwise_distance = None
        else:
            raise ValueError(f"Unsupported sparsevlm_budget_candidate_scnd layer mode: {mode}")

        keep_indices = selection["keep_indices"]
        if int(keep_indices.numel()) != int(target_keep):
            raise AssertionError(
                "sparsevlm_budget_candidate_scnd selection produced the wrong keep count, "
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
            "layer_policy": str(self.config.get("layer_policy", "auto")),
            "layer_strategy_effective": "sparsevlm_budget_candidate_scnd",
            "selection_rule": selection_rule,
            "profile_version": PROFILE_VERSION,
            "budget_target": budget_target,
            "candidate_alpha": profile.alpha,
            "candidate_beta": profile.beta,
            "candidate_rho": profile.rho,
            "candidate_cap_multiplier": profile.candidate_cap,
            "candidate_cap_fraction": profile.candidate_cap_fraction,
            "projection_dim": profile.projection_dim,
            "candidate_step_ratio": profile.step_ratio,
            "candidate_gate_floor": profile.gate_floor,
            "reservoir_min_fraction": profile.reservoir_min_fraction,
            "global_reservoir_fraction": profile.global_reservoir_fraction,
            "saliency_core_ratio": profile.saliency_core_ratio,
            "boundary_pool_multiplier": params["boundary_pool_multiplier"],
            "distance_metric": "cosine",
            "saliency_entropy": entropy_raw,
            "saliency_entropy_norm": entropy_norm,
            "margin_confidence": margin_confidence,
            "seed_ratio_min": params["seed_ratio_min"],
            "seed_ratio_max": params["seed_ratio_max"],
            "seed_pool_multiplier": params["seed_pool_multiplier"],
            "saliency_floor_min": params["saliency_floor_min"],
            "saliency_floor_max": params["saliency_floor_max"],
            "saliency_floor_effective_min": params["saliency_floor_min"],
            "saliency_floor_effective_max": params["saliency_floor_max"],
            "boundary_ratio": params["boundary_ratio"],
            "cached_diversity_weight": params["cached_diversity_weight"],
            "boundary_gate_min": params["boundary_gate_min"],
            "saliency_repair": params["saliency_repair"],
            "retained_saliency_mass_ratio": _saliency_mass_ratio(visual_scores, keep_indices),
            "global_prune_step": memory["global_prune_step"],
            "global_current_weight": memory["global_current_weight"],
            "global_ema_decay": memory["global_ema_decay"],
            "global_use_ema": memory["global_use_ema"],
            "use_score_memory": memory["use_score_memory"],
            "global_saliency_ema": memory["global_saliency_ema"].detach().cpu().numpy(),
        }
        if mean_pairwise_distance is not None:
            info["mean_selected_pairwise_distance"] = mean_pairwise_distance

        for key, value in selection.items():
            if key == "keep_indices":
                continue
            if torch.is_tensor(value):
                info[key] = value.detach().cpu().numpy()
            else:
                info[key] = value
        return keep_indices, info
