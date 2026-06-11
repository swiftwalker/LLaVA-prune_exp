"""SparseVLM SCND: saliency-constrained native DivPrune.

SCND keeps SparseVLM's text-conditioned saliency as the task gate, then uses
global max-min visual diversity inside that saliency constraint. The first
pruning layer can run native DivPrune-style selection (C), later layers can use
boundary-only diversity refinement (B), and any layer can fall back to pure
SparseVLM saliency top-k (S).
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .sparsevlm_adaptive_stratified import compute_target_keep_count
from .sparsevlm_diverse_mmr import (
    SparseVLMDiverseMMRStrategy,
    _cosine_distance_matrix,
    _saliency_mass_ratio,
    _selected_mean_pairwise_distance,
)
from .sparsevlm_score_memory import compute_entropy_stats, deterministic_descending_indices, rank_normalize_scores
from .scnd_triton_kernels import is_scnd_triton_available, scnd_native_c_select_triton


VALID_LAYER_MODES = {"C", "B", "S"}
VALID_SELECTION_BACKENDS = {"auto", "gpu", "python", "triton"}
VALID_C_SELECTION_RULES = {"native", "random_feasible"}
VALID_DISTANCE_METRICS = {"cosine", "euclidean", "dot"}


def _clamp01(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def _sync_if_cuda(tensor: Optional[torch.Tensor]) -> None:
    if tensor is not None and tensor.is_cuda:
        torch.cuda.synchronize(tensor.device)


def _scnd_distance_matrix(embeds: torch.Tensor, distance_metric: str) -> torch.Tensor:
    if embeds.ndim != 2:
        raise ValueError(f"current_visual_embeds must be 2D, got {tuple(embeds.shape)}")

    metric = str(distance_metric).lower()
    embeds_f = embeds.to(dtype=torch.float32)
    if metric == "cosine":
        distance = _cosine_distance_matrix(embeds_f)
    elif metric == "euclidean":
        distance = torch.cdist(embeds_f, embeds_f, p=2)
    elif metric == "dot":
        distance = -(embeds_f @ embeds_f.transpose(0, 1))
    else:
        raise ValueError(f"sparsevlm_scnd.distance_metric only supports {sorted(VALID_DISTANCE_METRICS)}, got {metric!r}")

    distance = distance.to(device=embeds.device, dtype=torch.float32)
    if int(distance.numel()) > 0:
        distance.fill_diagonal_(0.0)
    return distance


def _selected_mean_pairwise_distance_from_embeds(embeds: torch.Tensor, keep_indices: torch.Tensor) -> float:
    selected_count = int(keep_indices.numel())
    if selected_count <= 1:
        return 0.0
    normalized = F.normalize(embeds.to(dtype=torch.float32), dim=-1, eps=1e-6)
    selected = normalized.index_select(0, keep_indices.to(device=normalized.device, dtype=torch.long))
    distance = torch.clamp((1.0 - selected @ selected.transpose(0, 1)) / 2.0, min=0.0, max=1.0)
    total = distance.sum()
    denom = selected_count * (selected_count - 1)
    return float((total / denom).item())


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

    def _get_scnd_params(self) -> Dict[str, float | bool | str | int]:
        seed_ratio_min = float(self.config.get("seed_ratio_min", 0.15))
        seed_ratio_max = float(self.config.get("seed_ratio_max", 0.55))
        seed_pool_multiplier = float(self.config.get("seed_pool_multiplier", 2.0))
        saliency_floor_min = float(self.config.get("saliency_floor_min", 0.65))
        saliency_floor_max = float(self.config.get("saliency_floor_max", 0.90))
        boundary_ratio = float(self.config.get("boundary_ratio", 0.25))
        distance_metric = str(self.config.get("distance_metric", "cosine")).lower()
        saliency_repair = bool(self.config.get("saliency_repair", True))
        selection_backend = str(self.config.get("selection_backend", "auto")).lower()
        c_selection_rule = str(self.config.get("c_selection_rule", "native")).lower()
        seed = int(self.config.get("seed", 42))

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
        if distance_metric not in VALID_DISTANCE_METRICS:
            raise ValueError(
                f"sparsevlm_scnd.distance_metric only supports {sorted(VALID_DISTANCE_METRICS)}, "
                f"got {distance_metric!r}"
            )
        if distance_metric != "cosine" and "B" in self._layer_mode_map().values():
            raise ValueError(
                "sparsevlm_scnd.distance_metric values other than cosine are only supported when "
                "layer_modes do not include B"
            )
        if selection_backend not in VALID_SELECTION_BACKENDS:
            raise ValueError(
                f"sparsevlm_scnd.selection_backend only supports {sorted(VALID_SELECTION_BACKENDS)}, "
                f"got {selection_backend!r}"
            )
        if c_selection_rule not in VALID_C_SELECTION_RULES:
            raise ValueError(
                f"sparsevlm_scnd.c_selection_rule only supports {sorted(VALID_C_SELECTION_RULES)}, "
                f"got {c_selection_rule!r}"
            )

        return {
            "seed_ratio_min": seed_ratio_min,
            "seed_ratio_max": seed_ratio_max,
            "seed_pool_multiplier": seed_pool_multiplier,
            "saliency_floor_min": saliency_floor_min,
            "saliency_floor_max": saliency_floor_max,
            "boundary_ratio": boundary_ratio,
            "distance_metric": distance_metric,
            "saliency_repair": saliency_repair,
            "selection_backend": selection_backend,
            "c_selection_rule": c_selection_rule,
            "seed": seed,
        }

    @staticmethod
    def _score_list(scores: torch.Tensor) -> List[float]:
        return [float(value) for value in scores.detach().cpu().tolist()]

    @staticmethod
    def _descending_indices_tensor(scores: torch.Tensor) -> torch.Tensor:
        """Return score-descending indices with stable token-index tie breaking."""
        return torch.argsort(scores.to(dtype=torch.float32), descending=True, stable=True)

    @staticmethod
    def _best_high_gain_saliency_low_index(
        candidate_mask: torch.Tensor,
        gains: torch.Tensor,
        saliency_score: torch.Tensor,
    ) -> torch.Tensor:
        if not bool(candidate_mask.any().item()):
            raise ValueError("candidate_mask must contain at least one candidate")
        gains_f = gains.to(dtype=torch.float32)
        saliency_f = saliency_score.to(dtype=torch.float32)
        neg_inf = torch.tensor(float("-inf"), device=gains.device, dtype=torch.float32)
        masked_gains = torch.where(candidate_mask, gains_f, neg_inf)
        best_gain = masked_gains.max()
        gain_mask = candidate_mask & (gains_f == best_gain)
        masked_saliency = torch.where(gain_mask, saliency_f, neg_inf)
        best_saliency = masked_saliency.max()
        final_mask = gain_mask & (saliency_f == best_saliency)
        indices = torch.arange(int(candidate_mask.numel()), device=candidate_mask.device, dtype=torch.float32)
        priority = torch.where(final_mask, -indices, neg_inf)
        return torch.argmax(priority).to(dtype=torch.long)

    @staticmethod
    def _best_high_saliency_low_index(candidate_mask: torch.Tensor, saliency_score: torch.Tensor) -> torch.Tensor:
        if not bool(candidate_mask.any().item()):
            raise ValueError("candidate_mask must contain at least one candidate")
        saliency_f = saliency_score.to(dtype=torch.float32)
        neg_inf = torch.tensor(float("-inf"), device=saliency_score.device, dtype=torch.float32)
        masked_saliency = torch.where(candidate_mask, saliency_f, neg_inf)
        best_saliency = masked_saliency.max()
        final_mask = candidate_mask & (saliency_f == best_saliency)
        indices = torch.arange(int(candidate_mask.numel()), device=candidate_mask.device, dtype=torch.float32)
        priority = torch.where(final_mask, -indices, neg_inf)
        return torch.argmax(priority).to(dtype=torch.long)

    @staticmethod
    def _best_low_saliency_low_contribution_high_index(
        candidate_mask: torch.Tensor,
        saliency_score: torch.Tensor,
        contributions: torch.Tensor,
    ) -> torch.Tensor:
        if not bool(candidate_mask.any().item()):
            raise ValueError("candidate_mask must contain at least one candidate")
        saliency_f = saliency_score.to(dtype=torch.float32)
        contrib_f = contributions.to(dtype=torch.float32)
        pos_inf = torch.tensor(float("inf"), device=saliency_score.device, dtype=torch.float32)
        neg_inf = torch.tensor(float("-inf"), device=saliency_score.device, dtype=torch.float32)
        masked_saliency = torch.where(candidate_mask, saliency_f, pos_inf)
        min_saliency = masked_saliency.min()
        saliency_mask = candidate_mask & (saliency_f == min_saliency)
        masked_contrib = torch.where(saliency_mask, contrib_f, pos_inf)
        min_contrib = masked_contrib.min()
        final_mask = saliency_mask & (contrib_f == min_contrib)
        indices = torch.arange(int(candidate_mask.numel()), device=candidate_mask.device, dtype=torch.float32)
        priority = torch.where(final_mask, indices, neg_inf)
        return torch.argmax(priority).to(dtype=torch.long)

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

    def _max_min_select_from_pool_gpu(
        self,
        distance: torch.Tensor,
        saliency_score: torch.Tensor,
        pool_indices: torch.Tensor,
        select_count: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pool_indices = pool_indices.to(device=distance.device, dtype=torch.long)
        if select_count <= 0 or int(pool_indices.numel()) == 0:
            return (
                torch.empty(0, device=distance.device, dtype=torch.long),
                torch.empty(0, device=distance.device, dtype=torch.float32),
            )

        num_visual = int(saliency_score.numel())
        pool_mask = torch.zeros(num_visual, device=distance.device, dtype=torch.bool)
        pool_mask[pool_indices] = True
        saliency_desc = self._descending_indices_tensor(saliency_score)
        ordered_pool = saliency_desc[pool_mask.index_select(0, saliency_desc)]
        limit = min(int(select_count), int(ordered_pool.numel()))
        if limit <= 0:
            return (
                torch.empty(0, device=distance.device, dtype=torch.long),
                torch.empty(0, device=distance.device, dtype=torch.float32),
            )

        selected = torch.empty(limit, device=distance.device, dtype=torch.long)
        gains = torch.empty(limit, device=distance.device, dtype=torch.float32)
        selected_mask = torch.zeros(num_visual, device=distance.device, dtype=torch.bool)

        first = ordered_pool[0]
        selected[0] = first
        gains[0] = 0.0
        selected_mask[first] = True
        min_distance = distance.index_select(1, first.view(1)).flatten().to(dtype=torch.float32)
        selected_count = 1

        while selected_count < limit:
            candidate_mask = pool_mask & ~selected_mask
            best_idx = self._best_high_gain_saliency_low_index(candidate_mask, min_distance, saliency_score)
            selected[selected_count] = best_idx
            gains[selected_count] = min_distance[best_idx]
            selected_mask[best_idx] = True
            min_distance = torch.minimum(
                min_distance,
                distance.index_select(1, best_idx.view(1)).flatten().to(dtype=torch.float32),
            )
            selected_count += 1

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

    @staticmethod
    def _cpu_generator(seed: int) -> torch.Generator:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) % (2**63 - 1))
        return generator

    @staticmethod
    def _random_position(count: int, generator: torch.Generator) -> int:
        if count <= 0:
            raise ValueError("count must be positive for random selection")
        return int(torch.randint(int(count), (1,), generator=generator).item())

    def _derive_random_seed(
        self,
        context: Dict[str, object],
        layer_idx: int,
        global_prune_step: int,
        base_seed: int,
    ) -> int:
        sample_index = int(context.get("score_sample_index", 0))
        return int(base_seed + sample_index * 1_000_003 + int(layer_idx) * 9_176 + int(global_prune_step) * 131)

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

    def _repair_saliency_mass_gpu(
        self,
        selected_mask: torch.Tensor,
        seed_mask: torch.Tensor,
        saliency_score: torch.Tensor,
        distance: torch.Tensor,
        saliency_mass_floor: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[Dict[str, float | int]]]:
        positive_saliency = saliency_score.to(dtype=torch.float32).clamp_min(0.0)
        selected_mask = selected_mask.clone()
        seed_mask = seed_mask.to(device=selected_mask.device, dtype=torch.bool)
        repair_replacements: List[Dict[str, float | int]] = []

        selected_mass = positive_saliency[selected_mask].sum()
        while float((selected_mass + 1e-8 < saliency_mass_floor).item()):
            unselected_mask = ~selected_mask
            if not bool(unselected_mask.any().item()):
                break
            incoming = self._best_high_saliency_low_index(unselected_mask, positive_saliency)

            selected_indices = torch.nonzero(selected_mask, as_tuple=False).flatten()
            contributions = torch.zeros_like(positive_saliency)
            if int(selected_indices.numel()) > 1:
                selected_distance = distance.index_select(0, selected_indices).index_select(1, selected_indices)
                eye = torch.eye(int(selected_indices.numel()), device=distance.device, dtype=torch.bool)
                selected_distance = selected_distance.masked_fill(eye, float("inf"))
                contributions.index_copy_(0, selected_indices, selected_distance.min(dim=1).values)

            replaceable_mask = selected_mask & ~seed_mask
            if not bool(replaceable_mask.any().item()):
                replaceable_mask = selected_mask
            outgoing = self._best_low_saliency_low_contribution_high_index(
                replaceable_mask,
                positive_saliency,
                contributions,
            )

            incoming_saliency = positive_saliency[incoming]
            outgoing_saliency = positive_saliency[outgoing]
            if not bool((incoming_saliency > outgoing_saliency + 1e-8).item()):
                break

            selected_mask[outgoing] = False
            selected_mask[incoming] = True
            selected_mass = selected_mass + incoming_saliency - outgoing_saliency
            repair_replacements.append(
                {
                    "out": int(outgoing.item()),
                    "in": int(incoming.item()),
                    "mass_gain": float((incoming_saliency - outgoing_saliency).item()),
                }
            )

        return selected_mask, repair_replacements

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

    def _saliency_constrained_native_divprune_select_gpu(
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
        device = saliency_score.device
        if target_keep <= 0:
            empty_long = torch.empty(0, device=device, dtype=torch.long)
            return {
                "keep_indices": empty_long,
                "seed_ratio": 0.0,
                "seed_count": 0,
                "seed_pool_indices": empty_long,
                "seed_indices": empty_long,
                "mmr_selected_order": empty_long,
                "diversity_gain": torch.empty(0, device=device, dtype=torch.float32),
                "saliency_floor_eta": 0.0,
                "saliency_mass_floor": 0.0,
                "saliency_mass_selected": 0.0,
                "saliency_mass_topk": 0.0,
                "feasible_candidate_count": 0,
                "repair_replacements": [],
            }

        keep_count = min(int(target_keep), num_visual)
        entropy_norm = _clamp01(entropy_norm)
        positive_saliency = saliency_score.to(dtype=torch.float32).clamp_min(0.0)
        saliency_desc = self._descending_indices_tensor(saliency_score)
        topk_indices = saliency_desc[:keep_count]
        saliency_mass_topk = positive_saliency.index_select(0, topk_indices).sum()

        seed_ratio = seed_ratio_min + (seed_ratio_max - seed_ratio_min) * (1.0 - entropy_norm)
        seed_count = min(keep_count, int(math.ceil(float(keep_count) * seed_ratio)))
        if keep_count > 0 and seed_ratio > 0.0:
            seed_count = max(1, seed_count)
        seed_pool_count = min(num_visual, max(seed_count, int(math.ceil(float(seed_count) * seed_pool_multiplier))))
        seed_pool_indices = saliency_desc[:seed_pool_count]
        seed_indices, seed_gains = self._max_min_select_from_pool_gpu(
            distance=distance,
            saliency_score=saliency_score,
            pool_indices=seed_pool_indices,
            select_count=seed_count,
        )

        eta = saliency_floor_min + (saliency_floor_max - saliency_floor_min) * (1.0 - entropy_norm)
        saliency_mass_floor = saliency_mass_topk * float(eta)
        selected_mask = torch.zeros(num_visual, device=device, dtype=torch.bool)
        seed_mask = torch.zeros(num_visual, device=device, dtype=torch.bool)
        selected_order = torch.empty(keep_count, device=device, dtype=torch.long)
        diversity_gains = torch.zeros(keep_count, device=device, dtype=torch.float32)

        selected_count = int(seed_indices.numel())
        if selected_count > 0:
            selected_mask[seed_indices] = True
            seed_mask[seed_indices] = True
            selected_order[:selected_count] = seed_indices
            diversity_gains[:selected_count] = seed_gains
            min_distance = distance.index_select(1, seed_indices).min(dim=1).values.to(dtype=torch.float32)
        else:
            min_distance = torch.zeros(num_visual, device=device, dtype=torch.float32)
        selected_mass = positive_saliency[selected_mask].sum()
        feasible_candidate_count = 0

        while selected_count < keep_count:
            remaining_mask = ~selected_mask
            if not bool(remaining_mask.any().item()):
                break

            slots_after_candidate = keep_count - selected_count - 1
            if slots_after_candidate <= 0:
                possible_remaining = torch.zeros(num_visual, device=device, dtype=torch.float32)
            else:
                remaining_order = saliency_desc[remaining_mask.index_select(0, saliency_desc)]
                remaining_saliency = positive_saliency.index_select(0, remaining_order)
                if int(remaining_order.numel()) == 0:
                    possible_remaining = torch.zeros(num_visual, device=device, dtype=torch.float32)
                else:
                    rank_by_index = torch.full(
                        (num_visual,),
                        fill_value=num_visual + 1,
                        device=device,
                        dtype=torch.long,
                    )
                    rank_by_index.index_copy_(
                        0,
                        remaining_order,
                        torch.arange(int(remaining_order.numel()), device=device, dtype=torch.long),
                    )
                    top_s_count = min(slots_after_candidate, int(remaining_order.numel()))
                    top_s_sum = remaining_saliency[:top_s_count].sum()
                    top_s_plus_count = min(slots_after_candidate + 1, int(remaining_order.numel()))
                    top_s_plus_sum = remaining_saliency[:top_s_plus_count].sum()
                    candidate_in_top_s = rank_by_index < slots_after_candidate
                    possible_remaining = torch.where(
                        candidate_in_top_s,
                        top_s_plus_sum - positive_saliency,
                        top_s_sum,
                    )

            possible_mass = selected_mass + positive_saliency + possible_remaining
            feasible_mask = remaining_mask & (possible_mass + 1e-8 >= saliency_mass_floor)
            feasible_count = int(feasible_mask.sum().item())
            feasible_candidate_count += feasible_count
            candidate_mask = feasible_mask if feasible_count > 0 else remaining_mask

            gains = min_distance if selected_count > 0 else torch.zeros_like(min_distance)
            best_idx = self._best_high_gain_saliency_low_index(candidate_mask, gains, saliency_score)
            selected_order[selected_count] = best_idx
            diversity_gains[selected_count] = gains[best_idx]
            selected_mask[best_idx] = True
            selected_mass = selected_mass + positive_saliency[best_idx]
            if selected_count == 0:
                min_distance = distance.index_select(1, best_idx.view(1)).flatten().to(dtype=torch.float32)
            else:
                min_distance = torch.minimum(
                    min_distance,
                    distance.index_select(1, best_idx.view(1)).flatten().to(dtype=torch.float32),
                )
            selected_count += 1

        selected_order = selected_order[:selected_count]
        diversity_gains = diversity_gains[:selected_count]

        repair_replacements: List[Dict[str, float | int]] = []
        if saliency_repair and selected_count > 0:
            selected_mask, repair_replacements = self._repair_saliency_mass_gpu(
                selected_mask=selected_mask,
                seed_mask=seed_mask,
                saliency_score=saliency_score,
                distance=distance,
                saliency_mass_floor=saliency_mass_floor,
            )
            selected_mass = positive_saliency[selected_mask].sum()

        keep_indices = torch.nonzero(selected_mask, as_tuple=False).flatten()
        return {
            "keep_indices": keep_indices,
            "seed_ratio": float(seed_ratio),
            "seed_count": int(seed_count),
            "seed_pool_indices": seed_pool_indices.to(device=device, dtype=torch.long),
            "seed_indices": torch.sort(seed_indices).values,
            "mmr_selected_order": selected_order,
            "diversity_gain": diversity_gains,
            "saliency_floor_eta": float(eta),
            "saliency_mass_floor": float(saliency_mass_floor.item()),
            "saliency_mass_selected": float(selected_mass.item()),
            "saliency_mass_topk": float(saliency_mass_topk.item()),
            "feasible_candidate_count": int(feasible_candidate_count),
            "repair_replacements": repair_replacements,
        }

    def _saliency_constrained_native_divprune_select_triton(
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
        device = saliency_score.device
        if target_keep <= 0:
            empty_long = torch.empty(0, device=device, dtype=torch.long)
            return {
                "keep_indices": empty_long,
                "seed_ratio": 0.0,
                "seed_count": 0,
                "seed_pool_indices": empty_long,
                "seed_indices": empty_long,
                "mmr_selected_order": empty_long,
                "diversity_gain": torch.empty(0, device=device, dtype=torch.float32),
                "saliency_floor_eta": 0.0,
                "saliency_mass_floor": 0.0,
                "saliency_mass_selected": 0.0,
                "saliency_mass_topk": 0.0,
                "feasible_candidate_count": 0,
                "repair_replacements": [],
            }

        keep_count = min(int(target_keep), num_visual)
        entropy_norm = _clamp01(entropy_norm)
        positive_saliency = saliency_score.to(dtype=torch.float32).clamp_min(0.0)
        saliency_desc = self._descending_indices_tensor(saliency_score)
        topk_indices = saliency_desc[:keep_count]
        saliency_mass_topk = positive_saliency.index_select(0, topk_indices).sum()

        seed_ratio = seed_ratio_min + (seed_ratio_max - seed_ratio_min) * (1.0 - entropy_norm)
        seed_count = min(keep_count, int(math.ceil(float(keep_count) * seed_ratio)))
        if keep_count > 0 and seed_ratio > 0.0:
            seed_count = max(1, seed_count)
        seed_pool_count = min(num_visual, max(seed_count, int(math.ceil(float(seed_count) * seed_pool_multiplier))))
        seed_pool_indices = saliency_desc[:seed_pool_count]

        eta = saliency_floor_min + (saliency_floor_max - saliency_floor_min) * (1.0 - entropy_norm)
        saliency_mass_floor = saliency_mass_topk * float(eta)
        selected_mask, seed_mask, selected_order, diversity_gains, feasible_candidate_count = scnd_native_c_select_triton(
            distance=distance,
            saliency_score=saliency_score,
            saliency_desc=saliency_desc,
            keep_count=keep_count,
            seed_count=seed_count,
            seed_pool_count=seed_pool_count,
            saliency_mass_floor=saliency_mass_floor,
        )
        selected_mass = positive_saliency[selected_mask].sum()

        repair_replacements: List[Dict[str, float | int]] = []
        if saliency_repair and bool(selected_mask.any().item()):
            selected_mask, repair_replacements = self._repair_saliency_mass_gpu(
                selected_mask=selected_mask,
                seed_mask=seed_mask,
                saliency_score=saliency_score,
                distance=distance,
                saliency_mass_floor=saliency_mass_floor,
            )
            selected_mass = positive_saliency[selected_mask].sum()

        seed_indices = torch.nonzero(seed_mask, as_tuple=False).flatten()
        keep_indices = torch.nonzero(selected_mask, as_tuple=False).flatten()
        return {
            "keep_indices": keep_indices,
            "seed_ratio": float(seed_ratio),
            "seed_count": int(seed_count),
            "seed_pool_indices": seed_pool_indices.to(device=device, dtype=torch.long),
            "seed_indices": torch.sort(seed_indices).values,
            "mmr_selected_order": selected_order,
            "diversity_gain": diversity_gains,
            "saliency_floor_eta": float(eta),
            "saliency_mass_floor": float(saliency_mass_floor.item()),
            "saliency_mass_selected": float(selected_mass.item()),
            "saliency_mass_topk": float(saliency_mass_topk.item()),
            "feasible_candidate_count": int(feasible_candidate_count),
            "repair_replacements": repair_replacements,
        }

    def _saliency_constrained_random_feasible_select(
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
        random_seed: int,
    ) -> Dict[str, object]:
        num_visual = int(saliency_score.numel())
        if target_keep <= 0:
            empty_long = torch.empty(0, device=saliency_score.device, dtype=torch.long)
            return {
                "keep_indices": empty_long,
                "seed_ratio": 0.0,
                "seed_count": 0,
                "seed_pool_indices": empty_long,
                "seed_indices": empty_long,
                "mmr_selected_order": empty_long,
                "diversity_gain": torch.empty(0, device=saliency_score.device, dtype=torch.float32),
                "saliency_floor_eta": 0.0,
                "saliency_mass_floor": 0.0,
                "saliency_mass_selected": 0.0,
                "saliency_mass_topk": 0.0,
                "feasible_candidate_count": 0,
                "repair_replacements": [],
                "random_seed": int(random_seed),
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
        seed_pool_set = set(seed_pool_indices)

        eta = saliency_floor_min + (saliency_floor_max - saliency_floor_min) * (1.0 - entropy_norm)
        saliency_mass_floor = float(eta * saliency_mass_topk)
        generator = self._cpu_generator(random_seed)

        selected: List[int] = []
        seed_indices: List[int] = []
        selected_mask = [False for _ in range(num_visual)]
        selected_mass = 0.0
        selected_order: List[int] = []
        diversity_gains: List[float] = []
        feasible_candidate_count = 0

        def feasible_from(candidates: List[int], slots_after_candidate: int) -> List[int]:
            feasible: List[int] = []
            for idx in candidates:
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
            return feasible

        while len(selected) < keep_count:
            remaining = [idx for idx in range(num_visual) if not selected_mask[idx]]
            if not remaining:
                break

            in_seed_phase = len(seed_indices) < seed_count
            candidate_scope = [idx for idx in remaining if idx in seed_pool_set] if in_seed_phase else remaining
            if not candidate_scope:
                candidate_scope = remaining

            slots_after_candidate = keep_count - len(selected) - 1
            feasible = feasible_from(candidate_scope, slots_after_candidate)
            if in_seed_phase and not feasible:
                feasible = feasible_from(remaining, slots_after_candidate)
            feasible_candidate_count += len(feasible)
            candidate_pool = feasible if feasible else candidate_scope

            best_idx = int(candidate_pool[self._random_position(len(candidate_pool), generator)])
            if selected:
                selected_tensor = torch.tensor(selected, device=distance.device, dtype=torch.long)
                gain = float(distance.index_select(0, torch.tensor([best_idx], device=distance.device)).index_select(1, selected_tensor).min().item())
            else:
                gain = 0.0

            selected.append(best_idx)
            selected_mask[best_idx] = True
            selected_mass += max(saliency_values[best_idx], 0.0)
            selected_order.append(best_idx)
            diversity_gains.append(float(gain))
            if in_seed_phase:
                seed_indices.append(best_idx)

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
            "random_seed": int(random_seed),
        }

    def _saliency_constrained_random_feasible_select_gpu(
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
        random_seed: int,
    ) -> Dict[str, object]:
        num_visual = int(saliency_score.numel())
        device = saliency_score.device
        if target_keep <= 0:
            empty_long = torch.empty(0, device=device, dtype=torch.long)
            return {
                "keep_indices": empty_long,
                "seed_ratio": 0.0,
                "seed_count": 0,
                "seed_pool_indices": empty_long,
                "seed_indices": empty_long,
                "mmr_selected_order": empty_long,
                "diversity_gain": torch.empty(0, device=device, dtype=torch.float32),
                "saliency_floor_eta": 0.0,
                "saliency_mass_floor": 0.0,
                "saliency_mass_selected": 0.0,
                "saliency_mass_topk": 0.0,
                "feasible_candidate_count": 0,
                "repair_replacements": [],
                "random_seed": int(random_seed),
            }

        keep_count = min(int(target_keep), num_visual)
        entropy_norm = _clamp01(entropy_norm)
        positive_saliency = saliency_score.to(dtype=torch.float32).clamp_min(0.0)
        saliency_desc = self._descending_indices_tensor(saliency_score)
        topk_indices = saliency_desc[:keep_count]
        saliency_mass_topk = positive_saliency.index_select(0, topk_indices).sum()

        seed_ratio = seed_ratio_min + (seed_ratio_max - seed_ratio_min) * (1.0 - entropy_norm)
        seed_count = min(keep_count, int(math.ceil(float(keep_count) * seed_ratio)))
        if keep_count > 0 and seed_ratio > 0.0:
            seed_count = max(1, seed_count)
        seed_pool_count = min(num_visual, max(seed_count, int(math.ceil(float(seed_count) * seed_pool_multiplier))))
        seed_pool_indices = saliency_desc[:seed_pool_count]
        seed_pool_mask = torch.zeros(num_visual, device=device, dtype=torch.bool)
        if int(seed_pool_indices.numel()) > 0:
            seed_pool_mask[seed_pool_indices] = True

        eta = saliency_floor_min + (saliency_floor_max - saliency_floor_min) * (1.0 - entropy_norm)
        saliency_mass_floor = saliency_mass_topk * float(eta)
        generator = self._cpu_generator(random_seed)

        selected_mask = torch.zeros(num_visual, device=device, dtype=torch.bool)
        seed_mask = torch.zeros(num_visual, device=device, dtype=torch.bool)
        selected_order = torch.empty(keep_count, device=device, dtype=torch.long)
        diversity_gains = torch.zeros(keep_count, device=device, dtype=torch.float32)
        seed_indices = torch.empty(seed_count, device=device, dtype=torch.long)
        selected_count = 0
        seed_selected_count = 0
        selected_mass = positive_saliency[selected_mask].sum()
        min_distance = torch.zeros(num_visual, device=device, dtype=torch.float32)
        feasible_candidate_count = 0

        while selected_count < keep_count:
            remaining_mask = ~selected_mask
            if not bool(remaining_mask.any().item()):
                break

            slots_after_candidate = keep_count - selected_count - 1
            if slots_after_candidate <= 0:
                possible_remaining = torch.zeros(num_visual, device=device, dtype=torch.float32)
            else:
                remaining_order = saliency_desc[remaining_mask.index_select(0, saliency_desc)]
                remaining_saliency = positive_saliency.index_select(0, remaining_order)
                if int(remaining_order.numel()) == 0:
                    possible_remaining = torch.zeros(num_visual, device=device, dtype=torch.float32)
                else:
                    rank_by_index = torch.full(
                        (num_visual,),
                        fill_value=num_visual + 1,
                        device=device,
                        dtype=torch.long,
                    )
                    rank_by_index.index_copy_(
                        0,
                        remaining_order,
                        torch.arange(int(remaining_order.numel()), device=device, dtype=torch.long),
                    )
                    top_s_count = min(slots_after_candidate, int(remaining_order.numel()))
                    top_s_sum = remaining_saliency[:top_s_count].sum()
                    top_s_plus_count = min(slots_after_candidate + 1, int(remaining_order.numel()))
                    top_s_plus_sum = remaining_saliency[:top_s_plus_count].sum()
                    candidate_in_top_s = rank_by_index < slots_after_candidate
                    possible_remaining = torch.where(
                        candidate_in_top_s,
                        top_s_plus_sum - positive_saliency,
                        top_s_sum,
                    )

            possible_mass = selected_mass + positive_saliency + possible_remaining
            feasible_mask = remaining_mask & (possible_mass + 1e-8 >= saliency_mass_floor)
            in_seed_phase = seed_selected_count < seed_count
            scope_mask = (remaining_mask & seed_pool_mask) if in_seed_phase else remaining_mask
            if not bool(scope_mask.any().item()):
                scope_mask = remaining_mask

            candidate_mask = scope_mask & feasible_mask
            feasible_count = int(candidate_mask.sum().item())
            if in_seed_phase and feasible_count == 0:
                candidate_mask = feasible_mask
                feasible_count = int(candidate_mask.sum().item())
            feasible_candidate_count += feasible_count
            if feasible_count == 0:
                candidate_mask = scope_mask

            candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).flatten()
            best_idx = candidate_indices[self._random_position(int(candidate_indices.numel()), generator)]
            best_gain = min_distance[best_idx] if selected_count > 0 else torch.tensor(0.0, device=device)

            selected_order[selected_count] = best_idx
            diversity_gains[selected_count] = best_gain
            selected_mask[best_idx] = True
            selected_mass = selected_mass + positive_saliency[best_idx]
            if in_seed_phase:
                seed_indices[seed_selected_count] = best_idx
                seed_mask[best_idx] = True
                seed_selected_count += 1
            if selected_count == 0:
                min_distance = distance.index_select(1, best_idx.view(1)).flatten().to(dtype=torch.float32)
            else:
                min_distance = torch.minimum(
                    min_distance,
                    distance.index_select(1, best_idx.view(1)).flatten().to(dtype=torch.float32),
                )
            selected_count += 1

        selected_order = selected_order[:selected_count]
        diversity_gains = diversity_gains[:selected_count]
        seed_indices = seed_indices[:seed_selected_count]

        repair_replacements: List[Dict[str, float | int]] = []
        if saliency_repair and selected_count > 0:
            selected_mask, repair_replacements = self._repair_saliency_mass_gpu(
                selected_mask=selected_mask,
                seed_mask=seed_mask,
                saliency_score=saliency_score,
                distance=distance,
                saliency_mass_floor=saliency_mass_floor,
            )
            selected_mass = positive_saliency[selected_mask].sum()

        keep_indices = torch.nonzero(selected_mask, as_tuple=False).flatten()
        return {
            "keep_indices": keep_indices,
            "seed_ratio": float(seed_ratio),
            "seed_count": int(seed_count),
            "seed_pool_indices": seed_pool_indices.to(device=device, dtype=torch.long),
            "seed_indices": torch.sort(seed_indices).values,
            "mmr_selected_order": selected_order,
            "diversity_gain": diversity_gains,
            "saliency_floor_eta": float(eta),
            "saliency_mass_floor": float(saliency_mass_floor.item()),
            "saliency_mass_selected": float(selected_mass.item()),
            "saliency_mass_topk": float(saliency_mass_topk.item()),
            "feasible_candidate_count": int(feasible_candidate_count),
            "repair_replacements": repair_replacements,
            "random_seed": int(random_seed),
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

    def _boundary_refinement_select_gpu(
        self,
        saliency_score: torch.Tensor,
        margin_score: torch.Tensor,
        current_visual_embeds: torch.Tensor,
        target_keep: int,
        entropy_norm: float,
        boundary_ratio: float,
    ) -> Dict[str, object]:
        num_visual = int(saliency_score.numel())
        device = saliency_score.device
        keep_count = min(max(int(target_keep), 0), num_visual)
        if keep_count <= 0:
            empty = torch.empty(0, device=device, dtype=torch.long)
            return {
                "keep_indices": empty,
                "boundary_count": 0,
                "boundary_core_indices": empty,
                "boundary_pool_indices": empty,
                "boundary_fill_indices": empty,
                "margin_confidence": 1.0,
                "mmr_selected_order": empty,
                "diversity_gain": torch.empty(0, device=device, dtype=torch.float32),
                "distance_cost_proxy": 0,
            }

        order = self._descending_indices_tensor(saliency_score)
        margin_confidence = self._margin_confidence(margin_score, keep_count)
        raw_boundary = float(boundary_ratio) * float(keep_count) * _clamp01(entropy_norm) * (1.0 - margin_confidence)
        boundary_count = min(keep_count, int(math.ceil(raw_boundary)))

        if boundary_count <= 0:
            keep = torch.sort(order[:keep_count]).values
            return {
                "keep_indices": keep,
                "boundary_count": 0,
                "boundary_core_indices": keep,
                "boundary_pool_indices": torch.empty(0, device=device, dtype=torch.long),
                "boundary_fill_indices": torch.empty(0, device=device, dtype=torch.long),
                "margin_confidence": float(margin_confidence),
                "mmr_selected_order": keep,
                "diversity_gain": torch.zeros(int(keep.numel()), device=device, dtype=torch.float32),
                "distance_cost_proxy": 0,
            }

        core_count = keep_count - boundary_count
        core = order[:core_count]
        pool_count = min(num_visual - core_count, max(boundary_count, boundary_count * 2))
        boundary_pool = order[core_count:core_count + pool_count]
        normalized = F.normalize(current_visual_embeds.to(dtype=torch.float32), dim=-1, eps=1e-6)

        selected_mask = torch.zeros(num_visual, device=device, dtype=torch.bool)
        if int(core.numel()) > 0:
            selected_mask[core] = True
        active_pool_mask = torch.ones(int(boundary_pool.numel()), device=device, dtype=torch.bool)
        selected_order = torch.empty(keep_count, device=device, dtype=torch.long)
        diversity_gain = torch.zeros(keep_count, device=device, dtype=torch.float32)
        selected_count = int(core.numel())
        if selected_count > 0:
            selected_order[:selected_count] = core
            core_embeds = normalized.index_select(0, core)
            pool_embeds = normalized.index_select(0, boundary_pool)
            min_distance = torch.clamp((1.0 - pool_embeds @ core_embeds.transpose(0, 1)) / 2.0, min=0.0, max=1.0).min(dim=1).values
        else:
            min_distance = torch.zeros(int(boundary_pool.numel()), device=device, dtype=torch.float32)

        fills = torch.empty(boundary_count, device=device, dtype=torch.long)
        fill_gains = torch.zeros(boundary_count, device=device, dtype=torch.float32)
        fill_count = 0
        distance_cost_proxy = int(boundary_pool.numel()) * max(selected_count, 1)

        while fill_count < boundary_count and bool(active_pool_mask.any().item()):
            candidate_mask = torch.zeros(num_visual, device=device, dtype=torch.bool)
            candidate_indices = boundary_pool[active_pool_mask]
            candidate_mask[candidate_indices] = True
            gain_full = torch.zeros(num_visual, device=device, dtype=torch.float32)
            gain_full.index_copy_(0, boundary_pool, min_distance)
            best_idx = self._best_high_gain_saliency_low_index(candidate_mask, gain_full, saliency_score)
            best_gain = gain_full[best_idx]
            previous_selected_count = selected_count

            fills[fill_count] = best_idx
            fill_gains[fill_count] = best_gain
            selected_order[selected_count] = best_idx
            diversity_gain[selected_count] = best_gain
            selected_mask[best_idx] = True
            selected_count += 1
            fill_count += 1

            active_pool_mask = active_pool_mask & (boundary_pool != best_idx)
            if bool(active_pool_mask.any().item()):
                pool_embeds = normalized.index_select(0, boundary_pool)
                best_embed = normalized.index_select(0, best_idx.view(1)).transpose(0, 1)
                dist_to_best = torch.clamp((1.0 - (pool_embeds @ best_embed).flatten()) / 2.0, min=0.0, max=1.0)
                min_distance = dist_to_best if previous_selected_count == 0 else torch.minimum(min_distance, dist_to_best)
                distance_cost_proxy += int(boundary_pool.numel())

        if selected_count < keep_count:
            for idx_tensor in order:
                idx = int(idx_tensor.item())
                if bool(selected_mask[idx].item()):
                    continue
                selected_order[selected_count] = idx_tensor
                selected_mask[idx_tensor] = True
                selected_count += 1
                if selected_count == keep_count:
                    break

        keep = torch.nonzero(selected_mask, as_tuple=False).flatten()
        return {
            "keep_indices": keep,
            "boundary_count": int(boundary_count),
            "boundary_core_indices": torch.sort(core).values,
            "boundary_pool_indices": boundary_pool,
            "boundary_fill_indices": torch.sort(fills[:fill_count]).values,
            "margin_confidence": float(margin_confidence),
            "mmr_selected_order": selected_order[:selected_count],
            "diversity_gain": diversity_gain[:selected_count],
            "distance_cost_proxy": int(distance_cost_proxy),
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
        requested_backend = str(params["selection_backend"])
        c_selection_rule = str(params["c_selection_rule"])
        if requested_backend == "triton":
            if not current_visual_embeds.is_cuda:
                raise RuntimeError("sparsevlm_scnd.selection_backend='triton' requires CUDA tensors")
            if not is_scnd_triton_available():
                raise RuntimeError("sparsevlm_scnd.selection_backend='triton' requires a working Triton CUDA install")
            if mode == "C" and c_selection_rule != "native":
                raise ValueError(
                    "sparsevlm_scnd.selection_backend='triton' currently supports only "
                    "c_selection_rule='native'"
                )
            selection_backend_effective = "triton" if mode == "C" else "gpu"
        elif requested_backend == "python":
            selection_backend_effective = "python"
        elif requested_backend == "gpu" and current_visual_embeds.is_cuda:
            selection_backend_effective = "gpu"
        elif requested_backend == "auto" and current_visual_embeds.is_cuda:
            selection_backend_effective = "gpu"
        else:
            selection_backend_effective = "python"

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
        global_prune_step = int(memory["global_prune_step"])
        random_seed = self._derive_random_seed(context, layer_idx, global_prune_step, int(params["seed"]))
        entropy_raw, entropy_norm = compute_entropy_stats(visual_scores)
        prune_ratio = self.get_prune_ratio(layer_idx, visual_scores)
        min_visual_tokens_after_prune = int(self.config.get("min_visual_tokens_after_prune", 16))
        target_keep = compute_target_keep_count(
            num_visual=v_token_num,
            prune_ratio=prune_ratio,
            min_visual_tokens_after_prune=min_visual_tokens_after_prune,
        )
        distance: Optional[torch.Tensor] = None
        distance_time_ms = 0.0
        selection_start = 0.0
        selection_time_ms = 0.0
        distance_cost_proxy = 0

        needs_full_distance = selection_backend_effective == "python" or mode == "C"
        if needs_full_distance:
            _sync_if_cuda(current_visual_embeds)
            distance_start = time.perf_counter()
            distance = _scnd_distance_matrix(current_visual_embeds, str(params["distance_metric"])).to(
                device=mixed_score.device
            )
            _sync_if_cuda(distance)
            distance_time_ms = (time.perf_counter() - distance_start) * 1000.0
            distance_cost_proxy = int(v_token_num) * int(v_token_num)

        if mode == "C":
            assert distance is not None
            _sync_if_cuda(distance)
            selection_start = time.perf_counter()
            if c_selection_rule == "random_feasible" and selection_backend_effective == "gpu":
                selection = self._saliency_constrained_random_feasible_select_gpu(
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
                    random_seed=random_seed,
                )
            elif c_selection_rule == "random_feasible":
                selection = self._saliency_constrained_random_feasible_select(
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
                    random_seed=random_seed,
                )
            elif selection_backend_effective == "triton":
                selection = self._saliency_constrained_native_divprune_select_triton(
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
            elif selection_backend_effective == "gpu":
                selection = self._saliency_constrained_native_divprune_select_gpu(
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
            else:
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
            _sync_if_cuda(distance)
            selection_time_ms = (time.perf_counter() - selection_start) * 1000.0
            selection_rule = (
                "saliency_constrained_random_feasible"
                if c_selection_rule == "random_feasible"
                else "saliency_constrained_native_divprune"
            )
        elif mode == "B":
            _sync_if_cuda(current_visual_embeds)
            selection_start = time.perf_counter()
            if selection_backend_effective == "gpu":
                selection = self._boundary_refinement_select_gpu(
                    saliency_score=mixed_score,
                    margin_score=visual_scores,
                    current_visual_embeds=current_visual_embeds,
                    target_keep=target_keep,
                    entropy_norm=entropy_norm,
                    boundary_ratio=float(params["boundary_ratio"]),
                )
                distance_cost_proxy = int(selection.get("distance_cost_proxy", 0))
            else:
                assert distance is not None
                selection = self._boundary_refinement_select(
                    saliency_score=mixed_score,
                    margin_score=visual_scores,
                    distance=distance,
                    target_keep=target_keep,
                    entropy_norm=entropy_norm,
                    boundary_ratio=float(params["boundary_ratio"]),
                )
            _sync_if_cuda(current_visual_embeds)
            selection_time_ms = (time.perf_counter() - selection_start) * 1000.0
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
            "distance_metric": params["distance_metric"],
            "saliency_entropy": entropy_raw,
            "saliency_entropy_norm": entropy_norm,
            "seed_ratio_min": params["seed_ratio_min"],
            "seed_ratio_max": params["seed_ratio_max"],
            "seed_pool_multiplier": params["seed_pool_multiplier"],
            "seed": params["seed"],
            "saliency_floor_min": params["saliency_floor_min"],
            "saliency_floor_max": params["saliency_floor_max"],
            "boundary_ratio": params["boundary_ratio"],
            "saliency_repair": params["saliency_repair"],
            "c_selection_rule": params["c_selection_rule"],
            "selection_backend_requested": requested_backend,
            "selection_backend_effective": selection_backend_effective,
            "distance_time_ms": float(distance_time_ms),
            "selection_time_ms": float(selection_time_ms),
            "distance_cost_proxy": int(distance_cost_proxy),
            "mean_selected_pairwise_distance": (
                _selected_mean_pairwise_distance(distance, keep_indices)
                if distance is not None
                else _selected_mean_pairwise_distance_from_embeds(current_visual_embeds, keep_indices)
            ),
            "retained_saliency_mass_ratio": _saliency_mass_ratio(visual_scores, keep_indices),
            "global_prune_step": global_prune_step,
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
