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

from .scnd_entropy_calibration import (
    VALID_CALIBRATION_MODES,
    apply_entropy_calibration,
    load_calibration_artifact,
)
from .scnd_herc import (
    VALID_CONTINUITY_REFERENCES,
    VALID_PROFILE_LOCKS,
    VALID_RECONCILIATION_MODES,
    VALID_RECONCILIATION_STAGES,
    normalize_rater_distributions,
    profile_locked_pressure,
    reconcile_evidence,
    restrict_reference_distributions,
)
from .scnd_visual_roles import (
    ROLE_LOCAL_PATCH,
    visual_token_role_ids,
    visual_token_structure_ids,
)
from .sparsevlm_adaptive_stratified import compute_target_keep_count
from .sparsevlm_diverse_mmr import (
    SparseVLMDiverseMMRStrategy,
    _cosine_distance_matrix,
    _saliency_mass_ratio,
    _selected_mean_pairwise_distance,
)
from .sparsevlm_score_memory import compute_entropy_stats, deterministic_descending_indices, rank_normalize_scores


VALID_LAYER_MODES = {"C", "B", "S"}
VALID_SELECTION_BACKENDS = {"auto", "gpu", "python"}
VALID_C_SELECTION_RULES = {"native", "random_feasible"}
VALID_DISTANCE_METRICS = {"cosine", "euclidean", "dot"}
VALID_VISUAL_ROLE_CONSTRAINT_MODES = {
    "none",
    "local_floor",
    "budget_adaptive_local_floor",
    "budget_adaptive_saliency_gated_local_floor",
}


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

    def _get_evidence_reconciliation_params(self) -> Dict[str, object]:
        cfg = self.config.get("evidence_reconciliation", {}) or {}
        if not isinstance(cfg, dict):
            raise ValueError("sparsevlm_scnd.evidence_reconciliation must be a mapping")
        mode = str(cfg.get("mode", "none")).lower()
        apply_modes = cfg.get("apply_modes", ["C", "B", "S"])
        if isinstance(apply_modes, str):
            apply_modes = [value.strip() for value in apply_modes.split(",") if value.strip()]
        apply_modes = [str(value).upper() for value in apply_modes]
        profile_lock = str(cfg.get("profile_lock", "first_c")).lower()
        continuity_reference = str(cfg.get("continuity_reference", "first_c")).lower()
        candidate_pool_multiplier = cfg.get("candidate_pool_multiplier")
        max_swap_ratio = cfg.get("max_swap_ratio")
        raw_stages = cfg.get("stages", ["query", "context"])
        if isinstance(raw_stages, str):
            raw_stages = [value.strip() for value in raw_stages.split(",") if value.strip()]
        requested_stages = {str(value).lower() for value in raw_stages}

        if mode not in VALID_RECONCILIATION_MODES:
            raise ValueError(
                "sparsevlm_scnd.evidence_reconciliation.mode only supports "
                f"{sorted(VALID_RECONCILIATION_MODES)}, got {mode!r}"
            )
        invalid_modes = [value for value in apply_modes if value not in VALID_LAYER_MODES]
        if invalid_modes:
            raise ValueError(
                "sparsevlm_scnd.evidence_reconciliation.apply_modes only supports "
                f"{sorted(VALID_LAYER_MODES)}, got {invalid_modes}"
            )
        if profile_lock not in VALID_PROFILE_LOCKS:
            raise ValueError(
                "sparsevlm_scnd.evidence_reconciliation.profile_lock only supports "
                f"{sorted(VALID_PROFILE_LOCKS)}, got {profile_lock!r}"
            )
        if continuity_reference not in VALID_CONTINUITY_REFERENCES:
            raise ValueError(
                "sparsevlm_scnd.evidence_reconciliation.continuity_reference only supports "
                f"{sorted(VALID_CONTINUITY_REFERENCES)}, got {continuity_reference!r}"
            )
        if candidate_pool_multiplier is not None and float(candidate_pool_multiplier) < 1.0:
            raise ValueError("evidence_reconciliation.candidate_pool_multiplier must be >= 1 or null")
        if max_swap_ratio is not None and not 0.0 <= float(max_swap_ratio) <= 1.0:
            raise ValueError("evidence_reconciliation.max_swap_ratio must be in [0, 1] or null")
        invalid_stages = sorted(requested_stages - VALID_RECONCILIATION_STAGES)
        if invalid_stages:
            raise ValueError(
                "sparsevlm_scnd.evidence_reconciliation.stages only supports "
                f"{sorted(VALID_RECONCILIATION_STAGES)}, got {invalid_stages}"
            )
        if mode in {"herc_v1", "herc_v2", "herc_v3"} and not requested_stages:
            raise ValueError("Active HERC requires at least one evidence_reconciliation stage")
        if mode != "none":
            configured_modes = list(self._layer_mode_map().values())
            if "C" not in configured_modes:
                raise ValueError("Active evidence reconciliation requires a C layer for profile locking")
            if "C" not in apply_modes:
                raise ValueError("Active evidence reconciliation requires C in apply_modes")
            if str(self.config.get("c_selection_rule", "native")).lower() != "native":
                raise ValueError("Active evidence reconciliation requires c_selection_rule=native")

        return {
            "mode": mode,
            "apply_modes": tuple(apply_modes),
            "profile_lock": profile_lock,
            "continuity_reference": continuity_reference,
            "candidate_pool_multiplier": (
                None if candidate_pool_multiplier is None else float(candidate_pool_multiplier)
            ),
            "max_swap_ratio": None if max_swap_ratio is None else float(max_swap_ratio),
            "stages": tuple(
                stage for stage in ("query", "context") if stage in requested_stages
            ),
        }

    def _evidence_profile(self, params: Dict[str, object]) -> Dict[str, float | int]:
        mode_map = self._layer_mode_map()
        first_c_layer = next(layer for layer, mode in mode_map.items() if mode == "C")
        ratio_map = self.config.get("prune_ratio_map") or {}
        if int(first_c_layer) in ratio_map:
            removal_ratio = float(ratio_map[int(first_c_layer)])
        else:
            removal_ratio = float(self.config.get("prune_ratio", 0.5))
        nominal_keep_fraction = _clamp01(1.0 - removal_ratio)
        keep_fraction_low = float(params["entropy_calibration_budget_keep_fraction_low"])
        keep_fraction_high = float(params["entropy_calibration_budget_keep_fraction_high"])
        return {
            "first_c_layer": int(first_c_layer),
            "keep_fraction": nominal_keep_fraction,
            "pressure": profile_locked_pressure(
                nominal_keep_fraction,
                keep_fraction_low,
                keep_fraction_high,
            ),
        }

    @staticmethod
    def _evidence_bypass_stats(
        mode: str,
        profile: Dict[str, float | int],
        stages: Tuple[str, ...] = ("query", "context"),
    ) -> Dict[str, object]:
        return {
            "evidence_reconcile_mode": mode,
            "evidence_reconcile_stages": ",".join(stages),
            "evidence_reconcile_applied": False,
            "evidence_reconcile_bypassed": True,
            "evidence_reconcile_profile_pressure": float(profile["pressure"]),
            "evidence_reconcile_profile_keep_fraction": float(profile["keep_fraction"]),
            "evidence_reconcile_candidate_count": 0,
            "evidence_reconcile_swap_budget": 0,
            "evidence_reconcile_query_swap_count": 0,
            "evidence_reconcile_context_swap_count": 0,
            "evidence_reconcile_query_budget": 0,
            "evidence_reconcile_context_budget": 0,
            "evidence_reconcile_max_total_swaps": 0,
            "evidence_query_deficit": 0.0,
            "evidence_hierarchy_deficit": 0.0,
            "evidence_representative_deficit": 0.0,
            "evidence_context_co_deficit": 0.0,
            "evidence_anchor_count_before": 0,
            "evidence_anchor_count_after_query": 0,
            "evidence_anchor_fraction_before": 0.0,
            "evidence_anchor_fraction_after_query": 0.0,
            "evidence_pareto_safe": mode == "herc_v3",
            "evidence_reconcile_time_ms": 0.0,
        }

    def _get_scnd_params(self) -> Dict[str, object]:
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
        entropy_calibration_cfg = self.config.get("entropy_calibration", {}) or {}
        if not isinstance(entropy_calibration_cfg, dict):
            raise ValueError("sparsevlm_scnd.entropy_calibration must be a mapping")
        entropy_calibration_mode = str(entropy_calibration_cfg.get("mode", "identity")).lower()
        entropy_calibration_path = entropy_calibration_cfg.get("artifact_path")
        budget_keep_fraction_low = float(entropy_calibration_cfg.get("budget_keep_fraction_low", 0.125))
        budget_keep_fraction_high = float(entropy_calibration_cfg.get("budget_keep_fraction_high", 0.5))
        diversity_tail_gain = float(entropy_calibration_cfg.get("diversity_tail_gain", 1.0))
        visual_role_cfg = self.config.get("visual_role_constraint", {}) or {}
        if not isinstance(visual_role_cfg, dict):
            raise ValueError("sparsevlm_scnd.visual_role_constraint must be a mapping")
        visual_role_mode = str(visual_role_cfg.get("mode", "none")).lower()
        local_floor_ratio = float(visual_role_cfg.get("local_floor_ratio", 1.0))
        role_budget_keep_fraction_low = float(visual_role_cfg.get("budget_keep_fraction_low", 0.125))
        role_budget_keep_fraction_high = float(visual_role_cfg.get("budget_keep_fraction_high", 0.5))
        role_max_visual_tokens_before = int(visual_role_cfg.get("max_visual_tokens_before", 2200))
        role_min_local_saliency_mass_ratio = float(
            visual_role_cfg.get("min_local_saliency_mass_ratio", 0.70)
        )
        evidence_params = self._get_evidence_reconciliation_params()

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
        if entropy_calibration_mode not in VALID_CALIBRATION_MODES:
            raise ValueError(
                "sparsevlm_scnd.entropy_calibration.mode only supports "
                f"{sorted(VALID_CALIBRATION_MODES)}, got {entropy_calibration_mode!r}"
            )
        if entropy_calibration_mode != "identity" and not entropy_calibration_path:
            raise ValueError(f"{entropy_calibration_mode} entropy calibration requires artifact_path")
        if not 0.0 < budget_keep_fraction_low < budget_keep_fraction_high <= 1.0:
            raise ValueError(
                "entropy_calibration budget bounds must satisfy 0 < low < high <= 1, got "
                f"{budget_keep_fraction_low}, {budget_keep_fraction_high}"
            )
        if diversity_tail_gain < 0.0:
            raise ValueError(f"entropy_calibration.diversity_tail_gain must be >= 0, got {diversity_tail_gain}")
        if visual_role_mode not in VALID_VISUAL_ROLE_CONSTRAINT_MODES:
            raise ValueError(
                "sparsevlm_scnd.visual_role_constraint.mode only supports "
                f"{sorted(VALID_VISUAL_ROLE_CONSTRAINT_MODES)}, got {visual_role_mode!r}"
            )
        if not 0.0 <= local_floor_ratio <= 1.0:
            raise ValueError(
                "sparsevlm_scnd.visual_role_constraint.local_floor_ratio must be in [0, 1], "
                f"got {local_floor_ratio}"
            )
        if not 0.0 < role_budget_keep_fraction_low < role_budget_keep_fraction_high <= 1.0:
            raise ValueError(
                "visual_role_constraint budget bounds must satisfy 0 < low < high <= 1, got "
                f"{role_budget_keep_fraction_low}, {role_budget_keep_fraction_high}"
            )
        if role_max_visual_tokens_before <= 0:
            raise ValueError(
                "sparsevlm_scnd.visual_role_constraint.max_visual_tokens_before must be positive, "
                f"got {role_max_visual_tokens_before}"
            )
        if not 0.0 <= role_min_local_saliency_mass_ratio <= 1.0:
            raise ValueError(
                "sparsevlm_scnd.visual_role_constraint.min_local_saliency_mass_ratio must be in [0, 1], "
                f"got {role_min_local_saliency_mass_ratio}"
            )
        if visual_role_mode != "none" and c_selection_rule != "native":
            raise ValueError("visual_role_constraint is only supported with c_selection_rule=native")

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
            "entropy_calibration_mode": entropy_calibration_mode,
            "entropy_calibration_path": entropy_calibration_path,
            "entropy_calibration_budget_keep_fraction_low": budget_keep_fraction_low,
            "entropy_calibration_budget_keep_fraction_high": budget_keep_fraction_high,
            "entropy_calibration_diversity_tail_gain": diversity_tail_gain,
            "visual_role_constraint_mode": visual_role_mode,
            "visual_role_local_floor_ratio": local_floor_ratio,
            "visual_role_budget_keep_fraction_low": role_budget_keep_fraction_low,
            "visual_role_budget_keep_fraction_high": role_budget_keep_fraction_high,
            "visual_role_max_visual_tokens_before": role_max_visual_tokens_before,
            "visual_role_min_local_saliency_mass_ratio": role_min_local_saliency_mass_ratio,
            "evidence_reconciliation_mode": evidence_params["mode"],
            "evidence_reconciliation_apply_modes": evidence_params["apply_modes"],
            "evidence_reconciliation_profile_lock": evidence_params["profile_lock"],
            "evidence_reconciliation_continuity_reference": evidence_params["continuity_reference"],
            "evidence_reconciliation_candidate_pool_multiplier": evidence_params[
                "candidate_pool_multiplier"
            ],
            "evidence_reconciliation_max_swap_ratio": evidence_params["max_swap_ratio"],
            "evidence_reconciliation_stages": evidence_params["stages"],
        }

    @staticmethod
    def _visual_role_local_saliency_mass_ratio(
        visual_scores: torch.Tensor,
        visual_role_ids: torch.Tensor,
    ) -> float:
        scores = visual_scores.detach().to(device="cpu", dtype=torch.float64).flatten().clamp_min(0.0)
        roles = visual_role_ids.detach().to(device="cpu", dtype=torch.long).flatten()
        if int(scores.numel()) != int(roles.numel()):
            raise ValueError(
                "visual_scores and visual_role_ids must have the same length, "
                f"got {scores.numel()} vs {roles.numel()}"
            )
        total_mass = float(scores.sum().item())
        if total_mass <= 0.0:
            return 0.0
        local_mass = float(scores[roles == ROLE_LOCAL_PATCH].sum().item())
        return float(local_mass / total_mass)

    @staticmethod
    def _visual_role_constraint_control(
        mode: str,
        local_floor_ratio: float,
        keep_fraction: float,
        keep_fraction_low: float,
        keep_fraction_high: float,
        num_visual_tokens_before: Optional[int] = None,
        local_saliency_mass_ratio: Optional[float] = None,
        max_visual_tokens_before: int = 2200,
        min_local_saliency_mass_ratio: float = 0.70,
    ) -> Dict[str, float | int | str | bool | None]:
        keep_fraction = _clamp01(keep_fraction)
        layout_gate_passed = True
        saliency_gate_passed = True
        gate_passed = True
        if mode == "none":
            pressure = 0.0
        elif mode == "local_floor":
            pressure = 1.0
        elif mode in {
            "budget_adaptive_local_floor",
            "budget_adaptive_saliency_gated_local_floor",
        }:
            pressure = _clamp01(
                (float(keep_fraction_high) - keep_fraction)
                / (float(keep_fraction_high) - float(keep_fraction_low))
            )
            if mode == "budget_adaptive_saliency_gated_local_floor":
                layout_gate_passed = (
                    num_visual_tokens_before is not None
                    and int(num_visual_tokens_before) <= int(max_visual_tokens_before)
                )
                saliency_gate_passed = (
                    local_saliency_mass_ratio is not None
                    and float(local_saliency_mass_ratio) >= float(min_local_saliency_mass_ratio)
                )
                gate_passed = bool(layout_gate_passed and saliency_gate_passed)
                if not gate_passed:
                    pressure = 0.0
        else:
            raise ValueError(f"Unsupported visual role constraint mode: {mode}")
        return {
            "mode": str(mode),
            "applied": bool(mode != "none" and pressure > 0.0),
            "budget_keep_fraction": float(keep_fraction),
            "budget_pressure": float(pressure),
            "local_floor_ratio": float(local_floor_ratio),
            "local_floor_ratio_effective": float(local_floor_ratio * pressure),
            "budget_keep_fraction_low": float(keep_fraction_low),
            "budget_keep_fraction_high": float(keep_fraction_high),
            "num_visual_tokens_before": (
                int(num_visual_tokens_before) if num_visual_tokens_before is not None else None
            ),
            "local_saliency_mass_ratio": (
                float(local_saliency_mass_ratio) if local_saliency_mass_ratio is not None else None
            ),
            "max_visual_tokens_before": int(max_visual_tokens_before),
            "min_local_saliency_mass_ratio": float(min_local_saliency_mass_ratio),
            "layout_gate_passed": bool(layout_gate_passed),
            "saliency_gate_passed": bool(saliency_gate_passed),
            "gate_passed": bool(gate_passed),
        }

    def _entropy_control(
        self,
        raw_entropy_norm: float,
        layer_idx: int,
        layer_mode: str,
        params: Dict[str, object],
        keep_fraction: Optional[float] = None,
    ) -> Dict[str, object]:
        calibration_mode = str(params["entropy_calibration_mode"])
        if layer_mode != "C" or calibration_mode == "identity":
            return apply_entropy_calibration(raw_entropy_norm, "identity", None, keep_fraction=keep_fraction)

        artifact = getattr(self, "_entropy_calibration_artifact", None)
        if artifact is None:
            artifact = load_calibration_artifact(
                str(params["entropy_calibration_path"]),
                expected_model_name=self.config.get("_model_name"),
                expected_model_fingerprint=self.config.get("_model_config_fingerprint"),
                expected_layer=int(layer_idx),
            )
            self._entropy_calibration_artifact = artifact
        elif int(artifact["capture_layer"]) != int(layer_idx):
            raise ValueError(
                "Entropy calibration artifact is bound to layer "
                f"{artifact['capture_layer']}, but C selection was requested at layer {layer_idx}"
            )
        return apply_entropy_calibration(
            raw_entropy_norm,
            calibration_mode,
            artifact,
            keep_fraction=keep_fraction,
            budget_keep_fraction_low=float(params["entropy_calibration_budget_keep_fraction_low"]),
            budget_keep_fraction_high=float(params["entropy_calibration_budget_keep_fraction_high"]),
            diversity_tail_gain=float(params["entropy_calibration_diversity_tail_gain"]),
        )

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
    def _local_floor_count(
        saliency_desc: torch.Tensor,
        keep_count: int,
        visual_role_ids: Optional[torch.Tensor],
        local_floor_ratio: float,
    ) -> Tuple[int, int]:
        if visual_role_ids is None or local_floor_ratio <= 0.0 or keep_count <= 0:
            return 0, 0
        role_ids = visual_role_ids.to(device=saliency_desc.device, dtype=torch.long)
        topk = saliency_desc[: int(keep_count)]
        topk_local_count = int((role_ids.index_select(0, topk) == ROLE_LOCAL_PATCH).sum().item())
        floor_count = min(
            topk_local_count,
            int(math.ceil(float(topk_local_count) * float(local_floor_ratio) - 1e-12)),
        )
        return topk_local_count, floor_count

    @staticmethod
    def _rebalance_seed_local_capacity_gpu(
        seed_indices: torch.Tensor,
        seed_gains: torch.Tensor,
        saliency_desc: torch.Tensor,
        visual_role_ids: Optional[torch.Tensor],
        keep_count: int,
        local_floor_count: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if visual_role_ids is None or local_floor_count <= 0 or int(seed_indices.numel()) == 0:
            return seed_indices, seed_gains

        role_ids = visual_role_ids.to(device=seed_indices.device, dtype=torch.long)
        local_mask = role_ids == ROLE_LOCAL_PATCH
        seed_local_count = int(local_mask.index_select(0, seed_indices).sum().item())
        required_seed_local = max(0, int(local_floor_count) - (int(keep_count) - int(seed_indices.numel())))
        deficit = required_seed_local - seed_local_count
        if deficit <= 0:
            return seed_indices, seed_gains

        selected_mask = torch.zeros(int(role_ids.numel()), device=seed_indices.device, dtype=torch.bool)
        selected_mask[seed_indices] = True
        incoming = saliency_desc[
            local_mask.index_select(0, saliency_desc) & ~selected_mask.index_select(0, saliency_desc)
        ][:deficit]
        outgoing_positions = torch.nonzero(
            ~local_mask.index_select(0, seed_indices), as_tuple=False
        ).flatten().flip(0)[:deficit]
        if int(incoming.numel()) != deficit or int(outgoing_positions.numel()) != deficit:
            raise AssertionError("Could not preserve enough local-patch capacity during SCND seed selection")

        adjusted_indices = seed_indices.clone()
        adjusted_gains = seed_gains.clone()
        adjusted_indices[outgoing_positions] = incoming
        adjusted_gains[outgoing_positions] = 0.0
        return adjusted_indices, adjusted_gains

    @staticmethod
    def _rebalance_seed_local_capacity(
        seed_indices: List[int],
        seed_gains: List[float],
        saliency_desc: List[int],
        visual_role_ids: Optional[List[int]],
        keep_count: int,
        local_floor_count: int,
    ) -> Tuple[List[int], List[float]]:
        if visual_role_ids is None or local_floor_count <= 0 or not seed_indices:
            return seed_indices, seed_gains

        required_seed_local = max(0, int(local_floor_count) - (int(keep_count) - len(seed_indices)))
        seed_local_count = sum(visual_role_ids[int(index)] == ROLE_LOCAL_PATCH for index in seed_indices)
        deficit = required_seed_local - seed_local_count
        if deficit <= 0:
            return seed_indices, seed_gains

        selected = set(int(index) for index in seed_indices)
        incoming = [
            int(index)
            for index in saliency_desc
            if visual_role_ids[int(index)] == ROLE_LOCAL_PATCH and int(index) not in selected
        ][:deficit]
        outgoing_positions = [
            position
            for position in range(len(seed_indices) - 1, -1, -1)
            if visual_role_ids[int(seed_indices[position])] != ROLE_LOCAL_PATCH
        ][:deficit]
        if len(incoming) != deficit or len(outgoing_positions) != deficit:
            raise AssertionError("Could not preserve enough local-patch capacity during SCND seed selection")

        adjusted_indices = list(seed_indices)
        adjusted_gains = list(seed_gains)
        for position, incoming_index in zip(outgoing_positions, incoming):
            adjusted_indices[position] = int(incoming_index)
            adjusted_gains[position] = 0.0
        return adjusted_indices, adjusted_gains

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
        visual_role_ids: Optional[List[int]] = None,
        local_floor_count: int = 0,
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
            if visual_role_ids is not None and local_floor_count > 0:
                selected_local_count = sum(
                    visual_role_ids[int(index)] == ROLE_LOCAL_PATCH for index in selected_set
                )
                incoming_is_local = visual_role_ids[int(incoming)] == ROLE_LOCAL_PATCH
                replaceable = [
                    index
                    for index in replaceable
                    if (
                        selected_local_count
                        - int(visual_role_ids[int(index)] == ROLE_LOCAL_PATCH)
                        + int(incoming_is_local)
                        >= int(local_floor_count)
                    )
                ]
                if not replaceable and not incoming_is_local:
                    local_incoming = [
                        index
                        for index in unselected
                        if visual_role_ids[int(index)] == ROLE_LOCAL_PATCH
                    ]
                    if local_incoming:
                        incoming = max(local_incoming, key=lambda idx: (saliency_values[idx], -idx))
                        replaceable = [idx for idx in selected_set if idx not in seed_set] or list(selected_set)
                if not replaceable:
                    break
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
        visual_role_ids: Optional[torch.Tensor] = None,
        local_floor_count: int = 0,
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
            if visual_role_ids is not None and local_floor_count > 0:
                role_ids = visual_role_ids.to(device=selected_mask.device, dtype=torch.long)
                local_mask = role_ids == ROLE_LOCAL_PATCH
                selected_local_count = int((selected_mask & local_mask).sum().item())
                incoming_is_local = bool(local_mask[incoming].item())
                if not incoming_is_local and selected_local_count <= int(local_floor_count):
                    replaceable_mask = replaceable_mask & ~local_mask
                if not bool(replaceable_mask.any().item()) and not incoming_is_local:
                    local_incoming_mask = unselected_mask & local_mask
                    if bool(local_incoming_mask.any().item()):
                        incoming = self._best_high_saliency_low_index(local_incoming_mask, positive_saliency)
                        replaceable_mask = selected_mask & ~seed_mask
                        if not bool(replaceable_mask.any().item()):
                            replaceable_mask = selected_mask
                if not bool(replaceable_mask.any().item()):
                    break
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
        visual_role_ids: Optional[torch.Tensor] = None,
        local_floor_ratio: float = 0.0,
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
                "topk_local_patch_count": 0,
                "local_patch_floor_count": 0,
                "selected_local_patch_count": 0,
            }

        keep_count = min(int(target_keep), num_visual)
        saliency_values = self._score_list(saliency_score)
        saliency_desc = deterministic_descending_indices(saliency_score).detach().cpu().tolist()
        topk_indices = [int(idx) for idx in saliency_desc[:keep_count]]
        saliency_mass_topk = self._saliency_sum(saliency_values, topk_indices)
        role_values = (
            [int(value) for value in visual_role_ids.detach().cpu().tolist()]
            if visual_role_ids is not None
            else None
        )
        topk_local_count = (
            sum(role_values[int(index)] == ROLE_LOCAL_PATCH for index in topk_indices)
            if role_values is not None
            else 0
        )
        local_floor_count = min(
            topk_local_count,
            int(math.ceil(float(topk_local_count) * float(local_floor_ratio) - 1e-12)),
        )

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
        seed_indices, seed_gains = self._rebalance_seed_local_capacity(
            seed_indices=seed_indices,
            seed_gains=seed_gains,
            saliency_desc=saliency_desc,
            visual_role_ids=role_values,
            keep_count=keep_count,
            local_floor_count=local_floor_count,
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
            selected_local_count = (
                sum(role_values[int(index)] == ROLE_LOCAL_PATCH for index in selected)
                if role_values is not None
                else 0
            )
            remaining_local_count = (
                sum(role_values[int(index)] == ROLE_LOCAL_PATCH for index in remaining)
                if role_values is not None
                else 0
            )
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
                role_feasible = True
                if role_values is not None and local_floor_count > 0:
                    candidate_is_local = int(role_values[int(idx)] == ROLE_LOCAL_PATCH)
                    remaining_local_after = remaining_local_count - candidate_is_local
                    possible_local_count = (
                        selected_local_count
                        + candidate_is_local
                        + min(slots_after_candidate, remaining_local_after)
                    )
                    role_feasible = possible_local_count >= local_floor_count
                if possible_mass + 1e-8 >= saliency_mass_floor and role_feasible:
                    feasible.append(int(idx))
            feasible_candidate_count += len(feasible)
            if feasible:
                candidate_pool = feasible
            elif role_values is not None and local_floor_count > 0:
                role_feasible_remaining = []
                for idx in remaining:
                    candidate_is_local = int(role_values[int(idx)] == ROLE_LOCAL_PATCH)
                    possible_local_count = (
                        selected_local_count
                        + candidate_is_local
                        + min(slots_after_candidate, remaining_local_count - candidate_is_local)
                    )
                    if possible_local_count >= local_floor_count:
                        role_feasible_remaining.append(int(idx))
                candidate_pool = role_feasible_remaining if role_feasible_remaining else remaining
            else:
                candidate_pool = remaining

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
                visual_role_ids=role_values,
                local_floor_count=local_floor_count,
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
            "topk_local_patch_count": int(topk_local_count),
            "local_patch_floor_count": int(local_floor_count),
            "selected_local_patch_count": int(
                sum(role_values[int(index)] == ROLE_LOCAL_PATCH for index in selected)
                if role_values is not None
                else 0
            ),
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
        visual_role_ids: Optional[torch.Tensor] = None,
        local_floor_ratio: float = 0.0,
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
                "topk_local_patch_count": 0,
                "local_patch_floor_count": 0,
                "selected_local_patch_count": 0,
            }

        keep_count = min(int(target_keep), num_visual)
        entropy_norm = _clamp01(entropy_norm)
        positive_saliency = saliency_score.to(dtype=torch.float32).clamp_min(0.0)
        saliency_desc = self._descending_indices_tensor(saliency_score)
        topk_indices = saliency_desc[:keep_count]
        saliency_mass_topk = positive_saliency.index_select(0, topk_indices).sum()
        role_ids = (
            visual_role_ids.to(device=device, dtype=torch.long)
            if visual_role_ids is not None
            else None
        )
        topk_local_count, local_floor_count = self._local_floor_count(
            saliency_desc=saliency_desc,
            keep_count=keep_count,
            visual_role_ids=role_ids,
            local_floor_ratio=local_floor_ratio,
        )

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
        seed_indices, seed_gains = self._rebalance_seed_local_capacity_gpu(
            seed_indices=seed_indices,
            seed_gains=seed_gains,
            saliency_desc=saliency_desc,
            visual_role_ids=role_ids,
            keep_count=keep_count,
            local_floor_count=local_floor_count,
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
            saliency_feasible_mask = remaining_mask & (possible_mass + 1e-8 >= saliency_mass_floor)
            role_feasible_mask = remaining_mask
            if role_ids is not None and local_floor_count > 0:
                local_mask = role_ids == ROLE_LOCAL_PATCH
                selected_local_count = int((selected_mask & local_mask).sum().item())
                remaining_local_count = int((remaining_mask & local_mask).sum().item())
                candidate_is_local = local_mask.to(dtype=torch.long)
                remaining_local_after = remaining_local_count - candidate_is_local
                future_local_capacity = torch.clamp(
                    remaining_local_after,
                    min=0,
                    max=max(int(slots_after_candidate), 0),
                )
                possible_local_count = selected_local_count + candidate_is_local + future_local_capacity
                role_feasible_mask = remaining_mask & (possible_local_count >= int(local_floor_count))

            feasible_mask = saliency_feasible_mask & role_feasible_mask
            feasible_count = int(feasible_mask.sum().item())
            feasible_candidate_count += feasible_count
            if feasible_count > 0:
                candidate_mask = feasible_mask
            elif bool(role_feasible_mask.any().item()):
                candidate_mask = role_feasible_mask
            else:
                candidate_mask = remaining_mask

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
                visual_role_ids=role_ids,
                local_floor_count=local_floor_count,
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
            "topk_local_patch_count": int(topk_local_count),
            "local_patch_floor_count": int(local_floor_count),
            "selected_local_patch_count": int(
                ((role_ids == ROLE_LOCAL_PATCH) & selected_mask).sum().item()
                if role_ids is not None
                else 0
            ),
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

    def _run_evidence_reconciliation(
        self,
        *,
        attn_weights: torch.Tensor,
        v_token_start: int,
        v_token_num: int,
        text_token_start: int,
        layer_idx: int,
        layer_mode: str,
        params: Dict[str, object],
        context: Dict[str, object],
        current_patch_indices: torch.Tensor,
        current_visual_embeds: torch.Tensor,
        legacy_keep_indices: torch.Tensor,
        scalar_score: torch.Tensor,
        saliency_floor_eta: Optional[float],
        local_floor_count: int,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        reconcile_mode = str(params["evidence_reconciliation_mode"])
        profile = self._evidence_profile(params)
        context["evidence_reconcile_profile"] = dict(profile)
        if float(profile["pressure"]) <= 0.0 or layer_mode not in params[
            "evidence_reconciliation_apply_modes"
        ]:
            return legacy_keep_indices, self._evidence_bypass_stats(
                reconcile_mode,
                profile,
                tuple(params["evidence_reconciliation_stages"]),
            )
        if current_visual_embeds is None:
            raise ValueError("Active evidence reconciliation requires current_visual_embeds on every applied layer")
        visual_layout = context.get("visual_layout")
        if visual_layout is None:
            raise ValueError("Active evidence reconciliation requires visual_layout sample metadata")

        rater_scores = self.compute_rater_importance(
            attn_weights=attn_weights,
            v_token_start=v_token_start,
            v_token_num=v_token_num,
            text_token_start=text_token_start,
            layer_idx=layer_idx,
            device=current_visual_embeds.device,
        )
        current_distributions = normalize_rater_distributions(rater_scores)
        reference_distributions: Optional[torch.Tensor]
        if layer_mode == "C" and int(layer_idx) == int(profile["first_c_layer"]):
            if saliency_floor_eta is None:
                raise ValueError("C-layer evidence reconciliation requires the effective saliency floor eta")
            initial_count = int(context["initial_v_token_num"])
            reference_by_original = torch.zeros(
                (int(current_distributions.shape[0]), initial_count),
                device=current_distributions.device,
                dtype=torch.float32,
            )
            reference_by_original.index_copy_(
                1,
                current_patch_indices.to(device=current_distributions.device, dtype=torch.long),
                current_distributions,
            )
            context["evidence_c_reference"] = reference_by_original.detach()
            context["evidence_c_saliency_floor_eta"] = float(saliency_floor_eta)
            reference_distributions = None
        else:
            reference_by_original = context.get("evidence_c_reference")
            if reference_by_original is None:
                raise ValueError(
                    "Evidence reconciliation reached a B/S layer before the first-C reference was captured"
                )
            reference_distributions = restrict_reference_distributions(
                torch.as_tensor(reference_by_original, device=current_distributions.device),
                current_patch_indices,
            )
            saliency_floor_eta = float(context["evidence_c_saliency_floor_eta"])

        structure_ids = visual_token_structure_ids(
            current_patch_indices,
            visual_layout,
            device=current_distributions.device,
        )
        candidate_pool_multiplier = params["evidence_reconciliation_candidate_pool_multiplier"]
        if candidate_pool_multiplier is None:
            candidate_pool_multiplier = params["seed_pool_multiplier"]
        max_swap_ratio = params["evidence_reconciliation_max_swap_ratio"]
        if max_swap_ratio is None:
            max_swap_ratio = params["boundary_ratio"]

        _sync_if_cuda(current_visual_embeds)
        start = time.perf_counter()
        result = reconcile_evidence(
            mode=reconcile_mode,
            legacy_keep_indices=legacy_keep_indices,
            scalar_score=scalar_score,
            current_distributions=current_distributions,
            reference_distributions=reference_distributions,
            current_visual_embeds=current_visual_embeds,
            current_original_indices=current_patch_indices.to(
                device=current_distributions.device,
                dtype=torch.long,
            ),
            structure_ids=structure_ids,
            tau=float(saliency_floor_eta),
            profile_pressure=float(profile["pressure"]),
            candidate_pool_multiplier=float(candidate_pool_multiplier),
            max_swap_ratio=float(max_swap_ratio),
            local_floor_count=int(local_floor_count),
            stages=tuple(params["evidence_reconciliation_stages"]),
        )
        _sync_if_cuda(current_visual_embeds)
        stats = dict(result["stats"])
        stats.update(
            {
                "evidence_reconcile_bypassed": False,
                "evidence_reconcile_profile_keep_fraction": float(profile["keep_fraction"]),
                "evidence_reconcile_time_ms": (time.perf_counter() - start) * 1000.0,
            }
        )
        return torch.as_tensor(
            result["keep_indices"],
            device=legacy_keep_indices.device,
            dtype=torch.long,
        ), stats

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
        evidence_params = self._get_evidence_reconciliation_params()

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
            if str(evidence_params["mode"]) != "none":
                if current_visual_embeds is None:
                    raise ValueError("Active evidence reconciliation requires current_visual_embeds for S layers")
                params = self._get_scnd_params()
                keep_indices, evidence_stats = self._run_evidence_reconciliation(
                    attn_weights=attn_weights,
                    v_token_start=v_token_start,
                    v_token_num=v_token_num,
                    text_token_start=text_token_start,
                    layer_idx=layer_idx,
                    layer_mode=mode,
                    params=params,
                    context=context,
                    current_patch_indices=current_patch_indices,
                    current_visual_embeds=current_visual_embeds,
                    legacy_keep_indices=keep_indices,
                    scalar_score=visual_scores,
                    saliency_floor_eta=None,
                    local_floor_count=0,
                )
                pruned_mask = torch.ones(v_token_num, device=visual_scores.device, dtype=torch.bool)
                pruned_mask[keep_indices] = False
                pruned_indices = torch.nonzero(pruned_mask, as_tuple=False).flatten()
                patch_info = self._patch_index_info(current_patch_indices, keep_indices, pruned_indices)
                info.update(evidence_stats)
                info.update(
                    {
                        "num_visual_after": int(keep_indices.numel()),
                        "num_pruned": int(pruned_indices.numel()),
                        "keep_indices": keep_indices.detach().cpu().numpy(),
                        "pruned_indices": pruned_indices.detach().cpu().numpy(),
                        "keep_patch_indices": patch_info["keep_patch_indices"].detach().cpu().numpy(),
                        "pruned_patch_indices": patch_info["pruned_patch_indices"].detach().cpu().numpy(),
                    }
                )
                if bool(evidence_stats.get("evidence_reconcile_applied", False)):
                    info["layer_strategy_effective"] = "sparsevlm_scnd_herc"
                    info["selection_rule"] = f"sparsevlm_topk_{evidence_params['mode']}"
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
        if requested_backend == "python":
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
        entropy_control = self._entropy_control(
            entropy_norm,
            layer_idx,
            mode,
            params,
            keep_fraction=float(target_keep / v_token_num),
        )
        entropy_norm_control = float(entropy_control["control_value"])
        current_visual_role_ids: Optional[torch.Tensor] = None
        visual_role_local_saliency_mass_ratio: Optional[float] = None
        if mode == "C" and str(params["visual_role_constraint_mode"]) != "none":
            visual_layout = context.get("visual_layout")
            if visual_layout is None:
                raise ValueError(
                    "sparsevlm_scnd.visual_role_constraint requires visual_layout sample metadata"
                )
            current_visual_role_ids = visual_token_role_ids(
                current_patch_indices,
                visual_layout,
                device=mixed_score.device,
            )
            visual_role_local_saliency_mass_ratio = self._visual_role_local_saliency_mass_ratio(
                visual_scores,
                current_visual_role_ids,
            )
        visual_role_control = self._visual_role_constraint_control(
            mode=str(params["visual_role_constraint_mode"]),
            local_floor_ratio=float(params["visual_role_local_floor_ratio"]),
            keep_fraction=float(target_keep / v_token_num),
            keep_fraction_low=float(params["visual_role_budget_keep_fraction_low"]),
            keep_fraction_high=float(params["visual_role_budget_keep_fraction_high"]),
            num_visual_tokens_before=int(v_token_num) if mode == "C" else None,
            local_saliency_mass_ratio=visual_role_local_saliency_mass_ratio,
            max_visual_tokens_before=int(params["visual_role_max_visual_tokens_before"]),
            min_local_saliency_mass_ratio=float(params["visual_role_min_local_saliency_mass_ratio"]),
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
            c_selection_rule = str(params["c_selection_rule"])
            if c_selection_rule == "random_feasible" and selection_backend_effective == "gpu":
                selection = self._saliency_constrained_random_feasible_select_gpu(
                    saliency_score=mixed_score,
                    distance=distance,
                    target_keep=target_keep,
                    entropy_norm=entropy_norm_control,
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
                    entropy_norm=entropy_norm_control,
                    seed_ratio_min=float(params["seed_ratio_min"]),
                    seed_ratio_max=float(params["seed_ratio_max"]),
                    seed_pool_multiplier=float(params["seed_pool_multiplier"]),
                    saliency_floor_min=float(params["saliency_floor_min"]),
                    saliency_floor_max=float(params["saliency_floor_max"]),
                    saliency_repair=bool(params["saliency_repair"]),
                    random_seed=random_seed,
                )
            elif selection_backend_effective == "gpu":
                selection = self._saliency_constrained_native_divprune_select_gpu(
                    saliency_score=mixed_score,
                    distance=distance,
                    target_keep=target_keep,
                    entropy_norm=entropy_norm_control,
                    seed_ratio_min=float(params["seed_ratio_min"]),
                    seed_ratio_max=float(params["seed_ratio_max"]),
                    seed_pool_multiplier=float(params["seed_pool_multiplier"]),
                    saliency_floor_min=float(params["saliency_floor_min"]),
                    saliency_floor_max=float(params["saliency_floor_max"]),
                    saliency_repair=bool(params["saliency_repair"]),
                    visual_role_ids=current_visual_role_ids,
                    local_floor_ratio=float(visual_role_control["local_floor_ratio_effective"]),
                )
            else:
                selection = self._saliency_constrained_native_divprune_select(
                    saliency_score=mixed_score,
                    distance=distance,
                    target_keep=target_keep,
                    entropy_norm=entropy_norm_control,
                    seed_ratio_min=float(params["seed_ratio_min"]),
                    seed_ratio_max=float(params["seed_ratio_max"]),
                    seed_pool_multiplier=float(params["seed_pool_multiplier"]),
                    saliency_floor_min=float(params["saliency_floor_min"]),
                    saliency_floor_max=float(params["saliency_floor_max"]),
                    saliency_repair=bool(params["saliency_repair"]),
                    visual_role_ids=current_visual_role_ids,
                    local_floor_ratio=float(visual_role_control["local_floor_ratio_effective"]),
                )
            _sync_if_cuda(distance)
            selection_time_ms = (time.perf_counter() - selection_start) * 1000.0
            if c_selection_rule == "random_feasible":
                selection_rule = "saliency_constrained_random_feasible"
            elif bool(visual_role_control["applied"]):
                selection_rule = (
                    "saliency_constrained_native_divprune_anyres_saliency_gated_local_floor"
                    if visual_role_control["mode"] == "budget_adaptive_saliency_gated_local_floor"
                    else "saliency_constrained_native_divprune_anyres_local_floor"
                )
            else:
                selection_rule = "saliency_constrained_native_divprune"
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

        evidence_stats: Dict[str, object] = {}
        keep_indices = selection["keep_indices"]
        layer_strategy_effective = "sparsevlm_scnd"
        if str(params["evidence_reconciliation_mode"]) != "none":
            local_floor_count = int(selection.get("local_patch_floor_count", 0)) if mode == "C" else 0
            keep_indices, evidence_stats = self._run_evidence_reconciliation(
                attn_weights=attn_weights,
                v_token_start=v_token_start,
                v_token_num=v_token_num,
                text_token_start=text_token_start,
                layer_idx=layer_idx,
                layer_mode=mode,
                params=params,
                context=context,
                current_patch_indices=current_patch_indices,
                current_visual_embeds=current_visual_embeds,
                legacy_keep_indices=keep_indices,
                scalar_score=mixed_score,
                saliency_floor_eta=(
                    float(selection["saliency_floor_eta"])
                    if mode == "C" and "saliency_floor_eta" in selection
                    else None
                ),
                local_floor_count=local_floor_count,
            )
            selection["keep_indices"] = keep_indices
            selection_time_ms += float(evidence_stats.get("evidence_reconcile_time_ms", 0.0))
            if mode == "C" and "saliency_mass_selected" in selection:
                selection["saliency_mass_selected"] = float(
                    mixed_score.to(dtype=torch.float32)
                    .clamp_min(0.0)
                    .index_select(0, keep_indices)
                    .sum()
                    .item()
                )
            if mode == "C" and current_visual_role_ids is not None:
                selection["selected_local_patch_count"] = int(
                    (current_visual_role_ids.index_select(0, keep_indices) == ROLE_LOCAL_PATCH).sum().item()
                )
            if bool(evidence_stats.get("evidence_reconcile_applied", False)):
                layer_strategy_effective = "sparsevlm_scnd_herc"
                selection_rule = f"{selection_rule}_{params['evidence_reconciliation_mode']}"

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
            "layer_strategy_effective": layer_strategy_effective,
            "selection_rule": selection_rule,
            "distance_metric": params["distance_metric"],
            "saliency_entropy": entropy_raw,
            "saliency_entropy_norm": entropy_norm,
            "saliency_entropy_norm_control": entropy_norm_control,
            "entropy_calibration_mode": params["entropy_calibration_mode"],
            "entropy_calibration_applied": bool(mode == "C" and params["entropy_calibration_mode"] != "identity"),
            "entropy_calibration_q_low": entropy_control["q_low"],
            "entropy_calibration_q_high": entropy_control["q_high"],
            "entropy_calibration_clipped_low": entropy_control["clipped_low"],
            "entropy_calibration_clipped_high": entropy_control["clipped_high"],
            "entropy_calibration_artifact_path": entropy_control["artifact_path"],
            "entropy_calibration_quantile_value": entropy_control["quantile_value"],
            "entropy_calibration_budget_keep_fraction": entropy_control["budget_keep_fraction"],
            "entropy_calibration_budget_pressure": entropy_control["budget_pressure"],
            "entropy_calibration_budget_keep_fraction_low": entropy_control["budget_keep_fraction_low"],
            "entropy_calibration_budget_keep_fraction_high": entropy_control["budget_keep_fraction_high"],
            "entropy_calibration_diversity_tail_gain": entropy_control["diversity_tail_gain"],
            "entropy_calibration_diversity_tail_delta": entropy_control["diversity_tail_delta"],
            "visual_role_constraint_mode": visual_role_control["mode"],
            "visual_role_constraint_applied": bool(mode == "C" and visual_role_control["applied"]),
            "visual_role_budget_keep_fraction": visual_role_control["budget_keep_fraction"],
            "visual_role_budget_pressure": visual_role_control["budget_pressure"],
            "visual_role_local_floor_ratio": visual_role_control["local_floor_ratio"],
            "visual_role_local_floor_ratio_effective": visual_role_control["local_floor_ratio_effective"],
            "visual_role_budget_keep_fraction_low": visual_role_control["budget_keep_fraction_low"],
            "visual_role_budget_keep_fraction_high": visual_role_control["budget_keep_fraction_high"],
            "visual_role_num_visual_tokens_before": visual_role_control["num_visual_tokens_before"],
            "visual_role_local_saliency_mass_ratio": visual_role_control["local_saliency_mass_ratio"],
            "visual_role_max_visual_tokens_before": visual_role_control["max_visual_tokens_before"],
            "visual_role_min_local_saliency_mass_ratio": visual_role_control[
                "min_local_saliency_mass_ratio"
            ],
            "visual_role_layout_gate_passed": visual_role_control["layout_gate_passed"],
            "visual_role_saliency_gate_passed": visual_role_control["saliency_gate_passed"],
            "visual_role_gate_passed": visual_role_control["gate_passed"],
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

        info.update(evidence_stats)

        return keep_indices, info
