"""Budget-neutral evidence reconciliation for SCND selections."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


VALID_RECONCILIATION_MODES = {"none", "audit_only", "herc_v1", "herc_v2"}
VALID_PROFILE_LOCKS = {"first_c"}
VALID_CONTINUITY_REFERENCES = {"first_c"}
VALID_HIERARCHY_AGGREGATIONS = {"harmonic", "mass_weighted"}

_EPS = 1e-8


def profile_locked_pressure(
    keep_fraction: float,
    keep_fraction_low: float,
    keep_fraction_high: float,
) -> float:
    """Map the first-C nominal keep fraction to a fixed sample profile pressure."""
    return float(
        min(
            max(
                (float(keep_fraction_high) - float(keep_fraction))
                / (float(keep_fraction_high) - float(keep_fraction_low)),
                0.0,
            ),
            1.0,
        )
    )


def normalize_rater_distributions(rater_scores: torch.Tensor) -> torch.Tensor:
    """Normalize non-negative rater attention independently over visual tokens."""
    if rater_scores.ndim != 2 or int(rater_scores.shape[0]) == 0 or int(rater_scores.shape[1]) == 0:
        raise ValueError(f"rater_scores must be non-empty [R, N], got {tuple(rater_scores.shape)}")
    positive = rater_scores.to(dtype=torch.float32).clamp_min(0.0)
    positive = positive + torch.finfo(positive.dtype).eps
    return positive / positive.sum(dim=1, keepdim=True)


def restrict_reference_distributions(
    reference_by_original: torch.Tensor,
    current_original_indices: torch.Tensor,
) -> torch.Tensor:
    """Restrict first-C distributions to current survivors and renormalize rows."""
    if reference_by_original.ndim != 2:
        raise ValueError(
            f"reference_by_original must be [R, N_initial], got {tuple(reference_by_original.shape)}"
        )
    indices = current_original_indices.to(device=reference_by_original.device, dtype=torch.long).flatten()
    if int(indices.numel()) == 0:
        raise ValueError("current_original_indices must be non-empty")
    restricted = reference_by_original.index_select(1, indices).to(dtype=torch.float32)
    row_sum = restricted.sum(dim=1, keepdim=True)
    uniform = torch.full_like(restricted, 1.0 / float(indices.numel()))
    return torch.where(row_sum > _EPS, restricted / row_sum.clamp_min(_EPS), uniform)


def rater_js_disagreement(distributions: torch.Tensor) -> float:
    """Return generalized Jensen-Shannon disagreement, normalized to [0, 1]."""
    if distributions.ndim != 2:
        raise ValueError(f"distributions must be [R, N], got {tuple(distributions.shape)}")
    rater_count = int(distributions.shape[0])
    if rater_count <= 1:
        return 0.0
    probs = distributions.to(dtype=torch.float64).clamp_min(_EPS)
    probs = probs / probs.sum(dim=1, keepdim=True)
    mixture = probs.mean(dim=0, keepdim=True).clamp_min(_EPS)
    js = (probs * (probs.log() - mixture.log())).sum(dim=1).mean()
    normalized = js / math.log(float(rater_count))
    return float(normalized.clamp(0.0, 1.0).item())


def _stable_descending(scores: torch.Tensor) -> torch.Tensor:
    return torch.argsort(scores.to(dtype=torch.float32), descending=True, stable=True)


def _harmonic_coverage(values: torch.Tensor, dim: int = 0) -> torch.Tensor:
    count = int(values.shape[dim])
    if count == 0:
        shape = list(values.shape)
        del shape[dim]
        return torch.ones(shape, device=values.device, dtype=torch.float32)
    values_f = values.to(dtype=torch.float32).clamp(min=_EPS, max=1.0)
    return (float(count) / (1.0 / values_f).sum(dim=dim)).clamp(min=0.0, max=1.0)


def _weighted_harmonic_coverage(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    values_f = values.to(dtype=torch.float32).clamp(min=_EPS, max=1.0)
    weights_f = weights.to(device=values.device, dtype=torch.float32)
    weights_f = weights_f / weights_f.sum().clamp_min(_EPS)
    return (1.0 / (weights_f / values_f).sum()).clamp(min=0.0, max=1.0)


def _weighted_mass_coverage(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    values_f = values.to(dtype=torch.float32).clamp(min=0.0, max=1.0)
    weights_f = weights.to(device=values.device, dtype=torch.float32)
    weights_f = weights_f / weights_f.sum().clamp_min(_EPS)
    return (weights_f * values_f).sum().clamp(min=0.0, max=1.0)


def _hierarchy_aggregation_for_mode(mode: str) -> str:
    return "mass_weighted" if mode == "herc_v2" else "harmonic"


def _query_components(
    current: torch.Tensor,
    reference: Optional[torch.Tensor],
    selected: torch.Tensor,
    tau: float,
) -> Dict[str, torch.Tensor]:
    keep_count = int(selected.numel())
    current_target = float(tau) * torch.topk(
        current,
        k=min(keep_count, int(current.shape[1])),
        dim=1,
        largest=True,
        sorted=False,
    ).values.sum(dim=1)
    current_mass = current.index_select(1, selected).sum(dim=1)
    current_coverage = torch.clamp(current_mass / current_target.clamp_min(_EPS), max=1.0)

    if reference is None:
        reference_target = current_target
        reference_coverage = current_coverage
        effective = current_coverage
    else:
        reference_target = float(tau) * torch.topk(
            reference,
            k=min(keep_count, int(reference.shape[1])),
            dim=1,
            largest=True,
            sorted=False,
        ).values.sum(dim=1)
        reference_mass = reference.index_select(1, selected).sum(dim=1)
        reference_coverage = torch.clamp(reference_mass / reference_target.clamp_min(_EPS), max=1.0)
        effective = torch.sqrt((current_coverage * reference_coverage).clamp_min(0.0))

    return {
        "current_target": current_target,
        "reference_target": reference_target,
        "current_coverage": current_coverage,
        "reference_coverage": reference_coverage,
        "effective_coverage": effective,
        "aggregate": _harmonic_coverage(effective, dim=0),
        "current_aggregate": _harmonic_coverage(current_coverage, dim=0),
        "reference_aggregate": _harmonic_coverage(reference_coverage, dim=0),
    }


def _query_pair_components(
    current: torch.Tensor,
    reference: Optional[torch.Tensor],
    selected: torch.Tensor,
    incoming: torch.Tensor,
    outgoing: torch.Tensor,
    tau: float,
) -> Dict[str, torch.Tensor]:
    base = _query_components(current, reference, selected, tau)
    current_mass = current.index_select(1, selected).sum(dim=1)
    current_pair_mass = (
        current_mass[:, None, None]
        + current.index_select(1, incoming)[:, :, None]
        - current.index_select(1, outgoing)[:, None, :]
    )
    current_coverage = torch.clamp(
        current_pair_mass / base["current_target"][:, None, None].clamp_min(_EPS),
        max=1.0,
    )

    if reference is None:
        reference_coverage = current_coverage
        effective = current_coverage
    else:
        reference_mass = reference.index_select(1, selected).sum(dim=1)
        reference_pair_mass = (
            reference_mass[:, None, None]
            + reference.index_select(1, incoming)[:, :, None]
            - reference.index_select(1, outgoing)[:, None, :]
        )
        reference_coverage = torch.clamp(
            reference_pair_mass / base["reference_target"][:, None, None].clamp_min(_EPS),
            max=1.0,
        )
        effective = torch.sqrt((current_coverage * reference_coverage).clamp_min(0.0))

    return {
        "current_coverage": current_coverage,
        "reference_coverage": reference_coverage,
        "aggregate": _harmonic_coverage(effective, dim=0),
    }


def _build_hierarchy_level_template(
    token_group_ids: torch.Tensor,
    patch_mask: torch.Tensor,
    saliency: torch.Tensor,
    keep_count: int,
    tau: float,
) -> Dict[str, torch.Tensor]:
    valid = patch_mask & (token_group_ids >= 0)
    valid_positions = torch.nonzero(valid, as_tuple=False).flatten()
    if int(valid_positions.numel()) == 0:
        return {"empty": torch.tensor(True, device=saliency.device)}

    unique_ids, inverse = torch.unique(
        token_group_ids.index_select(0, valid_positions),
        sorted=True,
        return_inverse=True,
    )
    group_count = int(unique_ids.numel())
    token_group = torch.full_like(token_group_ids, -1)
    token_group[valid_positions] = inverse
    reachable = torch.zeros(group_count, device=saliency.device, dtype=torch.float32)
    for group_idx in range(group_count):
        members = valid_positions[inverse == group_idx]
        group_scores = saliency.index_select(0, members)
        reachable[group_idx] = torch.topk(
            group_scores,
            k=min(int(keep_count), int(members.numel())),
            largest=True,
            sorted=False,
        ).values.sum()
    weights = reachable / reachable.sum().clamp_min(_EPS)
    target = float(tau) * reachable

    return {
        "empty": torch.tensor(False, device=saliency.device),
        "unique_ids": unique_ids,
        "token_group": token_group,
        "target": target,
        "weights": weights,
    }


def _hierarchy_level_state(
    template: Dict[str, torch.Tensor],
    saliency: torch.Tensor,
    selected: torch.Tensor,
    aggregation: str,
) -> Dict[str, torch.Tensor | float]:
    if bool(template["empty"].item()):
        return {"empty": template["empty"], "coverage": 1.0}
    token_group = template["token_group"]
    target = template["target"]
    weights = template["weights"]
    group_count = int(target.numel())
    selected_groups = token_group.index_select(0, selected)
    selected_values = saliency.index_select(0, selected)
    selected_valid = selected_groups >= 0
    mass = torch.zeros(group_count, device=saliency.device, dtype=torch.float32)
    if bool(selected_valid.any().item()):
        mass.scatter_add_(0, selected_groups[selected_valid], selected_values[selected_valid])
    coverage_by_group = torch.clamp(mass / target.clamp_min(_EPS), max=1.0)
    if aggregation == "harmonic":
        coverage = _weighted_harmonic_coverage(coverage_by_group, weights)
    elif aggregation == "mass_weighted":
        coverage = _weighted_mass_coverage(coverage_by_group, weights)
    else:
        raise ValueError(f"Unsupported hierarchy aggregation: {aggregation!r}")
    return {
        **template,
        "token_group": token_group,
        "mass": mass,
        "coverage": coverage,
        "aggregation": aggregation,
    }


def _hierarchy_templates(
    structure_ids: Dict[str, torch.Tensor],
    saliency: torch.Tensor,
    keep_count: int,
    tau: float,
) -> Dict[str, Dict[str, torch.Tensor]]:
    patch_mask = structure_ids["patch_mask"]
    return {
        "view": _build_hierarchy_level_template(
            structure_ids["view_ids"], patch_mask, saliency, keep_count, tau
        ),
        "macrocell": _build_hierarchy_level_template(
            structure_ids["macrocell_ids"], patch_mask, saliency, keep_count, tau
        ),
        "row": _build_hierarchy_level_template(
            structure_ids["row_ids"], patch_mask, saliency, keep_count, tau
        ),
    }


def _hierarchy_states(
    structure_ids: Dict[str, torch.Tensor],
    saliency: torch.Tensor,
    selected: torch.Tensor,
    keep_count: int,
    tau: float,
    aggregation: str,
    templates: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
) -> Dict[str, Dict[str, torch.Tensor | float]]:
    if templates is None:
        templates = _hierarchy_templates(structure_ids, saliency, keep_count, tau)
    return {
        key: _hierarchy_level_state(template, saliency, selected, aggregation)
        for key, template in templates.items()
    }


def _hierarchy_aggregate(states: Dict[str, Dict[str, torch.Tensor | float]]) -> torch.Tensor:
    coverages = [
        torch.as_tensor(state["coverage"], device=next(iter(states.values()))["empty"].device, dtype=torch.float32)
        for state in states.values()
    ]
    stacked = torch.stack(coverages).clamp_min(_EPS)
    return torch.exp(torch.log(stacked).mean()).clamp(max=1.0)


def _level_pair_coverage(
    state: Dict[str, torch.Tensor | float],
    saliency: torch.Tensor,
    incoming: torch.Tensor,
    outgoing: torch.Tensor,
) -> torch.Tensor:
    if bool(torch.as_tensor(state["empty"]).item()):
        return torch.ones(
            (int(incoming.numel()), int(outgoing.numel())),
            device=saliency.device,
            dtype=torch.float32,
        )
    token_group = torch.as_tensor(state["token_group"], device=saliency.device, dtype=torch.long)
    mass = torch.as_tensor(state["mass"], device=saliency.device, dtype=torch.float32)
    target = torch.as_tensor(state["target"], device=saliency.device, dtype=torch.float32)
    weights = torch.as_tensor(state["weights"], device=saliency.device, dtype=torch.float32)
    aggregation = str(state["aggregation"])

    def harmonic_term(group: torch.Tensor, new_mass: torch.Tensor) -> torch.Tensor:
        valid = group >= 0
        safe_group = group.clamp_min(0)
        coverage = torch.clamp(
            new_mass / target.index_select(0, safe_group).clamp_min(_EPS),
            min=0.0,
            max=1.0,
        )
        values = weights.index_select(0, safe_group) / coverage.clamp_min(_EPS)
        return torch.where(valid, values, torch.zeros_like(values))

    def mass_term(group: torch.Tensor, new_mass: torch.Tensor) -> torch.Tensor:
        valid = group >= 0
        safe_group = group.clamp_min(0)
        coverage = torch.clamp(
            new_mass / target.index_select(0, safe_group).clamp_min(_EPS),
            min=0.0,
            max=1.0,
        )
        values = weights.index_select(0, safe_group) * coverage
        return torch.where(valid, values, torch.zeros_like(values))

    base_coverage = torch.clamp(mass / target.clamp_min(_EPS), min=0.0, max=1.0)
    if aggregation == "harmonic":
        term = harmonic_term
        base_terms = weights / base_coverage.clamp_min(_EPS)
    elif aggregation == "mass_weighted":
        term = mass_term
        base_terms = weights * base_coverage
    else:
        raise ValueError(f"Unsupported hierarchy aggregation: {aggregation!r}")
    base_total = base_terms.sum()
    incoming_group = token_group.index_select(0, incoming)
    outgoing_group = token_group.index_select(0, outgoing)
    incoming_mass = saliency.index_select(0, incoming)
    outgoing_mass = saliency.index_select(0, outgoing)

    incoming_safe = incoming_group.clamp_min(0)
    outgoing_safe = outgoing_group.clamp_min(0)
    add_delta = term(incoming_group, mass.index_select(0, incoming_safe) + incoming_mass) - torch.where(
        incoming_group >= 0,
        base_terms.index_select(0, incoming_safe),
        torch.zeros_like(incoming_mass),
    )
    remove_delta = term(outgoing_group, mass.index_select(0, outgoing_safe) - outgoing_mass) - torch.where(
        outgoing_group >= 0,
        base_terms.index_select(0, outgoing_safe),
        torch.zeros_like(outgoing_mass),
    )
    pair_delta = add_delta[:, None] + remove_delta[None, :]

    same_group = (incoming_group[:, None] == outgoing_group[None, :]) & (incoming_group[:, None] >= 0)
    if bool(same_group.any().item()):
        safe_pair_group = incoming_group[:, None].expand_as(same_group).clamp_min(0)
        exact_mass = (
            mass.index_select(0, safe_pair_group.flatten()).view_as(same_group)
            + incoming_mass[:, None]
            - outgoing_mass[None, :]
        )
        pair_target = target.index_select(0, safe_pair_group.flatten()).view_as(same_group)
        pair_weight = weights.index_select(0, safe_pair_group.flatten()).view_as(same_group)
        exact_coverage = torch.clamp(
            exact_mass / pair_target.clamp_min(_EPS),
            min=0.0,
            max=1.0,
        )
        if aggregation == "harmonic":
            exact_term = pair_weight / exact_coverage.clamp_min(_EPS)
        else:
            exact_term = pair_weight * exact_coverage
        base_pair_term = base_terms.index_select(0, safe_pair_group.flatten()).view_as(same_group)
        pair_delta = torch.where(same_group, exact_term - base_pair_term, pair_delta)

    pair_total = base_total + pair_delta
    if aggregation == "harmonic":
        return (1.0 / pair_total.clamp_min(_EPS)).clamp(min=0.0, max=1.0)
    return pair_total.clamp(min=0.0, max=1.0)


def _facility_template(
    embeds: torch.Tensor,
    candidate_pool: torch.Tensor,
    saliency: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    normalized = F.normalize(
        embeds.index_select(0, candidate_pool).to(dtype=torch.float32),
        dim=-1,
        eps=1e-6,
    )
    similarity = torch.clamp((1.0 + normalized @ normalized.transpose(0, 1)) / 2.0, 0.0, 1.0)
    weights = saliency.index_select(0, candidate_pool).to(dtype=torch.float32)
    weights = weights / weights.sum().clamp_min(_EPS)
    lookup = torch.full((int(embeds.shape[0]),), -1, device=embeds.device, dtype=torch.long)
    lookup[candidate_pool] = torch.arange(int(candidate_pool.numel()), device=embeds.device, dtype=torch.long)
    return {
        "similarity": similarity,
        "weights": weights,
        "lookup": lookup,
    }


def _facility_state_from_template(
    template: Dict[str, torch.Tensor],
    selected: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    similarity = template["similarity"]
    weights = template["weights"]
    lookup = template["lookup"]
    selected_positions = lookup.index_select(0, selected)
    represented = similarity.index_select(1, selected_positions).max(dim=1).values
    return {
        **template,
        "coverage": (weights * represented).sum().clamp(min=0.0, max=1.0),
    }


def _facility_state(
    embeds: torch.Tensor,
    candidate_pool: torch.Tensor,
    saliency: torch.Tensor,
    selected: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    return _facility_state_from_template(
        _facility_template(embeds, candidate_pool, saliency),
        selected,
    )


def _facility_pair_coverage(
    facility: Dict[str, torch.Tensor],
    selected: torch.Tensor,
    incoming: torch.Tensor,
    outgoing: torch.Tensor,
) -> torch.Tensor:
    similarity = facility["similarity"]
    weights = facility["weights"]
    lookup = facility["lookup"]
    selected_positions = lookup.index_select(0, selected)
    incoming_positions = lookup.index_select(0, incoming)
    outgoing_positions = lookup.index_select(0, outgoing)
    if bool((torch.cat((selected_positions, incoming_positions, outgoing_positions)) < 0).any().item()):
        raise AssertionError("Facility coverage received a token outside the candidate pool")

    selected_similarity = similarity.index_select(1, selected_positions)
    top1, top1_arg = selected_similarity.max(dim=1)
    if int(selected_positions.numel()) > 1:
        masked = selected_similarity.clone()
        masked.scatter_(1, top1_arg[:, None], -1.0)
        top2 = masked.max(dim=1).values.clamp_min(0.0)
    else:
        top2 = torch.zeros_like(top1)

    incoming_similarity = similarity.index_select(1, incoming_positions)
    base_add = (
        weights[:, None] * torch.clamp(incoming_similarity - top1[:, None], min=0.0)
    ).sum(dim=0)
    exact_if_top1_removed = torch.maximum(incoming_similarity, top2[:, None]) - top1[:, None]
    base_if_top1_kept = torch.clamp(incoming_similarity - top1[:, None], min=0.0)
    correction_by_point = weights[:, None] * (exact_if_top1_removed - base_if_top1_kept)
    corrections_by_selected = torch.zeros(
        (int(incoming.numel()), int(selected.numel())),
        device=similarity.device,
        dtype=torch.float32,
    )
    corrections_by_selected.scatter_add_(
        1,
        top1_arg[None, :].expand(int(incoming.numel()), -1),
        correction_by_point.transpose(0, 1),
    )
    selected_column_lookup = torch.full(
        (int(similarity.shape[0]),),
        -1,
        device=similarity.device,
        dtype=torch.long,
    )
    selected_column_lookup[selected_positions] = torch.arange(
        int(selected_positions.numel()),
        device=similarity.device,
        dtype=torch.long,
    )
    outgoing_columns = selected_column_lookup.index_select(0, outgoing_positions)
    if bool((outgoing_columns < 0).any().item()):
        raise AssertionError("Facility outgoing tokens must belong to the selected set")
    corrections = corrections_by_selected.index_select(1, outgoing_columns)

    current = facility["coverage"]
    return (current + base_add[:, None] + corrections).clamp(min=0.0, max=1.0)


def build_reconcile_candidate_pool(
    legacy_keep: torch.Tensor,
    scalar_score: torch.Tensor,
    current_distributions: torch.Tensor,
    reference_distributions: Optional[torch.Tensor],
    structure_ids: Dict[str, torch.Tensor],
    keep_count: int,
    pool_multiplier: float,
) -> torch.Tensor:
    """Build the union of scalar, per-rater, continuity, and structure candidates."""
    num_tokens = int(scalar_score.numel())
    selected = torch.zeros(num_tokens, device=scalar_score.device, dtype=torch.bool)
    selected[legacy_keep] = True
    scalar_count = min(num_tokens, max(keep_count, int(math.ceil(float(pool_multiplier) * keep_count))))
    selected[_stable_descending(scalar_score)[:scalar_count]] = True

    rater_count = int(current_distributions.shape[0])
    per_rater_count = min(
        num_tokens,
        max(1, int(math.ceil(float(pool_multiplier) * float(keep_count) / float(rater_count)))),
    )
    for distributions in (current_distributions, reference_distributions):
        if distributions is None:
            continue
        for row in distributions:
            selected[_stable_descending(row)[:per_rater_count]] = True

    scalar_saliency = current_distributions.mean(dim=0)
    patch_mask = structure_ids["patch_mask"]
    for key in ("view_ids", "macrocell_ids", "row_ids"):
        group_ids = structure_ids[key]
        valid_groups = torch.unique(group_ids[patch_mask & (group_ids >= 0)], sorted=True)
        for group_id in valid_groups.detach().cpu().tolist():
            members = torch.nonzero(patch_mask & (group_ids == int(group_id)), as_tuple=False).flatten()
            if int(members.numel()) == 0:
                continue
            member_scores = scalar_saliency.index_select(0, members)
            selected[members[_stable_descending(member_scores)[0]]] = True
    return torch.nonzero(selected, as_tuple=False).flatten()


def _pair_feasibility(
    scalar_score: torch.Tensor,
    scalar_mass: torch.Tensor,
    scalar_floor: float,
    structure_ids: Dict[str, torch.Tensor],
    selected: torch.Tensor,
    incoming: torch.Tensor,
    outgoing: torch.Tensor,
    local_floor_count: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    incoming_score = scalar_score.index_select(0, incoming)
    outgoing_score = scalar_score.index_select(0, outgoing)
    resulting_mass = scalar_mass + incoming_score[:, None] - outgoing_score[None, :]
    feasible = resulting_mass + 1e-7 >= float(scalar_floor)
    if local_floor_count > 0:
        role_ids = structure_ids["role_ids"]
        local = role_ids == 1
        selected_local = local.index_select(0, selected).sum()
        resulting_local = (
            selected_local
            + local.index_select(0, incoming).to(dtype=torch.long)[:, None]
            - local.index_select(0, outgoing).to(dtype=torch.long)[None, :]
        )
        feasible = feasible & (resulting_local >= int(local_floor_count))
    return feasible, resulting_mass


def _select_best_pair(
    gain: torch.Tensor,
    feasible: torch.Tensor,
    resulting_scalar_mass: torch.Tensor,
    incoming: torch.Tensor,
    outgoing: torch.Tensor,
    original_indices: torch.Tensor,
) -> Optional[Tuple[int, int]]:
    feasible = feasible & torch.isfinite(gain) & (gain > 1e-7)
    if not bool(feasible.any().item()):
        return None
    neg_inf = torch.tensor(float("-inf"), device=gain.device, dtype=gain.dtype)
    masked_gain = torch.where(feasible, gain, neg_inf)
    best_gain = masked_gain.max()
    candidates = feasible & (gain == best_gain)
    masked_mass = torch.where(candidates, resulting_scalar_mass, neg_inf)
    best_mass = masked_mass.max()
    candidates = candidates & (resulting_scalar_mass == best_mass)

    incoming_original = original_indices.index_select(0, incoming).to(dtype=torch.long)
    outgoing_original = original_indices.index_select(0, outgoing).to(dtype=torch.long)
    large = torch.iinfo(torch.long).max
    incoming_priority = torch.where(
        candidates,
        incoming_original[:, None].expand_as(candidates),
        torch.full_like(candidates, large, dtype=torch.long),
    )
    best_incoming = incoming_priority.min()
    candidates = candidates & (incoming_original[:, None] == best_incoming)
    outgoing_priority = torch.where(
        candidates,
        outgoing_original[None, :].expand_as(candidates),
        torch.full_like(candidates, large, dtype=torch.long),
    )
    best_outgoing = outgoing_priority.min()
    chosen = torch.nonzero(
        candidates & (outgoing_original[None, :] == best_outgoing),
        as_tuple=False,
    )[0]
    return int(chosen[0].item()), int(chosen[1].item())


def reconcile_evidence(
    *,
    mode: str,
    legacy_keep_indices: torch.Tensor,
    scalar_score: torch.Tensor,
    current_distributions: torch.Tensor,
    reference_distributions: Optional[torch.Tensor],
    current_visual_embeds: torch.Tensor,
    current_original_indices: torch.Tensor,
    structure_ids: Dict[str, torch.Tensor],
    tau: float,
    profile_pressure: float,
    candidate_pool_multiplier: float,
    max_swap_ratio: float,
    local_floor_count: int = 0,
) -> Dict[str, object]:
    """Audit or reconcile a legacy SCND proposal through budget-neutral swaps."""
    if mode not in VALID_RECONCILIATION_MODES - {"none"}:
        raise ValueError(f"Unsupported active evidence reconciliation mode: {mode!r}")
    hierarchy_aggregation = _hierarchy_aggregation_for_mode(mode)
    selected = torch.sort(legacy_keep_indices.to(dtype=torch.long)).values
    keep_count = int(selected.numel())
    if keep_count <= 0:
        raise ValueError("HERC requires a non-empty legacy selection")
    scalar_positive = scalar_score.to(dtype=torch.float32).clamp_min(0.0)
    scalar_mass_before = scalar_positive.index_select(0, selected).sum()
    scalar_topk_mass = torch.topk(
        scalar_positive,
        k=min(keep_count, int(scalar_positive.numel())),
        largest=True,
        sorted=False,
    ).values.sum()
    scalar_floor = torch.minimum(
        scalar_mass_before,
        scalar_topk_mass * float(tau),
    )
    scalar_saliency = current_distributions.mean(dim=0)
    candidate_pool = build_reconcile_candidate_pool(
        legacy_keep=selected,
        scalar_score=scalar_positive,
        current_distributions=current_distributions,
        reference_distributions=reference_distributions,
        structure_ids=structure_ids,
        keep_count=keep_count,
        pool_multiplier=candidate_pool_multiplier,
    )

    query_before = _query_components(current_distributions, reference_distributions, selected, tau)
    hierarchy_templates = _hierarchy_templates(
        structure_ids,
        scalar_saliency,
        keep_count,
        tau,
    )
    hierarchy_before = _hierarchy_states(
        structure_ids,
        scalar_saliency,
        selected,
        keep_count,
        tau,
        hierarchy_aggregation,
        templates=hierarchy_templates,
    )
    hierarchy_before_aggregate = _hierarchy_aggregate(hierarchy_before)
    facility_template = _facility_template(
        current_visual_embeds,
        candidate_pool,
        scalar_saliency,
    )
    facility_before = _facility_state_from_template(
        facility_template,
        selected,
    )
    deficit = 1.0 - float(
        (
            query_before["aggregate"]
            * hierarchy_before_aggregate
            * facility_before["coverage"]
        ).clamp(min=0.0, max=1.0).item()
    )
    swap_budget = min(
        keep_count,
        max(0, int(math.ceil(float(profile_pressure) * float(max_swap_ratio) * keep_count * deficit))),
    )

    query_swap_in: List[int] = []
    query_swap_out: List[int] = []
    context_swap_in: List[int] = []
    context_swap_out: List[int] = []
    query_after_query = query_before

    if mode in {"herc_v1", "herc_v2"} and swap_budget > 0:
        for _ in range(swap_budget):
            selected_mask = torch.zeros(
                int(scalar_score.numel()), device=scalar_score.device, dtype=torch.bool
            )
            selected_mask[selected] = True
            incoming = candidate_pool[~selected_mask.index_select(0, candidate_pool)]
            if int(incoming.numel()) == 0:
                break
            outgoing = selected
            pairs = _query_pair_components(
                current_distributions,
                reference_distributions,
                selected,
                incoming,
                outgoing,
                tau,
            )
            scalar_mass = scalar_positive.index_select(0, selected).sum()
            feasible, resulting_mass = _pair_feasibility(
                scalar_positive,
                scalar_mass,
                float(scalar_floor.item()),
                structure_ids,
                selected,
                incoming,
                outgoing,
                local_floor_count,
            )
            gain = pairs["aggregate"] - query_after_query["aggregate"]
            chosen = _select_best_pair(
                gain,
                feasible,
                resulting_mass,
                incoming,
                outgoing,
                current_original_indices,
            )
            if chosen is None:
                break
            incoming_index = int(incoming[chosen[0]].item())
            outgoing_index = int(outgoing[chosen[1]].item())
            selected = torch.sort(
                torch.cat(
                    (
                        selected[selected != outgoing_index],
                        torch.tensor([incoming_index], device=selected.device, dtype=torch.long),
                    )
                )
            ).values
            query_swap_in.append(incoming_index)
            query_swap_out.append(outgoing_index)
            query_after_query = _query_components(
                current_distributions,
                reference_distributions,
                selected,
                tau,
            )
            if bool((query_after_query["current_coverage"] >= 1.0 - 1e-7).all().item()) and bool(
                (query_after_query["reference_coverage"] >= 1.0 - 1e-7).all().item()
            ):
                break

        query_floor_current = query_after_query["current_coverage"].clone()
        query_floor_reference = query_after_query["reference_coverage"].clone()
        remaining_budget = swap_budget - len(query_swap_in)

        for _ in range(remaining_budget):
            selected_mask = torch.zeros(
                int(scalar_score.numel()), device=scalar_score.device, dtype=torch.bool
            )
            selected_mask[selected] = True
            incoming = candidate_pool[~selected_mask.index_select(0, candidate_pool)]
            if int(incoming.numel()) == 0:
                break
            outgoing = selected
            query_pairs = _query_pair_components(
                current_distributions,
                reference_distributions,
                selected,
                incoming,
                outgoing,
                tau,
            )
            scalar_mass = scalar_positive.index_select(0, selected).sum()
            feasible, resulting_mass = _pair_feasibility(
                scalar_positive,
                scalar_mass,
                float(scalar_floor.item()),
                structure_ids,
                selected,
                incoming,
                outgoing,
                local_floor_count,
            )
            feasible = feasible & (
                query_pairs["current_coverage"]
                >= query_floor_current[:, None, None] - 1e-7
            ).all(dim=0)
            feasible = feasible & (
                query_pairs["reference_coverage"]
                >= query_floor_reference[:, None, None] - 1e-7
            ).all(dim=0)

            hierarchy_states = _hierarchy_states(
                structure_ids,
                scalar_saliency,
                selected,
                keep_count,
                tau,
                hierarchy_aggregation,
                templates=hierarchy_templates,
            )
            hierarchy_pairs = torch.ones(
                (int(incoming.numel()), int(outgoing.numel())),
                device=scalar_score.device,
                dtype=torch.float32,
            )
            for state in hierarchy_states.values():
                hierarchy_pairs = hierarchy_pairs * _level_pair_coverage(
                    state,
                    scalar_saliency,
                    incoming,
                    outgoing,
                ).clamp_min(_EPS)
            hierarchy_pairs = hierarchy_pairs.pow(1.0 / float(len(hierarchy_states)))
            facility_state = _facility_state_from_template(
                facility_template,
                selected,
            )
            facility_pairs = _facility_pair_coverage(
                facility_state,
                selected,
                incoming,
                outgoing,
            )
            current_hierarchy = _hierarchy_aggregate(hierarchy_states)
            current_objective = torch.log(current_hierarchy.clamp_min(_EPS)) + torch.log(
                facility_state["coverage"].clamp_min(_EPS)
            )
            pair_objective = torch.log(hierarchy_pairs.clamp_min(_EPS)) + torch.log(
                facility_pairs.clamp_min(_EPS)
            )
            chosen = _select_best_pair(
                pair_objective - current_objective,
                feasible,
                resulting_mass,
                incoming,
                outgoing,
                current_original_indices,
            )
            if chosen is None:
                break
            incoming_index = int(incoming[chosen[0]].item())
            outgoing_index = int(outgoing[chosen[1]].item())
            selected = torch.sort(
                torch.cat(
                    (
                        selected[selected != outgoing_index],
                        torch.tensor([incoming_index], device=selected.device, dtype=torch.long),
                    )
                )
            ).values
            context_swap_in.append(incoming_index)
            context_swap_out.append(outgoing_index)

    query_after = _query_components(current_distributions, reference_distributions, selected, tau)
    hierarchy_after = _hierarchy_states(
        structure_ids,
        scalar_saliency,
        selected,
        keep_count,
        tau,
        hierarchy_aggregation,
        templates=hierarchy_templates,
    )
    facility_after = _facility_state_from_template(
        facility_template,
        selected,
    )
    all_in = query_swap_in + context_swap_in
    all_out = query_swap_out + context_swap_out
    in_tensor = torch.tensor(all_in, device=selected.device, dtype=torch.long)
    out_tensor = torch.tensor(all_out, device=selected.device, dtype=torch.long)

    stats: Dict[str, object] = {
        "evidence_reconcile_mode": mode,
        "evidence_reconcile_applied": bool(
            mode in {"herc_v1", "herc_v2"} and profile_pressure > 0.0
        ),
        "evidence_hierarchy_aggregation": hierarchy_aggregation,
        "evidence_reconcile_profile_pressure": float(profile_pressure),
        "evidence_reconcile_candidate_count": int(candidate_pool.numel()),
        "evidence_reconcile_swap_budget": int(swap_budget),
        "evidence_reconcile_query_swap_count": len(query_swap_in),
        "evidence_reconcile_context_swap_count": len(context_swap_in),
        "evidence_reconcile_deficit": float(deficit),
        "evidence_query_coverage_before": float(query_before["aggregate"].item()),
        "evidence_query_coverage_after_query": float(query_after_query["aggregate"].item()),
        "evidence_query_coverage_after_context": float(query_after["aggregate"].item()),
        "evidence_query_current_coverage_after": float(query_after["current_aggregate"].item()),
        "evidence_query_c_reference_coverage_after": float(query_after["reference_aggregate"].item()),
        "evidence_view_coverage_before": float(hierarchy_before["view"]["coverage"]),
        "evidence_view_coverage_after": float(hierarchy_after["view"]["coverage"]),
        "evidence_macrocell_coverage_before": float(hierarchy_before["macrocell"]["coverage"]),
        "evidence_macrocell_coverage_after": float(hierarchy_after["macrocell"]["coverage"]),
        "evidence_row_coverage_before": float(hierarchy_before["row"]["coverage"]),
        "evidence_row_coverage_after": float(hierarchy_after["row"]["coverage"]),
        "evidence_representative_coverage_before": float(facility_before["coverage"].item()),
        "evidence_representative_coverage_after": float(facility_after["coverage"].item()),
        "evidence_scalar_mass_floor": float(scalar_floor.item()),
        "evidence_scalar_mass_before": float(scalar_mass_before.item()),
        "evidence_scalar_mass_after": float(scalar_positive.index_select(0, selected).sum().item()),
        "evidence_rater_count": int(current_distributions.shape[0]),
        "evidence_rater_js_disagreement": rater_js_disagreement(current_distributions),
        "evidence_reconcile_in_indices": in_tensor,
        "evidence_reconcile_out_indices": out_tensor,
        "evidence_reconcile_in_patch_indices": current_original_indices.index_select(0, in_tensor),
        "evidence_reconcile_out_patch_indices": current_original_indices.index_select(0, out_tensor),
        "evidence_reconcile_candidate_indices": candidate_pool,
    }
    return {"keep_indices": selected, "stats": stats}
