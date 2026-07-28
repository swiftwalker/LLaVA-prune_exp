"""Exact next-layer counterfactual preview helpers for SCND routing."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _sync_if_cuda(tensor: torch.Tensor) -> None:
    if tensor.is_cuda:
        torch.cuda.synchronize(tensor.device)


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch_size, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch_size,
        num_key_value_heads,
        n_rep,
        seq_len,
        head_dim,
    )
    return hidden_states.reshape(batch_size, num_key_value_heads * n_rep, seq_len, head_dim)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    first = x[..., : x.shape[-1] // 2]
    second = x[..., x.shape[-1] // 2 :]
    return torch.cat((-second, first), dim=-1)


def _gather_rotary_cache(cache: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
    if cache.dim() == 2:
        gathered = cache.index_select(0, position_ids.reshape(-1))
        return gathered.view(*position_ids.shape, cache.shape[-1]).unsqueeze(1)
    if cache.dim() == 3 and cache.shape[0] == 1:
        gathered = cache[0].index_select(0, position_ids.reshape(-1))
        return gathered.view(*position_ids.shape, cache.shape[-1]).unsqueeze(1)
    if cache.dim() == 4 and cache.shape[0] == 1 and cache.shape[1] == 1:
        gathered = cache[0, 0].index_select(0, position_ids.reshape(-1))
        return gathered.view(*position_ids.shape, cache.shape[-1]).unsqueeze(1)
    raise NotImplementedError(f"Unsupported rotary cache shape: {tuple(cache.shape)}")


def _apply_rotary(
    states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    selected_cos = _gather_rotary_cache(cos, position_ids).to(dtype=states.dtype, device=states.device)
    selected_sin = _gather_rotary_cache(sin, position_ids).to(dtype=states.dtype, device=states.device)
    return (states * selected_cos) + (_rotate_half(states) * selected_sin)


def _required_rotary_seq_len(position_ids: torch.Tensor, minimum_seq_len: int) -> int:
    if int(position_ids.numel()) == 0:
        return int(minimum_seq_len)
    return max(int(minimum_seq_len), int(position_ids.max().item()) + 1)


def global_maxmin_candidate_exact(
    distance: torch.Tensor,
    saliency_score: torch.Tensor,
    keep_count: int,
) -> torch.Tensor:
    """Build the exact unconstrained max-min candidate with one host transfer.

    NeXT anyres prompts commonly contain more than 2K visual tokens. Running
    one tiny CUDA reduction per greedy step is launch-bound at that size. A
    single D2H transfer followed by vectorized NumPy reductions preserves the
    float32 gain/saliency/index ordering while avoiding hundreds of launches.
    """
    if distance.ndim != 2 or int(distance.shape[0]) != int(distance.shape[1]):
        raise ValueError(f"distance must have shape [N, N], got {tuple(distance.shape)}")
    num_visual = int(distance.shape[0])
    if int(saliency_score.numel()) != num_visual:
        raise ValueError(
            f"saliency_score must contain {num_visual} values, got {saliency_score.numel()}"
        )
    keep_count = min(max(int(keep_count), 0), num_visual)
    if keep_count == 0:
        return torch.empty(0, device=distance.device, dtype=torch.long)

    distance_cpu = (
        distance.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    )
    saliency_cpu = (
        saliency_score.detach().to(device="cpu", dtype=torch.float32).flatten().numpy()
    )
    selected = np.empty(keep_count, dtype=np.int64)
    selected_mask = np.zeros(num_visual, dtype=np.bool_)

    best_saliency = saliency_cpu.max()
    first = int(np.flatnonzero(saliency_cpu == best_saliency)[0])
    selected[0] = first
    selected_mask[first] = True
    min_distance = distance_cpu[:, first].copy()

    for position in range(1, keep_count):
        candidate_mask = ~selected_mask
        best_gain = min_distance[candidate_mask].max()
        gain_mask = candidate_mask & (min_distance == best_gain)
        best_saliency = saliency_cpu[gain_mask].max()
        best = int(np.flatnonzero(gain_mask & (saliency_cpu == best_saliency))[0])
        selected[position] = best
        selected_mask[best] = True
        np.minimum(min_distance, distance_cpu[:, best], out=min_distance)

    return torch.as_tensor(selected, device=distance.device, dtype=torch.long)


@dataclass(frozen=True)
class NextLayerRoutingContext:
    """Read-only state needed to preview the decoder layer after pruning."""

    current_layer_idx: int
    next_layer: Any
    hidden_states: torch.Tensor
    position_ids: torch.Tensor
    causal_mask: torch.Tensor
    visual_start: int
    visual_count: int
    text_start: int
    text_special_token_mask: torch.Tensor | None


class NextLayerPreview:
    """Candidate-independent Q/K/V state for exact query-only layer preview."""

    def __init__(
        self,
        *,
        layer: Any,
        query_positions: torch.Tensor,
        query_residual: torch.Tensor,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        query_attention_mask: torch.Tensor,
        visual_start: int,
        visual_count: int,
        sequence_length: int,
        projection_time_ms: float,
    ) -> None:
        self.layer = layer
        self.query_positions = query_positions
        self.query_residual = query_residual
        self.query_states = query_states
        self.key_states = key_states
        self.value_states = value_states
        self.query_attention_mask = query_attention_mask
        self.visual_start = int(visual_start)
        self.visual_count = int(visual_count)
        self.sequence_length = int(sequence_length)
        self.projection_time_ms = float(projection_time_ms)

    @property
    def query_count(self) -> int:
        return int(self.query_positions.numel())

    def evaluate(self, visual_keep_indices: torch.Tensor) -> torch.Tensor:
        """Return exact next-layer outputs for the selected prompt query rows."""
        keep = torch.as_tensor(
            visual_keep_indices,
            device=self.query_states.device,
            dtype=torch.long,
        ).flatten()
        if int(keep.numel()) > 0:
            if int(keep.min().item()) < 0 or int(keep.max().item()) >= self.visual_count:
                raise ValueError("visual_keep_indices contains an index outside the current visual block")
            if int(torch.unique(keep).numel()) != int(keep.numel()):
                raise ValueError("visual_keep_indices must be unique")

        full_keep_mask = torch.ones(
            self.sequence_length,
            device=self.query_states.device,
            dtype=torch.bool,
        )
        full_keep_mask[self.visual_start : self.visual_start + self.visual_count] = False
        if int(keep.numel()) > 0:
            full_keep_mask[self.visual_start + keep] = True
        key_positions = torch.nonzero(full_keep_mask, as_tuple=False).flatten()

        keys = self.key_states.index_select(2, key_positions)
        values = self.value_states.index_select(2, key_positions)
        attention_mask = self.query_attention_mask.index_select(3, key_positions)
        attention_weights = torch.matmul(self.query_states, keys.transpose(2, 3))
        attention_weights = attention_weights / math.sqrt(int(self.query_states.shape[-1]))
        attention_weights = attention_weights + attention_mask.to(
            dtype=attention_weights.dtype,
            device=attention_weights.device,
        )
        min_value = torch.tensor(
            torch.finfo(attention_weights.dtype).min,
            device=attention_weights.device,
            dtype=attention_weights.dtype,
        )
        attention_weights = torch.maximum(attention_weights, min_value)
        attention_weights = F.softmax(attention_weights, dim=-1, dtype=torch.float32).to(
            self.query_states.dtype
        )

        attention_output = torch.matmul(attention_weights, values)
        batch_size, _, query_count, head_dim = attention_output.shape
        hidden_size = int(self.query_residual.shape[-1])
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size,
            query_count,
            hidden_size,
        )
        attention_output = self.layer.self_attn.o_proj(attention_output)
        hidden = self.query_residual + attention_output
        return hidden + self.layer.mlp(self.layer.post_attention_layernorm(hidden))


def _query_positions(context: NextLayerRoutingContext) -> torch.Tensor:
    seq_len = int(context.hidden_states.shape[1])
    text_start = int(context.text_start)
    if not 0 <= text_start <= seq_len:
        raise ValueError(f"text_start must be in [0, {seq_len}], got {text_start}")
    positions = torch.arange(
        text_start,
        seq_len,
        device=context.hidden_states.device,
        dtype=torch.long,
    )
    special_mask = context.text_special_token_mask
    if special_mask is None:
        return positions
    special_mask = torch.as_tensor(
        special_mask,
        device=context.hidden_states.device,
        dtype=torch.bool,
    ).flatten()
    if int(special_mask.numel()) != int(positions.numel()):
        raise ValueError(
            "text_special_token_mask length must match the post-visual text length, "
            f"got {special_mask.numel()} vs {positions.numel()}"
        )
    return positions[~special_mask]


def build_next_layer_preview(context: NextLayerRoutingContext) -> NextLayerPreview | None:
    """Build the shared exact preview state, or return None when no query exists."""
    hidden_states = context.hidden_states
    position_ids = context.position_ids
    causal_mask = context.causal_mask
    if hidden_states.ndim != 3 or int(hidden_states.shape[0]) != 1:
        raise NotImplementedError(
            f"NLCR currently requires batch size 1 hidden states, got {tuple(hidden_states.shape)}"
        )
    seq_len = int(hidden_states.shape[1])
    if tuple(position_ids.shape) != (1, seq_len):
        raise ValueError(f"position_ids must have shape (1, {seq_len}), got {tuple(position_ids.shape)}")
    if tuple(causal_mask.shape) != (1, 1, seq_len, seq_len):
        raise ValueError(
            f"causal_mask must have shape (1, 1, {seq_len}, {seq_len}), got {tuple(causal_mask.shape)}"
        )

    layer = context.next_layer
    required_layer_attrs = ("input_layernorm", "self_attn", "post_attention_layernorm", "mlp")
    missing_layer_attrs = [name for name in required_layer_attrs if not hasattr(layer, name)]
    if missing_layer_attrs:
        raise NotImplementedError(
            f"NLCR requires a Llama-style decoder layer; missing {missing_layer_attrs}"
        )
    attention = layer.self_attn
    required_attention_attrs = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "rotary_emb",
        "num_heads",
        "head_dim",
    )
    missing_attention_attrs = [name for name in required_attention_attrs if not hasattr(attention, name)]
    if missing_attention_attrs:
        raise NotImplementedError(
            f"NLCR requires Llama-style self attention; missing {missing_attention_attrs}"
        )
    if int(getattr(getattr(attention, "config", None), "pretraining_tp", 1)) != 1:
        raise NotImplementedError("NLCR v1 does not support pretraining_tp values other than 1")

    queries = _query_positions(context)
    if int(queries.numel()) == 0:
        return None

    _sync_if_cuda(hidden_states)
    start = time.perf_counter()
    normalized = layer.input_layernorm(hidden_states)
    normalized_queries = normalized.index_select(1, queries)
    query_residual = hidden_states.index_select(1, queries)

    num_heads = int(attention.num_heads)
    head_dim = int(attention.head_dim)
    num_key_value_heads = int(getattr(attention, "num_key_value_heads", num_heads))
    num_key_value_groups = int(
        getattr(attention, "num_key_value_groups", max(num_heads // num_key_value_heads, 1))
    )
    query_states = attention.q_proj(normalized_queries).view(
        1,
        int(queries.numel()),
        num_heads,
        head_dim,
    ).transpose(1, 2)
    key_states = attention.k_proj(normalized).view(
        1,
        seq_len,
        num_key_value_heads,
        head_dim,
    ).transpose(1, 2)
    value_states = attention.v_proj(normalized).view(
        1,
        seq_len,
        num_key_value_heads,
        head_dim,
    ).transpose(1, 2)

    rotary_seq_len = _required_rotary_seq_len(position_ids, seq_len)
    try:
        cos, sin = attention.rotary_emb(value_states, seq_len=rotary_seq_len)
    except TypeError:
        cos, sin = attention.rotary_emb(value_states, rotary_seq_len)
    query_position_ids = position_ids.index_select(1, queries)
    query_states = _apply_rotary(query_states, cos, sin, query_position_ids)
    key_states = _apply_rotary(key_states, cos, sin, position_ids)
    key_states = _repeat_kv(key_states, num_key_value_groups)
    value_states = _repeat_kv(value_states, num_key_value_groups)
    query_attention_mask = causal_mask.index_select(2, queries)
    _sync_if_cuda(hidden_states)
    projection_time_ms = (time.perf_counter() - start) * 1000.0

    return NextLayerPreview(
        layer=layer,
        query_positions=queries,
        query_residual=query_residual,
        query_states=query_states,
        key_states=key_states,
        value_states=value_states,
        query_attention_mask=query_attention_mask,
        visual_start=context.visual_start,
        visual_count=context.visual_count,
        sequence_length=seq_len,
        projection_time_ms=projection_time_ms,
    )


def compute_visual_effect_errors(
    *,
    phi_keep: torch.Tensor,
    phi_empty: torch.Tensor,
    phi_full: torch.Tensor,
) -> Dict[str, float | bool]:
    """Compute normalized aggregate and worst-query counterfactual errors."""
    if phi_keep.shape != phi_empty.shape or phi_keep.shape != phi_full.shape:
        raise ValueError("phi_keep, phi_empty, and phi_full must have identical shapes")
    difference = phi_keep.to(dtype=torch.float32) - phi_full.to(dtype=torch.float32)
    visual_effect = phi_full.to(dtype=torch.float32) - phi_empty.to(dtype=torch.float32)
    difference_sq = difference.square().sum(dim=-1).flatten()
    effect_sq = visual_effect.square().sum(dim=-1).flatten()
    eps = float(torch.finfo(torch.float32).eps)
    effect_sum = float(effect_sq.sum().item())
    effect_max = float(effect_sq.max().item()) if int(effect_sq.numel()) > 0 else 0.0
    error_sum = float(difference_sq.sum().item()) / max(effect_sum, eps)
    error_max = (
        float(difference_sq.max().item()) / max(effect_max, eps)
        if int(difference_sq.numel()) > 0
        else 0.0
    )
    return {
        "error_sum": error_sum,
        "error_max": error_max,
        "visual_effect_norm_sum": effect_sum,
        "visual_effect_norm_max": effect_max,
        "degenerate_reference": effect_sum <= eps or effect_max <= eps,
    }


def numerical_tolerance(candidate: float, legacy: float) -> float:
    return 8.0 * float(torch.finfo(torch.float32).eps) * max(1.0, abs(candidate), abs(legacy))


def pareto_dominates_legacy(
    candidate: Mapping[str, float],
    legacy: Mapping[str, float],
) -> bool:
    candidate_sum = float(candidate["error_sum"])
    candidate_max = float(candidate["error_max"])
    legacy_sum = float(legacy["error_sum"])
    legacy_max = float(legacy["error_max"])
    if not all(math.isfinite(value) for value in (candidate_sum, candidate_max, legacy_sum, legacy_max)):
        raise ValueError("Counterfactual errors must be finite")
    sum_tolerance = numerical_tolerance(candidate_sum, legacy_sum)
    max_tolerance = numerical_tolerance(candidate_max, legacy_max)
    non_worse = (
        candidate_sum <= legacy_sum + sum_tolerance
        and candidate_max <= legacy_max + max_tolerance
    )
    strictly_better = (
        candidate_sum < legacy_sum - sum_tolerance
        or candidate_max < legacy_max - max_tolerance
    )
    return bool(non_worse and strictly_better)


def select_candidate_deterministically(
    errors: Mapping[str, Mapping[str, float]],
    candidate_order: Sequence[str],
    *,
    legacy_name: str = "legacy_scnd",
) -> Tuple[str, Tuple[str, ...]]:
    if legacy_name not in errors:
        raise ValueError(f"Missing legacy candidate {legacy_name!r}")
    legacy = errors[legacy_name]
    admissible = tuple(
        name
        for name in candidate_order
        if name != legacy_name
        and name in errors
        and pareto_dominates_legacy(errors[name], legacy)
    )
    if not admissible:
        return legacy_name, ()
    order_index = {name: index for index, name in enumerate(candidate_order)}
    selected = min(
        admissible,
        key=lambda name: (
            float(errors[name]["error_max"]),
            float(errors[name]["error_sum"]),
            order_index[name],
        ),
    )
    return selected, admissible
