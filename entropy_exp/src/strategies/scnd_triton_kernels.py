"""Triton kernels for exact SCND native C-layer selection."""

from __future__ import annotations

from typing import Tuple

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - exercised on machines without Triton.
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


MAX_TRITON_TOKENS = 1024


def is_scnd_triton_available() -> bool:
    return bool(_TRITON_AVAILABLE and torch.cuda.is_available())


def _next_power_of_2(value: int) -> int:
    return 1 << (max(int(value), 1) - 1).bit_length()


if _TRITON_AVAILABLE:

    @triton.jit
    def _best_high_gain_saliency_low_index(
        candidate_mask,
        gains,
        saliency,
        offsets,
        BLOCK_N: tl.constexpr,
    ):
        neg_inf = -float("inf")
        big_index = 2147483647
        masked_gain = tl.where(candidate_mask, gains, neg_inf)
        best_gain = tl.max(masked_gain, axis=0)
        gain_mask = candidate_mask & (gains == best_gain)
        masked_saliency = tl.where(gain_mask, saliency, neg_inf)
        best_saliency = tl.max(masked_saliency, axis=0)
        final_mask = gain_mask & (saliency == best_saliency)
        best_idx = tl.min(tl.where(final_mask, offsets, big_index), axis=0)
        return best_idx, best_gain

    @triton.jit
    def _scnd_native_c_select_kernel(
        distance_ptr,
        saliency_ptr,
        saliency_desc_ptr,
        selected_mask_ptr,
        seed_mask_ptr,
        selected_order_ptr,
        diversity_gain_ptr,
        feasible_count_ptr,
        distance_stride0: tl.constexpr,
        distance_stride1: tl.constexpr,
        num_visual: tl.constexpr,
        keep_count: tl.constexpr,
        seed_count,
        seed_pool_count,
        saliency_mass_floor_ptr,
        BLOCK_N: tl.constexpr,
    ):
        offsets = tl.arange(0, BLOCK_N)
        valid = offsets < num_visual
        saliency = tl.load(saliency_ptr + offsets, mask=valid, other=-float("inf")).to(tl.float32)
        positive_saliency = tl.maximum(saliency, 0.0)

        selected_mask = tl.zeros((BLOCK_N,), dtype=tl.int32)
        seed_mask = tl.zeros((BLOCK_N,), dtype=tl.int32)
        min_distance = tl.zeros((BLOCK_N,), dtype=tl.float32)
        selected_count = 0
        selected_mass = 0.0
        feasible_total = 0
        saliency_mass_floor = tl.load(saliency_mass_floor_ptr).to(tl.float32)

        rank_by_index = tl.full((BLOCK_N,), 2147483647, dtype=tl.int32)
        rank = 0
        while rank < num_visual:
            desc_idx = tl.load(saliency_desc_ptr + rank).to(tl.int32)
            rank_by_index = tl.where(offsets == desc_idx, rank, rank_by_index)
            rank += 1

        if seed_count > 0:
            first_idx = tl.load(saliency_desc_ptr).to(tl.int32)
            selected_mask = tl.where(offsets == first_idx, 1, selected_mask)
            seed_mask = tl.where(offsets == first_idx, 1, seed_mask)
            selected_order_offset = selected_order_ptr + selected_count
            diversity_gain_offset = diversity_gain_ptr + selected_count
            tl.store(selected_order_offset, first_idx)
            tl.store(diversity_gain_offset, 0.0)
            first_saliency = tl.load(saliency_ptr + first_idx).to(tl.float32)
            selected_mass += tl.maximum(first_saliency, 0.0)
            min_distance = tl.load(
                distance_ptr + offsets * distance_stride0 + first_idx * distance_stride1,
                mask=valid,
                other=0.0,
            ).to(tl.float32)
            selected_count += 1

        while selected_count < seed_count:
            candidate_mask = valid & (rank_by_index < seed_pool_count) & (selected_mask == 0)
            best_idx, best_gain = _best_high_gain_saliency_low_index(
                candidate_mask,
                min_distance,
                saliency,
                offsets,
                BLOCK_N,
            )
            selected_mask = tl.where(offsets == best_idx, 1, selected_mask)
            seed_mask = tl.where(offsets == best_idx, 1, seed_mask)
            tl.store(selected_order_ptr + selected_count, best_idx)
            tl.store(diversity_gain_ptr + selected_count, best_gain)
            best_saliency = tl.load(saliency_ptr + best_idx).to(tl.float32)
            selected_mass += tl.maximum(best_saliency, 0.0)
            distance_to_best = tl.load(
                distance_ptr + offsets * distance_stride0 + best_idx * distance_stride1,
                mask=valid,
                other=0.0,
            ).to(tl.float32)
            min_distance = tl.minimum(min_distance, distance_to_best)
            selected_count += 1

        while selected_count < keep_count:
            remaining_mask = valid & (selected_mask == 0)
            slots_after_candidate = keep_count - selected_count - 1
            remaining_rank_by_index = tl.full((BLOCK_N,), 2147483647, dtype=tl.int32)
            top_s_sum = 0.0
            top_s_plus_sum = 0.0
            remaining_rank = 0
            desc_pos = 0

            while desc_pos < num_visual:
                desc_idx = tl.load(saliency_desc_ptr + desc_pos).to(tl.int32)
                is_selected = tl.sum(tl.where(offsets == desc_idx, selected_mask, 0), axis=0) > 0
                if not is_selected:
                    desc_saliency = tl.maximum(tl.load(saliency_ptr + desc_idx).to(tl.float32), 0.0)
                    top_s_sum += tl.where(remaining_rank < slots_after_candidate, desc_saliency, 0.0)
                    top_s_plus_sum += tl.where(remaining_rank < slots_after_candidate + 1, desc_saliency, 0.0)
                    remaining_rank_by_index = tl.where(offsets == desc_idx, remaining_rank, remaining_rank_by_index)
                    remaining_rank += 1
                desc_pos += 1

            candidate_in_top_s = remaining_rank_by_index < slots_after_candidate
            possible_remaining = tl.where(
                candidate_in_top_s,
                top_s_plus_sum - positive_saliency,
                top_s_sum,
            )
            possible_mass = selected_mass + positive_saliency + possible_remaining
            feasible_mask = remaining_mask & (possible_mass + 1.0e-8 >= saliency_mass_floor)
            feasible_count = tl.sum(tl.where(feasible_mask, 1, 0), axis=0)
            feasible_total += feasible_count
            candidate_mask = tl.where(feasible_count > 0, feasible_mask, remaining_mask)
            gains = tl.where(selected_count > 0, min_distance, tl.zeros((BLOCK_N,), dtype=tl.float32))
            best_idx, best_gain = _best_high_gain_saliency_low_index(
                candidate_mask,
                gains,
                saliency,
                offsets,
                BLOCK_N,
            )

            selected_mask = tl.where(offsets == best_idx, 1, selected_mask)
            tl.store(selected_order_ptr + selected_count, best_idx)
            tl.store(diversity_gain_ptr + selected_count, best_gain)
            best_saliency = tl.load(saliency_ptr + best_idx).to(tl.float32)
            selected_mass += tl.maximum(best_saliency, 0.0)
            distance_to_best = tl.load(
                distance_ptr + offsets * distance_stride0 + best_idx * distance_stride1,
                mask=valid,
                other=0.0,
            ).to(tl.float32)
            min_distance = tl.where(
                selected_count > 0,
                tl.minimum(min_distance, distance_to_best),
                distance_to_best,
            )
            selected_count += 1

        tl.store(selected_mask_ptr + offsets, selected_mask, mask=valid)
        tl.store(seed_mask_ptr + offsets, seed_mask, mask=valid)
        tl.store(feasible_count_ptr, feasible_total)


def scnd_native_c_select_triton(
    *,
    distance: torch.Tensor,
    saliency_score: torch.Tensor,
    saliency_desc: torch.Tensor,
    keep_count: int,
    seed_count: int,
    seed_pool_count: int,
    saliency_mass_floor: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Run exact native SCND C-layer selection in one Triton program."""
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available; install triton before using selection_backend='triton'")
    if not distance.is_cuda or not saliency_score.is_cuda:
        raise RuntimeError("selection_backend='triton' requires CUDA tensors")
    if int(saliency_score.numel()) != int(distance.shape[0]) or int(distance.shape[0]) != int(distance.shape[1]):
        raise ValueError(
            "SCND Triton selection expects distance shape [N, N] matching saliency_score, "
            f"got distance={tuple(distance.shape)}, saliency={tuple(saliency_score.shape)}"
        )

    num_visual = int(saliency_score.numel())
    keep_count = min(max(int(keep_count), 0), num_visual)
    seed_count = min(max(int(seed_count), 0), keep_count)
    seed_pool_count = min(max(int(seed_pool_count), seed_count), num_visual)
    if num_visual > MAX_TRITON_TOKENS:
        raise RuntimeError(
            f"SCND Triton exact selection supports at most {MAX_TRITON_TOKENS} visual tokens, got {num_visual}"
        )

    distance = distance.contiguous()
    saliency_score = saliency_score.to(device=distance.device, dtype=torch.float32).contiguous()
    saliency_desc = saliency_desc.to(device=distance.device, dtype=torch.long).contiguous()
    selected_mask_i32 = torch.empty(num_visual, device=distance.device, dtype=torch.int32)
    seed_mask_i32 = torch.empty(num_visual, device=distance.device, dtype=torch.int32)
    selected_order = torch.empty(keep_count, device=distance.device, dtype=torch.long)
    diversity_gain = torch.empty(keep_count, device=distance.device, dtype=torch.float32)
    feasible_count = torch.empty(1, device=distance.device, dtype=torch.int64)
    block_n = _next_power_of_2(num_visual)

    _scnd_native_c_select_kernel[(1,)](
        distance,
        saliency_score,
        saliency_desc,
        selected_mask_i32,
        seed_mask_i32,
        selected_order,
        diversity_gain,
        feasible_count,
        distance.stride(0),
        distance.stride(1),
        num_visual,
        keep_count,
        seed_count,
        seed_pool_count,
        saliency_mass_floor.to(device=distance.device, dtype=torch.float32),
        BLOCK_N=block_n,
    )

    return (
        selected_mask_i32.to(dtype=torch.bool),
        seed_mask_i32.to(dtype=torch.bool),
        selected_order,
        diversity_gain,
        int(feasible_count.item()),
    )
