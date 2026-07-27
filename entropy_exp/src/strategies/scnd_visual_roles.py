"""Visual-token role helpers for LLaVA-NeXT any-resolution sequences."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import torch


ROLE_BASE = 0
ROLE_LOCAL_PATCH = 1
ROLE_NEWLINE = 2

VIEW_BASE = 0
VIEW_LOCAL = 1
STRUCTURE_UNASSIGNED = -1

ROLE_NAMES = {
    ROLE_BASE: "base",
    ROLE_LOCAL_PATCH: "local_patch",
    ROLE_NEWLINE: "newline",
}


def visual_token_role_ids(
    original_indices: Any,
    visual_layout: Dict[str, Any],
    *,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Map merged-sequence token indices to base/local/newline role ids."""
    indices = torch.as_tensor(original_indices, dtype=torch.long, device=device).flatten()
    base_count = int(visual_layout.get("base_token_count", 0))
    local_width = int(visual_layout.get("local_grid_width", 0))
    newline_count = int(visual_layout.get("newline_token_count", 0))
    total_count = int(visual_layout.get("total_token_count", 0))

    if total_count <= 0:
        raise ValueError(f"visual_layout.total_token_count must be positive, got {total_count}")
    if bool(((indices < 0) | (indices >= total_count)).any().item()):
        invalid = indices[(indices < 0) | (indices >= total_count)][0]
        raise ValueError(f"Visual token index {int(invalid.item())} is outside layout size {total_count}")

    roles = torch.full_like(indices, ROLE_LOCAL_PATCH)
    roles[indices < base_count] = ROLE_BASE
    if newline_count > 0 and local_width > 0:
        local_mask = indices >= base_count
        relative = indices - base_count
        newline_mask = local_mask & (relative.remainder(local_width + 1) == local_width)
        roles[newline_mask] = ROLE_NEWLINE
    return roles


def visual_token_role_names(original_indices: Any, visual_layout: Dict[str, Any]) -> List[str]:
    role_ids = visual_token_role_ids(original_indices, visual_layout, device=torch.device("cpu"))
    return [ROLE_NAMES[int(role_id)] for role_id in role_ids.tolist()]


def _square_side(token_count: int, field_name: str) -> int:
    side = int(round(math.sqrt(float(token_count))))
    if side <= 0 or side * side != int(token_count):
        raise ValueError(f"{field_name} must be a positive square token count, got {token_count}")
    return side


def visual_token_structure_ids(
    original_indices: Any,
    visual_layout: Dict[str, Any],
    *,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """Map original visual indices to view, macrocell, and row structure IDs."""
    indices = torch.as_tensor(original_indices, dtype=torch.long, device=device).flatten()
    role_ids = visual_token_role_ids(indices, visual_layout, device=indices.device)
    base_count = int(visual_layout.get("base_token_count", 0))
    total_count = int(visual_layout.get("total_token_count", 0))
    layout_kind = str(visual_layout.get("layout_kind", "square"))
    base_side = _square_side(base_count, "visual_layout.base_token_count")

    view_ids = torch.full_like(indices, STRUCTURE_UNASSIGNED)
    macrocell_ids = torch.full_like(indices, STRUCTURE_UNASSIGNED)
    row_ids = torch.full_like(indices, STRUCTURE_UNASSIGNED)
    patch_mask = role_ids != ROLE_NEWLINE

    base_mask = indices < base_count
    if bool(base_mask.any().item()):
        base_indices = indices[base_mask]
        view_ids[base_mask] = VIEW_BASE
        macrocell_ids[base_mask] = 0
        row_ids[base_mask] = torch.div(base_indices, base_side, rounding_mode="floor")

    local_mask = indices >= base_count
    if not bool(local_mask.any().item()):
        return {
            "role_ids": role_ids,
            "view_ids": view_ids,
            "macrocell_ids": macrocell_ids,
            "row_ids": row_ids,
            "patch_mask": patch_mask,
        }

    view_ids[local_mask] = VIEW_LOCAL
    local_indices = indices[local_mask]
    relative = local_indices - base_count

    if layout_kind == "anyres_flat":
        local_count = total_count - base_count
        if local_count < 0 or local_count % base_count != 0:
            raise ValueError(
                "anyres_flat local token count must be divisible by base_token_count, "
                f"got local={local_count}, base={base_count}"
            )
        crop_id = torch.div(relative, base_count, rounding_mode="floor")
        within_crop = relative.remainder(base_count)
        macrocell_ids[local_mask] = 1 + crop_id
        row_ids[local_mask] = base_side + crop_id * base_side + torch.div(
            within_crop,
            base_side,
            rounding_mode="floor",
        )
    elif layout_kind.startswith("anyres_spatial"):
        local_width = int(visual_layout.get("local_grid_width", 0))
        local_height = int(visual_layout.get("local_grid_height", 0))
        crop_grid_width = int(visual_layout.get("crop_grid_width", 0))
        crop_grid_height = int(visual_layout.get("crop_grid_height", 0))
        newline_count = int(visual_layout.get("newline_token_count", 0))
        if min(local_width, local_height, crop_grid_width, crop_grid_height) <= 0:
            raise ValueError(
                "Spatial anyres layout requires positive local and crop-grid dimensions, got "
                f"local={local_width}x{local_height}, crop={crop_grid_width}x{crop_grid_height}"
            )
        stride = local_width + (1 if newline_count > 0 else 0)
        local_y = torch.div(relative, stride, rounding_mode="floor")
        local_x = relative.remainder(stride)
        if bool((local_y >= local_height).any().item()):
            raise ValueError("Spatial anyres token index exceeds the declared local grid height")
        row_ids[local_mask] = base_side + local_y

        local_patch_mask = local_x < local_width
        local_positions = torch.nonzero(local_mask, as_tuple=False).flatten()
        patch_positions = local_positions[local_patch_mask]
        if int(patch_positions.numel()) > 0:
            patch_x = local_x[local_patch_mask]
            patch_y = local_y[local_patch_mask]
            cell_x = torch.clamp(
                torch.div(patch_x * crop_grid_width, local_width, rounding_mode="floor"),
                max=crop_grid_width - 1,
            )
            cell_y = torch.clamp(
                torch.div(patch_y * crop_grid_height, local_height, rounding_mode="floor"),
                max=crop_grid_height - 1,
            )
            macrocell_ids[patch_positions] = 1 + cell_y * crop_grid_width + cell_x
    elif layout_kind != "square":
        raise ValueError(f"Unsupported visual layout kind for SCND structure mapping: {layout_kind!r}")
    else:
        raise ValueError("Square visual layout cannot contain local visual-token indices")

    return {
        "role_ids": role_ids,
        "view_ids": view_ids,
        "macrocell_ids": macrocell_ids,
        "row_ids": row_ids,
        "patch_mask": patch_mask,
    }
