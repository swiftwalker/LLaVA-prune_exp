"""Visual-token role helpers for LLaVA-NeXT any-resolution sequences."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch


ROLE_BASE = 0
ROLE_LOCAL_PATCH = 1
ROLE_NEWLINE = 2

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
