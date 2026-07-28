"""
Pruning inference pipeline for LLaVA.

Runs inference on supported VQA-style datasets with visual token pruning.
Outputs:
  1. Answer JSONL files
  2. Pruning statistics JSONL (per-sample pruning details)

Usage:
    python entropy_exp/src/prune_inference.py \
        --config entropy_exp/configs/prune.yaml \
        --dataset mme \
        [--max-samples 10]
"""

import argparse
import os
import sys
import json
import random
import time
import datetime
import yaml
import re
from typing import Any, Optional
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
# Limit PyTorch CPU threads to avoid contention across concurrent experiments.
# Default is ALL cores (192 on this machine); N experiments = N*192 threads
# thrashing on 384 cores.  4 threads per experiment is sufficient for the
# GPU-bound inference workload here.
_cpu_threads = os.environ.get("OMP_NUM_THREADS")
if _cpu_threads is None:
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ.setdefault("MKL_NUM_THREADS", "4")
import h5py
import torch
if _cpu_threads is None:
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# Add LLaVA root to path
LLAVA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, LLAVA_ROOT)
# Add src/ to path so strategy/pruner imports work
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    tokenizer_image_token,
    process_images,
    load_image_from_base64,
    get_model_name_from_path,
    get_visual_token_layout,
)

from hooks import locate_image_tokens
from pruner import VisualTokenPruner, enable_sparse_position_ids_compat
from run_layout import build_run_dir, build_run_rel_dir
from strategies import get_strategy
from strategies.scnd_entropy_calibration import model_config_fingerprint
from strategies.scnd_visual_roles import visual_token_role_names
from dataset_adapters import SUPPORTED_DATASETS, load_dataset_samples

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


MODEL_CONFIG_METADATA_KEYS = (
    "_name_or_path",
    "model_type",
    "architectures",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "mm_vision_tower",
    "mm_vision_select_layer",
    "mm_vision_select_feature",
    "image_aspect_ratio",
    "image_grid_pinpoints",
    "mm_patch_merge_type",
    "mm_projector_type",
    "tokenizer_model_max_length",
    "max_position_embeddings",
    "torch_dtype",
)


# ---------------------------------------------------------------------------
# Dataset (same as inference.py)
# ---------------------------------------------------------------------------
class VQADataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config, conv_mode):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.conv_mode = conv_mode

    def __getitem__(self, index):
        line = self.questions[index]
        qs = line["text"]
        if line.get("has_image", False):
            if getattr(self.model_config, 'mm_use_im_start_end', False):
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image_tensor = None
        image_size = None
        image_path = line.get("image_path")
        image_base64 = line.get("image_base64")
        if image_path:
            resolved_image_path = image_path
            if not os.path.isabs(resolved_image_path):
                resolved_image_path = os.path.join(self.image_folder, resolved_image_path)
            image = Image.open(resolved_image_path).convert('RGB')
            image_tensor = process_images([image], self.image_processor, self.model_config)[0]
            image_size = image.size
        elif image_base64:
            image = load_image_from_base64(image_base64).convert('RGB')
            image_tensor = process_images([image], self.image_processor, self.model_config)[0]
            image_size = image.size

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        return input_ids, image_tensor, image_size

    def __len__(self):
        return len(self.questions)


def strip_prompt_prefix_if_present(output_ids: torch.Tensor, input_ids: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Remove the prompt prefix only when generate() returned prompt+continuation."""
    prompt_length = int(input_ids.shape[1])
    output_length = int(output_ids.shape[1])
    if output_length < prompt_length:
        return output_ids, False

    input_ids_for_compare = input_ids.to(device=output_ids.device)
    if torch.equal(output_ids[:, :prompt_length], input_ids_for_compare):
        return output_ids[:, prompt_length:], True

    return output_ids, False


def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    if image_tensors[0] is None:
        image_tensors = None
    else:
        image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes


def _normalize_question_id_set(raw_question_ids: Any) -> set[str]:
    if raw_question_ids is None:
        return set()
    if isinstance(raw_question_ids, (str, int)):
        return {str(raw_question_ids)}
    return {str(item) for item in raw_question_ids}


def _normalize_int_set(raw_values: Any) -> set[int]:
    if raw_values is None:
        return set()
    if isinstance(raw_values, int):
        return {int(raw_values)}
    if isinstance(raw_values, str):
        raw_values = raw_values.strip()
        if not raw_values:
            return set()
        return {int(raw_values)}
    return {int(item) for item in raw_values}


def _h5_dataset_kwargs(cfg: dict) -> dict:
    compression = cfg.get("compression", "gzip")
    if not compression:
        return {}
    kwargs = {"compression": compression}
    compression_opts = cfg.get("compression_opts", 4)
    if compression_opts is not None:
        kwargs["compression_opts"] = compression_opts
    return kwargs


def build_text_special_token_mask(tokenizer, text_token_ids: torch.Tensor) -> torch.Tensor:
    """Build a boolean mask over text-side token ids for tokenizer special tokens."""
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    ids = text_token_ids.tolist()
    mask = [int(token_id) in special_ids for token_id in ids]
    return torch.tensor(mask, dtype=torch.bool, device=text_token_ids.device)


def _serialize_optional_sequence(value: Any) -> Any:
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    return value


def _set_optional_layer_field(sample_stats: dict, field_name: str, value: Any) -> None:
    serialized = _serialize_optional_sequence(value)
    if serialized is not None:
        sample_stats[field_name] = serialized


def _normalized_positive_entropy(values: Any) -> tuple[float, float]:
    scores = torch.as_tensor(values, dtype=torch.float64).flatten().clamp_min(0.0)
    num_tokens = int(scores.numel())
    if num_tokens <= 1:
        return 0.0, 0.0
    score_sum = scores.sum()
    if float(score_sum.item()) <= 0.0:
        return 0.0, 0.0
    probabilities = scores / score_sum
    positive = probabilities > 0
    entropy = -(probabilities[positive] * torch.log(probabilities[positive])).sum()
    entropy_norm = torch.clamp(entropy / torch.log(scores.new_tensor(float(num_tokens))), 0.0, 1.0)
    return float(entropy.item()), float(entropy_norm.item())


def _visual_token_roles(original_indices: Any, visual_layout: dict[str, Any]) -> list[str]:
    indices = _serialize_optional_sequence(original_indices) or []
    return visual_token_role_names(indices, visual_layout)


def compute_visual_token_role_stats(layer_info: dict, visual_layout: Optional[dict]) -> dict[str, Any]:
    """Summarize anyres token roles without exporting per-token arrays."""
    if not visual_layout:
        return {}
    current_indices = layer_info.get("current_patch_indices")
    keep_indices = layer_info.get("keep_patch_indices", layer_info.get("keep_indices"))
    if current_indices is None or keep_indices is None:
        return {}

    current_roles = _visual_token_roles(current_indices, visual_layout)
    keep_roles = _visual_token_roles(keep_indices, visual_layout)
    role_stats: dict[str, Any] = {}
    for role, field_prefix in (
        ("base", "base_token"),
        ("local_patch", "local_patch_token"),
        ("newline", "newline_token"),
    ):
        before = current_roles.count(role)
        after = keep_roles.count(role)
        role_stats[f"{field_prefix}_count_before"] = before
        role_stats[f"{field_prefix}_count_after"] = after
        role_stats[f"{field_prefix}_keep_rate"] = float(after / before) if before else None

    importance_scores = layer_info.get("importance_scores")
    if importance_scores is None:
        return role_stats
    scores = torch.as_tensor(importance_scores, dtype=torch.float64).flatten()
    if int(scores.numel()) != len(current_roles):
        raise ValueError(
            "importance_scores length must match current visual-token indices for role diagnostics, "
            f"got {scores.numel()} vs {len(current_roles)}"
        )

    patch_mask = torch.tensor(
        [role != "newline" for role in current_roles],
        dtype=torch.bool,
    )
    patch_entropy, patch_entropy_norm = _normalized_positive_entropy(scores[patch_mask])
    role_stats.update(
        {
            "patch_only_saliency_entropy": patch_entropy,
            "patch_only_saliency_entropy_norm": patch_entropy_norm,
            "patch_only_saliency_token_count": int(patch_mask.sum().item()),
        }
    )
    positive_scores = scores.clamp_min(0.0)
    positive_sum = positive_scores.sum()
    for role, field_prefix in (
        ("base", "base_token"),
        ("local_patch", "local_patch_token"),
        ("newline", "newline"),
    ):
        role_mask = torch.tensor([value == role for value in current_roles], dtype=torch.bool)
        role_stats[f"{field_prefix}_saliency_mass_ratio"] = (
            float(positive_scores[role_mask].sum().item() / positive_sum.item())
            if float(positive_sum.item()) > 0.0
            else 0.0
        )

    target_keep = layer_info.get(
        "target_keep",
        layer_info.get("num_visual_after", len(keep_roles)),
    )
    if target_keep is not None:
        target_keep = min(max(int(target_keep), 0), len(current_roles))
        saliency_order = torch.argsort(scores, descending=True, stable=True)[:target_keep].tolist()
        topk_roles = [current_roles[int(index)] for index in saliency_order]
        for role, field_prefix in (
            ("base", "base_token"),
            ("local_patch", "local_patch_token"),
            ("newline", "newline_token"),
        ):
            role_stats[f"topk_{field_prefix}_count"] = topk_roles.count(role)
    return role_stats


def append_layer_stats_fields(
    sample_stats: dict,
    layer_idx: int,
    layer_info: dict,
    *,
    save_importance: bool,
    save_indices: bool,
    visual_layout: Optional[dict] = None,
) -> None:
    sample_stats[f"layer_{layer_idx}_ratio"] = layer_info["prune_ratio"]
    sample_stats[f"layer_{layer_idx}_before"] = layer_info["num_visual_before"]
    sample_stats[f"layer_{layer_idx}_after"] = layer_info["num_visual_after"]
    sample_stats[f"layer_{layer_idx}_pruned"] = layer_info["num_pruned"]

    if save_importance:
        _set_optional_layer_field(
            sample_stats,
            f"layer_{layer_idx}_importance",
            layer_info.get("importance_scores"),
        )
        _set_optional_layer_field(
            sample_stats,
            f"layer_{layer_idx}_text_relevance_scores",
            layer_info.get("text_relevance_scores"),
        )

    keep_indices = layer_info.get("keep_indices")
    pruned_indices = layer_info.get("pruned_indices")
    keep_patch_indices = layer_info.get("keep_patch_indices", keep_indices)
    pruned_patch_indices = layer_info.get("pruned_patch_indices", pruned_indices)

    if save_indices:
        for key, value in (
            ("keep_indices", keep_indices),
            ("pruned_indices", pruned_indices),
            ("keep_patch_indices", keep_patch_indices),
            ("pruned_patch_indices", pruned_patch_indices),
            ("rater_indices", layer_info.get("rater_indices")),
            ("current_patch_indices", layer_info.get("current_patch_indices")),
            ("high_keep_indices", layer_info.get("high_keep_indices")),
            ("low_keep_indices", layer_info.get("low_keep_indices")),
            ("high_keep_patch_indices", layer_info.get("high_keep_patch_indices")),
            ("low_keep_patch_indices", layer_info.get("low_keep_patch_indices")),
            ("stratum_selected_counts", layer_info.get("stratum_selected_counts")),
            ("stratum_candidate_counts", layer_info.get("stratum_candidate_counts")),
            ("stratum_deficits", layer_info.get("stratum_deficits")),
            ("stratum_quotas", layer_info.get("stratum_quotas")),
        ):
            _set_optional_layer_field(sample_stats, f"layer_{layer_idx}_{key}", value)

    for key, value in compute_visual_token_role_stats(layer_info, visual_layout).items():
        if value is not None:
            sample_stats[f"layer_{layer_idx}_{key}"] = value

    for key in (
        "target_keep",
        "strategy_keep_high",
        "strategy_keep_low",
        "grid_size",
        "patch_per_row",
        "grid_balance_weight",
        "high_ratio",
        "intra_stratum_mode",
        "adaptive_alpha",
        "saliency_entropy_norm",
        "saliency_entropy_norm_control",
        "entropy_calibration_mode",
        "entropy_calibration_applied",
        "entropy_calibration_q_low",
        "entropy_calibration_q_high",
        "entropy_calibration_clipped_low",
        "entropy_calibration_clipped_high",
        "entropy_calibration_artifact_path",
        "entropy_calibration_quantile_value",
        "entropy_calibration_budget_keep_fraction",
        "entropy_calibration_budget_pressure",
        "entropy_calibration_budget_keep_fraction_low",
        "entropy_calibration_budget_keep_fraction_high",
        "entropy_calibration_diversity_tail_gain",
        "entropy_calibration_diversity_tail_delta",
        "visual_role_constraint_mode",
        "visual_role_constraint_applied",
        "visual_role_budget_keep_fraction",
        "visual_role_budget_pressure",
        "visual_role_local_floor_ratio",
        "visual_role_local_floor_ratio_effective",
        "visual_role_budget_keep_fraction_low",
        "visual_role_budget_keep_fraction_high",
        "visual_role_num_visual_tokens_before",
        "visual_role_local_saliency_mass_ratio",
        "visual_role_max_visual_tokens_before",
        "visual_role_min_local_saliency_mass_ratio",
        "visual_role_layout_gate_passed",
        "visual_role_saliency_gate_passed",
        "visual_role_gate_passed",
        "topk_local_patch_count",
        "local_patch_floor_count",
        "selected_local_patch_count",
        "evidence_reconcile_mode",
        "evidence_reconcile_stages",
        "evidence_reconcile_applied",
        "evidence_reconcile_bypassed",
        "evidence_reconcile_profile_pressure",
        "evidence_reconcile_profile_keep_fraction",
        "evidence_reconcile_candidate_count",
        "evidence_reconcile_swap_budget",
        "evidence_reconcile_query_swap_count",
        "evidence_reconcile_context_swap_count",
        "evidence_reconcile_deficit",
        "evidence_reconcile_time_ms",
        "evidence_hierarchy_aggregation",
        "evidence_query_coverage_before",
        "evidence_query_coverage_after_query",
        "evidence_query_coverage_after_context",
        "evidence_query_current_coverage_after",
        "evidence_query_c_reference_coverage_after",
        "evidence_view_coverage_before",
        "evidence_view_coverage_after",
        "evidence_macrocell_coverage_before",
        "evidence_macrocell_coverage_after",
        "evidence_row_coverage_before",
        "evidence_row_coverage_after",
        "evidence_representative_coverage_before",
        "evidence_representative_coverage_after",
        "evidence_scalar_mass_floor",
        "evidence_scalar_mass_before",
        "evidence_scalar_mass_after",
        "evidence_rater_count",
        "evidence_rater_js_disagreement",
        "global_prune_step",
        "global_current_weight",
        "global_ema_decay",
        "global_debt_weight",
        "global_first_layer_fallback",
        "global_use_global",
        "global_use_ema",
        "use_score_memory",
        "layer_mode",
        "layer_strategy_effective",
        "selection_rule",
        "boost_weight",
        "boost_weight_min",
        "boost_weight_max",
        "beta",
        "beta_min",
        "beta_max",
        "shuffle_seed",
        "sampling_fill_count",
        "diversity_ratio",
        "anchor_ratio",
        "lambda_div",
        "diversity_weight_max",
        "core_keep_ratio",
        "candidate_pool_multiplier",
        "candidate_pool_multiplier_effective",
        "min_candidate_multiplier",
        "max_candidate_multiplier",
        "budget_target",
        "layer_policy",
        "candidate_alpha",
        "candidate_beta",
        "candidate_rho",
        "candidate_cap_multiplier",
        "candidate_cap_fraction",
        "candidate_pool_size",
        "candidate_step_ratio",
        "candidate_gate_floor",
        "candidate_cap_fraction_count",
        "reservoir_size",
        "saliency_pool_count",
        "global_reservoir_size",
        "global_reservoir_count",
        "global_reservoir_reserved_count",
        "global_reservoir_fraction",
        "candidate_cap_count",
        "candidate_pool_saliency_count",
        "candidate_pool_reservoir_count",
        "candidate_pool_reservoir_only_count",
        "candidate_pool_global_reservoir_count",
        "candidate_pool_global_reservoir_only_count",
        "saliency_core_count",
        "saliency_core_ratio",
        "diversity_fill_budget",
        "diversity_steps_budget",
        "diversity_steps_used",
        "diversity_gate",
        "diversity_gate_raw",
        "selection_time_ms",
        "distance_time_ms",
        "distance_cost_proxy",
        "selection_backend_requested",
        "selection_backend_effective",
        "profile_version",
        "diversity_decay_per_step",
        "distance_metric",
        "mean_selected_pairwise_distance",
        "retained_saliency_mass_ratio",
        "seed_ratio",
        "seed_count",
        "seed_ratio_min",
        "seed_ratio_max",
        "seed_pool_multiplier",
        "saliency_floor_eta",
        "saliency_mass_floor",
        "saliency_mass_selected",
        "saliency_mass_topk",
        "saliency_floor_min",
        "saliency_floor_max",
        "feasible_candidate_count",
        "boundary_ratio",
        "boundary_count",
        "boundary_pool_multiplier",
        "boundary_gate_min",
        "blite_enabled",
        "blite_disabled_reason",
        "blite_mode",
        "native_boundary_used",
        "margin_confidence",
        "saliency_repair",
        "core_ratio",
        "core_ratio_min",
        "core_ratio_max",
        "saliency_pool_multiplier",
        "reservoir_multiplier",
        "reservoir_rank_multiplier",
        "projection_dim",
        "greedy_steps",
        "greedy_steps_used",
        "reservoir_source_count",
        "tie_break_band_ratio",
        "tie_break_band_max",
        "tie_break_band_count",
        "cached_diversity_weight",
    ):
        if key in layer_info and layer_info[key] is not None:
            sample_stats[f"layer_{layer_idx}_{key}"] = _serialize_optional_sequence(layer_info[key])

    if not save_indices:
        return

    for key in (
        "global_selection_score",
        "global_saliency_ema",
        "current_saliency_norm",
        "current_rank_score",
        "mixed_score",
        "token_boost",
        "adjusted_scores",
        "stratum_deficit",
        "sampling_weights",
        "stratum_history_debt",
        "stratum_combined_debt",
        "core_indices",
        "candidate_pool_indices",
        "mmr_selected_order",
        "diversity_gain",
        "mmr_scores",
        "grid_saliency_topk_counts",
        "grid_quota",
        "grid_selected_counts",
        "grid_candidate_counts",
        "grid_anchor_indices",
        "grid_fill_indices",
        "grid_diverse_fill_indices",
        "grid_distance_selected_order",
        "grid_candidate_indices",
        "grid_global_fallback_indices",
        "seed_pool_indices",
        "seed_indices",
        "saliency_pool_indices",
        "reservoir_source_indices",
        "reservoir_indices",
        "reservoir_reserved_indices",
        "global_reservoir_indices",
        "global_reservoir_reserved_indices",
        "saliency_core_indices",
        "cached_diversity_gain",
        "tie_break_scores",
        "repair_replacements",
        "boundary_core_indices",
        "boundary_pool_indices",
        "boundary_fill_indices",
        "evidence_reconcile_in_indices",
        "evidence_reconcile_out_indices",
        "evidence_reconcile_in_patch_indices",
        "evidence_reconcile_out_patch_indices",
        "evidence_reconcile_candidate_indices",
    ):
        _set_optional_layer_field(sample_stats, f"layer_{layer_idx}_{key}", layer_info.get(key))


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def _parse_value(raw: str):
    """Parse a CLI override value string into a Python object.

    Supports: int, float, bool, null/None, JSON lists, and plain strings.
    """
    stripped = raw.strip()
    if stripped.lower() in ("null", "none", "~"):
        return None
    if stripped.lower() == "true":
        return True
    if stripped.lower() == "false":
        return False
    # JSON array: [1, 2, 3]
    if stripped.startswith("["):
        return json.loads(stripped)
    # Try int / float
    try:
        return int(stripped)
    except ValueError:
        pass
    try:
        return float(stripped)
    except ValueError:
        pass
    return stripped


def apply_overrides(config: dict, overrides: list) -> dict:
    """Apply dot-separated key=value overrides to a nested config dict.

    Examples:
        --set pruning.strategy=entropy
        --set pruning.prune_layers=[2,5,10]
        --set pruning.prune_ratio=[0.3,0.4,0.5]
        --set pruning.entropy.dynamic_ratio=true
        --set inference.max_new_tokens=256
    """
    import copy
    config = copy.deepcopy(config)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override format (expected key=value): {item!r}")
        key_path, raw_value = item.split("=", 1)
        keys = key_path.strip().split(".")
        value = _parse_value(raw_value)

        d = config
        for k in keys[:-1]:
            if k not in d or not isinstance(d[k], dict):
                d[k] = {}
            d = d[k]
        d[keys[-1]] = value
    return config


def get_primary_visible_cuda_device() -> str:
    """Use only the first GPU from CUDA_VISIBLE_DEVICES inside this process."""
    if not torch.cuda.is_available():
        return "cpu"

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        first_visible = visible.split(",")[0].strip()
        print(
            f"[device] CUDA_VISIBLE_DEVICES={visible}; "
            f"binding this run to the first visible GPU only (process-local cuda:0, physical GPU {first_visible})."
        )
    else:
        print("[device] CUDA_VISIBLE_DEVICES is not set; binding this run to process-local cuda:0.")

    return "cuda:0"


def _encode_run_name_value(value):
    if isinstance(value, float):
        text = format(value, "g")
    else:
        text = str(value)
    return text.replace(".", "p")


def _encode_run_name_list(values) -> str:
    if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple)):
        values = [values]
    return "-".join(_encode_run_name_value(value) for value in values)


def build_run_name(dataset_name: str, run_tag: str, prune_layers, prune_ratio, timestamp: str) -> str:
    layers_part = f"l{_encode_run_name_list(prune_layers)}"
    ratio_part = f"r{_encode_run_name_list(prune_ratio)}"
    return f"{dataset_name}_{run_tag}_{layers_part}_{ratio_part}__{timestamp}"


def sanitize_run_tag_suffix(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")
    return sanitized


def _normalize_configured_prune_layers(value) -> list[int]:
    if isinstance(value, int):
        return [int(value)]
    if isinstance(value, list):
        return [int(item) for item in value]
    raise ValueError(f"prune_layers must be an int or list of ints, got {type(value).__name__}")


def normalize_configured_v_token_num(value) -> int | str:
    """Normalize a fixed LLaVA-1.5 token count or dynamic ``auto`` mode."""
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "auto":
            return "auto"
        try:
            value = int(normalized)
        except ValueError as exc:
            raise ValueError(
                f"pruning.v_token_num must be a positive integer or 'auto', got {value!r}"
            ) from exc
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            f"pruning.v_token_num must be a positive integer or 'auto', got {value!r}"
        )
    return int(value)


def _normalize_single_tail_start_layer(value) -> int:
    layers = _normalize_configured_prune_layers(value)
    if len(layers) != 1:
        raise ValueError(
            f"tail_masking_attn_score requires exactly one configured start layer, got {layers}"
        )
    return int(layers[0])


def _normalize_single_tail_ratio(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError(
                "tail_masking_attn_score requires prune_ratio to be a scalar or a single-element list"
            )
        return float(value[0])
    raise ValueError(
        f"tail_masking_attn_score requires prune_ratio to be numeric, got {type(value).__name__}"
    )


def load_model_config_metadata(model_path: str) -> dict[str, Any]:
    """Read lightweight model metadata from config.json without loading weights."""
    config_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Model config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        model_config = json.load(f)

    metadata = {
        key: model_config.get(key)
        for key in MODEL_CONFIG_METADATA_KEYS
        if key in model_config
    }
    metadata["config_path"] = config_path
    return metadata


def _num_hidden_layers_from_metadata(model_config_metadata: dict[str, Any]) -> int:
    num_hidden_layers = model_config_metadata.get("num_hidden_layers")
    if not isinstance(num_hidden_layers, int):
        raise ValueError(
            "model config metadata is missing integer num_hidden_layers, "
            f"got {num_hidden_layers!r} from {model_config_metadata.get('config_path')}"
        )
    return num_hidden_layers


def _load_num_hidden_layers(model_path: str) -> int:
    return _num_hidden_layers_from_metadata(load_model_config_metadata(model_path))


def validate_fixed_prune_layers(
    prune_layers: list[int],
    num_hidden_layers: int,
    *,
    field_name: str = "prune_layers",
) -> None:
    if num_hidden_layers <= 0:
        raise ValueError(f"num_hidden_layers must be > 0, got {num_hidden_layers}")
    invalid_layers = [layer for layer in prune_layers if layer < 0 or layer >= num_hidden_layers]
    if invalid_layers:
        raise ValueError(
            f"{field_name} must be within [0, {num_hidden_layers - 1}] for this model, "
            f"got invalid layers {invalid_layers} from {prune_layers}"
        )


def derive_effective_prune_config(
    prune_cfg: dict,
    model_path: str,
    model_config_metadata: Optional[dict[str, Any]] = None,
) -> tuple[dict, list[int], Optional[int]]:
    import copy

    configured_prune_layers = _normalize_configured_prune_layers(prune_cfg["prune_layers"])
    effective_prune_cfg = copy.deepcopy(prune_cfg)
    strategy_name = prune_cfg["strategy"]
    model_config_metadata = model_config_metadata or load_model_config_metadata(model_path)
    num_hidden_layers = _num_hidden_layers_from_metadata(model_config_metadata)

    if prune_cfg.get("layer_selection", "fixed") == "fixed":
        validate_fixed_prune_layers(configured_prune_layers, num_hidden_layers)

    if strategy_name != "tail_masking_attn_score":
        effective_prune_cfg["prune_layers"] = configured_prune_layers
        return effective_prune_cfg, configured_prune_layers, None

    if prune_cfg.get("layer_selection", "fixed") != "fixed":
        raise ValueError("tail_masking_attn_score currently only supports layer_selection=fixed")

    start_layer = _normalize_single_tail_start_layer(prune_cfg["prune_layers"])
    shared_ratio = _normalize_single_tail_ratio(prune_cfg.get("prune_ratio", 0.5))
    if start_layer < 0 or start_layer >= num_hidden_layers:
        raise ValueError(
            f"tail_masking_attn_score start layer must be within [0, {num_hidden_layers - 1}], got {start_layer}"
        )

    effective_prune_layers = list(range(start_layer, num_hidden_layers))
    effective_prune_cfg["prune_layers"] = effective_prune_layers
    effective_prune_cfg["prune_ratio"] = shared_ratio
    return effective_prune_cfg, effective_prune_layers, start_layer


def _actual_decoder_layer_count(model) -> Optional[int]:
    backbone = getattr(model, "model", None)
    layers = getattr(backbone, "layers", None)
    if layers is None:
        return None
    try:
        return len(layers)
    except TypeError:
        return None


def validate_loaded_model_runtime(
    model,
    model_config_metadata: dict[str, Any],
    configured_v_token_num: int | str,
) -> dict[str, Any]:
    """Validate loaded model shape against config metadata and pruning settings."""
    runtime_metadata: dict[str, Any] = {}

    actual_num_layers = _actual_decoder_layer_count(model)
    runtime_metadata["actual_num_layers"] = actual_num_layers
    expected_num_layers = model_config_metadata.get("num_hidden_layers")
    if (
        isinstance(expected_num_layers, int)
        and actual_num_layers is not None
        and actual_num_layers != expected_num_layers
    ):
        raise RuntimeError(
            "Loaded model layer count does not match config.json: "
            f"actual={actual_num_layers}, config={expected_num_layers}"
        )

    vision_token_count = None
    vision_patches_per_side = None
    if hasattr(model, "get_vision_tower"):
        vision_tower = model.get_vision_tower()
        if vision_tower is not None:
            vision_token_count = getattr(vision_tower, "num_patches", None)
            vision_patches_per_side = getattr(vision_tower, "num_patches_per_side", None)
    runtime_metadata["vision_token_count"] = vision_token_count
    runtime_metadata["vision_patches_per_side"] = vision_patches_per_side
    runtime_metadata["visual_token_count_mode"] = configured_v_token_num
    runtime_metadata["image_aspect_ratio"] = model_config_metadata.get("image_aspect_ratio")
    runtime_metadata["mm_patch_merge_type"] = model_config_metadata.get("mm_patch_merge_type")

    uses_dynamic_anyres = model_config_metadata.get("image_aspect_ratio") == "anyres"
    if uses_dynamic_anyres and configured_v_token_num != "auto":
        raise RuntimeError(
            "LLaVA-NeXT anyres inputs require pruning.v_token_num=auto; "
            f"got fixed value {configured_v_token_num!r}"
        )
    if (
        configured_v_token_num != "auto"
        and isinstance(vision_token_count, int)
        and int(configured_v_token_num) != vision_token_count
    ):
        raise RuntimeError(
            "Configured pruning.v_token_num does not match the loaded vision tower: "
            f"configured={configured_v_token_num}, vision_tower={vision_token_count}"
        )

    meta_parameters = [name for name, parameter in model.named_parameters() if parameter.is_meta]
    if meta_parameters:
        raise RuntimeError(
            "Loaded model still contains meta parameters; first entries: "
            f"{meta_parameters[:5]}"
        )
    runtime_metadata["meta_parameter_count"] = 0

    return runtime_metadata


# ---------------------------------------------------------------------------
# Main inference loop with pruning
# ---------------------------------------------------------------------------
def run_prune_inference(
    config: dict,
    dataset_name: str,
    max_samples: int = None,
    run_mode: str = "prune",
):
    """
    Run inference with visual token pruning on a specific dataset.
    """

    def resolve(p):
        if p is None:
            return None
        if os.path.isabs(p):
            return p
        return os.path.join(LLAVA_ROOT, p)

    model_cfg = config["model"]
    infer_cfg = config["inference"]
    prune_cfg = config["pruning"]
    ds_cfg = config["datasets"][dataset_name]
    output_cfg = config["output"]

    if run_mode not in {"prune", "baseline"}:
        raise ValueError(f"Unknown run_mode: {run_mode}")

    model_path = resolve(model_cfg["path"])
    model_name = model_cfg["name"]
    model_config_metadata = load_model_config_metadata(model_path)
    question_file = resolve(ds_cfg["question_file"])
    image_folder = resolve(ds_cfg["image_folder"])
    max_new_tokens = int(ds_cfg.get("max_new_tokens", infer_cfg["max_new_tokens"]))
    strategy_name = prune_cfg["strategy"]
    configured_v_token_num = normalize_configured_v_token_num(prune_cfg.get("v_token_num", 576))
    configured_prune_layers = _normalize_configured_prune_layers(prune_cfg["prune_layers"])
    effective_prune_cfg, effective_prune_layers, tail_start_layer = derive_effective_prune_config(
        prune_cfg,
        model_path,
        model_config_metadata=model_config_metadata,
    )
    strategy_extra = effective_prune_cfg.get(strategy_name, {})
    strategy_config = {
        **effective_prune_cfg,
        "seed": infer_cfg.get("seed", 42),
        **strategy_extra,
        "_model_name": model_name,
        "_model_config_fingerprint": model_config_fingerprint(model_config_metadata),
    }
    strategy = get_strategy(strategy_name, strategy_config)

    # --- output paths: per-run directory ---
    # Use readable settings in the run name and keep a microsecond timestamp
    # suffix so concurrent launches remain unique.
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_tag = "baseline" if run_mode == "baseline" else strategy_name
    run_tag_suffix = sanitize_run_tag_suffix(output_cfg.get("run_tag_suffix"))
    if run_tag_suffix:
        run_tag = f"{run_tag}_{run_tag_suffix}"
    base_dir = resolve(output_cfg.get("base_dir", "entropy_exp/outputs"))

    base_run_name = build_run_name(
        dataset_name=dataset_name,
        run_tag=run_tag,
        prune_layers=prune_cfg["prune_layers"],
        prune_ratio=prune_cfg["prune_ratio"],
        timestamp=timestamp,
    )
    run_name = base_run_name
    strategy_branch = "baseline" if run_mode == "baseline" else strategy_name
    run_dir = os.fspath(build_run_dir(base_dir, strategy_branch, dataset_name, run_name))
    suffix = 1
    while True:
        try:
            os.makedirs(run_dir)
            break
        except FileExistsError:
            run_name = f"{base_run_name}_{suffix:02d}"
            run_dir = os.fspath(build_run_dir(base_dir, strategy_branch, dataset_name, run_name))
            suffix += 1

    answers_path = os.path.join(run_dir, "answers.jsonl")
    stats_path = os.path.join(run_dir, "stats.jsonl")
    capture_cfg = config.get("capture", {})
    save_attention = capture_cfg.get("save_attention", False)
    save_importance = capture_cfg.get("save_importance_scores", False)
    save_indices = capture_cfg.get("save_keep_indices", False)
    captures_h5_path = os.path.join(run_dir, "captures.h5") if save_attention else None
    rep_diag_cfg = config.get("diagnostics", {}).get("rep_bottleneck", {})
    rep_diag_enabled = bool(rep_diag_cfg.get("enabled", False))
    rep_diag_question_ids = _normalize_question_id_set(rep_diag_cfg.get("question_ids", []))
    rep_diag_sample_indices = _normalize_int_set(rep_diag_cfg.get("sample_indices", []))
    if rep_diag_enabled and not rep_diag_question_ids and not rep_diag_sample_indices:
        raise ValueError(
            "diagnostics.rep_bottleneck.enabled=true requires a non-empty "
            "diagnostics.rep_bottleneck.question_ids or sample_indices list."
        )
    rep_diag_layer = int(rep_diag_cfg.get("capture_layer", 2))
    rep_diag_output_name = str(rep_diag_cfg.get("output_name", "rep_bottleneck_hidden.h5"))
    rep_diag_h5_path = os.path.join(run_dir, rep_diag_output_name) if rep_diag_enabled else None

    # Parse capture_layers: "all" -> None (means all), list -> set
    capture_layers_cfg = capture_cfg.get("capture_layers", "all")
    if capture_layers_cfg == "all" or capture_layers_cfg is None:
        capture_layers_set = None  # None means all layers
    else:
        capture_layers_set = set(capture_layers_cfg)

    # --- save config snapshot (after --set overrides) ---
    config_snapshot = {
        **config,
        "model_config_metadata": model_config_metadata,
        "pruning": {
            **config["pruning"],
            "prune_layers": (
                configured_prune_layers
                if strategy_name == "tail_masking_attn_score"
                else config["pruning"]["prune_layers"]
            ),
            "effective_prune_layers": effective_prune_layers,
            "tail_start_layer": tail_start_layer,
        },
        "_run_meta": {
            "run_mode": run_mode,
            "dataset": dataset_name,
            "strategy": strategy_branch,
            "max_samples": max_samples,
            "timestamp": timestamp,
            "run_name": run_name,
            "run_rel_dir": build_run_rel_dir(base_dir, strategy_branch, dataset_name, run_name),
            "run_dir": run_dir,
        },
    }
    config_snapshot_path = os.path.join(run_dir, "config.yaml")
    with open(config_snapshot_path, 'w') as f:
        yaml.dump(config_snapshot, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    # --- seed ---
    seed = infer_cfg.get("seed", 42)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # --- load model ---
    print(f"Loading model from {model_path} …")
    disable_torch_init()
    load_kwargs = {}
    attn_impl = model_cfg.get("attn_implementation", "eager")
    if attn_impl == "eager":
        load_kwargs["attn_implementation"] = "eager"

    target_device = get_primary_visible_cuda_device()
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path,
        None,
        model_name,
        device_map=target_device,
        device=target_device,
        disable_mmap=bool(model_cfg.get("disable_mmap", True)),
        **load_kwargs,
    )
    if enable_sparse_position_ids_compat(model):
        family = getattr(model, "_sparse_position_ids_compat_family", "unknown")
        print(f"[position-ids] Enabled sparse position-id compatibility for {family} attention.")
    model.eval()
    model_runtime_metadata = validate_loaded_model_runtime(
        model,
        model_config_metadata=model_config_metadata,
        configured_v_token_num=configured_v_token_num,
    )
    config_snapshot["_run_meta"]["model_runtime_metadata"] = model_runtime_metadata
    with open(config_snapshot_path, 'w') as f:
        yaml.dump(config_snapshot, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(
        f"Model loaded.  attn={attn_impl}, device={target_device}, "
        f"hidden_size={model_config_metadata.get('hidden_size')}, "
        f"layers={model_runtime_metadata.get('actual_num_layers')}"
    )

    # --- build pruner ---
    pruner = VisualTokenPruner(model, strategy, effective_prune_cfg)
    effective_ratio_map = {str(k): float(v) for k, v in pruner.layer_ratio_map.items()}
    print(f"Pruner: strategy={strategy_name}, prune_layers={pruner.prune_layers}, "
          f"base_ratio={prune_cfg.get('prune_ratio', 0.5)}")

    # --- load questions ---
    questions = load_dataset_samples(dataset_name, question_file)
    rep_diag_missing_qids: list[str] = []
    rep_diag_missing_sample_indices: list[int] = []
    if rep_diag_enabled:
        indexed_questions = []
        for original_sample_idx, line in enumerate(questions):
            indexed_line = dict(line)
            indexed_line["_rep_bottleneck_original_sample_idx"] = original_sample_idx
            indexed_questions.append(indexed_line)
        questions = indexed_questions

        available_qids = {str(line["question_id"]) for line in questions}
        available_sample_indices = {int(line["_rep_bottleneck_original_sample_idx"]) for line in questions}
        rep_diag_missing_qids = sorted(rep_diag_question_ids - available_qids)
        rep_diag_missing_sample_indices = sorted(rep_diag_sample_indices - available_sample_indices)
        if rep_diag_missing_qids:
            print(
                "[rep-bottleneck] Warning: requested question_ids not found in "
                f"{dataset_name}: {rep_diag_missing_qids}"
            )
        if rep_diag_missing_sample_indices:
            print(
                "[rep-bottleneck] Warning: requested sample_indices not found in "
                f"{dataset_name}: {rep_diag_missing_sample_indices}"
            )
        questions = [
            line
            for line in questions
            if (
                (rep_diag_question_ids and str(line["question_id"]) in rep_diag_question_ids)
                or (
                    rep_diag_sample_indices
                    and int(line["_rep_bottleneck_original_sample_idx"]) in rep_diag_sample_indices
                )
            )
        ]
        if not questions:
            raise ValueError(
                "diagnostics.rep_bottleneck matched zero samples for "
                f"dataset={dataset_name}."
            )

    effective_max = max_samples or prune_cfg.get("max_samples")
    if effective_max is not None:
        questions = questions[:effective_max]
    print(f"Using {len(questions)} samples")
    if not questions:
        raise ValueError(f"No samples found for dataset={dataset_name}.")

    # --- dataset & loader ---
    dataset = VQADataset(
        questions, image_folder, tokenizer, image_processor,
        model.config, infer_cfg["conv_mode"],
    )
    num_workers = int(os.environ.get("DATALOADER_NUM_WORKERS", "0"))
    data_loader = DataLoader(dataset, batch_size=1, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)

    eos_token_id = tokenizer.eos_token_id or 2

    total_time = 0.0
    all_stats = []
    ans_file = open(answers_path, 'w')
    h5_file = h5py.File(captures_h5_path, "w") if captures_h5_path else None
    rep_diag_h5_file = h5py.File(rep_diag_h5_path, "w") if rep_diag_h5_path else None
    if rep_diag_h5_file is not None:
        rep_diag_h5_file.attrs["dataset"] = dataset_name
        rep_diag_h5_file.attrs["capture_layer"] = rep_diag_layer
        rep_diag_h5_file.attrs["requested_question_ids"] = json.dumps(sorted(rep_diag_question_ids))
        rep_diag_h5_file.attrs["requested_sample_indices"] = json.dumps(sorted(rep_diag_sample_indices))
        rep_diag_h5_file.attrs["missing_question_ids"] = json.dumps(rep_diag_missing_qids)
        rep_diag_h5_file.attrs["missing_sample_indices"] = json.dumps(rep_diag_missing_sample_indices)
        rep_diag_h5_file.attrs["hidden_precision"] = str(rep_diag_cfg.get("precision", "fp16"))
    progress_tag = run_tag

    for sample_idx, ((input_ids, image_tensor, image_sizes), line) in enumerate(
        tqdm(zip(data_loader, questions), total=len(questions), desc=f"[{dataset_name}|{progress_tag}]")
    ):
        question_id = line["question_id"]
        image_file = line.get("image_path") or (f"inline_base64:{question_id}" if line.get("image_base64") else None)
        has_image = bool(line.get("has_image", False))
        retry_count = 0
        while True:
            input_ids_cuda = None
            image_tensor_cuda = None
            _input_ids = None
            position_ids = None
            attention_mask = None
            inputs_embeds = None
            generated_ids = None
            output_ids = None
            visual_layout = None
            v_token_start = None
            sample_v_token_num = 0
            text_token_start = None

            try:
                input_ids_cuda = input_ids.to(device=target_device, non_blocking=True)
                if has_image:
                    # --- prepare multimodal embeddings ---
                    image_tensor_cuda = image_tensor.to(dtype=torch.float16, device=target_device, non_blocking=True)
                    if image_tensor_cuda.ndim == 5:
                        num_image_crops = int(image_tensor_cuda.shape[1])
                    elif image_tensor_cuda.ndim == 4:
                        num_image_crops = 1
                    else:
                        raise RuntimeError(
                            f"Unsupported image tensor shape: {tuple(image_tensor_cuda.shape)}"
                        )

                    vision_tower = model.get_vision_tower()
                    visual_layout = get_visual_token_layout(
                        image_sizes[0],
                        model.config,
                        patches_per_side=vision_tower.num_patches_per_side,
                        vision_image_size=vision_tower.config.image_size,
                        num_image_crops=num_image_crops,
                    )
                    sample_v_token_num = int(visual_layout["total_token_count"])
                    if (
                        configured_v_token_num != "auto"
                        and int(configured_v_token_num) != sample_v_token_num
                    ):
                        raise RuntimeError(
                            "Configured visual token count does not match this sample: "
                            f"configured={configured_v_token_num}, actual={sample_v_token_num}, "
                            f"layout={visual_layout}"
                        )
                    v_token_start, _, text_token_start = locate_image_tokens(
                        input_ids,
                        IMAGE_TOKEN_INDEX,
                        v_token_num=sample_v_token_num,
                    )

                    (
                        _input_ids,
                        position_ids,
                        attention_mask,
                        _,
                        inputs_embeds,
                        _,
                    ) = model.prepare_inputs_labels_for_multimodal(
                        input_ids_cuda, None, None, None, None,
                        image_tensor_cuda, image_sizes=list(image_sizes),
                    )
                    untruncated_seq_len = int(input_ids.shape[1]) - 1 + sample_v_token_num
                    tokenizer_max_length = getattr(model.config, "tokenizer_model_max_length", None)
                    expected_seq_len = (
                        min(untruncated_seq_len, int(tokenizer_max_length))
                        if tokenizer_max_length is not None
                        else untruncated_seq_len
                    )
                    actual_seq_len = int(inputs_embeds.shape[1])
                    if actual_seq_len != expected_seq_len:
                        raise RuntimeError(
                            f"Expanded sequence length mismatch for question_id={question_id}: "
                            f"expected {expected_seq_len}, got {actual_seq_len}; "
                            f"layout={visual_layout}."
                        )
                    if text_token_start > actual_seq_len:
                        raise RuntimeError(
                            "Tokenizer truncation reached the visual block, which is unsupported: "
                            f"visual_end={text_token_start}, sequence_length={actual_seq_len}"
                        )

                    actual_text_length = actual_seq_len - text_token_start
                    text_token_ids = input_ids[
                        0,
                        v_token_start + 1:v_token_start + 1 + actual_text_length,
                    ].clone()
                    text_special_token_mask = build_text_special_token_mask(tokenizer, text_token_ids)

                    # --- generate with pruning ---
                    t0 = time.time()
                    generated_ids, prune_info = pruner.pruned_generate(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        v_token_start=v_token_start,
                        v_token_num=sample_v_token_num,
                        text_token_start=text_token_start,
                        text_token_ids=text_token_ids,
                        text_special_token_mask=text_special_token_mask,
                        max_new_tokens=max_new_tokens,
                        eos_token_id=eos_token_id,
                        save_tv_attn=save_attention,
                        capture_layers=capture_layers_set,
                        capture_visual_hidden_layers={rep_diag_layer} if rep_diag_enabled else None,
                        visual_layout=visual_layout,
                    )
                    t1 = time.time()
                    total_time += (t1 - t0)

                    answer_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
                else:
                    t0 = time.time()
                    output_ids = model.generate(
                        inputs=input_ids_cuda,
                        do_sample=True if infer_cfg["temperature"] > 0 else False,
                        temperature=infer_cfg["temperature"],
                        top_p=infer_cfg["top_p"],
                        num_beams=infer_cfg["num_beams"],
                        max_new_tokens=max_new_tokens,
                        use_cache=True,
                    )
                    t1 = time.time()
                    total_time += (t1 - t0)

                    prompt_length = int(input_ids_cuda.shape[1])
                    generated_ids, stripped_prompt_prefix = strip_prompt_prefix_if_present(output_ids, input_ids_cuda)
                    answer_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                    prune_info = {
                        "original_seq_len": prompt_length,
                        "final_seq_len": prompt_length,
                        "output_seq_len": int(output_ids.shape[1]),
                        "num_generated_tokens": int(generated_ids.shape[1]),
                        "stripped_prompt_prefix": bool(stripped_prompt_prefix),
                        "prefill_time": None,
                        "decode_time": None,
                        "total_time": t1 - t0,
                        "prune_stage": "text_only_no_prune",
                        "layers": {},
                    }
                break
            except torch.cuda.OutOfMemoryError:
                retry_count += 1
                backoff_seconds = random.randint(1, 10)
                print(
                    f"[oom-retry] CUDA OOM on sample_idx={sample_idx}, question_id={question_id}, "
                    f"retry={retry_count}. Backing off for {backoff_seconds}s before retry."
                )
                if output_ids is not None:
                    del output_ids
                if generated_ids is not None:
                    del generated_ids
                if inputs_embeds is not None:
                    del inputs_embeds
                if attention_mask is not None:
                    del attention_mask
                if position_ids is not None:
                    del position_ids
                if _input_ids is not None:
                    del _input_ids
                if image_tensor_cuda is not None:
                    del image_tensor_cuda
                if input_ids_cuda is not None:
                    del input_ids_cuda
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                time.sleep(backoff_seconds)

        # --- write answer ---
        ans_file.write(json.dumps({
            "question_id": question_id,
            "prompt": line["text"],
            "text": answer_text,
            "model_id": model_name,
            "metadata": {},
        }) + "\n")

        # --- record pruning stats ---
        sample_stats = {
            "sample_idx": sample_idx,
            "run_mode": run_mode,
            "strategy_requested": strategy_name,
            "prune_stage": prune_info.get("prune_stage", strategy.prune_stage()),
            "effective_prune_ratio_map": effective_ratio_map,
            "configured_prune_layers": configured_prune_layers,
            "effective_prune_layers": effective_prune_layers,
            "prune_layers": pruner.prune_layers,
            "question_id": str(question_id),
            "image_file": image_file,
            "answer": answer_text,
            "original_seq_len": prune_info["original_seq_len"],
            "final_seq_len": prune_info["final_seq_len"],
            "num_generated_tokens": prune_info["num_generated_tokens"],
            "prefill_time": prune_info["prefill_time"],
            "decode_time": prune_info["decode_time"],
            "total_time": prune_info["total_time"],
        }
        if visual_layout is not None:
            sample_stats.update(
                {
                    "initial_visual_token_count": int(sample_v_token_num),
                    "visual_index_semantics": visual_layout["visual_index_semantics"],
                    "visual_token_layout": visual_layout,
                }
            )

        # Per-layer pruning details
        for layer_idx, linfo in prune_info.get("layers", {}).items():
            # Skip capture-only layers (no pruning stats)
            if "prune_ratio" not in linfo:
                continue
            append_layer_stats_fields(
                sample_stats,
                layer_idx,
                linfo,
                save_importance=save_importance,
                save_indices=save_indices,
                visual_layout=visual_layout,
            )

        # Write attention captures to HDF5 (same format as Phase 1)
        if h5_file is not None:
            sample_id = f"{sample_idx:06d}_{str(question_id).replace('/', '__')}"
            grp = h5_file.create_group(sample_id)
            grp.attrs["question_id"] = str(question_id)
            grp.attrs["image_file"] = str(image_file)
            grp.attrs["v_token_start"] = -1 if v_token_start is None else int(v_token_start)
            grp.attrs["v_token_num"] = int(sample_v_token_num)
            grp.attrs["text_token_start"] = -1 if text_token_start is None else int(text_token_start)
            if visual_layout is not None:
                grp.attrs["visual_token_layout"] = json.dumps(visual_layout, sort_keys=True)

            use_fp16 = capture_cfg.get("precision", "fp16") == "fp16"
            compression = capture_cfg.get("compression", "gzip")
            compression_opts = capture_cfg.get("compression_opts", 4)

            for layer_idx, linfo in prune_info.get("layers", {}).items():
                if "tv_attn" not in linfo:
                    continue
                tv = linfo["tv_attn"]  # [H, L_t, L_v] CPU tensor
                if use_fp16:
                    tv = tv.half()
                layer_grp = grp.create_group(f"layer_{layer_idx}")
                layer_grp.create_dataset(
                    "tv_attn", data=tv.numpy(),
                    compression=compression, compression_opts=compression_opts,
                )
                # Also save prune scores for Phase 1 compatibility
                if "importance_scores" in linfo:
                    scores = linfo["importance_scores"]
                    if hasattr(scores, 'cpu'):
                        scores = scores.cpu().numpy()
                    layer_grp.create_dataset(
                        "prune_scores", data=np.asarray(scores, dtype=np.float32),
                    )
            if sample_idx % 50 == 0:
                h5_file.flush()

        # Write L2 pre-prune visual hidden states for representation bottleneck figures.
        if rep_diag_h5_file is not None:
            sample_id = f"{sample_idx:06d}_{str(question_id).replace('/', '__')}"
            grp = rep_diag_h5_file.create_group(sample_id)
            grp.attrs["question_id"] = str(question_id)
            grp.attrs["image_file"] = str(image_file)
            grp.attrs["has_image"] = bool(has_image)
            grp.attrs["capture_layer"] = rep_diag_layer
            grp.attrs["original_sample_idx"] = int(line.get("_rep_bottleneck_original_sample_idx", sample_idx))
            grp.attrs["prompt"] = str(line.get("text", ""))
            grp.attrs["visual_index_semantics"] = (
                "none" if visual_layout is None else visual_layout["visual_index_semantics"]
            )
            if visual_layout is not None:
                grp.attrs["visual_token_layout"] = json.dumps(visual_layout, sort_keys=True)

            linfo = prune_info.get("layers", {}).get(rep_diag_layer, {})
            visual_hidden = linfo.get("visual_hidden")
            if visual_hidden is None:
                grp.attrs["missing_visual_hidden"] = True
            else:
                grp.attrs["missing_visual_hidden"] = False
                grp.attrs["v_token_start"] = int(linfo.get("visual_hidden_v_token_start", -1))
                grp.attrs["v_token_num"] = int(linfo.get("visual_hidden_v_token_num", visual_hidden.shape[0]))
                grp.attrs["hidden_shape"] = json.dumps(list(visual_hidden.shape))
                layer_grp = grp.create_group(f"layer_{rep_diag_layer}")

                use_fp16 = rep_diag_cfg.get("precision", "fp16") == "fp16"
                hidden_to_save = visual_hidden.half() if use_fp16 else visual_hidden.float()
                layer_grp.create_dataset(
                    "visual_hidden",
                    data=hidden_to_save.numpy(),
                    **_h5_dataset_kwargs(rep_diag_cfg),
                )
            if sample_idx % 10 == 0:
                rep_diag_h5_file.flush()

        all_stats.append(sample_stats)

    ans_file.close()
    if h5_file is not None:
        h5_file.close()
    if rep_diag_h5_file is not None:
        rep_diag_h5_file.close()

    # --- write stats ---
    with open(stats_path, 'w') as f:
        for s in all_stats:
            f.write(json.dumps(s) + "\n")

    avg_time = total_time / len(questions) if questions else 0

    # --- summary ---
    prune_layer = pruner.prune_layers[0] if pruner.prune_layers else -1
    avg_pruned = 0
    avg_visual_before = 0
    if all_stats:
        key = f"layer_{prune_layer}_pruned"
        vals = [s.get(key, 0) for s in all_stats]
        avg_pruned = sum(vals) / len(vals)
        before_key = f"layer_{prune_layer}_before"
        before_vals = [s[before_key] for s in all_stats if before_key in s]
        avg_visual_before = sum(before_vals) / len(before_vals) if before_vals else 0

    print(f"\nDone! {len(questions)} samples processed.")
    print(f"  Run mode:          {run_mode}")
    print(f"  Strategy:          {strategy_name}")
    print(f"  Prune stage:       {prune_info.get('prune_stage', strategy.prune_stage()) if questions else strategy.prune_stage()}")
    print(f"  Prune layer(s):    {pruner.prune_layers}")
    print(f"  Avg tokens pruned: {avg_pruned:.1f} / {avg_visual_before:.1f}")
    print(f"  Avg time/sample:   {avg_time:.3f}s")
    print(f"  Run dir:  {run_dir}")
    print(f"  Answers:  {answers_path}")
    print(f"  Stats:    {stats_path}")
    if captures_h5_path:
        print(f"  Captures: {captures_h5_path}")
    if rep_diag_h5_path:
        print(f"  Rep bottleneck hidden: {rep_diag_h5_path}")

    return run_dir, answers_path, stats_path


# ---------------------------------------------------------------------------
# Baseline (no pruning) for fair comparison
# ---------------------------------------------------------------------------
def run_baseline_inference(config: dict, dataset_name: str, max_samples: int = None):
    """
    Run inference WITHOUT pruning using the same custom decode loop,
    so timing comparison is fair (same overhead).
    """
    # temporarily disable pruning
    config = {**config, "pruning": {**config["pruning"], "prune_ratio": 0.0}}
    return run_prune_inference(config, dataset_name, max_samples, run_mode="baseline")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Pruning inference pipeline")
    parser.add_argument("--config", type=str, default="entropy_exp/configs/prune.yaml")
    parser.add_argument("--dataset", type=str, required=True, choices=list(SUPPORTED_DATASETS))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--baseline", action="store_true",
                        help="Run baseline (no pruning) for comparison")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY=VALUE",
                        help="Override config values, e.g. --set pruning.strategy=entropy "
                             "--set pruning.prune_layers=[2,5] --set pruning.prune_ratio=[0.3,0.5]")
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(LLAVA_ROOT, config_path)

    config = load_config(config_path)

    if args.overrides:
        config = apply_overrides(config, args.overrides)
        print(f"Config overrides applied: {args.overrides}")

    if args.baseline:
        run_baseline_inference(config, args.dataset, args.max_samples)
    else:
        run_prune_inference(config, args.dataset, args.max_samples)


if __name__ == "__main__":
    main()
