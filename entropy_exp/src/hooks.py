"""
Attention Capture Hook for LLaVA entropy experiments.

Captures text→vision attention sub-matrices from all (or selected) decoder layers
during inference. Supports two modes:
  - "capture": pure observation, no pruning
  - "prune":   (future) compute prune scores and return masks

Usage:
    hook = AttentionCaptureHook(model, config)
    hook.enable()
    outputs = model.generate(...)   # with output_attentions=True
    hook.save_sample(sample_id, outputs.attentions, metadata)
    hook.disable()
"""

import torch
import h5py
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union


class AttentionCaptureHook:
    """Manages capture of text→vision attention sub-matrices and writes to HDF5."""

    def __init__(self, h5_path: str, capture_cfg: dict):
        """
        Args:
            h5_path: path to the output HDF5 file
            capture_cfg: dict with keys: mode, layers, head_reduction, precision,
                         compression, compression_opts, v_token_num
        """
        self.h5_path = h5_path
        self.cfg = capture_cfg
        self.h5_file: Optional[h5py.File] = None

        # Parse layer config
        layers_cfg = self.cfg.get("layers", "all")
        if layers_cfg == "all":
            self.target_layers = None  # capture all
        else:
            self.target_layers = set(layers_cfg)

        self.head_reduction = self.cfg.get("head_reduction", "none")
        self.use_fp16 = self.cfg.get("precision", "fp16") == "fp16"
        self.compression = self.cfg.get("compression", "gzip")
        self.compression_opts = self.cfg.get("compression_opts", 4)

    def open(self):
        """Open HDF5 file for writing."""
        self.h5_file = h5py.File(self.h5_path, "w")

    def close(self):
        """Close HDF5 file."""
        if self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def save_config_snapshot(self, config: dict):
        """Save experiment config as HDF5 attributes on root group."""
        if self.h5_file is None:
            return
        for k, v in config.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    self.h5_file.attrs[f"{k}.{kk}"] = str(vv)
            else:
                self.h5_file.attrs[k] = str(v)

    def extract_tv_submatrix(
        self,
        attn_weights: torch.Tensor,
        v_token_start: int,
        v_token_num: int,
        text_token_start: int,
    ) -> np.ndarray:
        """
        Extract text→vision attention sub-matrix from full attention weights.

        Args:
            attn_weights: [B, H, L, L] attention weights for one layer (prefill step)
            v_token_start: index where vision tokens start
            v_token_num: number of vision tokens (576)
            text_token_start: index where text tokens start (= v_token_start + v_token_num)

        Returns:
            np.ndarray of shape [H, L_t, L_v] or [L_t, L_v] if head_reduction="mean"
        """
        # attn_weights: [B, H, L, L], B=1
        # text tokens query (rows) → vision tokens key (cols)
        v_end = v_token_start + v_token_num
        # text query positions: from text_token_start to end of sequence
        # vision key positions: from v_token_start to v_token_start + v_token_num
        tv = attn_weights[0, :, text_token_start:, v_token_start:v_end]  # [H, L_t, L_v]

        if self.head_reduction == "mean":
            tv = tv.mean(dim=0)  # [L_t, L_v]

        if self.use_fp16:
            tv = tv.half()

        return tv.cpu().numpy()

    def compute_prune_scores(
        self,
        tv_attn: np.ndarray,
    ) -> np.ndarray:
        """
        Compute per-visual-token prune scores from text→vision sub-matrix.
        Same logic as SparseVLM: head_mean → text_mean → per visual token score.

        Args:
            tv_attn: [H, L_t, L_v] or [L_t, L_v]

        Returns:
            np.ndarray of shape [L_v] — importance score per visual token
        """
        arr = tv_attn
        if arr.ndim == 3:
            arr = arr.mean(axis=0)  # [L_t, L_v]
        scores = arr.mean(axis=0)  # [L_v]
        return scores

    def save_sample(
        self,
        sample_id: str,
        attentions: Tuple[Tuple[torch.Tensor, ...], ...],
        v_token_start: int,
        v_token_num: int,
        text_token_start: int,
        metadata: Dict[str, Any],
    ):
        """
        Extract and save attention data for one sample.

        Args:
            sample_id: unique identifier for this sample
            attentions: output from model.generate(output_attentions=True)
                        For generate, attentions is a tuple over generation steps.
                        Each step is a tuple over layers.
                        We only care about the FIRST step (prefill) which contains
                        the full sequence attention.
            v_token_start: where vision tokens start in the sequence
            v_token_num: number of vision tokens
            text_token_start: where text tokens start (v_token_start + v_token_num)
            metadata: dict with question_id, image_file, answer, etc.
        """
        if self.h5_file is None:
            raise RuntimeError("HDF5 file not open. Call open() first.")

        grp = self.h5_file.create_group(sample_id)

        # Save metadata
        for k, v in metadata.items():
            if v is not None:
                grp.attrs[k] = str(v)
        grp.attrs["v_token_start"] = v_token_start
        grp.attrs["v_token_num"] = v_token_num
        grp.attrs["text_token_start"] = text_token_start

        # attentions from generate:
        # attentions[step][layer] = [B, H, L_step, L_kv]
        # Step 0 = prefill: L_step == full input length, L_kv == full input length
        # Step 1+  = decode: L_step == 1
        # We only need step 0 (prefill) for text→vision analysis
        prefill_attentions = attentions[0]  # tuple of num_layers tensors

        num_layers = len(prefill_attentions)

        for layer_idx in range(num_layers):
            if self.target_layers is not None and layer_idx not in self.target_layers:
                continue

            attn_w = prefill_attentions[layer_idx]  # [B, H, L, L]
            tv_attn = self.extract_tv_submatrix(
                attn_w, v_token_start, v_token_num, text_token_start
            )

            layer_grp = grp.create_group(f"layer_{layer_idx}")
            layer_grp.create_dataset(
                "tv_attn",
                data=tv_attn,
                compression=self.compression,
                compression_opts=self.compression_opts,
            )

            # Also compute and store prune scores
            prune_scores = self.compute_prune_scores(tv_attn)
            layer_grp.create_dataset("prune_scores", data=prune_scores.astype(np.float32))

        self.h5_file.flush()


def locate_image_tokens(
    input_ids: torch.Tensor,
    image_token_index: int = -200,
    v_token_num: int = 576,
) -> Tuple[int, int, int]:
    """
    Determine v_token_start and text_token_start from the original input_ids
    BEFORE prepare_inputs_labels_for_multimodal replaces image tokens with embeddings.

    For vicuna_v1 with LLaVA-1.5:
      input_ids = [system_prompt_tokens..., IMAGE_TOKEN(-200), question_tokens...]
      After multimodal preparation, the sequence becomes:
      [system_prompt_tokens..., 576_vision_embeddings, question_tokens...]

    So:
      v_token_start = index of IMAGE_TOKEN in input_ids
      v_token_num = configured visual token count
      text_token_start = v_token_start + v_token_num

    Args:
        input_ids: [1, L] input token ids (before multimodal preparation)
        image_token_index: the sentinel token id for image placeholder (default: -200)
        v_token_num: number of visual tokens after multimodal expansion

    Returns:
        (v_token_start, v_token_num, text_token_start)
    """
    ids = input_ids.squeeze()
    image_pos = (ids == image_token_index).nonzero(as_tuple=True)[0]
    if len(image_pos) == 0:
        raise ValueError("No image token found in input_ids")

    v_token_start = image_pos[0].item()
    # Count how many tokens come before the image token → that's the system prompt
    # The image token is replaced by v_token_num vision embeddings
    # Text (question) tokens start right after vision tokens
    # But we need to account for the fact that IMAGE_TOKEN is 1 token in input_ids
    # but becomes v_token_num tokens in the actual sequence.
    text_token_start = v_token_start + v_token_num

    return v_token_start, v_token_num, text_token_start
