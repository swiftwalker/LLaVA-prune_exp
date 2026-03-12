"""
Core visual token pruning engine.

Implements a custom layer-by-layer prefill that prunes visual tokens at
designated layers, followed by greedy autoregressive decoding using the
pruned KV cache.

Compatible with:
  - transformers 4.37.x DynamicCache
  - LLaVA-1.5 (LlamaForCausalLM backbone, eager attention)
"""

import math
import time
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import DynamicCache

from strategies.base import PruneStrategy


# ---------------------------------------------------------------------------
# Causal mask helpers (matches transformers 4.37 eager-attention format)
# ---------------------------------------------------------------------------

def _make_causal_mask(
    seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
    past_kv_len: int = 0,
) -> torch.Tensor:
    """
    Create a 4-D causal attention mask.

    Returns:
        [1, 1, seq_len, past_kv_len + seq_len]
        0 for positions that MAY be attended to, float('-inf') otherwise.
    """
    total_len = past_kv_len + seq_len
    mask = torch.full((seq_len, total_len), torch.finfo(dtype).min, device=device, dtype=dtype)
    rows = torch.arange(seq_len, device=device).unsqueeze(1)
    cols = torch.arange(total_len, device=device).unsqueeze(0)
    mask[cols <= (rows + past_kv_len)] = 0.0
    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, Q, KV]


# ---------------------------------------------------------------------------
# Pruner
# ---------------------------------------------------------------------------

class VisualTokenPruner:
    """
    Manages visual token pruning during LLaVA inference.

    Workflow:
      1. Receive merged embeddings from LLaVA's ``prepare_inputs_labels_for_multimodal``.
      2. Run layer-by-layer prefill; at each designated prune layer compute
         importance scores, remove unimportant visual tokens from hidden states
         **and** KV caches of all preceding layers.
      3. Finish prefill, then greedy-decode up to ``max_new_tokens``.
    """

    def __init__(self, model, strategy: PruneStrategy, prune_config: dict):
        self.model = model
        self.strategy = strategy
        self.config = prune_config
        self.prune_layers: List[int] = self._determine_prune_layers()
        self.layer_ratio_map: Dict[int, float] = self._build_layer_ratio_map()
        # Inject the mapping into the strategy so get_prune_ratio() can use it
        self.strategy.config["prune_ratio_map"] = self.layer_ratio_map

    # ------------------------------------------------------------------
    # Layer selection
    # ------------------------------------------------------------------
    def _determine_prune_layers(self) -> List[int]:
        method = self.config.get("layer_selection", "fixed")

        if method == "fixed":
            layers = self.config["prune_layers"]
            if isinstance(layers, int):
                layers = [layers]
            return list(layers)
        elif method == "dynamic":
            # placeholder — falls back to config prune_layers for now
            layers = self.config.get("prune_layers", [2, 3])
            if isinstance(layers, int):
                layers = [layers]
            return list(layers)
        else:
            raise ValueError(f"Unknown layer_selection method: {method}")

    def _build_layer_ratio_map(self) -> Dict[int, float]:
        """Build a mapping from layer index to its prune ratio.

        When ``prune_layers`` and ``prune_ratio`` are both lists of equal
        length, each element pairs a layer with its ratio.  A scalar
        ``prune_ratio`` is broadcast to all prune layers.
        """
        ratios = self.config.get("prune_ratio", 0.5)
        if isinstance(ratios, (int, float)):
            return {l: float(ratios) for l in self.prune_layers}
        ratios = list(ratios)
        if len(ratios) != len(self.prune_layers):
            raise ValueError(
                f"prune_layers ({self.prune_layers}) and prune_ratio ({ratios}) "
                f"must have the same length when both are lists"
            )
        return dict(zip(self.prune_layers, [float(r) for r in ratios]))

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def pruned_generate(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        v_token_start: int,
        v_token_num: int,
        text_token_start: int,
        max_new_tokens: int = 128,
        eos_token_id: int = 2,
        save_tv_attn: bool = False,
        capture_layers: Optional[set] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Generate with visual token pruning.

        Args:
            inputs_embeds: [1, L, D] merged embeddings (system + vision + question)
            attention_mask: [1, L] or None
            v_token_start:  index of first visual token
            v_token_num:    number of visual tokens (576)
            text_token_start: index of first text token after vision block
            max_new_tokens: maximum tokens to generate
            eos_token_id:   id of the EOS token for stopping
            save_tv_attn:   if True, capture text→vision attention sub-matrices
            capture_layers: set of layer indices to capture (None = all layers)

        Returns:
            (generated_ids [1, N], prune_info dict)
        """
        device = inputs_embeds.device

        # --- Pruned prefill ---
        t0 = time.time()
        hidden_states, past_kv, prune_info = self._pruned_prefill(
            inputs_embeds, v_token_start, v_token_num, text_token_start,
            save_tv_attn=save_tv_attn,
            capture_layers=capture_layers,
        )
        t_prefill = time.time() - t0

        # --- First token ---
        logits = self.model.lm_head(hidden_states[:, -1:])  # [1, 1, V]
        next_token = logits.argmax(dim=-1)                   # [1, 1]

        generated = [next_token]

        # --- Autoregressive decode ---
        t1 = time.time()
        cache_len = past_kv.get_seq_length()  # after prefill

        for step in range(max_new_tokens - 1):
            if next_token.item() == eos_token_id:
                break

            token_embeds = self.model.model.embed_tokens(next_token)  # [1, 1, D]
            pos_id = torch.tensor([[cache_len + step]], device=device, dtype=torch.long)
            causal_mask = _make_causal_mask(1, token_embeds.dtype, device, past_kv_len=cache_len + step)

            hidden = token_embeds
            for layer in self.model.model.layers:
                out = layer(
                    hidden,
                    attention_mask=causal_mask,
                    position_ids=pos_id,
                    past_key_value=past_kv,
                    use_cache=True,
                    output_attentions=False,
                )
                hidden = out[0]

            hidden = self.model.model.norm(hidden)
            logits = self.model.lm_head(hidden)
            next_token = logits.argmax(dim=-1)
            generated.append(next_token)

        t_decode = time.time() - t1
        generated_ids = torch.cat(generated, dim=1)  # [1, N]

        prune_info["prefill_time"] = t_prefill
        prune_info["decode_time"] = t_decode
        prune_info["total_time"] = t_prefill + t_decode
        prune_info["num_generated_tokens"] = generated_ids.shape[1]

        return generated_ids, prune_info

    # ------------------------------------------------------------------
    # Pruned prefill (layer-by-layer)
    # ------------------------------------------------------------------
    def _pruned_prefill(
        self,
        inputs_embeds: torch.Tensor,
        v_token_start: int,
        v_token_num: int,
        text_token_start: int,
        save_tv_attn: bool = False,
        capture_layers: Optional[set] = None,
    ) -> Tuple[torch.Tensor, DynamicCache, Dict[str, Any]]:
        device = inputs_embeds.device
        dtype = inputs_embeds.dtype
        batch_size, seq_len, _ = inputs_embeds.shape

        hidden_states = inputs_embeds
        position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
        causal_mask = _make_causal_mask(seq_len, dtype, device)

        past_kv = DynamicCache()
        prune_info: Dict[str, Any] = {"layers": {}}

        cur_v_start = v_token_start
        cur_v_num = v_token_num
        cur_text_start = text_token_start

        for layer_idx, layer in enumerate(self.model.model.layers):
            need_prune = layer_idx in self.prune_layers
            need_capture = save_tv_attn and (capture_layers is None or layer_idx in capture_layers)
            need_attn = need_prune or need_capture

            out = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_kv,
                use_cache=True,
                output_attentions=need_attn,
            )

            hidden_states = out[0]

            # Capture tv_attn for non-prune layers
            if need_capture and not need_prune and cur_v_num > 0:
                attn_weights = out[1]
                if attn_weights is not None:
                    v_end = cur_v_start + cur_v_num
                    tv_attn = attn_weights[0, :, cur_text_start:, cur_v_start:v_end].cpu()
                    prune_info["layers"][layer_idx] = {
                        "layer_idx": layer_idx,
                        "tv_attn": tv_attn,
                    }

            if need_prune and cur_v_num > 0:
                attn_weights = out[1]  # [B, H, cur_len, cur_len]

                if attn_weights is not None:
                    # Extract text→vision sub-matrix before pruning changes positions
                    if need_capture:
                        v_end = cur_v_start + cur_v_num
                        tv_attn = attn_weights[0, :, cur_text_start:, cur_v_start:v_end].cpu()

                    keep_indices, layer_info = self.strategy.compute_keep_mask(
                        attn_weights, cur_v_start, cur_v_num,
                        cur_text_start, layer_idx,
                    )

                    num_pruned = cur_v_num - len(keep_indices)
                    if num_pruned > 0:
                        full_keep = self._build_full_keep_mask(
                            hidden_states.shape[1], cur_v_start, cur_v_num,
                            keep_indices, device,
                        )

                        hidden_states = hidden_states[:, full_keep]
                        self._prune_kv_cache(past_kv, full_keep, up_to_layer=layer_idx)

                        new_len = hidden_states.shape[1]
                        position_ids = torch.arange(new_len, device=device, dtype=torch.long).unsqueeze(0)
                        causal_mask = _make_causal_mask(new_len, dtype, device)

                        cur_v_num = len(keep_indices)
                        cur_text_start = cur_v_start + cur_v_num

                    if need_capture:
                        layer_info["tv_attn"] = tv_attn  # [H, L_t, L_v]

                    layer_info["layer_idx"] = layer_idx
                    prune_info["layers"][layer_idx] = layer_info

        hidden_states = self.model.model.norm(hidden_states)

        prune_info["original_seq_len"] = seq_len
        prune_info["final_seq_len"] = hidden_states.shape[1]
        prune_info["prune_layers"] = self.prune_layers

        return hidden_states, past_kv, prune_info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _build_full_keep_mask(
        seq_len: int,
        v_start: int,
        v_num: int,
        keep_indices: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Boolean mask over the full sequence: True for kept positions."""
        mask = torch.ones(seq_len, dtype=torch.bool, device=device)
        mask[v_start: v_start + v_num] = False
        mask[v_start + keep_indices] = True
        return mask

    @staticmethod
    def _prune_kv_cache(
        cache: DynamicCache,
        keep_mask: torch.Tensor,
        up_to_layer: int,
    ):
        """Remove pruned positions from KV cache for layers 0 … up_to_layer."""
        for i in range(min(up_to_layer + 1, len(cache.key_cache))):
            cache.key_cache[i] = cache.key_cache[i][:, :, keep_mask, :]
            cache.value_cache[i] = cache.value_cache[i][:, :, keep_mask, :]
        # Fix the seen-tokens counter (tracked at layer 0)
        if len(cache.key_cache) > 0:
            cache._seen_tokens = cache.key_cache[0].shape[2]
