"""
Core visual token pruning engine.

Implements a custom layer-by-layer prefill that prunes visual tokens at
designated layers, followed by greedy autoregressive decoding using the
pruned KV cache.

Compatible with transformers 4.37.x DynamicCache and eager-attention
LLaVA Llama/Mistral decoder backbones.
"""

import math
import time
import types
import warnings
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


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads to match the number of attention heads."""
    if n_rep == 1:
        return hidden_states
    batch_size, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch_size, num_key_value_heads, n_rep, seq_len, head_dim
    )
    return hidden_states.reshape(batch_size, num_key_value_heads * n_rep, seq_len, head_dim)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension by half for RoPE."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


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


def _apply_rotary_pos_emb(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = _gather_rotary_cache(cos, position_ids).to(dtype=query_states.dtype, device=query_states.device)
    sin = _gather_rotary_cache(sin, position_ids).to(dtype=query_states.dtype, device=query_states.device)
    query_states = (query_states * cos) + (_rotate_half(query_states) * sin)
    key_states = (key_states * cos) + (_rotate_half(key_states) * sin)
    return query_states, key_states


def _required_rotary_seq_len(
    position_ids: Optional[torch.Tensor],
    minimum_seq_len: int,
) -> int:
    """Ensure rotary caches cover the largest preserved position id."""
    if position_ids is None:
        return minimum_seq_len
    if position_ids.numel() == 0:
        return minimum_seq_len
    return max(minimum_seq_len, int(position_ids.max().item()) + 1)


def enable_sparse_position_ids_compat(model) -> bool:
    """Patch Llama or Mistral attention for sparse, non-reindexed positions."""
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        return False
    if getattr(model, "_sparse_position_ids_compat_enabled", False):
        return True

    attention_modules = [
        getattr(layer, "self_attn", None)
        for layer in layers
        if getattr(layer, "self_attn", None) is not None
    ]
    if not attention_modules:
        return False

    try:
        from transformers.models.llama.modeling_llama import (
            LlamaAttention,
            apply_rotary_pos_emb as llama_apply_rotary_pos_emb,
            repeat_kv as llama_repeat_kv,
        )
        from transformers.models.mistral.modeling_mistral import (
            MistralAttention,
            apply_rotary_pos_emb as mistral_apply_rotary_pos_emb,
            repeat_kv as mistral_repeat_kv,
        )
    except Exception:
        return False

    if all(isinstance(module, LlamaAttention) for module in attention_modules):
        attention_family = "llama"
        hf_apply_rotary_pos_emb = llama_apply_rotary_pos_emb
        hf_repeat_kv = llama_repeat_kv
    elif all(isinstance(module, MistralAttention) for module in attention_modules):
        attention_family = "mistral"
        hf_apply_rotary_pos_emb = mistral_apply_rotary_pos_emb
        hf_repeat_kv = mistral_repeat_kv
    else:
        return False

    def _patched_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        bsz, q_len, _ = hidden_states.size()

        pretraining_tp = max(int(getattr(self.config, "pretraining_tp", 1)), 1)
        if pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        rotary_seq_len = _required_rotary_seq_len(position_ids, kv_seq_len)
        cos, sin = self.rotary_emb(value_states, seq_len=rotary_seq_len)
        query_states, key_states = hf_apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = hf_repeat_kv(key_states, self.num_key_value_groups)
        value_states = hf_repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = torch.nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // pretraining_tp, dim=1)
            attn_output = sum(F.linear(attn_output[i], o_proj_slices[i]) for i in range(pretraining_tp))
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    for layer in layers:
        attn_module = getattr(layer, "self_attn", None)
        if attn_module is None:
            continue
        if getattr(attn_module, "_sparse_position_ids_compat_wrapped", False):
            continue
        attn_module._original_forward = attn_module.forward
        attn_module.forward = types.MethodType(_patched_forward, attn_module)
        attn_module._sparse_position_ids_compat_wrapped = True

    model._sparse_position_ids_compat_enabled = True
    model._sparse_position_ids_compat_family = attention_family
    return True


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
        position_ids: Optional[torch.Tensor],
        v_token_start: int,
        v_token_num: int,
        text_token_start: int,
        text_token_ids: Optional[torch.Tensor] = None,
        text_special_token_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 128,
        eos_token_id: int = 2,
        save_tv_attn: bool = False,
        capture_layers: Optional[set] = None,
        capture_visual_hidden_layers: Optional[set[int]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Generate with visual token pruning.

        Args:
            inputs_embeds: [1, L, D] merged embeddings (system + vision + question)
            attention_mask: [1, L] or None
            position_ids: [1, L] or None; if provided, preserved across physical pruning
            v_token_start:  index of first visual token
            v_token_num:    number of merged visual tokens for this sample
            text_token_start: index of first text token after vision block
            max_new_tokens: maximum tokens to generate
            eos_token_id:   id of the EOS token for stopping
            save_tv_attn:   if True, capture text→vision attention sub-matrices
            capture_layers: set of layer indices to capture (None = all layers)
            capture_visual_hidden_layers: set of layer indices whose pre-prune
                visual hidden states should be exported in prune_info

        Returns:
            (generated_ids [1, N], prune_info dict)
        """
        device = inputs_embeds.device

        # --- Pruned prefill ---
        t0 = time.time()
        try:
            hidden_states, past_kv, prune_info = self._pruned_prefill(
                inputs_embeds,
                position_ids,
                v_token_start,
                v_token_num,
                text_token_start,
                text_token_ids=text_token_ids,
                text_special_token_mask=text_special_token_mask,
                save_tv_attn=save_tv_attn,
                capture_layers=capture_layers,
                capture_visual_hidden_layers=capture_visual_hidden_layers,
            )
        finally:
            self.strategy.clear_sample()
        t_prefill = time.time() - t0

        # --- First token ---
        logits = self.model.lm_head(hidden_states[:, -1:])  # [1, 1, V]
        next_token = logits.argmax(dim=-1)                   # [1, 1]

        generated = [next_token]

        # --- Autoregressive decode ---
        t1 = time.time()
        cache_len = past_kv.get_seq_length()  # after prefill
        next_position_id = int(prune_info["final_position_ids"][-1]) + 1

        for step in range(max_new_tokens - 1):
            if next_token.item() == eos_token_id:
                break

            token_embeds = self.model.model.embed_tokens(next_token)  # [1, 1, D]
            pos_id = torch.tensor([[next_position_id + step]], device=device, dtype=torch.long)
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
        initial_position_ids: Optional[torch.Tensor] = None,
        v_token_start: Optional[int] = None,
        v_token_num: Optional[int] = None,
        text_token_start: Optional[int] = None,
        text_token_ids: Optional[torch.Tensor] = None,
        text_special_token_mask: Optional[torch.Tensor] = None,
        save_tv_attn: bool = False,
        capture_layers: Optional[set] = None,
        capture_visual_hidden_layers: Optional[set[int]] = None,
    ) -> Tuple[torch.Tensor, DynamicCache, Dict[str, Any]]:
        if v_token_start is None or v_token_num is None or text_token_start is None:
            raise TypeError("v_token_start, v_token_num, and text_token_start are required")
        v_token_start = int(v_token_start)
        v_token_num = int(v_token_num)
        text_token_start = int(text_token_start)

        device = inputs_embeds.device
        dtype = inputs_embeds.dtype
        batch_size, seq_len, _ = inputs_embeds.shape

        hidden_states = inputs_embeds
        if initial_position_ids is None:
            position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
        else:
            position_ids = initial_position_ids.to(device=device, dtype=torch.long).clone()
        causal_mask = _make_causal_mask(seq_len, dtype, device)

        past_kv = DynamicCache()
        prune_stage = self.strategy.prune_stage()
        prune_info: Dict[str, Any] = {"layers": {}, "prune_stage": prune_stage}
        sample_info = self.strategy.prepare_sample(
            inputs_embeds=inputs_embeds,
            v_token_start=v_token_start,
            v_token_num=v_token_num,
            text_token_start=text_token_start,
            text_token_ids=text_token_ids,
            text_special_token_mask=text_special_token_mask,
        )
        if sample_info:
            prune_info["sample"] = sample_info

        cur_v_start = v_token_start
        cur_v_num = v_token_num
        cur_text_start = text_token_start

        for layer_idx, layer in enumerate(self.model.model.layers):
            need_prune = layer_idx in self.prune_layers
            need_capture = save_tv_attn and (capture_layers is None or layer_idx in capture_layers)
            need_visual_hidden_capture = (
                capture_visual_hidden_layers is not None
                and layer_idx in capture_visual_hidden_layers
                and cur_v_num > 0
            )
            if prune_stage == "pre" and need_prune and cur_v_num > 0:
                hidden_states, position_ids, causal_mask, cur_v_num, cur_text_start, layer_info = self._run_pre_prune_layer(
                    layer=layer,
                    layer_idx=layer_idx,
                    hidden_states=hidden_states,
                    position_ids=position_ids,
                    causal_mask=causal_mask,
                    past_kv=past_kv,
                    v_token_start=cur_v_start,
                    v_token_num=cur_v_num,
                    text_token_start=cur_text_start,
                    need_capture=need_capture,
                    dtype=dtype,
                    device=device,
                )
                layer_info["layer_idx"] = layer_idx
                layer_info["prune_stage"] = prune_stage
                prune_info["layers"][layer_idx] = layer_info
                continue
            if prune_stage == "masking" and need_prune and cur_v_num > 0:
                hidden_states, layer_info = self._run_masking_prune_layer(
                    layer=layer,
                    layer_idx=layer_idx,
                    hidden_states=hidden_states,
                    position_ids=position_ids,
                    causal_mask=causal_mask,
                    past_kv=past_kv,
                    v_token_start=cur_v_start,
                    v_token_num=cur_v_num,
                    text_token_start=cur_text_start,
                    need_capture=need_capture,
                )
                layer_info["layer_idx"] = layer_idx
                layer_info["prune_stage"] = prune_stage
                prune_info["layers"][layer_idx] = layer_info
                continue

            need_attn = need_capture or (need_prune and self.strategy.requires_attention())

            out = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_kv,
                use_cache=True,
                output_attentions=need_attn,
            )

            hidden_states = out[0]

            visual_hidden_info = None
            if need_visual_hidden_capture:
                visual_hidden_info = self._capture_visual_hidden(
                    hidden_states=hidden_states,
                    v_token_start=cur_v_start,
                    v_token_num=cur_v_num,
                )

            # Capture diagnostics for non-prune layers
            if (need_capture or visual_hidden_info is not None) and not need_prune and cur_v_num > 0:
                layer_capture_info = prune_info["layers"].setdefault(layer_idx, {"layer_idx": layer_idx})
                if visual_hidden_info is not None:
                    layer_capture_info.update(visual_hidden_info)
                attn_weights = out[1]
                if need_capture and attn_weights is not None:
                    v_end = cur_v_start + cur_v_num
                    tv_attn = attn_weights[0, :, cur_text_start:, cur_v_start:v_end].cpu()
                    layer_capture_info["tv_attn"] = tv_attn

            if need_prune and cur_v_num > 0:
                attn_weights = out[1] if need_attn else None

                # Extract text→vision sub-matrix before pruning changes positions
                if need_capture:
                    v_end = cur_v_start + cur_v_num
                    tv_attn = attn_weights[0, :, cur_text_start:, cur_v_start:v_end].cpu()

                keep_indices, layer_info = self.strategy.compute_keep_mask(
                    attn_weights, cur_v_start, cur_v_num,
                    cur_text_start, layer_idx, device=device,
                    current_visual_embeds=hidden_states[0, cur_v_start:cur_v_start + cur_v_num],
                )

                num_pruned = cur_v_num - len(keep_indices)
                if num_pruned > 0:
                    full_keep = self._build_full_keep_mask(
                        hidden_states.shape[1], cur_v_start, cur_v_num,
                        keep_indices, device,
                    )

                    hidden_states = hidden_states[:, full_keep]
                    self._prune_kv_cache(past_kv, full_keep, up_to_layer=layer_idx)

                    position_ids, causal_mask = self._refresh_sequence_state(
                        position_ids=position_ids[:, full_keep],
                        dtype=dtype,
                        device=device,
                    )

                    cur_v_num = len(keep_indices)
                    cur_text_start = cur_v_start + cur_v_num
                self.strategy.update_after_prune(keep_indices=keep_indices, layer_idx=layer_idx)

                if need_capture:
                    layer_info["tv_attn"] = tv_attn  # [H, L_t, L_v]
                if visual_hidden_info is not None:
                    layer_info.update(visual_hidden_info)

                layer_info["layer_idx"] = layer_idx
                layer_info["prune_stage"] = prune_stage
                prune_info["layers"][layer_idx] = layer_info

        hidden_states = self.model.model.norm(hidden_states)

        prune_info["original_seq_len"] = seq_len
        prune_info["final_seq_len"] = hidden_states.shape[1]
        prune_info["final_position_ids"] = position_ids[0].detach().cpu().tolist()
        prune_info["prune_layers"] = self.prune_layers

        return hidden_states, past_kv, prune_info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _capture_visual_hidden(
        hidden_states: torch.Tensor,
        v_token_start: int,
        v_token_num: int,
    ) -> Dict[str, Any]:
        visual_hidden = hidden_states[
            0,
            v_token_start:v_token_start + v_token_num,
        ].detach().cpu()
        return {
            "visual_hidden": visual_hidden,
            "visual_hidden_v_token_start": int(v_token_start),
            "visual_hidden_v_token_num": int(v_token_num),
            "visual_hidden_shape": list(visual_hidden.shape),
        }

    def _run_pre_prune_layer(
        self,
        layer,
        layer_idx: int,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        causal_mask: torch.Tensor,
        past_kv: DynamicCache,
        v_token_start: int,
        v_token_num: int,
        text_token_start: int,
        need_capture: bool,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, Dict[str, Any]]:
        tv_attn, importance_scores = self._compute_pre_prune_scores(
            layer=layer,
            hidden_states=hidden_states,
            position_ids=position_ids,
            attention_mask=causal_mask,
            v_token_start=v_token_start,
            v_token_num=v_token_num,
            text_token_start=text_token_start,
        )
        keep_indices, layer_info = self.strategy.compute_keep_mask_from_importance(
            importance_scores=importance_scores,
            layer_idx=layer_idx,
        )

        num_pruned = v_token_num - len(keep_indices)
        cur_v_num = v_token_num
        cur_text_start = text_token_start
        if num_pruned > 0:
            full_keep = self._build_full_keep_mask(
                hidden_states.shape[1], v_token_start, v_token_num, keep_indices, device
            )
            hidden_states = hidden_states[:, full_keep]
            self._prune_kv_cache(past_kv, full_keep, up_to_layer=layer_idx - 1)
            cur_v_num = len(keep_indices)
            cur_text_start = v_token_start + cur_v_num
            position_ids, causal_mask = self._refresh_sequence_state(
                position_ids=position_ids[:, full_keep],
                dtype=dtype,
                device=device,
            )
            self.strategy.update_after_prune(keep_indices=keep_indices, layer_idx=layer_idx)

        out = layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=past_kv,
            use_cache=True,
            output_attentions=False,
        )
        hidden_states = out[0]

        if need_capture:
            layer_info["tv_attn"] = tv_attn.cpu()

        return hidden_states, position_ids, causal_mask, cur_v_num, cur_text_start, layer_info

    def _run_masking_prune_layer(
        self,
        layer,
        layer_idx: int,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        causal_mask: torch.Tensor,
        past_kv: DynamicCache,
        v_token_start: int,
        v_token_num: int,
        text_token_start: int,
        need_capture: bool,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        tv_attn, importance_scores = self._compute_pre_prune_scores(
            layer=layer,
            hidden_states=hidden_states,
            position_ids=position_ids,
            attention_mask=causal_mask,
            v_token_start=v_token_start,
            v_token_num=v_token_num,
            text_token_start=text_token_start,
        )
        keep_indices, layer_info = self.strategy.compute_keep_mask_from_importance(
            importance_scores=importance_scores,
            layer_idx=layer_idx,
        )

        hidden_states = self._forward_masked_layer(
            layer=layer,
            layer_idx=layer_idx,
            hidden_states=hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_kv=past_kv,
            v_token_start=v_token_start,
            v_token_num=v_token_num,
            text_token_start=text_token_start,
            keep_indices=keep_indices.to(device=hidden_states.device),
        )

        if need_capture:
            layer_info["tv_attn"] = tv_attn.cpu()

        return hidden_states, layer_info

    def _compute_pre_prune_scores(
        self,
        layer,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        v_token_start: int,
        v_token_num: int,
        text_token_start: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if hidden_states.shape[0] != 1:
            raise NotImplementedError(
                f"pre_attn_score only supports batch size 1, got batch={hidden_states.shape[0]}"
            )
        required_layer_attrs = ["input_layernorm", "self_attn"]
        missing_layer_attrs = [name for name in required_layer_attrs if not hasattr(layer, name)]
        if missing_layer_attrs:
            raise NotImplementedError(
                f"pre_attn_score requires a Llama-style decoder layer with {missing_layer_attrs}, "
                f"but layer {type(layer).__name__} does not provide them"
            )

        attn_module = layer.self_attn
        required_attn_attrs = ["q_proj", "k_proj", "rotary_emb", "num_heads", "head_dim"]
        missing_attn_attrs = [name for name in required_attn_attrs if not hasattr(attn_module, name)]
        if missing_attn_attrs:
            raise NotImplementedError(
                f"pre_attn_score requires a Llama-style self attention module with {missing_attn_attrs}, "
                f"but {type(attn_module).__name__} does not provide them"
            )

        normed_hidden = layer.input_layernorm(hidden_states)
        batch_size, seq_len, _ = normed_hidden.size()
        num_heads = int(attn_module.num_heads)
        head_dim = int(attn_module.head_dim)
        num_key_value_heads = int(getattr(attn_module, "num_key_value_heads", num_heads))
        num_key_value_groups = int(getattr(attn_module, "num_key_value_groups", max(num_heads // num_key_value_heads, 1)))

        query_states = attn_module.q_proj(normed_hidden).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        key_states = attn_module.k_proj(normed_hidden).view(
            batch_size, seq_len, num_key_value_heads, head_dim
        ).transpose(1, 2)

        rotary_input = key_states
        rotary_seq_len = _required_rotary_seq_len(position_ids, seq_len)
        try:
            cos, sin = attn_module.rotary_emb(rotary_input, seq_len=rotary_seq_len)
        except TypeError:
            cos, sin = attn_module.rotary_emb(rotary_input, rotary_seq_len)
        query_states, key_states = _apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        key_states = _repeat_kv(key_states, num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
        if attention_mask is not None:
            if attention_mask.dim() != 4:
                raise NotImplementedError(
                    f"pre_attn_score expects a 4D causal mask, got shape {tuple(attention_mask.shape)}"
                )
            attn_weights = attn_weights + attention_mask.to(dtype=attn_weights.dtype, device=attn_weights.device)
            min_value = torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device, dtype=attn_weights.dtype)
            attn_weights = torch.max(attn_weights, min_value)

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        v_end = v_token_start + v_token_num
        tv_attn = attn_weights[0, :, text_token_start:, v_token_start:v_end]
        importance_scores = tv_attn.mean(dim=0).mean(dim=0)
        return tv_attn, importance_scores

    def _forward_masked_layer(
        self,
        layer,
        layer_idx: int,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: torch.Tensor,
        past_kv: DynamicCache,
        v_token_start: int,
        v_token_num: int,
        text_token_start: int,
        keep_indices: torch.Tensor,
    ) -> torch.Tensor:
        if hasattr(layer, "forward_with_attention_logits_mask"):
            return layer.forward_with_attention_logits_mask(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_kv,
                use_cache=True,
                v_token_start=v_token_start,
                v_token_num=v_token_num,
                text_token_start=text_token_start,
                keep_indices=keep_indices,
            )[0]

        required_layer_attrs = ["input_layernorm", "self_attn", "post_attention_layernorm", "mlp"]
        missing_layer_attrs = [name for name in required_layer_attrs if not hasattr(layer, name)]
        if missing_layer_attrs:
            raise NotImplementedError(
                f"masking_attn_score requires a Llama-style decoder layer with {missing_layer_attrs}, "
                f"but layer {type(layer).__name__} does not provide them"
            )

        attn_module = layer.self_attn
        required_attn_attrs = [
            "q_proj", "k_proj", "v_proj", "o_proj", "rotary_emb",
            "num_heads", "head_dim",
        ]
        missing_attn_attrs = [name for name in required_attn_attrs if not hasattr(attn_module, name)]
        if missing_attn_attrs:
            raise NotImplementedError(
                f"masking_attn_score requires a Llama-style self attention module with {missing_attn_attrs}, "
                f"but {type(attn_module).__name__} does not provide them"
            )

        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)

        batch_size, seq_len, hidden_size = hidden_states.size()
        num_heads = int(attn_module.num_heads)
        head_dim = int(attn_module.head_dim)
        num_key_value_heads = int(getattr(attn_module, "num_key_value_heads", num_heads))
        num_key_value_groups = int(
            getattr(attn_module, "num_key_value_groups", max(num_heads // num_key_value_heads, 1))
        )

        query_states = attn_module.q_proj(hidden_states).view(
            batch_size, seq_len, num_heads, head_dim
        ).transpose(1, 2)
        key_states = attn_module.k_proj(hidden_states).view(
            batch_size, seq_len, num_key_value_heads, head_dim
        ).transpose(1, 2)
        value_states = attn_module.v_proj(hidden_states).view(
            batch_size, seq_len, num_key_value_heads, head_dim
        ).transpose(1, 2)

        rotary_input = value_states
        rotary_seq_len = _required_rotary_seq_len(position_ids, seq_len)
        try:
            cos, sin = attn_module.rotary_emb(rotary_input, seq_len=rotary_seq_len)
        except TypeError:
            cos, sin = attn_module.rotary_emb(rotary_input, rotary_seq_len)
        query_states, key_states = _apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if len(past_kv.key_cache) <= layer_idx:
            past_kv.key_cache.append(key_states)
            past_kv.value_cache.append(value_states)
        else:
            past_kv.key_cache[layer_idx] = key_states
            past_kv.value_cache[layer_idx] = value_states
        past_kv._seen_tokens = key_states.shape[2]

        key_states_for_attn = _repeat_kv(key_states, num_key_value_groups)
        value_states_for_attn = _repeat_kv(value_states, num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states_for_attn.transpose(2, 3)) / math.sqrt(head_dim)
        if attention_mask is not None:
            if attention_mask.dim() != 4:
                raise NotImplementedError(
                    f"masking_attn_score expects a 4D causal mask, got shape {tuple(attention_mask.shape)}"
                )
            attn_weights = attn_weights + attention_mask.to(
                dtype=attn_weights.dtype,
                device=attn_weights.device,
            )

        masked_attn_bias = torch.zeros_like(attn_weights)
        drop_mask = torch.ones(v_token_num, dtype=torch.bool, device=hidden_states.device)
        drop_mask[keep_indices] = False
        text_slice = slice(text_token_start, seq_len)
        vision_drop_indices = (torch.arange(v_token_num, device=hidden_states.device)[drop_mask] + v_token_start)
        if vision_drop_indices.numel() > 0:
            min_value = torch.finfo(attn_weights.dtype).min
            masked_attn_bias[:, :, text_slice, vision_drop_indices] = min_value
            attn_weights = attn_weights + masked_attn_bias

        min_value = torch.tensor(
            torch.finfo(attn_weights.dtype).min,
            device=attn_weights.device,
            dtype=attn_weights.dtype,
        )
        attn_weights = torch.max(attn_weights, min_value)
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        attn_output = torch.matmul(attn_weights, value_states_for_attn)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        attn_output = attn_module.o_proj(attn_output)

        hidden_states = residual + attn_output
        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = residual + layer.mlp(hidden_states)
        return hidden_states

    @staticmethod
    def _refresh_sequence_state(
        position_ids: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if position_ids.dim() != 2:
            raise ValueError(f"position_ids must be 2D [B, L], got shape {tuple(position_ids.shape)}")
        preserved_position_ids = position_ids.to(device=device, dtype=torch.long)
        seq_len = preserved_position_ids.shape[1]
        causal_mask = _make_causal_mask(seq_len, dtype, device)
        return preserved_position_ids, causal_mask

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
