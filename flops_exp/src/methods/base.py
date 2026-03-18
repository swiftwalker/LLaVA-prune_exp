from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class LayerSegment:
    start_layer: int
    end_layer: int
    seq_len: int
    visual_tokens: int

    @property
    def num_layers(self) -> int:
        return self.end_layer - self.start_layer + 1

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MethodEstimate:
    prune_method_flops: float
    non_flop_overhead: dict[str, Any]


class BaseMethodEstimator:
    strategy_name = "base"

    def build_sequence_schedule(
        self,
        model_profile,
        prefill_seq_len: int,
        image_token_len: int,
        prune_spec,
    ) -> list[LayerSegment]:
        if not prune_spec.prune_layers:
            return [
                LayerSegment(
                    start_layer=0,
                    end_layer=model_profile.num_hidden_layers - 1,
                    seq_len=int(prefill_seq_len),
                    visual_tokens=int(image_token_len),
                )
            ]

        schedule: list[LayerSegment] = []
        current_layer = 0
        current_seq_len = int(prefill_seq_len)
        current_visual_tokens = int(image_token_len)

        for prune_layer in sorted(prune_spec.prune_layers):
            if prune_layer < current_layer or prune_layer >= model_profile.num_hidden_layers:
                raise ValueError(f"Invalid prune layer {prune_layer} for model depth {model_profile.num_hidden_layers}")

            schedule.append(
                LayerSegment(
                    start_layer=current_layer,
                    end_layer=prune_layer,
                    seq_len=current_seq_len,
                    visual_tokens=current_visual_tokens,
                )
            )

            ratio = float(prune_spec.prune_ratio_map.get(prune_layer, 0.0))
            num_pruned = int(current_visual_tokens * ratio)
            kept_visual_tokens = current_visual_tokens - num_pruned
            current_seq_len -= num_pruned
            current_visual_tokens = kept_visual_tokens
            current_layer = prune_layer + 1

        if current_layer <= model_profile.num_hidden_layers - 1:
            schedule.append(
                LayerSegment(
                    start_layer=current_layer,
                    end_layer=model_profile.num_hidden_layers - 1,
                    seq_len=current_seq_len,
                    visual_tokens=current_visual_tokens,
                )
            )
        return schedule

    def estimate_method_flops(
        self,
        model_profile,
        prefill_seq_len: int,
        image_token_len: int,
        prune_spec,
    ) -> MethodEstimate:
        return MethodEstimate(prune_method_flops=0.0, non_flop_overhead={})

