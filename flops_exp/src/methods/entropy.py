from methods.attn_score import AttnScoreEstimator
from methods.base import MethodEstimate


class EntropyEstimator(AttnScoreEstimator):
    strategy_name = "entropy"

    def estimate_method_flops(
        self,
        model_profile,
        prefill_seq_len: int,
        image_token_len: int,
        prune_spec,
    ) -> MethodEstimate:
        base = super().estimate_method_flops(
            model_profile=model_profile,
            prefill_seq_len=prefill_seq_len,
            image_token_len=image_token_len,
            prune_spec=prune_spec,
        )
        schedule = self.build_sequence_schedule(
            model_profile=model_profile,
            prefill_seq_len=prefill_seq_len,
            image_token_len=image_token_len,
            prune_spec=prune_spec,
        )
        schedule_by_layer = {
            segment.end_layer: segment
            for segment in schedule
            if segment.end_layer in prune_spec.prune_layers
        }
        extra = 0.0
        for prune_layer in prune_spec.prune_layers:
            segment = schedule_by_layer[prune_layer]
            text_tokens = max(segment.seq_len - segment.visual_tokens, 0)
            extra += float(
                5 * model_profile.num_attention_heads * text_tokens * segment.visual_tokens
            )
        notes = dict(base.non_flop_overhead)
        notes["dynamic_ratio"] = (
            "entropy dynamic ratio affects estimated runtime behavior but uses configured base ratios for length reduction"
        )
        return MethodEstimate(
            prune_method_flops=base.prune_method_flops + extra,
            non_flop_overhead=notes,
        )
