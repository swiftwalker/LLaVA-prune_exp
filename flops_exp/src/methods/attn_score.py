from methods.base import BaseMethodEstimator, MethodEstimate


class AttnScoreEstimator(BaseMethodEstimator):
    strategy_name = "attn_score"

    def estimate_method_flops(
        self,
        model_profile,
        prefill_seq_len: int,
        image_token_len: int,
        prune_spec,
    ) -> MethodEstimate:
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
        total = 0.0
        for prune_layer in prune_spec.prune_layers:
            segment = schedule_by_layer[prune_layer]
            text_tokens = max(segment.seq_len - segment.visual_tokens, 0)
            total += float(
                2 * model_profile.num_attention_heads * text_tokens * segment.visual_tokens
            )
        return MethodEstimate(
            prune_method_flops=total,
            non_flop_overhead={
                "selection": "top-k sorting/index selection not counted as FLOPs",
                "state_update": "kv-cache pruning and tensor movement not counted as FLOPs",
            },
        )
