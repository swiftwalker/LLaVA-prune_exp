from dataclasses import asdict, dataclass

from methods import get_method_estimator


@dataclass
class FlopsBreakdown:
    self_attention_flops: float
    mlp_flops: float
    prune_method_flops: float
    total_llm_flops: float
    non_flop_overhead: dict

    def to_dict(self) -> dict:
        return asdict(self)


def attention_flops_per_layer(seq_len: int, hidden_size: int) -> float:
    return float(8 * seq_len * (hidden_size ** 2) + 4 * (seq_len ** 2) * hidden_size)


def mlp_flops_per_layer(seq_len: int, hidden_size: int, intermediate_size: int) -> float:
    return float(6 * seq_len * hidden_size * intermediate_size)


def estimate_schedule_flops(schedule, hidden_size: int, intermediate_size: int) -> tuple[float, float]:
    self_attention = 0.0
    mlp = 0.0
    for segment in schedule:
        self_attention += segment.num_layers * attention_flops_per_layer(segment.seq_len, hidden_size)
        mlp += segment.num_layers * mlp_flops_per_layer(
            segment.seq_len, hidden_size, intermediate_size
        )
    return self_attention, mlp


def compute_prefill_seq_len(text_token_len: float, template_overhead_tokens: int, image_token_len: int) -> int:
    return int(round(text_token_len + template_overhead_tokens - 1 + image_token_len))


def compute_flops_report(
    model_profile,
    prune_spec,
    text_token_len: float,
    template_overhead_tokens: int,
    image_token_len: int,
) -> dict:
    prefill_seq_len = compute_prefill_seq_len(
        text_token_len=text_token_len,
        template_overhead_tokens=template_overhead_tokens,
        image_token_len=image_token_len,
    )

    baseline_schedule = [
        get_method_estimator("baseline").build_sequence_schedule(
            model_profile=model_profile,
            prefill_seq_len=prefill_seq_len,
            image_token_len=image_token_len,
            prune_spec=prune_spec.__class__(
                strategy_name="baseline",
                layer_selection="fixed",
                prune_layers=[],
                prune_ratio_map={},
                ratio_mode="static",
                scoring_mode="none",
                run_mode="baseline",
                config_source="synthetic_baseline",
                source_path="",
                source_dataset=None,
                source_run_name=None,
                notes=[],
            ),
        )[0]
    ]
    baseline_attention, baseline_mlp = estimate_schedule_flops(
        baseline_schedule,
        hidden_size=model_profile.hidden_size,
        intermediate_size=model_profile.intermediate_size,
    )
    baseline = FlopsBreakdown(
        self_attention_flops=baseline_attention,
        mlp_flops=baseline_mlp,
        prune_method_flops=0.0,
        total_llm_flops=baseline_attention + baseline_mlp,
        non_flop_overhead={},
    )

    estimator = get_method_estimator(prune_spec.strategy_name)
    pruned_schedule = estimator.build_sequence_schedule(
        model_profile=model_profile,
        prefill_seq_len=prefill_seq_len,
        image_token_len=image_token_len,
        prune_spec=prune_spec,
    )
    pruned_attention, pruned_mlp = estimate_schedule_flops(
        pruned_schedule,
        hidden_size=model_profile.hidden_size,
        intermediate_size=model_profile.intermediate_size,
    )
    method_estimate = estimator.estimate_method_flops(
        model_profile=model_profile,
        prefill_seq_len=prefill_seq_len,
        image_token_len=image_token_len,
        prune_spec=prune_spec,
    )
    pruned = FlopsBreakdown(
        self_attention_flops=pruned_attention,
        mlp_flops=pruned_mlp,
        prune_method_flops=method_estimate.prune_method_flops,
        total_llm_flops=pruned_attention + pruned_mlp + method_estimate.prune_method_flops,
        non_flop_overhead=method_estimate.non_flop_overhead,
    )
    reduction_ratio = 0.0
    if baseline.total_llm_flops > 0:
        reduction_ratio = (
            baseline.total_llm_flops - pruned.total_llm_flops
        ) / baseline.total_llm_flops

    return {
        "prefill_seq_len": prefill_seq_len,
        "baseline_schedule": [segment.to_dict() for segment in baseline_schedule],
        "pruned_schedule": [segment.to_dict() for segment in pruned_schedule],
        "baseline": baseline.to_dict(),
        "pruned": pruned.to_dict(),
        "reduction_ratio": reduction_ratio,
    }
