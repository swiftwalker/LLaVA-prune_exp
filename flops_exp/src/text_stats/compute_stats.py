from text_stats.dataset_registry import load_dataset_config, load_dataset_records
from text_stats.prompt_builder import (
    compute_canonical_template_overhead,
    compute_prompt_lengths,
)
from text_stats.schema import SampleTextRecord, TextLengthStats
from text_stats.tokenizer_utils import load_model_flags, load_tokenizer, tokenizer_display_name
from utils.stats import summarize_numeric


def build_sample_records(
    dataset_config: dict,
    tokenizer,
    conv_mode: str,
    image_token_len: int,
    mm_use_im_start_end: bool,
    sample_limit: int | None = None,
) -> list[SampleTextRecord]:
    rows = load_dataset_records(dataset_config, sample_limit=sample_limit)
    records: list[SampleTextRecord] = []
    dataset_name = dataset_config["dataset_name"]
    text_field = dataset_config.get("text_field", "text")
    image_field = dataset_config.get("image_field", "image")
    id_field = dataset_config.get("id_field", "question_id")

    for sample_idx, row in enumerate(rows):
        raw_text = row[text_field]
        lengths = compute_prompt_lengths(
            tokenizer=tokenizer,
            raw_text=raw_text,
            conv_mode=conv_mode,
            image_token_len=image_token_len,
            mm_use_im_start_end=mm_use_im_start_end,
        )
        records.append(
            SampleTextRecord(
                sample_idx=sample_idx,
                dataset_name=dataset_name,
                question_id=str(row[id_field]),
                image_file=str(row[image_field]),
                raw_text=raw_text,
                raw_text_token_len=lengths.raw_text_token_len,
                template_overhead_tokens=lengths.template_overhead_tokens,
                templated_input_ids_len_pre_mm=lengths.templated_input_ids_len_pre_mm,
                image_token_len=image_token_len,
                prefill_seq_len=lengths.prefill_seq_len,
            )
        )
    return records


def summarize_text_records(
    records: list[SampleTextRecord],
    dataset_name: str,
    tokenizer_name: str,
    conv_mode: str,
    canonical_template_overhead_tokens: int,
    image_token_len: int,
) -> TextLengthStats:
    summary = summarize_numeric(record.raw_text_token_len for record in records)
    return TextLengthStats(
        dataset_name=dataset_name,
        tokenizer_name=tokenizer_name,
        conv_mode=conv_mode,
        sample_count=len(records),
        raw_text_token_len_mean=summary["mean"],
        raw_text_token_len_median=summary["median"],
        raw_text_token_len_p25=summary["p25"],
        raw_text_token_len_p75=summary["p75"],
        raw_text_token_len_min=summary["min"],
        raw_text_token_len_max=summary["max"],
        raw_text_token_len_std=summary["std"],
        template_overhead_tokens=canonical_template_overhead_tokens,
        image_token_len=image_token_len,
    )


def compute_text_length_stats(
    model_path: str,
    tokenizer_path: str | None,
    conv_mode: str,
    image_token_len: int,
    dataset_name: str | None = None,
    dataset_config_path: str | None = None,
    sample_limit: int | None = None,
) -> tuple[TextLengthStats, list[SampleTextRecord], dict]:
    dataset_config = load_dataset_config(dataset_name=dataset_name, config_path=dataset_config_path)
    resolved_tokenizer_path = tokenizer_path or model_path
    tokenizer = load_tokenizer(resolved_tokenizer_path)
    flags = load_model_flags(model_path)
    canonical_overhead = compute_canonical_template_overhead(
        tokenizer=tokenizer,
        conv_mode=conv_mode,
        mm_use_im_start_end=flags["mm_use_im_start_end"],
    )
    records = build_sample_records(
        dataset_config=dataset_config,
        tokenizer=tokenizer,
        conv_mode=conv_mode,
        image_token_len=image_token_len,
        mm_use_im_start_end=flags["mm_use_im_start_end"],
        sample_limit=sample_limit,
    )
    stats = summarize_text_records(
        records=records,
        dataset_name=dataset_config["dataset_name"],
        tokenizer_name=tokenizer_display_name(resolved_tokenizer_path, tokenizer),
        conv_mode=conv_mode,
        canonical_template_overhead_tokens=canonical_overhead,
        image_token_len=image_token_len,
    )
    metadata = {
        "dataset_config": dataset_config,
        "sample_limit": sample_limit,
        "mm_use_im_start_end": flags["mm_use_im_start_end"],
        "observed_template_overhead_tokens": sorted(
            {record.template_overhead_tokens for record in records}
        ),
    }
    return stats, records, metadata

