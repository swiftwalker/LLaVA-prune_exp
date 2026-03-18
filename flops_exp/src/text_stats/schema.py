from dataclasses import asdict, dataclass


@dataclass
class SampleTextRecord:
    sample_idx: int
    dataset_name: str
    question_id: str
    image_file: str
    raw_text: str
    raw_text_token_len: int
    template_overhead_tokens: int
    templated_input_ids_len_pre_mm: int
    image_token_len: int
    prefill_seq_len: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TextLengthStats:
    dataset_name: str
    tokenizer_name: str
    conv_mode: str
    sample_count: int
    raw_text_token_len_mean: float
    raw_text_token_len_median: float
    raw_text_token_len_p25: float
    raw_text_token_len_p75: float
    raw_text_token_len_min: int
    raw_text_token_len_max: int
    raw_text_token_len_std: float
    template_overhead_tokens: int
    image_token_len: int

    def to_dict(self) -> dict:
        return asdict(self)

