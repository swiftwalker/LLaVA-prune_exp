from dataclasses import dataclass

from llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token


@dataclass
class PromptLengths:
    raw_text_token_len: int
    template_overhead_tokens: int
    templated_input_ids_len_pre_mm: int
    image_token_len: int
    prefill_seq_len: int


def build_question_text(raw_text: str, mm_use_im_start_end: bool) -> str:
    if mm_use_im_start_end:
        return DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + raw_text
    return DEFAULT_IMAGE_TOKEN + "\n" + raw_text


def build_prompt(raw_text: str, conv_mode: str, mm_use_im_start_end: bool) -> str:
    question = build_question_text(raw_text, mm_use_im_start_end)
    conversation = conv_templates[conv_mode].copy()
    conversation.append_message(conversation.roles[0], question)
    conversation.append_message(conversation.roles[1], None)
    return conversation.get_prompt()


def tokenize_raw_text(tokenizer, raw_text: str) -> int:
    return len(tokenizer(raw_text, add_special_tokens=False).input_ids)


def tokenize_prompt_pre_mm(tokenizer, prompt: str):
    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    )
    return input_ids


def compute_prompt_lengths(
    tokenizer,
    raw_text: str,
    conv_mode: str,
    image_token_len: int,
    mm_use_im_start_end: bool,
) -> PromptLengths:
    raw_text_token_len = tokenize_raw_text(tokenizer, raw_text)
    prompt = build_prompt(raw_text, conv_mode, mm_use_im_start_end)
    input_ids = tokenize_prompt_pre_mm(tokenizer, prompt)
    templated_len = int(input_ids.shape[0])
    template_overhead_tokens = templated_len - raw_text_token_len
    prefill_seq_len = templated_len - 1 + image_token_len
    return PromptLengths(
        raw_text_token_len=raw_text_token_len,
        template_overhead_tokens=template_overhead_tokens,
        templated_input_ids_len_pre_mm=templated_len,
        image_token_len=image_token_len,
        prefill_seq_len=prefill_seq_len,
    )


def compute_canonical_template_overhead(
    tokenizer,
    conv_mode: str,
    mm_use_im_start_end: bool,
) -> int:
    prompt = build_prompt("", conv_mode, mm_use_im_start_end)
    input_ids = tokenize_prompt_pre_mm(tokenizer, prompt)
    return int(input_ids.shape[0])

