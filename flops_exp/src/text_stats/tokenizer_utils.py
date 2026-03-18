from pathlib import Path

from transformers import AutoConfig, AutoTokenizer


def load_tokenizer(tokenizer_path: str):
    return AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)


def load_model_flags(model_path: str) -> dict:
    config = AutoConfig.from_pretrained(model_path)
    return {
        "mm_use_im_start_end": bool(getattr(config, "mm_use_im_start_end", False)),
    }


def tokenizer_display_name(tokenizer_path: str, tokenizer) -> str:
    name_or_path = getattr(tokenizer, "name_or_path", None)
    if name_or_path:
        return str(name_or_path)
    return str(Path(tokenizer_path))

