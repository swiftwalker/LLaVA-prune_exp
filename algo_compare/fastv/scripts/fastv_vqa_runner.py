#!/usr/bin/env python3
"""Run FastV official LLaVA code on local VQA-style datasets."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import shortuuid
import torch
from torch import nn
from PIL import Image
from tqdm import tqdm


def _install_transformers_compat() -> None:
    if os.environ.get("FASTV_ALLOW_TOKENIZERS_015", "1") != "1":
        return
    import importlib.metadata

    real_version = importlib.metadata.version

    def compat_version(distribution_name: str) -> str:
        normalized = distribution_name.lower().replace("_", "-")
        if normalized == "tokenizers":
            return "0.13.3"
        return real_version(distribution_name)

    importlib.metadata.version = compat_version


_install_transformers_compat()


def _add_fastv_paths() -> Path:
    repo = Path(os.environ.get("FASTV_OFFICIAL_REPO") or os.getcwd()).resolve()
    algo_src = Path(__file__).resolve().parents[2] / "src"
    candidates = [
        algo_src,
        repo / "src" / "transformers" / "src",
        repo / "src" / "FastV",
        repo / "src" / "LLaVA",
        repo / "src" / "FastV" / "llava-hf" / "transformers" / "src",
        repo,
        Path(os.getcwd()).resolve(),
    ]
    for path in reversed(candidates):
        if path.exists():
            sys.path.insert(0, str(path))
    return repo


FASTV_REPO = _add_fastv_paths()

from algo_compare.vqa_compat import (  # noqa: E402
    SUPPORTED_DATASETS as VQA_DATASETS,
    load_questions as load_compat_questions,
    normalize_item as normalize_compat_item,
    open_image as open_compat_image,
)
from llava.constants import (  # noqa: E402
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import conv_templates  # noqa: E402
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token  # noqa: E402
from llava.model.builder import load_pretrained_model  # noqa: E402
from llava.utils import disable_torch_init  # noqa: E402


FASTV_DIRECT_ATTRS = {
    "use_fast_v",
    "fast_v_sys_length",
    "fast_v_image_token_length",
    "fast_v_attention_rank",
    "fast_v_agg_layer",
    "fast_v_inplace",
    "fast_v_token_mask",
}


def split_list(items: list[dict[str, Any]], n: int) -> list[list[dict[str, Any]]]:
    chunk_size = math.ceil(len(items) / n)
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def get_chunk(items: list[dict[str, Any]], n: int, k: int) -> list[dict[str, Any]]:
    return split_list(items, n)[k]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_questions(dataset: str, path: Path) -> list[dict[str, Any]]:
    return load_compat_questions(dataset, path)


def normalize_item(dataset: str, item: dict[str, Any]) -> dict[str, Any]:
    return normalize_compat_item(dataset, item)


def configure_fastv(
    model: Any,
    args: argparse.Namespace,
    enabled: bool = True,
    sys_length_override: int | None = None,
) -> None:
    sys_length = int(sys_length_override if sys_length_override is not None else args.fastv_sys_length)
    enabled = bool(enabled)
    values = {
        "fastv_k": args.fastv_k,
        "fast_v_k": args.fastv_k,
        "fastv_r": args.fastv_r,
        "fast_v_r": args.fastv_r,
        "fastv_mode": args.fastv_mode,
        "fast_v_mode": args.fastv_mode,
        "fast_v_agg_layer": args.fastv_k,
        "fastv_agg_layer": args.fastv_k,
        "fast_v_attention_rank": args.fastv_attention_rank,
        "fastv_attention_rank": args.fastv_attention_rank,
        "fast_v_image_token_length": args.fastv_image_token_length,
        "fastv_image_token_length": args.fastv_image_token_length,
        "fast_v_sys_length": sys_length,
        "fastv_sys_length": sys_length,
        "use_fast_v": enabled,
        "fast_v_token_mask": enabled and args.fastv_mode == "token_mask",
        "fast_v_inplace": enabled and args.fastv_mode == "inplace",
    }

    targets: list[Any] | None = getattr(model, "_algo_compare_fastv_targets", None)
    if targets is None:
        targets = []
        seen: set[int] = set()

        def add_target(target: Any) -> None:
            if target is None:
                return
            marker = id(target)
            if marker in seen:
                return
            seen.add(marker)
            targets.append(target)

        add_target(model)
        for attr_name in ("model", "base_model"):
            add_target(getattr(model, attr_name, None))
        if hasattr(model, "get_model"):
            try:
                add_target(model.get_model())
            except Exception:
                pass

        if isinstance(model, nn.Module):
            for _, module in model.named_modules():
                if hasattr(module, "reset_fastv") or any(hasattr(module, attr) for attr in FASTV_DIRECT_ATTRS):
                    add_target(module)

        setattr(model, "_algo_compare_fastv_targets", targets)

    for target in targets:
        config = getattr(target, "config", None)
        if config is None:
            continue
        for key, value in values.items():
            setattr(config, key, value)

    for target in targets:
        if hasattr(target, "reset_fastv"):
            target.reset_fastv()

    direct_values = {key: value for key, value in values.items() if key in FASTV_DIRECT_ATTRS}
    for target in targets:
        for key, value in direct_values.items():
            if hasattr(target, key):
                setattr(target, key, value)


def build_prompt_and_image(
    item: dict[str, Any],
    image_folder: Path,
    model_config: Any,
    image_processor: Any,
) -> tuple[str, str, torch.Tensor | None, list[tuple[int, int]] | None]:
    qs = item["text"]
    cur_prompt = qs
    image_tensor = None
    image_sizes = None

    image = open_compat_image(item, image_folder)
    if image is not None:
        image_tensor = process_images([image], image_processor, model_config)[0].unsqueeze(0).half().cuda()
        image_sizes = [image.size]
        if getattr(model_config, "mm_use_im_start_end", False):
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if item.get("single_pred_prompt"):
        suffix = "Answer with the option's letter from the given choices directly."
        qs = qs + "\n" + suffix
        cur_prompt = cur_prompt + "\n" + suffix

    return qs, cur_prompt, image_tensor, image_sizes


def eval_model(args: argparse.Namespace) -> None:
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, args.model_base, model_name)

    questions = load_questions(args.dataset, Path(args.question_file))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = Path(os.path.expanduser(args.answers_file))
    answers_file.parent.mkdir(parents=True, exist_ok=True)

    with answers_file.open("w", encoding="utf-8") as ans_file:
        for raw_item in tqdm(questions):
            item = normalize_item(args.dataset, raw_item)
            qs, cur_prompt, image_tensor, image_sizes = build_prompt_and_image(
                item,
                Path(args.image_folder),
                model.config,
                image_processor,
            )
            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
            image_positions = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=False)
            image_start = int(image_positions[0].item()) if image_positions.numel() else args.fastv_sys_length
            input_token_count = int(input_ids.shape[1])
            has_image = image_tensor is not None
            estimated_expanded_tokens = (
                input_token_count + args.fastv_image_token_length - 1 if has_image else input_token_count
            )
            fastv_enabled = has_image
            fastv_disabled_reason = None
            if (
                fastv_enabled
                and args.fastv_max_expanded_tokens > 0
                and estimated_expanded_tokens > args.fastv_max_expanded_tokens
            ):
                fastv_enabled = False
                fastv_disabled_reason = f"estimated_expanded_tokens>{args.fastv_max_expanded_tokens}"
            elif not fastv_enabled:
                fastv_disabled_reason = "no_image"
            configure_fastv(model, args, enabled=fastv_enabled, sys_length_override=image_start)

            generate_args = {
                "images": image_tensor,
                "do_sample": args.temperature > 0,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "num_beams": args.num_beams,
                "max_new_tokens": args.max_new_tokens,
                "use_cache": True,
                "output_attentions": True,
                "output_scores": True,
                "return_dict_in_generate": True,
            }
            if args.top_p is None:
                generate_args.pop("top_p")
            with torch.inference_mode():
                output_ids = model.generate(input_ids, **generate_args)

            if hasattr(output_ids, "sequences"):
                sequences = output_ids.sequences
            elif isinstance(output_ids, dict):
                sequences = output_ids["sequences"]
            else:
                sequences = output_ids
            if sequences.shape[1] > input_ids.shape[1]:
                sequences = sequences[:, input_ids.shape[1] :]
            outputs = tokenizer.batch_decode(sequences, skip_special_tokens=True)[0].strip()
            ans_file.write(
                json.dumps(
                    {
                        "question_id": item["question_id"],
                        "prompt": cur_prompt,
                        "text": outputs,
                        "answer_id": shortuuid.uuid(),
                        "model_id": model_name,
                        "metadata": {
                            "fastv_k": args.fastv_k,
                            "fastv_r": args.fastv_r,
                            "fastv_attention_rank": args.fastv_attention_rank,
                            "fastv_mode": args.fastv_mode,
                            "fastv_enabled": fastv_enabled,
                            "fastv_disabled_reason": fastv_disabled_reason,
                            "input_token_count": input_token_count,
                            "estimated_expanded_tokens": estimated_expanded_tokens,
                            "fastv_max_expanded_tokens": args.fastv_max_expanded_tokens,
                        },
                    }
                )
                + "\n"
            )
            ans_file.flush()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=list(VQA_DATASETS), required=True)
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--fastv-k", type=int, default=2)
    parser.add_argument("--fastv-r", type=float, default=0.5)
    parser.add_argument("--fastv-attention-rank", type=int, default=288)
    parser.add_argument("--fastv-image-token-length", type=int, default=576)
    parser.add_argument("--fastv-sys-length", type=int, default=35)
    parser.add_argument("--fastv-mode", type=str, default="token_mask")
    parser.add_argument(
        "--fastv-max-expanded-tokens",
        type=int,
        default=900,
        help="Disable FastV per sample when estimated expanded prompt length exceeds this value; <=0 disables fallback.",
    )
    return parser


if __name__ == "__main__":
    eval_model(build_parser().parse_args())
