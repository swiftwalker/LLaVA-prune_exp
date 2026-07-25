#!/usr/bin/env python3
"""Run pinned official CDPruner on local LLaVA-NeXT VQA datasets."""

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
from tqdm import tqdm


def _add_import_paths() -> None:
    official_repo = Path(os.environ.get("CDPRUNER_OFFICIAL_REPO") or os.getcwd()).resolve()
    algo_src = Path(__file__).resolve().parents[2] / "src"
    sys.path.insert(0, str(official_repo))
    sys.path.insert(1, str(algo_src))


_add_import_paths()

from algo_compare.llava_next_official import (  # noqa: E402
    configure_cdpruner_model,
    generate_with_cdpruner,
    image_crop_count,
    install_cdpruner_next_image_adapter,
    is_cdpruner_mistral,
    is_llava_next_config,
    llava_language_backbone,
)
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
import llava.mm_utils as llava_mm_utils  # noqa: E402
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token  # noqa: E402
from llava.model.builder import load_pretrained_model  # noqa: E402
from llava.utils import disable_torch_init  # noqa: E402


def split_list(items: list[dict[str, Any]], chunks: int) -> list[list[dict[str, Any]]]:
    if chunks <= 0:
        raise ValueError(f"num_chunks must be positive, got {chunks}")
    if not items:
        return [[]]
    chunk_size = math.ceil(len(items) / chunks)
    return [items[index : index + chunk_size] for index in range(0, len(items), chunk_size)]


def get_chunk(items: list[dict[str, Any]], chunks: int, chunk_idx: int) -> list[dict[str, Any]]:
    parts = split_list(items, chunks)
    if chunk_idx < 0 or chunk_idx >= len(parts):
        raise ValueError(f"chunk_idx {chunk_idx} is outside [0, {len(parts)})")
    return parts[chunk_idx]


def build_prompt_and_image(
    item: dict[str, Any],
    image_folder: Path,
    model_config: Any,
    image_processor: Any,
) -> tuple[str, str, torch.Tensor | None, list[tuple[int, int]] | None]:
    question = str(item["text"])
    prompt_text = question
    image_tensor = None
    image_sizes = None

    image = open_compat_image(item, image_folder)
    if image is not None:
        image_tensor = process_images([image], image_processor, model_config)[0].unsqueeze(0).half().cuda()
        image_sizes = [image.size]
        if getattr(model_config, "mm_use_im_start_end", False):
            prompt_text = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + prompt_text
        else:
            prompt_text = DEFAULT_IMAGE_TOKEN + "\n" + prompt_text

    if item.get("single_pred_prompt"):
        suffix = "Answer with the option's letter from the given choices directly."
        prompt_text += "\n" + suffix
        question += "\n" + suffix

    return prompt_text, question, image_tensor, image_sizes


def _requested_model_is_mistral(model_path: str, model_name: str) -> bool:
    if "mistral" in model_name.lower():
        return True
    config_path = Path(model_path) / "config.json"
    if not config_path.is_file():
        return False
    with config_path.open("r", encoding="utf-8") as handle:
        return str(json.load(handle).get("model_type", "")).lower() == "llava_mistral"


def eval_model(args: argparse.Namespace) -> None:
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    load_kwargs: dict[str, Any] = {}
    if not _requested_model_is_mistral(model_path, model_name):
        load_kwargs["visual_token_num"] = args.visual_token_num
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path,
        args.model_base,
        model_name,
        **load_kwargs,
    )
    configure_cdpruner_model(model, args.visual_token_num)
    mistral_bridge = is_cdpruner_mistral(model)
    language_backbone = llava_language_backbone(model)
    use_next_adapter = args.llava_next_compat == "on" or (
        args.llava_next_compat == "auto" and is_llava_next_config(model.config)
    )
    if use_next_adapter:
        install_cdpruner_next_image_adapter(llava_mm_utils)

    questions = load_compat_questions(args.dataset, Path(args.question_file))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    if args.max_samples is not None:
        questions = questions[: args.max_samples]

    answers_file = Path(os.path.expanduser(args.answers_file))
    answers_file.parent.mkdir(parents=True, exist_ok=True)
    image_folder = Path(args.image_folder)

    with answers_file.open("w", encoding="utf-8") as answer_stream:
        progress = tqdm(questions)
        for raw_item in progress:
            item = normalize_compat_item(args.dataset, raw_item)
            prompt_text, selection_text, image_tensor, image_sizes = build_prompt_and_image(
                item,
                image_folder,
                model.config,
                image_processor,
            )
            conversation = conv_templates[args.conv_mode].copy()
            conversation.append_message(conversation.roles[0], prompt_text)
            conversation.append_message(conversation.roles[1], None)
            prompt = conversation.get_prompt()
            input_ids = tokenizer_image_token(
                prompt,
                tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors="pt",
            ).unsqueeze(0).cuda()

            generate_args: dict[str, Any] = {
                "do_sample": args.temperature > 0,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "num_beams": args.num_beams,
                "max_new_tokens": args.max_new_tokens,
                "use_cache": True,
            }
            if args.top_p is None:
                generate_args.pop("top_p")

            with torch.inference_mode():
                output_ids, effective_visual_tokens = generate_with_cdpruner(
                    model,
                    input_ids,
                    images=image_tensor,
                    image_sizes=image_sizes,
                    texts=selection_text.replace(
                        "\nAnswer the question using a single word or phrase.",
                        "",
                    ),
                    **generate_args,
                )

            if hasattr(output_ids, "sequences"):
                output_ids = output_ids.sequences
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            crop_count = image_crop_count(image_tensor)
            progress.set_postfix(vtn=effective_visual_tokens, crops=crop_count)
            answer_stream.write(
                json.dumps(
                    {
                        "question_id": item["question_id"],
                        "prompt": str(item["text"]),
                        "text": outputs,
                        "answer_id": shortuuid.uuid(),
                        "model_id": model_name,
                        "metadata": {
                            "cdpruner_enabled": image_tensor is not None,
                            "cdpruner_visual_token_num_per_crop": int(args.visual_token_num),
                            "cdpruner_visual_token_num_effective": int(effective_visual_tokens),
                            "cdpruner_crop_count": crop_count,
                            "cdpruner_expected_crop_product": crop_count * int(args.visual_token_num),
                            "cdpruner_llava_next_mistral_bridge": mistral_bridge,
                            "cdpruner_language_backbone": language_backbone,
                            "cdpruner_generation_path": (
                                "mistral_bridge" if mistral_bridge else "official_llama_native"
                            ),
                            "cdpruner_llava_next_compat": use_next_adapter,
                            "cdpruner_image_adapter": (
                                "canonical_anyres_resolution" if use_next_adapter else "official_fixed_672"
                            ),
                            "cdpruner_official_unpad_disabled": "unpad"
                            in str(getattr(model.config, "mm_patch_merge_type", "")),
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            answer_stream.flush()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=list(VQA_DATASETS), required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--image-folder", default="")
    parser.add_argument("--question-file", required=True)
    parser.add_argument("--answers-file", required=True)
    parser.add_argument("--conv-mode", default="vicuna_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--visual-token-num", type=int, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--llava-next-compat", choices=("auto", "on", "off"), default="auto")
    return parser


if __name__ == "__main__":
    eval_model(build_parser().parse_args())
