#!/usr/bin/env python3
"""Run official VisionZip on local VQA-style datasets."""

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
from PIL import Image
from tqdm import tqdm
from types import MethodType


def _add_runtime_paths() -> None:
    official_repo = Path(os.environ.get("VISIONZIP_OFFICIAL_REPO") or os.getcwd()).resolve()
    llava_root = Path(os.environ.get("VISIONZIP_LLAVA_ROOT") or Path(__file__).resolve().parents[3]).resolve()
    algo_src = Path(__file__).resolve().parents[2] / "src"
    for path in (algo_src, llava_root, official_repo):
        if path.exists():
            sys.path.insert(0, str(path))


_add_runtime_paths()

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
from visionzip import visionzip  # noqa: E402
from visionzip.clip_encoder import CLIPVisionTower_VisionZip  # noqa: E402


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
        cur_prompt = "<image>\n" + cur_prompt if item.get("single_pred_prompt") else cur_prompt

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
    model = visionzip(model, dominant=args.visionzip_dominant, contextual=args.visionzip_contextual)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.forward = MethodType(CLIPVisionTower_VisionZip.forward, vision_tower)

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

            generate_args = {
                "images": image_tensor,
                "image_sizes": image_sizes,
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
                            "visionzip_dominant": args.visionzip_dominant,
                            "visionzip_contextual": args.visionzip_contextual,
                            "visionzip_retained_visual_tokens": args.visionzip_dominant + args.visionzip_contextual,
                            "visionzip_enabled": image_tensor is not None,
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
    parser.add_argument("--visionzip-dominant", type=int, default=54)
    parser.add_argument("--visionzip-contextual", type=int, default=10)
    return parser


if __name__ == "__main__":
    eval_model(build_parser().parse_args())
