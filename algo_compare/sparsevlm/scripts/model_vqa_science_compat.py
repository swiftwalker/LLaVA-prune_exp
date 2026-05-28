#!/usr/bin/env python3
"""ScienceQA runner compatible with the official SparseVLM checkout.

This mirrors the official ``llava.eval.model_vqa_science`` input and prompt
handling, but does not pass ``retained_tokens`` into ``model.generate``. The
official SparseVLM pruning code already reads the retain-token budget from the
``RETAIN_TOKN`` environment variable; passing it again as a generation kwarg
breaks with the transformers version in this workspace.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import shortuuid
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.getcwd())

from llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


def split_list(items: list[dict], n: int) -> list[list[dict]]:
    chunk_size = math.ceil(len(items) / n)
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def get_chunk(items: list[dict], n: int, k: int) -> list[dict]:
    return split_list(items, n)[k]


def eval_model(args: argparse.Namespace) -> None:
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, args.model_base, model_name)

    with open(os.path.expanduser(args.question_file), "r", encoding="utf-8") as f:
        questions = json.load(f)
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    answers_file = Path(os.path.expanduser(args.answers_file))
    answers_file.parent.mkdir(parents=True, exist_ok=True)

    with answers_file.open("w", encoding="utf-8") as ans_file:
        for line in tqdm(questions):
            idx = line["id"]
            question = line["conversations"][0]
            qs = question["value"].replace("<image>", "").strip()
            cur_prompt = qs

            if "image" in line:
                image = Image.open(os.path.join(args.image_folder, line["image"])).convert("RGB")
                image_tensor = process_images([image], image_processor, model.config)[0]
                images = image_tensor.unsqueeze(0).half().cuda()
                image_sizes = [image.size]
                if getattr(model.config, "mm_use_im_start_end", False):
                    qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
                else:
                    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
                cur_prompt = "<image>" + "\n" + cur_prompt
            else:
                images = None
                image_sizes = None

            if args.single_pred_prompt:
                suffix = "Answer with the option's letter from the given choices directly."
                qs = qs + "\n" + suffix
                cur_prompt = cur_prompt + "\n" + suffix

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

            generate_kwargs = {
                "images": images,
                "image_sizes": image_sizes,
                "do_sample": args.temperature > 0,
                "temperature": args.temperature,
                "max_new_tokens": args.max_new_tokens,
                "use_cache": True,
            }
            with torch.inference_mode():
                output_ids = model.generate(input_ids, **generate_kwargs)

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            ans_file.write(
                json.dumps(
                    {
                        "question_id": idx,
                        "prompt": cur_prompt,
                        "text": outputs,
                        "answer_id": shortuuid.uuid(),
                        "model_id": model_name,
                        "metadata": {},
                    }
                )
                + "\n"
            )
            ans_file.flush()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.json")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v0")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--answer-prompter", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--retained_tokens", type=int, default=192)
    parser.add_argument("--max-new-tokens", "--max_new_tokens", dest="max_new_tokens", type=int, default=1024)
    return parser


if __name__ == "__main__":
    eval_model(build_parser().parse_args())
