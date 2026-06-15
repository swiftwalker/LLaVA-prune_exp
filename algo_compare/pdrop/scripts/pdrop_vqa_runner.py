#!/usr/bin/env python3
"""Run official PyramidDrop on local VQA-style datasets with benchmark stats."""

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


def _add_runtime_paths() -> None:
    official_repo = Path(os.environ.get("PDROP_OFFICIAL_REPO") or os.getcwd()).resolve()
    algo_src = Path(__file__).resolve().parents[2] / "src"
    for path in (algo_src, official_repo):
        if path.exists():
            sys.path.insert(0, str(path))


_add_runtime_paths()

from algo_compare.benchmark import (  # noqa: E402
    BenchmarkRecorder,
    add_benchmark_args,
    continuation_sequences,
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
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token  # noqa: E402
from llava.model.builder import load_pretrained_model  # noqa: E402
from llava.utils import disable_torch_init  # noqa: E402


def split_list(items: list[dict[str, Any]], n: int) -> list[list[dict[str, Any]]]:
    chunk_size = math.ceil(len(items) / n)
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def get_chunk(items: list[dict[str, Any]], n: int, k: int) -> list[dict[str, Any]]:
    return split_list(items, n)[k]


def load_questions(dataset: str, path: Path) -> list[dict[str, Any]]:
    return load_compat_questions(dataset, path)


def normalize_item(dataset: str, item: dict[str, Any]) -> dict[str, Any]:
    return normalize_compat_item(dataset, item)


def configure_pdrop_model(model: Any, args: argparse.Namespace) -> None:
    if type(model).__name__ != "LlavaLlamaForCausalLM_PDrop":
        return
    model.model.layer_list = list(eval(args.layer_list))
    model.model.image_token_ratio_list = list(eval(args.image_token_ratio_list))
    model.model.image_token_ratio_list.insert(0, 1.0)


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
    pdrop_infer = bool(args.pdrop_infer or args.layer_list)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path,
        args.model_base,
        model_name,
        pdrop_infer,
    )
    configure_pdrop_model(model, args)

    questions = load_questions(args.dataset, Path(args.question_file))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    if args.max_samples is not None:
        questions = questions[: args.max_samples]
    answers_file = Path(os.path.expanduser(args.answers_file))
    answers_file.parent.mkdir(parents=True, exist_ok=True)
    benchmark = BenchmarkRecorder.from_args(args, answers_file=answers_file, method="pdrop", dataset=args.dataset)

    with answers_file.open("w", encoding="utf-8") as ans_file:
        for sample_idx, raw_item in enumerate(tqdm(questions)):
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
            bench_token = benchmark.start_sample()
            with torch.inference_mode():
                output_ids = model.generate(input_ids, **generate_args)
            sequences = continuation_sequences(output_ids, input_ids)
            benchmark.finish_sample(
                bench_token,
                sample_idx=sample_idx,
                question_id=item["question_id"],
                input_token_count=int(input_ids.shape[1]),
                has_image=image_tensor is not None,
                num_generated_tokens=int(sequences.shape[1]),
                metadata={
                    "pdrop_layer_list": args.layer_list,
                    "pdrop_image_token_ratio_list": args.image_token_ratio_list,
                    "pdrop_infer": pdrop_infer,
                    "pdrop_enabled": image_tensor is not None,
                },
            )
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
                            "pdrop_layer_list": args.layer_list,
                            "pdrop_image_token_ratio_list": args.image_token_ratio_list,
                            "pdrop_infer": pdrop_infer,
                            "pdrop_enabled": image_tensor is not None,
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
    parser.add_argument("--max_new_tokens", "--max-new-tokens", dest="max_new_tokens", type=int, default=128)
    parser.add_argument("--layer_list", type=str, default="[8,16,24]")
    parser.add_argument("--image_token_ratio_list", type=str, default="[0.5,0.25,0.125]")
    parser.add_argument("--pdrop_infer", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--lang", default="en")
    add_benchmark_args(parser)
    return parser


if __name__ == "__main__":
    eval_model(build_parser().parse_args())
