"""
Inference pipeline for entropy experiment.

Runs LLaVA inference on GQA/MME/POPE with attention capture.
Outputs:
  1. Answer JSONL files (compatible with existing eval scripts)
  2. HDF5 files with per-layer text→vision attention sub-matrices
"""

import argparse
import os
import sys
import json
import time
import datetime
import yaml
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
_cpu_threads = os.environ.get("OMP_NUM_THREADS")
if _cpu_threads is None:
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ.setdefault("MKL_NUM_THREADS", "4")
import torch
if _cpu_threads is None:
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# Add LLaVA root to path
LLAVA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, LLAVA_ROOT)

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from hooks import AttentionCaptureHook, locate_image_tokens

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


class VQADataset(Dataset):
    """Dataset that loads questions and images for VQA inference."""

    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config, conv_mode):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.conv_mode = conv_mode

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]

        if getattr(self.model_config, 'mm_use_im_start_end', False):
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor, image.size

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_inference(config: dict, dataset_name: str, max_samples: int = None):
    """
    Run inference with attention capture on a specific dataset.

    Args:
        config: experiment configuration dict
        dataset_name: one of 'gqa', 'mme', 'pope'
        max_samples: override max samples for debugging
    """
    # Resolve paths relative to LLaVA root
    def resolve(p):
        if os.path.isabs(p):
            return p
        return os.path.join(LLAVA_ROOT, p)

    model_cfg = config["model"]
    infer_cfg = config["inference"]
    capture_cfg = config["capture"]
    ds_cfg = config["datasets"][dataset_name]
    output_cfg = config["output"]

    model_path = resolve(model_cfg["path"])
    model_name = model_cfg["name"]
    question_file = resolve(ds_cfg["question_file"])
    image_folder = resolve(ds_cfg["image_folder"])

    # Output paths
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_dir = resolve(output_cfg["raw_dir"])
    answers_dir = resolve(output_cfg["answers_dir"])
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(answers_dir, exist_ok=True)

    h5_path = os.path.join(raw_dir, f"{dataset_name}_{timestamp}.h5")
    answers_path = os.path.join(answers_dir, f"{dataset_name}_{timestamp}.jsonl")

    # Set seed
    seed = infer_cfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load model
    print(f"Loading model from {model_path} ...")
    disable_torch_init()

    load_kwargs = {}
    attn_impl = model_cfg.get("attn_implementation", "eager")
    if attn_impl == "eager":
        # Force eager attention to get full attention weights
        load_kwargs["attn_implementation"] = "eager"

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name, **load_kwargs
    )
    model.eval()
    print(f"Model loaded. Attention implementation: {attn_impl}")

    # Load questions
    with open(question_file, 'r') as f:
        questions = [json.loads(line) for line in f]

    effective_max = max_samples or capture_cfg.get("max_samples")
    if effective_max is not None:
        questions = questions[:effective_max]
        print(f"Using {len(questions)} samples (max_samples={effective_max})")
    else:
        print(f"Using all {len(questions)} samples")

    # Create dataset & loader (batch_size=1)
    dataset = VQADataset(
        questions, image_folder, tokenizer, image_processor,
        model.config, infer_cfg["conv_mode"]
    )
    num_workers = int(os.environ.get("DATALOADER_NUM_WORKERS", "0"))
    data_loader = DataLoader(
        dataset, batch_size=1, num_workers=num_workers, shuffle=False, collate_fn=collate_fn
    )

    # Open HDF5 and answer file
    with AttentionCaptureHook(h5_path, capture_cfg) as hook:
        hook.save_config_snapshot(config)
        ans_file = open(answers_path, 'w')

        v_token_num = capture_cfg.get("v_token_num", 576)
        total_time = 0.0

        for sample_idx, ((input_ids, image_tensor, image_sizes), line) in enumerate(tqdm(
            zip(data_loader, questions), total=len(questions), desc=f"[{dataset_name}]"
        )):
            question_id = line["question_id"]
            image_file = line["image"]

            # Locate image token position BEFORE multimodal preparation
            v_token_start, _, text_token_start = locate_image_tokens(
                input_ids, IMAGE_TOKEN_INDEX, v_token_num=v_token_num
            )
            # Expected sequence length after replacing one IMAGE token with v_token_num vision tokens
            expected_seq_len = v_token_start + v_token_num + (input_ids.shape[1] - (v_token_start + 1))

            input_ids = input_ids.to(device='cuda', non_blocking=True)
            image_tensor = image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True)

            t0 = time.time()
            with torch.inference_mode():
                outputs = model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=image_sizes,
                    do_sample=False,
                    temperature=infer_cfg["temperature"],
                    top_p=infer_cfg.get("top_p"),
                    num_beams=infer_cfg["num_beams"],
                    max_new_tokens=infer_cfg["max_new_tokens"],
                    use_cache=True,
                    output_attentions=True,
                    return_dict_in_generate=True,
                )
            t1 = time.time()
            total_time += (t1 - t0)

            # Validate multimodal expansion length against prefill attention length
            prefill_seq_len = outputs.attentions[0][0].shape[-1]
            if prefill_seq_len != expected_seq_len:
                raise RuntimeError(
                    f"Expanded sequence length mismatch for question_id={question_id}: "
                    f"expected {expected_seq_len}, got {prefill_seq_len}. "
                    f"Check v_token_num (configured={v_token_num})."
                )

            # Decode answer
            generated_ids = outputs.sequences
            answer_text = tokenizer.batch_decode(
                generated_ids[:, input_ids.shape[1]:],  # only new tokens, but generate may change shape
                skip_special_tokens=True
            )[0].strip()

            # Write answer (compatible with existing eval scripts)
            ans_file.write(json.dumps({
                "question_id": question_id,
                "prompt": line["text"],
                "text": answer_text,
                "model_id": model_name,
                "metadata": {}
            }) + "\n")

            # Save attention data
            metadata = {
                "question_id": question_id,
                "image_file": image_file,
                "answer": answer_text,
                "question": line["text"],
            }

            # Sanitize question_id: replace / with __ to avoid nested HDF5 groups
            safe_qid = str(question_id).replace("/", "__")
            hook.save_sample(
                sample_id=f"sample_{sample_idx:06d}_{safe_qid}",
                attentions=outputs.attentions,
                v_token_start=v_token_start,
                v_token_num=v_token_num,
                text_token_start=text_token_start,
                metadata=metadata,
            )

        ans_file.close()

    avg_time = total_time / len(questions) if questions else 0
    print(f"\nDone! {len(questions)} samples processed.")
    print(f"  Average inference time: {avg_time:.3f}s/sample")
    print(f"  Answers: {answers_path}")
    print(f"  Attention data: {h5_path}")

    return answers_path, h5_path


def main():
    parser = argparse.ArgumentParser(description="Entropy experiment inference pipeline")
    parser.add_argument("--config", type=str, default="entropy_exp/configs/default.yaml",
                        help="Path to config YAML file")
    parser.add_argument("--dataset", type=str, required=True, choices=["gqa", "mme", "pope"],
                        help="Dataset to run inference on")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Override max samples for debugging")
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(LLAVA_ROOT, config_path)

    config = load_config(config_path)
    run_inference(config, args.dataset, args.max_samples)


if __name__ == "__main__":
    main()
