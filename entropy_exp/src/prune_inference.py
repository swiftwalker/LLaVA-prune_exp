"""
Pruning inference pipeline for LLaVA.

Runs inference on GQA/MME/POPE with visual token pruning.
Outputs:
  1. Answer JSONL files (compatible with existing eval scripts)
  2. Pruning statistics JSONL (per-sample pruning details)

Usage:
    python entropy_exp/src/prune_inference.py \
        --config entropy_exp/configs/prune.yaml \
        --dataset mme \
        [--max-samples 10]
"""

import argparse
import os
import sys
import json
import time
import datetime
import yaml
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
import h5py
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# Add LLaVA root to path
LLAVA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, LLAVA_ROOT)
# Add src/ to path so strategy/pruner imports work
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from hooks import locate_image_tokens
from pruner import VisualTokenPruner
from strategies import get_strategy

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


# ---------------------------------------------------------------------------
# Dataset (same as inference.py)
# ---------------------------------------------------------------------------
class VQADataset(Dataset):
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
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def _parse_value(raw: str):
    """Parse a CLI override value string into a Python object.

    Supports: int, float, bool, null/None, JSON lists, and plain strings.
    """
    stripped = raw.strip()
    if stripped.lower() in ("null", "none", "~"):
        return None
    if stripped.lower() == "true":
        return True
    if stripped.lower() == "false":
        return False
    # JSON array: [1, 2, 3]
    if stripped.startswith("["):
        return json.loads(stripped)
    # Try int / float
    try:
        return int(stripped)
    except ValueError:
        pass
    try:
        return float(stripped)
    except ValueError:
        pass
    return stripped


def apply_overrides(config: dict, overrides: list) -> dict:
    """Apply dot-separated key=value overrides to a nested config dict.

    Examples:
        --set pruning.strategy=entropy
        --set pruning.prune_layers=[2,5,10]
        --set pruning.prune_ratio=[0.3,0.4,0.5]
        --set pruning.entropy.dynamic_ratio=true
        --set inference.max_new_tokens=256
    """
    import copy
    config = copy.deepcopy(config)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override format (expected key=value): {item!r}")
        key_path, raw_value = item.split("=", 1)
        keys = key_path.strip().split(".")
        value = _parse_value(raw_value)

        d = config
        for k in keys[:-1]:
            if k not in d or not isinstance(d[k], dict):
                d[k] = {}
            d = d[k]
        d[keys[-1]] = value
    return config


# ---------------------------------------------------------------------------
# Main inference loop with pruning
# ---------------------------------------------------------------------------
def run_prune_inference(
    config: dict,
    dataset_name: str,
    max_samples: int = None,
    run_mode: str = "prune",
):
    """
    Run inference with visual token pruning on a specific dataset.
    """

    def resolve(p):
        if os.path.isabs(p):
            return p
        return os.path.join(LLAVA_ROOT, p)

    model_cfg = config["model"]
    infer_cfg = config["inference"]
    prune_cfg = config["pruning"]
    ds_cfg = config["datasets"][dataset_name]
    output_cfg = config["output"]

    if run_mode not in {"prune", "baseline"}:
        raise ValueError(f"Unknown run_mode: {run_mode}")

    strategy_name = prune_cfg["strategy"]
    strategy_extra = prune_cfg.get(strategy_name, {})
    strategy_config = {**prune_cfg, **strategy_extra}
    strategy = get_strategy(strategy_name, strategy_config)

    model_path = resolve(model_cfg["path"])
    model_name = model_cfg["name"]
    question_file = resolve(ds_cfg["question_file"])
    image_folder = resolve(ds_cfg["image_folder"])

    # --- output paths: per-run directory ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = "baseline" if run_mode == "baseline" else strategy_name
    run_name = f"{dataset_name}_{run_tag}_{timestamp}"
    base_dir = resolve(output_cfg.get("base_dir", "entropy_exp/outputs"))
    run_dir = os.path.join(base_dir, "runs", run_name)
    os.makedirs(run_dir, exist_ok=True)

    answers_path = os.path.join(run_dir, "answers.jsonl")
    stats_path = os.path.join(run_dir, "stats.jsonl")
    capture_cfg = config.get("capture", {})
    save_attention = capture_cfg.get("save_attention", False)
    save_importance = capture_cfg.get("save_importance_scores", False)
    save_indices = capture_cfg.get("save_keep_indices", False)
    captures_h5_path = os.path.join(run_dir, "captures.h5") if save_attention else None

    # Parse capture_layers: "all" -> None (means all), list -> set
    capture_layers_cfg = capture_cfg.get("capture_layers", "all")
    if capture_layers_cfg == "all" or capture_layers_cfg is None:
        capture_layers_set = None  # None means all layers
    else:
        capture_layers_set = set(capture_layers_cfg)

    # --- save config snapshot (after --set overrides) ---
    config_snapshot = {
        **config,
        "_run_meta": {
            "run_mode": run_mode,
            "dataset": dataset_name,
            "max_samples": max_samples,
            "timestamp": timestamp,
            "run_dir": run_dir,
        },
    }
    config_snapshot_path = os.path.join(run_dir, "config.yaml")
    with open(config_snapshot_path, 'w') as f:
        yaml.dump(config_snapshot, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    # --- seed ---
    seed = infer_cfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # --- load model ---
    print(f"Loading model from {model_path} …")
    disable_torch_init()
    load_kwargs = {}
    attn_impl = model_cfg.get("attn_implementation", "eager")
    if attn_impl == "eager":
        load_kwargs["attn_implementation"] = "eager"

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name, **load_kwargs
    )
    model.eval()
    print(f"Model loaded.  attn={attn_impl}")

    # --- build pruner ---
    pruner = VisualTokenPruner(model, strategy, prune_cfg)
    effective_ratio_map = {str(k): float(v) for k, v in pruner.layer_ratio_map.items()}
    print(f"Pruner: strategy={strategy_name}, prune_layers={pruner.prune_layers}, "
          f"base_ratio={prune_cfg.get('prune_ratio', 0.5)}")

    # --- load questions ---
    with open(question_file, 'r') as f:
        questions = [json.loads(line) for line in f]

    effective_max = max_samples or prune_cfg.get("max_samples")
    if effective_max is not None:
        questions = questions[:effective_max]
    print(f"Using {len(questions)} samples")

    # --- dataset & loader ---
    dataset = VQADataset(
        questions, image_folder, tokenizer, image_processor,
        model.config, infer_cfg["conv_mode"],
    )
    data_loader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False, collate_fn=collate_fn)

    v_token_num = prune_cfg.get("v_token_num", 576)
    eos_token_id = tokenizer.eos_token_id or 2

    total_time = 0.0
    all_stats = []
    ans_file = open(answers_path, 'w')
    h5_file = h5py.File(captures_h5_path, "w") if captures_h5_path else None
    progress_tag = run_tag

    for sample_idx, ((input_ids, image_tensor, image_sizes), line) in enumerate(
        tqdm(zip(data_loader, questions), total=len(questions), desc=f"[{dataset_name}|{progress_tag}]")
    ):
        question_id = line["question_id"]
        image_file = line["image"]

        v_token_start, _, text_token_start = locate_image_tokens(
            input_ids, IMAGE_TOKEN_INDEX, v_token_num=v_token_num
        )
        expected_seq_len = v_token_start + v_token_num + (input_ids.shape[1] - (v_token_start + 1))

        # --- prepare multimodal embeddings ---
        input_ids_cuda = input_ids.to(device='cuda', non_blocking=True)
        image_tensor_cuda = image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True)

        (
            _input_ids,
            position_ids,
            attention_mask,
            _,
            inputs_embeds,
            _,
        ) = model.prepare_inputs_labels_for_multimodal(
            input_ids_cuda, None, None, None, None,
            image_tensor_cuda, image_sizes=list(image_sizes),
        )
        if inputs_embeds.shape[1] != expected_seq_len:
            raise RuntimeError(
                f"Expanded sequence length mismatch for question_id={question_id}: "
                f"expected {expected_seq_len}, got {inputs_embeds.shape[1]}. "
                f"Check v_token_num (configured={v_token_num})."
            )

        # --- generate with pruning ---
        t0 = time.time()
        generated_ids, prune_info = pruner.pruned_generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            v_token_start=v_token_start,
            v_token_num=v_token_num,
            text_token_start=text_token_start,
            max_new_tokens=infer_cfg["max_new_tokens"],
            eos_token_id=eos_token_id,
            save_tv_attn=save_attention,
            capture_layers=capture_layers_set,
        )
        t1 = time.time()
        total_time += (t1 - t0)

        answer_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

        # --- write answer ---
        ans_file.write(json.dumps({
            "question_id": question_id,
            "prompt": line["text"],
            "text": answer_text,
            "model_id": model_name,
            "metadata": {},
        }) + "\n")

        # --- record pruning stats ---
        sample_stats = {
            "sample_idx": sample_idx,
            "run_mode": run_mode,
            "strategy_requested": strategy_name,
            "effective_prune_ratio_map": effective_ratio_map,
            "prune_layers": pruner.prune_layers,
            "question_id": str(question_id),
            "image_file": image_file,
            "answer": answer_text,
            "original_seq_len": prune_info["original_seq_len"],
            "final_seq_len": prune_info["final_seq_len"],
            "num_generated_tokens": prune_info["num_generated_tokens"],
            "prefill_time": prune_info["prefill_time"],
            "decode_time": prune_info["decode_time"],
            "total_time": prune_info["total_time"],
        }

        # Per-layer pruning details
        for layer_idx, linfo in prune_info.get("layers", {}).items():
            # Skip capture-only layers (no pruning stats)
            if "prune_ratio" not in linfo:
                continue
            sample_stats[f"layer_{layer_idx}_ratio"] = linfo["prune_ratio"]
            sample_stats[f"layer_{layer_idx}_before"] = linfo["num_visual_before"]
            sample_stats[f"layer_{layer_idx}_after"] = linfo["num_visual_after"]
            sample_stats[f"layer_{layer_idx}_pruned"] = linfo["num_pruned"]

            if save_importance:
                sample_stats[f"layer_{layer_idx}_importance"] = linfo["importance_scores"].tolist()
            if save_indices:
                sample_stats[f"layer_{layer_idx}_keep_indices"] = linfo["keep_indices"].tolist()

        # Write attention captures to HDF5 (same format as Phase 1)
        if h5_file is not None:
            sample_id = f"{sample_idx:06d}_{str(question_id).replace('/', '__')}"
            grp = h5_file.create_group(sample_id)
            grp.attrs["question_id"] = str(question_id)
            grp.attrs["image_file"] = str(image_file)
            grp.attrs["v_token_start"] = v_token_start
            grp.attrs["v_token_num"] = v_token_num
            grp.attrs["text_token_start"] = text_token_start

            use_fp16 = capture_cfg.get("precision", "fp16") == "fp16"
            compression = capture_cfg.get("compression", "gzip")
            compression_opts = capture_cfg.get("compression_opts", 4)

            for layer_idx, linfo in prune_info.get("layers", {}).items():
                if "tv_attn" not in linfo:
                    continue
                tv = linfo["tv_attn"]  # [H, L_t, L_v] CPU tensor
                if use_fp16:
                    tv = tv.half()
                layer_grp = grp.create_group(f"layer_{layer_idx}")
                layer_grp.create_dataset(
                    "tv_attn", data=tv.numpy(),
                    compression=compression, compression_opts=compression_opts,
                )
                # Also save prune scores for Phase 1 compatibility
                if "importance_scores" in linfo:
                    scores = linfo["importance_scores"]
                    if hasattr(scores, 'cpu'):
                        scores = scores.cpu().numpy()
                    layer_grp.create_dataset(
                        "prune_scores", data=np.asarray(scores, dtype=np.float32),
                    )
            h5_file.flush()

        all_stats.append(sample_stats)

    ans_file.close()
    if h5_file is not None:
        h5_file.close()

    # --- write stats ---
    with open(stats_path, 'w') as f:
        for s in all_stats:
            f.write(json.dumps(s) + "\n")

    avg_time = total_time / len(questions) if questions else 0

    # --- summary ---
    prune_layer = pruner.prune_layers[0] if pruner.prune_layers else -1
    avg_pruned = 0
    if all_stats:
        key = f"layer_{prune_layer}_pruned"
        vals = [s.get(key, 0) for s in all_stats]
        avg_pruned = sum(vals) / len(vals)

    print(f"\nDone! {len(questions)} samples processed.")
    print(f"  Run mode:          {run_mode}")
    print(f"  Strategy:          {strategy_name}")
    print(f"  Prune layer(s):    {pruner.prune_layers}")
    print(f"  Avg tokens pruned: {avg_pruned:.1f} / {v_token_num}")
    print(f"  Avg time/sample:   {avg_time:.3f}s")
    print(f"  Run dir:  {run_dir}")
    print(f"  Answers:  {answers_path}")
    print(f"  Stats:    {stats_path}")
    if captures_h5_path:
        print(f"  Captures: {captures_h5_path}")

    return run_dir, answers_path, stats_path


# ---------------------------------------------------------------------------
# Baseline (no pruning) for fair comparison
# ---------------------------------------------------------------------------
def run_baseline_inference(config: dict, dataset_name: str, max_samples: int = None):
    """
    Run inference WITHOUT pruning using the same custom decode loop,
    so timing comparison is fair (same overhead).
    """
    # temporarily disable pruning
    config = {**config, "pruning": {**config["pruning"], "prune_ratio": 0.0}}
    return run_prune_inference(config, dataset_name, max_samples, run_mode="baseline")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Pruning inference pipeline")
    parser.add_argument("--config", type=str, default="entropy_exp/configs/prune.yaml")
    parser.add_argument("--dataset", type=str, required=True, choices=["gqa", "mme", "pope"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--baseline", action="store_true",
                        help="Run baseline (no pruning) for comparison")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY=VALUE",
                        help="Override config values, e.g. --set pruning.strategy=entropy "
                             "--set pruning.prune_layers=[2,5] --set pruning.prune_ratio=[0.3,0.5]")
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(LLAVA_ROOT, config_path)

    config = load_config(config_path)

    if args.overrides:
        config = apply_overrides(config, args.overrides)
        print(f"Config overrides applied: {args.overrides}")

    if args.baseline:
        run_baseline_inference(config, args.dataset, args.max_samples)
    else:
        run_prune_inference(config, args.dataset, args.max_samples)


if __name__ == "__main__":
    main()
