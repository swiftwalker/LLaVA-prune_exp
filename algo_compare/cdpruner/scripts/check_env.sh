#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METHOD_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$METHOD_DIR/../.." && pwd)"
OFFICIAL_REPO="${OFFICIAL_REPO:-$METHOD_DIR/third_party/CDPruner}"
MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/entropy_exp/models/llava-v1.6-vicuna-7b}"
PYTHON_BIN="${PYTHON_BIN:-/data_ssd/liuyu/.conda/envs/llava-next/bin/python}"
status=0

check_path() {
  local label="$1"
  local path="$2"
  if [[ -e "$path" ]]; then
    echo "OK      $label: $path"
  else
    echo "MISSING $label: $path"
    status=1
  fi
}

check_path "official repo" "$OFFICIAL_REPO/.git"
check_path "official CDPruner llava_arch" "$OFFICIAL_REPO/llava/model/llava_arch.py"
check_path "official Llama model" "$OFFICIAL_REPO/llava/model/language_model/llava_llama.py"
check_path "official CLIP tower" "$OFFICIAL_REPO/llava/model/multimodal_encoder/clip_encoder.py"
check_path "model path" "$MODEL_PATH"
check_path "GQA questions" "$REPO_ROOT/entropy_exp/eval_questions/gqa/llava_gqa_testdev_balanced.jsonl"
check_path "GQA images" "$REPO_ROOT/entropy_exp/datasets/gqa/images"
check_path "MME questions" "$REPO_ROOT/entropy_exp/eval_questions/MME/llava_mme.jsonl"
check_path "MME data" "$REPO_ROOT/entropy_exp/datasets/MME_Benchmark_release_version"
check_path "POPE questions" "$REPO_ROOT/entropy_exp/eval_questions/pope/llava_pope_test.jsonl"
check_path "POPE images" "$REPO_ROOT/entropy_exp/datasets/pope/val2014"
check_path "TextVQA questions" "$REPO_ROOT/entropy_exp/eval_questions/textvqa/llava_textvqa_val_v051_ocr.jsonl"
check_path "TextVQA images" "$REPO_ROOT/entropy_exp/datasets/textvqa/train_images"
check_path "ScienceQA questions" "$REPO_ROOT/entropy_exp/eval_questions/scienceqa/llava_test_CQM-A.json"
check_path "ScienceQA images" "$REPO_ROOT/entropy_exp/eval_questions/scienceqa/test"

if [[ -x "$PYTHON_BIN" ]]; then
  echo "OK      python: $PYTHON_BIN"
  if ! "$PYTHON_BIN" - <<'PY'
import accelerate
import torch
import transformers
from transformers import CLIPTextModelWithProjection, CLIPTokenizerFast

print(f"VERSIONS torch={torch.__version__} transformers={transformers.__version__} accelerate={accelerate.__version__}")
print(f"CUDA available={torch.cuda.is_available()} devices={torch.cuda.device_count()}")
CLIPTokenizerFast.from_pretrained("openai/clip-vit-large-patch14-336", local_files_only=True)
CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14-336", local_files_only=True)
print("CACHE   CLIP text tokenizer/model: available")
PY
  then
    status=1
  fi
else
  echo "MISSING python: $PYTHON_BIN"
  status=1
fi

exit "$status"
