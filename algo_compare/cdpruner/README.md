# CDPruner Official LLaVA-NeXT Wrapper

This directory runs the pinned official CDPruner implementation on
LLaVA-v1.6-Vicuna-7B without modifying the third-party checkout.

The official repository contains a LLaVA-NeXT anyres path, but its CDPruner
generation wiring is implemented in `llava_llama.py`, so the Vicuna checkpoint
uses the official native path directly. Question text enters the official CLIP
text tower and the official conditional-DPP selector prepares visual
embeddings. A local generation bridge remains available for explicit Mistral
runs; it does not replace the selector.

The official fork also forces every input to a `672x672` anyres canvas. The
local NeXT adapter restores the model's canonical resolution selection so that
CDPruner and other methods receive the same image crops. This changes only
preprocessing policy; the official conditional-DPP selector remains unchanged.

The official NeXT implementation selects `visual_token_num` tokens from each
576-token crop and changes `spatial_unpad` to `spatial`. Consequently, K is a
per-crop budget and the final image count is approximately `crop_count * K`.

## Setup And Smoke

```bash
bash algo_compare/cdpruner/scripts/fetch_official.sh
bash algo_compare/cdpruner/scripts/check_env.sh

PYTHON_BIN=/data_ssd/liuyu/.conda/envs/llava-next/bin/python \
MODEL_PATH=entropy_exp/models/llava-v1.6-vicuna-7b \
CONV_MODE=vicuna_v1 RETAIN_TOKEN=126 MAX_SAMPLES=3 \
bash algo_compare/cdpruner/scripts/run_official_gqa.sh --no-eval
```

Command preview through the shared official runner:

```bash
/data_ssd/liuyu/.conda/envs/llava-next/bin/python \
  algo_compare/scripts/run_official.py \
  --method cdpruner --dataset gqa --retain-token 126 \
  --max-samples 3 --no-eval --dry-run
```
