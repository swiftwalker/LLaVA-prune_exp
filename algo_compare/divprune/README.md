# DivPrune Official Wrapper

This directory contains the local wrapper for the official DivPrune method.
Official source is fetched into `third_party/divprune` and is not committed.

## Fetch

```bash
bash algo_compare/divprune/scripts/fetch_official.sh
```

## Dry Run

```bash
python3 algo_compare/scripts/run_official.py \
  --method divprune \
  --dataset mme \
  --variant divprune_r0p098 \
  --eval \
  --dry-run
```

## Default Config

- official repo: `https://github.com/vbdi/divprune`
- variant: `divprune_r0p098`
- `BASELINE=OURS`
- `LAYER_INDEX=0`
- `SUBSET_RATIO=0.098`
- expected retained visual tokens for 576-token LLaVA-1.5 images: `round(0.098 * 576) = 56`

The official implementation lives in the repository's `LLaVA/` checkout and is
activated by environment variables. The wrapper only adapts local dataset paths,
model paths, command metadata, and local evaluation.

## LLaVA-NeXT Vicuna

The NeXT wrapper keeps the official max-min selector but replaces its fixed
`SYS_TOKEN_LEN=35` assumption with dynamic visual-span discovery. For anyres
inputs, `SUBSET_RATIO` is applied to each sample's merged visual sequence, so
the absolute retained count varies with image shape.

```bash
PYTHON_BIN=/data_ssd/liuyu/.conda/envs/llava-next/bin/python \
MODEL_PATH=entropy_exp/models/llava-v1.6-vicuna-7b \
CONV_MODE=vicuna_v1 MAX_SAMPLES=3 \
DIVPRUNE_SUBSET_RATIO=0.21875 \
bash algo_compare/divprune/scripts/run_official_gqa.sh --no-eval
```
