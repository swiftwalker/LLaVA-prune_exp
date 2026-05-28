# FastV Official Reproduction

This directory is for running the official FastV implementation from
`pkunlp-icler/FastV` against local LLaVA models and local evaluation datasets.

Official source, generated outputs, and large artifacts are intentionally kept
out of git.

## Setup

```bash
bash algo_compare/fastv/scripts/fetch_official.sh --dry-run
NO_GIT_PROXY=1 bash algo_compare/fastv/scripts/fetch_official.sh
bash algo_compare/fastv/scripts/check_env.sh
```

## Run

```bash
python3 algo_compare/scripts/run_official.py \
  --method fastv \
  --dataset mme \
  --variant fastv_token_mask \
  --fastv-k 2 \
  --fastv-r 0.5 \
  --eval \
  --dry-run
```

Dataset shell wrappers are provided for convenience:

```bash
bash algo_compare/fastv/scripts/run_official_mme.sh --dry-run
bash algo_compare/fastv/scripts/run_official_gqa.sh --dry-run
bash algo_compare/fastv/scripts/run_official_pope.sh --dry-run
bash algo_compare/fastv/scripts/run_official_textvqa.sh --dry-run
bash algo_compare/fastv/scripts/run_official_scienceqa.sh --dry-run
```

The default FastV setting is `K=2`, `R=0.5`, `image_token_length=576`, and
`attention_rank=288`.

The wrapper prepends FastV's modified Transformers tree to `PYTHONPATH`.
Because the local LLaVA env has `tokenizers==0.15.1` while the official
Transformers checkout hard-checks `<0.14`, wrapper runs also enable a
process-local compatibility shim via `FASTV_ALLOW_TOKENIZERS_015=1`. This does
not modify the official source or the conda environment.
