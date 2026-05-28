# PDROP / PyramidDrop Official Wrapper

This directory contains metadata and thin local wrappers for the official
PyramidDrop / PDROP implementation.

Official source code is fetched into `third_party/PyramidDrop` and is ignored by
git. The wrapper only resolves local model, dataset, output, and eval paths; it
does not modify the official algorithm.

## Fetch

```bash
bash algo_compare/pdrop/scripts/fetch_official.sh --dry-run
bash algo_compare/pdrop/scripts/fetch_official.sh
```

If `git clone` is unreliable on this machine, the fetch script can fall back to
the GitHub zip archive at the pinned commit.

## Dry Run

```bash
python3 algo_compare/scripts/run_official.py \
  --method pdrop \
  --dataset mme \
  --variant pdrop_v1_5 \
  --eval \
  --dry-run
```

Default PDROP parameters follow the official LLaVA-1.5 examples:
`layer_list=[8,16,24]` and `image_token_ratio_list=[0.5,0.25,0.125]`.

## Run

```bash
CUDA_VISIBLE_DEVICES=0 python3 algo_compare/scripts/run_official.py \
  --method pdrop \
  --dataset gqa \
  --variant pdrop_v1_5 \
  --eval
```

Outputs are written under `algo_compare/pdrop/outputs/` and are ignored by git.
