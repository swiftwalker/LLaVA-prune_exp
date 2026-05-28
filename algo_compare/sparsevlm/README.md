# SparseVLM Official Reproduction

This method directory is for official SparseVLM / SparseVLM-v2 reproduction
work. It is not the home for local exploratory variants such as entropy-alpha,
global memory, boost, or compensated sampling.

## First-Time Setup

Preview the fetch:

```bash
bash algo_compare/sparsevlm/scripts/fetch_official.sh --dry-run
```

Fetch the official repository:

```bash
bash algo_compare/sparsevlm/scripts/fetch_official.sh
```

Check local paths:

```bash
bash algo_compare/sparsevlm/scripts/check_env.sh
```

## Run Wrappers

The preferred entrypoint is the shared Python wrapper. It executes official
SparseVLM inference code from `third_party/SparseVLMs`, while this workspace
controls local paths, output directories, local eval, and result collection.

Example:

```bash
python3 algo_compare/scripts/run_official.py \
  --method sparsevlm \
  --dataset mme \
  --variant sparsevlm_v2 \
  --retain-token 192 \
  --eval \
  --dry-run
```

Dataset shell scripts are thin convenience wrappers around the same CLI:

```bash
bash algo_compare/sparsevlm/scripts/run_official_mme.sh --dry-run
bash algo_compare/sparsevlm/scripts/run_official_gqa.sh --dry-run
bash algo_compare/sparsevlm/scripts/run_official_pope.sh --dry-run
bash algo_compare/sparsevlm/scripts/run_official_textvqa.sh --dry-run
bash algo_compare/sparsevlm/scripts/run_official_scienceqa.sh --dry-run
```

Generated outputs are written below `algo_compare/sparsevlm/outputs/` and are
ignored by git.

The five local datasets currently resolved by default are `gqa`, `mme`,
`pope`, `textvqa`, and `scienceqa`. `scienceqa` uses the local path recorded in
`entropy_exp/configs/prune.yaml`; if the official loader needs a stricter JSONL
format for a full non-dry run, add a dataset adapter in a later wrapper revision
without changing the official SparseVLM source.
