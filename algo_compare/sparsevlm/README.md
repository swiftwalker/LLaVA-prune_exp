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

The run wrappers execute official SparseVLM inference code from
`third_party/SparseVLMs`, while letting this workspace control paths and output
directories.

Example:

```bash
USE_VERSION=2_0 RETAIN_TOKN=192 \
MODEL_PATH=entropy_exp/models/llava-v1.5-7b \
bash algo_compare/sparsevlm/scripts/run_official_mme.sh --dry-run
```

Generated outputs are written below `algo_compare/sparsevlm/outputs/` and are
ignored by git.
