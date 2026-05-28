# algo_compare

`algo_compare` is a lightweight workspace for runnable reproductions of
published / official pruning methods.

This directory is intentionally separate from `entropy_exp`:

- `entropy_exp` remains the local exploration and implementation workspace.
- `algo_compare` tracks official method metadata, fetch scripts, run wrappers,
  manifests, and reports.
- Official source trees, model weights, datasets, and generated outputs are not
  committed here.

## Layout

Each top-level method directory is named after the method family:

```text
algo_compare/
  fastv/
  divprune/
  pdrop/
  sparsevlm/
  visionzip/
  scripts/
  src/algo_compare/
```

Future method families should follow the same shape, for example
`algo_compare/prumerge/`.

## Basic Commands

List registered official methods:

```bash
python3 algo_compare/scripts/list_methods.py
```

Check a manifest:

```bash
python3 algo_compare/scripts/check_manifests.py \
  algo_compare/sparsevlm/manifests/local_reference_runs.yaml
```

Summarize manifest coverage:

```bash
python3 algo_compare/scripts/summarize_manifest.py \
  algo_compare/sparsevlm/manifests/local_reference_runs.yaml
```

Fetch official SparseVLM code:

```bash
bash algo_compare/sparsevlm/scripts/fetch_official.sh
```

Fetch official FastV code:

```bash
NO_GIT_PROXY=1 bash algo_compare/fastv/scripts/fetch_official.sh
```

Fetch official PyramidDrop / PDROP code:

```bash
bash algo_compare/pdrop/scripts/fetch_official.sh
```

Fetch official DivPrune code:

```bash
bash algo_compare/divprune/scripts/fetch_official.sh
```

Fetch official VisionZip code:

```bash
bash algo_compare/visionzip/scripts/fetch_official.sh
```

Run an official method through the local wrapper:

```bash
python3 algo_compare/scripts/run_official.py \
  --method sparsevlm \
  --dataset mme \
  --variant sparsevlm_v2 \
  --retain-token 192 \
  --eval \
  --dry-run
```

Run official PDROP through the same local wrapper:

```bash
python3 algo_compare/scripts/run_official.py \
  --method pdrop \
  --dataset mme \
  --variant pdrop_v1_5 \
  --eval \
  --dry-run
```

Run official DivPrune through the same local wrapper:

```bash
python3 algo_compare/scripts/run_official.py \
  --method divprune \
  --dataset mme \
  --variant divprune_r0p098 \
  --eval \
  --dry-run
```

Run official VisionZip through the same local wrapper:

```bash
python3 algo_compare/scripts/run_official.py \
  --method visionzip \
  --dataset mme \
  --variant visionzip_64 \
  --eval \
  --dry-run
```

The wrapper resolves local model, question, image, output, and eval paths from
`entropy_exp/configs/prune.yaml` plus each method's `env.yaml`. It does not
modify official algorithm code.
