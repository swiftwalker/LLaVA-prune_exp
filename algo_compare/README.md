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
  sparsevlm/
  scripts/
  src/algo_compare/
```

Future method families should follow the same shape, for example
`algo_compare/fastv/` or `algo_compare/prumerge/`.

## Basic Commands

List registered official methods:

```bash
python algo_compare/scripts/list_methods.py
```

Check a manifest:

```bash
python algo_compare/scripts/check_manifests.py \
  algo_compare/sparsevlm/manifests/local_reference_runs.yaml
```

Summarize manifest coverage:

```bash
python algo_compare/scripts/summarize_manifest.py \
  algo_compare/sparsevlm/manifests/local_reference_runs.yaml
```

Fetch official SparseVLM code:

```bash
bash algo_compare/sparsevlm/scripts/fetch_official.sh
```
