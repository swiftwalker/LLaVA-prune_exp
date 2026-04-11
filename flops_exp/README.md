# flops_exp

`flops_exp` is a standalone workspace for:

- collecting dataset-side text token length statistics
- estimating run-config-driven LLM FLOPs for saved `entropy_exp` experiments
- batch FLOPs estimation across multiple saved runs

It lives alongside `entropy_exp`, but reuses the same tokenizer, prompt
template, and run-config semantics so the FLOPs estimate stays aligned with the
actual LLaVA pruning experiments.

## What It Counts

The current implementation focuses on:

- LLM prefill FLOPs
- pruning method overhead inside the LLM path

It does **not** include:

- vision tower FLOPs
- multimodal projector FLOPs
- sorting/top-k/cache movement in the total FLOPs number

Those non-FLOP costs are emitted as notes in the report instead.

## Environment

The bash entry scripts prefer the `llava` conda environment instead of `base`.
If you are not already inside `llava`, the scripts will try to activate it
automatically.

Manual activation is still the safest option:

```bash
conda activate llava
cd ~/LLaVA-prune_exp
```

## Workspace Layout

```text
flops_exp/
├── configs/
├── outputs/
├── scripts/
├── src/
└── tests/
```

## Main Commands

### 1. Collect text-length statistics

```bash
bash flops_exp/scripts/run_text_stats.sh --dataset mme --sample-limit 8
```

The script prints the output run directory as its last line. That directory
contains:

- `config.yaml`
- `summary.json`
- `summary.csv`
- `per_sample.csv`

### 2. Single-run FLOPs from a saved run dir

For single-run FLOPs, you must provide a saved `entropy_exp` run snapshot by
`--run-dir` or `--run-config`.

Stats-based example:

```bash
STATS_DIR="$(bash flops_exp/scripts/run_text_stats.sh --dataset mme --sample-limit 8 | tail -n 1)"
bash flops_exp/scripts/run_flops.sh \
  --run-dir entropy_exp/outputs/runs/attn_score/mme/mme_attn_score_l1_r0p2__20260410_123456_000001 \
  --stats-json "$STATS_DIR/summary.json"
```

Fixed-length example:

```bash
bash flops_exp/scripts/run_flops.sh \
  --run-dir entropy_exp/outputs/runs/baseline/mme/mme_baseline_l1_r0__20260410_123456_000001 \
  --fixed-text-len 32
```

### 3. One-shot stats + single-run FLOPs

`run_all.sh` is now a convenience wrapper for:

1. generating text stats for one dataset
2. running a single FLOPs estimate against one saved run

```bash
bash flops_exp/scripts/run_all.sh \
  mme \
  entropy_exp/outputs/runs/attn_score/mme/mme_attn_score_l1_r0p2__20260410_123456_000001 \
  8
```

### 4. Batch FLOPs over many saved runs

All runs:

```bash
bash flops_exp/scripts/run_batch_flops.sh \
  --scope runs \
  --fixed-text-len 32
```

Dataset-filtered runs:

```bash
STATS_DIR="$(bash flops_exp/scripts/run_text_stats.sh --dataset mme --sample-limit 8 | tail -n 1)"
bash flops_exp/scripts/run_batch_flops.sh \
  --dataset mme \
  --stats-json "$STATS_DIR/summary.json"
```

## CLI Notes

### `run_text_stats.sh`

Forwards to:

```bash
python flops_exp/src/interface.py text-stats ...
```

Supported arguments:

- `--dataset {mme,gqa,pope}`
- `--sample-limit N`
- `--dataset-config /path/to/custom_dataset.yaml`

### `run_flops.sh`

Forwards to:

```bash
python flops_exp/src/interface.py flops ...
```

Required input groups:

- run source:
  - `--run-dir /path/to/entropy_exp/outputs/runs/...`
  - or `--run-config /path/to/config.yaml`
- text length source:
  - `--stats-json /path/to/summary.json`
  - or `--fixed-text-len FLOAT`

Optional arguments:

- `--method baseline|attn_score|entropy`
- `--length-stat mean|median|p25|p75`
- `--set pruning.prune_ratio=[0.4,0.4]`

### `run_batch_flops.sh`

Forwards to:

```bash
python flops_exp/src/interface.py batch-flops ...
```

Supported arguments:

- `--runs-root entropy_exp/outputs/runs`
- `--scope runs`
- `--dataset mme|gqa|pope`
- `--stats-json /path/to/summary.json`
- `--fixed-text-len FLOAT`
- `--length-stat mean|median|p25|p75`
- `--set ...`

## Outputs

### Single-run FLOPs

Single-run outputs are written to:

```text
flops_exp/outputs/flops/{source_run_name}_{timestamp}/
```

Each run contains:

- `config.yaml`
- `resolved_method.json`
- `report.json`
- `report.csv`
- `notes.txt`

### Batch FLOPs

Batch outputs are written to:

```text
flops_exp/outputs/batch_flops/{scope_or_dataset}_{timestamp}/
```

Each batch run contains:

- `config.yaml`
- `summary.json`
- `summary.csv`
- `per_run.csv`

## Run-Config Semantics

FLOPs estimation is bound to a saved `entropy_exp` run config.

Current behavior:

- `_run_meta.run_mode == baseline` resolves to baseline even if
  `pruning.strategy` is still `attn_score`
- `strategy=entropy` with `entropy.dynamic_ratio=true` is treated as
  `dynamic_estimated`
- `layer_selection=dynamic` is rejected unless a fully explicit plan is added
  later

This prevents estimating FLOPs from a generic default prune config and drifting
away from the actual experiment that produced the run.

## Validation Status

The current implementation has been checked with:

- `python -m unittest discover -s flops_exp/tests -p 'test_*.py'` in the `llava` environment
- a single-run smoke flow using `--run-dir + --stats-json`
- a single-run fixed-length smoke flow
- a batch smoke flow using `--dataset mme`
