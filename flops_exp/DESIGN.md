# flops_exp Design

## Current Scope

`flops_exp` is the FLOPs-specific companion workspace for saved pruning runs in
this repository.

The current code implements:

- dataset text token length statistics
- model-profile loading from local LLaVA configs
- prune-spec resolution from saved `entropy_exp/outputs/runs/*/config.yaml`
- LLM prefill FLOPs estimation
- method overhead estimation for `baseline`, `attn_score`, and `entropy`
- batch FLOPs estimation across multiple saved runs
- structured output export to JSON and CSV

The current code does **not** implement:

- vision tower FLOPs
- multimodal projector FLOPs
- decode-stage FLOPs
- fully dynamic layer-selection resolution

## Counting Rules

- Unit: FLOPs
- One multiply-add is counted as `2 FLOPs`
- Main total:
  - `self_attention_flops`
  - `mlp_flops`
  - `prune_method_flops`
- Excluded from the total and reported only as notes:
  - sorting / top-k selection
  - cache pruning / tensor movement
  - other non-arithmetic control overhead

## Implemented Data Flow

### 1. Text statistics

`src/text_stats/compute_stats.py`:

- loads dataset records from `flops_exp/configs/datasets/*.yaml`
- tokenizes raw dataset `text`
- rebuilds the same Vicuna-style prompt used by the LLaVA path
- computes:
  - `raw_text_token_len`
  - `template_overhead_tokens`
  - `templated_input_ids_len_pre_mm`
  - `prefill_seq_len`

The exported summary keeps `raw_text_token_len_*` separate from
`template_overhead_tokens`, so later FLOPs runs can reuse aggregate stats
without rescanning the dataset.

### 2. Run-config-driven prune resolution

`src/prune_config_resolver.py` reads:

- `run_dir/config.yaml`
- or an explicit `run_config_path`

and resolves a normalized `ResolvedPruneSpec`.

Current resolution behavior:

- `_run_meta.run_mode == baseline`
  - resolves to baseline directly
- `layer_selection=fixed`
  - resolves directly to `prune_layers + prune_ratio_map`
- `strategy=entropy` with `dynamic_ratio=true`
  - marked as `dynamic_estimated`
  - still uses configured base prune ratios for sequence-length reduction
- `layer_selection=dynamic`
  - currently rejected with an explicit error

This is deliberate: FLOPs estimates must stay tied to a real saved experiment
run instead of a generic default prune config.

### 3. Method estimation

`src/methods/` contains lightweight method models:

- `baseline.py`
- `attn_score.py`
- `entropy.py`

Each method estimator does two things:

- build the per-segment sequence schedule after pruning
- estimate the arithmetic overhead of the pruning method itself

### 4. LLM FLOPs

`src/llm_flops.py` uses:

- `ModelProfile`
- `ResolvedPruneSpec`
- either fixed text length or saved text statistics

The implemented formula per decoder layer is:

- attention projections + output:
  - `8 * L * H^2`
- attention score/value path:
  - `4 * L^2 * H`
- MLP:
  - `6 * L * H * I`

where:

- `L` = effective sequence length at that layer
- `H` = hidden size
- `I` = intermediate size

The initial prefill length is:

- `prefill_seq_len = text_token_len + template_overhead_tokens - 1 + image_token_len`

### 5. Batch FLOPs

`src/batch_flops.py`:

- scans `entropy_exp/outputs/runs`
- keeps only run dirs containing `config.yaml`
- optionally filters by dataset prefix such as `mme_`
- calls the single-run `compute_llm_flops(...)` path for each matched run
- aggregates run-level summaries into a batch output directory

This keeps single and batch FLOPs on the same code path and avoids metric drift.

## Implemented Output Structure

### Text-stats runs

```text
flops_exp/outputs/text_stats/{dataset}_{timestamp}/
├── config.yaml
├── summary.json
├── summary.csv
└── per_sample.csv
```

### Single-run FLOPs

```text
flops_exp/outputs/flops/{source_run_name}_{timestamp}/
├── config.yaml
├── resolved_method.json
├── report.json
├── report.csv
└── notes.txt
```

### Batch FLOPs

```text
flops_exp/outputs/batch_flops/{scope_or_dataset}_{timestamp}/
├── config.yaml
├── summary.json
├── summary.csv
└── per_run.csv
```

## Entry Points

### Bash

- `scripts/run_text_stats.sh`
- `scripts/run_flops.sh`
- `scripts/run_all.sh`
- `scripts/run_batch_flops.sh`

All scripts prefer the `llava` conda environment and try to activate it
automatically.

### Python

`src/interface.py` provides three CLI entrypoints:

- `text-stats`
- `flops`
- `batch-flops`

## Repository Boundaries

`flops_exp` reuses:

- `llava.conversation.conv_templates`
- `llava.mm_utils.tokenizer_image_token`
- local model `config.json`
- saved `entropy_exp/outputs/runs/*/config.yaml`

`flops_exp` does not change:

- pruning runtime execution
- dataset source files
- entropy experiment outputs

## Tests

The current test suite covers:

- text-stat summary behavior
- prune resolution from saved run configs
- baseline-vs-prune handling via `_run_meta.run_mode`
- FLOPs monotonicity and method comparisons
- batch dataset filtering and all-runs traversal

The tests live in `flops_exp/tests/` and are runnable with:

```bash
conda activate llava
python -m unittest discover -s flops_exp/tests -p 'test_*.py'
```
