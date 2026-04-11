# Phase 1 Secondary Workflow

This document covers the retained **Phase 1** attention-capture and offline
entropy-analysis path inside `entropy_exp`.

Status:

- support level: `secondary`
- default pruning workflow: **not** this path
- use this only when you explicitly need raw attention capture or historical
  entropy-analysis behavior

Primary pruning users should start from:

- [`USAGE.md`](./USAGE.md)
- [`SCHEDULER.md`](./SCHEDULER.md)
- [`RESULTS_WORKFLOW.md`](./RESULTS_WORKFLOW.md)

## When To Use It

Use this secondary workflow only if you need one of these:

- capture full-layer `text -> vision` attention data without pruning
- generate HDF5 artifacts for offline metric analysis
- reproduce or compare earlier entropy-analysis experiments

Do **not** use it for:

- current pruning benchmarks
- scheduler-driven large matrices
- current result summarization mainline

## Retained Entry Points

- `entropy_exp/src/inference.py`
- `entropy_exp/src/hooks.py`
- `entropy_exp/src/metrics.py`
- `entropy_exp/analysis/entropy_analysis.py`
- `entropy_exp/scripts/run_capture.sh`
- `entropy_exp/scripts/run_analysis.sh`

## Minimal Workflow

### 1. Capture attention data

```bash
conda activate llava
cd ~/LLaVA-prune_exp

bash entropy_exp/scripts/run_capture.sh mme 10
```

This writes raw artifacts under:

```text
entropy_exp/outputs/raw/
entropy_exp/outputs/answers/
```

### 2. Analyze captured HDF5 files

```bash
bash entropy_exp/scripts/run_analysis.sh
```

This writes processed CSV outputs under:

```text
entropy_exp/outputs/processed/
```

### 3. Optional evaluation

If you also need task metrics for the captured answers:

```bash
bash entropy_exp/scripts/run_eval.sh gqa entropy_exp/outputs/answers/gqa_*.jsonl
```

## Relationship To Current Mainline

- `prune_inference.py` is the supported pruning entrypoint
- `run_scheduler.py` is the supported large-matrix orchestration entrypoint
- `summarize_results.py` is the supported matrix-level result aggregation tool

This Phase 1 path is kept because it still has diagnostic and historical value,
but it is intentionally separated from the current pruning mainline.
