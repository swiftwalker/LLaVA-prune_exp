# Repository Audit

Snapshot date: `2026-04-12`

This document records the reasoning used to classify the repository into
`supported`, `secondary`, and `archived` areas.

## Audit Basis

The audit used three signals:

1. Recent git history on the active branch family:
   `prune-exp -> keep-position-ids`
2. Current primary-document references
3. Existing tests and runtime entrypoints

## High-Signal Git-History Findings

Recent active commits are heavily concentrated in:

- `entropy_exp/`
  - scheduler
  - pruning strategies
  - result summarization
  - run layout and recovery
- `flops_exp/`
  - companion analysis workflow
- a small portion of `llava/`
  - dependency fixes required by the experiment workspaces
- one retained root helper script
  - `scripts/convert_gqa_for_eval.py`, still called by `entropy_exp/src/eval_datasets.py`

Representative recent topics include:

- `tmux`-based scheduler introduction and follow-up fixes
- keep-position-ids compatibility
- tail masking strategy
- adaptive stratified SparseVLM pruning
- POPE metric normalization
- results workflow documentation

These are all consistent with an experiment-first repository rather than a
general upstream LLaVA distribution.

## Current-Reference Findings

Primary docs now cluster around:

- `entropy_exp/USAGE.md`
- `entropy_exp/SCHEDULER.md`
- `entropy_exp/RESULTS_WORKFLOW.md`
- `entropy_exp/TRANSFORMER_BLOCK_STRATEGY_PATHS.md`
- `flops_exp/README.md`

By contrast:

- the old root `README.md`
- the old root `docs/`
- the old root `scripts/`

mainly described upstream LLaVA training, serving, model-zoo, and evaluation
flows, which no longer match the current pruning-focused development history.

## Classification Decisions

### Supported

- `entropy_exp` pruning runtime, scheduler, eval, summary, and canonical plans
- `flops_exp` runtime and README
- `llava/` as dependency layer

### Secondary

- Phase 1 capture and offline entropy analysis
- FLOPs design reference doc

### Archived

- upstream LLaVA root README/docs/scripts
- with one exception: `scripts/convert_gqa_for_eval.py` remains on the supported path because current GQA evaluation still imports it by location
- historical experiment design and replan docs
- run-layout migration utility
- non-canonical or one-off plan files

## Why Archive Instead of Delete

The repository has already gone through multiple target shifts. Several older
documents still contain useful rationale, path history, and recovery context.
Moving them under `archive/` gives us:

- a cleaner mainline
- preserved historical traceability
- lower risk than immediate hard deletion

## Follow-Up Candidates

If the secondary Phase 1 workflow remains unused for another cleanup cycle, it
can be reconsidered for archival or deletion. The same applies to any
historical plan or maintenance script that no longer has a concrete operational
consumer.
