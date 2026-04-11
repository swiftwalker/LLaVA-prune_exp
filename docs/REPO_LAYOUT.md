# Repository Layout

This repository is intentionally organized around the current pruning experiment
workflow, not around the original upstream LLaVA documentation structure.

## Support Levels

### Supported

These paths define the current mainline and should remain discoverable from the
root README.

| Area | Paths | Role |
| --- | --- | --- |
| Pruning workflow | `entropy_exp/USAGE.md`, `entropy_exp/SCHEDULER.md`, `entropy_exp/RESULTS_WORKFLOW.md` | Main operator docs |
| Strategy/runtime internals | `entropy_exp/TRANSFORMER_BLOCK_STRATEGY_PATHS.md` | Code-path reference |
| Pruning runtime | `entropy_exp/src/prune_inference.py`, `entropy_exp/src/pruner.py`, `entropy_exp/src/scheduler.py`, `entropy_exp/src/run_layout.py`, `entropy_exp/src/eval_datasets.py`, `entropy_exp/src/summarize_results.py`, `entropy_exp/src/strategies/*` | Core implementation |
| Pruning scripts | `entropy_exp/scripts/run_prune.sh`, `entropy_exp/scripts/run_scheduler.py`, `entropy_exp/scripts/run_eval.sh`, `entropy_exp/scripts/run_summary.sh`, `entropy_exp/scripts/build_recovery_plan_from_scheduler_state.py`, `entropy_exp/scripts/run_dir_common.sh`, `entropy_exp/scripts/run_dir_helper.py`, `entropy_exp/scripts/check_incomplete_runs.sh` | Main CLI surface |
| Eval compatibility helper | `scripts/convert_gqa_for_eval.py` | Retained root-level helper still required by `entropy_exp/src/eval_datasets.py` |
| Canonical plans | `entropy_exp/plans/keep_position_ids_sparsevlm_adaptive_stratified_full_matrix.yaml`, `entropy_exp/plans/tail_masking_attn_score_54.yaml`, `entropy_exp/plans/scheduler_smoke_test.yaml` | Operational plans |
| FLOPs workflow | `flops_exp/README.md`, `flops_exp/src/*`, `flops_exp/scripts/*` | Secondary but supported workspace |
| Dependency layer | `llava/` | Shared model/runtime dependency |

### Secondary

These paths still have value, but they are no longer the default narrative or
first-class workflow.

| Area | Paths | Role |
| --- | --- | --- |
| Phase 1 capture/analysis | `entropy_exp/PHASE1_SECONDARY_WORKFLOW.md` | Entry doc for retained capture workflow |
| Phase 1 implementation | `entropy_exp/src/inference.py`, `entropy_exp/src/hooks.py`, `entropy_exp/src/metrics.py`, `entropy_exp/analysis/entropy_analysis.py`, `entropy_exp/scripts/run_capture.sh`, `entropy_exp/scripts/run_analysis.sh` | Historical capture/analysis path retained for diagnostics |
| FLOPs design notes | `flops_exp/DESIGN.md` | Internal design reference, not main entry |

### Archived

These paths are preserved for traceability, but should not be used as current
entry points.

| Area | Paths | Role |
| --- | --- | --- |
| Upstream LLaVA docs | `archive/upstream_llava/docs/` | Archived upstream documentation |
| Upstream top-level scripts | `archive/upstream_llava/scripts/` | Archived upstream training/conversion scripts (except the retained GQA conversion helper) |
| Upstream repository README snapshot | `archive/upstream_llava/README.md` | Archived upstream narrative |
| Historical experiment docs | `archive/docs/entropy_exp/EXPERIMENT_DESIGN.md`, `archive/docs/entropy_exp/KEEP_POSITION_IDS_EXPERIMENT_REPLAN.md`, `archive/docs/entropy_exp/viewer/APP_DESIGN.md` | Design/replan history |
| Historical plans | `archive/plans/entropy_exp/` | Non-canonical experimental plans |
| Historical maintenance script | `archive/scripts/entropy_exp/migrate_runs_layout.py` | Migration utility kept only for traceability |

## Current Navigation Rules

- Start from the root [`README.md`](../README.md) for repository orientation.
- Use `entropy_exp/` docs for pruning execution and result analysis.
- Use `flops_exp/README.md` for FLOPs-specific follow-up work.
- Treat `archive/` as read-only historical context unless a recovery or audit
  explicitly needs it.
