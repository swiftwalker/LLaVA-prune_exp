# Transition Guide

This document is a handoff guide for moving the current `LLaVA-prune_exp`
workflow to a new machine and letting a new agent continue work without
re-discovering the recent project state.

## 1. Current Branch State

- Primary historical experiment branch: `keep-position-ids`
- Current integration branch: `prune-exp-sync-non-position-id`
- Current HEAD when this document was last updated: `b95802a`

### What this branch contains

This branch starts from `prune-exp` and syncs over the generic improvements
from `keep-position-ids`, while intentionally excluding the position-id
preservation runtime changes.

Included:

- run layout refactor
- transformer block path documentation
- tmux-based scheduler
- scheduler GPU memory estimation fix
- CLIP local-cache / no-proxy loading fix
- POPE `macro_f1` evaluation and summary logic
- `prune-exp`-specific full-matrix scheduler plan

Intentionally excluded:

- physical-pruning position-id preservation
- sparse-position-id runtime compatibility patch
- keep-position-ids-specific experiment notes

## 2. Completed Results Already Available

### Keep-position-ids full matrix

The full `keep-position-ids` experiment matrix has already been completed and
evaluated.

Key output locations:

- run root:
  - `entropy_exp/outputs/runs/`
- full summary:
  - `entropy_exp/outputs/summary/keep_position_ids_all_strategies_all_datasets/summary.csv`
  - `entropy_exp/outputs/summary/keep_position_ids_all_strategies_all_datasets/summary.json`
  - `entropy_exp/outputs/summary/keep_position_ids_all_strategies_all_datasets/skipped_runs.json`

Current counts:

- completed run directories in `entropy_exp/outputs/runs/`: `324`
- included runs in the final summary: `324`
- skipped runs in the final summary: `0`

### Archived historical material

Archived outputs are under:

- `entropy_exp/outputs/runs_archive/pre_keep_position_ids_20260323`
- `entropy_exp/outputs/runs_archive/formal_failed_attempts_20260324`

Important note:

- archived `POPE` results are not a complete multi-strategy matrix
- usable archived `POPE` results are only:
  - `attn_score` old runs
  - one old `baseline` run
- failed-attempt archive directories should not be treated as valid evaluated
  results

## 3. Current POPE Metric Definition

The repo no longer uses POPE weighted-average F1 in the entropy-layer wrapper.

Current POPE metric behavior:

- per-category scoring still comes from `llava/eval/eval_pope.py`
- final repo-level POPE metric is `macro_f1`
- `macro_f1` is the simple mean of all category `f1_score` values parsed from
  POPE evaluation output
- this is the current project source of truth for POPE aggregation

Relevant files:

- `entropy_exp/src/eval_datasets.py`
- `entropy_exp/src/summarize_results.py`

## 4. Environment Requirements

Expected runtime environment on the current machine:

- Conda env: `llava`
- Conda init script:
  - `/data_ssd/liuyu/miniconda3/etc/profile.d/conda.sh`
- `tmux` installed
- `nvidia-smi` available

Project assumes the following local assets exist:

- model root:
  - `entropy_exp/models/llava-v1.5-7b`
- datasets:
  - `entropy_exp/datasets/gqa/...`
  - `entropy_exp/datasets/MME_Benchmark_release_version/...`
  - `entropy_exp/datasets/pope/...`
- eval question files:
  - `entropy_exp/eval_questions/gqa/...`
  - `entropy_exp/eval_questions/MME/...`
  - `entropy_exp/eval_questions/pope/...`

CLIP loading note:

- the branch includes a fix to prefer local Hugging Face cache for
  `openai/clip-vit-large-patch14-336`
- if local cache is incomplete, the fallback path temporarily bypasses proxy
  env vars and talks directly to `HF_ENDPOINT`

## 5. Plans And Entry Points

Main scheduler plan for this branch:

- `entropy_exp/plans/prune_exp_all_strategies_all_datasets.yaml`

Main operational entry points:

- pruning:
  - `entropy_exp/scripts/run_prune.sh`
- evaluation:
  - `entropy_exp/scripts/run_eval.sh`
- scheduler:
  - `entropy_exp/scripts/run_scheduler.py`
- aggregate summary:
  - `entropy_exp/src/summarize_results.py`

Useful docs:

- `entropy_exp/USAGE.md`
- `entropy_exp/SCHEDULER.md`
- `entropy_exp/TRANSFORMER_BLOCK_STRATEGY_PATHS.md`
- `entropy_exp/EXPERIMENT_DESIGN.md`

## 6. Validation Already Done On This Branch

The following test set is passing on `prune-exp-sync-non-position-id`:

- `python -m unittest entropy_exp.tests.test_scheduler`
- `python -m unittest entropy_exp.tests.test_clip_encoder`
- `python -m unittest entropy_exp.tests.test_eval_datasets`
- `python -m unittest entropy_exp.tests.test_summarize_results`
- `python -m unittest entropy_exp.tests.test_run_layout`

Combined run:

```bash
source /data_ssd/liuyu/miniconda3/etc/profile.d/conda.sh
conda activate llava
python -m unittest \
  entropy_exp.tests.test_scheduler \
  entropy_exp.tests.test_clip_encoder \
  entropy_exp.tests.test_eval_datasets \
  entropy_exp.tests.test_summarize_results \
  entropy_exp.tests.test_run_layout
```

Scheduler dry-run already verified:

```bash
source /data_ssd/liuyu/miniconda3/etc/profile.d/conda.sh
conda activate llava
python entropy_exp/scripts/run_scheduler.py \
  --plan entropy_exp/plans/prune_exp_all_strategies_all_datasets.yaml \
  --dry-run
```

## 7. Known Caveats

- A tiny manual `POPE` smoke on this branch hit `ZeroDivisionError` when using
  only one sample because `llava/eval/eval_pope.py` expects all categories to
  be represented.
- A follow-up 3-category smoke setup was prepared, but final execution was
  blocked by temporary GPU OOM because the machine was saturated at that time.
- This means branch-level functionality is validated by unit tests and dry-run,
  but a fresh end-to-end `POPE` smoke may still need to be re-run on a machine
  with enough free GPU memory.

## 8. Recommended Next Actions On A New Machine

### If the goal is to inspect completed keep-position-ids results

1. Check out `keep-position-ids`
2. Verify summary files under:
   - `entropy_exp/outputs/summary/keep_position_ids_all_strategies_all_datasets/`
3. Use `summary.csv` / `summary.json` as the main result source

### If the goal is to continue prune-exp work

1. Check out `prune-exp-sync-non-position-id`
2. Recreate Conda env and data/model paths
3. Run the passing unit tests listed above
4. Run scheduler dry-run with:

```bash
python entropy_exp/scripts/run_scheduler.py \
  --plan entropy_exp/plans/prune_exp_all_strategies_all_datasets.yaml \
  --dry-run
```

5. If GPU capacity is available, run a small smoke test first:
   - one `POPE` run with a question file containing at least one sample from
     each category
   - then `run_eval.sh` on that exact run
   - verify `stdout.txt` ends with `Macro-F1: ...`
   - verify run-local `eval/summary.json` contains `macro_f1`

6. Only then launch larger prune-exp experiments

## 9. Commands Worth Reusing

Activate environment:

```bash
source /data_ssd/liuyu/miniconda3/etc/profile.d/conda.sh
conda activate llava
```

Run all key tests:

```bash
python -m unittest \
  entropy_exp.tests.test_scheduler \
  entropy_exp.tests.test_clip_encoder \
  entropy_exp.tests.test_eval_datasets \
  entropy_exp.tests.test_summarize_results \
  entropy_exp.tests.test_run_layout
```

Scheduler dry-run:

```bash
python entropy_exp/scripts/run_scheduler.py \
  --plan entropy_exp/plans/prune_exp_all_strategies_all_datasets.yaml \
  --dry-run
```

Evaluate all completed runs:

```bash
bash entropy_exp/scripts/run_eval.sh runs
```

Aggregate all completed runs:

```bash
python entropy_exp/src/summarize_results.py \
  --output-dir entropy_exp/outputs/summary/keep_position_ids_all_strategies_all_datasets \
  --selection-label keep-position-ids-all-strategies-all-datasets \
  --run-dir $(find entropy_exp/outputs/runs -mindepth 3 -maxdepth 3 -type d | sort)
```
