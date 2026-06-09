# LLaVA-prune_exp

`LLaVA-prune_exp` is now organized as an **experiment-first workspace** for
visual-token pruning research built on top of LLaVA.

Current repository focus:

- `entropy_exp/`
  Main pruning experiment workspace
- `flops_exp/`
  Companion FLOPs and text-length analysis workspace
- `llava/`
  Underlying model/runtime dependency used by both experiment workspaces

This repository is no longer documented as a general-purpose upstream LLaVA
home. The upstream training, serving, and evaluation materials are preserved
for traceability under `archive/upstream_llava/`.

## Start Here

- Final pruning method: [`entropy_exp/SCND_GPU_FINAL_METHOD.md`](./entropy_exp/SCND_GPU_FINAL_METHOD.md)
- Pruning experiment usage: [`entropy_exp/USAGE.md`](./entropy_exp/USAGE.md)
- Scheduler and large-matrix orchestration: [`entropy_exp/SCHEDULER.md`](./entropy_exp/SCHEDULER.md)
- Result collection and matrix summarization: [`entropy_exp/RESULTS_WORKFLOW.md`](./entropy_exp/RESULTS_WORKFLOW.md)
- Transformer-block strategy paths: [`entropy_exp/TRANSFORMER_BLOCK_STRATEGY_PATHS.md`](./entropy_exp/TRANSFORMER_BLOCK_STRATEGY_PATHS.md)
- FLOPs workspace: [`flops_exp/README.md`](./flops_exp/README.md)

## Repository Layout

The repository uses three support levels:

- `supported`
  The current experiment mainline. These paths are the ones we keep linked from
  primary entry docs and validate in routine tests.
- `secondary`
  Still usable, but no longer presented as the default workflow.
- `archived`
  Historical or upstream material kept for reference only.

See:

- support matrix: [`docs/REPO_LAYOUT.md`](./docs/REPO_LAYOUT.md)
- audit basis from current docs and git history: [`docs/REPO_AUDIT.md`](./docs/REPO_AUDIT.md)

## Environment

The main workflows assume the `llava` conda environment:

```bash
conda activate llava
cd ~/LLaVA-prune_exp
```

The authoritative runtime and path details live in the experiment workspace
docs rather than this root README.

## Archived Materials

Archived materials are intentionally preserved, not deleted:

- upstream LLaVA snapshot docs and top-level scripts:
  [`archive/upstream_llava/README.md`](./archive/upstream_llava/README.md)
- historical `entropy_exp` design/replan documents:
  `archive/docs/entropy_exp/`

If you need the original upstream LLaVA narrative or legacy experiment context,
start from those archive locations instead of the primary experiment entry docs.
