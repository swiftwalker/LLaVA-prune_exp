# DivPrune Official Repo Notes

- Repository: `https://github.com/vbdi/divprune`
- Target commit: `799e2d950aa01ba7860907f5a6d86061f885dca6`
- Paper: `DivPrune: Diversity-based Visual Token Pruning for Large Multimodal Models`
- Official activation path:
  - import the official `LLaVA/` package from the repo
  - set `BASELINE=OURS`
  - set `LAYER_INDEX=0`
  - set `SUBSET_RATIO=0.098`
- The official README uses `lmms_eval` for paper tables. The local wrapper uses
  the same official LLaVA implementation but produces local `answers.jsonl`
  compatible with `entropy_exp/src/eval_datasets.py`.
