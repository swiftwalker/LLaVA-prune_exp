# DivPrune Official Repo Notes

- Repository: `https://github.com/vbdi/divprune`
- Target commit: `799e2d950aa01ba7860907f5a6d86061f885dca6`
- The official checkout recognizes LLaVA-1.6 image features and Mistral model
  loading, but its pruning block assumes a fixed 35-token prompt prefix.
- The local NeXT wrapper repairs only span discovery; max-min selection remains
  the pinned official implementation.
- Paper: `DivPrune: Diversity-based Visual Token Pruning for Large Multimodal Models`
- Official activation path:
  - import the official `LLaVA/` package from the repo
  - set `BASELINE=OURS`
  - set `LAYER_INDEX=0`
  - set `SUBSET_RATIO=0.098`
- The official README uses `lmms_eval` for paper tables. The local wrapper uses
  the same official LLaVA implementation but produces local `answers.jsonl`
  compatible with `entropy_exp/src/eval_datasets.py`.
