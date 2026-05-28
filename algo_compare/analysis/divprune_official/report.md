# DivPrune Official Results

Scope: official full runs only; smoke runs excluded.

## Config

- Method: `divprune`
- Variant: `divprune_r0p098`
- Official commit: `799e2d950aa01ba7860907f5a6d86061f885dca6`
- DivPrune params: `BASELINE=OURS`, `LAYER_INDEX=0`, `SUBSET_RATIO=0.098`
- Retained visual tokens: 56 / 576
- Metrics: GQA/TextVQA/ScienceQA accuracy, POPE macro_f1, MME overall_total_score.

## Matrix

| dataset | metric | DivPrune-r0.098 | answers | complete |
|---|---|---:|---:|---|
| gqa | accuracy | 57.01 | 12578 | true |
| textvqa | accuracy | 54.21 | 5000 | true |
| pope | macro_f1 | 0.8526 | 8910 | true |
| mme | overall_total_score | 1606.99 | 2374 | true |
| scienceqa | accuracy | 69.68 | 4241 | true |

## Run Paths

| dataset | summary |
|---|---|
| gqa | `algo_compare/divprune/outputs/official_gqa_divprune_r0p098_full_20260526_140456/official_summary.json` |
| textvqa | `algo_compare/divprune/outputs/official_textvqa_divprune_r0p098_full_20260526_140456/official_summary.json` |
| pope | `algo_compare/divprune/outputs/official_pope_divprune_r0p098_full_20260526_140456/official_summary.json` |
| mme | `algo_compare/divprune/outputs/official_mme_divprune_r0p098_full_20260526_140456/official_summary.json` |
| scienceqa | `algo_compare/divprune/outputs/official_scienceqa_divprune_r0p098_full_20260526_140456/official_summary.json` |

## Notes

- Five full datasets completed and evaluated successfully.
- The wrapper uses local dataset/eval compatibility while leaving the official DivPrune algorithm path isolated under `algo_compare/divprune/third_party/`.
