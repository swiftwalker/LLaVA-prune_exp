# PDROP / FastV Official Results

Scope: official full runs only; smoke runs excluded.

## Configs

- FastV: `fastv_token_mask`, K=2, R=0.5, attention rank=288, image token length=576.
- PDROP: `pdrop_v1_5`, layers=[8,16,24], image_token_ratio_list=[0.5,0.25,0.125].
- Metrics: GQA/TextVQA/ScienceQA accuracy, POPE macro_f1, MME overall_total_score.

## Matrix

| dataset | metric | FastV | PDROP | PDROP-FastV | winner |
|---|---|---:|---:|---:|---|
| gqa | accuracy | 60.05 | 60.22 | +0.17 | pdrop |
| textvqa | accuracy | 58.27 | 57.44 | -0.83 | fastv |
| pope | macro_f1 | 0.8236 | 0.8475 | +0.0239 | pdrop |
| mme | overall_total_score | 1867.90 | 1857.21 | -10.68 | fastv |
| scienceqa | accuracy | 69.96 | 70.12 | +0.17 | pdrop |

## Run Status

| method | dataset | answers | summary |
|---|---|---:|---|
| fastv | gqa | 12578 | `algo_compare/fastv/outputs/official_gqa_fastv_token_mask_k2_r0p5_full_20260525_153346/official_summary.json` |
| pdrop | gqa | 12578 | `algo_compare/pdrop/outputs/official_gqa_pdrop_v1_5_layers8-16-24_ratios0p5-0p25-0p125_full_20260526_094709/official_summary.json` |
| fastv | textvqa | 5000 | `algo_compare/fastv/outputs/official_textvqa_fastv_token_mask_k2_r0p5_full_compat_20260526_102809/official_summary.json` |
| pdrop | textvqa | 5000 | `algo_compare/pdrop/outputs/official_textvqa_pdrop_v1_5_layers8-16-24_ratios0p5-0p25-0p125_full_20260526_094709/official_summary.json` |
| fastv | pope | 8910 | `algo_compare/fastv/outputs/official_pope_fastv_token_mask_k2_r0p5_full_20260525_153346/official_summary.json` |
| pdrop | pope | 8910 | `algo_compare/pdrop/outputs/official_pope_pdrop_v1_5_layers8-16-24_ratios0p5-0p25-0p125_full_20260526_094709/official_summary.json` |
| fastv | mme | 2374 | `algo_compare/fastv/outputs/official_mme_fastv_token_mask_k2_r0p5_full_20260525_153346/official_summary.json` |
| pdrop | mme | 2374 | `algo_compare/pdrop/outputs/official_mme_pdrop_v1_5_layers8-16-24_ratios0p5-0p25-0p125_full_20260526_094709/official_summary.json` |
| fastv | scienceqa | 4241 | `algo_compare/fastv/outputs/official_scienceqa_fastv_token_mask_k2_r0p5_full_compat_20260526_102809/official_summary.json` |
| pdrop | scienceqa | 4241 | `algo_compare/pdrop/outputs/official_scienceqa_pdrop_v1_5_layers8-16-24_ratios0p5-0p25-0p125_full_20260526_094709/official_summary.json` |

## Notes

- FastV TextVQA and ScienceQA use the local compatibility patch documented in `algo_compare/fastv/patches/fastv_token_mask_bounds_guard.patch`; `max_expanded_tokens=0`, so FastV token-mask is not disabled by length fallback.
- ScienceQA contains no-image examples; those naturally have no visual tokens to prune.
