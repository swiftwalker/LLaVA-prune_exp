# VisionZip Official Results

Scope: official full runs only; smoke runs excluded.

## Config

- Method: `visionzip`
- Variant: `visionzip_64`
- Official commit: `8f86b55c6f000eb033e6912538af2dd7dcb30502`
- Retained visual tokens: 64 = dominant 54 + contextual 10
- Metrics: GQA/TextVQA/ScienceQA accuracy, POPE macro_f1, MME overall_total_score.

## Matrix

| dataset | metric | VisionZip-64 | answers |
|---|---|---:|---:|
| gqa | accuracy | 55.15 | 12578 |
| textvqa | accuracy | 55.50 | 5000 |
| pope | macro_f1 | 0.7708 | 8910 |
| mme | overall_total_score | 1695.13 | 2374 |
| scienceqa | accuracy | 69.98 | 4241 |

## Run Paths

| dataset | summary |
|---|---|
| gqa | `algo_compare/visionzip/outputs/official_gqa_visionzip_64_full_20260526_112201/official_summary.json` |
| textvqa | `algo_compare/visionzip/outputs/official_textvqa_visionzip_64_full_20260526_112201/official_summary.json` |
| pope | `algo_compare/visionzip/outputs/official_pope_visionzip_64_full_20260526_112201/official_summary.json` |
| mme | `algo_compare/visionzip/outputs/official_mme_visionzip_64_full_20260526_112201/official_summary.json` |
| scienceqa | `algo_compare/visionzip/outputs/official_scienceqa_visionzip_64_full_20260526_112201/official_summary.json` |
