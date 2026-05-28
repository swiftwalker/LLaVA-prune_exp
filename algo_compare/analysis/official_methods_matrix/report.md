# Official Methods Result Matrices

Scope: full official runs only; smoke/linkcheck runs excluded. Metrics follow local unified eval: GQA/TextVQA/ScienceQA accuracy, POPE macro_f1, MME overall_total_score.

## All Official Runs

| dataset | metric | SparseVLM-v1-192 | SparseVLM-v2-192 | FastV-k2-r0.5 | PDROP-v1.5 | VisionZip-64 | DivPrune-r0.098 |
|---|---|---:|---:|---:|---:|---:|---:|
| gqa | accuracy | 59.42 | 60.89 | 60.05 | 60.22 | 55.15 | 57.01 |
| textvqa | accuracy | 57.72 | 57.96 | 58.27 | 57.44 | 55.50 | 54.21 |
| pope | macro_f1 | 0.8538 | 0.8475 | 0.8236 | 0.8475 | 0.7708 | 0.8526 |
| mme | overall_total_score | 1782.88 | 1862.84 | 1867.90 | 1857.21 | 1695.13 | 1606.99 |
| scienceqa | accuracy | 69.84 | MISSING | 69.96 | 70.12 | 69.98 | 69.68 |

## Extra Official Methods Only

| dataset | metric | FastV-k2-r0.5 | PDROP-v1.5 | VisionZip-64 | DivPrune-r0.098 |
|---|---|---:|---:|---:|---:|
| gqa | accuracy | 60.05 | 60.22 | 55.15 | 57.01 |
| textvqa | accuracy | 58.27 | 57.44 | 55.50 | 54.21 |
| pope | macro_f1 | 0.8236 | 0.8475 | 0.7708 | 0.8526 |
| mme | overall_total_score | 1867.90 | 1857.21 | 1695.13 | 1606.99 |
| scienceqa | accuracy | 69.96 | 70.12 | 69.98 | 69.68 |

## SV1 Compute-Equivalent Runs

| dataset | metric | SparseVLM-v1-192 | SparseVLM-v2-192 | FastV-sv1compute | PDROP-sv1compute | VisionZip-sv1compute | DivPrune-sv1compute |
|---|---|---:|---:|---:|---:|---:|---:|
| gqa | accuracy | 59.42 | 60.89 | 56.99 | 57.16 | 59.07 | 60.01 |
| textvqa | accuracy | 57.72 | 57.96 | 57.63 | 55.99 | 57.40 | 56.50 |
| pope | macro_f1 | 0.8538 | 0.8475 | 0.7562 | 0.7963 | 0.8566 | 0.8701 |
| mme | overall_total_score | 1782.88 | 1862.84 | 1820.39 | 1769.03 | 1760.14 | 1775.34 |
| scienceqa | accuracy | 69.84 | MISSING | 70.01 | 70.31 | 69.82 | 69.87 |

## Config Notes

- SparseVLM-v1/v2: official SparseVLM wrapper, retain token 192; SparseVLM-v2 ScienceQA full result is currently missing.
- FastV: `fastv_token_mask`, K=2, R=0.5, attention rank 288, image token length 576.
- PDROP: `pdrop_v1_5`, layers [8,16,24], ratios [0.5,0.25,0.125].
- VisionZip: `visionzip_64`, retained 64 visual tokens = dominant 54 + contextual 10.
- DivPrune: `divprune_r0p098`, SUBSET_RATIO=0.098, retained 56/576 visual tokens.

## Official Commits

- SparseVLM-v1-192: `87fe4319430e079a778c25d4ddf4588a3c4038ca`
- SparseVLM-v2-192: `87fe4319430e079a778c25d4ddf4588a3c4038ca`
- FastV-k2-r0.5: `d1659729b5bf1be225e99ee15783deeea80f63b1`
- PDROP-v1.5: `6444f304aee4b7edcb022839fb9e8cd07859704a`
- VisionZip-64: `8f86b55c6f000eb033e6912538af2dd7dcb30502`
- DivPrune-r0.098: `799e2d950aa01ba7860907f5a6d86061f885dca6`

## Files

- `official_methods_long.csv`: one row per method/dataset, with paths and completion status.
- `official_methods_matrix.csv`: SparseVLM official references plus extra methods.
- `extra_official_methods_matrix.csv`: FastV/PDROP/VisionZip/DivPrune only.
- `sv1_compute_equivalent_results_matrix.csv`: SparseVLM-v1/v2 plus SV1-compute-equivalent FastV/PDROP/VisionZip/DivPrune.
- `sv1_compute_equivalent_results_long.csv`: source paths and metadata for the SV1-compute-equivalent matrix.
- `sv1_retain128_64_equivalent_configs.md`: planned configs for SV1 retain-128/64 compute-equivalent official runs.
- `sv1_retain128_64_equivalent_configs.csv`: 12 method-level retain-128/64 configs.
- `sv1_retain128_64_equivalent_runs.csv`: 60 dataset-expanded retain-128/64 run rows.
- `official_methods_status.csv`: completeness matrix.
