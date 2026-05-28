# SV1 Compute-Equivalent Official Results

Scope: compare SparseVLM official v1/v2 retain-192 references with FastV/PDROP/VisionZip/DivPrune parameters aligned to the SparseVLM-v1 32-layer token-layer budget where available. Metrics use the local unified eval: GQA/TextVQA/ScienceQA accuracy, POPE macro_f1, MME overall_total_score.

| dataset | metric | SparseVLM-v1-192 | SparseVLM-v2-192 | FastV-sv1compute | PDROP-sv1compute | VisionZip-sv1compute | DivPrune-sv1compute |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gqa | accuracy | 59.42 | 60.89 | 56.99 | 57.16 | 59.07 | 60.01 |
| textvqa | accuracy | 57.72 | 57.96 | 57.63 | 55.99 | 57.40 | 56.50 |
| pope | macro_f1 | 0.8538 | 0.8475 | 0.7562 | 0.7963 | 0.8566 | 0.8701 |
| mme | overall_total_score | 1782.88 | 1862.84 | 1820.39 | 1769.03 | 1760.14 | 1775.34 |
| scienceqa | accuracy | 69.84 | MISSING | 70.01 | 70.31 | 69.82 | 69.87 |

## Notes

- `SparseVLM-v1-192` and `SparseVLM-v2-192` are official SparseVLM wrapper results with `RETAIN_TOKN=192`.
- `FastV-sv1compute`: `K=2, R=0.713542, attention_rank=165`, approximate token-layer budget 6102.
- `PDROP-sv1compute`: layers `[8,16,24]`, ratios `[0.248264,0.062500,0.015625]`, token-layer budget 6112.
- `VisionZip-sv1compute`: dominant/contextual `161/30`, retained visual tokens 191.
- `DivPrune-sv1compute`: `subset_ratio=0.331598`, retained visual tokens 191.
- SparseVLM-v2 ScienceQA full official result is not present locally, so it is marked `MISSING`.

## Files

- `sv1_compute_equivalent_results_matrix.csv`: wide matrix with SV1/SV2 and aligned official methods.
- `sv1_compute_equivalent_results_long.csv`: run-level source paths and metadata.
- `sv1_compute_equivalent_results_status.csv`: completeness status for each dataset/method key.
