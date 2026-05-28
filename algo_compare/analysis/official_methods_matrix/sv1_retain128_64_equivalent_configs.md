# SV1 Retain-128/64 Compute-Equivalent Experiment Configs

This file defines the experiment configuration set for aligning official comparison methods to the SparseVLM-v1 `RETAIN_TOKN=128` and `RETAIN_TOKN=64` intended token-layer budgets.

Primary convention follows the earlier `sv1compute` setting: use SparseVLM-v1 `score.py` top-k stage lists and the intended 32-layer grouping `2*576 + 4*a + 10*b + 16*c`.

## Budget Reference

| target | reference | schedule | token-layer sum | avg tokens/layer | compute prune |
|---:|---|---|---:|---:|---:|
| 128 | SparseVLM-v1 intended topk | 2x576 + 4x303 + 10x110 + 16x36 | 4040 | 126.25 | 78.08% |
| 64 | SparseVLM-v1 intended topk | 2x576 + 4x66 + 10x30 + 16x17 | 1988 | 62.12 | 89.21% |

## Method Configs

| target | label | approx token-layer sum | avg tokens/layer | compute prune | params | note |
|---:|---|---:|---:|---:|---|---|
| 128 | SparseVLM-v1-128 | 4040 | 126.25 | 78.08% | `--retain-token 128 --use-version 1_0` | Official SV1 reference; primary budget source. |
| 128 | SparseVLM-v2-128 | 4144 | 129.50 | 77.52% | `--retain-token 128 --use-version 2_0` | Official SV2 same RETAIN_TOKN reference; compute differs from SV1 schedule. |
| 128 | FastV-sv1r128compute | 4032 | 126.00 | 78.12% | `--fastv-k 2 --fastv-r 0.833333 --fastv-attention-rank 96` | Exact continuous rank 96.267; integer rank 96. |
| 128 | PDROP-sv1r128compute | 4040 | 126.25 | 78.08% | `--pdrop-layer-list '[2,6,16]' --pdrop-image-token-ratio-list '[0.526043,0.190974,0.062502]'` | Uses 1-based PDROP boundaries [2,6,16] to mirror SV1 2/4/10/16 stage lengths. |
| 128 | VisionZip-sv1r128compute | 4032 | 126.00 | 78.12% | `--visionzip-dominant 106 --visionzip-contextual 20` | Retains 126 visual tokens; dominant/contextual split preserves 54:10 ratio. |
| 128 | DivPrune-sv1r128compute | 4032 | 126.00 | 78.12% | `--divprune-subset-ratio 0.218750` | round(0.218750 * 576) = 126 retained visual tokens. |
| 64 | SparseVLM-v1-64 | 1988 | 62.12 | 89.21% | `--retain-token 64 --use-version 1_0` | Official SV1 reference; primary budget source. |
| 64 | SparseVLM-v2-64 | 2076 | 64.88 | 88.74% | `--retain-token 64 --use-version 2_0` | Official SV2 same RETAIN_TOKN reference; compute differs from SV1 schedule. |
| 64 | FastV-sv1r64compute | 1992 | 62.25 | 89.19% | `--fastv-k 2 --fastv-r 0.951389 --fastv-attention-rank 28` | Exact continuous rank 27.867; integer rank 28. |
| 64 | PDROP-sv1r64compute | 1988 | 62.12 | 89.21% | `--pdrop-layer-list '[2,6,16]' --pdrop-image-token-ratio-list '[0.114585,0.052085,0.029516]'` | Uses 1-based PDROP boundaries [2,6,16] to mirror SV1 2/4/10/16 stage lengths. |
| 64 | VisionZip-sv1r64compute | 1984 | 62.00 | 89.24% | `--visionzip-dominant 52 --visionzip-contextual 10` | Retains 62 visual tokens; dominant/contextual split preserves 54:10 ratio. |
| 64 | DivPrune-sv1r64compute | 1984 | 62.00 | 89.24% | `--divprune-subset-ratio 0.107639` | round(0.107639 * 576) = 62 retained visual tokens. |

## Dataset-Expanded Matrix

- Datasets: `gqa, textvqa, pope, mme, scienceqa`
- Total jobs: `60` = 2 targets x 6 method configs x 5 datasets

## Important Caveats

- SparseVLM-v1 has a merge branch after top-k. The primary configs intentionally use the same intended top-k budget convention as the previous 192-token `sv1compute` runs, so results remain comparable across 192/128/64 planning.
- PDROP default boundaries `[8,16,24]` cannot match SV1-128 or SV1-64 budgets because the first 8 full-token layers alone cost `4608` token-layers. The aligned configs therefore use `[2,6,16]` to mirror the SV1 2/4/10/16 stage lengths.
- VisionZip and DivPrune are single-budget methods here, so they are aligned by nearest integer average visual tokens per layer.
- SparseVLM-v2 rows are same-`RETAIN_TOKN` official references, not forced to match the SV1 token-layer sum.

## Files

- `sv1_retain128_64_budget_reference.csv`: budget formulas and diagnostic reference rows.
- `sv1_retain128_64_equivalent_configs.csv`: 12 method-level configs.
- `sv1_retain128_64_equivalent_runs.csv`: 60 dataset-expanded run rows.
- `sv1_retain128_64_equivalent_commands.sh`: dry command list for future launch scripting.
