# SV1 Compute-Equivalent FastV / PDROP Parameters

Reference uses the official SparseVLM-v1 intended 32-layer visual-token schedule for `RETAIN_TOKN=192`:

```text
C_sv1 = 2*576 + 4*300 + 10*200 + 16*110 = 6112 visual-token-layers
avg = 6112 / 32 = 191.0 visual tokens per layer
equivalent compute pruning = 1 - 6112 / (32*576) = 66.84%
```

| method | parameters | per-layer token schedule | token-layer sum | avg tokens/layer | compute pruning |
|---|---|---|---:|---:|---:|
| SparseVLM-v1 reference | `RETAIN_TOKN=192, USE_VERSION=1_0` | 2x576 + 4x300 + 10x200 + 16x110 | 6112 | 191.00 | 66.84% |
| FastV aligned | `--fastv-k 2 --fastv-r 0.713542 --fastv-attention-rank 165` | 2x576 + 30x165 | 6102 | 190.69 | 66.89% |
| PDROP aligned | `--pdrop-layer-list '[8,16,24]' --pdrop-image-token-ratio-list '[0.248264,0.062500,0.015625]'` | 8x576 + 8x143 + 8x36 + 8x9 | 6112 | 191.00 | 66.84% |

## Commands

```bash
python3 algo_compare/scripts/run_official.py --method fastv --dataset <dataset> --variant fastv_token_mask --fastv-k 2 --fastv-r 0.713542 --fastv-attention-rank 165 --eval
python3 algo_compare/scripts/run_official.py --method pdrop --dataset <dataset> --variant pdrop_v1_5 --pdrop-layer-list '[8,16,24]' --pdrop-image-token-ratio-list '[0.248264,0.062500,0.015625]' --eval
```

## Caveats

- This aligns token-layer compute, not final retained token count.
- FastV exact solution is rank 165.333; `165` is closer to SV1 than `166` under integer token count.
- PDROP must be much more aggressive than its default because its first 8 layers remain uncompressed. The aligned final stage keeps only 9 visual tokens.
- If you prefer the nominal `192*32=6144` target instead of the official SV1 schedule sum 6112, use FastV rank `166` and PDROP ratios `[0.253472,0.064236,0.015625]`.
