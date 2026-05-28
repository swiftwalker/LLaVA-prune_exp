# 66.67% Token-Pruning Alignment Plan

Target budget: retain 192 out of 576 LLaVA-1.5 visual patch tokens, i.e. keep rate 33.33% and pruning rate 66.67%. SparseVLM-v1/v2 already use this budget via `retain_token=192`.

| method | aligned variant | retained | keep | prune | parameters |
|---|---|---:|---:|---:|---|
| FastV | `fastv_token_mask_192` | 192 / 576 | 33.33% | 66.67% | `--fastv-k 2 --fastv-r 0.6666667 --fastv-attention-rank 192` |
| PDROP | `pdrop_v1_5_final192_geometric` | 192 / 576 | 33.33% | 66.67% | `--pdrop-layer-list '[8,16,24]' --pdrop-image-token-ratio-list '[0.693361,0.480750,0.333334]'` |
| VisionZip | `visionzip_192` | 192 / 576 | 33.33% | 66.67% | `--visionzip-dominant 162 --visionzip-contextual 30` |
| DivPrune | `divprune_r0p333334` | 192 / 576 | 33.33% | 66.67% | `--divprune-subset-ratio 0.333334` |

## Recommended Run Commands

Replace `<dataset>` with `gqa`, `textvqa`, `pope`, `mme`, or `scienceqa`.

```bash
python3 algo_compare/scripts/run_official.py --method fastv --dataset <dataset> --variant fastv_token_mask --fastv-k 2 --fastv-r 0.6666667 --fastv-attention-rank 192 --eval
python3 algo_compare/scripts/run_official.py --method pdrop --dataset <dataset> --variant pdrop_v1_5 --pdrop-layer-list '[8,16,24]' --pdrop-image-token-ratio-list '[0.693361,0.480750,0.333334]' --eval
python3 algo_compare/scripts/run_official.py --method visionzip --dataset <dataset> --variant visionzip_64 --visionzip-dominant 162 --visionzip-contextual 30 --eval
python3 algo_compare/scripts/run_official.py --method divprune --dataset <dataset> --variant divprune_r0p098 --divprune-subset-ratio 0.333334 --eval
```

## Notes

- This aligns final/equivalent retained visual tokens, not measured FLOPs.
- FastV and PDROP prune after intermediate layers, so their layer-average token budgets are still less aggressive than methods that compress before all LLM layers.
- For PDROP, the proposed ratio list is geometric from 1.0 to 1/3. A linear alternative would be `[0.75,0.50,0.333334]`, but geometric is closer to the original pyramid design.
- For VisionZip, `162+30` is exactly 3x the official quick-start `54+10`, preserving the dominant/contextual proportion.
