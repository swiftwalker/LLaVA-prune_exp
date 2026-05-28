# Official Method Token Pruning Rates

Base visual token count: 576 patch tokens for LLaVA-1.5 7B. Final/equivalent pruning rate is `(576 - retained_visual_tokens) / 576`.

| method | retained | keep rate | prune rate | schedule | rough 32-layer average prune |
|---|---:|---:|---:|---|---:|
| SparseVLM-v1-192 | 192 / 576 | 33.33% | 66.67% | retain_token=192 | N/A |
| SparseVLM-v2-192 | 192 / 576 | 33.33% | 66.67% | retain_token=192 | N/A |
| FastV-k2-r0.5 | 288 / 576 | 50.00% | 50.00% | K=2, R=0.5, attention_rank=288 | 46.88% |
| PDROP-v1.5 | 72 / 576 | 12.50% | 87.50% | layers=[8,16,24], keep ratios=[0.5,0.25,0.125] => 576->288->144->72 | 53.12% |
| VisionZip-64 | 64 / 576 | 11.11% | 88.89% | dominant=54 + contextual=10 | 88.89% |
| DivPrune-r0.098 | 56 / 576 | 9.72% | 90.28% | SUBSET_RATIO=0.098 => round(0.098*576)=56 | 90.28% |

Notes:
- The final/equivalent pruning rate is the cleanest cross-method comparison, because all methods are reduced to retained visual-token count out of 576.
- The rough 32-layer average is only a compute intuition for methods that prune after intermediate layers; it is not a measured FLOPs number.
