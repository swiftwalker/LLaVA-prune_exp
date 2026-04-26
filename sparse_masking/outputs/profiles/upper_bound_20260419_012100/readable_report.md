# Phase-0 Readable Report

## Scope

- Source runs: GPU4 `baseline + masking_attn_score` and GPU5 `tail_masking_attn_score`.
- Merged profile dir: `sparse_masking/outputs/profiles/phase0_merged_gpu45_20260419_012055`
- Upper-bound dir: `sparse_masking/outputs/profiles/upper_bound_20260419_012100`
- Matrix size: 111 lightweight profile cases + 9 anchor traces.

## Executive Summary

- `masking_attn_score` keeps latency relatively close to baseline: average prefill slowdown is `+10.10%` on GQA, `+3.49%` on MME, and `+1.56%` on POPE.
- `tail_masking_attn_score` is substantially slower in the current eager implementation: average prefill slowdown is `+75.87%` on GQA, `+74.38%` on MME, and `+70.63%` on POPE.
- Under the current upper-bound model, even a perfect sparse-aware kernel only gives a modest prefill win because sequence length and QKV projection cost remain unchanged. The best observed bound is about `1.046x` prefill speedup (`layer=1`, `ratio=0.7`, tail masking).
- The same best-case bound saves about `195.2 MiB` of decode-stage KV reads per generated token (`204,685,312` bytes), so decode bandwidth reduction is more promising than prefill FLOPs reduction.

## What Is Slow Right Now

- The dominant issue is still dense eager attention plus full-length hidden-state processing. Current masking does not remove sequence positions, so the runtime does not get structural compute savings.
- `tail_masking_attn_score` is especially expensive because the custom masked path is executed repeatedly from the chosen start layer to the end of the decoder.
- The anchor traces show no evidence of a sparse compute skip today. At `layer=2, ratio=0.5`, tail masking still spends comparable or larger time in `QK`, `softmax`, and `AV` than baseline.

## Anchor Cases (`layer=2`, `ratio=0.5`)

| Dataset | Strategy | Prefill ms | QK ms | Softmax ms | AV ms | Sparse UB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GQA | baseline | 43.698 | 4.556 | 1.830 | 3.972 | 1.000x |
| GQA | basic masking | 46.614 | 2.777 | 1.261 | 2.222 | 1.001x |
| GQA | tail masking | 73.424 | 4.889 | 2.747 | 2.381 | 1.032x |
| MME | baseline | 46.011 | 2.457 | 1.172 | 2.226 | 1.000x |
| MME | basic masking | 48.078 | 2.533 | 1.204 | 2.117 | 1.001x |
| MME | tail masking | 79.254 | 5.038 | 2.690 | 2.540 | 1.032x |
| POPE | baseline | 46.680 | 2.466 | 1.140 | 2.164 | 1.000x |
| POPE | basic masking | 47.913 | 2.555 | 1.169 | 2.145 | 1.001x |
| POPE | tail masking | 79.621 | 4.643 | 2.646 | 2.340 | 1.032x |

## Dataset-Level Readout

### GQA

- Baseline prefill: `43.698 ms`
- Basic masking average prefill: `48.109 ms`
- Tail masking average prefill: `76.850 ms`
- Best sparse upper bound: `1.046x` at `gqa__tail_masking_attn_score__l1__r0p7`

### MME

- Baseline prefill: `46.011 ms`
- Basic masking average prefill: `47.617 ms`
- Tail masking average prefill: `80.234 ms`
- Best sparse upper bound: `1.046x` at `mme__tail_masking_attn_score__l1__r0p7`

### POPE

- Baseline prefill: `46.680 ms`
- Basic masking average prefill: `47.410 ms`
- Tail masking average prefill: `79.652 ms`
- Best sparse upper bound: `1.046x` at `pope__tail_masking_attn_score__l1__r0p7`

## Important Caveat

- The lightweight `attn_total_ms` field undercounts tail masking. The reason is code-path-specific: tail masking executes custom attention math inside `VisualTokenPruner._forward_masked_layer(...)` instead of going through `layer.self_attn.forward(...)`, while the lightweight timer hooks only attach to `self_attn`.
- Because of that, `prefill_ms_mean` and the 9 anchor `QK / softmax / AV` traces are the trustworthy signals for comparing baseline vs masking vs tail masking in this report.

## Recommendation

- For phase 1, prioritize a sparse-aware attention kernel path rather than additional Python-side masking variants. The current data says quality-preserving masking is viable, but runtime only improves if the kernel skips masked KV columns directly.
- Also fix profiling instrumentation before any large follow-up benchmark so that tail masking attention time is measured inside the custom masked path, not only through `self_attn` hooks.
