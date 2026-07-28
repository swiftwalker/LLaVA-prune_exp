# SCND on LLaVA-NeXT Vicuna-7B: HERC Ultra Report

## 1. Scope

This report records the HERC investigation on the branch
`exp/scnd-next-entropy-recalibration`. The goal was to improve SCND at the
LLaVA-NeXT Vicuna-7B Ultra compression point without changing the per-layer
token budget, adding training, or introducing dataset-specific parameters.

The result is a negative but informative gate: conservative HERC v3 reduces
the POPE regression caused by HERC v2, but does not improve the five-task
prefix benchmark. A full Ultra run was therefore not launched.

## 2. Protocol

- Model: `llava-v1.6-vicuna-7b`
- Conversation mode: `vicuna_v1`
- Pruning layers: `[2, 6, 16]`
- Layer modes: `C-B-S`
- Ultra removal ratios:
  `[0.8854166666666666, 0.5454545454545454, 0.43333333333333335]`
- Selection backend: GPU
- Datasets: GQA, TextVQA, POPE, MME, ScienceQA
- Gate size: first 400 evaluation samples per dataset
- Seed: 42
- Entropy calibration: `budget_adaptive_quantile`
- Visual role constraint:
  `budget_adaptive_saliency_gated_local_floor`

The legacy and HERC variants use identical model, sample order, pruning
layers, removal ratios, and final keep counts.

## 3. Starting Point

The current full-dataset Improved SCND Ultra result is 92.249 Mean Ret%.
The comparison targets available in the workspace are:

| Method | Ultra Mean Ret% | Delta over Improved SCND |
|---|---:|---:|
| Improved SCND | 92.249 | 0.000 |
| DivPrune | 93.629 | +1.380 |
| CDPruner | approximately 94.791 | +2.542 |

The CDPruner number remains subject to the separate official 672-resolution
adjudication. The unambiguous target for an SCND improvement is therefore
DivPrune at 93.629 Mean Ret%.

## 4. HERC Iterations

### 4.1 HERC v1

HERC v1 added per-rater query evidence, anyres hierarchy coverage, facility
coverage, and first-C continuity as budget-neutral post-selection swaps.
Its weighted harmonic hierarchy aggregation collapsed under anyres layouts:
almost every missing low-mass group drove coverage toward zero, producing an
almost maximal 25% swap budget. The full v1 matrix was stopped after this
failure mode was established.

### 4.2 HERC v2

HERC v2 replaced harmonic hierarchy aggregation with evidence-mass-weighted
aggregation. This restored an adaptive budget, but a five-task prefix-400 gate
showed a clear task trade-off:

| Variant | GQA | TextVQA | POPE F1 | MME | ScienceQA |
|---|---:|---:|---:|---:|---:|
| Legacy Improved SCND | 58.000 | 52.125 | 0.834254 | 157.778 | 73.000 |
| HERC v2, query + context | 58.000 | 53.375 | 0.817927 | 160.556 | 72.750 |
| HERC v2, context only | 58.750 | 53.375 | 0.817927 | 161.389 | 72.750 |
| HERC v2, query only | 56.250 | 52.475 | 0.828729 | 151.944 | 73.000 |
| HERC v2, C-only full stages | 57.500 | 53.125 | 0.821229 | 157.222 | 72.750 |

The stage ablation establishes two causal points:

1. Active query reconciliation is harmful to GQA and MME.
2. Context reconciliation produces the POPE regression even when it preserves
   aggregate per-rater evidence coverage.

Facility coverage at L2 was already approximately 0.98. Most v2 context swaps
were therefore driven by row and macrocell deficits rather than a meaningful
representation deficit.

### 4.3 HERC v3

HERC v3 was implemented as an explicit, default-off mode with four changes:

1. Evidence anchors: for scalar, current-rater, and first-C-reference channels,
   protect the smallest selected-token prefix that carries the effective
   saliency-floor fraction of selected evidence mass.
2. Context-only gate: the formal experiment disables active query swaps.
3. Co-deficit budget: context budget is proportional to the geometric mean of
   hierarchy and facility deficits, so either near-saturated component closes
   the budget.
4. Pareto-safe swaps: every accepted swap must not reduce scalar mass, any
   current/reference rater coverage, view/macrocell/row coverage, or facility
   coverage.

The implementation preserves the old `none`, `audit_only`, `herc_v1`, and
`herc_v2` semantics. High compression settings with zero profile pressure still
take the exact legacy bypass before constructing HERC tensors.

## 5. HERC v3 Results

| Variant | GQA | TextVQA | POPE F1 | MME | ScienceQA |
|---|---:|---:|---:|---:|---:|
| Legacy Improved SCND | 58.000 | 52.125 | **0.834254** | **157.778** | **73.000** |
| HERC v2, context only | **58.750** | **53.375** | 0.817927 | 161.389 | 72.750 |
| HERC v3, context only | 57.500 | 51.950 | 0.827778 | 152.778 | 72.750 |
| v3 minus legacy | -0.500 | -0.175 | -0.006476 | -5.000 | -0.250 |

HERC v3 recovers about 0.985 F1 points relative to v2 on POPE, but remains
about 0.648 F1 points below the legacy selector. It also removes the v2 gains
on GQA, TextVQA, and MME. The variant fails the prefix gate on every task.

## 6. Selector Diagnostics

| Dataset | L2 context swaps | L6 context swaps | L16 context swaps | L2 anchor fraction |
|---|---:|---:|---:|---:|
| GQA | 3.483 | 0.980 | 0.005 | 0.535 |
| TextVQA | 3.933 | 0.965 | 0.003 | 0.532 |
| POPE | 3.590 | 0.985 | 0.005 | 0.528 |
| MME | 3.658 | 0.990 | 0.003 | 0.513 |
| ScienceQA | 1.593 | 0.450 | 0.000 | 0.250 |

The remaining guard deviations are limited to FP32 rounding, with a maximum
observed magnitude of approximately `1.19e-7`. The method therefore satisfies
its declared proxy constraints, yet performance still regresses.

This is the main finding: scalar mass, per-rater mass coverage, spatial
hierarchy coverage, and facility coverage are not sufficient statistics for
answer-critical semantic evidence. A swap can be Pareto-safe under all four
proxies and still remove a token whose identity or local interaction is
causally important to generation.

## 7. Verification

- Full unit suite: 261/261 passed.
- Real POPE GPU smoke: 20/20 samples completed.
- Prefix scheduler: 5/5 jobs completed, 0 failed.
- Evaluation summaries: 5/5 present.
- Full Ultra HERC v3: not launched because the prefix gate failed.

Authoritative result root:

```text
entropy_exp/outputs/scnd_next_vicuna_herc_v3_context_only_prefix400_core5
```

## 8. Decision

HERC v3 should be retained as a negative ablation and diagnostic backend, not
used as the default SCND selector and not reported as an Ultra improvement.
The next optimization should not add another coverage weight or post-hoc swap
guard. It should instead reconsider how query-conditioned evidence and
representation coverage enter the original selection process, or develop a
stronger no-swap confidence criterion tied to cross-layer evidence stability.
