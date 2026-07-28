# SCND-NeXT NLCR Experiment Report

## Scope

This branch evaluates Next-Layer Counterfactual Routing (NLCR) for the first
SCND C layer on LLaVA-NeXT Vicuna-7B. NLCR compares the legacy SCND set,
saliency top-K, and unconstrained native max-min at the same keep count. It
uses exact layer-3 query-state distortion to decide whether a non-legacy set
Pareto-dominates the legacy set.

The feature is default-off. High-budget profiles bypass candidate construction
and preview before any additional Q/K/V work.

## Implementation

- Exact Llama/Mistral next-layer query-only preview with preserved sparse
  position IDs, RoPE, GQA/MHA handling, FP32 attention softmax, output
  projection, and token-wise MLP.
- Deterministic candidate construction and numerical Pareto routing.
- `none`, `audit_only`, `route`, and offline-only `force_candidate` modes.
- Canonical final keep/pruned fields plus candidate, error, timing, Jaccard,
  activation, and bypass diagnostics.
- Exact host-vectorized max-min candidate construction for NeXT anyres token
  counts. This avoids hundreds of launch-bound CUDA reductions while
  preserving float32 gain/saliency/index tie semantics.

## Stage Results

### Stage 0: forensic capture

Passed. The five prefix-400 capture runs exactly match the existing HERC v3
answers (`2000/2000` rows, zero mismatches). The report includes L2/L6/L16 set
differences, raw/rank score changes, visual roles, first-C absolute survival
mass, and correctness transitions.

### Stage 2: audit smoke and High bypass

Passed.

| Dataset | Legacy prefill (ms) | Audit prefill (ms) | Increase | Peak delta (GB) |
|---|---:|---:|---:|---:|
| POPE | 130.975 | 142.023 | 8.436% | 0.000 |
| TextVQA | 145.593 | 159.936 | 9.851% | 0.000 |
| MME | 94.746 | 104.923 | 10.741% | 0.000 |

All audit answers are exact. High route mode also preserves answers and keep
indices exactly and records a zero-cost `high_budget_profile` bypass.

### Stage 3: candidate headroom

Failed the gate. Each cell below uses the first 400 samples. Utilities are
sample-level correctness except TextVQA, which uses the official soft score.

| Dataset | Legacy | Top-K | Max-min | Oracle | Simulated route | Route gain (pp) | Comparator accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| GQA | 56.750 | 58.250 | 56.250 | 69.500 | 57.000 | +0.250 | 46.429% |
| TextVQA | 52.475 | 52.400 | 52.375 | 58.775 | 51.975 | -0.500 | 47.967% |
| POPE | 84.500 | 82.250 | 87.000 | 90.750 | 85.000 | +0.500 | 67.925% |
| MME | 65.000 | 69.000 | 62.750 | 73.500 | 65.250 | +0.250 | 30.469% |
| ScienceQA | 73.000 | 72.500 | 73.000 | 74.750 | 72.750 | -0.250 | 55.882% |

Gate summary:

- Oracle Mean Ret headroom: `+10.280 pp` (pass, required `>= +0.30 pp`).
- Counterfactual pair comparator: `280/587 = 47.700%` (fail, required
  `>55%`).
- Simulated routing aggregate utility gain: `+0.050 pp`, equivalent to about
  `+0.042 Mean Ret pp` (positive but negligible).
- Non-negative tasks: `3/5` (fail, required `>=4/5`).
- Route never selected saliency top-K under the strict Pareto rule. Across
  2,000 samples it selected legacy 1,587 times and max-min 413 times.

The candidate space has substantial oracle headroom, but exact next-layer
query-state distortion does not predict which candidate improves downstream
utility. The failure is therefore in the routing signal, not candidate
availability.

## Decision

Per the predefined gate, prefix route and full Ultra were not started. No
threshold, error weighting, or routing sweep was added after observing the
failure. NLCR remains a default-off negative ablation on this branch.

## Artifacts

- Stage 0: `entropy_exp/outputs/scnd_next_vicuna_nlcr_analysis/stage0_forensic_with_candidate_outcomes`
- Stage 2: `entropy_exp/outputs/scnd_next_vicuna_nlcr_analysis/stage2_audit_smoke_readonly_v3`
- High bypass: `entropy_exp/outputs/scnd_next_vicuna_nlcr_analysis/stage2_high_bypass`
- Stage 3: `entropy_exp/outputs/scnd_next_vicuna_nlcr_analysis/stage3_headroom`
- Stage 3 runs: `entropy_exp/outputs/scnd_next_vicuna_nlcr_candidate_headroom_prefix400_core5`

## Validation

- Scheduler: `15/15` Stage 3 jobs completed, `failed_final=0`.
- Evaluation: `15/15` summaries present.
- Unit tests: `281/281` passed.
- Stages 4 and 5: intentionally not run because Stage 3 failed.
