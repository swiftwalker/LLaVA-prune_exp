# Prompt: Continue Optimizing SCND Ultra on LLaVA-NeXT Vicuna-7B

Use the following prompt for the next research and implementation round.

```text
You are working on SCND (also called ProSCD in the paper) for
LLaVA-NeXT Vicuna-7B.

Repository:
https://github.com/swiftwalker/LLaVA-prune_exp

Branch:
exp/scnd-next-entropy-recalibration

Before proposing changes, inspect the branch implementation, tests, experiment
plans, and this report:
entropy_exp/SCND_NEXT_HERC_ULTRA_REPORT.md

Task
====

Design the next technically defensible method for improving SCND at the Ultra
compression point. Start with a code-level diagnosis and an experiment plan.
Do not implement or launch experiments until the plan has been reviewed.

Fixed experimental setting
==========================

- Model: LLaVA-NeXT Vicuna-7B (`llava-v1.6-vicuna-7b`)
- Conversation mode: `vicuna_v1`
- Pruning layers: `[2,6,16]`
- Layer modes: `C-B-S`
- Ultra removal ratios:
  `[0.8854166666666666,0.5454545454545454,0.43333333333333335]`
- Five tasks: GQA, TextVQA, POPE, MME, ScienceQA
- Same final keep counts and equivalent token budget as the existing runs
- Training-free, deterministic seed 42
- No dataset-specific parameters, labels, or calibration artifacts
- High compression behavior must remain exactly unchanged unless a separate
  ablation explicitly proves an improvement

Current evidence
================

1. Full-dataset Ultra Mean Ret%:
   - Improved SCND: 92.249
   - DivPrune: 93.629
   - CDPruner: approximately 94.791, but this number remains conditional on
     the official 672-resolution adjudication

2. Entropy recalibration restored useful variation on LLaVA-NeXT, but SCND is
   still 1.380 Mean Ret% below DivPrune at Ultra.

3. HERC v2 used post-selection evidence and coverage swaps. It improved some
   GQA/TextVQA/MME prefix metrics but caused a clear POPE regression.

4. Conservative HERC v3 protected evidence anchors and accepted only swaps
   that were Pareto-safe for all available proxies: scalar saliency mass,
   per-rater mass, spatial hierarchy coverage, and facility coverage. Despite
   satisfying these constraints to FP32 tolerance, it regressed against legacy
   SCND on every prefix task:
   - GQA: -0.500 percentage points
   - TextVQA: -0.175 percentage points
   - POPE macro-F1: -0.006476 absolute
   - MME: -5.000 points
   - ScienceQA: -0.250 percentage points

5. Therefore aggregate saliency mass, per-rater coverage, spatial quotas, and
   facility coverage are not sufficient statistics for answer-critical token
   identity or token interactions. Do not assume that satisfying these proxies
   makes a replacement semantically safe.

Design constraints
==================

- Do not propose another scalar-weight sweep, fixed spatial quota, or a larger
  stack of post-hoc HERC guards as the primary solution.
- Do not replace SCND with DPP, clustering, or random coverage without a causal
  argument tied to the observed failure.
- Prefer changing the original C-layer selection objective, introducing a
  principled no-replacement confidence test, or using cross-layer evidence
  stability before destructive decisions.
- Preserve exact token budgets, anyres image layouts, batch-size-one inference,
  and backward compatibility of old modes.
- Added computation must be quantified. Avoid an extra full model forward or
  any training stage.
- Distinguish query-conditioned evidence from generic context coverage. The v2
  ablation showed that both can be harmful in different ways.

Directions worth evaluating, not mandatory conclusions
=======================================================

1. Selection-time robust multi-rater objective:
   preserve candidate identity and interactions while optimizing a worst-case
   or lower-confidence-bound score across current and reference saliency views.

2. Cross-layer evidence stability:
   estimate whether a token or local token group remains salient under nearby
   layers/raters, then use stability to gate pruning rather than repairing a
   selected set afterward.

3. Survival-aware progressive allocation:
   optimize the C-layer set for downstream survival through B and S, instead of
   maximizing only immediate diversity at L2.

4. Interaction-sensitive evidence:
   model local complementarity or redundancy using attention transport,
   low-cost local Jacobian/logit sensitivity, or another forward-available
   signal. Explain how the signal captures information that aggregate mass and
   facility coverage miss.

Required deliverables
=====================

1. A code-level root-cause analysis of the current C/B/S pipeline, including
   where answer-critical identity can be lost and why Ultra magnifies it.
2. At least three candidate mechanisms, each with:
   - mathematical objective;
   - causal rationale;
   - expected compute and memory cost;
   - failure mode and explicit rejection criterion.
3. Recommend one mechanism and provide precise equations and pseudocode.
4. Show how exact per-layer keep counts are preserved and how the High path can
   take a bitwise-equivalent legacy bypass.
5. Specify config keys, diagnostic fields, unit tests, CUDA tests, and rollback
   behavior. New behavior must be explicit and default-off first.
6. Design a staged experiment schedule:
   - toy/unit invariants;
   - real GPU smoke;
   - five-task prefix-400 gate against legacy Improved SCND;
   - full Ultra matrix only after the prefix gate passes.
7. Define a strict go/no-go gate. At minimum, the candidate should avoid clear
   regressions on all five prefix tasks and should improve their aggregate. A
   full run should target Ultra Mean Ret% above DivPrune's 93.629, equivalent to
   at least +1.380 points over current Improved SCND.
8. State what negative result would falsify the proposed mechanism rather than
   merely motivate another hyperparameter sweep.

Reporting discipline
====================

- Report raw task metrics as well as Mean Ret%.
- Keep CDPruner conclusions conditional until the official 672-resolution
  adjudication is complete.
- Do not describe a proxy-preserving swap as semantically safe without direct
  evidence.
- If the prefix gate fails, stop the full run and report the negative result.
```
