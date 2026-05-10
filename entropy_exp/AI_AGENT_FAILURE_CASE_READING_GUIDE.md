# AI Agent Reading Guide For Failure-Case Visualization Archive

## Purpose

This guide is written for an AI agent that needs to inspect the packaged failure-case visualization outputs and explain them accurately. The archive studies two visual-token pruning methods by finding samples where the reference method answers correctly but the analyzed method answers incorrectly, then visualizes how their kept visual patches differ across pruning layers.

The agent should treat this package as an analysis artifact, not as raw benchmark ground truth or a new model-training dataset.

## Recommended Reading Order

1. Read `docs/FAILURE_CASE_VISUALIZATION_PROJECT.md` first to understand the project goal, notation, sample-selection logic, and visualization design.
2. Read `analysis/.../summary.json` to confirm the scope of the archive, including datasets, keep levels, selected-case count, and source matrix.
3. Read `analysis/.../gap_rank.csv` to identify which dataset and keep level have the largest net failure gap.
4. Read `analysis/.../case_summary.csv` for the selected high-priority samples and their patch-difference statistics.
5. Read `analysis/.../selected_cases.jsonl` only if row-level JSON is easier than CSV for downstream processing.
6. Use `analysis/.../index.md` as a human-friendly browsing entry for the selected figures.
7. Inspect figures under `analysis/.../figures/` only after checking the corresponding row in `case_summary.csv`.

## Core Interpretation Rules

- `sv2` or `reference` means the baseline/reference pruning method.
- `ours` or `analyzed` means the entropy-alpha spatial-compensation pruning method.
- The main failure type is `sv2_correct_ours_wrong`: the reference method is correct and the analyzed method is wrong on the same sample.
- `ours_correct_sv2_wrong` is the reverse case and should be interpreted as a gain for the analyzed method.
- `net_loss = sv2_correct_ours_wrong - ours_correct_sv2_wrong`.
- A positive `net_loss` means the analyzed method loses more samples than it gains for that dataset and keep level.
- A negative `net_loss` means the analyzed method gains more samples than it loses, even if individual failure examples still exist.

## Dataset-Specific Correctness

- TextVQA uses soft answer scoring. In this archive, a sample is counted as correct if the soft score is greater than zero. Do not reinterpret TextVQA correctness as strict exact match unless explicitly asked.
- POPE is a yes/no task. Model answers are normalized to `yes` or `no` before comparison with the label.
- The package focuses only on TextVQA and POPE. Do not infer conclusions for GQA, MME, or ScienceQA from this archive alone.

## Table Field Guide

In `gap_rank.csv`:

- `dataset`: dataset name.
- `keep`: final visual-token level, such as 118, 60, 28, or 20.
- `sv2_metric`: reference-method aggregate metric.
- `ours_metric`: analyzed-method aggregate metric.
- `metric_delta`: `ours_metric - sv2_metric`.
- `metric_delta_relative_percent`: relative change from the reference method.
- `both_correct`: number of samples both methods answer correctly.
- `both_wrong`: number of samples both methods answer incorrectly.
- `sv2_correct_ours_wrong`: number of reference-only correct samples.
- `ours_correct_sv2_wrong`: number of analyzed-only correct samples.
- `net_loss`: reference-only correct count minus analyzed-only correct count.

In `case_summary.csv`:

- `score_gap`: per-sample score difference, `sv2_score - ours_score`.
- `patch_diff_score`: heuristic score for how different the kept patch sets are.
- `max_sv2_only_keep_ratio`: maximum layer-wise fraction of reference-kept patches not kept by the analyzed method.
- `max_ours_only_keep_ratio`: maximum layer-wise fraction of analyzed-kept patches not kept by the reference method.
- `max_ours_low_keep_ratio`: maximum layer-wise fraction of analyzed-method kept patches that came from low/compensation selection.
- `max_sv2_only_saliency_mass`: saliency mass assigned by the analyzed method to patches kept only by the reference method.
- `selection_score`: heuristic ranking score used to choose high-information visualization cases.

## Figure Reading Rules

Each selected case figure is organized as a matrix:

- Rows are pruning layers: layer 2, layer 6, and layer 15.
- Columns are visualization views.

The five columns are:

1. Reference keep mask: patches kept by the reference method.
2. Analyzed keep mask: patches kept by the analyzed method.
3. High/low keep mask: analyzed-method high-saliency keeps and low/compensation keeps.
4. Difference map: shared keeps, reference-only keeps, and analyzed-only keeps.
5. Saliency heatmap: analyzed-method saliency scores projected back to the image grid.

Color conventions:

- Green: kept patches in a single-method keep view.
- Yellow: analyzed-method high-saliency keep.
- Cyan: analyzed-method low/compensation keep.
- Gray: kept by both methods.
- Red: kept only by the reference method.
- Blue: kept only by the analyzed method.
- Bright heatmap color: higher analyzed-method saliency.

When explaining a figure, prefer statements such as:

- "The reference method retained this region while the analyzed method removed it."
- "The analyzed method allocated compensation tokens to this area."
- "The saliency heatmap suggests this removed region had non-trivial saliency."

Avoid overclaiming causal conclusions from a single figure. Use multiple samples with similar patterns before stating a likely failure mode.

## Safe Analysis Workflow

1. Start with the highest `net_loss` rows in `gap_rank.csv`.
2. For each target row, sort `case_summary.csv` by `selection_score` or `patch_diff_score`.
3. Inspect the top cases and compare the text question, ground truth, both answers, and patch-difference metrics.
4. Open the corresponding figure and describe patch-level differences layer by layer.
5. Summarize recurring failure patterns across at least several cases.

## Common Pitfalls To Avoid

- Do not treat `selection_score` as an official benchmark metric. It is only a diagnostic ranking heuristic.
- Do not claim the analyzed method is globally worse based only on `sv2_correct_ours_wrong`; always compare it with `ours_correct_sv2_wrong` and `net_loss`.
- Do not compare TextVQA soft-score correctness with POPE binary correctness as if they had the same statistical meaning.
- Do not assume every red region is semantically important; it is only a region kept by the reference method and not by the analyzed method.
- Do not infer that a low/compensation token is bad by default. It is suspicious only when repeated failure cases show compensation displacing task-critical regions.
- Do not ignore keep level. A pattern at 20 tokens may not apply at 118 tokens.

## Expected Archive Contents

The archive should contain:

- `README.md`: short package-level overview.
- `docs/FAILURE_CASE_VISUALIZATION_PROJECT.md`: human-facing project explanation.
- `docs/AI_AGENT_FAILURE_CASE_READING_GUIDE.md`: this AI-agent guide.
- `analysis/.../gap_rank.csv`: aggregate gap ranking.
- `analysis/.../case_summary.csv`: selected cases with diagnostic fields.
- `analysis/.../all_cases.jsonl`: all sample-level classifications.
- `analysis/.../selected_cases.jsonl`: selected visualization cases.
- `analysis/.../index.md`: figure browsing index.
- `analysis/.../figures/`: visualization PNG files.

If any of these are missing, report the missing item before drawing conclusions.
