# Historical Workflows

本文收纳历史分支、旧矩阵和次级工作流，避免它们和当前 `cleanup/experiment-mainline` 主线文档混在一起。

## 1. `keep-position-ids` 历史分支

`keep-position-ids` 是历史分支，用来验证物理剪枝后是否保留原始 `position_ids`。它关注的是位置编号语义，不是新的 token scoring 策略。

核心差异：

- 当前主线物理剪枝后会刷新序列状态。
- `keep-position-ids` 分支在物理剪枝后保留 surviving token 的原始 position id。
- masking 系列策略不缩短序列，因此不受 position id 重建问题影响。

历史参考结果和 parked run root 可能位于：

```text
entropy_exp/outputs/runs_keep_position_ids/
```

这个目录应按只读参考数据处理，除非明确要复现实验。

## 2. Historical canonical adaptive plan

历史 full-matrix plan：

```text
entropy_exp/plans/keep_position_ids_sparsevlm_adaptive_stratified_full_matrix.yaml
```

它的设计约束：

- strategy: `sparsevlm_adaptive_stratified`
- datasets: `gqa` / `mme` / `pope`
- prune layers: `1` / `2` / `3`
- prune ratios: `0.2` / `0.3` / `0.4` / `0.5` / `0.6` / `0.7`
- total jobs: `54`
- pool size: `12`

如果只是当前主线新实验，不要把这个 plan 当默认模板；新矩阵应在 `entropy_exp/plans/` 下另建明确 label 的 plan。

## 3. Historical result summary example

旧结果示例中常见 label：

```text
keep-position-ids-sparsevlm-adaptive-stratified-full-matrix
```

结果处理方式仍然和当前 [RESULTS_WORKFLOW.md](./RESULTS_WORKFLOW.md) 相同：

1. 从 `outputs/scheduler/<label>/attempts/*.json` 收集 completed run dirs。
2. 用 `run_eval.sh` 补齐 `eval/summary.json`。
3. 用 `summarize_results.py` 生成 summary。

区别只是该 label 代表历史分支语义，不应和当前主线结果直接混在同一个 summary 中。

## 4. Phase 1 secondary workflow

Phase 1 attention capture / entropy analysis 是保留的次级流程，不是当前 pruning benchmark 主线。详细命令保留在：

```text
entropy_exp/PHASE1_SECONDARY_WORKFLOW.md
```

适用场景：

- 只想捕获 attention 数据
- 复现早期 entropy analysis
- 检查 HDF5 capture 格式

不适用场景：

- 当前 pruning benchmark
- scheduler-first 大矩阵
- repo-native summary 主流程

## 5. 使用历史流程的安全规则

- 明确写出 `output.base_dir`，避免把历史分支结果写进当前主线 run root。
- 不要把 historical summary 和当前主线 summary 合并。
- 复现实验前先检查 `git branch --show-current`、`tmux ls` 和相关进程。
- 若要比较历史结果和当前结果，应在报告里明确标注 branch、position-id 语义和 output root。

