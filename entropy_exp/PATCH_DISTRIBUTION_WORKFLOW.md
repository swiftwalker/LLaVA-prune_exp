# Patch 分布分析流程

本文只覆盖 patch retention distribution：从已完成 run 的 `stats.jsonl` 统计 patch keep-rate、high/low keep-rate、stratum quota，并生成 heatmap 和对比图。

## 1. 当前支持范围

当前工具入口：

```text
entropy_exp/src/patch_distribution_report.py
```

当前内置策略顺序只覆盖：

- `baseline`
- `random`
- `sparsevlm`
- `sparsevlm_adaptive_stratified`

注意：`sparsevlm_entropy_alpha` 当前已经支持 inference、scheduler 和 summary，但 **patch distribution 工具尚未扩展到该策略**。如果要分析 entropy-alpha 的 patch 分布，需要单独扩展 `STRATEGY_ORDER`、adaptive heatmap 分支和测试。

## 2. 推荐 plan

现有 full-matrix plan：

```text
entropy_exp/plans/gqa_textvqa_patch_distribution_full_matrix.yaml
```

它面向 `GQA + TextVQA`，固定开启：

- `capture.save_attention=false`
- `capture.save_importance_scores=false`
- `capture.save_keep_indices=true`

这样可以减少磁盘占用，同时保留 patch distribution 所需的 keep indices。

## 3. 执行矩阵

```bash
python entropy_exp/scripts/run_scheduler.py \
  --plan entropy_exp/plans/gqa_textvqa_patch_distribution_full_matrix.yaml \
  --dry-run

python entropy_exp/scripts/run_scheduler.py \
  --plan entropy_exp/plans/gqa_textvqa_patch_distribution_full_matrix.yaml
```

完成后确认：

```bash
cat entropy_exp/outputs/scheduler/gqa-textvqa-patch-distribution-full-matrix/progress.txt
```

## 4. 生成 patch distribution report

```bash
python entropy_exp/src/patch_distribution_report.py \
  --state-dir entropy_exp/outputs/scheduler/gqa-textvqa-patch-distribution-full-matrix \
  --output-dir entropy_exp/outputs/analysis/patch_distribution/gqa_textvqa_patch_distribution_full_matrix
```

输出结构：

```text
entropy_exp/outputs/analysis/patch_distribution/<label>/
├── manifest/
├── tables/
├── heatmaps/
└── compare/
```

常用产物：

| 路径 | 说明 |
| --- | --- |
| `manifest/completed_run_dirs.json` | 本轮 completed run dir 清单 |
| `tables/per_config_patch_summary.csv` | 每个配置的 patch 分布摘要 |
| `heatmaps/{dataset}/{strategy}/...` | 单策略 heatmap |
| `compare/{dataset}/...` | 同一 layer/ratio 下跨策略对比图 |

## 5. 数据要求

每个 run 需要：

- `config.yaml`
- `stats.jsonl`
- `stats.jsonl` 中存在 keep indices 信息

如果运行时关闭了 `capture.save_keep_indices`，patch distribution 无法完整恢复 patch 保留分布。

## 6. 扩展到新策略的检查点

如果后续要支持 `sparsevlm_entropy_alpha`，至少同步修改：

- `entropy_exp/src/patch_distribution_report.py::STRATEGY_ORDER`
- adaptive 类策略的 high/low keep-rate 图生成逻辑
- 对应 scheduler plan 的 strategies 和 strategy-specific overrides
- `entropy_exp/tests/test_patch_distribution_report.py`
- 本文档的支持范围

