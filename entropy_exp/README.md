# `entropy_exp` 文档入口

本文档只回答一个问题：你现在应该看哪一篇文档。

当前主线按本工作区事实处理为 `cleanup/experiment-mainline`。历史分支、旧矩阵和二级工作流会明确标注为 historical reference，避免和当前实验主线混用。

## 快速导航

| 目标 | 阅读文档 | 说明 |
| --- | --- | --- |
| 跑一次 pruning / baseline inference | [USAGE.md](./USAGE.md) | 单次命令、配置覆盖、输出文件 |
| 了解当前有哪些策略分支 | [STRATEGY_BRANCH_SUMMARY.md](./STRATEGY_BRANCH_SUMMARY.md) | `baseline` + 9 个剪枝策略的方法总览 |
| 理解策略如何改 transformer block | [TRANSFORMER_BLOCK_STRATEGY_PATHS.md](./TRANSFORMER_BLOCK_STRATEGY_PATHS.md) | pre/post/masking 对 hidden、KV、position、logits 的影响 |
| 批量跑实验矩阵 | [SCHEDULER.md](./SCHEDULER.md) | scheduler plan、dry-run、tmux、resume、失败恢复 |
| 评测和汇总结果矩阵 | [RESULTS_WORKFLOW.md](./RESULTS_WORKFLOW.md) | `run_eval.sh`、`summarize_results.py`、top-k 和趋势整理 |
| 分析 patch 保留分布 | [PATCH_DISTRIBUTION_WORKFLOW.md](./PATCH_DISTRIBUTION_WORKFLOW.md) | 当前 patch distribution 工具的支持范围和使用方式 |
| 查历史/次级流程 | [HISTORICAL_WORKFLOWS.md](./HISTORICAL_WORKFLOWS.md) | `keep-position-ids`、Phase 1 attention capture 等历史参考 |

## 当前事实源

- 策略入口：`entropy_exp/src/strategies/__init__.py`
- 单次运行入口：`entropy_exp/scripts/run_prune.sh`
- scheduler 策略白名单：`entropy_exp/src/scheduler.py::SUPPORTED_STRATEGIES`
- run 目录识别：`entropy_exp/src/run_layout.py::KNOWN_STRATEGIES`
- 默认配置：`entropy_exp/configs/prune.yaml`
- 7B/13B 切换入口：`model.path` 和 `model.name`；默认 7B，13B 建议通过
  `--set model.path=entropy_exp/models/llava-v1.5-13b --set model.name=llava-v1.5-13b`
  或 plan override 显式启用，详见 [USAGE.md](./USAGE.md)

## 能力边界

- Inference 数据集：`gqa`、`mme`、`pope`、`textvqa`、`scienceqa`、`mmbench`、`mmvet`、`ai2d`
- Repo-native 本地评测和 summary：`gqa`、`mme`、`pope`、`textvqa`、`scienceqa`、`mmbench`、`ai2d`
- `mmvet` 当前只导出 inference-only 结果和 official/GPT judge 所需 JSON
- Patch distribution 当前只覆盖 `baseline`、`random`、`sparsevlm`、`sparsevlm_adaptive_stratified`
