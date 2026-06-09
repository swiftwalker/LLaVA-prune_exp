# `entropy_exp` 文档入口

本文档只回答一个问题：当前实验开发应该从哪里读起。根目录只保留主线开发文档；一次性分析报告、scheduler 状态、run dir 清单等生成物统一放在 `entropy_exp/outputs/` 下，不再作为开发文档维护。

当前工作分支：`exp/adaptive-saliency-diversity-pruning`。

## 快速导航

| 目标 | 阅读文档 | 说明 |
| --- | --- | --- |
| 了解最终定稿方法 | [SCND_GPU_FINAL_METHOD.md](./SCND_GPU_FINAL_METHOD.md) | `sparsevlm_scnd + selection_backend=gpu` 的最终口径、结果和历史索引 |
| 跑一次 pruning / baseline inference | [USAGE.md](./USAGE.md) | 单次命令、配置覆盖、输出文件 |
| 了解当前策略方法 | [STRATEGY_BRANCH_SUMMARY.md](./STRATEGY_BRANCH_SUMMARY.md) | `baseline` + 18 个剪枝策略的方法总览 |
| 理解策略如何进入 transformer block | [TRANSFORMER_BLOCK_STRATEGY_PATHS.md](./TRANSFORMER_BLOCK_STRATEGY_PATHS.md) | post / pre / masking 路径，以及新策略的运行形态 |
| 批量跑实验矩阵 | [SCHEDULER.md](./SCHEDULER.md) | scheduler plan、dry-run、tmux、resume、失败恢复 |
| 评测和汇总结果矩阵 | [RESULTS_WORKFLOW.md](./RESULTS_WORKFLOW.md) | `run_eval.sh`、`summarize_results.py`、top-k 和趋势整理 |

## 当前定稿方法

当前论文/实验笔记主线方法是 **SCND-GPU**：

```text
sparsevlm_scnd + pruning.sparsevlm_scnd.selection_backend=gpu
```

它保留 full SCND 的 saliency-constrained native diversity 语义，并把 C/B 层选择循环迁移到 GPU tensor backend。`sparsevlm_fast_scnd`、`sparsevlm_budget_candidate_scnd` 等保留为过程方法和消融对照，不再作为最终主方法表述。

## 当前事实源

- 策略入口：`entropy_exp/src/strategies/__init__.py`
- 单次运行入口：`entropy_exp/scripts/run_prune.sh`
- scheduler 策略白名单：`entropy_exp/src/scheduler.py::SUPPORTED_STRATEGIES`
- run 目录识别：`entropy_exp/src/run_layout.py::KNOWN_STRATEGIES`
- 默认配置：`entropy_exp/configs/prune.yaml`
- report / analysis 脚本：`entropy_exp/src/*_report.py`

新增策略时至少同步以上入口和本文档集合，避免出现“代码能跑但文档仍是旧策略表”的漂移。

## 当前策略家族

| 家族 | 策略 |
| --- | --- |
| 基础对照 | `baseline`、`random` |
| Attention 路径 | `attn_score`、`pre_attn_score`、`masking_attn_score`、`tail_masking_attn_score`、`entropy` |
| SparseVLM / Ours | `sparsevlm`、`sparsevlm_adaptive_stratified`、`sparsevlm_entropy_alpha`、`sparsevlm_entropy_alpha_global` |
| Boost / sampling | `sparsevlm_boost`、`sparsevlm_boost_hybrid`、`sparsevlm_compensated` |
| Diversity / SCND | `sparsevlm_diverse_mmr`、`sparsevlm_adaptive_diverse_mmr`、`sparsevlm_scnd`、`sparsevlm_fast_scnd`、`sparsevlm_budget_candidate_scnd` |

策略细节和参数含义见 [STRATEGY_BRANCH_SUMMARY.md](./STRATEGY_BRANCH_SUMMARY.md)。

## 数据集能力边界

- Inference：`gqa`、`mme`、`pope`、`textvqa`、`scienceqa`、`mmbench`、`mmvet`、`ai2d`
- Repo-native 本地评测和 summary：`gqa`、`mme`、`pope`、`textvqa`、`scienceqa`、`mmbench`、`ai2d`
- `mmvet` 当前只导出 inference-only 结果和 official/GPT judge 所需 JSON

## 文档维护原则

- 根目录文档只放稳定开发说明；单次实验报告写入 `entropy_exp/outputs/analysis/<label>/report.md`。
- 旧分支、失败案例包、远端迁移 manifest、run-dir 临时清单不再放在 root docs 中维护。
- 结果引用以 `config.yaml`、`eval/summary.json`、`summary.csv/json` 为准，文档只记录方法口径和流程口径。
