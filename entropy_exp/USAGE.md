# 剪枝推理实验使用指南

本文只覆盖 **单次 pruning / baseline inference**：怎么启动、怎么覆盖配置、输出在哪里。批量矩阵、结果汇总和策略原理分别见：

- 批量调度：[SCHEDULER.md](./SCHEDULER.md)
- 结果评测与汇总：[RESULTS_WORKFLOW.md](./RESULTS_WORKFLOW.md)
- 策略方法总览：[STRATEGY_BRANCH_SUMMARY.md](./STRATEGY_BRANCH_SUMMARY.md)

## 1. 环境准备

```bash
conda activate llava
cd ~/LLaVA-prune_exp
```

确认这些路径可用：

```text
entropy_exp/models
entropy_exp/datasets
entropy_exp/eval_questions
```

当前默认配置文件是：

```text
entropy_exp/configs/prune.yaml
```

### 7B / 13B 模型切换入口

剪枝实验的模型大小由 `model.path` 和 `model.name` 决定。默认入口在
`entropy_exp/configs/prune.yaml`，当前保持 7B：

```yaml
model:
  path: "entropy_exp/models/llava-v1.5-7b"
  name: "llava-v1.5-7b"
```

如果要全局默认切到 13B，可以把这两项改成：

```yaml
model:
  path: "entropy_exp/models/llava-v1.5-13b"
  name: "llava-v1.5-13b"
```

更推荐在单次命令或 scheduler plan 中显式覆盖，避免把 7B/13B 结果混入同一实验口径：

```bash
bash entropy_exp/scripts/run_prune.sh sparsevlm gqa 4 --no-auto-gpu \
  --set model.path=entropy_exp/models/llava-v1.5-13b \
  --set model.name=llava-v1.5-13b
```

LLaVA-1.5 7B/13B 当前都使用 `v_token_num=576`、`patch_per_row=24`。13B
使用 40 层 decoder，`prune_layers` 仍按绝对层号写，例如 `[2,6,15]`；代码会从
模型 `config.json` 读取 `hidden_size`、`num_hidden_layers` 等元信息并校验层号范围。

## 2. 快速开始

推荐入口是 `run_prune.sh`：

```bash
# attention score 剪枝，限制 10 个样本
bash entropy_exp/scripts/run_prune.sh attn_score mme 10

# pre-layer 物理剪枝
bash entropy_exp/scripts/run_prune.sh pre_attn_score gqa 10

# target-layer attention logits masking
bash entropy_exp/scripts/run_prune.sh masking_attn_score gqa 10

# 从指定起始层到最后一层持续 masking
bash entropy_exp/scripts/run_prune.sh tail_masking_attn_score gqa 10 \
  --set pruning.prune_layers=[3] \
  --set pruning.prune_ratio=[0.2]

# SparseVLM text-rater 剪枝
bash entropy_exp/scripts/run_prune.sh sparsevlm mme 10

# SparseVLM + 固定 alpha 分层补偿
bash entropy_exp/scripts/run_prune.sh sparsevlm_adaptive_stratified mme 10

# SparseVLM + saliency 熵自适应 alpha 分层补偿
bash entropy_exp/scripts/run_prune.sh sparsevlm_entropy_alpha mme 10

# SparseVLM + score boosting
bash entropy_exp/scripts/run_prune.sh sparsevlm_boost mme 10

# O-S-S / O-O-S 等层级混合；S=SparseVLM，O=boost
bash entropy_exp/scripts/run_prune.sh sparsevlm_boost_hybrid mme 10 \
  --set 'pruning.prune_layers=[2,6,16]' \
  --set 'pruning.prune_ratio=[0.4791667,0.3333333,0.45]' \
  --set 'pruning.sparsevlm_boost_hybrid.layer_modes=["O","S","S"]'

# Grid-local pure-distance diversity；D=diverse，S=SparseVLM
bash entropy_exp/scripts/run_prune.sh sparsevlm_diverse_mmr pope 10 \
  --set 'pruning.prune_layers=[2,6,16]' \
  --set 'pruning.prune_ratio=[0.4791667,0.3333333,0.45]' \
  --set 'pruning.sparsevlm_diverse_mmr.layer_modes=["D","S","S"]'

# Adaptive saliency-diversity；A=adaptive diverse，S=SparseVLM
bash entropy_exp/scripts/run_prune.sh sparsevlm_adaptive_diverse_mmr pope 10 \
  --set 'pruning.prune_layers=[2,6,16]' \
  --set 'pruning.prune_ratio=[0.4791667,0.3333333,0.45]' \
  --set 'pruning.sparsevlm_adaptive_diverse_mmr.layer_modes=["A","S","S"]'

# SCND / Fast-SCND；C/F 为首层 saliency-constrained native diversity
bash entropy_exp/scripts/run_prune.sh sparsevlm_scnd pope 10 \
  --set 'pruning.prune_layers=[2,6,16]' \
  --set 'pruning.prune_ratio=[0.4791667,0.3333333,0.45]' \
  --set 'pruning.sparsevlm_scnd.layer_modes=["C","B","B"]'

bash entropy_exp/scripts/run_prune.sh sparsevlm_fast_scnd pope 10 \
  --set 'pruning.prune_layers=[2,6,16]' \
  --set 'pruning.prune_ratio=[0.4791667,0.3333333,0.45]' \
  --set 'pruning.sparsevlm_fast_scnd.layer_modes=["F","T","S"]'

# 无剪枝 baseline，同一套自定义 decode 路径
bash entropy_exp/scripts/run_prune.sh baseline mme 10
```

也可以直接调用 Python：

```bash
python entropy_exp/src/prune_inference.py \
  --config entropy_exp/configs/prune.yaml \
  --dataset mme \
  --max-samples 10

python entropy_exp/src/prune_inference.py \
  --config entropy_exp/configs/prune.yaml \
  --dataset mme \
  --max-samples 10 \
  --baseline
```

## 3. CLI 参数

### `run_prune.sh`

```bash
bash entropy_exp/scripts/run_prune.sh <strategy|baseline> <dataset|all> [max_samples] [--auto-gpu|--no-auto-gpu] [--set key=val ...]
```

| 参数 | 可选值 | 说明 |
| --- | --- | --- |
| `strategy` | `baseline` 或 17 个剪枝策略 | 当前策略列表见 [STRATEGY_BRANCH_SUMMARY.md](./STRATEGY_BRANCH_SUMMARY.md) |
| `dataset` | `gqa` / `mme` / `pope` / `textvqa` / `scienceqa` / `mmbench` / `mmvet` / `ai2d` / `all` | `mmvet` 当前为 inference-only 导出口径 |
| `max_samples` | 整数，可选 | 省略则使用配置中的 `pruning.max_samples`；仍为空则跑全量 |
| `--auto-gpu` | flag | 运行前自动选择可用显存最多的 GPU |
| `--no-auto-gpu` | flag | 不自动选择 GPU，使用当前 `CUDA_VISIBLE_DEVICES` |
| `--set` | `key=value`，可重复 | 覆盖配置项 |

### `prune_inference.py`

```bash
python entropy_exp/src/prune_inference.py \
  --config entropy_exp/configs/prune.yaml \
  --dataset gqa \
  --set pruning.strategy=sparsevlm_entropy_alpha
```

| 参数 | 说明 |
| --- | --- |
| `--config` | 配置文件路径，默认 `entropy_exp/configs/prune.yaml` |
| `--dataset` | 单个数据集名 |
| `--max-samples` | 样本数限制 |
| `--baseline` | 运行无剪枝 baseline |
| `--set KEY=VALUE` | 覆盖任意配置项，可重复 |

## 4. 配置覆盖

`--set` 支持点分隔嵌套 key，并会做基础类型推断：

```bash
bash entropy_exp/scripts/run_prune.sh sparsevlm_entropy_alpha textvqa \
  --set inference.seed=42 \
  --set pruning.prune_layers=[2] \
  --set pruning.prune_ratio=[0.3] \
  --set pruning.sparsevlm_entropy_alpha.alpha_min=0.4 \
  --set pruning.sparsevlm_entropy_alpha.alpha_max=1.0
```

常用覆盖：

| 覆盖项 | 说明 |
| --- | --- |
| `inference.seed=42` | 固定随机种子 |
| `pruning.strategy=entropy` | 切换策略 |
| `pruning.prune_layers=[2,3]` | 指定目标层 |
| `pruning.prune_ratio=[0.5,0.5]` | 每个目标层的剪枝比例 |
| `pruning.max_samples=100` | 限制样本数 |
| `capture.save_attention=true` | 保存 text->vision attention 到 `captures.h5` |
| `capture.save_importance_scores=true` | 保存 importance / visual scores 到 `stats.jsonl` |
| `capture.save_keep_indices=true` | 保存 keep / pruned indices 到 `stats.jsonl` |

常用层模式：

| 策略 | 模式字母 | 含义 |
| --- | --- | --- |
| `sparsevlm_boost_hybrid` | `S` / `O` | `S` 为纯 SparseVLM，`O` 为 boost |
| `sparsevlm_diverse_mmr` | `D` / `S` | `D` 为 grid-local pure-distance diversity，`S` 为纯 SparseVLM |
| `sparsevlm_adaptive_diverse_mmr` | `A` / `S` | `A` 为 prune-ratio 自适应 saliency-diversity，`S` 为纯 SparseVLM |
| `sparsevlm_scnd` | `C` / `B` / `S` | `C` 为 global SCND，`B` 为 boundary-only diversity refinement，`S` 为纯 SparseVLM |
| `sparsevlm_fast_scnd` | `F` / `T` / `S` | `F` 为 micro-greedy Fast-SCND，`T` 为 cached diversity tie-break，`S` 为纯 SparseVLM |

`prune_layers` 和 `prune_ratio` 的规则：

- 二者都是列表时，长度必须相同，并按位置配对。
- `prune_ratio` 是标量时，会广播到所有 `prune_layers`。
- 多层剪枝是渐进式的：后一层在前一层保留下来的 visual tokens 上继续剪。

## 5. 核心配置结构

只列运行时最常改的部分：

```yaml
model:
  path: "entropy_exp/models/llava-v1.5-7b"
  name: "llava-v1.5-7b"
  attn_implementation: "eager"

inference:
  temperature: 0
  num_beams: 1
  max_new_tokens: 128
  conv_mode: "vicuna_v1"
  seed: 42

pruning:
  strategy: "attn_score"
  layer_selection: "fixed"
  prune_layers: [2, 3]
  prune_ratio: [0.5, 0.5]
  v_token_num: 576
  max_samples: null

capture:
  save_attention: false
  capture_layers: "all"
  save_importance_scores: true
  save_keep_indices: true

output:
  base_dir: "entropy_exp/outputs"
```

策略专属参数都在 `pruning.<strategy_name>` 下。当前策略数量、方法差异和参数方向见 [STRATEGY_BRANCH_SUMMARY.md](./STRATEGY_BRANCH_SUMMARY.md)。

## 6. 输出文件

单次运行会创建：

```text
entropy_exp/outputs/runs/{strategy}/{dataset}/{run_name}/
```

常见文件：

| 文件 | 说明 |
| --- | --- |
| `config.yaml` | 本次运行的完整配置快照，包含 `--set` 覆盖后结果 |
| `answers.jsonl` | 模型输出 |
| `stats.jsonl` | 剪枝统计；是否包含 scores/indices 由 `capture` 配置控制 |
| `captures.h5` | 可选 attention 捕获文件，仅 `capture.save_attention=true` 时生成 |
| `eval/summary.json` | 评测后生成，不由 inference 自动生成 |

评测和汇总请走 [RESULTS_WORKFLOW.md](./RESULTS_WORKFLOW.md)。批量实验请走 [SCHEDULER.md](./SCHEDULER.md)。

## 7. 数据集能力

| 数据集 | Inference | Repo-native local eval/summary |
| --- | --- | --- |
| `gqa` | 支持 | 支持，主指标 `accuracy` |
| `mme` | 支持 | 支持，主指标 `overall_total_score` |
| `pope` | 支持 | 支持，主指标 `macro_f1` |
| `textvqa` | 支持 | 支持，主指标 `accuracy` |
| `scienceqa` | 支持 | 支持，主指标 `accuracy` |
| `mmbench` | 支持 | 支持，主指标 `accuracy` |
| `ai2d` | 支持 | 支持，主指标 `accuracy` |
| `mmvet` | 支持 | inference-only，导出 official/GPT judge JSON |
