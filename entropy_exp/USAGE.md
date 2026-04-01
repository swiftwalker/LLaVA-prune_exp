# 剪枝推理实验 — 使用指南

## 〇、环境准备

```bash
conda activate llava
cd ~/LLaVA-prune_exp
```

确保以下软链接存在：

```
entropy_exp/models         → ~/models
entropy_exp/datasets       → ~/datasets/SparseVLMs/data
entropy_exp/eval_questions → ~/datasets/SparseVLMs/eval
```

---

## 一、快速开始

### 1.1 Shell 脚本（推荐）

```bash
# 用 attn_score 策略跑 MME，限 10 个样本（冒烟测试）
bash entropy_exp/scripts/run_prune.sh attn_score mme 10

# 用 masking_attn_score 策略跑 GQA，当前层 mask 掉被剪枝视觉 token
bash entropy_exp/scripts/run_prune.sh masking_attn_score gqa 10

# 用 entropy 策略跑 POPE 全量
bash entropy_exp/scripts/run_prune.sh entropy pope

# 用 random 策略跑 MME（由 inference.seed 控制可复现）
bash entropy_exp/scripts/run_prune.sh random mme 10

# 用 sparsevlm 策略跑 MME（text rater + text->vision attention）
bash entropy_exp/scripts/run_prune.sh sparsevlm mme 10

# 跑无剪枝 baseline（使用同一 decode 路径，计时公平）
bash entropy_exp/scripts/run_prune.sh baseline mme 10

# 跑所有三个数据集
bash entropy_exp/scripts/run_prune.sh attn_score all
```

**参数说明：**

| 位置 | 参数 | 可选值 | 说明 |
|:--|:--|:--|:--|
| $1 | strategy | `attn_score` / `pre_attn_score` / `masking_attn_score` / `entropy` / `random` / `sparsevlm` / `baseline` | 剪枝策略；`pre_attn_score` 在目标层前物理裁剪，`masking_attn_score` 只在目标层 attention logits 中屏蔽被剪枝视觉 token，`random` 使用固定 seed 可复现，`sparsevlm` 使用 text raters |
| $2 | dataset | `gqa` / `mme` / `pope` / `all` | 数据集 |
| $3 | max_samples | 整数（可选） | 限制样本数，省略则跑全量 |
| -- | `--set key=val` | 任意（可多次） | 覆盖 yaml 配置项，见 §1.3 |

### 1.2 直接调用 Python

```bash
# 剪枝推理
python entropy_exp/src/prune_inference.py \
    --config entropy_exp/configs/prune.yaml \
    --dataset mme \
    --max-samples 10

# 无剪枝 baseline
python entropy_exp/src/prune_inference.py \
    --config entropy_exp/configs/prune.yaml \
    --dataset mme \
    --max-samples 10 \
    --baseline
```

**CLI 参数：**

| 参数 | 说明 |
|:--|:--|
| `--config` | 配置文件路径，默认 `entropy_exp/configs/prune.yaml` |
| `--dataset` | 数据集名，必选：`gqa` / `mme` / `pope` |
| `--max-samples` | 限制样本数（可选） |
| `--baseline` | 加此 flag 则不剪枝，用于对照 |
| `--set KEY=VALUE` | 覆盖配置项（可多次使用），见下文 §1.3 |

### 1.3 `--set` 命令行覆盖机制

不需要复制 yaml 文件，用 `--set` 即可在命令行中覆盖任意配置项。支持 **点分隔嵌套 key** 和自动类型推导：

```bash
# 覆盖策略和剪枝层/比例
python entropy_exp/src/prune_inference.py \
    --config entropy_exp/configs/prune.yaml \
    --dataset mme \
    --set pruning.strategy=entropy \
    --set pruning.prune_layers=[2,5,10] \
    --set pruning.prune_ratio=[0.3,0.4,0.5]

# 覆盖嵌套参数
python entropy_exp/src/prune_inference.py \
    --config entropy_exp/configs/prune.yaml \
    --dataset mme \
    --set pruning.entropy.dynamic_ratio=true \
    --set pruning.entropy.max_prune_ratio=0.8

# 通过 shell 脚本传递 --set
bash entropy_exp/scripts/run_prune.sh attn_score mme 10 \
    --set pruning.prune_layers=[5] \
    --set pruning.prune_ratio=[0.7]
```

**自动类型推导规则：**

| 输入 | 解析结果 | 类型 |
|:--|:--|:--|
| `42` | `42` | int |
| `0.5` | `0.5` | float |
| `true` / `false` | `True` / `False` | bool |
| `null` / `none` | `None` | NoneType |
| `[2,5,10]` | `[2, 5, 10]` | list (JSON) |
| `[0.3,0.5]` | `[0.3, 0.5]` | list (JSON) |
| `entropy` | `"entropy"` | str |
| `pre_attn_score` | `"pre_attn_score"` | str |
| `masking_attn_score` | `"masking_attn_score"` | str |
| `random` | `"random"` | str |
| `sparsevlm` | `"sparsevlm"` | str |

---

## 二、配置文件详解

配置文件：**`entropy_exp/configs/prune.yaml`**

### 2.1 完整配置结构

```yaml
model:
  path: "entropy_exp/models/llava-v1.5-7b"
  name: "llava-v1.5-7b"
  attn_implementation: "eager"      # 必须为 eager 才能获取 attention weights

inference:
  temperature: 0                    # 0 = 确定性推理
  top_p: null
  num_beams: 1
  max_new_tokens: 128               # 最大生成 token 数
  conv_mode: "vicuna_v1"
  seed: 42                          # 随机种子

pruning:
  strategy: "attn_score"            # 剪枝策略：attn_score / pre_attn_score / masking_attn_score / entropy / random / sparsevlm
  layer_selection: "fixed"          # 层选择方法
  prune_layers: [2, 3]             # 剪枝层列表
  prune_ratio: [0.5, 0.5]          # 对应每层的剪枝比例
  v_token_num: 576                  # 视觉 token 数量
  max_samples: null                 # 样本数限制
  attn_score: {}                    # attn_score 策略额外参数
  pre_attn_score: {}                # pre_attn_score 策略额外参数
  masking_attn_score: {}            # masking_attn_score 策略额外参数
  entropy:                          # entropy 策略额外参数
    dynamic_ratio: false
    dynamic_scale: 0.5
    max_prune_ratio: 0.9
  random: {}                        # random 策略额外参数（当前为空）
  sparsevlm:                        # sparsevlm 策略额外参数
    fallback_topk: 4
    exclude_special_tokens: true
    min_visual_tokens_after_prune: 16

capture:
  save_attention: false             # 保存注意力矩阵到 captures.h5（HDF5）
  capture_layers: "all"             # 捕获哪些层："all" 或层索引列表如 [0, 1, 2, 3]
  save_importance_scores: true      # 保存重要性分数到 stats.jsonl
  save_keep_indices: true           # 保存保留索引到 stats.jsonl
  precision: "fp16"                 # HDF5 存储精度
  compression: "gzip"               # HDF5 压缩方式
  compression_opts: 4               # 压缩等级

datasets: ...                       # 数据集路径
output:
  base_dir: "entropy_exp/outputs"   # 输出根目录，每次运行创建 runs/{tag}/ 子目录
```

### 2.2 三者关系：`strategy` / `prune_layers` / `prune_ratio`

这三个参数**互相独立、正交组合**，各司其职：

```
strategy     →  决定「怎么算重要性」（importance 计算方法）
prune_layers →  决定「在哪里剪」（哪些层执行剪枝）
prune_ratio  →  决定「剪多少」（每层移除 visual token 的比例）
```

它们在代码中的协作流程：

```
prune.yaml
  ├─ pruning.strategy: "attn_score"     ─┐
  ├─ pruning.attn_score: {}              ─┤  ① get_strategy(name, config)
  │  └─ (或 pruning.pre_attn_score/masking_attn_score/entropy/random/sparsevlm: {})  ─┘
  │                                              → 实例化对应策略
  │
  ├─ pruning.prune_layers: [2, 3]       ─┐
  │                                       ├  ② VisualTokenPruner.__init__()
  └─ pruning.prune_ratio: [0.5, 0.5]   ─┘     → _determine_prune_layers() 读 prune_layers
                                                → _build_layer_ratio_map()  将两者 zip 成
                                                  {2: 0.5, 3: 0.5} 并注入 strategy.config

第 N 层执行剪枝时:
  strategy.compute_importance(attn_weights, ...)  →  [576] 重要性分数
  strategy.get_prune_ratio(layer_idx, scores)     →  从 ratio_map 查出该层比例
  strategy.compute_keep_mask(...)                 →  保留 top-K 个 token
```

**关键点**：`strategy` 只决定 importance 怎么算，以及剪枝动作如何施加；`prune_layers` 和 `prune_ratio` 独立控制在哪层剪、剪多少。同一组 layers/ratio 可以搭配任意 strategy。`pre_attn_score` 会在目标层前物理裁剪 visual token 并同步更新前序 KV cache；`masking_attn_score` 会保留完整 hidden states / position ids / KV cache，只在目标层的 text->vision attention logits 中屏蔽被剪枝视觉 token。`random` 会根据 `inference.seed` 生成可复现的随机重要性分数，`sparsevlm` 会先选 text raters，再用当前层的 text->vision attention 给 visual token 打分。

### 2.3 典型配置示例

#### 示例 1：最简 — 一个策略，一个层，一个比例

```yaml
pruning:
  strategy: "attn_score"
  prune_layers: [2]
  prune_ratio: [0.5]
```

直接 `python entropy_exp/src/prune_inference.py --config ... --dataset mme` 即可运行。

#### 示例 2：两层配对，不同比例

```yaml
pruning:
  strategy: "entropy"
  prune_layers: [2, 3]
  prune_ratio: [0.3, 0.6]    # 第 2 层剪 30%，第 3 层在剩余中再剪 60%
```

> **注意**：多层剪枝是**渐进的**。第 2 层从 576 个 token 中剪 30%（保留 403），第 3 层再从 403 中剪 60%（保留 161）。

#### 示例 3：标量广播

```yaml
pruning:
  prune_layers: [2, 5, 10]
  prune_ratio: 0.5            # 标量，自动广播到每层都用 0.5
```

#### 示例 4：搭配 entropy 动态比例

```yaml
pruning:
  strategy: "entropy"
  prune_layers: [2, 3]
  prune_ratio: [0.5, 0.5]    # 作为 base ratio
  entropy:
    dynamic_ratio: true       # 开启后，实际比例 ≥ base ratio
    dynamic_scale: 0.5
    max_prune_ratio: 0.9
```

此时 `prune_ratio` 中的值作为基准，`EntropyStrategy.get_prune_ratio()` 会根据 importance 分布的集中度向上调整（但不超过 `max_prune_ratio`）。

> **注意**：当 `prune_layers` 和 `prune_ratio` 都是列表时长度必须相等，否则启动时会报错。

#### 示例 5：可复现的 random 剪枝

```yaml
pruning:
  strategy: "random"
  prune_layers: [2, 3]
  prune_ratio: [0.5, 0.5]
```

在相同的 `inference.seed` 下，多次运行会得到相同的随机保留结果。

#### 示例 6：sparsevlm text-rater 剪枝

```yaml
pruning:
  strategy: "sparsevlm"
  prune_layers: [8, 12, 16]
  prune_ratio: [0.25, 0.5, 0.5]
  sparsevlm:
    fallback_topk: 4
    exclude_special_tokens: true
    min_visual_tokens_after_prune: 16
```

该策略会先从文本 token 中选择 image-relevant raters，再基于每个剪枝层的 text->vision attention 计算 visual token 分数。

### 2.4 如何跑实验

**方式 1：`--set` 覆盖（推荐，无需创建文件）**

保留一份 `prune.yaml` 作为默认值，用 `--set` 覆盖要改的参数：

```bash
# 实验 A：attn_score 在第 2 层剪 50%（yaml 默认值，直接跑）
bash entropy_exp/scripts/run_prune.sh attn_score mme

# 实验 B：entropy 在第 2、3 层分别剪 30%、60%
bash entropy_exp/scripts/run_prune.sh entropy mme \
    --set pruning.prune_layers=[2,3] \
    --set pruning.prune_ratio=[0.3,0.6]

# 实验 C：第 5 层 70%
bash entropy_exp/scripts/run_prune.sh attn_score mme \
    --set pruning.prune_layers=[5] \
    --set pruning.prune_ratio=[0.7]

# 实验 D：random，第 2、3 层各剪 50%
bash entropy_exp/scripts/run_prune.sh random mme \
    --set pruning.prune_layers=[2,3] \
    --set pruning.prune_ratio=[0.5,0.5]

# 实验 E：sparsevlm，第 8/12/16 层按给定比例剪枝
bash entropy_exp/scripts/run_prune.sh sparsevlm mme \
    --set pruning.prune_layers=[8,12,16] \
    --set pruning.prune_ratio=[0.25,0.5,0.5]
```

**方式 2：修改 yaml**

直接编辑 `prune.yaml` 中的三个字段后运行：

```bash
python entropy_exp/src/prune_inference.py \
    --config entropy_exp/configs/prune.yaml \
    --dataset mme
```

**方式 3：显式逐策略对比**

当前不再提供 `compare` 一键模式，建议显式逐个策略运行，保证对比矩阵和输出目录更可控：

```bash
bash entropy_exp/scripts/run_prune.sh baseline mme
bash entropy_exp/scripts/run_prune.sh attn_score mme
bash entropy_exp/scripts/run_prune.sh pre_attn_score mme
bash entropy_exp/scripts/run_prune.sh masking_attn_score mme
bash entropy_exp/scripts/run_prune.sh entropy mme
bash entropy_exp/scripts/run_prune.sh random mme
```

### 2.5 其他参数速查

#### `pruning.entropy` — entropy 策略的额外参数

| 参数 | 说明 |
|:--|:--|
| `dynamic_ratio: false` | 设为 `true` 时，剪枝比例根据 importance 分布集中度动态向上调整 |
| `dynamic_scale: 0.5` | 动态调整幅度系数（越大调整越激进） |
| `max_prune_ratio: 0.9` | 动态调整上限，防止过度剪枝 |

#### `pruning.layer_selection` — 层选择方法

| 值 | 说明 |
|:--|:--|
| `fixed` | 使用 `prune_layers` 列表指定的层（默认，推荐） |
| `dynamic` | 预留接口，当前回退读 `prune_layers` |

---

## 三、捕获（Capture）配置

### 3.1 剪枝推理时的捕获

`prune.yaml` 的 `capture` 段控制两类输出：

```yaml
capture:
  # === 注意力捕获 → captures.h5（HDF5，与 Phase 1 格式一致）===
  save_attention: false           # 保存 text→vision 注意力子矩阵到 HDF5
  capture_layers: "all"           # 捕获哪些层："all"（全部 32 层）或层索引列表
  precision: "fp16"               # fp16 | fp32
  compression: "gzip"
  compression_opts: 4

  # === 剪枝结果 → stats.jsonl ===
  save_importance_scores: true    # 保存每个 visual token 的重要性分数
  save_keep_indices: true         # 保存被保留的 token 索引列表
```

**`save_attention: true`**（HDF5 捕获）：
- 保存指定层的 text→vision 注意力子矩阵 `[H, L_t, L_v]`
- 剪枝层同时保存 `prune_scores [L_v]`；非剪枝捕获层仅保存 `tv_attn`
- 输出到 `captures.h5`，格式与 Phase 1 `AttentionCaptureHook` 完全一致
- Phase 1 的分析脚本可直接处理此文件
- 数据量较大，默认关闭

**`capture_layers`**（捕获层控制）：
- `"all"`（默认）：捕获全部 32 层的注意力，适合离线分析
- 层索引列表如 `[0, 1, 2, 3]`：仅捕获指定层，减少磁盘占用
- 与 `prune_layers` 独立：可以剪枝第 2、3 层，但捕获全部 32 层的注意力数据

```bash
# 捕获全部 32 层（默认）
bash entropy_exp/scripts/run_prune.sh attn_score mme 10 \
    --set capture.save_attention=true

# 仅捕获前 4 层
bash entropy_exp/scripts/run_prune.sh attn_score mme 10 \
    --set capture.save_attention=true \
    --set capture.capture_layers=[0,1,2,3]

# 仅捕获剪枝层（与 prune_layers 相同）
bash entropy_exp/scripts/run_prune.sh attn_score mme 10 \
    --set capture.save_attention=true \
    --set capture.capture_layers=[2,3]
```

**`save_importance_scores` / `save_keep_indices`**（JSONL 剪枝结果）：
- 保存到 `stats.jsonl` 的对应字段中（轻量级）
- `importance_scores [576]`：重要性分数向量，适合分析哪些 token 被认为重要
- `keep_indices [K]`：保留 token 索引，方便可视化
- 大规模正式实验只看精度时可关闭以减小体积

### 3.2 纯注意力捕获（Phase 1，entropy-exp 分支功能）

如果只想捕获全层注意力数据（不做剪枝），使用 `configs/default.yaml` 配合 `inference.py`：

```bash
# 捕获 MME 前 10 个样本的全部 32 层 attention
python entropy_exp/src/inference.py \
    --config entropy_exp/configs/default.yaml \
    --dataset mme \
    --max-samples 10

# 或使用脚本
bash entropy_exp/scripts/run_capture.sh mme 10
```

此模式使用 `AttentionCaptureHook`，输出 HDF5 文件到 `outputs/raw/`。

---

## 四、输出文件说明

每次运行会创建一个独立的 **run 目录**，包含该次实验的完整配置、结果和中间变量：

```
outputs/runs/{strategy}/{dataset}/{dataset}_{strategy}_l{layers}_r{ratios}__{YYYYMMDD_HHMMSS_microseconds}/
  ├── config.yaml        # 实验配置快照（含 --set 覆盖后的最终值）
  ├── answers.jsonl      # 模型回答（与评测脚本兼容）
  ├── stats.jsonl        # 每样本剪枝统计（timing, ratios, seq_len）
  ├── eval/summary.json  # 评测结果
  └── captures.h5        # 注意力中间变量捕获（HDF5，与 Phase 1 格式一致）
                         # 仅当 capture.save_attention=true 时生成
```

其中 leaf `run_name` 保持不变，只是父目录层级改成了 `{strategy}/{dataset}`。例如：

```text
outputs/runs/pre_attn_score/pope/pope_pre_attn_score_l2_r0p3__20260320_192913_375869
outputs/runs/pre_attn_score/pope/pope_pre_attn_score_l2-3_r0p5-0p5__20260320_192913_486301
```

`run_eval.sh`、`run_summary.sh`、`run_analysis.sh`、`check_incomplete_runs.sh` 会递归扫描 `outputs/runs/`，所以旧的 leaf `run_name` / prefix 用法仍然可用，例如 `pope_random_`、`mme_masking_attn_score_` 这类筛选方式不需要改。

### 4.1 config.yaml — 配置快照

每次运行自动保存当前生效的完整配置（包括 `--set` 覆盖后的值）。可直接用于复现：

```bash
python entropy_exp/src/prune_inference.py \
    --config entropy_exp/outputs/runs/attn_score/mme/mme_attn_score_l2-3_r0p5-0p5__20260309_143000_123456/config.yaml \
    --dataset mme
```

### 4.2 answers.jsonl — 模型回答

```json
{"question_id": "...", "prompt": "...", "text": "模型回答", "model_id": "llava-v1.5-7b", "metadata": {}}
```

与 GQA/MME/POPE 评测脚本兼容，直接传路径即可：

```bash
bash entropy_exp/scripts/run_eval.sh gqa entropy_exp/outputs/runs/gqa_attn_score_*/answers.jsonl
```

### 4.3 stats.jsonl — 剪枝统计

每行一个样本，记录剪枝过程的聚合统计：

| 字段 | 类型 | 说明 |
|:--|:--|:--|
| `sample_idx` | int | 样本序号 |
| `run_mode` | str | `"prune"` 或 `"baseline"` |
| `strategy_requested` | str | 使用的策略名 |
| `effective_prune_ratio_map` | dict | 实际的 `{层号: 比例}` 映射 |
| `prune_layers` | list | 剪枝层列表 |
| `question_id` | str | 问题 ID |
| `image_file` | str | 图片文件路径 |
| `answer` | str | 模型回答 |
| `original_seq_len` | int | 剪枝前序列总长度 |
| `final_seq_len` | int | 剪枝后序列总长度 |
| `num_generated_tokens` | int | 生成的新 token 数 |
| `prefill_time` | float | prefill 阶段耗时（秒） |
| `decode_time` | float | decode 阶段耗时（秒） |
| `total_time` | float | 样本总耗时（秒） |
| `layer_{N}_ratio` | float | 第 N 层实际剪枝比例 |
| `layer_{N}_before` | int | 第 N 层剪枝前 visual token 数 |
| `layer_{N}_after` | int | 第 N 层剪枝后 visual token 数 |
| `layer_{N}_pruned` | int | 第 N 层被剪掉的 token 数 |
| `layer_{N}_importance` | list[float] | （`save_importance_scores` 开启时）每个 visual token 的重要性分数 |
| `layer_{N}_keep_indices` | list[int] | （`save_keep_indices` 开启时）被保留 token 的索引 |

### 4.4 captures.h5 — 注意力中间变量捕获（HDF5）

仅当 `capture.save_attention: true` 时生成。保存指定层的 **text→vision 注意力子矩阵**，格式与 Phase 1 的 `AttentionCaptureHook` 输出完全一致：

```
captures.h5
  /{sample_id}/
    .attrs: question_id, image_file, v_token_start, v_token_num, text_token_start
    layer_{N}/
      tv_attn          # [H, L_t, L_v]  text→vision 注意力子矩阵 (gzip 压缩)
      prune_scores     # [L_v]          每个 visual token 的重要性分数（仅剪枝层）
```

- **捕获范围**由 `capture.capture_layers` 控制：`"all"` 捕获全部 32 层，指定列表如 `[0, 1, 2, 3]` 则只捕获这些层
- **剪枝层**：同时写入 `tv_attn` 和 `prune_scores`
- **非剪枝捕获层**：仅写入 `tv_attn`（无 `prune_scores`）
- **`tv_attn`**：原始 text→vision 注意力矩阵，`H`=注意力头数，`L_t`=文本 token 数，`L_v`=视觉 token 数（剪枝前）
- **`prune_scores`**：从 `tv_attn` 计算得出的重要性分数，与 Phase 1 格式一致
- Phase 1 的分析脚本可直接处理此文件

> **注意**：注意力捕获数据量较大（每层 [32, ~60, 576] float16 ≈ 2.2MB），正式大规模实验时建议关闭或仅捕获关键层以节省磁盘空间。

---

## 五、常用实验流程

### 5.1 冒烟测试

```bash
# 快速验证代码能跑通
bash entropy_exp/scripts/run_prune.sh attn_score mme 2
# 结果在 outputs/runs/mme_attn_score_l2-3_r0p5-0p5__YYYYMMDD_HHMMSS_microseconds/ 下
```

### 5.2 单策略实验

```bash
bash entropy_exp/scripts/run_prune.sh attn_score gqa
# 评测时指定 run 目录下的 answers.jsonl
bash entropy_exp/scripts/run_eval.sh gqa entropy_exp/outputs/runs/gqa_attn_score_*/answers.jsonl

# random 策略同理
bash entropy_exp/scripts/run_prune.sh random gqa
bash entropy_exp/scripts/run_eval.sh gqa entropy_exp/outputs/runs/gqa_random_*/answers.jsonl
```

### 5.3 策略对比实验

```bash
# 显式跑多种策略，保证每个 run 目录和 override 可追踪
bash entropy_exp/scripts/run_prune.sh baseline mme
bash entropy_exp/scripts/run_prune.sh attn_score mme
bash entropy_exp/scripts/run_prune.sh pre_attn_score mme
bash entropy_exp/scripts/run_prune.sh masking_attn_score mme
bash entropy_exp/scripts/run_prune.sh entropy mme

# 分别评测（每种策略生成独立的 run 目录）
bash entropy_exp/scripts/run_eval.sh mme entropy_exp/outputs/runs/mme_baseline_*/answers.jsonl
bash entropy_exp/scripts/run_eval.sh mme entropy_exp/outputs/runs/mme_attn_score_*/answers.jsonl
bash entropy_exp/scripts/run_eval.sh mme entropy_exp/outputs/runs/mme_pre_attn_score_*/answers.jsonl
bash entropy_exp/scripts/run_eval.sh mme entropy_exp/outputs/runs/mme_masking_attn_score_*/answers.jsonl
bash entropy_exp/scripts/run_eval.sh mme entropy_exp/outputs/runs/mme_entropy_*/answers.jsonl
```

### 5.4 调参实验（`--set` 快速切换）

不需要复制 yaml，用 `--set` 直接在命令行覆盖参数，跑多组实验：

```bash
# 实验 A：attn_score，第 2 层 50%
bash entropy_exp/scripts/run_prune.sh attn_score mme \
    --set pruning.prune_layers=[2] \
    --set pruning.prune_ratio=[0.5]

# 实验 B：entropy，第 2、3 层分别 30%、60%
bash entropy_exp/scripts/run_prune.sh entropy mme \
    --set pruning.prune_layers=[2,3] \
    --set pruning.prune_ratio=[0.3,0.6]

# 实验 C：attn_score，第 5 层 70%
bash entropy_exp/scripts/run_prune.sh attn_score mme \
    --set pruning.prune_layers=[5] \
    --set pruning.prune_ratio=[0.7]

# 实验 D：entropy 动态比例
bash entropy_exp/scripts/run_prune.sh entropy mme \
    --set pruning.entropy.dynamic_ratio=true \
    --set pruning.entropy.dynamic_scale=0.8 \
    --set pruning.entropy.max_prune_ratio=0.9

# 实验 E：random（固定 seed，可复现实验）
bash entropy_exp/scripts/run_prune.sh random mme \
    --set pruning.prune_layers=[2,3] \
    --set pruning.prune_ratio=[0.5,0.5]
```

> **Tip**：把上述命令写成一个批跑脚本，就可以一次性提交多组实验。`--set` 优先级高于 yaml 文件中的值，不会修改原始 yaml。

### 5.5 关闭捕获（减少磁盘占用）

```yaml
capture:
  save_attention: false           # 关闭 HDF5 注意力捕获
  save_importance_scores: false   # 关闭 stats 中的 importance scores
  save_keep_indices: false        # 关闭 stats 中的 keep indices
```

此时 run 目录下不会生成 `captures.h5`，`stats.jsonl` 只包含聚合统计。

或通过命令行关闭：

```bash
bash entropy_exp/scripts/run_prune.sh attn_score mme \
    --set capture.save_attention=false \
    --set capture.save_importance_scores=false \
    --set capture.save_keep_indices=false
```

如果只需要注意力捕获（用于分析），不需要 stats 中的 scores/indices，可以只开 `save_attention`：

```bash
bash entropy_exp/scripts/run_prune.sh attn_score mme \
    --set capture.save_attention=true \
    --set capture.save_importance_scores=false \
    --set capture.save_keep_indices=false
```

---

## 六、效率测量注意事项

进行正式效率对比实验时，需严格控制环境：

| 条件 | 说明 |
|:--|:--|
| **GPU** | 独占显卡，`nvidia-smi` 确认无其他进程 |
| **CPU** | `top` / `htop` 确认总体占用 < 80% |
| **磁盘 I/O** | 数据放 SSD，或提前预加载到内存 |
| **Baseline** | 必须使用 `--baseline` 而非原始 `inference.py`，因为它走的是同一个自定义 decode 路径，计时才公平 |
| **CUDA warmup** | 首个样本的耗时通常偏高，可考虑跑前几个样本作 warmup 后再统计 |
