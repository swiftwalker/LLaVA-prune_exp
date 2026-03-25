# 实验设计与实现文档

本文档覆盖两个阶段的实验：

- **阶段 1（entropy-exp 分支）**：熵变趋势分析 — 捕获注意力数据、离线计算指标
- **阶段 2（prune-exp 分支）**：视觉 token 剪枝实验 — 对比不同剪枝策略的性能与效率

---

# 阶段 1：熵变趋势分析

## 一、研究问题

1. **熵变最大点 vs 熵最小点**：哪个更适合作为最佳剪枝判断依据？
2. **熵变的图片依赖性**：不同图片的熵变趋势是否一致？熵变由图片+模型共同决定还是模型单独决定？

---

## 二、实验总体设计

### 2.1 数据集选取

| 数据集 | 题型 | 评测重点 | 选取理由 |
|:---|:---|:---|:---|
| **GQA** | 开放式问答 | 视觉推理与空间关系 | 对 token 剪枝敏感，区分度好 |
| **MME** | 感知 + 认知 | 多维度综合能力 | 覆盖 Perception/Cognition 双维度 |
| **POPE** | 二分类（Yes/No） | 物体幻觉检测 | 快速验证，对幻觉敏感 |

### 2.2 实验配置

| 变量 | 设置 |
|:---|:---|
| 模型 | llava-v1.5-7b |
| 推理后端 | transformers（eager attention, batch=1） |
| 捕获内容 | 全部 32 层的 text→vision 注意力子矩阵 |
| Hook 模式 | `mode="capture"`（纯捕获，不剪枝） |
| 温度 | 0（确定性推理） |
| 随机种子 | 42 |
| GPU | cuda:0 |

### 2.3 分析指标

| 指标 | 计算方式 | 分析目的 |
|:---|:---|:---|
| Shannon 熵 $H(q)$ | Prune Scores → 归一化 → $-\sum p \log_2 p$ | 跨层熵变曲线，定位熵变最大点 / 熵最小点 |
| Rényi 熵 $H_\alpha(q)$ | $\alpha=2$ | 对集中分布更敏感，与 Shannon 熵对比 |
| Gini 系数 $G$ | 基于 Prune Scores 的不均匀度 | 注意力不均匀程度的另一种度量 |
| Top-K 集中度 $C_k$ | K=32,64,128 | 直观衡量注意力集中程度 |

### 2.4 关键分析

- **分析 1**：绘制每个样本的独立熵曲线（叠加在同一张图上），观察**样本间一致性** → 回答问题 2
- **分析 2**：绘制熵变（层间差分）曲线，统计最大熵变点 / 最小熵点的分布 → 回答问题 1
- **分析 3**：跨数据集对比均值±σ曲线，观察数据集间差异
- **分析 4**：变异系数（CV = σ/μ）分析，判断各层注意力模式是模型主导还是图片主导

---

## 三、工程实现

### 3.1 项目位置与隔离策略

- **项目路径**：`~/data/LLaVA/entropy_exp/`
- **代码基础**：基于 `~/data/LLaVA` 仓库的 `master` 分支，创建 `entropy-exp` 分支
- **与 SparseVLMs 的关系**：完全解耦。通过软连接复用数据集和模型权重，不引入 SparseVLMs 代码
- **模型加载**：使用原始 `LlavaLlamaForCausalLM`（非 Sparse 版本），通过 `dynamic_sparse=False` 等效实现

### 3.2 目录结构

```
~/data/LLaVA/entropy_exp/
├── models/              → ln -s /home/liuyu/data/LM_models
├── datasets/            → ln -s /home/liuyu/data/SparseVLMs/data
├── eval_questions/      → ln -s /home/liuyu/data/SparseVLMs/eval
├── src/
│   ├── inference.py     # 推理主入口（改造自 model_vqa_loader.py）
│   ├── hooks.py         # AttentionCaptureHook + HDF5 写入
│   ├── metrics.py       # Shannon/Rényi 熵、Gini、Top-K 计算
│   └── eval_datasets.py # GQA/MME/POPE 评测脚本整合
├── analysis/
│   └── entropy_analysis.py   # 离线指标提取（HDF5 → CSV）
├── outputs/
│   ├── raw/             # HDF5 原始注意力数据
│   ├── processed/       # 指标 CSV 文件
│   └── answers/         # 推理答案 JSONL（兼容现有评测）
├── configs/
│   └── default.yaml     # 实验配置文件
└── scripts/
    ├── run_capture.sh   # 一键推理+捕获
    ├── run_analysis.sh  # 一键离线分析（不含可视化）
    └── run_eval.sh      # 一键评测
```

### 3.3 核心模块说明

#### `src/hooks.py` — 注意力捕获

- **`AttentionCaptureHook`** 类：管理 HDF5 文件的打开/关闭，提取 text→vision 子矩阵并写盘
  - `extract_tv_submatrix()`: 从完整 attention `[B,H,L,L]` 中切出 `[H, L_t, L_v]` 或 `[L_t, L_v]`
  - `compute_prune_scores()`: head 均值 → text 均值 → 每个 visual token 的重要性分数 `[L_v]`
  - `save_sample()`: 处理 `model.generate()` 的 `output_attentions` 输出，仅取 prefill 阶段
- **`locate_image_tokens()`**: 从 input_ids 中定位 IMAGE_TOKEN 位置，结合配置 `v_token_num` 推算 `v_token_start` 和 `text_token_start`

#### `src/inference.py` — 推理主流程

- 基于 LLaVA 原始 `model_vqa_loader.py` 改造
- 使用 `attn_implementation="eager"` 加载模型，确保 `output_attentions=True` 可用
- 每个样本：推理 → 提取 32 层注意力 → 写入 HDF5 + 写入答案 JSONL
- 答案 JSONL 格式与原始评测脚本兼容

#### `src/metrics.py` — 指标计算

- `shannon_entropy()`: $H(p) = -\sum p \log_2 p$
- `renyi_entropy()`: $H_\alpha(p) = \frac{1}{1-\alpha} \log_2(\sum p^\alpha)$，默认 $\alpha=2$
- `gini_coefficient()`: 基于排序的 Gini 系数
- `topk_concentration()`: Top-K 注意力质量占比
- `compute_all_metrics()`: 一次性计算所有指标

#### `analysis/entropy_analysis.py` — 离线分析

- 读取 HDF5 → 对每个 (sample, layer) 计算全部指标 → 输出 `per_sample_layer.csv`
- 计算层间熵变（差分）；多文件场景按 `sample_uid=source_file::sample_id` 分组，避免跨运行样本冲突
- 输出每个 HDF5 文件、每数据集、全数据集的 summary 统计
- 服务器端可视化已解耦到独立流程，本目录不再维护绘图脚本

### 3.4 数据流

```
[推理]  inference.py
   ├──→ outputs/raw/*.h5          (32层 text→vision 注意力子矩阵)
   └──→ outputs/answers/*.jsonl   (模型回答，兼容现有评测)

[评测]  eval_datasets.py
   └──→ 调用 GQA/MME/POPE 原始评测脚本

[分析]  entropy_analysis.py
   └──→ outputs/processed/*.csv   (每样本×每层的熵/Gini/TopK指标)
```

### 3.5 HDF5 存储结构

```
output_<dataset>_<timestamp>.h5
├── sample_<question_id>/
│   ├── attrs: {question_id, image_file, answer, v_token_start, v_token_num, text_token_start}
│   ├── layer_0/
│   │   ├── tv_attn          # [H, L_t, L_v] fp16, gzip 压缩
│   │   └── prune_scores     # [L_v] fp32
│   ├── layer_1/
│   │   └── ...
│   └── layer_31/
│       └── ...
└── attrs: {实验配置快照}
```

### 3.6 存储开销估算

| 模式 | 单样本 | 100 样本 | 1k 样本 |
|:---|:---|:---|:---|
| 全 32 层 + 全 head（`head_reduction=none`） | ~48 MB | ~4.8 GB | ~48 GB |
| 全 32 层 + head 均值（`head_reduction=mean`） | ~1.5 MB | ~150 MB | ~1.5 GB |

默认使用 `head_reduction=none`（保留全部 head 维度），初期 100 样本的 4.8 GB 可接受。

---

## 四、运行方法

### 环境准备

```bash
conda activate llava
cd ~/data/LLaVA
```

### 4.1 推理 + 捕获

```bash
# 快速验证（2 样本）
bash entropy_exp/scripts/run_capture.sh mme 2

# 初期分析（100 样本）
bash entropy_exp/scripts/run_capture.sh mme 100

# 全量运行某个数据集
bash entropy_exp/scripts/run_capture.sh gqa

# 运行全部三个数据集
bash entropy_exp/scripts/run_capture.sh all
```

### 4.2 离线分析

```bash
# 分析所有已捕获的 HDF5 文件
bash entropy_exp/scripts/run_analysis.sh

# 仅分析某个数据集
bash entropy_exp/scripts/run_analysis.sh gqa
```

### 4.3 评测

```bash
# 运行评测（需要先完成推理）
bash entropy_exp/scripts/run_eval.sh gqa entropy_exp/outputs/answers/gqa_*.jsonl
bash entropy_exp/scripts/run_eval.sh pope entropy_exp/outputs/answers/pope_*.jsonl
```

### 4.4 直接运行 Python

```bash
# 推理
python entropy_exp/src/inference.py --config entropy_exp/configs/default.yaml --dataset mme --max-samples 10

# 分析
python entropy_exp/analysis/entropy_analysis.py \
    --h5 entropy_exp/outputs/raw/*.h5 \
    --output entropy_exp/outputs/processed
```

---

## 五、配置说明

配置文件：`entropy_exp/configs/default.yaml`

| 配置项 | 默认值 | 说明 |
|:---|:---|:---|
| `model.attn_implementation` | `eager` | 必须为 eager 才能获取 attention weights |
| `capture.layers` | `all` | 捕获哪些层，可设为 `[2, 6, 15]` 减少存储 |
| `capture.head_reduction` | `none` | `none` 保留全部 head，`mean` 做 head 均值 |
| `capture.max_samples` | `null` | 限制样本数，`null` 表示全量 |
| `capture.v_token_num` | `576` | LLaVA-1.5 视觉 token 数（24×24），用于 image token 定位与序列长度一致性校验 |
| `inference.temperature` | `0` | 确定性推理 |
| `inference.seed` | `42` | 随机种子 |

---

## 六、后续扩展接口

### 剪枝模块预留

`hooks.py` 中的 `AttentionCaptureHook` 设计了 `mode` 参数：
- `mode="capture"`: 当前阶段，纯观测不干预
- `mode="prune"`: 后续阶段，基于捕获的 prune scores 计算剪枝 mask

`compute_prune_scores()` 的计算路径与 SparseVLM 一致（head 均值 → text 均值 → per visual token score），确保从观测到剪枝的逻辑一致性。

### 配置化层级控制

通过 `capture.layers` 配置项控制捕获/剪枝的目标层，不硬编码。后续可以根据分析结果动态选择关键层。

---
---

# 阶段 2：视觉 token 剪枝实验（prune-exp 分支）

## 一、实验目标

在固定剪枝层条件下，对比多种 visual token 剪枝策略，包括显式裁剪与 masking-based 隐式剪枝：

| 方案 | 策略名 | 核心思路 |
|:--|:--|:--|
| **A** | `attn_score` | SparseVLM 风格：head 均值 → text 均值 → 每 visual token 重要性 |
| **B** | `pre_attn_score` | 与 `attn_score` 相同打分，但在目标层前物理裁剪 visual token，并同步更新前序 KV cache |
| **C** | `masking_attn_score` | 与 `pre_attn_score` 相同打分，但仅在目标层的 text->vision attention logits 中屏蔽被剪枝 token，不改变 hidden states / position ids / KV cache 长度 |
| **D** | `entropy` | 熵加权聚合：低熵 head 权重更高，可选动态调整剪枝比例 |

### 评估维度

| 维度 | 指标 | 说明 |
|:--|:--|:--|
| **性能** | GQA Accuracy / MME Score / POPE macro-F1 | 剪枝后 answer 质量 vs baseline；POPE 使用各 category F1 的简单平均 |
| **效率** | prefill time / decode time / total time per sample | 同一 decode 路径，公平对比 |

### 效率测量注意事项

- **GPU**：独占显卡，避免资源抢占
- **CPU**：整体占用控制在 80% 以下
- **磁盘 I/O**：数据放 SSD 或预加载至内存

---

## 二、代码框架

### 2.1 新增目录结构

```
entropy_exp/
├── src/
│   ├── strategies/              # NEW — 剪枝策略模块
│   │   ├── __init__.py          # 策略注册表 + get_strategy()
│   │   ├── base.py              # PruneStrategy 抽象基类
│   │   ├── attn_score.py        # 方案 A：Attention Score
│   │   ├── pre_attn_score.py    # 方案 B：Pre-layer physical pruning
│   │   ├── masking_attn_score.py # 方案 C：Layer-local logits masking
│   │   └── entropy.py           # 方案 D：Entropy-weighted
│   ├── pruner.py                # NEW — 核心剪枝引擎 VisualTokenPruner
│   ├── prune_inference.py       # NEW — 剪枝推理主入口
│   ├── hooks.py                 # 原有（兼容）
│   ├── inference.py             # 原有（兼容）
│   ├── metrics.py               # 原有（兼容）
│   └── eval_datasets.py         # 原有（兼容）
├── configs/
│   ├── default.yaml             # 原有（capture 用）
│   └── prune.yaml               # NEW — 剪枝实验配置
├── scripts/
│   ├── run_prune.sh             # NEW — 剪枝实验启动脚本
│   ├── run_capture.sh           # 原有
│   ├── run_analysis.sh          # 原有
│   └── run_eval.sh              # 原有
└── outputs/
    ├── runs/                    # NEW — 每次运行一个独立子目录
    │   └── {dataset}_{strategy}_{YYYYMMDD_HHMMSS}/
    │       ├── config.yaml      #   实验配置快照（含 --set 覆盖后的最终值）
    │       ├── answers.jsonl    #   模型回答（兼容现有评测脚本）
    │       ├── stats.jsonl      #   逐样本剪枝统计
    │       └── captures.h5      #   注意力中间变量（HDF5，仅 save_attention=true）
    ├── raw/                     # 原有（Phase 1 capture）
    └── processed/               # 原有（Phase 1 分析结果）
```

### 2.2 核心模块说明

#### `strategies/base.py` — PruneStrategy 抽象基类

```python
class PruneStrategy(ABC):
    def compute_importance(attn_weights, v_start, v_num, text_start, layer_idx) -> Tensor[L_v]
    def get_prune_ratio(layer_idx, importance_scores) -> float     # 可覆写为动态比例
    def compute_keep_mask(...) -> (keep_indices, info_dict)         # 模板方法
```

| 兼容性维度 | 实现方式 |
|:--|:--|
| 固定 / 动态剪枝层 | `VisualTokenPruner._determine_prune_layers()` 支持 `fixed`（配置列表） / `dynamic`（预留） |
| 固定 / 动态剪枝比例 | `prune_layers` 与 `prune_ratio` 为等长列表，逐层配对；`get_prune_ratio()` 从 `prune_ratio_map` 查表，`EntropyStrategy` 可覆写为动态 |
| 不同计算策略 | 通过策略注册表 `STRATEGY_REGISTRY` 按名称实例化 |
| 与 capture 代码兼容 | 原有 `hooks.py`, `inference.py` 完全不修改 |

#### `pruner.py` — VisualTokenPruner

核心引擎，实现：

1. **自定义 layer-by-layer prefill**：逐层运行 decoder layers，在指定层执行剪枝
2. **KV cache 一致性**：显式剪枝策略会同步更新所有层（0 … prune_layer）的 KV cache；`masking_attn_score` 保留完整 KV cache
3. **Greedy decode**：使用剪枝后的 KV cache 进行自回归生成

关键方法：
- `pruned_generate()`: 公开入口，返回 `(generated_ids, prune_info)`
- `_pruned_prefill()`: 逐层 prefill，在 prune layer 调用策略计算 keep mask，并根据 `prune_stage` 选择显式裁剪或 masking
- `_prune_kv_cache()`: 从 DynamicCache 中移除被剪枝位置的 K/V（仅显式裁剪策略）

#### `prune_inference.py` — 推理主入口

- 复用 `VQADataset` + `locate_image_tokens()` 等已有逻辑
- 调用 `model.prepare_inputs_labels_for_multimodal()` 获取合并 embeddings
- 使用 `VisualTokenPruner.pruned_generate()` 进行剪枝推理
- 每次运行创建独立目录 `outputs/runs/{strategy}/{dataset}/{dataset}_{strategy}_{timestamp}/`
- 输出 `config.yaml`（实验配置快照）、`answers.jsonl`（兼容评测）、`stats.jsonl`（剪枝统计）
- 当 `capture.save_attention=true` 时，额外输出 `captures.h5`（HDF5 注意力数据）
- 支持 `--set KEY=VALUE` 命令行覆盖任意配置项（点分隔嵌套 key，自动类型推导）

### 2.3 监控与记录

#### stats.jsonl — 剪枝统计

每行一个样本，包含：

| 字段 | 说明 |
|:--|:--|
| `sample_idx` | 样本序号 |
| `question_id` / `image_file` | 样本标识 |
| `run_mode` | `prune` / `baseline` |
| `strategy_requested` | 配置请求的策略名（如 `attn_score` / `pre_attn_score` / `masking_attn_score` / `entropy`） |
| `prune_stage` | `post` / `pre` / `masking`，表示剪枝动作在层内的施加方式 |
| `effective_prune_ratio_map` | 实际生效的层→比例映射 |
| `prune_layers` | 剪枝层列表 |
| `original_seq_len` / `final_seq_len` | 剪枝前/后序列长度；`masking_attn_score` 下二者相等 |
| `num_generated_tokens` | 生成的新 token 数 |
| `prefill_time` / `decode_time` / `total_time` | 计时信息 |
| `layer_N_ratio` | 第 N 层的实际剪枝比例 |
| `layer_N_before` / `after` / `pruned` | 剪枝前/后 visual token 数量 |
| `layer_N_importance` | (`save_importance_scores` 开启时) 重要性分数 |
| `layer_N_keep_indices` | (`save_keep_indices` 开启时) 保留的 token 索引 |

#### captures.h5 — 注意力中间变量（HDF5）

仅当 `capture.save_attention: true` 时生成，格式与 Phase 1 `AttentionCaptureHook` 一致：

```
captures.h5
  /{sample_idx}_{question_id}/
    .attrs: question_id, image_file, v_token_start, v_token_num, text_token_start
    layer_{N}/
      tv_attn          # [H, L_t, L_v]  text→vision 注意力子矩阵 (gzip)
      prune_scores     # [L_v]          重要性分数（仅剪枝层）
```

- `capture_layers` 控制捕获哪些层的注意力：`"all"` 或指定层列表如 `[0, 1, 2, 3]`
- 剪枝层同时写入 `tv_attn` 和 `prune_scores`；非剪枝捕获层仅写入 `tv_attn`

---

## 三、运行方法

### 3.1 剪枝推理

```bash
# 快速验证（2 样本, attn_score 策略）
bash entropy_exp/scripts/run_prune.sh attn_score mme 2

# 快速验证（2 样本, masking_attn_score 策略）
bash entropy_exp/scripts/run_prune.sh masking_attn_score gqa 2

# Entropy 策略，全量 POPE
bash entropy_exp/scripts/run_prune.sh entropy pope

# Baseline（无剪枝，同一 decode 路径）
bash entropy_exp/scripts/run_prune.sh baseline mme 10

# 所有数据集
bash entropy_exp/scripts/run_prune.sh attn_score all

# 使用 --set 覆盖配置项
bash entropy_exp/scripts/run_prune.sh entropy mme 10 \
    --set pruning.prune_layers=[2,5,10] \
    --set pruning.prune_ratio=[0.3,0.4,0.5]
```

结果输出到 `outputs/runs/{strategy}/{dataset}/{dataset}_{strategy}_{YYYYMMDD_HHMMSS}/` 下。

### 3.2 评测

```bash
# 使用 run 目录下的 answers.jsonl
bash entropy_exp/scripts/run_eval.sh gqa entropy_exp/outputs/runs/gqa_attn_score_*/answers.jsonl
bash entropy_exp/scripts/run_eval.sh pope entropy_exp/outputs/runs/pope_entropy_*/answers.jsonl
```

### 3.3 直接运行 Python

```bash
# 剪枝推理
python entropy_exp/src/prune_inference.py \
    --config entropy_exp/configs/prune.yaml \
    --dataset mme --max-samples 10

# baseline
python entropy_exp/src/prune_inference.py \
    --config entropy_exp/configs/prune.yaml \
    --dataset mme --max-samples 10 --baseline

# 使用 --set 覆盖
python entropy_exp/src/prune_inference.py \
    --config entropy_exp/configs/prune.yaml \
    --dataset mme \
    --set pruning.strategy=entropy \
    --set capture.capture_layers=[0,1,2,3]
```

---

## 四、配置说明

配置文件：`entropy_exp/configs/prune.yaml`

| 配置项 | 默认值 | 说明 |
|:--|:--|:--|
| `pruning.strategy` | `attn_score` | 策略选择：`attn_score` \| `pre_attn_score` \| `masking_attn_score` \| `entropy` \| `random` \| `sparsevlm` |
| `pruning.layer_selection` | `fixed` | 层选择方法：`fixed`（配置列表）\| `dynamic`（预留） |
| `pruning.prune_layers` | `[2, 3]` | 剪枝层列表（0-indexed），与 `prune_ratio` 等长 |
| `pruning.prune_ratio` | `[0.5, 0.5]` | 与 `prune_layers` 等长的剪枝比例列表（标量则广播到所有层） |
| `pruning.pre_attn_score` | `{}` | 目标层前物理裁剪策略的额外参数（当前为空） |
| `pruning.masking_attn_score` | `{}` | 目标层内 logits masking 策略的额外参数（当前为空） |
| `pruning.entropy.dynamic_ratio` | `false` | 是否用熵动态调整比例 |
| `pruning.entropy.dynamic_scale` | `0.5` | 动态调整的缩放系数 |
| `capture.save_attention` | `true` | 保存注意力矩阵到 captures.h5（HDF5） |
| `capture.capture_layers` | `"all"` | 捕获哪些层：`"all"` 或层索引列表如 `[0, 1, 2, 3]` |
| `capture.save_importance_scores` | `true` | 是否在 stats 中保存重要性分数 |
| `capture.save_keep_indices` | `true` | 是否在 stats 中保存保留索引 |
| `capture.precision` | `"fp16"` | HDF5 存储精度：`fp16` \| `fp32` |
| `capture.compression` | `"gzip"` | HDF5 压缩方式 |
| `capture.compression_opts` | `4` | 压缩等级 |

---

## 五、后续扩展

1. **动态层选择**：`layer_selection: "dynamic"` — 根据运行时熵变选择剪枝层
2. **多层剪枝**：在多个层级上渐进剪枝
3. **其他策略**：新增策略只需继承 `PruneStrategy` 并注册到 `STRATEGY_REGISTRY`
4. **效率优化**：SDPA / FlashAttention 兼容（需实现无 `output_attentions` 的 score 计算路径）
