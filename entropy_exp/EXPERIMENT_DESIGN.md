# 熵变趋势分析实验 — 设计与实现文档

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
│   ├── entropy_analysis.py   # 离线指标提取（HDF5 → CSV）
│   └── visualize.py          # 可视化绘图（6 种图表）
├── outputs/
│   ├── raw/             # HDF5 原始注意力数据
│   ├── processed/       # 指标 CSV 文件
│   ├── answers/         # 推理答案 JSONL（兼容现有评测）
│   └── figures/         # 可视化图表
├── configs/
│   └── default.yaml     # 实验配置文件
└── scripts/
    ├── run_capture.sh   # 一键推理+捕获
    ├── run_analysis.sh  # 一键离线分析+可视化
    └── run_eval.sh      # 一键评测
```

### 3.3 核心模块说明

#### `src/hooks.py` — 注意力捕获

- **`AttentionCaptureHook`** 类：管理 HDF5 文件的打开/关闭，提取 text→vision 子矩阵并写盘
  - `extract_tv_submatrix()`: 从完整 attention `[B,H,L,L]` 中切出 `[H, L_t, L_v]` 或 `[L_t, L_v]`
  - `compute_prune_scores()`: head 均值 → text 均值 → 每个 visual token 的重要性分数 `[L_v]`
  - `save_sample()`: 处理 `model.generate()` 的 `output_attentions` 输出，仅取 prefill 阶段
- **`locate_image_tokens()`**: 从 input_ids 中定位 IMAGE_TOKEN 位置，推算 `v_token_start` 和 `text_token_start`

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

- 读取 HDF5 → 对每个 (sample, layer) 计算全部指标 → 输出 `entropy_per_sample_layer.csv`
- 计算层间熵变（差分）
- 输出每数据集、全数据集的 summary 统计

#### `analysis/visualize.py` — 可视化

生成 6 类图表：

| 图表 | 描述 | 对应问题 |
|:---|:---|:---|
| `*_overlay.png` | 样本级熵曲线叠加图 | Q2: 样本间一致性 |
| `*_mean_band.png` | 均值±σ 带状图 | 整体趋势 |
| `*_entropy_delta.png` | 熵变曲线 + 关键层分布直方图 | Q1: 最佳剪枝点 |
| `cross_dataset_comparison.png` | 跨数据集对比 | Q2: 数据集间差异 |
| `*_variance_analysis.png` | CV + 箱线图 | Q2: 图片vs模型主导 |

### 3.4 数据流

```
[推理]  inference.py
   ├──→ outputs/raw/*.h5          (32层 text→vision 注意力子矩阵)
   └──→ outputs/answers/*.jsonl   (模型回答，兼容现有评测)

[评测]  eval_datasets.py
   └──→ 调用 GQA/MME/POPE 原始评测脚本

[分析]  entropy_analysis.py
   └──→ outputs/processed/*.csv   (每样本×每层的熵/Gini/TopK指标)

[可视化]  visualize.py
   └──→ outputs/figures/           (PNG 图表)
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

### 4.2 离线分析 + 可视化

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
python entropy_exp/analysis/entropy_analysis.py --h5 entropy_exp/outputs/raw/*.h5

# 可视化
python entropy_exp/analysis/visualize.py --csv entropy_exp/outputs/processed/entropy_per_sample_layer.csv
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
| `capture.v_token_num` | `576` | LLaVA-1.5 视觉 token 数（24×24） |
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
