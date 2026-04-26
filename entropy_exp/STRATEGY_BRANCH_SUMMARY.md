# Strategy Branch Summary

本文档总结 `entropy_exp` 当前可运行的策略分支及其大致方法。这里的“策略分支”指运行输出目录中的
`outputs/runs/{strategy}/{dataset}/...` 分支，以及 `run_prune.sh` / scheduler 可直接调度的策略名。

当前共有 **10 个运行分支**：

- **1 个无剪枝对照分支**：`baseline`
- **9 个剪枝策略分支**：`attn_score`、`pre_attn_score`、`masking_attn_score`、`tail_masking_attn_score`、`entropy`、`random`、`sparsevlm`、`sparsevlm_adaptive_stratified`、`sparsevlm_entropy_alpha`

其中 9 个剪枝策略注册在 `entropy_exp/src/strategies/__init__.py` 的 `STRATEGY_REGISTRY` 中；`baseline`
通过 `prune_inference.py --baseline` 进入同一套自定义 decode 路径，但不执行剪枝。

## 1. 总览表

| 分支 | 类型 | 大致方法 | 是否物理删 token | 主要用途 |
| --- | --- | --- | --- | --- |
| `baseline` | 无剪枝对照 | 走同一套 inference/decode 入口，但不做视觉 token 剪枝 | 否 | 质量和速度对照 |
| `attn_score` | post-layer 物理剪枝 | 当前层完整 forward 后，取 text->vision attention，对 head 和 text query 求均值作为 visual token 重要性 | 是 | 基础 attention-score 剪枝 |
| `pre_attn_score` | pre-layer 物理剪枝 | 进入目标层前手工计算该层 QK attention，用同 `attn_score` 的聚合规则提前打分，再先剪后 forward | 是 | 验证“剪在目标层之前”的效果 |
| `masking_attn_score` | target-layer logits masking | 进入目标层前计算 attention score，但不删 token，只在目标层内屏蔽 text query 到被剪 visual key 的 attention logits | 否 | 验证非物理删除的 layer-local mask |
| `tail_masking_attn_score` | tail logits masking | 从配置的起始层开始直到最后一层，都使用 `masking_attn_score` 的 layer-local mask 路径 | 否 | 验证后半段连续 masking |
| `entropy` | post-layer 物理剪枝 | 当前层完整 forward 后，按每个 head 在 visual token 维度的熵给 head 加权，低熵 head 权重更高 | 是 | 引入 head confidence 的 attention 剪枝 |
| `random` | post-layer 物理剪枝 | 给 visual token 生成随机重要性分数，再复用 top-k keep 路径 | 是 | 随机对照和 sanity check |
| `sparsevlm` | post-layer 物理剪枝 | 先从文本中选 image-relevant text raters，再用 raters 的 text->vision attention 给 visual token 打分 | 是 | SparseVLM 风格 text-rater 剪枝 |
| `sparsevlm_adaptive_stratified` | post-layer 物理剪枝 + 空间补偿 | 复用 `sparsevlm` 打分，先保留固定比例高分 token，再按空间 stratum 缺口补偿低分 token | 是 | 缓解纯 top-k 的空间覆盖不足 |
| `sparsevlm_entropy_alpha` | post-layer 物理剪枝 + 熵自适应空间补偿 | 复用 adaptive stratified 逻辑，但把固定 `high_ratio` 换成由 saliency 空间熵动态计算的 alpha | 是 | instance-level 自适应平衡高分聚焦和空间补偿 |

## 2. 公共控制参数

大多数剪枝分支共享以下外层控制项：

| 参数 | 作用 |
| --- | --- |
| `pruning.strategy` | 选择使用哪个策略分支 |
| `pruning.prune_layers` | 指定在哪些 transformer layer 执行剪枝或 masking |
| `pruning.prune_ratio` | 指定每个目标层移除 visual token 的比例；标量会广播到所有目标层，列表会和 `prune_layers` 配对 |
| `pruning.layer_selection` | 当前主要使用 `fixed`；`dynamic` 仍是预留路径 |
| `capture.save_importance_scores` | 是否把每层 importance / visual scores 写入 `stats.jsonl` |
| `capture.save_keep_indices` | 是否把每层 keep / pruned indices 写入 `stats.jsonl` |

注意：`strategy` 只决定“怎么打分”和“怎么施加剪枝动作”；`prune_layers` 和 `prune_ratio`
仍然独立决定“在哪里剪”和“剪多少”。

## 3. 各分支方法说明

### 3.1 `baseline`

`baseline` 是无剪枝对照分支。它不在 `STRATEGY_REGISTRY` 中，而是通过 `run_prune.sh baseline ...`
调用 `prune_inference.py --baseline`。它仍使用仓库的自定义推理路径，因此和剪枝分支的计时、数据读取、
输出目录结构保持可比。

输出目录形如：

```text
entropy_exp/outputs/runs/baseline/{dataset}/{run_name}/
```

### 3.2 `attn_score`

`attn_score` 是最基础的 attention-score 剪枝：

- 目标层先正常 forward，并打开 `output_attentions`
- 从当前层 attention 中取 `text query -> visual key` 子矩阵
- 对 head 维和 text query 维求平均，得到每个 visual token 的 importance
- 按 importance 从高到低保留预算内 token
- 当前层输出后物理裁剪 `hidden_states`，并同步裁剪当前层及之前层的 KV cache

它属于 **post-layer physical pruning**：目标层完整处理原始视觉 token，下一层才看到更短序列。

### 3.3 `pre_attn_score`

`pre_attn_score` 使用和 `attn_score` 相同的 attention 聚合规则，但剪枝时机提前到目标层 forward 之前：

- 在目标层前手工执行 attention 打分所需的前半段计算：layer norm、Q/K projection、RoPE、QK softmax
- 得到 text->vision attention score 后先决定 keep/drop
- 先裁剪目标层输入的 `hidden_states` 和目标层之前已有的 KV cache
- 再让目标层在裁剪后的短序列上正常 forward

它属于 **pre-layer physical pruning**：目标层本身从一开始就只处理保留下来的 visual token。

### 3.4 `masking_attn_score`

`masking_attn_score` 的打分阶段与 `pre_attn_score` 相同，但执行方式不同：

- 进入目标层前先算出将被保留和被屏蔽的 visual token
- 不物理删除 token，`hidden_states`、`position_ids`、KV cache 长度都不变
- 目标层 forward 时走 masking 路径，只对 `text query -> dropped visual key` 的 attention logits 加 mask
- target layer 之外的序列形态保持完整

它适合回答一个很干净的问题：如果不改变序列长度，只切断目标层内部的部分 text->vision attention 边，
性能会如何变化。

### 3.5 `tail_masking_attn_score`

`tail_masking_attn_score` 继承 `masking_attn_score` 的打分和 masking 行为，但层选择方式特殊：

- 用户在 `prune_layers` 中配置一个起始层
- `prune_inference.py` 会把它展开成从该起始层到模型最后一层的连续 layer 列表
- 每个 tail layer 都执行同样的 attention logits masking

它不物理删除 token，更像是在模型后半段持续削弱被判定为低重要性的视觉 token 连接。

### 3.6 `entropy`

`entropy` 和 `attn_score` 走同一条 post-layer 物理剪枝路径，区别在 importance 聚合：

- 先取当前层 text->vision attention
- 对每个 head 计算其 visual-token attention 分布的熵
- 熵越低表示该 head 越集中、越“有决断性”
- 用 `softmax(-entropy)` 得到 head 权重
- 加权聚合 head，再对 text query 求平均得到 visual token 分数

额外地，`entropy` 支持 `dynamic_ratio`：当 importance 分布更集中时，可以在配置上限内动态提高剪枝比例。

### 3.7 `random`

`random` 不依赖 attention：

- `requires_attention()` 返回 `False`
- 每个 visual token 的 importance 来自 `torch.rand(v_token_num)`
- 后续仍复用基础 top-k keep 和 post-layer 物理剪枝路径

它主要作为随机对照，用来判断一个策略是否真的优于无信息的 token 选择。

### 3.8 `sparsevlm`

`sparsevlm` 是 text-rater-guided 的 visual token 剪枝：

- 在 sample 准备阶段，从初始 multimodal embedding 中拆出 visual block `H_v` 和 text block `H_q`
- 用 `H_v @ H_q^T` 估计 text token 与图像的相关性
- 排除 special tokens 后，用均值阈值选择 image-relevant text raters；若为空则 fallback 到 top-k
- 到每个 prune layer 时，只使用这些 text raters 的 text->vision attention 来给 visual token 打分
- 按低分优先删除，同时受 `min_visual_tokens_after_prune` 保护

它仍然是 post-layer 物理剪枝，但 importance 的 text query 来源不再是所有文本 token，而是预先筛出的 raters。

### 3.9 `sparsevlm_adaptive_stratified`

`sparsevlm_adaptive_stratified` 继承 `sparsevlm` 的 text-rater 选择和 visual score 计算，改动重点在 keep 策略：

- 先根据 `prune_ratio` 计算目标保留数 `target_keep`
- 用固定 `high_ratio` 计算高分保留数 `n_high = floor(target_keep * high_ratio)`
- 高分部分直接按 saliency top-k 保留
- 剩余 `n_low` 个 token 作为空间补偿预算
- 将原始 patch 网格划成 `grid_size x grid_size` 个 strata
- 根据各 stratum 的 selected deficit 分配补偿 quota
- stratum 内部可用 `random` 或 `farthest` 方式选择补偿 token

它的目标是在保持 saliency top-k 能力的同时，避免视觉 token 过度集中在少数局部区域。

### 3.10 `sparsevlm_entropy_alpha`

`sparsevlm_entropy_alpha` 是 `sparsevlm_adaptive_stratified` 的 instance-level 自适应版本。它复用同一套
text-rater、stratum quota 和补偿选择逻辑，但不再使用固定 `high_ratio`。每个样本、每个剪枝层都会从
当前 `visual_scores` 的空间熵动态计算 alpha：

```text
p = visual_scores / visual_scores.sum()
H = -sum(p * log(p))
H_norm = H / log(N)
alpha = alpha_min + (1 - H_norm) * (alpha_max - alpha_min)
```

映射方向是：

- saliency 高熵、分布更分散 -> alpha 更低 -> 高分 top-k 比例降低，空间补偿更多
- saliency 低熵、分布更集中 -> alpha 更高 -> 高分 top-k 比例提高，空间补偿更少

这让策略可以在全局上下文型样本和局部聚焦型样本之间动态调整，而不需要为所有样本固定同一个
`high_ratio`。

## 4. 代码位置

| 分支 | 主要实现位置 |
| --- | --- |
| `baseline` | `entropy_exp/src/prune_inference.py::run_baseline_inference()` |
| `attn_score` | `entropy_exp/src/strategies/attn_score.py` |
| `pre_attn_score` | `entropy_exp/src/strategies/pre_attn_score.py`、`entropy_exp/src/pruner.py::_run_pre_prune_layer()` |
| `masking_attn_score` | `entropy_exp/src/strategies/masking_attn_score.py`、`entropy_exp/src/pruner.py::_run_masking_prune_layer()` |
| `tail_masking_attn_score` | `entropy_exp/src/strategies/tail_masking_attn_score.py`、`entropy_exp/src/prune_inference.py::derive_effective_prune_config()` |
| `entropy` | `entropy_exp/src/strategies/entropy.py` |
| `random` | `entropy_exp/src/strategies/random.py` |
| `sparsevlm` | `entropy_exp/src/strategies/sparsevlm.py` |
| `sparsevlm_adaptive_stratified` | `entropy_exp/src/strategies/sparsevlm_adaptive_stratified.py` |
| `sparsevlm_entropy_alpha` | `entropy_exp/src/strategies/sparsevlm_entropy_alpha.py` |

更细的 transformer block 修改路径见 `entropy_exp/TRANSFORMER_BLOCK_STRATEGY_PATHS.md`。

## 5. 继承和复用关系

```text
PruneStrategy
├── AttnScoreStrategy
│   ├── PreAttnScoreStrategy
│   ├── MaskingAttnScoreStrategy
│   │   └── TailMaskingAttnScoreStrategy
├── EntropyStrategy
├── RandomStrategy
└── SparseVLMStrategy
    └── SparseVLMAdaptiveStratifiedStrategy
        └── SparseVLMEntropyAlphaStrategy
```

`baseline` 不继承 `PruneStrategy`，因为它不是剪枝策略；它是同推理路径下的无剪枝运行模式。

