# Strategy Branch Summary

本文档总结 `entropy_exp` 当前可运行的策略分支及其大致方法。这里的“策略分支”指运行输出目录中的
`outputs/runs/{strategy}/{dataset}/...` 分支，以及 `run_prune.sh` / scheduler 可直接调度的策略名。

当前共有 **18 个运行分支**：

- **1 个无剪枝对照分支**：`baseline`
- **17 个剪枝策略分支**：见下表

剪枝策略注册在 `entropy_exp/src/strategies/__init__.py` 的 `STRATEGY_REGISTRY` 中；`baseline`
通过 `prune_inference.py --baseline` 进入同一套自定义 decode 路径，但不执行剪枝。

## 1. 总览表

| 分支 | 家族 | 大致方法 | 执行形态 | 主要用途 |
| --- | --- | --- | --- | --- |
| `baseline` | 对照 | 同一推理入口，无视觉 token 剪枝 | 不删 token | 质量和速度上限 |
| `attn_score` | attention | 当前层 forward 后，用所有 text query 到 visual key 的 attention 均值打分 | post-layer 物理剪枝 | 基础 attention 剪枝 |
| `pre_attn_score` | attention | 进入目标层前手工计算 QK attention，先剪再 forward | pre-layer 物理剪枝 | 验证目标层前剪枝 |
| `masking_attn_score` | attention | 进入目标层前打分，但只屏蔽 text->vision attention logits | target-layer masking | 不缩短序列的 layer-local 对照 |
| `tail_masking_attn_score` | attention | 从起始层到最后一层持续执行 masking | target-layer masking | 验证 tail layers 连续 masking |
| `entropy` | attention | text->vision attention 按 head 熵加权，低熵 head 权重更高 | post-layer 物理剪枝 | 引入 head confidence |
| `random` | 对照 | 随机 visual importance，复用 top-k keep | post-layer 物理剪枝 | 随机 sanity check |
| `sparsevlm` | SparseVLM | 先选 image-relevant text raters，再用 raters 的 text->vision attention 打分 | post-layer 物理剪枝 | SV2 / text-rater baseline |
| `sparsevlm_adaptive_stratified` | Ours alpha | SparseVLM saliency top 部分 + 固定比例空间 stratum 补偿 | post-layer 物理剪枝 | 固定空间补偿 |
| `sparsevlm_entropy_alpha` | Ours alpha | 用 saliency 熵动态决定 high-saliency 与空间补偿比例 alpha | post-layer 物理剪枝 | 单层/逐层自适应补偿 |
| `sparsevlm_entropy_alpha_global` | Ours alpha | 在 entropy-alpha 上加入跨层 saliency EMA 与 stratum debt | post-layer 物理剪枝 | 多层全局记忆补偿 |
| `sparsevlm_boost` | Boost | 对欠表达 grid 的 token 加确定性 boost，再统一 top-k | post-layer 物理剪枝 | 替代硬配额补偿 |
| `sparsevlm_boost_hybrid` | Boost | 按层在纯 SparseVLM (`S`) 与 boost (`O`) 间切换 | post-layer 物理剪枝 | O-S-S / O-O-S 层级混合 |
| `sparsevlm_compensated` | Sampling | saliency 分布幂次平滑后固定 seed 顺序采样 | post-layer 物理剪枝 | 无 grid 的概率补偿对照 |
| `sparsevlm_diverse_mmr` | Diversity | grid 内 saliency anchors + candidate pool，补选只看 max-min 距离 | post-layer 物理剪枝 | grid-local pure-distance diversity |
| `sparsevlm_adaptive_diverse_mmr` | Diversity | 随当前层 prune ratio 自动调节 saliency anchors 与 diversity fills | post-layer 物理剪枝 | 低剪枝靠 saliency，高剪枝靠 diversity |
| `sparsevlm_scnd` | SCND | saliency-constrained native DivPrune + 后层 boundary refinement | post-layer 物理剪枝 | 全局 native diversity under saliency constraints |
| `sparsevlm_fast_scnd` | SCND | saliency core + 小候选池 + 最多 16 步 micro-greedy，后层 cached tie-break | post-layer 物理剪枝 | 低复杂度 SCND |

## 2. 公共控制参数

| 参数 | 作用 |
| --- | --- |
| `pruning.strategy` | 选择使用哪个策略分支 |
| `pruning.prune_layers` | 指定在哪些 transformer layer 执行剪枝或 masking |
| `pruning.prune_ratio` | 指定每个目标层移除 visual token 的比例；标量会广播到所有目标层，列表会和 `prune_layers` 配对 |
| `pruning.layer_selection` | 当前主要使用 `fixed`；`dynamic` 仍是预留路径 |
| `capture.save_importance_scores` | 是否把 importance / visual scores 写入 `stats.jsonl` |
| `capture.save_keep_indices` | 是否把 keep / pruned indices 写入 `stats.jsonl` |

注意：`strategy` 只决定“怎么打分”和“怎么施加剪枝动作”；`prune_layers` 和 `prune_ratio`
仍然独立决定“在哪里剪”和“剪多少”。

## 3. 策略家族说明

### 3.1 基础 attention 系列

`attn_score`、`pre_attn_score`、`masking_attn_score`、`tail_masking_attn_score`、`entropy` 是最早的机制对照：

- `attn_score` 在目标层完整 forward 后，用 text->vision attention 聚合得到 visual importance。
- `pre_attn_score` 把相同打分提前到目标层前，目标层自身只处理保留 token。
- `masking_attn_score` 不物理删 token，只在目标层屏蔽被剪 visual key 的 attention logits。
- `tail_masking_attn_score` 把 masking 从起始层连续应用到最后一层。
- `entropy` 用 head-level entropy 给 attention head 加权，低熵 head 被视为更可信。

这些策略主要用于回答“剪枝发生位置”和“attention 聚合方式”对结果的影响。

### 3.2 `sparsevlm`

`sparsevlm` 是当前 SV2 口径的本地实现分支。它先从初始 multimodal embedding 中筛出与图像相关的 text raters，
再在每个 prune layer 只用这些 raters 的 text->vision attention 给 visual token 打分，最后按 saliency top-k 保留。

它是后续 Ours / diversity 策略的共同 saliency 基座。

### 3.3 空间补偿 alpha 系列

`sparsevlm_adaptive_stratified`、`sparsevlm_entropy_alpha`、`sparsevlm_entropy_alpha_global` 都保留 SparseVLM text-rater saliency，
区别在如何处理“只按 saliency top-k 可能过度集中”的问题：

- `sparsevlm_adaptive_stratified`：固定 `high_ratio`，先保留高 saliency token，再把剩余预算分给空间欠覆盖的 grid / stratum。
- `sparsevlm_entropy_alpha`：用当前层 saliency 分布熵计算 alpha。高熵表示 saliency 不确定、补偿更多；低熵表示 saliency 更集中、更接近 SparseVLM。
- `sparsevlm_entropy_alpha_global`：在线维护跨层 saliency EMA、已观测状态、drop layer 和 stratum keep exposure debt。第一层可退化到旧 entropy-alpha，后续层用当前 saliency 与历史 saliency / 历史空间欠账共同决定高分保留和补偿配额。

这组策略的核心变量是“saliency 主导”和“空间覆盖补偿”的比例。

### 3.4 Boost / compensated 系列

`sparsevlm_boost` 用确定性加分替代硬配额补偿：

```text
adjusted_score = mixed_saliency_score + boost_weight * grid_deficit
```

其中 `boost_weight` 随 saliency 熵增大而增大。高熵时适当增强覆盖，低熵时更接近 pure saliency top-k。
`use_score_memory=true` 时，多层可使用 saliency EMA；混合实验通常显式设为 `false`，避免和 O-S-S 口径混淆。

`sparsevlm_boost_hybrid` 支持层模式：

| 模式 | 含义 |
| --- | --- |
| `S` | pure `sparsevlm` saliency top-k |
| `O` | `sparsevlm_boost` score boosting |

典型矩阵是 `O-S-S` 和 `O-O-S`，用来验证浅层 boost、深层回到 SparseVLM 是否更稳。

`sparsevlm_compensated` 不使用 grid。它把 saliency / mixed score 经过 `score ** beta` 平滑后，用固定 seed 的顺序采样保留 token。高熵映射到更低 beta，选择更均匀；低熵映射到更高 beta，更信任 saliency。

### 3.5 Diversity 系列

`sparsevlm_diverse_mmr` 当前是 **grid-local pure-distance** 版本：

1. 仍用 SparseVLM saliency 做“进门”：决定每个 grid 的 quota、saliency anchors 和 candidate pool。
2. 每个 grid 内先保留 saliency anchors。
3. grid 内剩余 token 只按与已选集合的 max-min cosine distance 补选，不再用 saliency + lambda 加权。
4. 若某个 grid 候选不足，再用全局 saliency 补齐，保证最终 keep 数严格等于目标 K。

层模式：

| 模式 | 含义 |
| --- | --- |
| `D` | grid-local pure-distance diversity |
| `S` | pure `sparsevlm` |

`sparsevlm_adaptive_diverse_mmr` 在上述结构上把 diversity 强度绑定到当前层 `prune_ratio`：

```text
diversity_ratio = clamp(prune_ratio, 0, 1)
anchor_ratio = 1 - diversity_ratio
```

低剪枝率时 saliency anchors 占主导；高剪枝率时 candidate pool 自动变宽，diversity fills 增加。它只引入一个主要自适应量，避免 task-specific 超参堆叠。

层模式：

| 模式 | 含义 |
| --- | --- |
| `A` | adaptive grid saliency-diversity |
| `S` | pure `sparsevlm` |

### 3.6 SCND 系列

`sparsevlm_scnd` 是 **Saliency-Constrained Native DivPrune**：

- `C` 层：不再按 grid quota，而是在全局 visual token 上做 native max-min diversity；但 seed、候选可行性和 repair 都受 saliency mass floor 约束，防止低 saliency outlier 过度进入。
- `B` 层：主体是 saliency top-k，只在 cutoff 附近 boundary band 用 diversity tie-break；低熵或 cutoff margin 很明确时自动接近 pure saliency。
- `S` 层：pure `sparsevlm`。

典型层模式：

| 模式串 | 含义 |
| --- | --- |
| `C-B-B` | 首层全局 SCND，后两层 boundary refinement |
| `C-B-S` | 首层 SCND，中层 boundary，末层 SparseVLM |
| `C-S-S` | 只在首层引入 SCND |

`sparsevlm_fast_scnd` 是低复杂度版本：

- `F` 层：saliency core + 小候选池 + fixed low-dim projection，最多执行 `greedy_steps=16` 步 max-min micro-greedy，剩余 budget 用 saliency 补齐。
- `T` 层：不重新计算 pairwise distance，只用 `F` 层缓存的 diversity gain 在 saliency cutoff 附近做轻量 tie-break。
- `S` 层：pure `sparsevlm`。

默认层模式会根据剪枝层数自动展开：单层 `[F]`，两层 `[F,T]`，三层及以上 `[F,T,S,...]`。正式对比中常用 `F-T-S` 和 `F-S-S`。

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
| `sparsevlm_entropy_alpha_global` | `entropy_exp/src/strategies/sparsevlm_entropy_alpha_global.py` |
| `sparsevlm_boost` | `entropy_exp/src/strategies/sparsevlm_boost.py` |
| `sparsevlm_boost_hybrid` | `entropy_exp/src/strategies/sparsevlm_boost_hybrid.py` |
| `sparsevlm_compensated` | `entropy_exp/src/strategies/sparsevlm_compensated.py` |
| `sparsevlm_diverse_mmr` | `entropy_exp/src/strategies/sparsevlm_diverse_mmr.py` |
| `sparsevlm_adaptive_diverse_mmr` | `entropy_exp/src/strategies/sparsevlm_adaptive_diverse_mmr.py` |
| `sparsevlm_scnd` | `entropy_exp/src/strategies/sparsevlm_scnd.py` |
| `sparsevlm_fast_scnd` | `entropy_exp/src/strategies/sparsevlm_fast_scnd.py` |

更细的 transformer block 修改路径见 [TRANSFORMER_BLOCK_STRATEGY_PATHS.md](./TRANSFORMER_BLOCK_STRATEGY_PATHS.md)。

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
    ├── SparseVLMAdaptiveStratifiedStrategy
    │   └── SparseVLMEntropyAlphaStrategy
    │       └── SparseVLMEntropyAlphaGlobalStrategy
    └── SparseVLMScoreMemoryStrategy
        ├── SparseVLMBoostStrategy
        │   └── SparseVLMBoostHybridStrategy
        ├── SparseVLMCompensatedStrategy
        ├── SparseVLMDiverseMMRStrategy
        │   ├── SparseVLMAdaptiveDiverseMMRStrategy
        │   └── SparseVLMSCNDStrategy
        └── SparseVLMFastSCNDStrategy
```

`baseline` 不继承 `PruneStrategy`，因为它不是剪枝策略；它是同推理路径下的无剪枝运行模式。
