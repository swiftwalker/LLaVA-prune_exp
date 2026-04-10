# Transformer Block 修改路径总结

本文只总结剪枝分支，不包含 `baseline`。目标是回答两个问题：

1. 每个分支的剪枝决策是在哪个阶段、通过哪条函数路径进入 `transformer block` 的。
2. 每个分支到底改了 block 内的什么对象：`hidden_states`、`KV cache`、`position_ids`、`attention logits`，还是都不改。

---

## 1. 公共调用链

所有剪枝分支的公共入口都是：

```text
entropy_exp/scripts/run_prune.sh
  -> entropy_exp/src/prune_inference.py::run_prune_inference()
  -> VisualTokenPruner.pruned_generate()
  -> VisualTokenPruner._pruned_prefill()
  -> 根据 strategy.prune_stage() 进入不同 block 修改路径
```

公共职责分工如下：

- `run_prune.sh`
  - 解析策略名、数据集、`--set` 覆盖项，最后调用 `prune_inference.py`
- `prune_inference.py::run_prune_inference()`
  - 构造多模态输入
  - 创建 strategy 实例
  - 创建 `VisualTokenPruner`
  - 调用 `pruned_generate()`
- `VisualTokenPruner._pruned_prefill()`
  - 逐层遍历 `self.model.model.layers`
  - 在目标 prune layer 根据 `strategy.prune_stage()` 选择不同改动路径

---

## 2. 统一观察点

所有分支都围绕下面几个对象做文章：

- `hidden_states`
  - 当前层输入/输出的整条序列表示
- `past_kv`
  - 各层 KV cache
- `position_ids`
  - 物理裁剪后是否需要重建位置编号
- `causal_mask`
  - 物理裁剪后是否需要重建 causal mask
- `attn_weights`
  - 当前层 `text -> vision` 注意力子矩阵

在 `_pruned_prefill()` 里，差异主要由下面两个条件决定：

- `prune_stage == "post"`
- `prune_stage == "pre"`
- `prune_stage == "masking"`

---

## 3. `attn_score`

### 3.1 入口路径

```text
_pruned_prefill()
  -> 普通 layer forward(output_attentions=True)
  -> AttnScoreStrategy.compute_keep_mask()
  -> AttnScoreStrategy.compute_importance()
  -> 物理裁剪 hidden_states + KV cache
```

### 3.2 block 内修改方式

- 当前 target layer 先正常 forward
- 重要性分数来自当前层已经算出的 `attn_weights`
- 取子矩阵：
  - `attn_weights[0, :, text_token_start:, v_token_start:v_end]`
- 再做：
  - head 维平均
  - text query 维平均
  - 得到每个 visual token 的 importance

### 3.3 对 block 状态的影响

- 当前层:
  - 已经按完整序列执行完 forward
- 当前层输出后立刻发生:
  - `hidden_states = hidden_states[:, full_keep]`
  - `_prune_kv_cache(past_kv, full_keep, up_to_layer=layer_idx)`
- 所以会同步裁剪：
  - 当前层及之前所有层的 KV cache
  - 当前层输出后的序列长度
- 然后会重建：
  - `position_ids`
  - `causal_mask`
- 结果:
  - 下一层开始看到的是更短的序列

### 3.4 核心结论

- `attn_score` 是典型的 **post-layer 物理裁剪**
- 改的是：
  - 当前层输出后的 `hidden_states`
  - 当前层及之前层的 `KV cache`
  - 后续层使用的 `position_ids / causal_mask`
- 不改的是：
  - 当前层内部 attention logits

---

## 4. `pre_attn_score`

### 4.1 入口路径

```text
_pruned_prefill()
  -> _run_pre_prune_layer()
  -> _compute_pre_prune_scores()
  -> PreAttnScoreStrategy.compute_keep_mask_from_importance()
  -> 先裁剪 hidden_states / 旧 KV
  -> 再执行 target layer forward
```

### 4.2 block 内修改方式

`pre_attn_score` 不等当前层 forward 结束后再剪，而是在进入目标层之前，手工走一遍当前层 attention 前半段，提前得到打分：

- `layer.input_layernorm(hidden_states)`
- `self_attn.q_proj / k_proj`
- `rotary_emb`
- `QK^T / sqrt(d)`
- `softmax`
- 提取 `text -> vision` 子矩阵
- 用与 `attn_score` 相同的聚合规则生成 importance

注意，这一步只是为了“预判当前层会如何看待 visual token”，并没有执行完整 block。

### 4.3 对 block 状态的影响

在 target layer 正式 forward 前，先做：

- `hidden_states = hidden_states[:, full_keep]`
- `_prune_kv_cache(past_kv, full_keep, up_to_layer=layer_idx - 1)`

然后立即重建：

- `position_ids`
- `causal_mask`

最后 target layer 在更短的序列上正常 forward。

因此：

- 目标层之前的 KV cache 被裁短
- 目标层本身从一开始就只处理保留下来的 visual token
- 目标层产生的 KV cache 天生就是短的

### 4.4 核心结论

- `pre_attn_score` 是 **pre-layer 物理裁剪**
- 改动发生得比 `attn_score` 更早
- 它真正改变的是：
  - target layer 的输入序列
  - target layer 之前的 KV cache
  - target layer 及后续层的序列长度

---

## 5. `masking_attn_score`

### 5.1 入口路径

```text
_pruned_prefill()
  -> _run_masking_prune_layer()
  -> _compute_pre_prune_scores()
  -> MaskingAttnScoreStrategy.compute_keep_mask_from_importance()
  -> _forward_masked_layer()
```

### 5.2 block 内修改方式

前半段打分与 `pre_attn_score` 完全相同：

- 在 target layer forward 前先手工计算当前层 `text -> vision` attention score
- 得到 keep/drop visual token 集合

但真正执行 target layer 时，不做物理裁剪，而是进入 `_forward_masked_layer()`，手工重写一遍 Llama-style decoder layer 的 attention 路径：

- `input_layernorm`
- `q_proj / k_proj / v_proj`
- `rotary_emb`
- 写入当前层 `past_kv`
- 计算 attention logits
- 只对 `text query -> dropped vision key` 位置加上 `-inf`
- 再 `softmax`
- 再做 `attn_output`
- 再走 residual + MLP

关键 masking 位置是：

- query 侧: `text_slice = slice(text_token_start, seq_len)`
- key 侧: `vision_drop_indices = dropped_visual + v_token_start`

因此被屏蔽的是：

- 当前 target layer 内部
- 文本 token 对被剪枝 visual token 的注意力路径

### 5.3 对 block 状态的影响

不会物理裁剪：

- `hidden_states` 长度不变
- `position_ids` 不重建
- `causal_mask` 不重建
- 当前层及之前层 `KV cache` 长度不变

只会改变：

- 当前 target layer 内部 attention logits

所以从序列形态上看：

- `original_seq_len == final_seq_len`
- 视觉 token 仍然存在，只是 target layer 中对其中一部分“看不见”

### 5.4 核心结论

- `masking_attn_score` 是 **layer-local logits masking**
- 它不是删 token，而是在 target layer 里“禁用部分 text -> vision attention 边”
- 改的是：
  - 当前层 attention logits
- 不改的是：
  - hidden/KV 的物理长度
  - 后续层的 position_ids

---

## 6. `entropy`

### 6.1 入口路径

```text
_pruned_prefill()
  -> 普通 layer forward(output_attentions=True)
  -> EntropyStrategy.compute_keep_mask()
  -> EntropyStrategy.compute_importance()
  -> 物理裁剪 hidden_states + KV cache
```

### 6.2 block 内修改方式

`entropy` 和 `attn_score` 共用同一条 **post-layer 物理裁剪** 路径，区别只在 importance 聚合方式：

- 仍然先取 `text -> vision` attention 子矩阵
- 但不是 head 直接平均
- 而是先算每个 head 在 visual token 维上的 entropy
- 低熵 head 权重更高
- 再做加权 head 聚合
- 最后再对 text query 维平均

### 6.3 动态比例

`EntropyStrategy.get_prune_ratio()` 可选按 importance 分布的熵动态调大剪枝比例：

- importance 越集中
- 归一化熵越低
- 可以 prune 得更激进

但这只改变“剪多少”，不改变 block 修改位置。

### 6.4 核心结论

- `entropy` 的 block 修改位置与 `attn_score` 相同
- 差异只在：
  - importance score 的 head 聚合方式
  - 可选动态 `prune_ratio`

---

## 7. `random`

### 7.1 入口路径

```text
_pruned_prefill()
  -> 普通 layer forward(output_attentions=False, unless capture needed)
  -> RandomStrategy.compute_keep_mask()
  -> RandomStrategy.compute_importance()
  -> 物理裁剪 hidden_states + KV cache
```

### 7.2 block 内修改方式

`random` 不依赖 attention：

- `requires_attention() -> False`
- importance 直接由 `torch.rand(v_token_num)` 生成

因此在 prune layer：

- 当前层通常不需要打开 `output_attentions`
- target layer 正常 forward 完
- 然后像 `attn_score` 一样做 post-layer 物理裁剪

### 7.3 对 block 状态的影响

与 `attn_score` 相同：

- 当前层输出后裁剪 `hidden_states`
- 同步裁剪当前层及之前层 `KV cache`
- 重建 `position_ids / causal_mask`
- 下一层看到更短序列

### 7.4 核心结论

- `random` 的 block 改动位置与 `attn_score` 一致
- 唯一差别是：
  - 剪枝决策不来自 block attention，而是随机数

---

## 8. `sparsevlm`

### 8.1 入口路径

```text
_pruned_prefill()
  -> strategy.prepare_sample()
     -> SparseVLMStrategy.prepare_sample()
     -> select_text_raters()
  -> prune layer 普通 forward(output_attentions=True)
  -> SparseVLMStrategy.compute_keep_mask()
  -> SparseVLMStrategy.compute_importance()
  -> compute_visual_scores_from_attention()
  -> prune_visual_tokens()
  -> 物理裁剪 hidden_states + KV cache
```

### 8.2 block 外的预处理

`sparsevlm` 比其他策略多一步 sample-level 预处理：

- 在进入 layer-by-layer prefill 前
- 从 `inputs_embeds` 中拆出：
  - visual block `H_v`
  - text block `H_q`
- 先选出一小组 text raters

选 rater 的逻辑是：

- 先用 `H_v @ H_q^T`
- 对 text 维做 softmax
- 得到 text relevance
- 去掉 special token（可选）
- 再用均值阈值筛选
- 若为空则 fallback top-k

这一步把 `rater_indices` 缓存在 `sample_context` 中。

### 8.3 block 内修改方式

真正到 prune layer 时，仍然先让当前 layer 正常 forward，拿到 `attn_weights`。

但 importance 不再由“所有 text token 平均”得到，而是只看选中的 text raters：

- 从当前层 `attn_weights` 里取:
  - 当前 text positions
  - 当前 visual positions
- 只保留 `rater_indices` 对应的 text query
- 在这些 raters 上对 visual token 聚合打分

之后进入 `prune_visual_tokens()`：

- 按得分从低到高删 token
- 同时保证至少保留 `min_visual_tokens_after_prune`

### 8.4 对 block 状态的影响

物理改动位置与 `attn_score` 相同：

- 当前层 forward 后裁剪 `hidden_states`
- 同步裁剪当前层及之前层 `KV cache`
- 重建 `position_ids / causal_mask`

额外不同点只有：

- importance 依赖 `sample_context` 里的 text raters
- keep/drop 时有最小保留 token 下限

### 8.5 核心结论

- `sparsevlm` 本质上仍是 **post-layer 物理裁剪**
- 但它把“哪些 text token 有资格给 visual token 打分”提前固定下来
- 因而 block 内部真正用到的是：
  - 当前层 attention
  - 预先选出的 text raters

---

## 9. `sparsevlm_adaptive_stratified`

### 9.1 入口路径

```text
_pruned_prefill()
  -> strategy.prepare_sample()
     -> SparseVLMAdaptiveStratifiedStrategy.prepare_sample()
     -> SparseVLMStrategy.prepare_sample()
     -> select_text_raters()
  -> prune layer 普通 forward(output_attentions=True)
  -> SparseVLMAdaptiveStratifiedStrategy.compute_keep_mask()
     -> SparseVLMStrategy.compute_importance()
     -> compute_visual_scores_from_attention()
     -> build_stratum_index()
     -> allocate_adaptive_stratified_quotas()
  -> 物理裁剪 hidden_states + KV cache
  -> strategy.update_after_prune()
```

### 9.2 与 `sparsevlm` 的区别

`sparsevlm_adaptive_stratified` 的打分部分和 `sparsevlm` 完全相同：

- 先选 text raters
- 再用当前层 text->vision attention 计算 visual token 分数

不同点只在 keep/drop 决策阶段：

- 不是直接按 visual score 取 top-k
- 而是先保留一部分高分 patch
- 再按空间 strata 的缺口分配补偿名额
- 在每个 stratum 内部做 `random` 或 `farthest` 选择

### 9.3 对 block 状态的影响

它仍然沿用 `sparsevlm` 的 **post-layer 物理裁剪** 路径：

- 当前层先完整 forward
- forward 后裁剪 `hidden_states`
- 同步裁剪当前层及之前层 `KV cache`
- 保留 `keep-position-ids` 分支已有的 `position_ids` 语义

额外新增的是 sample-level 状态：

- `current_patch_indices`
- 每次物理剪枝后用当前层 `keep_indices` 更新
- 后续层始终按原始 patch 坐标做 strata 映射

---

## 10. 总结对照表

| 分支 | 决策时机 | target block 是否先完整 forward | 是否物理裁剪 token | 是否裁剪 KV cache | 是否改 attention logits | 后续层序列长度是否变化 |
| --- | --- | --- | --- | --- | --- | --- |
| `attn_score` | target layer 之后 | 是 | 是 | 是 | 否 | 是 |
| `pre_attn_score` | target layer 之前 | 否 | 是 | 是（先到 `layer_idx-1`，目标层生成短 KV） | 否 | 是 |
| `masking_attn_score` | target layer 之前决策，target layer 内执行 mask | 否，改为手写 masked forward | 否 | 否 | 是 | 否 |
| `entropy` | target layer 之后 | 是 | 是 | 是 | 否 | 是 |
| `random` | target layer 之后 | 是 | 是 | 是 | 否 | 是 |
| `sparsevlm` | target layer 之后 | 是 | 是 | 是 | 否 | 是 |
| `sparsevlm_adaptive_stratified` | target layer 之后 | 是 | 是 | 是 | 否 | 是 |

---

## 11. 一句话区分

- `attn_score / entropy / random / sparsevlm / sparsevlm_adaptive_stratified`
  - 都是 **post-layer 物理裁剪**
- `pre_attn_score`
  - 是 **pre-layer 物理裁剪**
- `masking_attn_score`
  - 是 **target layer 内部 attention logits masking**

如果后面你要继续扩展新策略，最先要决定的其实就是两件事：

1. 它属于 `pre`、`post` 还是 `masking`
2. 它修改的是 token 物理存在性，还是只修改 attention 边

---

## 12. `keep-position-ids` git 分支说明

### 12.1 分支目的

该分支名为：

```text
keep-position-ids
```

它基于当前 `entropy_exp` 剪枝框架，专门验证一个改动：

- 在 **物理剪枝仍然发生** 的情况下
- 不再把 surviving token 的 `position_ids` 重建为连续的 `0..new_len-1`
- 而是保留它们在原始多模态序列中的位置编号

换句话说，这个分支改的是：

- **位置编号语义**

但不改：

- 策略本身的打分规则
- 哪些 token 被删
- `hidden_states` / `KV cache` 的物理裁剪行为
- `masking_attn_score` 的 layer-local masking 路径

### 12.2 代码层改动入口

这个分支的核心改动都在下面两个文件：

- `entropy_exp/src/pruner.py`
- `entropy_exp/src/prune_inference.py`

实现路径是：

```text
prune_inference.py
  -> model.prepare_inputs_labels_for_multimodal()
  -> 拿到上游返回的 position_ids
  -> pruner.pruned_generate(..., position_ids=position_ids)

pruner.py
  -> _pruned_prefill(initial_position_ids)
  -> 物理剪枝后 position_ids = position_ids[:, full_keep]
  -> 只重建 causal_mask，不重建 position_ids
  -> decode 时新 token 从 final_position_ids[-1] + 1 继续编号
```

### 12.3 与主分支行为的差异

主分支（连续重建位置）在物理剪枝后会做：

```text
position_ids = [0, 1, ..., new_len-1]
causal_mask  = new_len 对应的新 mask
```

`keep-position-ids` 分支则改为：

```text
position_ids = old_position_ids[:, full_keep]
causal_mask  = new_len 对应的新 mask
```

因此当前分支采用的是下面这套兼容语义：

- **序列物理长度**：跟着剪枝后的 `hidden_states / KV cache` 走
- **RoPE 位置坐标**：沿用 surviving token 的原始位置编号

举例：

```text
原始:
tokens        = [BOS, V1, V2, V3, T1, T2]
position_ids  = [10, 11, 12, 13, 14, 15]

若剪掉 V3:
tokens        = [BOS, V1, V2, T1, T2]
position_ids  = [10, 11, 12, 14, 15]
```

这里可以看到：

- 长度已经缩短
- 但位置编号没有压紧成 `[0,1,2,3,4]`

### 12.4 当前兼容性处理

这个分支并不是简单地“保留 position_ids”而已，它同时做了三件兼容动作：

1. **物理剪枝后按 `full_keep` 同步裁剪 `position_ids`**
   - 这样 surviving token 和缩短后的 `hidden_states` 仍然一一对齐

2. **物理剪枝后仍然重建 `causal_mask`**
   - 因为 attention 的长度兼容取决于当前实际序列长度，而不是取决于原始位置编号

3. **decode 时新 token 从最后一个保留位置继续编号**
   - 不是从 `cache_len` 继续
   - 而是从 `final_position_ids[-1] + 1` 继续

所以这个分支本质上采用的是：

- `length compatibility by compressed tensors`
- `position semantics by preserved position ids`

### 12.5 哪些策略受影响

只影响会做 **物理剪枝** 的策略：

- `attn_score`
- `pre_attn_score`
- `entropy`
- `random`
- `sparsevlm`
- `sparsevlm_adaptive_stratified`

不影响：

- `masking_attn_score`

原因是 `masking_attn_score` 不缩短序列，也就没有位置重建问题。

### 12.6 这个分支适合回答的问题

它主要用来区分下面两种效应：

1. 性能变化来自 **token 真被删掉**
2. 性能变化来自 **删掉 token 之后位置坐标也被压缩**

因此它最适合与主分支做成对实验，用来观察：

- 在同一套剪枝策略、同一套 layer/ratio 下
- “连续重建位置” 与 “保留原始位置” 的结果差异
