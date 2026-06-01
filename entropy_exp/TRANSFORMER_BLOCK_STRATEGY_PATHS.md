# Transformer Block 策略路径

本文面向策略开发者，只说明各剪枝策略如何进入 transformer block，以及会修改哪些运行时对象。策略方法总览见 [STRATEGY_BRANCH_SUMMARY.md](./STRATEGY_BRANCH_SUMMARY.md)。

## 1. 公共入口

所有剪枝分支共享入口：

```text
run_prune.sh
  -> prune_inference.py::run_prune_inference()
  -> VisualTokenPruner.pruned_generate()
  -> VisualTokenPruner._pruned_prefill()
  -> 根据 strategy.prune_stage() 选择 post / pre / masking 路径
```

核心对象：

| 对象 | 含义 |
| --- | --- |
| `hidden_states` | 当前序列表示 |
| `past_kv` | transformer KV cache |
| `position_ids` | 位置编号 |
| `causal_mask` | causal attention mask |
| `attn_weights` | 当前层 attention，部分策略从中取 text->vision 子矩阵 |
| `current_visual_embeds` | 当前层可见 visual token hidden states，diversity / SCND 策略用来计算表征距离 |

## 2. 三类执行路径

### post-layer physical pruning

路径：

```text
target layer full forward
  -> strategy.compute_keep_mask(...)
  -> hidden_states = hidden_states[:, full_keep]
  -> _prune_kv_cache(..., up_to_layer=layer_idx)
  -> refresh sequence state
  -> strategy.update_after_prune(...)
```

特点：

- target layer 先完整处理原始 visual tokens。
- 剪枝发生在 target layer 输出后。
- 后续层看到更短序列。
- 会物理裁剪 `hidden_states` 和当前层及之前层的 KV cache。
- `current_patch_indices` 会随物理剪枝同步更新，让后续层仍能追踪 surviving token 的原始 patch id。

使用该路径：

- `attn_score`
- `entropy`
- `random`
- `sparsevlm`
- `sparsevlm_adaptive_stratified`
- `sparsevlm_entropy_alpha`
- `sparsevlm_entropy_alpha_global`
- `sparsevlm_boost`
- `sparsevlm_boost_hybrid`
- `sparsevlm_compensated`
- `sparsevlm_diverse_mmr`
- `sparsevlm_adaptive_diverse_mmr`
- `sparsevlm_scnd`
- `sparsevlm_fast_scnd`

### pre-layer physical pruning

路径：

```text
_compute_pre_prune_scores(...)
  -> strategy.compute_keep_mask_from_importance(...)
  -> hidden_states = hidden_states[:, full_keep]
  -> _prune_kv_cache(..., up_to_layer=layer_idx - 1)
  -> target layer forward on shortened sequence
```

特点：

- 进入 target layer 前先手工计算该层 attention score。
- target layer 本身只处理保留下来的 visual tokens。
- 目标层之前的 KV cache 被裁剪；目标层生成的 KV cache 天然是短序列。

使用该路径：

- `pre_attn_score`

### target-layer logits masking

路径：

```text
_compute_pre_prune_scores(...)
  -> strategy.compute_keep_mask_from_importance(...)
  -> _forward_masked_layer(...)
  -> mask text query -> dropped visual key attention logits
```

特点：

- 不物理删除 token。
- `hidden_states`、`position_ids`、KV cache 长度都不变。
- 只在目标层内部屏蔽 `text query -> dropped visual key` 的 attention logits。

使用该路径：

- `masking_attn_score`
- `tail_masking_attn_score`

`tail_masking_attn_score` 的差异在层展开：用户配置一个起始层，`prune_inference.py::derive_effective_prune_config()` 会展开成从该层到最后一层的连续 masking。

## 3. 策略 block 行为矩阵

| 策略 | 决策时机 | 是否物理裁剪 token | 是否裁剪 KV cache | 是否改 attention logits | 后续层序列长度变化 |
| --- | --- | --- | --- | --- | --- |
| `attn_score` | target layer 后 | 是 | 是 | 否 | 是 |
| `pre_attn_score` | target layer 前 | 是 | 是，先到 `layer_idx - 1` | 否 | 是 |
| `masking_attn_score` | target layer 前决策，层内执行 | 否 | 否 | 是 | 否 |
| `tail_masking_attn_score` | 起始层到最后一层，每层前决策并层内执行 | 否 | 否 | 是 | 否 |
| `entropy` | target layer 后 | 是 | 是 | 否 | 是 |
| `random` | target layer 后 | 是 | 是 | 否 | 是 |
| `sparsevlm` | target layer 后 | 是 | 是 | 否 | 是 |
| `sparsevlm_adaptive_stratified` | target layer 后 | 是 | 是 | 否 | 是 |
| `sparsevlm_entropy_alpha` | target layer 后 | 是 | 是 | 否 | 是 |
| `sparsevlm_entropy_alpha_global` | target layer 后 | 是 | 是 | 否 | 是 |
| `sparsevlm_boost` | target layer 后 | 是 | 是 | 否 | 是 |
| `sparsevlm_boost_hybrid` | target layer 后 | 是 | 是 | 否 | 是 |
| `sparsevlm_compensated` | target layer 后 | 是 | 是 | 否 | 是 |
| `sparsevlm_diverse_mmr` | target layer 后 | 是 | 是 | 否 | 是 |
| `sparsevlm_adaptive_diverse_mmr` | target layer 后 | 是 | 是 | 否 | 是 |
| `sparsevlm_scnd` | target layer 后 | 是 | 是 | 否 | 是 |
| `sparsevlm_fast_scnd` | target layer 后 | 是 | 是 | 否 | 是 |

## 4. 打分和选择差异摘要

| 策略 | importance / keep 决策来源 |
| --- | --- |
| `attn_score` | 当前层 text->vision attention，对 head 和 text query 求均值 |
| `pre_attn_score` | target layer 前手工算 QK softmax，再用 `attn_score` 聚合 |
| `masking_attn_score` | 同 `pre_attn_score`，但结果用于 logits mask |
| `tail_masking_attn_score` | 同 `masking_attn_score`，应用到 tail layers |
| `entropy` | text->vision attention，但按 head 熵加权，低熵 head 权重更高 |
| `random` | `torch.rand(v_token_num)` |
| `sparsevlm` | sample-level text raters 的 text->vision attention |
| `sparsevlm_adaptive_stratified` | SparseVLM saliency + 固定 high ratio 的空间分层补偿 |
| `sparsevlm_entropy_alpha` | SparseVLM saliency + saliency 熵自适应 alpha |
| `sparsevlm_entropy_alpha_global` | 当前 saliency 与跨层 saliency EMA / stratum debt 混合 |
| `sparsevlm_boost` | SparseVLM saliency + grid deficit 加分后统一 top-k |
| `sparsevlm_boost_hybrid` | 按层选择 pure SparseVLM (`S`) 或 boost (`O`) |
| `sparsevlm_compensated` | saliency 幂次平滑后固定 seed 顺序采样 |
| `sparsevlm_diverse_mmr` | grid quota / anchors 由 saliency 决定，补选只看 grid 内 max-min distance |
| `sparsevlm_adaptive_diverse_mmr` | 用当前 prune ratio 自动调节 saliency anchors 与 distance fills |
| `sparsevlm_scnd` | 首层全局 saliency-constrained native diversity，后层可做 boundary refinement |
| `sparsevlm_fast_scnd` | 首层低成本 micro-greedy diversity，后层 cached diversity tie-break |

## 5. 新策略接入检查表

新增策略时，先明确三件事：

- `prune_stage()` 返回 `post`、`pre` 还是 `masking`。
- 策略是否需要 attention，即 `requires_attention()`。
- 决策结果是物理删除 token，还是只修改 attention 边。

然后同步补齐：

- 策略实现和单测
- `STRATEGY_REGISTRY`
- `entropy_exp/configs/prune.yaml`
- `run_prune.sh` 策略白名单和帮助文本
- scheduler / run layout / run discovery 白名单
- [STRATEGY_BRANCH_SUMMARY.md](./STRATEGY_BRANCH_SUMMARY.md)
- 需要时扩展 report 脚本，让 `stats.jsonl` 中的新字段可被汇总
