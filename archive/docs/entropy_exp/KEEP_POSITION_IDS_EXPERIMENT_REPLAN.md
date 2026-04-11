# Keep Position IDs 分支实验重排方案

本文记录 `keep-position-ids` 分支上的新一轮实验安排。目标不是立即启动全部任务，而是：

- 暂时归档旧分支 `prune-exp` 留下的 `outputs/runs/` 结果
- 在空的 `outputs/runs/` 下，为 `keep-position-ids` 分支重新规划全量实验
- 保持跨策略、跨数据集的公平对比

---

## 1. 适用范围

本轮计划覆盖所有 **策略分支**，不包含 `baseline`：

- `attn_score`
- `pre_attn_score`
- `masking_attn_score`
- `entropy`
- `random`
- `sparsevlm`

覆盖全部数据集：

- `gqa`
- `mme`
- `pope`

统一的 layer / ratio 网格：

- `prune_layers = [1] / [2] / [3]`
- `prune_ratio = [0.2] / [0.3] / [0.4] / [0.5] / [0.6] / [0.7]`

总规模：

- `3 datasets × 6 strategies × 3 layers × 6 ratios = 324 runs`

---

## 2. 统一实验约束

所有 run 默认保持以下对齐：

- `inference.seed=42`
- `pruning.layer_selection=fixed`
- `max_samples=null`，即跑全量数据集
- 不引入额外策略特有改动，除非该策略本身必须依赖对应配置

策略特有的保留项：

- `sparsevlm`
  - 保留默认 `fallback_topk=4`
  - 保留 `exclude_special_tokens=true`
  - 保留 `min_visual_tokens_after_prune=16`
- `entropy`
  - 默认使用固定比例
  - 本轮不单独打开 `dynamic_ratio`

---

## 3. 批次组织方式

推荐把每个 `(dataset, layer, ratio)` 视为一个对比批次，在同一批次里跑 6 个策略：

```text
one batch = one dataset + one layer + one ratio + six strategies
```

因此一共是：

- `54` 个批次
- 每批 `6` 个 run

推荐顺序：

1. 先 `gqa`
2. 再 `mme`
3. 最后 `pope`

每个数据集内部：

1. `layer=1`, `ratio=0.2 -> 0.7`
2. `layer=2`, `ratio=0.2 -> 0.7`
3. `layer=3`, `ratio=0.2 -> 0.7`

这样做的好处：

- 每个批次内部比较最公平
- 每完成一个批次，就能直接比较 6 个策略
- 即使中途暂停，也不会只留下单个策略的半截矩阵

---

## 4. 计划文件

为这轮实验生成一份可直接用于 `plan_batch_runs.py --dry-run` 的 YAML 计划：

```text
entropy_exp/plans/keep_position_ids_all_strategies_all_datasets.yaml
```

计划特征：

- `label`: `keep-position-ids-all-strategies-all-datasets`
- `evaluate: true`
- `summarize: true`
- `summary_output_dir` 指向新的 branch 专属 summary 目录，避免和旧结果冲突

推荐汇总输出目录：

```text
entropy_exp/outputs/summary/keep_position_ids_all_strategies_all_datasets
```

---

## 5. dry-run 校验目标

在真正执行前，dry-run 至少要确认：

- 总 planned runs 数量为 `324`
- 每条命令都落在 `keep-position-ids` 分支当前工作树
- 每条命令都带有：
  - 正确的 `strategy`
  - 正确的 `dataset`
  - 正确的 `pruning.prune_layers`
  - 正确的 `pruning.prune_ratio`
  - 固定 `inference.seed=42`

---

## 6. 与旧结果的关系

旧 `prune-exp` 分支结果只做临时归档，不删除。

本轮新实验的目标是：

- 在 `keep-position-ids` 分支下重新获得完整矩阵
- 之后再与旧分支归档结果做成对比较

因此这轮计划的重点是：

- 先把 **新分支自己的完整矩阵跑齐**
- 再谈跨分支比较

---

## 7. Follow-Up：`sparsevlm_adaptive_stratified` 54-run 单策略矩阵

在上面的 324-run 全策略矩阵之外，当前分支又补了一条单独的 follow-up 实验线：

- strategy: `sparsevlm_adaptive_stratified`
- datasets: `gqa` / `mme` / `pope`
- prune layers: `1` / `2` / `3`
- prune ratios: `0.2` / `0.3` / `0.4` / `0.5` / `0.6` / `0.7`
- total jobs: `54`

这条 follow-up 线不并入原来的 `324 runs` 统计，而是用 repo 内置 scheduler 单独调度：

```text
entropy_exp/plans/keep_position_ids_sparsevlm_adaptive_stratified_full_matrix.yaml
```

与旧的 `plan_batch_runs.py` 路线不同，这条矩阵采用 `scheduler-first`：

- `pool_size: 12`
- `gpu.min_free_gib: 16`
- `tmux.session_name: sched_keep_position_ids_adaptive`
- `environment.conda_sh: /data/liuyu/anaconda3/etc/profile.d/conda.sh`

推荐执行顺序：

```bash
source /data/liuyu/anaconda3/etc/profile.d/conda.sh
conda activate llava

python entropy_exp/scripts/run_scheduler.py \
  --plan entropy_exp/plans/keep_position_ids_sparsevlm_adaptive_stratified_full_matrix.yaml \
  --dry-run

python entropy_exp/scripts/run_scheduler.py \
  --plan entropy_exp/plans/keep_position_ids_sparsevlm_adaptive_stratified_full_matrix.yaml
```

中断后恢复：

```bash
python entropy_exp/scripts/run_scheduler.py \
  --state-dir entropy_exp/outputs/scheduler/keep-position-ids-sparsevlm-adaptive-stratified-full-matrix \
  --resume
```

这份 plan 的作用是：

- 给 `keep-position-ids` 分支补齐 adaptive stratified 的完整单策略网格
- 保持与现有 `layer / ratio / dataset` 轴一致，便于后续和其他策略或旧分支结果对齐比较
- 避免旧文档继续给出“当前分支只覆盖到 `sparsevlm`”的过时印象
