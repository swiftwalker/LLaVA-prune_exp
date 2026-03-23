# Scheduler

`run_scheduler.py` 是 `entropy_exp` 的实验调度器入口。它只负责：

- 读取计划 YAML
- 维护待做 / 运行中 / 已完成 / 最终失败队列
- 根据 GPU 可用显存把实验挂到 `tmux`
- 在失败时按全局重试额度回队
- 输出并持久化简单进度摘要

它 **不负责**：

- `eval`
- `summary`
- 自动比较实验结果

---

## 用法

```bash
python entropy_exp/scripts/run_scheduler.py --plan entropy_exp/plans/keep_position_ids_all_strategies_all_datasets.yaml --dry-run

python entropy_exp/scripts/run_scheduler.py --plan entropy_exp/plans/keep_position_ids_all_strategies_all_datasets.yaml

python entropy_exp/scripts/run_scheduler.py --state-dir entropy_exp/outputs/scheduler/keep-position-ids-all-strategies-all-datasets --resume
```

可选参数：

- `--pool-size`
  - 覆盖计划中的 `pool_size`
- `--state-dir`
  - 覆盖默认状态目录
- `--dry-run`
  - 只展开队列和命令，不启动 tmux
- `--resume`
  - 从状态目录恢复

---

## 计划 YAML

```yaml
version: 1
label: "keep-position-ids-all-strategies-all-datasets"
pool_size: 6

gpu:
  min_free_gib: 16
  selection: "max_free"
  sample_seconds: 3
  poll_interval_seconds: 15

retry:
  budget_ratio: 0.1
  rounding: "ceil"

tmux:
  session_name: "sched_keep_position_ids"
  log_dir: "entropy_exp/outputs/logs/tmux"

environment:
  conda_sh: "/data/liuyu/anaconda3/etc/profile.d/conda.sh"
  conda_env: "llava"

defaults:
  max_samples: null
  extra_sets:
    - "inference.seed=42"
    - "pruning.layer_selection=fixed"

experiments:
  - name: "gqa_l1_r0.2"
    dataset: "gqa"
    strategies:
      - "attn_score"
      - "pre_attn_score"
      - "masking_attn_score"
      - "entropy"
      - "random"
      - "sparsevlm"
    extra_sets:
      - "pruning.prune_layers=[1]"
      - "pruning.prune_ratio=[0.2]"
```

解释：

- `experiments[*]` 是批次定义
- `strategies` 会展开成多个 job
- 队列顺序严格按 YAML 展开顺序保持 FIFO
- 调度器统一调用 `run_prune.sh --no-auto-gpu`

---

## GPU 规则

- 每轮派发前采样 `nvidia-smi --query-gpu=index,memory.free`
- 连续采样 `sample_seconds`
- 只考虑空闲显存至少 `min_free_gib`
- 默认按 `selection=max_free` 选择平均空闲显存最大的卡
- 调度器会把“每个运行中实验预估占用 16 GiB”计入本轮剩余可用显存估算，避免刚派发完又重复塞满同一张卡

---

## tmux 规则

- 一个计划对应一个 tmux session
- 每个 job attempt 对应一个独立 window
- window 名格式：
  - `<dataset>_<strategy>_l<layer>_r<ratio>_try<attempt>`
- 日志路径格式：
  - `entropy_exp/outputs/logs/tmux/<label>/<job_id>__try<attempt>.log`

---

## 完成与失败判定

一次 attempt 被视为完成，必须同时满足：

- 命令退出码为 `0`
- 唯一发现本次新 run dir
- `run_dir/config.yaml` 存在
- `run_dir/answers.jsonl` 存在
- `answers.jsonl` 行数与该数据集 question_file 一致
- 每行 answers 都能被 JSON 解析

否则视为失败。

失败后：

- 若 `retry_budget_remaining > 0`
  - 扣减 1 次额度
  - job 回到 `pending`
- 否则
  - 进入 `failed_final`

---

## 状态目录

默认状态目录：

```text
entropy_exp/outputs/scheduler/<label>/
```

固定内容：

- `plan.snapshot.yaml`
- `state.json`
- `jobs/<job_id>.json`
- `attempts/<job_id>__try<attempt>.json`
- `launchers/<job_id>__try<attempt>.sh`
- `progress.json`
- `progress.txt`

---

## 进度显示

调度器会在启动、每次状态变化、每轮轮询结束后刷新一行进度：

```text
[progress] 12/324 completed (3.7%) | running=6 pending=304 failed=2 | retry=31/33 | active_gpus=0,2,4,5
```

同样的信息也会写入：

- `progress.json`
- `progress.txt`

这样即使调度器退出，`--resume` 前也能直接查看当前进度。
