# Scheduler

本文是 `entropy_exp/scripts/run_scheduler.py` 的权威说明，只覆盖批量实验调度：plan YAML、dry-run、tmux、状态目录、resume 和失败恢复。单次运行见 [USAGE.md](./USAGE.md)，评测汇总见 [RESULTS_WORKFLOW.md](./RESULTS_WORKFLOW.md)。

## 1. 职责边界

Scheduler 负责：

- 读取 plan YAML 并展开 `experiments[*] -> jobs`
- 按全局 `pool_size` 控制并发
- 根据 GPU 可用显存派发 job
- 为每个 attempt 创建 tmux window、日志和结果记录
- 维护 `pending`、`running`、`completed`、`failed_final`
- 失败时按 retry budget 回队
- 从已有状态目录 resume

Scheduler 不负责：

- 评测 `answers.jsonl`
- 汇总 `summary.csv/json`
- 自动比较结果矩阵
- 自定义 run root 发现
- plan YAML 内置 GPU allowlist / denylist；需要时用启动环境 `LLAVA_SCHEDULER_VISIBLE_GPUS` 限制可见 GPU
- 每卡固定并发额度

## 2. 快速命令

```bash
# 新 plan 先 dry-run
python entropy_exp/scripts/run_scheduler.py \
  --plan entropy_exp/plans/demo.yaml \
  --dry-run

# 正式执行
python entropy_exp/scripts/run_scheduler.py \
  --plan entropy_exp/plans/demo.yaml

# 从状态目录恢复
python entropy_exp/scripts/run_scheduler.py \
  --state-dir entropy_exp/outputs/scheduler/<label> \
  --resume

# 只查看 resume 状态，不继续派发
python entropy_exp/scripts/run_scheduler.py \
  --state-dir entropy_exp/outputs/scheduler/<label> \
  --resume \
  --dry-run
```

当前代码默认 `conda_sh` 是 `~/miniconda3/etc/profile.d/conda.sh`，但 plan 可以显式覆盖。不同机器请按本机实际 Conda 路径调整。

## 3. 启动前检查

```bash
git branch --show-current
tmux ls 2>/dev/null || true
ps -ef | rg 'run_prune.sh|prune_inference.py|run_scheduler.py' || true
```

建议确认：

- 当前分支是目标实验分支；本工作区当前主线按 `cleanup/experiment-mainline` 处理
- 没有同一实验线的旧 scheduler 还在跑
- 没有旧进程还在写同一批输出目录

## 4. Plan YAML

最小示例：

```yaml
version: 1
label: "entropy-alpha-smoke"
pool_size: 2

gpu:
  min_free_gib: 16
  selection: "max_free"
  sample_seconds: 3
  poll_interval_seconds: 15

retry:
  budget_ratio: 0.1
  rounding: "ceil"

tmux:
  session_name: "sched_entropy_alpha_smoke"
  log_dir: "entropy_exp/outputs/logs/tmux"

environment:
  conda_sh: "/home/liuyu/miniconda3/etc/profile.d/conda.sh"
  conda_env: "llava"

defaults:
  max_samples: 10
  extra_sets:
    - "inference.seed=42"
    - "pruning.layer_selection=fixed"

experiments:
  - name: "gqa_l2_r0p3"
    dataset: "gqa"
    strategies:
      - "sparsevlm_entropy_alpha"
    extra_sets:
      - "pruning.prune_layers=[2]"
      - "pruning.prune_ratio=[0.3]"
      - "pruning.sparsevlm_entropy_alpha.alpha_min=0.4"
      - "pruning.sparsevlm_entropy_alpha.alpha_max=0.95"
```

### 顶层字段

| 字段 | 说明 |
| --- | --- |
| `version` | 当前必须是 `1` |
| `label` | 状态目录、日志目录和进度显示用的批次名 |
| `pool_size` | 全局并发上限，不是每卡并发上限；按下面的显存容量原则确定 |
| `gpu.min_free_gib` | 单个 job 派发前要求的最小空闲显存；默认按每个 run 占用 `16 GiB` 设置 |
| `gpu.selection` | 当前只支持 `max_free` |
| `gpu.sample_seconds` | 采样 `nvidia-smi` 的秒数 |
| `gpu.poll_interval_seconds` | 没有可派发 GPU 时的轮询间隔 |
| `retry.budget_ratio` | retry 预算占总 jobs 的比例 |
| `retry.rounding` | `ceil` / `floor` / `round` |
| `tmux.session_name` | scheduler 使用的 tmux session |
| `tmux.log_dir` | attempt 日志根目录 |
| `environment.conda_sh` | `conda.sh` 路径 |
| `environment.conda_env` | `conda activate` 的环境名或绝对前缀 |
| `defaults.max_samples` | 默认样本数；`null` 表示全量 |
| `defaults.extra_sets` | 所有 job 共享的 `--set` 覆盖 |
| `experiments` | 非空 experiment 列表 |

### `pool_size` 确定原则

默认把每个 pruning run 视为占用 `16 GiB` 显存，因此 `gpu.min_free_gib` 默认保持 `16`。`pool_size` 应该按当前可见 GPU 的空闲显存尽可能放大，不再使用固定保守值。

推荐计算流程：

1. 用 `nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits` 查看每张可用 GPU 的空闲显存。
2. 若需要保护某些 GPU，先通过 `LLAVA_SCHEDULER_VISIBLE_GPUS=0,1,2` 这类启动环境限制 scheduler 可见 GPU；未限制或设为 `all` 时默认所有 GPU 都可参与调度。
3. 对每张可见 GPU 计算可承载槽位：`slots_i = floor(free_mib_i / (16 * 1024))`。
4. 令 `estimated_slots = sum(slots_i)`，再设置 `pool_size = min(total_jobs, estimated_slots)`。
5. 如果机器是本批实验专用，且希望 scheduler 在别的任务释放显存后继续尽量补满，可以把 `pool_size` 设为 `total_jobs`；实际并发仍会被 `gpu.min_free_gib=16` 的显存门控限制。

示例：

```text
8 张 96 GiB GPU 全空，每张约 97,250 MiB free：
slots_i = floor(97250 / 16384) = 5
estimated_slots = 8 * 5 = 40
pool_size = min(total_jobs, 40)
```

如果某张 GPU 只剩约 `19 GiB` free，按激进最大吞吐原则仍可分配 `1` 个 run；如果想给其他用户或系统留更多余量，就不要把这张卡暴露给 scheduler，或手动降低 `pool_size`。

### `experiments[*]`

| 字段 | 说明 |
| --- | --- |
| `name` | 可读批次名 |
| `dataset` | `gqa` / `mme` / `pope` / `textvqa` / `scienceqa` / `mmbench` |
| `strategies` | 策略列表，按顺序展开；支持范围以 `entropy_exp/src/scheduler.py::SUPPORTED_STRATEGIES` 为准 |
| `extra_sets` | experiment 级别 `--set` 覆盖 |
| `max_samples` | 可选；缺省继承 `defaults.max_samples` |

合并规则：

- `defaults.extra_sets` 和 `experiments[*].extra_sets` 会去重合并。
- 每个 job 最终执行 `bash entropy_exp/scripts/run_prune.sh <strategy> <dataset> ... --no-auto-gpu --set ...`。
- `baseline` 也可以出现在 `strategies` 中。

## 5. 支持范围

数据集：

- Inference：`gqa`、`mme`、`pope`、`textvqa`、`scienceqa`、`mmbench`
- 本地 eval/summary：`gqa`、`mme`、`pope`、`textvqa`、`scienceqa`
- `mmbench`：当前 inference-only

策略：

- `baseline`
- `attn_score`
- `pre_attn_score`
- `masking_attn_score`
- `tail_masking_attn_score`
- `entropy`
- `random`
- `sparsevlm`
- `sparsevlm_adaptive_stratified`
- `sparsevlm_entropy_alpha`

策略方法说明见 [STRATEGY_BRANCH_SUMMARY.md](./STRATEGY_BRANCH_SUMMARY.md)。

## 6. 状态目录和日志

默认状态目录：

```text
entropy_exp/outputs/scheduler/{sanitize(label)}/
```

关键文件：

| 路径 | 说明 |
| --- | --- |
| `plan.snapshot.yaml` | 首次启动时保存的 plan 快照 |
| `state.json` | 当前 pending/running/completed/failed 状态 |
| `progress.txt` | 人类可读进度摘要 |
| `jobs/*.json` | 展开后的 job 定义 |
| `attempts/*.json` | 每次 attempt 的状态、命令、日志、run_dir |
| `launchers/*.sh` | tmux window 中执行的 launcher 脚本 |

日志目录来自 `tmux.log_dir`，通常形如：

```text
entropy_exp/outputs/logs/tmux/{label}/job_...log
```

## 7. 完成与失败判定

一个 attempt 视为成功需要同时满足：

- `run_prune.sh` 退出码为 0
- 能从输出中或 run discovery 中定位到 run directory
- run directory 中存在基础产物，例如 `config.yaml` 和 `answers.jsonl`

常见恢复流程：

```bash
python entropy_exp/scripts/run_scheduler.py \
  --state-dir entropy_exp/outputs/scheduler/<label> \
  --resume \
  --dry-run

python entropy_exp/scripts/run_scheduler.py \
  --state-dir entropy_exp/outputs/scheduler/<label> \
  --resume
```

如果 retry budget 已耗尽，需要先人工判断失败原因，再决定是否新开一轮 plan 或清理状态目录。不要在仍有运行进程时删除 run 目录。

## 8. 新策略接入检查表

新增策略后，至少同步检查：

- `entropy_exp/src/strategies/__init__.py`
- `entropy_exp/configs/prune.yaml`
- `entropy_exp/scripts/run_prune.sh`
- `entropy_exp/src/scheduler.py::SUPPORTED_STRATEGIES`
- `entropy_exp/src/run_layout.py::KNOWN_STRATEGIES`
- `entropy_exp/scripts/run_dir_helper.py --strategy`
- `entropy_exp/tests/test_scheduler.py`
- `entropy_exp/tests/test_run_layout.py`
- 策略自身测试
- [STRATEGY_BRANCH_SUMMARY.md](./STRATEGY_BRANCH_SUMMARY.md)

`sparsevlm_entropy_alpha` 是当前已接入 scheduler、run layout 和 run pruning 入口的参考案例。

## 9. 后续步骤

Scheduler 完成后，通常按这个顺序处理：

1. 用 `progress.txt` 或 `state.json` 确认没有 pending/running/failed。
2. 从 `attempts/*.json` 收集本轮 completed run directories。
3. 按 [RESULTS_WORKFLOW.md](./RESULTS_WORKFLOW.md) 补评测和生成 summary。
4. 如果需要 patch 保留分布，再按 [PATCH_DISTRIBUTION_WORKFLOW.md](./PATCH_DISTRIBUTION_WORKFLOW.md) 处理。

历史 keep-position-ids canonical plan 和旧工作流见 [HISTORICAL_WORKFLOWS.md](./HISTORICAL_WORKFLOWS.md)。
