# Scheduler

`entropy_exp/scripts/run_scheduler.py` 是 `entropy_exp` 的批量实验调度入口。本文档是 scheduler 的权威使用说明，覆盖：

- 日常启动、dry-run、resume
- plan YAML 字段含义
- 运行中如何看进度、日志、状态目录
- 当前实现边界与排查方法
- 新策略如何接入 scheduler 生态

## 1. 职责边界

Scheduler **负责**：

- 读取 plan YAML
- 按 FIFO 展开 `experiments[*] -> jobs`
- 维护 `pending / running / completed / failed_final`
- 根据 GPU 可用显存选择派发目标卡
- 把每个 attempt 挂到 `tmux`
- 为每个 attempt 记录 launcher、log、result 和进度摘要
- 在失败时按全局 retry budget 回队

Scheduler **不负责**：

- `eval`
- `summary`
- 自动比较不同实验结果
- 自定义 run root 发现
- GPU 白名单 / 黑名单控制
- 每卡固定并发额度控制

如果需要结果汇总，请在 run 完成后再单独调用 `run_eval.sh` / `run_summary.sh`。

## 2. CLI 用法

入口脚本：

```bash
python entropy_exp/scripts/run_scheduler.py --plan <plan.yaml> --dry-run
python entropy_exp/scripts/run_scheduler.py --plan <plan.yaml>
python entropy_exp/scripts/run_scheduler.py --state-dir <state_dir> --resume
```

真实支持的参数来自 `build_parser()`：

- `--plan`
  - scheduler plan YAML 路径
- `--pool-size`
  - 临时覆盖 plan 里的 `pool_size`
- `--state-dir`
  - 首次启动时覆盖默认状态目录
  - `--resume` 时指定已有状态目录
- `--dry-run`
  - 与 `--plan` 一起使用时：只展开并打印 jobs，不启动 `tmux`
  - 与 `--resume` 一起使用时：只输出当前进度摘要，不重新展开队列
- `--resume`
  - 从已有状态目录恢复

补充说明：

- 脚本启动时会优先尝试重进程到 `~/miniconda3/envs/llava/bin/python`，但只有该路径实际存在时才会生效。
- 在当前主机上，更稳妥的调用方式仍是先执行 `source /data/liuyu/anaconda3/etc/profile.d/conda.sh && conda activate llava`，不要依赖失效的 preferred-python 路径。
- `--resume` 且未显式传 `--state-dir` 时，如果给了 `--plan`，状态目录会按 `entropy_exp/outputs/scheduler/<sanitize(label)>/` 推导。

## 3. 推荐操作流程

### 3.1 启动前检查

```bash
git branch --show-current
tmux ls 2>/dev/null || true
ps -ef | rg 'run_prune.sh|prune_inference.py|run_scheduler.py' || true
```

建议确认：

- 当前分支是你想跑实验的分支
- 没有旧的 scheduler session 还在跑
- 没有同一实验线的旧进程还在写结果目录

### 3.2 先做 dry-run

```bash
python entropy_exp/scripts/run_scheduler.py --plan entropy_exp/plans/demo.yaml --dry-run
```

dry-run 会打印：

- `Plan label`
- `Repository root`
- `Pool size`
- `Retry budget`
- `Planned jobs`
- 每个 job 的完整 `run_prune.sh` 命令

这是校验 plan 是否写对的第一步。

### 3.3 正式启动

```bash
python entropy_exp/scripts/run_scheduler.py --plan entropy_exp/plans/demo.yaml
```

首次启动会：

- 创建状态目录
- 写入 `plan.snapshot.yaml`
- 初始化 `jobs/`、`attempts/`、`launchers/`
- 创建或复用对应 `tmux session`
- 按 GPU 可用显存逐个派发 job

### 3.4 中断后恢复

```bash
python entropy_exp/scripts/run_scheduler.py \
  --state-dir entropy_exp/outputs/scheduler/<label> \
  --resume
```

恢复逻辑会先：

- 读取 `plan.snapshot.yaml`
- 读取 `state.json`
- reconcile 当前 `running` job
- 对已有 attempt result 做收敛
- 再继续派发剩余队列

如果只想看当前状态，不继续跑：

```bash
python entropy_exp/scripts/run_scheduler.py \
  --state-dir entropy_exp/outputs/scheduler/<label> \
  --resume --dry-run
```

这里的 `--dry-run` 只会刷新并打印 `progress`，不会重新打印全部 jobs。

## 4. Plan YAML 结构

顶层字段固定为：

- `version`
- `label`
- `pool_size`
- `gpu`
- `retry`
- `tmux`
- `environment`
- `defaults`
- `experiments`

最小可运行示例：

```yaml
version: 1
label: "scheduler-demo-single"
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
  session_name: "sched_demo_single"
  log_dir: "entropy_exp/outputs/logs/tmux"

environment:
  conda_sh: "/data/liuyu/anaconda3/etc/profile.d/conda.sh"
  conda_env: "llava"

defaults:
  max_samples: 2
  extra_sets:
    - "inference.seed=42"
    - "pruning.layer_selection=fixed"

experiments:
  - name: "mme_l2_r0.3"
    dataset: "mme"
    strategies:
      - "sparsevlm_adaptive_stratified"
    extra_sets:
      - "pruning.prune_layers=[2]"
      - "pruning.prune_ratio=[0.3]"
```

### 4.1 顶层字段说明

- `version`
  - 当前必须是 `1`
- `label`
  - 非空字符串
  - 用于默认状态目录名、日志目录子目录名和进度显示
- `pool_size`
  - 全局并发上限
  - **不是**每卡并发上限
- `gpu.min_free_gib`
  - 单个 job 派发前要求的最小可用显存门槛
- `gpu.selection`
  - v1 只支持 `"max_free"`
- `gpu.sample_seconds`
  - 连续采样 `nvidia-smi` 的秒数
- `gpu.poll_interval_seconds`
  - 没有可派发 GPU 时，两轮轮询之间的休眠时间
- `retry.budget_ratio`
  - 总 retry 预算占总 jobs 数的比例
- `retry.rounding`
  - `ceil` / `floor` / `round`
- `tmux.session_name`
  - 该 plan 对应的 tmux session
- `tmux.log_dir`
  - 日志根目录；单次 attempt 日志会写到 `<log_dir>/<label>/`
- `environment.conda_sh`
  - `conda.sh` 路径
- `environment.conda_env`
  - `conda activate` 的目标，既可写环境名，也可写绝对前缀路径
- `defaults.max_samples`
  - 默认样本数；`null` 表示全量
- `defaults.extra_sets`
  - 所有 experiment 默认附加的 `--set key=value`
- `experiments`
  - 非空列表，每一项会继续按 `strategies` 展开成多个 job

### 4.2 `experiments[*]` 说明

每个 experiment 包含：

- `name`
  - 批次名，只用于可读性和进度
- `dataset`
  - 必须是 `gqa` / `mme` / `pope`
- `strategies`
  - 策略列表，按给定顺序展开
- `extra_sets`
  - experiment 级别附加 overrides
- `max_samples`
  - 可选；缺省时继承 `defaults.max_samples`

实际合并规则：

- `defaults.extra_sets` 与 `experiments[*].extra_sets` 会去重合并
- job 命令统一形如：

```bash
bash entropy_exp/scripts/run_prune.sh <strategy> <dataset> [max_samples] --no-auto-gpu --set ...
```

### 4.3 比率 sweep 示例

下面示例只演示 plan 写法，不代表本轮待执行计划：

```yaml
version: 1
label: "scheduler-demo-ratio-sweep"
pool_size: 12

gpu:
  min_free_gib: 16
  selection: "max_free"
  sample_seconds: 3
  poll_interval_seconds: 15

retry:
  budget_ratio: 0.1
  rounding: "ceil"

tmux:
  session_name: "sched_demo_ratio_sweep"
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
  - name: "gqa_l2_r0.2"
    dataset: "gqa"
    strategies: ["sparsevlm_adaptive_stratified"]
    extra_sets:
      - "pruning.prune_layers=[2]"
      - "pruning.prune_ratio=[0.2]"

  - name: "gqa_l2_r0.3"
    dataset: "gqa"
    strategies: ["sparsevlm_adaptive_stratified"]
    extra_sets:
      - "pruning.prune_layers=[2]"
      - "pruning.prune_ratio=[0.3]"

  - name: "gqa_l2_r0.4"
    dataset: "gqa"
    strategies: ["sparsevlm_adaptive_stratified"]
    extra_sets:
      - "pruning.prune_layers=[2]"
      - "pruning.prune_ratio=[0.4]"

  - name: "gqa_l2_r0.5"
    dataset: "gqa"
    strategies: ["sparsevlm_adaptive_stratified"]
    extra_sets:
      - "pruning.prune_layers=[2]"
      - "pruning.prune_ratio=[0.5]"

  - name: "gqa_l2_r0.6"
    dataset: "gqa"
    strategies: ["sparsevlm_adaptive_stratified"]
    extra_sets:
      - "pruning.prune_layers=[2]"
      - "pruning.prune_ratio=[0.6]"

  - name: "gqa_l2_r0.7"
    dataset: "gqa"
    strategies: ["sparsevlm_adaptive_stratified"]
    extra_sets:
      - "pruning.prune_layers=[2]"
      - "pruning.prune_ratio=[0.7]"
```

### 4.4 示例中的新策略说明

上面的最小示例和比率 sweep 示例都使用了 `sparsevlm_adaptive_stratified`，因为它是当前仓库里最新接入 scheduler 生态的新策略，适合作为“新策略如何被批量调度”的参考模板。

这个策略的行为可以简化理解为：

- 先沿用 `sparsevlm` 的 text-rater 选择和 text->vision attention 打分
- 再优先保留一部分高分 patch
- 最后按空间 strata 的分布缺口补足剩余 keep budget

它与 scheduler 相关的关键点有三条：

- 对 scheduler 来说，它和其他普通策略一样，只是 `strategies` 列表中的一个字符串，不需要特殊调度分支。
- 它仍走现有的 `run_prune.sh -> prune_inference.py` 路径，因此 scheduler 的 job 展开、run prefix 生成、tmux 启动和完成判定逻辑都不需要为它单独定制。
- 如果只使用默认参数，plan 里只需要写：
  - `strategies: ["sparsevlm_adaptive_stratified"]`
  - `pruning.prune_layers=[...]`
  - `pruning.prune_ratio=[...]`

如果后续要在批量实验里显式覆盖它的策略专属参数，可继续通过 `extra_sets` 传给 plan，例如：

```yaml
extra_sets:
  - "pruning.prune_layers=[2]"
  - "pruning.prune_ratio=[0.3]"
  - "pruning.sparsevlm_adaptive_stratified.high_ratio=0.7"
  - "pruning.sparsevlm_adaptive_stratified.grid_size=6"
  - "pruning.sparsevlm_adaptive_stratified.patch_per_row=24"
  - "pruning.sparsevlm_adaptive_stratified.intra_stratum_mode=random"
```

这些参数的直观含义是：

- `high_ratio`
  - 先按高分直接保留的预算比例
- `grid_size`
  - 空间分层的网格数，实际分成 `grid_size x grid_size` 个 strata
- `patch_per_row`
  - 原始视觉 patch 网格边长，LLaVA-1.5 默认是 `24`
- `intra_stratum_mode`
  - 每个 stratum 内补点的方式，当前支持 `random` 和 `farthest`

如果你只是做标准 sweep，通常不需要在 scheduler plan 里重复写这些默认值；只有在做策略消融时才建议显式覆盖。

### 4.5 `keep-position-ids` 分支的 canonical adaptive 54-run plan

当前分支已经补上一份单策略全矩阵 plan：

```text
entropy_exp/plans/keep_position_ids_sparsevlm_adaptive_stratified_full_matrix.yaml
```

它的固定约束是：

- strategy: `sparsevlm_adaptive_stratified`
- datasets: `gqa` / `mme` / `pope`
- prune layers: `1` / `2` / `3`
- prune ratios: `0.2` / `0.3` / `0.4` / `0.5` / `0.6` / `0.7`
- total jobs: `54`
- pool size: `12`
- retry budget: `ceil(54 * 0.1) = 6`

推荐启动顺序：

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

dry-run 至少应看到：

- `Pool size: 12`
- `Retry budget: 6`
- `Planned jobs: 54`

大矩阵跑完后，建议先从 scheduler `attempts/*.json` 精确收集 `status=completed` 的 `run_dir`，再做 `eval / summary`：

```bash
python - <<'PY'
import json
from pathlib import Path

state_dir = Path("entropy_exp/outputs/scheduler/keep-position-ids-sparsevlm-adaptive-stratified-full-matrix")
run_dirs = []
for attempt_path in sorted((state_dir / "attempts").glob("job_*__try*.json")):
    payload = json.loads(attempt_path.read_text(encoding="utf-8"))
    if payload.get("status") == "completed" and payload.get("run_dir"):
        run_dirs.append(payload["run_dir"])

for run_dir in dict.fromkeys(run_dirs):
    print(run_dir)
PY
```

完整的结果整理与汇总流程（含批量补评测、生成 `summary.csv`、提取 `layer × ratio` 矩阵）统一参考 [`RESULTS_WORKFLOW.md`](./RESULTS_WORKFLOW.md)。

如果存在 `failed_final`，优先使用 `entropy_exp/scripts/build_recovery_plan_from_scheduler_state.py` 生成 recovery plan，而不是手工重拼命令。

## 5. GPU 选择与并发语义

当前 GPU 选择逻辑来自 `collect_average_gpu_free_mib()` 和 `select_gpu_for_dispatch()`：

1. 每轮派发前调用：

```text
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits
```

2. 连续采样 `sample_seconds`
3. 对每张卡取平均 `memory.free`
4. 只保留满足 `projected_free_mib >= min_free_gib * 1024` 的候选卡
5. 从候选卡里选平均空闲显存最大的卡

补充说明：

- `pool_size` 是**全局**最多同时 running 的 job 数
- `pool_size=12` 的真实含义是“整个 scheduler 最多 12 个 running”，**不是**“每张卡 6 个任务”
- v1 只有“显存门槛 + max_free”这一套派发逻辑，没有每卡额度表
- v1 只对“本轮刚刚派发但 `nvidia-smi` 还没反映出来”的新任务做临时显存预留

当前实现**没有**以下能力：

- GPU allowlist / denylist
- 每卡任务上限
- 按 GPU 组做固定轮转

如果后续需要“只使用 GPU 5/6”或“每卡固定 5 个任务”，应作为 scheduler 能力增强单独实现，不应靠文档约定模拟。

## 6. tmux、日志与状态目录

### 6.1 tmux 规则

- 一个 plan 对应一个 tmux session
- session 不存在时会自动创建一个 `__controller` window
- 每个 job attempt 会启动一个独立 window

window 名来自：

```text
<dataset>_<strategy>_l<layers>_r<ratios>_try<attempt>
```

日志路径来自：

```text
entropy_exp/outputs/logs/tmux/<label>/<job_id>__try<attempt>.log
```

### 6.2 状态目录结构

默认状态目录：

```text
entropy_exp/outputs/scheduler/<sanitize(label)>/
```

固定内容：

- `plan.snapshot.yaml`
  - 启动时拍下来的 plan 快照，resume 时以它为准
- `state.json`
  - scheduler 主状态：`pending/running/completed/failed_final`
- `jobs/<job_id>.json`
  - 每个 job 的命令、attempt 历史和最后状态
- `attempts/<job_id>__try<attempt>.json`
  - 每个 attempt 的结果
- `launchers/<job_id>__try<attempt>.sh`
  - 实际被 tmux 执行的 launcher 脚本
- `progress.json`
  - 结构化进度摘要
- `progress.txt`
  - 一行人类可读进度

## 7. 完成与失败判定

一个 attempt 被视为 `completed`，必须同时满足：

- 进程退出码是 `0`
- 只发现一个新的 run dir
- `run_dir/config.yaml` 存在
- `run_dir/answers.jsonl` 存在
- `answers.jsonl` 行数与该数据集 question file 对应
- 每行 answers 都能被 JSON 解析

否则会进入失败分支，例如：

- `failed_process_exit`
- `failed_missing_run_dir`
- `failed_multiple_run_dirs`
- `failed_missing_config`
- `failed_missing_answers`
- `failed_incomplete_answers`

失败后的处理逻辑：

- 若 `retry_budget_remaining > 0`
  - 扣减 1
  - job 回到 `pending`
- 否则进入 `failed_final`

## 8. 重要限制

### 8.1 不支持自定义 run root

`finalize_attempt_result()` 目前固定在：

```text
entropy_exp/outputs/runs/
```

下查找新 run dir。  
因此 **不要** 在 scheduler plan 的 `extra_sets` 里写 `output.base_dir=...`。否则实际 run 可能成功，但 scheduler 仍会因为找不到新 run dir 而判失败。

### 8.2 不支持 GPU 限制与每卡额度

当前 plan YAML 里没有：

- GPU allowlist
- GPU denylist
- per-GPU quota
- “每卡固定 5 个任务”之类配置

### 8.3 `--dry-run` 和 `--resume --dry-run` 不是一回事

- `--plan --dry-run`
  - 打印队列展开结果
- `--state-dir --resume --dry-run`
  - 只输出当前进度，不打印全部 jobs

## 9. 常见误区与排查

### 9.1 `conda_sh` 路径失效

如果 launcher 里 `source <conda_sh>` 失败：

- 优先检查 `environment.conda_sh`
- 本机当前已验证可用的示例值：

```yaml
environment:
  conda_sh: "/data/liuyu/anaconda3/etc/profile.d/conda.sh"
  conda_env: "llava"
```

### 9.2 run 实际成功，但 scheduler 判失败

优先检查：

- plan 是否偷偷传了 `output.base_dir=...`
- `run_prefix` 是否唯一
- `answers.jsonl` 是否完整写完

如果日志显示 prune 已结束，但 attempt 是 `failed_missing_run_dir`，最常见原因就是 run root 不在 `entropy_exp/outputs/runs/`。

### 9.3 dry-run 正常，但正式运行不派发

最常见原因：

- 没有任何 GPU 满足 `min_free_gib`
- 现有运行中的 job 已达到 `pool_size`
- `tmux` session/窗口状态异常

建议先看：

```bash
cat entropy_exp/outputs/scheduler/<label>/progress.txt
tmux ls
ps -ef | rg 'run_prune.sh|prune_inference.py|run_scheduler.py'
nvidia-smi
```

### 9.4 状态目录已存在，首次启动失败

这是正常保护行为。首次启动如果检测到目标状态目录已存在，会报错并要求：

- 用新的 `label`
- 或显式换 `--state-dir`
- 或使用 `--resume`

## 10. 如何接入新策略到 Scheduler

新增策略后，要让它能被 scheduler、run 发现和文档完整支持，至少同步检查这些入口：

### 10.1 代码入口

- `entropy_exp/src/scheduler.py`
  - 更新 `SUPPORTED_STRATEGIES`
- `entropy_exp/scripts/run_prune.sh`
  - 更新策略白名单与帮助示例
- `entropy_exp/src/run_layout.py`
  - 更新 `KNOWN_STRATEGIES`
- `entropy_exp/scripts/run_dir_helper.py`
  - 更新 `--strategy` 的 `choices`

### 10.2 文档入口

- `entropy_exp/USAGE.md`
  - 更新策略列表和示例
- `entropy_exp/SCHEDULER.md`
  - 如有 scheduler 相关限制或示例变化，一并更新
- 其他策略路径说明文档
  - 例如 `TRANSFORMER_BLOCK_STRATEGY_PATHS.md`

### 10.3 测试入口

- `entropy_exp/tests/test_scheduler.py`
  - 验证 plan 接受新策略、run prefix 生成正确
- `entropy_exp/tests/test_run_layout.py`
  - 验证 run 名 fallback 能识别新策略
- 新策略自身测试
  - helper / pruner / strategy 行为测试

### 10.4 反向校验案例

当前已接入的 `sparsevlm_adaptive_stratified` 可以作为参考案例：

- scheduler 接受该策略
- `run_prune.sh` 接受该策略
- run layout 能识别该策略
- 对应测试已覆盖

## 11. 文档验收建议

改完或新增 plan 后，至少做一次：

```bash
python entropy_exp/scripts/run_scheduler.py --plan <plan.yaml> --dry-run
```

核对：

- `Pool size`
- `Retry budget`
- `Planned jobs`
- 每个 job 命令里的 `dataset / strategy / prune_layers / prune_ratio`

如果文档示例和 dry-run 输出对不上，应以代码行为为准回修文档。
