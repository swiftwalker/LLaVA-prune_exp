# `entropy_exp` 结果整理与汇总流程

本文档总结一套 **跨策略、跨数据集** 都可复用的结果整理流程，适用于 repo-native `scheduler-first` 大矩阵实验。

它回答的是：

- 一轮矩阵实验跑完后，如何确认结果已经可整理
- 如何从 scheduler 状态中**精确**收集本轮 run
- 如何批量补评测、生成汇总表、再提取 `layer × ratio` 矩阵

不覆盖的内容：

- 如何启动实验、写 scheduler plan、恢复中断任务
- legacy `plan_batch_runs.py` 的完整工作流

这些内容分别参考：

- 调度与 plan：[`SCHEDULER.md`](./SCHEDULER.md)
- 单次 pruning / 配置说明：[`USAGE.md`](./USAGE.md)

---

## 1. 确认矩阵已经结束

对 scheduler 驱动的大矩阵，先看：

```text
entropy_exp/outputs/scheduler/<label>/
  ├── progress.txt
  ├── progress.json
  ├── state.json
  └── attempts/job_*__try*.json
```

最直接的判断方式：

```bash
cat entropy_exp/outputs/scheduler/<label>/progress.txt
```

只有在下面这个状态下，才进入结果整理：

- `completed = total_jobs`
- `running = 0`
- `pending = 0`
- `failed = 0`

如果还想核对 `state.json`：

```bash
python - <<'PY'
import json
from pathlib import Path

label = "<label>"
state_path = Path(f"entropy_exp/outputs/scheduler/{label}/state.json")
state = json.loads(state_path.read_text(encoding="utf-8"))

print({
    "completed": len(state["completed"]),
    "running": len(state["running"]),
    "pending": len(state["pending"]),
    "failed_final": len(state["failed_final"]),
    "retry_budget_remaining": state["retry_budget_remaining"],
})
PY
```

判读规则：

- `failed_final > 0`
  - 先处理 recovery，不要直接汇总
- `running > 0` 或 `pending > 0`
  - 说明矩阵还没结束
- `completed == total_jobs`
  - 可以进入结果整理阶段

---

## 2. 从 `attempts/*.json` 精确收集本轮 run

对大矩阵，**scheduler state 才是结果归属的 source of truth**。

不要直接：

- 用宽泛 prefix 扫 `outputs/runs/`
- 用 `bash entropy_exp/scripts/run_summary.sh runs` 扫全仓库
- 混用不同 matrix / recovery / 手工重跑留下的 run 目录

推荐做法是只读取 `status=completed` 的 `run_dir`：

```bash
python - <<'PY'
import json
from pathlib import Path

label = "<label>"
attempt_dir = Path(f"entropy_exp/outputs/scheduler/{label}/attempts")
run_dirs = []

for attempt_path in sorted(attempt_dir.glob("job_*__try*.json")):
    payload = json.loads(attempt_path.read_text(encoding="utf-8"))
    if payload.get("status") == "completed" and payload.get("run_dir"):
        run_dirs.append(payload["run_dir"])

run_dirs = list(dict.fromkeys(run_dirs))
for run_dir in run_dirs:
    print(run_dir)
PY
```

建议把结果先落成临时清单文件，后续 `eval / summary` 都复用这一份：

```bash
label="<label>"
run_dir_list="/tmp/${label}_run_dirs.txt"

RESULT_LABEL="$label" python - <<'PY' > "$run_dir_list"
import json
import os
from pathlib import Path

label = os.environ["RESULT_LABEL"]
attempt_dir = Path(f"entropy_exp/outputs/scheduler/{label}/attempts")
run_dirs = []

for attempt_path in sorted(attempt_dir.glob("job_*__try*.json")):
    payload = json.loads(attempt_path.read_text(encoding="utf-8"))
    if payload.get("status") == "completed" and payload.get("run_dir"):
        run_dirs.append(payload["run_dir"])

for run_dir in dict.fromkeys(run_dirs):
    print(run_dir)
PY

wc -l "$run_dir_list"
head "$run_dir_list"
tail "$run_dir_list"
```

这里的行数应当和这轮矩阵的完成 job 数一致。

---

## 3. 批量补评测

每个 `run_dir` 必须先拥有：

```text
<run_dir>/eval/summary.json
```

如果还没有，就批量补 `run_eval.sh`：

```bash
while IFS= read -r run_dir; do
  echo "===== EVAL $run_dir"
  bash entropy_exp/scripts/run_eval.sh "$run_dir"
done < "$run_dir_list"
```

评测完成后，可以快速检查是否还有缺口：

```bash
python - <<'PY'
from pathlib import Path

run_dir_list = Path("/tmp/<label>_run_dirs.txt")
missing = []

for line in run_dir_list.read_text(encoding="utf-8").splitlines():
    run_dir = Path(line.strip())
    if line.strip() and not (run_dir / "eval" / "summary.json").is_file():
        missing.append(str(run_dir))

print("missing_eval", len(missing))
for run_dir in missing[:10]:
    print(run_dir)
PY
```

只有当 `missing_eval = 0` 时，才进入汇总。

---

## 4. 生成本轮聚合 summary

主入口是：

```bash
python entropy_exp/src/summarize_results.py \
  --selection-label "<label>" \
  --output-dir "entropy_exp/outputs/summary/<label>" \
  --run-dir $(tr '\n' ' ' < "$run_dir_list")
```

这会生成：

```text
entropy_exp/outputs/summary/<label>/
  ├── summary.csv
  ├── summary.json
  └── skipped_runs.json
```

三个文件的用途：

- `summary.csv`
  - 最适合做表格、矩阵、快速比较
- `summary.json`
  - 保留完整 records，适合脚本二次分析
- `skipped_runs.json`
  - 记录没纳入 summary 的 run 以及原因

补充说明：

- 大矩阵主线推荐直接调 `summarize_results.py`
- `bash entropy_exp/scripts/run_summary.sh <selector> [output_dir]` 更适合：
  - 单 run
  - 经过人工审核的 prefix
  - 小范围手工结果集合
- 对完整 scheduler matrix，不建议 `run_summary.sh runs`

---

## 5. 从 `summary.csv` 提取全量结果矩阵

跨数据集统一入口是：

- `primary_metric_name`
- `primary_metric_value`

数据集主指标映射：

- `gqa -> accuracy`
- `mme -> overall_total_score`
- `pope -> macro_f1`
- `textvqa -> accuracy`
- `scienceqa -> accuracy`

说明：

- `mmbench` 当前不进入 repo-native 本地指标汇总；它只保留 inference 输入兼容能力

把 `summary.csv` 组织成 `layer × ratio` 矩阵的一个通用脚本：

```bash
python - <<'PY'
import ast
import csv
from pathlib import Path

csv_path = Path("entropy_exp/outputs/summary/<label>/summary.csv")
rows = list(csv.DictReader(csv_path.open()))

for dataset in ["gqa", "mme", "pope"]:
    dataset_rows = [row for row in rows if row["dataset"] == dataset]
    ratios = sorted({ast.literal_eval(row["prune_ratio"])[0] for row in dataset_rows})
    layers = sorted({ast.literal_eval(row["effective_prune_layers"])[0] for row in dataset_rows})

    print("===", dataset)
    print("metric:", dataset_rows[0]["primary_metric_name"])
    print("layer," + ",".join(str(ratio) for ratio in ratios))

    for layer in layers:
        values = []
        for ratio in ratios:
            row = next(
                row
                for row in dataset_rows
                if ast.literal_eval(row["effective_prune_layers"])[0] == layer
                and ast.literal_eval(row["prune_ratio"])[0] == ratio
            )
            values.append(row["primary_metric_value"])
        print(str(layer) + "," + ",".join(values))
    print()
PY
```

这一步输出的就是我们日常所说的“全量结果矩阵”。

---

## 6. 二次整理：最优配置、层均值、比率趋势、Top-k

在 `summary.csv` 之上，最常做的二次整理有四类：

### 6.1 找每个数据集的最优配置

```bash
python - <<'PY'
import ast
import csv
from pathlib import Path

rows = list(csv.DictReader(Path("entropy_exp/outputs/summary/<label>/summary.csv").open()))

for dataset in ["gqa", "mme", "pope"]:
    dataset_rows = [row for row in rows if row["dataset"] == dataset]
    best = max(dataset_rows, key=lambda row: float(row["primary_metric_value"]))
    print(
        dataset,
        "layer=", ast.literal_eval(best["effective_prune_layers"])[0],
        "ratio=", ast.literal_eval(best["prune_ratio"])[0],
        "metric=", best["primary_metric_value"],
    )
PY
```

### 6.2 看分层均值

```bash
python - <<'PY'
import csv
from collections import defaultdict
from pathlib import Path

rows = list(csv.DictReader(Path("entropy_exp/outputs/summary/<label>/summary.csv").open()))

for dataset in ["gqa", "mme", "pope"]:
    bucket = defaultdict(list)
    for row in rows:
        if row["dataset"] != dataset:
            continue
        bucket[row["effective_prune_layers"]].append(float(row["primary_metric_value"]))
    print(dataset, {layer: sum(vals) / len(vals) for layer, vals in sorted(bucket.items())})
PY
```

### 6.3 看分剪枝率趋势

```bash
python - <<'PY'
import ast
import csv
from collections import defaultdict
from pathlib import Path

rows = list(csv.DictReader(Path("entropy_exp/outputs/summary/<label>/summary.csv").open()))

for dataset in ["gqa", "mme", "pope"]:
    bucket = defaultdict(list)
    for row in rows:
        if row["dataset"] != dataset:
            continue
        ratio = ast.literal_eval(row["prune_ratio"])[0]
        bucket[ratio].append(float(row["primary_metric_value"]))
    print(dataset, {ratio: sum(vals) / len(vals) for ratio, vals in sorted(bucket.items())})
PY
```

### 6.4 提取 Top-k 结果表

```bash
python - <<'PY'
import csv
from pathlib import Path

rows = list(csv.DictReader(Path("entropy_exp/outputs/summary/<label>/summary.csv").open()))

for dataset in ["gqa", "mme", "pope"]:
    dataset_rows = [row for row in rows if row["dataset"] == dataset]
    topk = sorted(dataset_rows, key=lambda row: float(row["primary_metric_value"]), reverse=True)[:3]
    print("\\nTOP3", dataset)
    for row in topk:
        print(row["run_name"], row["effective_prune_layers"], row["prune_ratio"], row["primary_metric_value"])
PY
```

---

## 7. 常见误区

- `attempts/*.json` 为空，或还有 `running / pending / failed_final`
  - 不要直接汇总，先把矩阵收口或做 recovery
- 直接用 `run_summary.sh runs`
  - 容易把别的策略、别的批次、旧结果一并扫进去
- `run_dir` 里缺 `eval/summary.json`
  - 先补 `run_eval.sh`，再做 summary
- 重跑或 recovery 后仍沿用旧的 `selection_label` / `output_dir`
  - 会把不同轮次结果混在一起，难以追溯
- 只看 `run_name` 前缀，不看 scheduler state
  - 对单 run / 小范围 prefix 可以，但对大矩阵不够稳

---

## 8. Worked Example

下面以这次完成过的一轮矩阵为例：

```text
label = keep-position-ids-sparsevlm-adaptive-stratified-full-matrix
```

在你自己的实验里，把 `<label>` 替换成当前 matrix 的 label 即可。

### 8.1 收集 run 清单

```bash
label="keep-position-ids-sparsevlm-adaptive-stratified-full-matrix"
run_dir_list="/tmp/${label}_run_dirs.txt"

RESULT_LABEL="$label" python - <<'PY' > "$run_dir_list"
import json
import os
from pathlib import Path

label = os.environ["RESULT_LABEL"]
attempt_dir = Path(f"entropy_exp/outputs/scheduler/{label}/attempts")
run_dirs = []

for attempt_path in sorted(attempt_dir.glob("job_*__try*.json")):
    payload = json.loads(attempt_path.read_text(encoding="utf-8"))
    if payload.get("status") == "completed" and payload.get("run_dir"):
        run_dirs.append(payload["run_dir"])

for run_dir in dict.fromkeys(run_dirs):
    print(run_dir)
PY
```

### 8.2 补评测

```bash
while IFS= read -r run_dir; do
  bash entropy_exp/scripts/run_eval.sh "$run_dir"
done < "$run_dir_list"
```

### 8.3 生成总 summary

```bash
python entropy_exp/src/summarize_results.py \
  --selection-label "$label" \
  --output-dir "entropy_exp/outputs/summary/keep_position_ids_sparsevlm_adaptive_stratified_full_matrix" \
  --run-dir $(tr '\n' ' ' < "$run_dir_list")
```

### 8.4 取出矩阵

```bash
python - <<'PY'
import ast
import csv
from pathlib import Path

csv_path = Path("entropy_exp/outputs/summary/keep_position_ids_sparsevlm_adaptive_stratified_full_matrix/summary.csv")
rows = list(csv.DictReader(csv_path.open()))

for dataset in ["gqa", "mme", "pope"]:
    dataset_rows = [row for row in rows if row["dataset"] == dataset]
    ratios = sorted({ast.literal_eval(row["prune_ratio"])[0] for row in dataset_rows})
    layers = sorted({ast.literal_eval(row["effective_prune_layers"])[0] for row in dataset_rows})

    print("===", dataset)
    print("metric:", dataset_rows[0]["primary_metric_name"])
    print("layer," + ",".join(str(ratio) for ratio in ratios))

    for layer in layers:
        values = []
        for ratio in ratios:
            row = next(
                row
                for row in dataset_rows
                if ast.literal_eval(row["effective_prune_layers"])[0] == layer
                and ast.literal_eval(row["prune_ratio"])[0] == ratio
            )
            values.append(row["primary_metric_value"])
        print(str(layer) + "," + ",".join(values))
    print()
PY
```

这就是一套可以复用于任何 `scheduler-first` 大矩阵的标准结果整理流程。
