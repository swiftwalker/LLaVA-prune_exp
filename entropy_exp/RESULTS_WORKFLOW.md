# 结果评测与汇总流程

本文是 `entropy_exp` 的 repo-native 结果处理流程，只覆盖：如何确认矩阵结束、补评测、生成 `summary.csv/json`、整理结果矩阵和 top-k。批量调度见 [SCHEDULER.md](./SCHEDULER.md)，单次运行见 [USAGE.md](./USAGE.md)。

## 1. 结果处理顺序

标准顺序：

1. 确认 scheduler 批次已经结束。
2. 从 scheduler `attempts/*.json` 精确收集本轮 completed run dirs。
3. 对每个 run dir 执行 `run_eval.sh`。
4. 用 `summarize_results.py` 生成本轮 summary。
5. 从 `summary.csv` 提取矩阵、top-k 和趋势。

不要直接扫整个 `entropy_exp/outputs/runs/` 做正式对比。这个目录可能混有不同策略、不同日期、不同参数或历史分支结果。

## 2. 数据集和主指标

| 数据集 | Local eval | Summary 主指标 |
| --- | --- | --- |
| `gqa` | 支持 | `accuracy` |
| `mme` | 支持 | `overall_total_score` |
| `pope` | 支持 | `macro_f1` |
| `textvqa` | 支持 | `accuracy` |
| `scienceqa` | 支持 | `accuracy` |
| `mmbench` | 不支持本地 official score | 不进入 repo-native summary 主指标 |

`summarize_results.py` 会把各数据集主指标统一写入：

- `primary_metric_name`
- `primary_metric_value`

## 3. 确认 scheduler 完成

```bash
label="<scheduler-label>"
state_dir="entropy_exp/outputs/scheduler/${label}"

cat "${state_dir}/progress.txt"
python -m json.tool "${state_dir}/state.json" | head -n 80
```

继续汇总前应确认：

- `pending = 0`
- `running = 0`
- `failed_final = 0`
- completed 数量等于 plan 期望 job 数

## 4. 收集本轮 run dirs

从 scheduler attempts 精确收集 completed run：

```bash
label="<scheduler-label>"
state_dir="entropy_exp/outputs/scheduler/${label}"
run_dir_list="/tmp/${label}_run_dirs.txt"

RESULT_STATE_DIR="$state_dir" python - <<'PY' > "$run_dir_list"
import json
import os
from pathlib import Path

attempts = Path(os.environ["RESULT_STATE_DIR"]) / "attempts"
run_dirs = []
for path in sorted(attempts.glob("*.json")):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") == "completed" and payload.get("run_dir"):
        run_dirs.append(payload["run_dir"])

for run_dir in dict.fromkeys(run_dirs):
    print(run_dir)
PY

wc -l "$run_dir_list"
```

## 5. 补评测

```bash
while IFS= read -r run_dir; do
  if [ -f "$run_dir/eval/summary.json" ]; then
    echo "[skip] $run_dir"
  else
    echo "[eval] $run_dir"
    bash entropy_exp/scripts/run_eval.sh "$run_dir"
  fi
done < "$run_dir_list"
```

也可以评测单个 run：

```bash
bash entropy_exp/scripts/run_eval.sh entropy_exp/outputs/runs/sparsevlm_entropy_alpha/gqa/<run_name>
```

评测完成后，每个 run 应有：

```text
<run_dir>/eval/summary.json
```

## 6. 生成 summary

```bash
label="<scheduler-label>"
run_dir_list="/tmp/${label}_run_dirs.txt"
summary_dir="entropy_exp/outputs/summary/${label}"

python entropy_exp/src/summarize_results.py \
  --selection-label "$label" \
  --output-dir "$summary_dir" \
  --run-dir $(tr '\n' ' ' < "$run_dir_list")
```

输出：

| 文件 | 说明 |
| --- | --- |
| `summary.json` | 完整 records 和 skipped runs |
| `summary.csv` | 便于表格处理的汇总 |
| `skipped_runs.json` | 缺文件或解析失败的 run |

验收点：

- `included_run_count` 等于本轮 completed run 数
- `skipped_run_count = 0`
- 每行都有 `primary_metric_value`

## 7. 提取结果矩阵

通用读取方式：

```bash
summary_csv="entropy_exp/outputs/summary/<label>/summary.csv"

python - <<'PY'
import ast
import csv
from pathlib import Path

csv_path = Path("entropy_exp/outputs/summary/<label>/summary.csv")
rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))

for row in sorted(rows, key=lambda r: (r["dataset"], r["strategy"], r["run_name"])):
    print(
        row["dataset"],
        row["strategy"],
        row["effective_prune_layers"],
        row["prune_ratio"],
        row["primary_metric_name"],
        row["primary_metric_value"],
    )
PY
```

如果要按 `dataset x layer x ratio` 组织：

```bash
python - <<'PY'
import ast
import csv
from collections import defaultdict
from pathlib import Path

rows = list(csv.DictReader(Path("entropy_exp/outputs/summary/<label>/summary.csv").open(encoding="utf-8")))
groups = defaultdict(list)

for row in rows:
    layers = ast.literal_eval(row["effective_prune_layers"])
    ratios = ast.literal_eval(row["prune_ratio"])
    layer = layers[0] if layers else None
    ratio = ratios[0] if isinstance(ratios, list) else ratios
    groups[(row["dataset"], row["strategy"], layer, ratio)].append(float(row["primary_metric_value"]))

for key, values in sorted(groups.items()):
    print(*key, sum(values) / len(values))
PY
```

## 8. Top-k 和趋势

每个数据集 top-k：

```bash
python - <<'PY'
import csv
from pathlib import Path

rows = list(csv.DictReader(Path("entropy_exp/outputs/summary/<label>/summary.csv").open(encoding="utf-8")))

for dataset in sorted({row["dataset"] for row in rows}):
    dataset_rows = [row for row in rows if row["dataset"] == dataset]
    topk = sorted(dataset_rows, key=lambda row: float(row["primary_metric_value"]), reverse=True)[:5]
    print("\\nTOP5", dataset)
    for row in topk:
        print(row["strategy"], row["effective_prune_layers"], row["prune_ratio"], row["primary_metric_value"], row["run_name"])
PY
```

跨数据集比较时，先确认：

- 使用相同 `max_samples`
- 使用相同 seed
- 使用相同 `prune_layers` / `prune_ratio` 搜索空间
- 不把 inference-only 的 `mmbench` 混入本地指标均值

## 9. 常见误区

- `attempts/*.json` 还存在 running 或 failed，却直接汇总。
- 直接扫 `outputs/runs`，把旧结果混入新矩阵。
- run dir 还没有 `eval/summary.json` 就运行 summary。
- 重跑 recovery 后继续沿用旧 `selection_label`，导致轮次混在一起。
- 只看 run_name 前缀，不看 scheduler attempt 记录。

历史 keep-position-ids worked example 已移到 [HISTORICAL_WORKFLOWS.md](./HISTORICAL_WORKFLOWS.md)。

