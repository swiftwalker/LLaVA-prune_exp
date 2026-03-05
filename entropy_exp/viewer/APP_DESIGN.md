# Entropy Viewer — 桌面可视化分析工具

## 一、项目定位

**独立的本地 Windows 桌面应用**，用于分析和可视化服务器端熵实验（entropy_exp）产出的 CSV 指标数据。

| 属性 | 说明 |
|---|---|
| 运行平台 | Windows 本地（有 GUI 能力） |
| 技术栈 | Python 3.10+ / PyQt5 / matplotlib / pandas / numpy |
| 数据来源 | 从服务器下载的 `processed/` 目录（含子目录和 CSV 文件） |
| 与服务器的关系 | **完全解耦**，不依赖 LLaVA / SparseVLMs 任何代码 |

---

## 二、数据格式约定

### 2.1 目录结构

用户从服务器下载 `entropy_exp/outputs/processed/` 目录到本地：

```
processed/                           ← 用户通过「打开目录」选择此目录
├── mme_20260305_140915/             ← 数据源 A（一次捕获实验）
│   ├── per_sample_layer.csv         ← 详细数据：每样本 × 每层的所有指标
│   └── summary.csv                  ← 汇总数据：每层的统计量
├── gqa_20260306_091000/             ← 数据源 B
│   ├── per_sample_layer.csv
│   └── summary.csv
├── pope_20260307_103000/            ← 数据源 C
│   ├── per_sample_layer.csv
│   └── summary.csv
└── merged/                          ← 可选：跨数据集合并
    ├── per_sample_layer.csv
    ├── summary.csv
    └── mme/
        └── summary.csv
```

### 2.2 per_sample_layer.csv 列定义

| 列名 | 类型 | 说明 |
|---|---|---|
| `sample_id` | str | 样本唯一标识（如 `sample_000000_code_reasoning__0020.png`） |
| `question_id` | str | 原始问题 ID |
| `image_file` | str | 图片文件路径 |
| `layer` | int | 层索引（0–31） |
| `prune_scores_mean` | float | 该层 prune scores 均值 |
| `prune_scores_std` | float | 该层 prune scores 标准差 |
| `prune_scores_max` | float | 该层 prune scores 最大值 |
| `prune_scores_min` | float | 该层 prune scores 最小值 |
| `shannon` | float | Shannon 熵 |
| `renyi_2` | float | Rényi 熵（α=2） |
| `gini` | float | Gini 系数 |
| `topk_32` | float | Top-32 集中度 |
| `topk_64` | float | Top-64 集中度 |
| `topk_128` | float | Top-128 集中度 |
| `attn_rank` | int | 注意力矩阵有效秩 |
| `attn_rank_ratio` | float | 秩 / min(L_t, L_v) |
| `attn_sv_top1_ratio` | float | 第一奇异值占比 |
| `source_file` | str | 来源 HDF5 文件名 |
| `dataset` | str | 数据集名（mme / gqa / pope） |
| `shannon_delta` | float | Shannon 熵层间变化 |
| `renyi_2_delta` | float | Rényi 熵层间变化 |
| `gini_delta` | float | Gini 系数层间变化 |
| `attn_rank_delta` | float | 秩的层间变化 |

### 2.3 summary.csv 列定义

每层的聚合统计：`{metric}_{agg}`，其中 agg ∈ {mean, std, median, min, max}。

---

## 三、项目结构

```
entropy_viewer/
├── main.py                        # 入口点
├── requirements.txt               # 依赖声明
│
├── app/
│   ├── __init__.py
│   ├── main_window.py             # 主窗口：布局组装 + 信号连接
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   └── manager.py             # DataManager：目录扫描、CSV 加载、过滤、缓存
│   │
│   ├── views/
│   │   ├── __init__.py
│   │   ├── plot_canvas.py         # MplCanvas 基类（FigureCanvasQTAgg 封装）
│   │   ├── entropy_curve.py       # 熵曲线视图（叠加 + 均值±σ）
│   │   ├── delta_view.py          # 熵变视图（变化曲线 + 关键层分布）
│   │   ├── rank_view.py           # 秩分析视图（rank / ratio / sv_top1）
│   │   ├── variance_view.py       # 方差分析视图（CV + 箱线图）
│   │   └── comparison_view.py     # 跨数据源对比视图
│   │
│   └── panels/
│       ├── __init__.py
│       ├── data_panel.py          # 数据源导航面板
│       ├── metric_panel.py        # 指标 / 图表类型选择面板
│       └── detail_panel.py        # 样本详情面板
│
└── resources/
    └── style.qss                  # Qt 样式表（可选）
```

---

## 四、核心模块说明

### 4.1 DataManager (`app/data/manager.py`)

**职责**：数据层的唯一入口，负责目录发现、CSV 解析、数据过滤和缓存。

```
DataManager
├── scan_directory(root_path: str) → List[DataSource]
│   扫描目录，识别包含 per_sample_layer.csv 的子目录为数据源
│
├── load_source(source_id: str) → DataFrame
│   懒加载：首次访问时读取 CSV 并缓存
│
├── get_samples(source_id) → List[str]
│   返回该数据源的所有 sample_id
│
├── get_layers(source_id) → List[int]
│   返回可用层范围
│
├── filter(source_id, samples=None, layers=None, dataset=None) → DataFrame
│   条件过滤，返回子集
│
├── get_summary(source_id) → DataFrame
│   加载 summary.csv
│
└── compare(source_ids: List[str]) → DataFrame
    合并多个数据源用于对比
```

**DataSource 数据类**：
```python
@dataclass
class DataSource:
    source_id: str        # 目录名，如 "mme_20260305_140915"
    path: str             # 绝对路径
    dataset: str          # 数据集名（从目录名提取）
    timestamp: str        # 时间戳（从目录名提取）
    sample_count: int     # 样本数（快速预览，读取 CSV 行数/32）
    has_summary: bool     # 是否有 summary.csv
```

### 4.2 主窗口 (`app/main_window.py`)

**布局**：

```
┌──────────────────────────────────────────────────────────┐
│  菜单栏：文件(打开目录/最近目录/导出图表) | 视图 | 帮助   │
├──────────┬───────────────────────────────────┬───────────┤
│          │                                   │           │
│  数据源  │                                   │  指标     │
│  导航    │         图表视图区域               │  选择     │
│  面板    │      (QTabWidget 多标签页)         │  面板     │
│          │                                   │           │
│  200px   │         自适应                      │  180px    │
│          │                                   │           │
├──────────┴───────────────────────────────────┴───────────┤
│                    样本详情面板 (可折叠)                    │
│            QTableView 显示选中样本的各层指标               │
└──────────────────────────────────────────────────────────┘
│                     状态栏                                │
└──────────────────────────────────────────────────────────┘
```

**标签页组织**：

| Tab | 视图类 | 功能 |
|---|---|---|
| 📈 熵曲线 | `EntropyCurveView` | 样本叠加图 / 均值±σ 带状图 |
| 📉 熵变分析 | `DeltaView` | 层间变化曲线 + 关键层分布直方图 |
| 🔢 秩分析 | `RankView` | attn_rank / rank_ratio / sv_top1 随层变化 |
| 📊 方差分析 | `VarianceView` | CV 曲线 + 关键层箱线图 |
| ⚖️ 对比 | `ComparisonView` | 多数据源 / 跨数据集叠加对比 |

### 4.3 图表视图 (`app/views/`)

**基类 `MplCanvas`**：
- 封装 `matplotlib.backends.backend_qt5agg.FigureCanvasQTAgg`
- 提供 `NavigationToolbar2QT`（缩放/平移/保存）
- 统一的 `update_plot(df, metric, **kwargs)` 接口
- 鼠标悬停 tooltip：显示 (layer, value, sample_id)
- 点击事件信号：`sample_clicked(sample_id: str)`

**各视图核心逻辑**：

#### EntropyCurveView

```
模式切换：[叠加模式] / [均值±σ 模式]

叠加模式：
  - 每个样本一条半透明曲线（alpha=0.15）
  - 叠加红色均值曲线
  - 超过 300 条时随机采样显示
  - 支持鼠标点击高亮单条曲线

均值±σ 模式：
  - 粗线均值 + 浅色填充 ±1σ 区域
  - 可叠加 ±2σ 外边界
```

#### DeltaView

```
左半：熵变曲线叠加图
  - X: layer, Y: shannon_delta
  - 零线参考 (y=0)
  - 均值曲线叠加

右半：关键层分布直方图
  - 蓝色：每个样本的 min-entropy 层分布
  - 橙色：每个样本的 max-|ΔH| 层分布
  - 可在右侧面板切换到 renyi_2_delta / gini_delta
```

#### RankView

```
上半：attn_rank 随层变化（叠加 + 均值）
下半：attn_rank_ratio 和 attn_sv_top1_ratio 双 Y 轴曲线

交互：
  - 鼠标悬停显示具体 rank 值
  - 点击层索引在 DetailPanel 中过滤该层
```

#### VarianceView

```
左半：CV 曲线（σ/μ 逐层变化）
  - shannon / gini / attn_rank 三条 CV 曲线
  - 高 CV → 图片依赖；低 CV → 模型主导

右半：箱线图
  - 在指定层（0, 4, 8, 12, 16, 20, 24, 28, 31）显示指标分布
  - 可切换指标
```

#### ComparisonView

```
加载多个数据源叠加对比：
  - 每个数据源一种颜色
  - 均值±σ 模式
  - 支持选择对比指标
  - 图例显示数据源名（数据集_时间戳）
```

### 4.4 面板模块 (`app/panels/`)

#### DataPanel（左侧）

```
┌─ 数据源 ──────────┐
│ [📂 打开目录...]   │
│                    │
│ ▼ mme_20260305...  │ ← QTreeWidget
│   ├ 100 samples    │
│   └ dataset: mme   │
│ ▷ gqa_20260306...  │
│ ▷ pope_20260307... │
│                    │
│ ─── 过滤 ─────── │
│ 样本搜索: [_____ ] │ ← QLineEdit（模糊搜索 sample_id）
│ 层范围:  [0] ~ [31]│ ← QSpinBox
│ [✓] 仅有效 delta   │ ← QCheckBox（排除 layer=0 的 NaN delta）
│                    │
│ [应用过滤]          │
└────────────────────┘
```

**信号**：
- `source_selected(source_id: str)` — 切换当前数据源
- `sources_for_compare(source_ids: List[str])` — 勾选多个用于对比
- `filter_changed(samples, layers)` — 过滤条件变更

#### MetricPanel（右侧）

```
┌─ 指标选择 ─────────┐
│                     │
│ 主指标:             │
│ ○ Shannon           │ ← QRadioButton 组
│ ● Rényi-2           │
│ ○ Gini              │
│ ○ Top-K (32/64/128) │
│ ○ attn_rank         │
│ ○ attn_rank_ratio   │
│ ○ attn_sv_top1      │
│                     │
│ ─── 显示选项 ───── │
│ [✓] 显示均值曲线    │
│ [✓] 显示 ±σ 带     │
│ [ ] 显示 ±2σ 带    │
│ 最大叠加数: [200]   │
│                     │
│ ─── 导出 ──────── │
│ [📷 保存当前图表]   │
│ [📊 批量导出 PNG]   │
└─────────────────────┘
```

**信号**：
- `metric_changed(metric: str)` — 指标切换
- `display_options_changed(opts: dict)` — 显示选项变更

#### DetailPanel（底部，可折叠）

```
┌─ 样本详情 ────────────────────────────────────────────────────────┐
│ 当前选中: sample_000042_artwork/0003.png                          │
│                                                                    │
│ Layer │ Shannon │ Rényi-2 │ Gini  │ TopK-64 │ Rank │ Δ Shannon   │
│ ──────┼─────────┼─────────┼───────┼─────────┼──────┼──────────── │
│   0   │  9.062  │  8.934  │ 0.208 │  0.213  │  11  │     —      │
│   1   │  8.756  │  8.277  │ 0.279 │  0.265  │   9  │  -0.306    │
│  ...  │  ...    │  ...    │ ...   │  ...    │ ...  │   ...      │
│  31   │  7.234  │  6.891  │ 0.412 │  0.389  │   6  │  -0.123    │
│                                                                    │
│ [复制到剪贴板]  [导出该样本 CSV]                                    │
└────────────────────────────────────────────────────────────────────┘
```

**触发方式**：在图表中点击某条曲线 → 发出 `sample_clicked` 信号 → DetailPanel 更新。

---

## 五、信号-槽连接全景

```
┌─────────────┐     source_selected      ┌──────────────┐
│  DataPanel   │ ───────────────────────→ │  DataManager  │
│  (左侧)      │     filter_changed       │  (数据层)     │
│              │ ───────────────────────→ │               │
└─────────────┘                           └──────┬───────┘
                                                  │ data_ready(df)
                                                  ▼
┌─────────────┐     metric_changed       ┌──────────────┐
│ MetricPanel  │ ───────────────────────→ │   当前活跃    │
│  (右侧)      │   display_opts_changed   │   View 标签   │
│              │ ───────────────────────→ │  (中央区域)   │
└─────────────┘                           └──────┬───────┘
                                                  │ sample_clicked(id)
                                                  ▼
                                          ┌──────────────┐
                                          │ DetailPanel   │
                                          │  (底部)       │
                                          └──────────────┘

Tab 切换：QTabWidget.currentChanged → 触发当前 View 的 update_plot()
对比模式：DataPanel.sources_for_compare → ComparisonView.update_comparison()
```

---

## 六、典型使用流程

### 流程 1：单数据源分析

```
1. 启动 entropy_viewer
2. 文件 → 打开目录 → 选择本地的 processed/ 文件夹
3. 左侧 DataPanel 显示目录树，列出 mme_20260305_140915 等数据源
4. 点击 mme_20260305_140915 → DataManager 加载 per_sample_layer.csv
5. 中央默认显示「熵曲线」标签页 → 100 条样本曲线叠加 + 红色均值线
6. 右侧切换指标为 attn_rank → 曲线自动更新
7. 切换到「熵变分析」标签 → 查看 ΔH 曲线和关键层分布
8. 在叠加图中点击某条异常曲线 → 底部 DetailPanel 显示该样本各层数值
9. 文件 → 保存当前图表 → 导出 PNG
```

### 流程 2：跨数据集对比

```
1. 打开包含多个子目录的 processed/ 文件夹
2. 左侧 DataPanel 中勾选 mme_xxx 和 gqa_xxx 两个数据源
3. 切换到「对比」标签 → 两组数据以不同颜色的均值±σ 叠加显示
4. 右侧选择指标 attn_rank_ratio → 观察不同数据集的低秩趋势差异
5. 结论：若曲线高度重合 → 模型主导；若显著分离 → 图片主导
```

### 流程 3：样本级深入分析

```
1. 加载数据源后，左侧过滤框输入图片名关键词（如 "artwork"）
2. 过滤后熵曲线只显示匹配样本
3. 在「方差分析」标签观察 CV 曲线 → 判断哪些层的样本间差异最大
4. 在箱线图中选择高 CV 层（如 layer 15）→ 查看该层的分布
5. 点击箱线图中的离群点 → DetailPanel 显示该异常样本
```

---

## 七、快捷键设计

| 快捷键 | 功能 |
|---|---|
| `Ctrl+O` | 打开目录 |
| `Ctrl+S` | 保存当前图表为 PNG |
| `Ctrl+Shift+S` | 批量导出所有图表 |
| `Ctrl+1~5` | 切换标签页 1~5 |
| `Ctrl+F` | 聚焦样本搜索框 |
| `Ctrl+C` | 复制 DetailPanel 选中行 |
| `Escape` | 清除高亮 / 关闭 DetailPanel |

---

## 八、依赖与运行

### requirements.txt

```
PyQt5>=5.15
matplotlib>=3.5
pandas>=1.4
numpy>=1.21
```

### 启动方式

```bash
cd entropy_viewer
pip install -r requirements.txt
python main.py
```

### 打包（可选）

```bash
pip install pyinstaller
pyinstaller --onefile --windowed main.py
```

---

## 九、后续扩展预留

| 功能 | 说明 |
|---|---|
| **HDF5 直接加载** | 可选支持直接加载 raw/*.h5，内置指标计算 |
| **注意力热图** | 加载 HDF5 中的 tv_attn 矩阵，可视化每层的 text→vision 注意力模式 |
| **交互式剪枝模拟** | 基于当前熵/rank 数据，模拟不同剪枝阈值的效果 |
| **报告生成** | 一键导出包含所有图表 + 统计摘要的 HTML/PDF 报告 |
