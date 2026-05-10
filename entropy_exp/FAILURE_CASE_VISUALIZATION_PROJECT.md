# Failure-Case Mining And Pruned-Token Visualization

## 1. 项目目标

这个可视化项目用于分析两种视觉 token 剪枝方法之间的失败差异。我们关心的不是整体平均分本身，而是更细粒度的问题：

> 在同一张图、同一个问题、同一个总保留 token 档位下，为什么方法 A 能答对，而方法 B 答错？

其中方法 A 是参考剪枝方法，主要保留高显著性视觉 token；方法 B 是带空间补偿的自适应剪枝方法，除了保留高显著性 token，还会从空间网格中补充一部分 token 以提高覆盖范围。

项目的核心用途是把“平均指标下降”转化为可检查的样本集合和可视化证据，帮助判断方法 B 的损失来自哪里，例如：

- 是否把关键文字区域剪掉了。
- 是否为了空间覆盖保留了过多低价值区域。
- 是否在早层剪掉的信息导致后续层无法恢复。
- 是否某些任务类型更依赖局部高显著性 token，而不适合强空间补偿。

## 2. 分析对象

分析限定在两个任务类型上：

- TextVQA：需要读取图像中的文字或细粒度局部信息。
- POPE：需要判断图像中目标是否存在，常用于观察视觉幻觉和目标存在性判断。

比较使用四个最终视觉 token 档位：

| 档位 | 含义 |
|---:|---|
| 118 | 中等剪枝，视觉信息保留较多 |
| 60 | 较强剪枝 |
| 28 | 极强剪枝 |
| 20 | 非常极端的低 token 档 |

每个档位都比较同一组剪枝层和同一组逐层剪枝率，从而尽量隔离“剪枝策略本身”的影响，而不是让层位置或计算量差异混入结论。

## 3. 样本判定算法

对于每个样本，记参考方法的输出为 `y_A`，待分析方法的输出为 `y_B`，标准答案或评分函数为 `g`。

每个样本先被划分为四类：

| 类型 | 条件 | 含义 |
|---|---|---|
| both_correct | `g(y_A)=1, g(y_B)=1` | 两者都答对 |
| both_wrong | `g(y_A)=0, g(y_B)=0` | 两者都答错 |
| A_correct_B_wrong | `g(y_A)=1, g(y_B)=0` | 参考方法答对，待分析方法答错 |
| B_correct_A_wrong | `g(y_A)=0, g(y_B)=1` | 待分析方法答对，参考方法答错 |

本项目重点挑选 `A_correct_B_wrong`，因为这些样本最能解释待分析方法相对参考方法的性能缺口。

### TextVQA 评分

TextVQA 使用软分数。给定模型答案 `y` 和 10 个人类标注答案集合 `{a_1, ..., a_10}`，先进行标准答案归一化，然后计算 VQA-style soft accuracy。

在样本筛选阶段：

```text
correct(y) = score(y) > 0
```

也就是说，只要答案命中了任意有效的人类标注形式，就视为正确。同时保留原始 soft score，便于后续区分“完全答错”和“部分命中”。

### POPE 评分

POPE 是 yes/no 判断任务。模型输出先被归一化为二值答案：

```text
if output contains "no" or "not":
    prediction = no
else:
    prediction = yes
```

然后与样本标签比较：

```text
correct(y) = 1[prediction == label]
```

同时保留 POPE 的类别信息，例如 random、popular、adversarial，方便观察失败是否集中在特定干扰类型。

## 4. Gap Ranking 设计

对每个 `dataset × keep_level` 组合，统计以下数量：

```text
N_loss = count(A_correct_B_wrong)
N_gain = count(B_correct_A_wrong)
net_loss = N_loss - N_gain
```

其中：

- `N_loss` 表示待分析方法相对参考方法新增的错误样本数。
- `N_gain` 表示待分析方法修复了参考方法错误的样本数。
- `net_loss` 是净损失，用于决定优先分析哪个任务和 token 档位。

同时记录整体指标差：

```text
metric_delta = metric_B - metric_A
relative_delta = (metric_B - metric_A) / metric_A
```

这样可以避免只看平均指标或只看 disagreement 数量。理想的优先分析目标通常同时满足：

- `metric_delta < 0`
- `net_loss` 较大
- `N_loss` 足够多，能提供稳定样本池

## 5. 重点样本排序

在所有 `A_correct_B_wrong` 样本中，每个任务和档位最多选择 30 个样本进入可视化。排序不是随机的，而是优先选择剪枝行为差异更明显的样本。

对每个剪枝层 `l`，定义：

```text
K_A^l = reference method kept patch set at layer l
K_B^l = analyzed method kept patch set at layer l
L_B^l = low-compensation kept patch set of method B at layer l
S_B^l(i) = saliency score of patch i under method B at layer l
```

计算以下诊断量：

```text
reference_only_ratio_l = |K_A^l \ K_B^l| / |K_A^l|
analyzed_only_ratio_l  = |K_B^l \ K_A^l| / |K_B^l|
low_comp_ratio_l       = |L_B^l| / |K_B^l|
```

再计算参考方法保留、但待分析方法丢弃区域在待分析方法 saliency 中占据的质量：

```text
reference_only_saliency_mass_l =
    sum_{i in K_A^l \ K_B^l} S_B^l(i) / sum_j S_B^l(j)
```

跨层取最大值：

```text
R_ref_only = max_l reference_only_ratio_l
R_low      = max_l low_comp_ratio_l
M_ref_only = max_l reference_only_saliency_mass_l
```

最终样本排序分数为：

```text
patch_diff_score = R_ref_only + 0.5 * M_ref_only + 0.25 * R_low
selection_score = answer_score_gap + patch_diff_score
```

其中 `answer_score_gap = score_A - score_B`。这个排序倾向于优先选择两类样本：

- 参考方法保留了大量待分析方法丢弃的 patch。
- 待分析方法的空间补偿比例高，并且可能牺牲了关键显著区域。

## 6. 可视化设计

每个重点样本生成一张综合图。图中每一行对应一个剪枝层，每一列对应一种观察视角。

### 行：剪枝层

当前分析使用三层渐进剪枝：

```text
layer = 2, 6, 15
```

这样可以观察错误是从早层就出现，还是在中后层累计放大。

### 列：五种视角

| 面板 | 内容 | 解释 |
|---|---|---|
| Reference keep | 参考方法保留的视觉 patch | 显示参考方法认为应保留的信息区域 |
| Analyzed keep | 待分析方法保留的视觉 patch | 显示待分析方法最终留下的信息区域 |
| High/low keep | 待分析方法中高显著性保留与空间补偿保留 | 区分“按 saliency 保留”和“按空间补偿保留” |
| Difference map | 两种方法保留集合差异 | 标出 shared、reference-only、analyzed-only 区域 |
| Saliency heatmap | 待分析方法的显著性热力图 | 检查被剪区域是否其实具有较高 saliency |

### 颜色语义

图中使用固定颜色语义：

- 绿色：某方法保留的 patch。
- 黄色：待分析方法的 high-saliency keep。
- 青色：待分析方法的 low-compensation keep。
- 灰色：两种方法都保留。
- 红色：只有参考方法保留。
- 蓝色：只有待分析方法保留。
- 热力图：待分析方法的 saliency，颜色越亮表示显著性越高。

每张图顶部还包含：

- 数据集和 token 档位。
- 样本 ID。
- 问题文本。
- 标准答案。
- 参考方法答案。
- 待分析方法答案。
- 两者答案分数差。

## 7. 输出内容

项目产出分为四类。

### 1. Gap ranking 表

记录每个任务和 token 档位的整体差异：

| 字段 | 含义 |
|---|---|
| dataset | 数据集 |
| keep | 最终 token 档位 |
| metric_delta | 待分析方法相对参考方法的指标差 |
| A_correct_B_wrong | 参考方法正确、待分析方法错误的样本数 |
| B_correct_A_wrong | 待分析方法正确、参考方法错误的样本数 |
| net_loss | 净损失样本数 |

这张表用于决定最值得优先诊断的任务和档位。

### 2. 全量样本判定表

记录所有样本的答案、分数和四分类结果。它回答：

```text
对于每个样本，两种方法分别答对还是答错？
```

### 3. 重点样本表

记录进入可视化的样本集合。每行包含：

- 样本 ID。
- 问题和标准答案。
- 两种方法的预测答案。
- 两种方法的 correctness score。
- 样本所属的 disagreement 类型。
- patch-level 差异指标。
- 最终 selection score。

这张表用于从定量角度解释为什么某个样本被选中。

### 4. 可视化图片

每个重点样本对应一张图片，展示多层剪枝过程中的保留区域、补偿区域、差异区域和 saliency 分布。

这部分用于人工诊断，例如观察：

- 参考方法是否保留了文字区域，而待分析方法没有。
- 空间补偿是否把 token 分配到了背景或无关区域。
- 失败是否在第 2 层已经出现，还是第 6/15 层才明显分化。
- 高显著性区域是否被低补偿策略稀释。

## 8. 当前运行结果概览

本轮分析覆盖：

```text
2 datasets × 4 token levels = 8 comparison groups
```

共判定：

```text
48,304 sample-level comparisons
```

每组选择 30 个重点失败样本，共：

```text
240 selected cases
240 visualization figures
```

当前 gap ranking 的主要结论是：

| 排名 | 数据集 | token 档位 | 指标差 | net loss |
|---:|---|---:|---:|---:|
| 1 | TextVQA | 28 | -1.836 | 66 |
| 2 | POPE | 28 | -0.851 | 52 |
| 3 | POPE | 60 | -0.607 | 49 |
| 4 | TextVQA | 20 | -1.804 | 45 |
| 5 | TextVQA | 60 | -0.954 | 40 |
| 6 | TextVQA | 118 | -0.384 | 11 |
| 7 | POPE | 118 | +0.005 | 4 |
| 8 | POPE | 20 | +0.465 | -17 |

这说明最值得优先人工检查的是 TextVQA 的 28-token 档，其次是 POPE 的 28-token 和 60-token 档。POPE 的 20-token 档虽然仍存在大量单样本差异，但净损失为负，说明待分析方法在该极端档位修复的样本数多于新增错误样本数。

## 9. 如何解读这套产出

建议按以下顺序阅读：

1. 先看 gap ranking，确定哪个任务和 token 档位的净损失最大。
2. 再看重点样本表，筛选 `selection_score` 高的样本。
3. 打开对应可视化图片，从第 2 层到第 15 层逐行观察差异如何形成。
4. 对同一任务下多个样本归纳共性失败模式。

如果多个失败样本都显示“红色 reference-only 区域集中在文字或目标主体上”，说明待分析方法可能过度牺牲局部关键区域。如果多个失败样本显示“青色 low-compensation 区域集中在背景”，说明空间补偿策略可能需要更强的 saliency 约束。如果失败主要出现在早层，则多层剪枝的第一层剪枝率或补偿策略可能是优先优化对象。

## 10. 方法边界

这套分析是离线诊断工具，不重新运行模型，也不改变剪枝策略。它解释的是“已有实验结果中，两种方法在样本和 patch 层面的差异”，不能单独证明某个改动一定会提升指标。

此外，样本排序分数是诊断启发式，不是新的评测指标。它用于帮助人更快找到高信息量失败样本，而不是替代最终 benchmark 结果。
