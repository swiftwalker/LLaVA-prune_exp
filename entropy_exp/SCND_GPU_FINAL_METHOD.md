# SCND-GPU 最终方法说明

本文档记录当前定稿方法的稳定口径。历史实验报告、临时 run-dir 清单、scheduler 状态和大矩阵生成物仍保留在 `entropy_exp/outputs/` 下，但不作为主线开发文档维护。

## 1. 最终方法定位

最终定稿方法是：

```text
strategy = sparsevlm_scnd
selection_backend = gpu
prune_layers = [2, 6, 16]
```

方法名可在论文和实验笔记中写作 **SCND-GPU**，即 **Saliency-Constrained Native DivPrune with GPU selection backend**。

它不是 `sparsevlm_fast_scnd` 或 `sparsevlm_budget_candidate_scnd` 的新分支，而是原版 `sparsevlm_scnd` 的正式执行口径：算法语义保持 full SCND，选择过程使用 GPU tensor 后端减少 Python/CPU 控制流。

## 2. 算法思想

SCND-GPU 的核心目标是：在 SparseVLM 的文本条件 saliency 约束下，引入视觉 token 的 native diversity，避免只保留一批相似的高 saliency token。

层模式仍采用 C/B/S：

| 模式 | 含义 | 使用方式 |
| --- | --- | --- |
| `C` | Saliency-constrained native DivPrune | 通常放在第一剪枝层，对全局 alive visual tokens 做 max-min diversity。 |
| `B` | Boundary-only diversity refinement | 后续层只在 saliency cutoff 附近做 diversity tie-break。 |
| `S` | SparseVLM saliency top-k | 后续层回到纯 saliency refinement。 |

C 层流程：

1. 复用 SparseVLM text-rater saliency，得到 visual token saliency。
2. 计算 saliency entropy，用于调整 seed ratio 和 saliency mass floor。
3. 对当前 alive visual tokens 计算 cosine distance matrix。
4. 在 saliency seed pool 内做 max-min selection，得到多样化 seed set。
5. 从 seed set 出发继续按 max-min distance 选择 token，但每一步都要满足 saliency mass feasibility。
6. 若最终 saliency mass 不足，启用 saliency repair，用高 saliency token 替换低价值 token。

B 层流程：

1. 先保留 saliency top `K - b` 的核心 token。
2. 根据 entropy 和 cutoff margin 自动确定 boundary size。
3. 只在 cutoff boundary pool 内做 max-min distance tie-break。
4. entropy 低或 cutoff margin 明确时，自动接近 pure SparseVLM。

## 3. GPU Backend 优化

原版 SCND 的主要瓶颈不是 `576 x 576` 距离矩阵本身，而是选择循环里的 Python list、`.cpu().tolist()` 和逐候选 feasibility / repair。

SCND-GPU 保持算法不变，只改变执行后端：

| 模块 | Python 后端 | GPU 后端 |
| --- | --- | --- |
| C 层主循环 | Python list 枚举 remaining / feasible candidate | GPU mask 和 tensor selection |
| saliency feasibility | 逐候选计算 possible mass | 对所有 remaining token 向量化计算 |
| saliency repair | Python 搜索替换对 | GPU tensor mask 搜索 |
| B 层 distance | 依赖 full distance | boundary pool 局部距离计算 |
| stats 输出 | 循环中频繁同步 | 最终统一 detach/cpu |

配置入口：

```yaml
pruning:
  sparsevlm_scnd:
    selection_backend: "gpu"   # auto | gpu | python
```

默认配置保留 `auto`，正式实验建议显式设为 `gpu`。如果 CUDA 不可用或需要复现旧路径，可设为 `python`。

## 4. 正式实验口径

主线 7B 实验：

| 项 | 配置 |
| --- | --- |
| model | LLaVA-1.5-7B |
| datasets | `gqa,textvqa,pope,mme,scienceqa` |
| strategy | `sparsevlm_scnd` |
| backend | `selection_backend=gpu` |
| prune layers | `[2,6,16]` |
| configs | `C-S-S`, `C-B-S`, `C-B-B` |
| targets | `retain192`, `retain128`, `retain64` |
| runs | `5 * 3 * 3 = 45` |

三档 target：

| target | prune_ratio `[L2,L6,L16]` |
| --- | --- |
| retain192 | `[0.4791667,0.3333333,0.45]` |
| retain128 | `[0.4739583,0.6369637,0.6727273]` |
| retain64 | `[0.8854167,0.5454545,0.4333333]` |

关键结果矩阵：

| dataset | target | metric | C-S-S | C-B-S | C-B-B | best |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| GQA | 192 | accuracy | **61.46** | 61.42 | 61.39 | C-S-S |
| GQA | 128 | accuracy | 60.41 | 60.74 | **60.76** | C-B-B |
| GQA | 64 | accuracy | 56.79 | **56.89** | 56.85 | C-B-S |
| MME | 192 | overall_total_score | 1849.51 | 1844.27 | **1851.63** | C-B-B |
| MME | 128 | overall_total_score | 1842.27 | **1853.89** | 1846.12 | C-B-S |
| MME | 64 | overall_total_score | **1764.56** | 1728.90 | 1739.48 | C-S-S |
| POPE | 192 | macro_f1 | 0.8613 | 0.8620 | **0.8624** | C-B-B |
| POPE | 128 | macro_f1 | 0.8555 | 0.8576 | **0.8577** | C-B-B |
| POPE | 64 | macro_f1 | 0.8012 | **0.8077** | 0.8068 | C-B-S |
| ScienceQA | 192 | accuracy | 69.8184 | **69.9128** | **69.9128** | C-B-S/C-B-B |
| ScienceQA | 128 | accuracy | **69.9599** | 69.9363 | 69.9128 | C-S-S |
| ScienceQA | 64 | accuracy | 69.8184 | **69.8656** | **69.8656** | C-B-S/C-B-B |
| TextVQA | 192 | accuracy | **57.8840** | 57.7160 | 57.8620 | C-S-S |
| TextVQA | 128 | accuracy | **57.6400** | 57.5360 | 57.4760 | C-S-S |
| TextVQA | 64 | accuracy | 54.4560 | 54.8160 | **54.8180** | C-B-B |

结论简述：

- GQA/POPE 是 SCND-GPU 最稳定的收益场景，说明 saliency-constrained native diversity 对全局覆盖和 hallucination detection 有价值。
- MME 对层模式敏感，retain128 偏 `C-B-S`，retain64 偏 `C-S-S`。
- TextVQA 中低压缩更偏 `C-S-S`，说明局部/OCR 任务不宜在后层过度引入 diversity。
- ScienceQA 差异很小，更适合作为不明显退化的 sanity check。

## 5. 方法演进时间线

| 阶段 | 方法 | 结论 |
| --- | --- | --- |
| 1 | entropy-alpha / stratified compensation | 空间补偿能缓解 saliency 过度集中，但硬配额和随机补偿不够稳定。 |
| 2 | score boost / O-S-S | 浅层 boost、深层回到 SparseVLM 是稳健 baseline，成为后续主对照。 |
| 3 | grid-local diverse MMR / adaptive diverse | diversity 有潜力，但 grid-local 或 prune-ratio-only 自适应不足以复现 full native diversity。 |
| 4 | full SCND | saliency-constrained native diversity 在 GQA/POPE/MME 高压缩下优势明显。 |
| 5 | Fast-SCND | 低复杂度版本更均衡，但高压缩上限不如 full SCND。 |
| 6 | Candidate-SCND v1-v3 | 候选池近似能降低成本，但中低压缩下仍难追平 full SCND 的全局 native diversity。 |
| 7 | SCND-GPU | 保留 full SCND 语义，用 GPU backend 提升可运行性，作为最终定稿口径。 |

## 6. 结果和记录索引

### Final evidence

| 内容 | 路径 |
| --- | --- |
| SCND-GPU 45-run summary | `entropy_exp/outputs/summary/scnd_gpu_full_l2_6_16/summary.csv` |
| SCND-GPU report | `entropy_exp/outputs/analysis/scnd_gpu_full_l2_6_16/report.md` |
| SCND-GPU 详细设计报告 | `entropy_exp/outputs/analysis/scnd_gpu_backend_design_report.md` |

### Supporting ablations

| 内容 | 路径 |
| --- | --- |
| Full SCND python/original run family | `entropy_exp/outputs/analysis/scnd_full_l2_6_16/report.md` |
| Fast-SCND | `entropy_exp/outputs/analysis/fast_scnd_full_l2_6_16/report.md` |
| Candidate-SCND v3 | `entropy_exp/outputs/analysis/budget_candidate_scnd_v3_l2_6_16/report.md` |
| O-S-S boost baseline | `entropy_exp/outputs/analysis/ours_boost_sv1_7b_bw0p25_l2_6_16/report.md` |
| SCND / Fast-SCND 设计报告 | `entropy_exp/outputs/analysis/scnd_fast_scnd_design_report.md` |

### Historical / debug artifacts

这些记录保留用于追溯，不再作为主线方法入口：

- entropy-alpha search 和 fullgrid plans：`entropy_exp/plans/entropy_alpha_*`、`entropy_exp/plans/fullgrid_sparsevlm_alpha_020_090_*`
- boost / hybrid sweep：`entropy_exp/outputs/analysis/hybrid_sv2_boost_*`
- diverse/adaptive MMR：`entropy_exp/outputs/analysis/diverse_mmr_*`、`entropy_exp/outputs/analysis/adaptive_diverse_mmr_*`
- Candidate-SCND v1/v2/partial：`entropy_exp/outputs/analysis/budget_candidate_scnd_l2_6_16*`
- remote transfer / run-dir 清单：`entropy_exp/outputs/transfer/*`、`entropy_exp/outputs/*run_dirs*.txt`
- smoke / precheck / missing 文件：仅用于当时调度和排查。

## 7. 代码入口

| 功能 | 路径 |
| --- | --- |
| SCND 策略实现 | `entropy_exp/src/strategies/sparsevlm_scnd.py` |
| 策略注册 | `entropy_exp/src/strategies/__init__.py` |
| 默认配置 | `entropy_exp/configs/prune.yaml` |
| 单次运行入口 | `entropy_exp/scripts/run_prune.sh` |
| scheduler 策略白名单 | `entropy_exp/src/scheduler.py` |
| run 目录识别 | `entropy_exp/src/run_layout.py` |
| SCND 单测 | `entropy_exp/tests/test_sparsevlm_scnd_strategy.py` |

正式复现实验时，优先从 `entropy_exp/plans/scnd_gpu_full_l2_6_16_gpu*.yaml` 或等价 scheduler plan 启动，并确保显式覆盖 `pruning.sparsevlm_scnd.selection_backend=gpu`。
