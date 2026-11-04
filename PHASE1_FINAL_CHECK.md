# PHASE 1 FINAL CHECK — 最终针对性核查

日期：2026-11-03  
范围：仅核查 4 项（Rank-5 身份级排序、ROC 端点/并列、SGM 逐样本语义、batch size 一致性）  
约束：未重构工程；未修改 CompNet/CCNet 官方代码、权重、数据与现有 split；未重跑全量 600 人实验；原始 `results/phase1_baseline.json` 未覆盖。

---

## 0. 结论摘要

| # | 核查项 | 结论 | 处理 |
|---|---|---|---|
| 1 | Rank-5 身份级排序 | **发现真实缺陷**：原实现直接取 top-5 注册图像 | 已修复为“每个身份取最大相似度，再对身份排序”；新增回归测试 |
| 2 | 完整 ROC 的 AUC/EER 端点与并列 | **未发现问题** | 不改代码；新增构造数据核查，AUC 与 sklearn 完全一致 |
| 3 | SGM 逐样本梯度 | **数学实现正确**；GPU batch<4 存在背包差异 | 增加最小 batch 补齐（不改变真实样本梯度） |
| 4 | batch size 1/4/16 一致性 | 修复前 CCNet batch 16 差异 5.8%；修复后 ≤1.4e-5 | 关闭 TF32 + min_batch=4；全部 PASS |

单元测试：**48/48 OK**（核查前 46 项，新增 2 项）。

---

## 1. 核查 1：Rank-5 是否身份级排序

### 1.1 问题

`evaluation/official_matching.py::rank_accuracy_image_gallery` 原实现：

```python
_, idx = torch.topk(sims, k=kmax, dim=1)       # 直接取 top-k 注册图像
top_labels = gl[idx]
hit = (top_labels[:, :k] == query).any(dim=1)
```

这会直接取 **top-5 张 Gallery 图像**。如果同一身份的多张注册图占据前五，真实身份会被挤掉。

官方代码只计算 Rank-1，因此 `inference.py` 不受影响；但本项目把同一协议扩展到 Rank-5，必须使用**身份级排序**：

```text
identity_score[j] = max_{i: label_i = j} cosine(query, image_i)
rank identities by identity_score
```

### 1.2 构造反例（核查前）

Gallery：identity 0 有 5 张图 `[1,0]`、`[1,0.01]`、`[1,0.02]`、`[1,0.03]`、`[1,0.04]`；identity 1 只有 1 张图 `[0,1]`。查询 `[1,0.05]`，真实身份 1。

- 原实现：top-5 图像全部是 identity 0 → **Rank-5 = 0%**
- 身份级正确结果：identity 0 最大相似度 ≈0.999，identity 1 最大相似度 ≈0.05 → 身份 1 排第 2 → **Rank-5 = 100%**

### 1.3 修复

`evaluation/official_matching.py`：

```python
unique_labels, inverse = torch.unique(gl, sorted=True, return_inverse=True)
id_scores = torch.full((b, num_ids), -inf)
id_scores.scatter_reduce_(1, inverse.view(1, -1).expand(b, -1), sims,
                         reduce="amax", include_self=True)
_, top_idx = torch.topk(id_scores, k=kmin, dim=1)
```

- Rank-1 语义不变（单图最近邻 = 身份最大值的 argmax）。
- Rank-k 现在按身份排序。
- 新增测试 `tests/test_official_matching.py::test_rank5_uses_identity_max_not_top5_images`。

### 1.4 对已有全量结果的影响

Phase 1 Review 中官方协议的 Rank-5 在四个模型/划分组合上**全部是 100%**。身份级 Rank-5 只可能 ≥ 图像级 Rank-5，因此仍为 100%，**无需重跑全量实验**，已有数值不变。Rank-1 不受影响。

---

## 2. 核查 2：完整 ROC 的 AUC/EER 端点与并列

### 2.1 检查方法

逐行审查 `evaluation/verification_metrics.py::_stream_roc`：

- 起点：`prev_fpr=0, prev_tpr=0, prev_thr=+inf`（全拒绝端点）。
- 遍历阈值：`thresholds = unique(genuine ∪ impostor)`，按降序处理。
- 终点：循环结束后追加 `(FPR=1, TPR=1)` 全接受端点，AUC 累加梯形面积。
- EER：只扫描内部点（排除两个端点）；精确 `|FAR−FRR|<=1e-12` 优先，否则在 FAR/FRR 相邻点间线性插值；无交叉时回退最接近点。
- 并列分数：`np.searchsorted(..., side="left")`，等价于 `score >= threshold` 接受，与 `at_threshold` 一致。
- 内存：18M 阈值分块流式处理，不保留完整 FPR/TPR 数组。

### 2.2 构造数据核查结果

`AUC` 与 `sklearn.metrics.roc_auc_score` 对比：

| 场景 | EER | EER 模式 | 本实现 AUC | sklearn AUC | 差异 |
|---|---|---|---|---|---|
| 完美分离 | 0.000000% | exact | 1.00000000 | 1.00000000 | 0 |
| 完全反向 | 100.000000% | exact | 0.00000000 | 0.00000000 | 0 |
| 同分布 | 50.000000% | exact | 0.50000000 | 0.50000000 | 0 |
| 全部并列 | 50.000000% | closest | 0.50000000 | 0.50000000 | 0 |
| 部分并列 | 33.333333% | interpolated | 0.77777778 | 0.77777778 | 2.22e-9 |
| 随机连续 | 2.000000% | exact | 0.99820000 | 0.99820000 | 0 |
| 离散并列 | 51.750000% | interpolated | 0.47080000 | 0.47080000 | 5.55e-17 |

结论：**端点包含正确、并列处理正确、AUC/EER 与独立参考一致**。未修改 `verification_metrics.py`。

---

## 3. 核查 3：SGM 是否逐样本计算

### 3.1 代码路径

`spectral/sgm.py::amplitude_gradient`：

1. 每个样本独立做 `rfft2`，取幅值、相位。
2. 幅值设为可微变量，相位 detach，`irfft2` 重建。
3. 重建图逐样本经过官方预处理（每图 z-score）与冻结模型。
4. `losses = loss_fn(logits, labels)` 为 `[B]` 逐样本损失。
5. `torch.autograd.grad(losses.sum(), amp_var)`。
6. `grad` 形状 `[B, 1, H, W//2+1]`，保留每个样本独立梯度。

`gradient_metrics`：

```python
per_sample_l1 = abs_grad.sum(dim=(1, 2, 3))      # 逐样本
grad_l1_mean = per_sample_l1.mean()              # 再跨样本平均
band_energy = (abs_grad * mask).sum(dim=(1,2,3)) # 逐样本频带能量
band_ratio  = band_energy / total_energy         # 逐样本比例
```

CompNet/CCNet 没有 BatchNorm，预处理也是逐图 z-score；模型中不存在跨样本算子，因此 `sum` 的梯度就是逐样本梯度，不存在 batch 耦合。

### 3.2 CPU 与 GPU 实证

- CPU（无 cuDNN 核选择差异）：batch 1 vs 8/16 的梯度相对 L1 差异约 **1.5e-4 / 1.8e-4**，属于浮点累加顺序差异；逐样本梯度独立。
- GPU：见第 4 节。差异来自 cuDNN 对不同 batch 选择不同卷积核，不是逐样本语义错误。

### 3.3 最小改动（数值稳定性，不是语义修复）

在 `amplitude_gradient` 中增加 `min_batch=4`：当 `0 < B < 4` 时，用第一个样本的副本补齐到 4，计算后只返回原始 B 个样本的梯度。

- 由于无跨样本算子，重复样本不会改变任何真实样本的逐样本梯度。
- 作用：强制小 batch 与大 batch 使用同一条 cuDNN 路径，消除 batch=1 的单图卷积核差异。
- 新增测试 `tests/test_spectral_sgm.py::test_min_batch_padding_preserves_sample_gradient`：同一批真实样本中，样本 0 在 batch=1（补齐）与 batch=4 下的梯度一致。

---

## 4. 核查 4：batch size 1/4/16 数值一致性

### 4.1 实验设置

脚本：`scripts/check_batch_size_invariance.py`  
图像：16 张真实 validation 图（4 个身份 × 4 张）  
模型：CompNet、CCNet  
损失：CE、Margin（scale=10）  
batch：1、4、16  
容差（预设）：梯度相对 L1 ≤ **5e-3**；逐样本 L1/L2/max 相对误差 ≤ **5e-3**；频带比例绝对误差 ≤ **1e-4**。

容差依据：关闭 TF32 后，batch≥4 使用同一 cuDNN 批卷积路径；batch=1 与批路径仍有约 0.12% 的单图核差异，5e-3 覆盖该差异。加入 min_batch 补齐后，实际差异远低于容差。

### 4.2 修复前的发现

| 配置 | batch 1 vs 16 梯度相对 L1 | 频带比例最大绝对差 | 结果 |
|---|---|---|---|
| CompNet CE | 5.87e-4 | 2.57e-5 | 通过旧 1e-3 阈值 |
| CompNet Margin | 5.03e-4 | 2.14e-5 | 通过 |
| CCNet CE | **4.79e-2** | **2.51e-2** | **FAIL** |
| CCNet Margin | **5.79e-2** | **2.90e-2** | **FAIL** |

定位：

- 同一 batch 重复运行结果完全相同 → 不是随机性。
- CPU batch 1/8/16 一致 → 数学上逐样本独立。
- `torch.backends.cudnn.allow_tf32` 默认 **True**。关闭 TF32 后，CCNet batch≥4 之间完全一致，与 batch=1 的差异降到 0.12%（频带 8.7e-4），说明主因是 **TF32 降低精度且不同 batch 选择了不同卷积核**。
- 进一步测试 `min_batch=4` 补齐后，batch 1/4/16 差异降到 ≤1.4e-5。

### 4.3 修复

1. `utils/seed.py::set_deterministic(True)`：
   ```python
   torch.backends.cudnn.allow_tf32 = False
   torch.backends.cuda.matmul.allow_tf32 = False
   torch.set_float32_matmul_precision("highest")
   ```
2. `spectral/sgm.py::amplitude_gradient(..., min_batch=4)`：小 batch 补齐。

### 4.4 修复后的最终结果

| 模型 | 损失 | 对比 | 梯度相对 L1 | 最大绝对差 | 逐样本 L1 相对误差 | 频带比例最大绝对差 | 结果 |
|---|---|---|---|---|---|---|---|
| CompNet | CE | bs1 vs 4 | 0 | 0 | 0 | 0 | PASS |
| CompNet | CE | bs1 vs 16 | 7.22e-7 | 4.77e-7 | 1.65e-6 | 8.94e-8 | PASS |
| CompNet | Margin | bs1 vs 4 | 0 | 0 | 0 | 0 | PASS |
| CompNet | Margin | bs1 vs 16 | 5.80e-7 | 9.24e-7 | 1.04e-7 | 8.94e-8 | PASS |
| CCNet | CE | bs1 vs 4 | 0 | 0 | 0 | 0 | PASS |
| CCNet | CE | bs1 vs 16 | 1.15e-6 | 3.82e-6 | 3.91e-6 | 1.79e-7 | PASS |
| CCNet | Margin | bs1 vs 4 | 0 | 0 | 0 | 0 | PASS |
| CCNet | Margin | bs1 vs 16 | 1.02e-6 | 1.34e-5 | 3.70e-7 | 7.45e-8 | PASS |

**结论：修复后 SGM 结果在 batch 1/4/16 下一致，全部 PASS。**  
结果文件：`results/phase1_review/batch_size_invariance.json`。

> 说明：batch 1 vs 4 完全为 0，是因为 min_batch 补齐后两者都走了相同的 4 样本批路径；这验证了“逐样本、无跨样本耦合”的语义。

---

## 5. SGM 稳定性结果（FP32 修正后）

由于 TF32 对 CCNet 梯度影响明显，使用修正后的配置（TF32 off + min_batch 补齐）重跑了 **小规模** SGM 稳定性实验（100 张 validation 图，未重跑全量）。旧的 TF32-on 结果保留为 `sgm_loss_stability_tf32on.{json,csv}` / `sgm_per_sample_tf32on.csv`。

**CompNet**

| Loss | Scale | L1 均值 | Max | 相对近零(<1%模型最大) | 低/中/高 |
|---|---|---|---|---|---|
| CE | 1 | 26.7248 | 0.2314 | 51.0% | 0.401/0.420/0.179 |
| CE | 10 | 88.7219 | 1.5467 | 35.0% | 0.397/0.424/0.180 |
| CE | 30 | 0.0873 | 0.00836 | 100.0% | 0.380/0.439/0.181 |
| Margin | 1 | 52.7082 | 0.4803 | 0% | 0.392/0.425/0.182 |
| Margin | 10 | 527.082 | 4.8032 | 0% | 0.392/0.425/0.182 |
| Margin | 30 | 1581.25 | 14.4096 | 0% | 0.392/0.425/0.182 |

**CCNet**

| Loss | Scale | L1 均值 | Max | 相对近零(<1%模型最大) | 低/中/高 |
|---|---|---|---|---|---|
| CE | 1 | 41.6239 | 9.1419 | 99.0% | 0.532/0.303/0.165 |
| CE | 10 | 114.329 | 54.7111 | 77.0% | 0.524/0.308/0.168 |
| CE | 30 | 0.1669 | 0.05017 | 100.0% | 0.445/0.360/0.195 |
| Margin | 1 | 93.8647 | 16.8832 | 82.0% | 0.545/0.294/0.161 |
| Margin | 10 | 938.647 | 168.832 | 0% | 0.545/0.294/0.161 |
| Margin | 30 | 2815.94 | 506.495 | 0% | 0.545/0.294/0.161 |

绝对近零（L1<1e-10）仍全部为 0。相对近零以“该模型所有配置最大单样本 L1 的 1%”为阈值，只用于描述相对量级；Margin scale=1 的 82% 是相对于 scale=30 的幅度，不是梯度消失。

**结论不变：**

- CE scale=30 在 clean 样本上饱和，梯度塌缩 3–4 个数量级，100% 样本低于模型最大值的 1%。
- Margin 梯度随 scale 严格线性，频带比例完全稳定。
- Phase 2 主损失仍建议 **Margin (scale=10)**，CE (scale=10) 作对照；不使用 CE scale=30 做 clean SGM。
- TF32 修正后 CCNet 的绝对梯度数值比旧报告更大（例如 Margin scale=30 max 506 vs 153），但趋势与结论一致。

---

## 6. 修改文件

| 文件 | 修改 |
|---|---|
| `evaluation/official_matching.py` | Rank-k 改为身份级最大相似度排序 |
| `tests/test_official_matching.py` | 新增 `test_rank5_uses_identity_max_not_top5_images` |
| `utils/seed.py` | 确定性模式关闭 TF32（cuDNN + matmul）并设为 highest precision |
| `spectral/sgm.py` | `amplitude_gradient` 增加 `min_batch=4` 补齐 |
| `tests/test_spectral_sgm.py` | 新增 `test_min_batch_padding_preserves_sample_gradient` |
| `scripts/check_batch_size_invariance.py` | 新增 batch 一致性核查脚本 |
| `PHASE1_REVIEW_FIXES.md` | 增加 TF32/重跑说明 |

未修改：

- `evaluation/verification_metrics.py`（核查 2 未发现问题）
- `official_project/` 全部官方代码
- `weights/`、`data/`、`splits/tongji_A/`
- CompNet/CCNet 网络结构与权重

---

## 7. 测试结果

```text
Ran 48 tests in 1.779s
OK
```

- 核查前：46 项通过。
- 核查后：**48 项通过**（新增 Rank-5 身份级排序测试、min-batch 补齐测试）。

运行命令：

```bash
D:/code/research/envs/pr/python.exe -m unittest discover -s tests -v
```

---

## 8. 对已有结果的影响

1. **官方 Rank-5**：语义已修正；由于原全量结果均为 100%，数值不变，无需重跑。
2. **clean Rank-1/EER**：未改动 `verification_metrics.py`，数值不变。
3. **SGM 稳定性**：TF32 修正后 CCNet 数值有变化；已重跑小规模实验并保留旧结果。
4. **后续实验**：`set_deterministic(True)` 现在全局关闭 TF32。后续攻击/防御实验将以全 FP32 运行，数值更稳定，但速度可能略慢。

---

## 9. 仍未解决的细节

1. 即便关闭 TF32 并补齐 min_batch，batch 1 与 batch 16 仍有约 1e-6 的残余差异（GEMM/卷积累加顺序），远低于预设容差；不是语义问题。
2. `min_batch=4` 的前提是模型前向不存在跨样本算子。当前 CompNet/CCNet 满足；若未来加入 BatchNorm/跨样本注意力，需要重新评估。
3. Phase 2 应固定实验 batch size（建议 ≥16），并保持 `set_deterministic(True)`，以便数值复现。
4. Phase 2 的 SGM 分析仍需扩展到全部 validation 身份和对抗样本；本次仅核查 batch 一致性与 clean 小样本稳定性。

---

## 10. 复现命令

```bash
PY="D:/code/research/envs/pr/python.exe"

# 核查 4：batch size 1/4/16
$PY scripts/check_batch_size_invariance.py --model all

# 修正后的 SGM 稳定性（100 张 validation 图）
$PY scripts/review_sgm_losses.py --model all --n-ids 20

# 全部单元测试
$PY -m unittest discover -s tests -v
```
