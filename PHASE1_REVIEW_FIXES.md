# PHASE 1 REVIEW FIXES — 审查意见核查与修正

完成日期：2026-11-03  
范围：Phase 1 审查任务 1–6  
约束：未修改 CompNet/CCNet 官方代码、权重、数据集与现有 split；原始 Phase 1 结果 `results/phase1_baseline.json` 未被覆盖。  
所有新结果位于 `results/phase1_review/`。

---

## 0. 结论摘要

| 任务 | 是否发现真实代码问题 | 处理 |
|---|---|---|
| 1. Gallery 匹配协议 | 否，但 Phase 1 的 Rank-1（原型）与 EER（图对图）口径混合 | 新增官方逐图匹配协议，两套协议完整对比 |
| 2. EER 实现 | **是**：阈值只取 genuine 分数，可能漏掉 impostor-only 交叉点 | 改为完整 ROC（genuine ∪ impostor），流式实现；新增 12 项测试 |
| 3. TAR@FAR 阈值来源 | **是**：原 TAR@FAR 由测试集自身 ROC 决定，是 test-oracle | 新增验证集阈值校准 + 测试集最终评价；原指标显式标记为 oracle |
| 4. 数据划分顺序偏差 | 未发现显著特征/精度偏差；发现极小的原始亮度漂移 | 保留协议 A 不变；提出备选划分但未应用 |
| 5. SGM 损失稳定性 | 不适用（新增实验） | CE 在 scale=30 对 clean 样本饱和；Margin 线性且稳定；据此确定 Phase 2 主损失 |
| 6. 测试 | — | **46/46 通过**（修正前 24/24） |

---

## 1. 任务 1：Gallery 匹配协议核实

### 1.1 官方代码的实际实现

阅读了：

- `official_project/compnet/CompNet/inference.py`
- `official_project/CCNet/inference.py`
- 两个仓库的 `getGI.py`、`getEER.py`

官方 `inference.py` 的 Rank-1 流程：

```python
# featDB_train = 全部 session1 注册图像特征（不做身份平均）
for i in range(ntest):
    for j in range(ntrain):
        cosdis = np.dot(feat_test[i], feat_train[j])
        dis = np.arccos(np.clip(cosdis, -1, 1)) / np.pi
    idx = np.argmin(dis)            # 最近的一张注册图
    galleryID = iddb_train[idx]
```

因此官方协议是 **查询图 vs 每一张注册图，取最小距离（等价于最大余弦）**，不是注册特征均值原型。  
官方 `getEER.py` 也使用 **全部 test × train 图像对** 的 genuine/impostor 分数。

### 1.2 Phase 1 原状态

- Rank-1/5：使用 `prototype = normalize(mean(normalize(f_i)))`。
- EER：使用 probe × gallery **图像**对（与官方一致）。
- 即 Rank 与 EER 分别来自两套不同协议。这不是模型或数据错误，但口径不统一，报告需要明确。

### 1.3 修正

- **保留**当前 prototype 协议（规范要求，且作为主协议）。
- 新增 `evaluation/official_matching.py`：
  - `rank_accuracy_image_gallery`：官方逐图最近邻 Rank-k。
  - `nearest_gallery_identity`。
- 新增脚本 `scripts/review_gallery_eer.py`，对两种协议分别输出 Rank-1/5、EER、AUC。
- 没有为了对齐结果修改任何模型、特征或预处理。

### 1.4 全量 600 人对比结果

数据：gallery = session1 全部 6000 张；validation = session2 前 5 张/人；test = session2 后 5 张/人。

| 模型 | 划分 | 协议 | Rank-1 | Rank-5 | EER（新） | AUC |
|---|---|---|---|---|---|---|
| CompNet | validation | prototype | 100.000% | 100.000% | 0.002282% | 0.99999997 |
| CompNet | validation | official | 100.000% | 100.000% | 0.013790% | 0.99999991 |
| CompNet | test | prototype | 100.000% | 100.000% | 0.016583% | 0.99999993 |
| CompNet | test | official | 100.000% | 100.000% | 0.023333% | 0.99999985 |
| CCNet | validation | prototype | 100.000% | 100.000% | 0.000000% | 1.00000000 |
| CCNet | validation | official | 100.000% | 100.000% | 0.000006% | 1.00000000 |
| CCNet | test | prototype | 100.000% | 100.000% | 0.000000% | 1.00000000 |
| CCNet | test | official | 100.000% | 100.000% | 0.000006% | 1.00000000 |

**差异说明：**

- 在 clean 数据上 Rank-1/5 两种协议都是 100%，无法区分；差异主要体现在 EER。
- 原型平均降低了同一身份的 genuine 分数方差，因此 EER 更低；官方逐图协议保留单张注册图的困难样本，EER 更高。
- 两种协议不可混用。后续实验以 **prototype 为主协议**（规范要求），**official 为对照协议**；攻击/防御实验必须固定使用其中一种并在报告中注明。

结果文件：
- `results/phase1_review/gallery_eer_review_full.json`
- `results/phase1_review/gallery_eer_review_full.csv`

---

## 2. 任务 2：EER 实现审查

### 2.1 原实现与真实问题

原 `evaluation/verification_metrics.py`：

```python
thresholds = np.unique(np.sort(genuine))      # 只取 genuine 分数
tpr = 1 - searchsorted(genuine, thresholds)/n
fpr = 1 - searchsorted(impostor, thresholds)/m
idx = argmin(|tpr - (1-fpr)|)
eer = (fpr[idx] + (1-tpr[idx])) / 2
```

问题：**ROC 的候选阈值只来自 genuine 分数**。当 FAR/FRR 的交叉点落在某个仅出现在 impostor 分数集合中的阈值上时，会漏掉该点。

人工反例：

- genuine = `[0.6, 0.7, 0.8]`
- impostor = `[0.5, 0.75, 0.9]`
- 真实交叉点在 `t = 0.75`（只存在于 impostor 分数中），EER = **66.6667%**
- 旧实现只能取 0.8/0.7/0.6，报告约 **50%**

该反例已写入 `tests/test_verification_metrics.py::test_impostor_only_threshold_is_required`。

### 2.2 修正实现

`evaluation/verification_metrics.py` 重写：

1. 阈值集合 = `genuine ∪ impostor` 的全部不同分数（完整 ROC）。
2. 相同分数按 `score >= threshold` 接受，与 `at_threshold`、官方 `getEER.py` 的评分方向一致。
3. 交叉点：
   - 优先取内部 `|FAR − FRR| <= 1e-12` 的精确点；
   - 否则在 FAR/FRR 相邻点之间线性插值；
   - 若完全没有交叉，回退到最接近点并标记 `"closest"`。
4. 同时输出 `eer_discrete_pct` / `eer_discrete_threshold`，对应官方 `getEER.py` 的 `eer_1/2`。
5. 保留 AUC（trapezoid）。
6. **流式 ROC**：完整阈值集合最多 1800 万项，不保留完整 FPR/TPR 数组，按 100 万点分块遍历并即时更新 EER/AUC/TAR。这样既满足“完整 ROC”，又避免 18M 长数组导致的内存失败（修正前全量 CCNet 曾因此失败）。

新增/扩充测试（`tests/test_verification_metrics.py`）：

- 完美分离 → EER 0
- 同分布 → EER 50%
- impostor-only 阈值反例 → 66.6667%
- 相同分数/ties 的处理
- ROC 阈值集合包含伪造/真实两侧的所有分数
- 20 组随机分布与独立暴力枚举参考实现逐项对比
- 验证集阈值校准、测试集应用、FAR=0/1 边界

### 2.3 EER 修改前后（全量 600 人）

**官方图对图协议**（与 Phase 1 baseline 的 EER 口径一致）：

| 模型 | 划分 | 旧实现 EER | 新实现 EER | Phase 1 原报告 |
|---|---|---|---|---|
| CompNet | validation | 0.013561% | **0.013790%** | 0.0136% |
| CompNet | test | 0.022334% | **0.023333%** | 0.0223% |
| CCNet | validation | 0.000003% | **0.000006%** | 0.0000% |
| CCNet | test | 0.000003% | **0.000006%** | 0.0000% |

**Prototype 协议**（Phase 1 未单独报告）：

| 模型 | 划分 | 旧实现 EER | 新实现 EER |
|---|---|---|---|
| CompNet | validation | 0.001141% | **0.002282%** |
| CompNet | test | 0.008292% | **0.016583%** |
| CCNet | validation | 0.000000% | **0.000000%** |
| CCNet | test | 0.000000% | **0.000000%** |

结论：在该数据集上 clean 分数高度可分，修正带来的数值变化很小；但实现层面上旧版本确实会漏阈值点，反例可差 16.7 个百分点。新实现满足完整 ROC 要求。

---

## 3. 任务 3：TAR@FAR 的阈值来源

### 3.1 原状态与问题

原 `verification_metrics()` 中的 `tar_at_far_1pct` / `0.1pct` 是：

> 在**同一组测试分数**的 ROC 上，找 FAR ≤ target 时能达到的最大 TAR。

这等价于用测试集自身选择阈值，属于 **test-oracle**，不能用于最终报告中的“验证集校准”声明。它是真实代码问题（口径问题），不是数值 bug。

### 3.2 修正

新增：

- `threshold_at_far(genuine_val, impostor_val, target_far)`：
  - 在验证集分数上选择满足 `FAR <= target` 的**最小阈值**（从而最大化 TAR）；
  - 相同 impostor 分数按 `>= threshold` 整体接受/拒绝；
  - 若没有阈值能满足（例如最高分并列过多），阈值取 `nextafter(max_impostor, +inf)`，并报告实际 FAR=0。
- `evaluate_at_threshold(genuine_test, impostor_test, threshold)`：
  - 将验证集阈值原样应用到测试集，报告测试集实际 FAR/TAR/FRR/TNR。
- `verification_metrics()` 中的 oracle 字段增加 `*_is_test_oracle: True`，防止误用。

### 3.3 验证集校准 → 测试集评价（全量）

| 模型 | 协议 | 目标 FAR | 验证集阈值 | 验证集 FAR | 测试集 FAR | 测试集 TAR | 测试 EER | 测试 oracle TAR@FAR |
|---|---|---|---|---|---|---|---|---|
| CompNet | prototype | 1% | 0.170839 | 0.010000 | **0.009956** | 1.000000 | 0.016583% | 100% |
| CompNet | prototype | 0.1% | 0.226600 | 0.001000 | **0.000984** | 1.000000 | 0.016583% | 100% |
| CompNet | official | 1% | 0.172152 | 0.010000 | **0.009935** | 1.000000 | 0.023333% | 100% |
| CompNet | official | 0.1% | 0.228519 | 0.001000 | **0.000987** | 1.000000 | 0.023333% | 100% |
| CCNet | prototype | 1% | 0.114067 | 0.010000 | **0.010046** | 1.000000 | 0.000000% | 100% |
| CCNet | prototype | 0.1% | 0.158184 | 0.001000 | **0.001051** | 1.000000 | 0.000000% | 100% |
| CCNet | official | 1% | 0.114922 | 0.010000 | **0.010017** | 1.000000 | 0.000006% | 100% |
| CCNet | official | 0.1% | 0.158986 | 0.001000 | **0.001046** | 1.000000 | 0.000006% | 100% |

关键点：

- 测试集只执行评价，阈值完全来自验证集。
- CCNet 的测试集实际 FAR 略高于目标（1.0046% vs 1%；0.1051% vs 0.1%），说明验证/测试分布存在轻微差异；这是真实结果，不做测试集回调。
- 本数据 clean TAR 在 1% 和 0.1% FAR 下都是 100%，阈值校准主要验证 FAR 控制；攻击场景下 TAR 才会下降。

结果文件：`results/phase1_review/gallery_eer_review_full.json` 中的 `calibration` 字段。

---

## 4. 任务 4：数据划分与采集顺序

### 4.1 可验证的信息

- BMP 文件没有 EXIF，`PIL.Image.info` 只有 `compression` 与 `dpi`。
- 文件 `mtime` 均为 2016-05-29，且按图像批量分布，不能作为逐张采集时间。
- 因此**无法独立验证** session2 内 10 张图的真实采集顺序；文件名/排序是唯一可用顺序。官方 `genText_tongji.py` 也是排序文件名生成列表。

### 4.2 顺序偏差分析（全量 session2，6000 张）

脚本：`scripts/analyze_split_order.py`  
结果：`results/phase1_review/split_order_analysis.{json,csv}`、`split_order_drift.png`

| 指标 | CompNet | CCNet |
|---|---|---|
| 相似度 vs 序号 Spearman | rho = 0.0126, p = 0.331 | rho = 0.0079, p = 0.540 |
| Val vs Test 相似度 KS | 0.0240, p = 0.353 | 0.0200, p = 0.586 |
| 按身份配对的 val−test 均值差 | -0.00133, 95% CI [-0.00412, 0.00146], p = 0.349 | +0.000087, CI [-0.00253, 0.00271], p = 0.948 |
| Wilcoxon（配对差） | p = 0.0599（效应量极小） | p = 0.255 |
| 每个 index 的 Rank-1 | 全部 100% | 全部 100% |

原始像素亮度：

- 平均亮度对 index 的 Spearman：**rho = -0.0354, p = 0.0061**。
- 均值范围：index 0 为 0.53035，index 9 为 0.51993；存在极轻微的整体变暗趋势。
- 该亮度漂移没有体现在特征相似度、Rank-1 或 val/test 分布上（见上表）。

### 4.3 结论

- 暂未发现会显著影响实验结论的采集顺序偏差。
- 存在一个很小的原始亮度漂移（统计显著但效应很小）；若审稿要求彻底排除顺序因素，可考虑：
  1. **交错划分**：val = index {0,2,4,6,8}，test = {1,3,5,7,9}；
  2. **固定种子随机 5/5 划分**（按身份内随机）；
  3. **session 划分**：gallery = session1 前 5 张，val = session1 后 5 张，test = session2 全部。
- **本次未修改 split**，协议 A 保持不变；上述方案仅在用户确认后才可应用。任何修改都会使现有所有结果失效，需要重跑。

---

## 5. 任务 5：SGM 损失稳定性（CE vs Margin）

> **2026-11-03 final-check note.** 下表的 CCNet 数值是在 TF32 默认开启、batch=32 下得到的。最终核查（`PHASE1_FINAL_CHECK.md`）发现 TF32 会造成 CCNet 梯度约 5.8% 的 batch 依赖，已在确定性配置中关闭 TF32，并为小 batch 增加最小 batch 补齐。修正后的 FP32 结果在新的 `sgm_loss_stability.{json,csv}` 中；旧结果保留为 `sgm_loss_stability_tf32on.*`。定性结论不变（CE scale=30 饱和、Margin 线性稳定）。

### 5.1 实现

新增：

- `spectral/frequency.py`：`rho = sqrt(fy² + fx²)/sqrt(0.5²+0.5²)`，`rho∈[0,1]`；低/中/高掩码互斥且覆盖全谱。
- `spectral/losses.py`：
  - CE：`F.cross_entropy(logits, labels, reduction="none")`
  - Margin：`max_{j≠y} s_j − s_y`
- `spectral/sgm.py`：
  - 固定相位，把幅值设为可微变量，重建后送入“官方预处理 + 冻结模型 + Gallery 原型”。
  - 梯度对幅值求导，保留完整计算图；预处理（每图 z-score）在图中。
  - `gradient_metrics` 输出 L1/L2/max、绝对与相对近零比例、频带能量占比。
- `scripts/review_sgm_losses.py`。

### 5.2 实验设置

- 模型：CompNet、CCNet（权重冻结，官方代码未改）。
- 图像：validation 前 20 个身份 × 每身份 5 张 = **100 张真实图像**。
- Gallery：完整 600 人 session1 注册集，prototype 分类器。
- 损失：CE、Margin；logit scale：1、10、30。
- 同一批图像、同一 FFT、同一模型；相位固定。
- 幅值重建误差：CompNet / CCNet 均为 **5.36e-7**。

### 5.3 结果

**CompNet**

| Loss | Scale | 平均 loss | L1 均值 | L1 总和 | L2 均值 | Max | 绝对近零 | 相对近零(<1%模型最大) | 低/中/高 |
|---|---|---|---|---|---|---|---|---|---|
| CE | 1 | 5.61225 | 26.726 | 2672.6 | 0.7551 | 0.2310 | 0 | 51.0% | 0.401/0.420/0.179 |
| CE | 10 | 0.359055 | 88.731 | 8873.1 | 2.4913 | 1.5460 | 0 | 35.0% | 0.397/0.424/0.180 |
| CE | 30 | 5.83e-5 | **0.0873** | 8.73 | 0.00243 | 0.00836 | 0 | **100.0%** | 0.380/0.439/0.181 |
| Margin | 1 | -0.5745 | 52.709 | 5270.9 | 1.4757 | 0.4803 | 0 | 0% | 0.392/0.425/0.182 |
| Margin | 10 | -5.7451 | 527.09 | 52709 | 14.757 | 4.8037 | 0 | 0% | 0.392/0.425/0.182 |
| Margin | 30 | -17.235 | 1581.27 | 158127 | 44.272 | 14.410 | 0 | 0% | 0.392/0.425/0.182 |

**CCNet**

| Loss | Scale | 平均 loss | L1 均值 | L1 总和 | L2 均值 | Max | 绝对近零 | 相对近零(<1%模型最大) | 低/中/高 |
|---|---|---|---|---|---|---|---|---|---|
| CE | 1 | 5.56526 | 39.728 | 3972.8 | 1.6060 | 2.8338 | 0 | 88.0% | 0.514/0.315/0.171 |
| CE | 10 | 0.257188 | 104.822 | 10482 | 3.9483 | 16.377 | 0 | 64.0% | 0.487/0.333/0.181 |
| CE | 30 | 6.62e-5 | **0.1659** | 16.59 | 0.00509 | 0.0502 | 0 | **100.0%** | 0.443/0.362/0.196 |
| Margin | 1 | -0.6865 | 91.930 | 9193.0 | 3.8663 | 5.0963 | 0 | 29.0% | 0.536/0.299/0.164 |
| Margin | 10 | -6.8654 | 919.28 | 91928 | 38.662 | 50.963 | 0 | 0% | 0.536/0.299/0.164 |
| Margin | 30 | -20.596 | 2757.84 | 275784 | 115.986 | 152.89 | 0 | 0% | 0.536/0.299/0.164 |

说明：

- **绝对近零**：`L1 < 1e-10` 的样本比例；全部配置均为 0。
- **相对近零**：`L1 < 1% × 该模型所有配置中的最大单样本 L1`；用于衡量相对饱和，不是“梯度精确为零”。
- CE 平均 loss 在 scale=30 接近 0（CompNet 5.8e-5，CCNet 6.6e-5），因为 clean 样本全部正确且 `cos≈0.99`，softmax 饱和。
- Margin 的 L1 随 scale 严格线性（×10、×30），三种 scale 的频带比例完全一致；CE 的频带比例随 scale 有轻微变化（CCNet 低頻 0.514→0.443）。
- 频带分布：两种模型都以低/中频为主，高频约占 16–20%；CompNet 约 39/43/18，CCNet 约 54/30/16（Margin）。**不能预先假设低频一定更适合识别**，这只是 clean 梯度分布。

### 5.4 对 Phase 2 的决策

基于实际结果（不是预设）：

1. **Phase 2 SGM 主损失：Margin，scale=10。**
   - 理由：在 1/10/30 三个尺度上梯度线性、频带比例完全稳定；clean 样本无饱和。
   - scale 只改变整体幅度，不改变频带比例；选 10 是为了保持与 CE-10 可比的量级，避免 scale=30 的极端最大值。
2. **CE 作为对照：scale=10。**
   - CE-10 的梯度量级与 Margin-1 接近（CompNet 88.7 vs 52.7；CCNet 104.8 vs 91.9），且未饱和。
3. **不将 CE scale=30 用于 clean 样本 SGM**：100% 样本的梯度低于模型最大值的 1%，3–4 个数量级的塌缩，会污染频谱结论。
4. 若 Phase 2 要分析对抗样本，必须重新验证 CE 是否仍然饱和（攻击后样本可能被误分类，CE 梯度可能恢复）；不能直接把 clean 结论外推。

结果文件：
- `results/phase1_review/sgm_loss_stability.json`
- `results/phase1_review/sgm_loss_stability.csv`
- `results/phase1_review/sgm_per_sample.csv`

---

## 6. 修改/新增文件

### 6.1 修改

| 文件 | 修改内容 |
|---|---|
| `evaluation/verification_metrics.py` | 完整 ROC、流式 EER、验证集阈值校准、oracle 标记 |
| `evaluation/__init__.py` | 导出新函数 |

### 6.2 新增

| 文件 | 作用 |
|---|---|
| `evaluation/official_matching.py` | 官方逐图 Gallery Rank-k |
| `spectral/__init__.py` | 频谱模块导出 |
| `spectral/frequency.py` | 径向频率与频带掩码 |
| `spectral/losses.py` | CE / Margin 逐样本损失 |
| `spectral/sgm.py` | 幅值梯度与梯度统计 |
| `scripts/review_gallery_eer.py` | Gallery 双协议 + EER + 阈值校准 |
| `scripts/analyze_split_order.py` | 划分顺序偏差分析 |
| `scripts/review_sgm_losses.py` | CE/Margin SGM 稳定性实验 |
| `tests/test_verification_metrics.py` | EER/阈值 12 项测试 |
| `tests/test_official_matching.py` | 官方协议 3 项测试 |
| `tests/test_spectral_sgm.py` | 频谱/SGM 7 项测试 |

### 6.3 未修改

- `official_project/` 下所有官方代码
- `weights/CompNet/compnet.pth`、`weights/CCNet/ccnet.pth`
- `data/TongjI/ROI/`
- `splits/tongji_A/`（三个 CSV 的 SHA256 与 `split_meta.json` 一致）
- `results/phase1_baseline.json`（原始结果保留）

---

## 7. 测试结果（修改前后）

| 阶段 | 测试数 | 结果 | 耗时 |
|---|---|---|---|
| 修正前（Phase 1） | 24 | OK | 1.75 s |
| 修正后 | **46** | **OK** | **1.83 s** |

新增 22 项：

- EER：完美分离、同分布、impostor-only 阈值、ties、完整阈值集合、20 组随机分布 vs 暴力参考、oracle 标记。
- 阈值校准：目标 FAR、FAR=0、FAR=1、应用到 held-out 测试、ties。
- 官方匹配：最近邻、top-k、预测身份。
- 频谱/SGM：频率范围、频带互斥覆盖、幅值重建、margin/CE 数值、dummy 模型幅值梯度、低頻能量比例。

运行命令：

```bash
D:/code/research/envs/pr/python.exe -m unittest discover -s tests -v
```

实际输出：

```text
Ran 46 tests in 1.831s
OK
```

---

## 8. 仍未解决的问题

1. **协议 A 的采集顺序无法独立验证**。BMP 无采集时间；只能依赖文件名顺序。特征/精度层面无显著偏差，但原始亮度有极小漂移（rho=-0.035, p=0.006）。备选划分已提出，未应用。
2. **CCNet 测试集 FAR 略超目标**（1.0046% vs 1%；0.1051% vs 0.1%），说明验证/测试分布有轻微差异。最终检测阈值（Phase 5）需要重新在验证集上校准，并明确报告测试集实际 FAR；不得用测试集回调。
3. **两种 Gallery 协议在 clean 下 Rank-1 都是 100%**，无法据此选择主协议。EER 差异存在（prototype 更低）。Phase 3–5 必须固定一种协议；本报告建议以 prototype 为主、official 为对照，最终选择应由攻击实验的可区分性决定。
4. **相对近零指标依赖模型最大梯度**，只用于描述 scale 饱和；绝对近零（<1e-10）全部为 0。两者不能混用。
5. **SGM 稳定性实验仅用 100 张 clean validation 图像**。Phase 2 应扩展到全部 validation 身份，并加入 adversarial 样本；clean 结论不能外推到攻击样本。
6. **autoattack 已安装但尚未接入**；Phase 3 会做接口适配和 APGD 交叉验证。
7. **Windows 偶发 GPU 单次分配 OOM**：已有 3 次重试；长时间实验前确认显存空闲 >6 GiB。
8. **内存**：完整 ROC 的 18M 阈值已改为流式处理；若后续同时保留多组 18M 分数数组，仍需显式 `del + gc.collect()`（本次脚本已加入）。

---

## 9. 复现命令

```bash
PY="D:/code/research/envs/pr/python.exe"

# 任务 1–3：双协议、EER、阈值校准（全量 600 人）
$PY scripts/review_gallery_eer.py --model all --tag full

# 任务 4：划分顺序分析
$PY scripts/analyze_split_order.py

# 任务 5：CE vs Margin SGM 稳定性（100 张 validation 图）
$PY scripts/review_sgm_losses.py --model all --n-ids 20

# 全部单元测试
$PY -m unittest discover -s tests -v
```
