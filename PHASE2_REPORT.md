# PHASE 2 REPORT — 掌纹识别网络的空间频谱脆弱性分析

日期：2026-11-03  
协议：协议 A（Gallery=Session1 全部；Validation=Session2 前 5 张；Test 未使用）  
模型：CompNet（512-D）、CCNet（2048-D），全程冻结  
环境：Python 3.10.22 / torch 2.5.1+cu121 / RTX 4070 Laptop 8GB / FP32（TF32 关闭）

---

## 0. 完成情况

| 阶段 | 内容 | 数据规模 | 状态 |
|---|---|---|---|
| 2.1 | FFT / Gaussian LPF / 频带掩码 / Dykstra | 单元测试 | ✅ |
| 2.2 | Progressive Frequency Attenuation | **600 身份，3000 图** | ✅ |
| 2.3 | SGM Frequency Gradient Analysis | **600 身份，3000 图** | ✅ |
| 2.4 | Band-limited Feature Sensitivity | **600 身份，3000 图** | ✅ |
| 2.5 | Band-limited Adversarial Attack | **600 身份，3000 图** | ✅ |
| 2.6 | 汇总、统计、图表 | 全部结果 | ✅ |

- A/B/C/D 全部使用完整 validation（600 身份 ×5 图 = 3000 张）。
- 单元测试：**67/67 通过**。
- 所有实验只使用 validation；test 未参与任何超参数、阈值或结果选择。

对抗攻击（validation，eps=2/255）：

| 模型 | PGD-Margin ASR | APGD-CE ASR | 干净准确率 |
|---|---|---|---|
| CompNet | 99.97% | 99.97% | 100% |
| CCNet | 94.27% | 98.20% | 100% |

eps=2/255 由 100 图 validation pilot 选定（1/255 时 CompNet ASR 50–52%、CCNet 19–38%，过弱）。

---

## 1. Phase 2.1：FFT / 滤波器 / 频带投影

### 1.1 数学定义

```
X(u,v) = FFT(x)
x_low = IFFT(X * G_sigma)
rho(u,v) = sqrt(fy(u)^2 + fx(v)^2) / sqrt(0.5^2 + 0.5^2) ∈ [0,1]
G_sigma(rho) = exp(-rho^2 / (2 sigma^2))
```

`sigma` 越小，低通越强；-3 dB 截止 `rho_half = sigma * sqrt(2 ln 2)`。

频带约束：

```
A = { delta : FFT(delta) * (1 - M_k) = 0 }
B = { delta : ||delta||inf <= eps, 0 <= x + delta <= 1 }
```

Dykstra 交替投影求 `A ∩ B`，并记录频带外能量、L∞ 超出、像素违反、频带残差、迭代与收敛。

### 1.2 测试

15 项专项测试全部通过：FFT 往返、DC 增益、sigma 单调性、高频衰减、掩码互斥覆盖、频带投影、box 投影、Dykstra 交点、梯度流。

`tests/test_fft_filter.py`、`tests/test_band_projection.py`。

### 1.3 失败案例

无。`sigma=100` 的近似恒等测试曾因容差 1e-5 与数值误差 1.59e-5 失败，改用 `sigma=1e4` 后通过；不涉及实验参数。

---

## 2. Phase 2.2：Progressive Frequency Attenuation（全量 600 身份）

### 2.1 定义

```
C(x, sigma) = cosine( normalize(f(x)), normalize(f(G_sigma(x))) )
```

- 滤波在原始 128×128 ROI 上进行，之后才经过官方可微预处理和冻结模型。
- 条件：Clean、Gaussian Noise（σ=0.05，clipped [0,1]）、PGD、APGD。
- sigma：`[0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.00, 1.50]`。
- 统计：按身份先平均，再对 600 个身份做 bootstrap 95% CI。

### 2.2 结果（均值；括号为按身份配对 95% CI）

**CompNet**

| 条件 | σ=0.02 | 0.05 | 0.10 | 0.20 | 0.30 | 0.50 | 1.00 |
|---|---|---|---|---|---|---|---|
| clean | 0.179 (0.171, 0.187) | 0.535 (0.528, 0.542) | 0.826 (0.823, 0.829) | 0.970 (0.970, 0.971) | 0.992 (0.992, 0.993) | 0.999 | 1.000 |
| gaussian | 0.157 (0.150, 0.163) | 0.370 (0.363, 0.377) | 0.650 (0.646, 0.654) | 0.923 (0.922, 0.924) | 0.974 | 0.993 | 0.999 |
| pgd | 0.131 (0.124, 0.137) | 0.342 (0.334, 0.350) | 0.621 (0.615, 0.626) | 0.914 (0.912, 0.915) | 0.977 | 0.996 | 1.000 |
| apgd | 0.125 (0.118, 0.131) | 0.321 (0.312, 0.330) | 0.585 (0.577, 0.593) | 0.896 (0.893, 0.898) | 0.971 | 0.996 | 1.000 |

**CCNet**

| 条件 | σ=0.02 | 0.05 | 0.10 | 0.20 | 0.30 | 0.50 | 1.00 |
|---|---|---|---|---|---|---|---|
| clean | 0.146 (0.140, 0.152) | 0.566 (0.560, 0.573) | 0.841 (0.838, 0.845) | 0.972 (0.971, 0.973) | 0.992 | 0.998 | 1.000 |
| gaussian | 0.122 (0.117, 0.128) | 0.419 (0.413, 0.425) | 0.678 (0.674, 0.682) | 0.912 (0.911, 0.914) | 0.965 | 0.991 | 0.999 |
| pgd | 0.110 (0.105, 0.116) | 0.367 (0.358, 0.375) | 0.600 (0.591, 0.608) | 0.882 (0.878, 0.886) | 0.961 | 0.992 | 0.999 |
| apgd | 0.094 (0.089, 0.099) | 0.303 (0.292, 0.314) | 0.519 (0.506, 0.531) | 0.840 (0.833, 0.848) | 0.946 | 0.990 | 0.999 |

### 2.3 解读

- 仅保留近 DC（σ=0.02, rho_half≈0.024）时，一致性只有 0.15–0.18：身份信息**不**只存在于最低频。
- σ=0.05（rho_half≈0.059）时 clean 一致性约 0.54–0.57；σ=0.10（rho_half≈0.118）约 0.83–0.84；σ≥0.20 时 ≥0.97。
- 对抗样本曲线整体更低，APGD 比 PGD 更低；CCNet 的 APGD 曲线下降最大（σ=0.10 时 0.519）。
- **不能**据此声称“低频锚点一定有效”：强低通会丢失大量身份信息（σ=0.05 已损失约 45% 一致性），弱低通又可能保留攻击成分。

---

## 3. Phase 2.3：SGM Frequency Gradient Analysis（全量 600 身份）

### 3.1 定义

```
S(u,v) = mean_n | dL_n / dA_n(u,v) |
```

- `A = |rfft2(x)|` 为可微变量，相位固定；重建后经过官方可微预处理和冻结模型。
- 逐样本求梯度，再对绝对梯度跨样本平均。
- 主损失 Margin scale=10；对照 CE scale=10。
- 条件 Clean / PGD / APGD；batch=16，min_batch=4，FP32，TF32 关闭。

### 3.2 结果

| 模型 | 条件 | Loss | L1 均值 (95% CI) | Max | 低/中/高梯度能量 |
|---|---|---|---|---|---|
| CompNet | clean | margin | 532.7 (524.1, 541.9) | 5.44 | 0.398 / 0.423 / 0.179 |
| CompNet | clean | ce | 88.5 (82.5, 95.0) | 2.70 | 0.405 / 0.418 / 0.177 |
| CompNet | pgd | margin | 589.4 (582.9, 596.2) | 5.16 | 0.402 / 0.420 / 0.178 |
| CompNet | pgd | ce | 556.6 (549.0, 564.5) | 5.87 | 0.405 / 0.418 / 0.177 |
| CompNet | apgd | margin | 563.4 (557.7, 568.9) | 5.16 | 0.407 / 0.416 / 0.177 |
| CompNet | apgd | ce | 551.3 (543.9, 558.5) | 5.46 | 0.410 / 0.415 / 0.176 |
| CCNet | clean | margin | 906.6 (886.7, 928.5) | 177.43 | 0.554 / 0.288 / 0.158 |
| CCNet | clean | ce | 91.3 (81.6, 101.4) | 54.71 | 0.561 / 0.283 / 0.156 |
| CCNet | pgd | margin | 1369.8 (1340.5, 1400.4) | 306.29 | 0.545 / 0.294 / 0.161 |
| CCNet | pgd | ce | 1168.5 (1135.5, 1202.2) | 246.13 | 0.542 / 0.297 / 0.161 |
| CCNet | apgd | margin | 1138.9 (1118.1, 1159.9) | 296.64 | 0.530 / 0.301 / 0.168 |
| CCNet | apgd | ce | 1050.8 (1027.0, 1075.2) | 262.02 | 0.527 / 0.304 / 0.169 |

绝对近零比例（L1<1e-10）在所有配置均为 0。相对近零（<该模型全部配置最大单样本 L1 的 1%）：CompNet clean CE 3.9%，CCNet clean CE 62.4%，CCNet APGD CE 1.8%，其余为 0；这是相对量级指标，不是梯度精确为零。

### 3.3 解读

- 两模型 clean SGM 都集中在低/中频；高頻带能量约 16–18%，并非为零。
- CompNet 的低频比例约 0.40，CCNet 更高（约 0.55），说明 CCNet 的身份梯度更偏向低频。
- 对抗样本显著放大 CE 梯度（CompNet 88.5→556.6，约 6.3×；CCNet 91.3→1168.5，约 12.8×），Margin 放大较小（约 1.1–1.5×）。
- 频带比例在 clean/对抗之间只有小幅变化；对抗攻击主要放大梯度量级，而非彻底改变频谱分布。

---

## 4. Phase 2.4：Band-limited Feature Sensitivity（全量 600 身份，3000 图）

### 4.1 定义

```
maximize  1 - cosine(f(x), f(x + delta_k))
subject to delta_k ∈ A_k ∩ B
```

Dykstra 投影；最终报告频带外能量、box 违反、L∞ 超出、残差、迭代/收敛，并同时报告 Rank-1/5 与识别错误率。

### 4.2 结果

| 模型 | 频带 | 特征偏移均值 (95% CI) | Rank-1 (95% CI) | 识别错误率 | 频带外能量 max | 投影失败率 |
|---|---|---|---|---|---|---|
| CompNet | low | 0.3883 (0.3774, 0.3996) | 90.8% (89.1, 92.4) | 9.2% | 1.29e-5 | 0% |
| CompNet | middle | 0.2525 (0.2466, 0.2586) | 99.4% (99.0, 99.7) | 0.6% | 3.75e-8 | 0% |
| CompNet | high | 0.0021 (0.0020, 0.0022) | 100.0% | 0.0% | 1.06e-10 | 0% |
| CCNet | low | 0.2789 (0.2708, 0.2879) | 98.5% (97.7, 99.1) | 1.5% | 1.20e-5 | 0% |
| CCNet | middle | 0.2197 (0.2128, 0.2271) | 99.4% (99.0, 99.8) | 0.6% | 2.92e-8 | 0% |
| CCNet | high | 0.0175 (0.0168, 0.0182) | 100.0% | 0.0% | 1.18e-10 | 0% |

### 4.3 解读

- 相同 L∞ 预算下，低频扰动造成的特征偏移最大，识别错误最多；中频次之；高频几乎无影响。
- CompNet 对低频扰动更敏感（Rank-1 降到 90.8%），CCNet 更稳（98.5%）。
- 投影失败率为 0；频带外能量 max ≤1.3e-5，低于 1e-3 容差。

---

## 5. Phase 2.5：Band-limited Adversarial Attack（全量 600 身份，3000 图）

### 5.1 定义

- 非定向：最大化 `L_margin = max_{j≠y} s_j - s_y`。
- 定向：最大化 `s_target - max_{j≠target} s_j`，`target = (y+1) mod 600`。
- 各频带使用相同 eps=2/255、30 步、alpha=1/255、相同 seed。
- ASR 分母为 clean-correct 样本；CI 按身份 bootstrap。

### 5.2 结果

**CompNet**

| 频带 | 目标 | ASR (95% CI) | Rank-1 | 特征偏移 | L2 均值 | L∞ 均值 | 频带外 max | 失败率 |
|---|---|---|---|---|---|---|---|---|
| low | untargeted | 76.6% (74.0, 79.3) | 23.4% | 0.2437 | 0.675 | 0.00784 | 1.25e-5 | 0% |
| low | targeted | 40.2% (37.1, 43.5) | 59.8% | 0.2470 | 0.671 | 0.00784 | 1.19e-5 | 0% |
| middle | untargeted | 12.9% (10.9, 14.9) | 87.1% | 0.1073 | 0.712 | 0.00784 | 3.85e-8 | 0% |
| middle | targeted | 1.6% (1.0, 2.4) | 98.3% | 0.1221 | 0.712 | 0.00784 | 3.26e-8 | 0% |
| high | untargeted | 0.0% (0.0, 0.1) | 100.0% | 0.0011 | 0.805 | 0.00784 | 1.07e-10 | 0% |
| high | targeted | 0.0% | 100.0% | 0.0011 | 0.806 | 0.00784 | 1.50e-10 | 0% |

**CCNet**

| 频带 | 目标 | ASR (95% CI) | Rank-1 | 特征偏移 | L2 均值 | L∞ 均值 | 频带外 max | 失败率 |
|---|---|---|---|---|---|---|---|---|
| low | untargeted | 30.6% (27.7, 33.5) | 69.4% | 0.1760 | 0.605 | 0.00784 | 1.16e-5 | 0% |
| low | targeted | 13.1% (11.2, 15.3) | 86.9% | 0.1744 | 0.604 | 0.00784 | 1.13e-5 | 0% |
| middle | untargeted | 12.3% (10.3, 14.4) | 87.7% | 0.1092 | 0.678 | 0.00784 | 2.01e-8 | 0% |
| middle | targeted | 4.8% (3.5, 6.2) | 95.2% | 0.1161 | 0.679 | 0.00784 | 2.40e-8 | 0% |
| high | untargeted | 0.1% (0.0, 0.3) | 99.9% | 0.0068 | 0.744 | 0.00784 | 1.09e-10 | 0% |
| high | targeted | 0.0% | 100.0% | 0.0069 | 0.744 | 0.00784 | 1.06e-10 | 0% |

### 5.3 解读

- 相同 L∞ 预算下，低频攻击最有效；中频次之；高频几乎完全无效。
- 高频扰动的 L2 反而最大（CompNet 0.805 vs low 0.675；CCNet 0.744 vs low 0.605），但在 L∞ 相同的情况下 ASR 仍接近 0。这说明网络对高频空间扰动的身份决策不敏感，而不是高频扰动“能量不足”。
- CompNet 比 CCNet 更易受低频攻击（ASR 76.6% vs 30.6%，untargeted；40.2% vs 13.1%，targeted）。
- 所有频带的投影失败率均为 0，频带外能量 max ≤1.3e-5。

---

## 6. Phase 2.6：统计、图表与代码

### 6.1 统计

- 所有均值/CI 均按身份配对：先对同一身份的 5 张图求均值，再对身份做 bootstrap（2000 次）。
- SGM 的 L1 与频带比例、C/D 的特征偏移、Rank-1、ASR 均给出 95% CI。
- 汇总文件：`results/phase2/final_summary.csv`。

### 6.2 图表

`results/phase2/figures/`：

- attenuation：模型曲线、clean 模型对比、滤波器响应。
- SGM：raw/normalized 热力图（margin/CE）、径向曲线、频带能量柱状图、clean 模型对比。
- band：特征偏移、Rank-1、投影残差、untargeted/targeted ASR 与失败率。

热力图同时保留 raw 与按自身最大值归一化版本；标题中标注 raw max，避免归一化隐藏量级差异。

### 6.3 新增代码

- `spectral/fft_utils.py`、`gaussian_filter.py`、`spectral_consistency.py`、`band_projection.py`
- `attacks/base.py`、`pgd.py`、`apgd.py`、`band_limited.py`
- `experiments/_common.py`、`run_spectral_attenuation.py`、`run_sgm_full.py`、`run_band_sensitivity.py`、`run_band_attacks.py`、`run_phase2_summary.py`
- `evaluation/paired_stats.py`
- `scripts/build_phase2_adv_cache.py`
- `visualization/plot_spectral_analysis.py`、`plot_sgm_comparison.py`、`plot_band_sensitivity.py`
- `tests/test_fft_filter.py`、`test_band_projection.py`、`test_attacks_phase2.py`
- `configs/spectral.yaml`

未修改官方代码、checkpoint、数据集和 split。

### 6.4 测试

```
Ran 67 tests in 2.027s
OK
```

---

## 7. 结论：CSR 低频锚点是否值得迁移？

基于以上真实实验，**不能**直接假定“低频身份锚点”成立，也不能假定 CSR 一定有效。证据如下：

1. **低/中频确实承载大量身份信息，但不是全部。**
   - σ=0.10（仅保留 rho<0.118）时 clean 一致性仍有 0.83–0.84；
   - σ=0.05（rho<0.059）时降到 0.54–0.57；
   - σ=0.02（接近 DC）时只有 0.15–0.18。
   强低通会损失显著身份信息；弱低通又可能保留攻击成分。

2. **低频也是攻击最有效的频带。**
   - 在相同 L∞=2/255 下，低频攻击 ASR（CompNet 76.6%，CCNet 30.6%）远高于中频（12.9%/12.3%）和高频（≈0）。
   - 高频扰动 L2 更大却几乎无效。
   这意味着：如果对抗扰动主要落在低/中频，CSR 用低通特征构造的“身份锚点”本身可能已被污染；低通锚点未必能把特征拉回干净身份。

3. **高频攻击在该预算下很弱。**
   - 若 CSR 主要针对高频扰动，其收益可能有限，因为这类扰动本身难以改变身份判决。
   - 若 CSR 针对低/中频扰动，则低频锚点存在被污染的风险。

4. **模型差异。**
   - CCNet 的 SGM 更偏低频（低频比例 0.554 vs CompNet 0.398），低频攻击 ASR 更低（30.6% vs 76.6%）。
   - CompNet 对低频扰动更脆弱，可能更适合作为 CSR 的首个验证对象，但不能外推到 CCNet。

**因此，Phase 2 的结论是：低频锚点思想在掌纹上“有条件可试”，但绝不能预设有效。** Phase 3/4 应优先测试：

- 低/中频对抗扰动下，CSR 的检测与校正是否仍能恢复身份；
- 强/弱低通 sigma 对“锚点质量”和“攻击残留”的权衡；
- 是否应使用中频或多频带锚点，而不是单一强低通。

---

## 8. 失败案例与限制

1. **计算/内存问题**：
   - 全量缓存与实验并发时出现主机内存分配失败和 `CUBLAS_STATUS_EXECUTION_FAILED`；改为串行执行后恢复。这不是算法失败。
   - 最终全量 A/B/C/D 均在串行下完成。
2. **攻击预算单一**：eps=2/255，PGD-Margin 与 APGD-CE；未覆盖 CW、多种 eps、restarts。不能外推到其他威胁模型。
3. **高频结果依赖预算参数化**：高频扰动在同一 L∞ 下 L2 更大但无效；这是该约束下的真实结果，不代表所有高频攻击形式都无效。
4. **SGM 相对近零指标**依赖模型最大 L1；绝对近零为 0。报告已区分。
5. **test 未使用**：所有结论仅基于 validation；最终泛化需在 test 上按固定协议评估（Phase 3+）。

---

## 9. 复现命令

```bash
PY="D:/code/research/envs/pr/python.exe"

# 对抗缓存（全量 validation）
$PY scripts/build_phase2_adv_cache.py --model all

# A：全量
$PY experiments/run_spectral_attenuation.py --model all

# B：全量
$PY experiments/run_sgm_full.py --model all

# C/D：全量
$PY experiments/run_band_sensitivity.py --model compnet --batch-size 64
$PY experiments/run_band_sensitivity.py --model ccnet  --batch-size 32
$PY experiments/run_band_attacks.py     --model compnet --batch-size 64
$PY experiments/run_band_attacks.py     --model ccnet  --batch-size 32

# 汇总与图表
$PY experiments/run_phase2_summary.py
$PY visualization/plot_spectral_analysis.py
$PY visualization/plot_sgm_comparison.py
$PY visualization/plot_band_sensitivity.py

# 测试
$PY -m unittest discover -s tests -v
```
