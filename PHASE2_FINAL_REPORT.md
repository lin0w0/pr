# PHASE 2 FINAL REPORT — 补充验证与机制分析

日期：2026-11-04  
范围：Phase 2 主体实验的可信度补充（Task 1）与 CompNet/CCNet 机制差异（Task 2）  
约束：保留 Phase 2 原始 CSV，不覆盖；不修改官方代码、checkpoint、数据集、split；不实现 CSR；test 集未使用。

原始 Phase 2 报告：[PHASE2_REPORT.md](PHASE2_REPORT.md)  
补充结果目录：`results/phase2_supplement/`

---

## 0. 数据与配置

| 项 | 值 |
|---|---|
| Gallery | Session1，600 身份 ×10 图 |
| Query | Validation，Session2 前 5 图，600 身份 ×5 = 3000 |
| Test | 未使用 |
| CompNet | 512-D，冻结 |
| CCNet | 2048-D，冻结 |
| 主匹配协议 | Prototype（均值原型） |
| 攻击预算（原始） | L∞ ε=2/255；PGD-Margin 50 步；APGD-CE 50 步 |
| 等 L2 补充 | CompNet r=0.6750；CCNet r=0.6049（原 low-band L∞ 攻击的平均 L2） |
| 频带 | low [0,0.2]，middle [0.2,0.5]，high [0.5,1.0] |
| 运行环境 | torch 2.5.1+cu121，FP32，TF32 关闭 |

补充实验的规模：
- 滤波器/权重/残差审计：使用全部已有 Phase 2 结果。
- 动态 vs 固定掩码：100 身份（500 图）。
- 等 L2 攻击：100 身份（500 图）。
- 机制分析：clean margin 与频带梯度使用全部 3000 图；频带攻击 margin 使用 100 身份。
- 原始 Phase 2 的 A/B/C/D 主结果仍为完整 600 身份。

---

## 1. 修正 Gaussian LPF 的 -3 dB 解释

### 1.1 数学

```
G_sigma(rho) = exp(-rho^2 / (2 sigma^2))
```

**半增益（G=0.5）**：

```
rho_half = sigma * sqrt(2 ln 2) = 1.17741 sigma
```

这对应幅度 **-6.0206 dB**，不是 -3 dB。

**真正的 -3 dB（幅度 G=1/√2=0.70711）**：

```
rho_minus3db = sigma * sqrt(ln 2) = 0.83255 sigma
```

原 Phase 2 报告把“半增益截止”误标为 -3 dB；这只影响解释文字，不影响任何实验结果。

### 1.2 修正后的截止频率

| σ | 半增益 rho (G=0.5, -6.02 dB) | 真正 -3 dB rho (G=0.707) |
|---|---|---|
| 0.02 | 0.0235 | 0.0167 |
| 0.05 | 0.0589 | 0.0416 |
| 0.10 | 0.1177 | 0.0833 |
| 0.15 | 0.1766 | 0.1249 |
| 0.20 | 0.2355 | 0.1665 |
| 0.30 | 0.3532 | 0.2498 |
| 0.50 | 0.5887 | 0.4163 |
| 0.75 | 0.8831 | 0.6244 |
| 1.00 | 1.1774 | 0.8326 |
| 1.50 | 1.7661 | 1.2488 |

因此 Phase 2 中“σ=0.10 保留 rho<0.118”应表述为“半增益截止 0.118；-3 dB 截止 0.083”。

结果文件：
- `results/phase2_supplement/audit/filter_response_cutoffs.csv`
- `results/phase2_supplement/audit/filter_response_corrected.csv`

---

## 2. SGM 频带平均梯度幅值与 rfft2 单边权重

### 2.1 rfft2 单边权重

对实图像，`rfft2` 返回 `[128, 65]`：
- fx=0 与 fx=Nyquist 列是自共轭，权重 1；
- 其余列代表共轭频率对，权重 2。

权重矩阵和为 16384（=128×128），但半频谱只有 8320 个 bin。若不加权，直接求和会把内部频率的能量少算约一半。

实测 Parseval：
- 加权：`|E_weighted - E_real|_max = 1.95e-3`
- 未加权：`8.12e3`
- 未加权/加权平均能量比：**0.508**

结果文件：`results/phase2_supplement/audit/rfft_weights.json`

### 2.2 加权 vs 未加权频带比例

从 Phase 2 保存的 `mean |grad|` 半频谱图重新计算（线性运算，等价于逐样本求和）：

| 模型 | 条件 | Loss | 频带 | 未加权比例 | 加权比例 | 加权平均 |grad| |
|---|---|---|---|---|---|---|---|
| CompNet | clean | margin | low | 0.398 | 0.397 | 0.4051 |
| CompNet | clean | margin | middle | 0.423 | 0.423 | 0.0827 |
| CompNet | clean | margin | high | 0.179 | 0.180 | 0.0191 |
| CompNet | clean | CE | low | 0.405 | 0.404 | 0.0686 |
| CompNet | clean | CE | middle | 0.418 | 0.418 | 0.0136 |
| CompNet | clean | CE | high | 0.177 | 0.178 | 0.0031 |
| CCNet | clean | margin | low | 0.554 | 0.552 | 0.9576 |
| CCNet | clean | margin | middle | 0.288 | 0.290 | 0.0963 |
| CCNet | clean | margin | high | 0.158 | 0.158 | 0.0284 |
| CCNet | clean | CE | low | 0.561 | 0.559 | 0.0977 |
| CCNet | clean | CE | middle | 0.283 | 0.285 | 0.0095 |
| CCNet | clean | CE | high | 0.156 | 0.156 | 0.0028 |

观察：

- 加权与未加权比例差异很小（<0.002），因为 SGM 分布较平滑；结论不依赖权重选择。
- **频带平均 |grad|** 与“频带比例”回答不同问题：low 带占面积小但平均梯度大；high 带面积大但平均梯度小。CompNet clean margin 的低/中/高平均梯度为 0.405/0.083/0.019；CCNet 为 0.958/0.096/0.028。
- 因此不能只用“频带能量比例”比较不同频带的重要性；平均梯度幅值更能反映单个频率 bin 的敏感性。

结果文件：`results/phase2_supplement/audit/sgm_band_weighting.csv`

---

## 3. 动态有效像素掩码 vs 固定原始 ROI 掩码

### 3.1 定义

- **动态掩码**（官方 `NormSingleROI`）：`pixel > 0`，对选中像素做 z-score，非正像素保持原值。
- **固定掩码**（诊断）：固定 128×128 全 ROI，对所有像素做 z-score。

### 3.2 结果（100 身份，500 图）

| 检查 | 结果 |
|---|---|
| Clean / PGD / APGD 的非正像素数 | **0** |
| σ=0.05/0.10/0.20 低通后的非正像素数 | **0** |
| 动态 vs 固定特征余弦（clean/PGD/APGD） | 1.000000（min 0.9999998） |
| 预测一致性 | 100% |
| 两种掩码下的 consistency 差 | 约 1e-8 |
| consistency 相关 | ≈1.0 |

原因：Tongji ROI 全部像素为正；Gaussian 低通由 DC 主导，输出仍为正，因此两种掩码在当前数据和滤波强度下**完全等价**。

结论：动态/固定掩码差异在本数据上不影响 Phase 2 结果；若未来使用更激进的滤波、裁剪或含零背景数据，才可能产生差异。官方动态掩码保持不变。

结果文件：`results/phase2_supplement/audit/mask_comparison.csv`

---

## 4. 等 L2 预算的频带攻击

### 4.1 设置

- 100 身份（500 图）。
- 每个模型三个频带使用**相同 L2 预算**：
  - CompNet r=0.6750
  - CCNet r=0.6049
  - 取自原始 low-band L∞ 攻击的平均 L2。
- 投影：Dykstra 到 band ∩ L2 ball ∩ pixel box；无 L∞ 上限。
- 非定向 margin 攻击，30 步，alpha=1/255，相同 seed。

### 4.2 结果

**CompNet（r=0.6750）**

| 频带 | 实际 L2 | ASR | Rank-1 | 特征偏移 | 频带外 max | 失败率 |
|---|---|---|---|---|---|---|
| low | 0.6750 | **77.0%** | 23.0% | 0.2806 | 5.99e-14 | 0% |
| middle | 0.6750 | 15.4% | 84.6% | 0.1212 | 2.76e-11 | 0% |
| high | 0.6750 | 0.2% | 99.8% | 0.0010 | 2.51e-14 | 0% |

**CCNet（r=0.6049）**

| 频带 | 实际 L2 | ASR | Rank-1 | 特征偏移 | 频带外 max | 失败率 |
|---|---|---|---|---|---|---|
| low | 0.6049 | **38.6%** | 61.4% | 0.1908 | 6.14e-14 | 0% |
| middle | 0.6049 | 11.2% | 88.8% | 0.0979 | 3.61e-11 | 0% |
| high | 0.6049 | 0.2% | 99.8% | 0.0049 | 2.52e-14 | 0% |

结论：**低频优势不是 L2 能量差异造成的**。在等 L2 下，low 仍远强于 middle/high；high 仍几乎无效。CompNet 对低频更脆弱。

结果目录：`results/phase2_supplement/equal_l2/{compnet,ccnet}/`

---

## 5. 攻击投影残差与频带外能量审计

### 5.1 全局 PGD/APGD（100 身份）

| 模型 | 攻击 | L∞ max | L∞ 超出 | box 违反 | L2 均值 | ASR |
|---|---|---|---|---|---|---|
| CompNet | PGD | 0.00784317 | 2.89e-8 | 0 | 0.9547 | 99.8% |
| CompNet | APGD | 0.00784317 | 2.89e-8 | 0 | 0.9717 | 99.8% |
| CCNet | PGD | 0.00784317 | 2.89e-8 | 0 | 0.8519 | 88.4% |
| CCNet | APGD | 0.00784317 | 2.89e-8 | 0 | 0.8343 | 94.8% |

（100 身份子集 ASR；全量 3000 的 ASR 为 CompNet 99.97%/99.97%，CCNet 94.27%/98.20%。）

### 5.2 频带攻击（全量 3000）

| 检查 | 最大值 |
|---|---|
| 频带外能量比例（low） | 1.29e-5 |
| 频带外能量比例（middle） | 3.85e-8 |
| 频带外能量比例（high） | 1.50e-10 |
| box 违反 | 0 |
| L∞ 超出 | 0 |
| 频带残差 L∞ | 4.12e-4（low） |
| 基于残差的失败率 | **0%** |

**重要说明：** Dykstra 的 `converged` 标记按“相邻迭代变化 ≤1e-6”判断；low/middle 在 20 次迭代内通常为 0%，但**频带外能量比例仍 ≤1.3e-5**，远低于 1e-3 容差。因此不能把 `converged=0` 误读为攻击失败；失败判据是残差指标，实际失败率为 0。high band 的 `converged` 为 100%。

结果文件：
- `results/phase2_supplement/audit/global_attack_constraints.csv`
- `results/phase2_supplement/audit/attack_constraint_audit.csv`

---

## 6. CompNet vs CCNet 机制差异

### 6.1 Clean identity margin（logit scale=10，全量 3000）

| 模型 | 均值 | 中位数 | P5 | P95 | 按身份 95% CI |
|---|---|---|---|---|---|
| CompNet | 5.58 | 5.78 | 3.27 | 7.17 | [5.49, 5.67] |
| CCNet | 6.91 | 7.17 | 4.61 | 8.26 | [6.82, 6.99] |

CCNet 的 clean margin 明显更大（约 +1.33 logit，约 +24%）。

### 6.2 频带梯度投影（全量 3000）

定义：
```
g = d L_margin / d x
g_k = P_band_k(g)
band_projection_ratio = ||g_k||_2 / ||g||_2
directional_derivative = ||g_k||_2
```

| 模型 | 频带 | 投影比例 | 梯度范数 | 与 full grad 余弦 |
|---|---|---|---|---|
| CompNet | low | 0.829 | 12.49 | 0.829 |
| CompNet | middle | 0.534 | 8.04 | 0.534 |
| CompNet | high | 0.162 | 2.44 | 0.162 |
| CCNet | low | **0.940** | **38.37** | 0.940 |
| CCNet | middle | 0.289 | 8.63 | 0.289 |
| CCNet | high | 0.120 | 3.84 | 0.120 |

CCNet 的低频梯度占比与绝对范数都更高，这与 SGM 低频比例（0.554 vs 0.398）一致。**但“梯度更偏低频”不等于“低频攻击更成功”。**

### 6.3 频带攻击后的 margin（100 身份，L∞=2/255）

| 模型 | 频带 | 成功数/500 | clean margin（成功） | clean margin（失败） | adv margin（成功） | adv margin（失败） | margin drop 均值 |
|---|---|---|---|---|---|---|---|
| CompNet | low | 332 | 5.25 | 6.62 | -2.38 | 1.36 | 6.83 |
| CompNet | middle | 55 | 3.22 | 6.02 | -1.11 | 2.57 | 3.54 |
| CompNet | high | 1 | 0.23 | 5.72 | -0.10 | 5.39 | 0.33 |
| CCNet | low | 127 | 5.53 | 7.46 | -1.89 | 2.39 | 5.67 |
| CCNet | middle | 56 | 4.55 | 7.27 | -2.05 | 3.32 | 4.25 |
| CCNet | high | 2 | 1.16 | 6.99 | -0.28 | 5.96 | 1.04 |

规律：

- 成功攻击的样本在 clean 时 margin 更低；失败样本 margin 更高。
- CCNet 的失败样本 clean margin 更高（low band：7.46 vs 6.62），是低 ASR 的直接原因之一。
- 相同 L∞ 下，CCNet low-band 的 margin drop（5.67）小于 CompNet（6.83），特征偏移也更小（全量：0.176 vs 0.244）。

### 6.4 特征偏移与攻击成功

全量 D 的逐样本结果：

| 模型 | 频带 | 目标 | 成功特征偏移 | 失败特征偏移 | AUC(shift→success) | 点二列相关 |
|---|---|---|---|---|---|---|
| CompNet | low | untargeted | 0.270 | 0.158 | 0.876 | 0.477 |
| CompNet | low | targeted | 0.319 | 0.199 | 0.876 | 0.631 |
| CompNet | middle | untargeted | 0.140 | 0.102 | 0.713 | 0.280 |
| CCNet | low | untargeted | 0.254 | 0.142 | **0.919** | 0.680 |
| CCNet | low | targeted | 0.290 | 0.157 | **0.919** | 0.596 |
| CCNet | middle | untargeted | 0.202 | 0.096 | 0.902 | 0.592 |

特征偏移与成功强相关（AUC 0.71–0.92），但同一特征偏移下 CCNet 的成功率仍更低，说明**特征偏移不是唯一决定因素，margin/决策边界距离同样关键**。

### 6.5 为什么 CCNet 低频 SGM 占比更高，但低频 ASR 更低？

基于以上实测，解释如下（按证据强度排序）：

1. **CCNet 的 clean margin 更大**：6.91 vs 5.58（logit），需要更大的 margin drop 才能翻转。CCNet 失败样本的 margin 尤其高（low band 7.46 vs CompNet 6.62）。
2. **同预算下 CCNet 的有效 margin drop 与特征偏移更小**：low-band margin drop 5.67 vs 6.83；特征偏移 0.176 vs 0.244。尽管 CCNet 的低频梯度范数更大（38.37 vs 12.49），但 sign-PGD 在 L∞ 与频带约束下产生的实际特征移动更小。
3. **SGM 频带比例是“梯度能量分布”，不是“可攻击性”**：CCNet 的低频梯度占比更高（0.940 投影比例；SGM 0.554 vs 0.398），但攻击成功率取决于 margin、梯度方向与有效扰动幅度的共同作用；高占比不等于高 ASR。
4. **低频梯度方向与 margin 降低方向不完全一致**：CCNet 的低频梯度能量大，但在 L∞ sign 步和 Dykstra 投影下，部分能量被频带/像素约束抵消；其失败样本仍保持较大正 margin。
5. **特征偏移与成功的关系在两模型都强**（AUC 0.88–0.92 low band），但 CCNet 达到相同成功率需要更大的 feature shift 或更低的 clean margin；这与第 1、2 点一致。

因此，**不能仅凭 SGM 低频占比预测模型脆弱性**。防御设计（CSR）应同时考虑 margin、梯度方向与频带扰动效率，而不是只看频谱能量。

结果目录：`results/phase2_supplement/mechanism/`

---

## 7. 代码修改记录

新增/修改：

| 文件 | 内容 |
|---|---|
| `spectral/fft_utils.py` | `one_sided_weights`、`weighted_spectral_energy` |
| `spectral/gaussian_filter.py` | `rho_at_gain`；修正 `filter_response_table`，输出半增益与真正 -3 dB |
| `datasets/preprocessing.py` | `fixed_mask_normalize`（诊断，不改官方流程） |
| `spectral/band_projection.py` | `project_l2`、`project_pixel_box`、`constraint_metrics_l2`、`dykstra_project_band_l2_box` |
| `attacks/band_limited.py` | `band_limited_margin_l2` |
| `evaluation/detection_metrics.py` | 检测阈值/ROC/AUC/多尺度分数 |
| `experiments/run_phase2_audit.py` | Task 1 审计 |
| `experiments/run_equal_l2_attacks.py` | 等 L2 攻击 |
| `experiments/run_mechanism_analysis.py` | Task 2 机制分析 |
| `experiments/run_detection_preliminary.py` | Task 3 检测分析 |
| `tests/test_phase2_supplement.py` | 权重、截止频率、固定掩码、L2 投影、检测指标 9 项测试 |

未修改：
- `official_project/`、权重、数据、现有 split
- 原始 Phase 2 CSV/JSON 结果

测试：**76/76 OK**（新增 9 项）。

---

## 8. 失败案例与限制

1. **CCNet mask 对比显存不足**：一次处理 500 图导致 OOM；改为 batch=32 后完成。结果不受影响。
2. **CSV 字段不一致**：mask 对比中 clean/adv 行与 filtered 行字段不同，首次保存失败；显式字段列表后修复。
3. **Dykstra `converged` 标记**：low/middle 在 20 次迭代内通常为 0%，但频带外能量 ≤1.3e-5，残差失败率 0%。报告必须区分“迭代变化收敛”和“约束满足”。
4. **高頻带 targeted 特征偏移缺失**：成功样本为 0，统计为 NaN；不解释为 0 偏移。
5. **等 L2 与机制攻击使用 100 身份**（500 图），不是全 600；结论方向与全量 C/D 一致，但具体数值需在全量复核。
6. **等 L2 半径取自 low-band L∞ 攻击的平均 L2**，不是逐样本预算匹配；不同预算可能改变 ASR 绝对值。
7. **掩码对比只在当前数据/滤波范围成立**：Tongji ROI 无零像素，低通后也无非正像素；不能推广到含黑边或强滤波数据。
8. **检测为初步分析**：calibration 100 身份、evaluation 500 身份；尚未做自适应攻击或跨数据集验证。

---

## 9. 输出目录

```
results/phase2_supplement/
  audit/
    filter_response_corrected.csv
    filter_response_cutoffs.csv
    rfft_weights.json
    rfft_weights_grid.csv
    sgm_band_weighting.csv
    mask_comparison.csv
    global_attack_constraints.csv
    attack_constraint_audit.csv
    audit_summary.json
  equal_l2/
    compnet/{per_sample.csv,summary.csv,run_meta.json}
    ccnet/{per_sample.csv,summary.csv,run_meta.json}
  mechanism/
    clean_margins_{model}.csv
    gradient_bands_{model}.csv
    gradient_bands_summary_{model}.csv
    attack_margins_{model}.csv
    attack_margins_summary_{model}.csv
    feature_shift_success.csv
    mechanism_summary.json
```

---

## 10. 复现命令

```bash
PY="D:/code/research/envs/pr/python.exe"

$PY experiments/run_phase2_audit.py
$PY experiments/run_equal_l2_attacks.py --model all --max-ids 100
$PY experiments/run_mechanism_analysis.py --model all --attack-max-ids 100
$PY -m unittest discover -s tests -v
```
