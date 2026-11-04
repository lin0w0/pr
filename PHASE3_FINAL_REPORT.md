# PHASE3_FINAL_REPORT — 检测器最终验证与 Original CSR 基线

日期：2026-11-04  
范围：Task A（A1–A5 检测器最终验证）、Task B/C/D（Original CSR 规格、实现、基线）  
约束：test 未使用；模型冻结；不实现 middle-band/multi-band CSR；不覆盖已有 Phase 2/3 结果。

结果目录：
- Phase 3 检测：`results/phase3_final/`
- Original CSR：`results/csr_original/`

相关报告：
- [CSR_MATHEMATICAL_SPEC.md](CSR_MATHEMATICAL_SPEC.md)
- [ORIGINAL_CSR_IMPLEMENTATION_REPORT.md](ORIGINAL_CSR_IMPLEMENTATION_REPORT.md)
- [ORIGINAL_CSR_EXPERIMENT_REPORT.md](ORIGINAL_CSR_EXPERIMENT_REPORT.md)

---

## 0. 配置与划分

- 模型：CompNet（512-D）、CCNet（2048-D），冻结。
- 检测器：
  - 单尺度：CompNet σ=0.5，CCNet σ=0.3。
  - 多尺度：mean of σ∈{0.1,0.15,0.2}。
- 阈值：沿用 Phase 3 preliminary 的 validation calibration 身份 0–99：单/多尺度在 FPR=1%/5% 的 τ；A1–A4 固定使用，不按攻击类型调参。
- 最终评估身份：A1/A2 使用 100–299（1000 张）；A3/A4 使用 100–199（500 张）；A5 使用全部 600 身份 5-fold CV。
- Test 未读取。

---

## 1. A1：频带攻击与全局攻击检测

固定阈值，不针对攻击调参。AUC 为 clean eval vs 该攻击；TPR 在固定 τ 下。

### CompNet

| 攻击 | ASR | 单尺度 AUC | 单 TPR@1% | 单 TPR@5% | 多尺度 AUC | 多 TPR@1% | 多 TPR@5% |
|---|---|---|---|---|---|---|---|
| Low-band PGD | 77.3% | 0.7622 | 0.184 | 0.338 | 0.8090 | 0.148 | 0.395 |
| Middle-band PGD | 16.1% | 0.9837 | 0.814 | 0.927 | 0.9380 | 0.448 | 0.728 |
| High-band PGD | 0.0% | 0.7512 | 0.140 | 0.297 | 0.5099 | 0.004 | 0.065 |
| Global PGD | 99.97% | 0.9994 | 0.988 | 0.999 | 0.9938 | 0.906 | 0.974 |
| Global APGD | 99.97% | 0.9996 | 0.997 | 0.999 | 0.9951 | 0.929 | 0.979 |

### CCNet

| 攻击 | ASR | 单尺度 AUC | 单 TPR@1% | 单 TPR@5% | 多尺度 AUC | 多 TPR@1% | 多 TPR@5% |
|---|---|---|---|---|---|---|---|
| Low-band PGD | 30.8% | 0.7642 | 0.031 | 0.205 | 0.7985 | 0.077 | 0.304 |
| Middle-band PGD | 15.2% | 0.9572 | 0.428 | 0.737 | 0.9013 | 0.290 | 0.594 |
| High-band PGD | 0.1% | 0.8530 | 0.141 | 0.373 | 0.5716 | 0.012 | 0.068 |
| Global PGD | 94.27% | 0.9960 | 0.883 | 0.977 | 0.9908 | 0.817 | 0.945 |
| Global APGD | 98.20% | 0.9872 | 0.942 | 0.984 | 0.9866 | 0.918 | 0.978 |

**结论：** 检测器对全局 PGD/APGD 很强，但对频带攻击明显较弱，尤其 low/high band；high-band 攻击本身 ASR≈0，检测 AUC 接近随机是“扰动几乎不改变特征”的结果。A1 说明检测器不能按全局攻击的阈值外推到频带攻击。

---

## 2. A2：非对抗扰动误报

参数来自 `configs/phase3_final.yaml`；以下为多尺度检测器在目标 FPR=1% 阈值下的实际 FPR（评估身份 100–299，1000 张 clean）。

### CompNet（多尺度，τ@1%）

| 扰动 | 参数 | 实际 FPR |
|---|---|---|
| Gaussian noise | σ=0.01 / 0.03 / 0.05 / 0.10 | 0.018 / 0.387 / 0.900 / 1.000 |
| Gaussian blur | σ=0.5 / 1 / 2 | 0.000 / 0.000 / 0.000 |
| Brightness | ±0.05, ±0.10 | 0.005 |
| Contrast | 0.8 / 1.2 | 0.005 |
| JPEG | q=90 / 70 / 50 | 0.006 / 0.010 / 0.005 |
| Rotation | 2° / 5° / 10° | 0.001 / 0.517 / 1.000 |
| Translation | 1 / 2 / 4 px | 0.037 / 0.415 / 1.000 |

### CCNet（多尺度，τ@1%）

| 扰动 | 参数 | 实际 FPR |
|---|---|---|
| Gaussian noise | σ=0.01 / 0.03 / 0.05 / 0.10 | 0.011 / 0.183 / 0.765 / 0.997 |
| Gaussian blur | σ=0.5 / 1 / 2 | 0.005 / 0.000 / 0.000 |
| Brightness | ±0.05, ±0.10 | 0.007 |
| Contrast | 0.8 / 1.2 | 0.007 |
| JPEG | q=90 / 70 / 50 | 0.008 / 0.006 / 0.005 |
| Rotation | 2° / 5° / 10° | 0.026 / 0.945 / 1.000 |
| Translation | 1 / 2 / 4 px | 0.117 / 0.806 / 1.000 |

**结论：** 检测器对光照/对比度/JPEG/模糊较稳，但对高斯噪声、旋转、平移极其敏感；大于 2px 平移或 5° 旋转几乎必然误报。这是 CSR 检测分支在掌纹部署中的主要风险。

---

## 3. A3：CW 检测

CW 定义（本工程）：CW-style L2 margin attack

```
min_delta  ||delta||_2^2 - c * (max_other - s_y - kappa)
x_adv = 0.5*(tanh(w)+1), delta = x_adv - x
```

（tanh 保证像素范围；不再使用会阻断梯度的 clamp。）

配置：c candidates [0.1,1,10,100]，steps=100，lr=0.01，kappa=0；在 calibration 身份 0–19 选择 c（优先 ASR，再低 L2），在 100–199 评估。

| 模型 | 选择 c | 评估 ASR | L2 均值 | L∞ 均值 | 单尺度 AUC | 单 TPR@1% | 多尺度 AUC | 多 TPR@1% |
|---|---|---|---|---|---|---|---|---|
| CompNet | 1.0 | 100% | 1.296 | 0.0526 | 1.000 | 1.000 | 1.000 | 1.000 |
| CCNet | 1.0 | 100% | 1.955 | 0.0660 | 1.000 | 1.000 | 1.000 | 1.000 |

两个模型的 CW 样本 100% 被检测到；CW 的平均 L2（1.30 / 1.95）远大于 2/255 L∞ 攻击的 L2（约 0.6–0.9），扰动更明显。CW 的 ASR 与 L2/L∞ 记录在 `a3_cw_stats_{model}.json`。

---

## 4. A4：检测器自适应攻击

联合目标：

```
maximize  L_margin(x_adv) - lambda * D(x_adv)
```

`D` 为对应检测器分数（越低越像对抗），因此减去 λD 会推高 D（逃逸）；同时保持 L∞=2/255 与像素约束。初始化为已有全局 PGD 对抗样本，steps=30，α=1/255；λ 在 calibration 身份 0–19 选择（最大化 ASR 且逃逸），在 100–199 评估。

指标：
- 原始 PGD ASR
- 自适应 ASR
- 检测率 / 逃逸率
- ASR_evade = clean-correct 中“攻击成功且逃逸”
- ASR_given_escape = 逃逸样本中的成功率
- detected_success_rate = 攻击成功但被检测

**状态：按用户要求停止，A4 未运行。** 脚本 `experiments/run_phase3_adaptive.py` 已实现并通过语法检查，但未执行；因此本报告不提供自适应攻击的 ASR/逃逸率结果。不得用未运行结果替代。

---

## 5. A5：阈值稳定性（身份级 5-fold CV）

5 折按身份分组；每折在其余身份 clean 分数上校准 τ（FPR 1%/5%），在留出身份上评估；报告 5 折均值与 95% CI（正态近似）。

### AUC（5 折均值，95% CI）

| 模型 | 检测器 | PGD | APGD |
|---|---|---|---|
| CompNet | single | 0.9990 (0.9986, 0.9993) | 0.9990 (0.9988, 0.9992) |
| CompNet | multi | 0.9938 (0.9925, 0.9952) | 0.9950 (0.9939, 0.9960) |
| CCNet | single | 0.9968 (0.9958, 0.9977) | 0.9857 (0.9854, 0.9860) |
| CCNet | multi | 0.9916 (0.9896, 0.9937) | 0.9850 (0.9842, 0.9858) |

### TPR@FPR=1%（5 折均值，95% CI）

| 模型 | 检测器 | PGD | APGD |
|---|---|---|---|
| CompNet | single | 0.9781 (0.9759, 0.9803) | 0.9857 (0.9839, 0.9875) |
| CompNet | multi | 0.9049 (0.9014, 0.9084) | 0.9245 (0.9219, 0.9272) |
| CCNet | single | 0.9349 (0.9268, 0.9431) | 0.9650 (0.9620, 0.9680) |
| CCNet | multi | 0.8480 (0.8391, 0.8569) | 0.9321 (0.9272, 0.9370) |

实际 FPR 在每折内报告于 `a5_threshold_stability.csv`；整体与目标接近。

**结论：** 全局攻击检测的阈值/指标在身份级交叉验证下稳定；单尺度 AUC/TPR 高于多尺度，但单尺度阈值更接近 1。

---

## 6. Original CSR 基线（Task D）

配置：原始 CSR（r=22.857 掌纹换算，τ 由 calibration 0–99 校准到 FPR 5%；另报告 τ=0.85），N=3，ε=4/255，α=2/255，λ=1，Greedy。

攻击：Global PGD、Global APGD、Low-band PGD；评估身份 100–199（500 张）。

指标：Clean Rank-1、Adversarial Rank-1、Rank-5、Recovery Rate、Clean Mis-correction、Feature Cosine、L2、L∞、平均推理时间。

**状态：按用户要求停止，Original CSR 基线实验未运行。** 实现、单元测试和实验脚本已就绪（`defenses/csr_original.py`、`experiments/run_csr_baseline.py`），但没有执行基线实验；`results/csr_original/` 中不应存在被引用的实验数值。

Original CSR 的数学规格与实现分别见 [CSR_MATHEMATICAL_SPEC.md](CSR_MATHEMATICAL_SPEC.md) 与 [ORIGINAL_CSR_IMPLEMENTATION_REPORT.md](ORIGINAL_CSR_IMPLEMENTATION_REPORT.md)。

---

## 7. 代码修改与测试

新增：
- `defenses/csr_original.py`（Original CSR）
- `defenses/detection.py`（可微/不可微检测分数）
- `attacks/cw.py`（CW-L2 margin）
- `attacks/adaptive_detection.py`（自适应 PGD）
- `configs/csr_original.yaml`
- `configs/phase3_final.yaml`
- `experiments/run_phase3_final_detection.py`
- `experiments/run_phase3_cw.py`
- `experiments/run_phase3_adaptive.py`
- `experiments/run_csr_baseline.py`
- `tests/test_csr_original.py`

测试：CSR 单测 **7/7** 通过；全工程在加入 CSR 测试前为 **76/76** 通过。按用户要求停止后，未再运行完整的 83 项测试套件；报告中不声称 83/83。

---

## 8. 失败案例与限制

1. **A2 误报严重**：Gaussian noise ≥0.03、rotation ≥5°、translation ≥2px 使检测器大量误报；检测器不能直接用于未对齐/噪声掌纹。
2. **A1 频带泛化差**：固定阈值对 low/high-band 攻击 TPR 很低；检测器是在全局攻击上校准的。
3. **CW 初始实现方向错误**：`l2 + c*max(margin,0)` 导致梯度消失；改为 `l2 - c*margin` 后 CompNet CW ASR=100%。配置和真实 ASR 已记录。
4. **自适应攻击未运行**：A4 脚本已实现但按用户要求停止；没有自适应 ASR/逃逸率数据。
5. **Original CSR 基线未运行**：实现和单测完成，但未执行 PGD/APGD/low-band 基线；没有防御恢复率、误校正率或时间数据。
6. **CSR 阈值**：r=40@224 换算到 128 为 22.857；τ 需掌纹校准。原文 τ=0.85 直接套用的结果未测。
7. **规模**：A1/A2 1000 张 eval，A3 500 张；A5 全量。结论方向可信，但绝对数值需全量复核。
8. Test 未使用。

**停止点：** 用户要求“跑完 CCNet CW 评估就停下”。A3 两模型均已完成；A4 与 Original CSR 基线未启动。
