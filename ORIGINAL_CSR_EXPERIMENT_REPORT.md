# ORIGINAL_CSR_EXPERIMENT_REPORT

日期：2026-11-04  
实现：[defenses/csr_original.py](defenses/csr_original.py)  
规格：[CSR_MATHEMATICAL_SPEC.md](CSR_MATHEMATICAL_SPEC.md)  
结果目录：`results/csr_original/`

---

## 1. 实验设置

| 项 | 值 |
|---|---|
| 模型 | CompNet（512-D）、CCNet（2048-D），冻结 |
| 数据 | Validation 身份 100–199（500 张 clean）；Session1 Gallery 全部 600 |
| 攻击 | Global PGD（ε=2/255）、Global APGD（ε=2/255）、Low-band PGD |
| 防御 | No Defense、Gaussian LPF（与 CSR 同一 LPF）、Original CSR |
| CSR 参数 | r=22.857（40@224 的归一化换算），N=3，ε=4/255，α=2/255，λ=1 |
| CSR 阈值 | τ 由 validation calibration 身份 0–99 clean 分数校准到 FPR 5%；另报告原文 τ=0.85 |
| Test | 未使用 |

---

## 2. 评价定义

- **Clean Rank-1 / Rank-5**：对 clean 输入执行防御后，用 prototype 协议识别。
- **Adversarial Rank-1 / Rank-5**：对每种攻击输入执行防御后识别。
- **Recovery Rate**：No Defense 下 clean-correct 且被攻击致错的样本中，防御后恢复正确身份的比例。
- **Clean Mis-correction Rate**：clean-correct 的干净样本中，防御后识别错误的比例（CSR 检测门控误触发后校正导致）。
- **Feature Cosine**：`cos(f(defended), f(clean reference))`，逐样本平均。
- **L2 / L∞**：`defended - input` 的扰动范数。
- **Average Inference Time**：防御流程每张图耗时（不含模型加载）。
- 检测率/触发率单独报告；不与身份恢复率混为一个指标。

---

## 3. 结果

**状态：未运行。** 用户要求“跑完 CCNet CW 评估就停下”，因此 Original CSR 基线实验没有执行。

- 没有 Clean/Adversarial Rank-1、Recovery Rate、Mis-correction、Feature Cosine、L2/L∞ 或时间数据。
- `results/csr_original/` 中不应存在被引用的实验数值。
- 实现与实验脚本已就绪，可在后续阶段直接运行；不得用占位值或推测值替代。

---

## 4. 每步记录

CSR 对每个被检测样本保留每步：

- `sim_lpf`, `sim_adv`, `lrec`
- `selected`（Greedy 选中）
- `l2`, `linf`
- step

文件：

```
results/csr_original/{model}/csr_{tau_name}_{attack}_records.csv
```

---

## 5. 失败案例与限制

- CSR 只校正被检测为对抗的样本；检测逃逸的攻击不会被校正。
- τ 与 LPF 半径是掌纹适配参数；原文 τ=0.85 与换算半径的结果需与校准版分开看。
- 评估使用 100 身份；全量 600 身份的扩展由脚本支持。
- 未实现 middle-band/multi-band CSR；未做 CSR 特定改进。
- 未使用 test。

**停止点：** A3 CW 完成后停止；A4 自适应攻击与 Original CSR 基线均未运行。
