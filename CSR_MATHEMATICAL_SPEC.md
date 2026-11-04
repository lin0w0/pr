# CSR_MATHEMATICAL_SPEC — Original CSR 数学与算法规格

来源：
- 论文：*Contrastive Spectral Rectification: Test-Time Defense towards Zero-shot Adversarial Robustness of CLIP*，arXiv:2601.19210（ICML 2026）。
- 官方实现：`https://github.com/Summu77/CSR`，本地克隆 `D:/code/research/csr_official`，commit `a5009cc`。
- 本规格以论文 Algorithm 1、Eq. (5)(6)(7) 与官方 `csr/defense/csr.py`、`csr_fast.py`、`csr/config.py` 为准。

本文件描述**原始 CSR**；不包含任何掌纹特定改进。掌纹适配只做输入尺寸换算与检测阈值校准，二者单独标注。

---

## 1. 符号

| 符号 | 含义 |
|---|---|
| `x ∈ [0,1]^{H×W}` | 输入图像（掌纹为单通道 128×128；CLIP 为 3×224×224） |
| `f(·)` | 冻结的图像编码器；本工程为 CompNet/CCNet `getFeatureCode` |
| `G_r(·)` | 半径为 r 的 Gaussian 低通滤波器 |
| `x_low = G_r(x)` | 低频锚点图像 |
| `sim(a,b)` | 余弦相似度 `a·b/(||a||₂||b||₂)` |
| `τ` | 检测阈值 |
| `N` | rectification 迭代步数 |
| `α` | 步长 |
| `ε` | 扰动预算（L∞） |
| `λ` | repulsion 权重 |

所有特征在 `sim` 前 L2 归一化。

---

## 2. 频谱锚点

Gaussian 低通在频域构造（官方实现使用 `fft2/fftshift`）：

```
F = FFT2(x)
F_shift = fftshift(F)
dist(u,v)² = (u - H/2)² + (v - W/2)²
M_r(u,v) = exp( -dist(u,v)² / (2 r²) )
x_low = real( IFFT2( ifftshift(F_shift * M_r) ) )
```

官方实现最后执行 `clamp(x_low, 0, 1)`。`r` 是**像素频率半径**，不是归一化 rho。

论文默认 `r = 40`（输入 224×224）。CLIP 归一化前的图像范围 [0,1]。

---

## 3. 检测分数（Spectral Consistency Gating）

```
C(x) = sim( f(x), f(x_low) )
     = f(x)·f(x_low) / (||f(x)||₂ ||f(x_low)||₂)
```

判定：

```
C(x) >= τ  -> benign, return x
C(x) <  τ  -> adversarial, enter rectification
```

不依赖身份标签、Gallery 或分类器。

---

## 4. Attraction / Repulsion

设当前 rectified 输入为 `x' = x + δ`。

Attraction：

```
L_att(δ) = sim( f(x + δ), f(x_low) )
```

Repulsion：

```
L_rep(δ) = - sim( f(x + δ), f(x) )
```

完整目标（论文 Eq. 6）：

```
L_rec(δ) = sim(f(x+δ), f(x_low)) - λ · sim(f(x+δ), f(x))
```

- `f(x_low)` 为低频身份锚点，由当前输入自身构造，不使用真实身份标签。
- `f(x)` 为输入原始特征（对抗样本特征或干净特征）。
- 论文正文/附录未给出 λ 的具体数值；官方 `csr.py`/`csr_fast.py` 直接计算 `sim_lpf - sim_adv`，即 **λ = 1**。本工程默认 λ=1，可配置，但不得把调 λ 当作“原文改进”。

---

## 5. 优化与约束

初始化：

```
δ = 0, x' = x, x* = x, L_best = -inf
```

每步：

```
L_rec = sim(f(x'), f(x_low)) - λ · sim(f(x'), f(x))
if L_rec > L_best:
    L_best = L_rec
    x* = x'
δ = clip(δ + α · sign(∇_{x'} L_rec), -ε, ε)
x' = x + δ
```

论文 Algorithm 1 用 `clip(δ+α sign(∇L_rec), -ε, ε)`。官方 PyTorch 实现在此之后还会：

```
x' = clamp(x + δ, 0, 1)
```

即同时满足 L∞ 预算与像素范围。本工程采用两者。

梯度只对输入 `x'` 求；模型参数冻结，不更新。

---

## 6. Greedy Selection

```
x* = x + argmax_{δ_t ∈ {δ_0, δ_1, ..., δ_{N-1}}} L_rec(δ_t)
```

- `δ_0 = 0` 也是候选；不一定返回最后一步。
- 论文 Algorithm 1 在 `t=1..N` 每步先评估再更新，因此候选为 `δ_0..δ_{N-1}`；官方 `csr_fast.py` 与之一致（最后一步不反传）。
- 官方 `csr.py` 在循环后额外评估一次 `δ_N`；本工程按论文 Algorithm 1 / `csr_fast.py` 实现，并在报告中注明该差异。

---

## 7. 推理流程

```
if C(x) >= τ:
    return x
else:
    rectify and return greedy-selected x*
```

- 只对检测为对抗的样本执行 rectification；干净样本直接通过。
- 检测阶段与校正阶段分开评价（检测指标与身份指标不能混为一个指标）。
- 不需要身份标签。

---

## 8. 停止条件

- 固定迭代次数 `N`；没有收敛阈值或提前停止。
- Greedy Selection 保存历史最优；最终返回 `x*`。

---

## 9. 原文默认超参数

| 参数 | 论文/官方默认 | 说明 |
|---|---|---|
| LPF 类型 | Gaussian | `filter_type="gaussian"` |
| 半径 r | 40 | 输入 224×224 像素频率半径 |
| 检测阈值 τ | 0.85 | CLIP 特征 |
| rectification 步数 N | 3 | 论文实现细节与消融 |
| 扰动预算 ε | 4/255 | L∞ |
| 步长 α | 2/255 | sign-PGD |
| λ | 未给出；官方代码 λ=1 | 本工程默认 1 |
| 优化器 | sign-PGD | 无 Adam |
| Greedy | 是 | 历史最大 L_rec |
| 身份标签 | 否 | 仅用图像特征 |

---

## 10. 掌纹适配（仅尺寸与阈值，不改算法）

1. **半径换算**：论文 r=40 对应 224 输入的归一化半径
   `rho = 40 / (224/√2) = 0.2525`。
   对 128×128 掌纹，保持相同归一化半径：
   `r_128 = 0.2525 * (128/√2) = 22.86` 像素。
   本工程默认 `r=22.857`。这不是调参改进，而是输入尺寸换算；同时会在实验中报告直接使用 `r=40` 的结果作为尺寸不换算的对照。
2. **检测阈值 τ**：CLIP 特征与掌纹特征分布不同，原文 τ=0.85 不能直接视为最优。Task D 允许在 validation 上确定参数：
   - 报告原文 τ=0.85 的直接结果；
   - 另用 validation calibration 身份（0–99）的 clean 分数校准 τ 到目标 FPR（1%/5%），作为主要基线设置；
   - Test 不参与阈值选择。
3. **其余参数保持原文**：N=3，ε=4/255，α=2/255，λ=1，Gaussian LPF，sign-PGD，Greedy Selection。

---

## 11. 与“简化 CSR”的区别（不得混用）

本规格不使用：
- Gallery prototype logits 或 softmax 身份损失；
- 温度缩放；
- 真实身份标签；
- 仅用低频特征替代 `f(x_low)` 的额外设计；
- 多频带/中频带 CSR。

CSR 的目标只包含两个余弦项；校正不依赖分类器或身份标签。

---

## 12. 官方代码对应位置

| 内容 | 文件 |
|---|---|
| LPF / detection / purify | `csr/defense/csr.py`、`csr/defense/csr_fast.py` |
| 默认参数 | `csr/config.py`、`configs/default.yaml` |
| 论文 Algorithm 1 | PDF 第 5 页 |
| 题目/作者/arXiv | README、PDF 首页 |

---

## 13. 未解决/需在报告中说明

- 论文未明确 λ 数值；官方代码等价 λ=1。
- 官方 `csr.py` 与 `csr_fast.py` 在是否评估 `δ_N` 上略有差异；本工程采用 Algorithm 1 / FastCSR。
- 掌纹 128 输入的 r 换算与 τ 校准属于必要适配，需与原文 CLIP 结果分开表述。
- CSR 的检测器与 Phase 2/3 的频谱一致性检测器同源，但阈值/半径设置需分别报告；不得把检测 AUC 当作校正恢复率。
