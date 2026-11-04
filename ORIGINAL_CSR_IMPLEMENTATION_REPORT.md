# ORIGINAL_CSR_IMPLEMENTATION_REPORT

日期：2026-11-04  
实现：`defenses/csr_original.py`  
规格：[CSR_MATHEMATICAL_SPEC.md](CSR_MATHEMATICAL_SPEC.md)  
配置：[configs/csr_original.yaml](configs/csr_original.yaml)

---

## 1. 实现范围

严格实现论文 Algorithm 1 / 官方 FastCSR 的核心流程：

1. Gaussian 低通 `x_low = G_r(x)`；
2. 检测分数 `C(x)=cos(f(x),f(x_low))`；
3. 仅对 `C(x)<τ` 的样本执行 rectification；
4. `L_rec = cos(f(x+δ),f(x_low)) - λ cos(f(x+δ),f(x))`；
5. sign-PGD，L∞ 预算与像素范围约束；
6. Greedy Selection（候选 δ₀…δ_{N-1}）；
7. 返回历史最优 `x*`。

未加入任何掌纹特定改进（无 Gallery logits、无身份标签、无多频带、无 middle-band CSR）。

---

## 2. 关键实现映射

| 论文/官方 | 本工程 |
|---|---|
| `G_r(x)` | `OriginalCSR.lowpass(x)`，`fft2→fftshift→Gaussian mask→ifftshift→ifft2`，`clamp(0,1)` |
| `f(·)` | `adapter.extract_features`（CompNet/CCNet `getFeatureCode`，含官方预处理） |
| `C(x)` | `score(x)`：`cos(f(x), f(lowpass(x)))` |
| `L_rec` | `sim_lpf - λ*sim_adv` |
| `δ update` | `δ = clip(δ + α sign(∇L_rec), -ε, ε)`；随后 `x'=clamp(x+δ,0,1)` |
| Greedy | 每个候选计算 `L_rec`，保留严格更大者；初始 `x` 也是候选 |
| 停止 | 固定 `N` 步，无收敛判定 |
| 标签 | 不使用 |

### 与官方代码的差异

- 官方 `csr.py` 在循环结束后额外评估一次 `δ_N`；官方 `csr_fast.py` 和论文 Algorithm 1 只评估 `δ_0…δ_{N-1}`。本实现采用后者。
- 论文 Eq. (6) 有 λ；论文与官方配置未给出 λ 数值，官方代码等价 λ=1。本实现默认 λ=1，可配置。
- 官方代码在 `δ` 更新后执行像素 `clamp(0,1)`；本实现同样执行，并记录有效 `δ`。

---

## 3. 掌纹适配

1. **LPF 半径**：论文 r=40（224 输入）。保持归一化频率等价：
   `r_128 = 40 * 128/224 = 22.857`。
   `configs/csr_original.yaml` 默认此值；报告中会同时说明原始 r=40 的直接套用。
2. **检测阈值 τ**：CLIP 的 τ=0.85 不能直接假定适合掌纹特征。基线实验在 validation calibration 身份（0–99）的 clean 分数上校准 τ 到 FPR 5%，并同时报告原文 τ=0.85 的结果。Test 不参与。
3. 其余参数保持原文：N=3，ε=4/255，α=2/255，λ=1，Gaussian LPF，sign-PGD，Greedy。

---

## 4. 单元测试

`tests/test_csr_original.py`，7 项：

1. 输入输出尺寸一致；
2. 像素范围 [0,1]；
3. L∞ 扰动预算；
4. 模型参数梯度关闭；
5. 输入梯度可传播；
6. Greedy 选择为记录中的最大 `L_rec`；
7. 同一输入重复运行结果一致（CSR 本身无随机性）。

运行：

```bash
D:/code/research/envs/pr/python.exe -m unittest tests.test_csr_original -v
Ran 7 tests ... OK
```

CSR 专项测试：**7/7 通过**。全工程在加入 CSR 测试前为 **76/76 通过**；按用户要求停止后未再运行完整 83 项套件，因此不声称 83/83。

---

## 5. 记录与输出

`OriginalCSR.rectify(..., return_records=True)` 记录每个样本每一步的：

- `sim_lpf`, `sim_adv`, `lrec`
- `selected`（是否 greedy 选中）
- `l2`, `linf`（当前扰动）
- `step`

`OriginalCSR.process(x, tau)` 返回：

- defended images
- detection mask
- 所有 flagged 样本的逐步骤记录
- detection scores

基线脚本 `experiments/run_csr_baseline.py` 将记录保存到：

```
results/csr_original/{model}/
  csr_{tau_name}_{attack}_records.csv
  per_sample.csv
  summary.csv
  meta.json
```

---

## 6. 已知限制

- 未实现官方 `csr.py` 的“额外评估 δ_N”变体；采用论文/FastCSR。
- λ 的原文数值缺失；默认 1。
- LPF 半径与 τ 的掌纹适配是必要工程选择，不是算法改进。
- 未实现多频带、middle-band CSR 或任何掌纹特定模块。
- 检测与校正分开评价；不把检测 AUC 当作恢复率。

---

## 7. 使用示例

```python
from defenses.csr_original import OriginalCSR

csr = OriginalCSR(adapter, radius=22.857, lam=1.0, steps=3,
                  eps=4/255, alpha=2/255)
defended, mask, records, scores = csr.process(x, tau=0.98)
```
