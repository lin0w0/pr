# PHASE 1 REPORT — 基础接口、数据划分、适配器与评价

完成日期：2026-11-03  
环境：Python 3.10.22 / torch 2.5.1+cu121 / RTX 4070 Laptop 8GB  
状态：**通过验收**（24/24 单元测试，全量 600 人 Rank-1=100%，环境自检全通过）

---

## 1. 本阶段新增文件

```
configs/
  dataset.yaml                  # 数据根目录、session、协议 A 划分、温度尺度
  compnet.yaml                  # 官方模块路径、checkpoint SHA256、batch
  ccnet.yaml                    # 含 36 个已知忽略键的显式声明

utils/
  __init__.py
  paths.py                      # 项目根路径解析（禁止硬编码绝对路径）
  config.py                     # YAML 加载 + 配置哈希
  seed.py                       # 随机种子 / 确定性设置

datasets/
  __init__.py
  preprocessing.py              # 官方预处理的可微复刻 + 原始 [0,1] 图像读取
  tongji_dataset.py             # Sample / TongjiDataset / split CSV 读取
  dataset_split.py              # 扫描、协议 A 划分、持久化、哈希校验、泄漏检查
  gallery.py                    # Gallery 特征、prototype、GalleryClassifier

models/
  __init__.py                   # build_adapter 工厂
  model_adapter.py              # 统一冻结适配器；严格加载；显存重试
  compnet_adapter.py
  ccnet_adapter.py

evaluation/
  __init__.py
  identification_metrics.py     # Rank-1/5
  verification_metrics.py       # EER / TAR@FAR / 阈值指标

tests/
  __init__.py
  test_preprocessing.py         # 与官方流程逐位对齐、梯度、零像素、尺寸保护
  test_dataset_split.py         # 标签映射、协议 A 计数、无泄漏、存取往返
  test_adapters.py              # 接口、logits 关系、输入梯度、参数冻结
  test_gallery_metrics.py       # prototype、classifier、Rank-k、EER

scripts/
  build_splits.py               # 生成并校验划分
  check_environment.py          # 环境/数据/权重/模型自检
  verify_phase1.py              # Phase 1 验收（全量或 --max-ids 小规模）

splits/tongji_A/
  gallery.csv                   # 6000 张（session1，10/人）
  validation.csv                # 3000 张（session2，1–5/人）
  test.csv                      # 3000 张（session2，6–10/人）
  split_meta.json               # 协议、计数、逐文件 SHA256、生成时间
```

Phase 0 审计脚本仍在 `scripts/audit_*.py`，不属于本阶段交付物。

## 2. 修改文件

- `.gitignore`：新增 `__pycache__/`、`results/`、`cache/` 等；`splits/` 保留纳入版本管理。
- `PROJECT_AUDIT.md`：补充已确认决策、依赖安装、Windows 显存风险 R16。
- 环境：安装 `openpyxl 3.1.5`、`autoattack 0.1`（GitHub 源）。

**未修改** `official_project/` 下任何官方代码；未修改 `data/`、`weights/`。

## 3. 核心实现说明

### 3.1 预处理（`datasets/preprocessing.py`）

与官方 `MyDataset(train=False)` 完全一致：

```
PIL convert("L") → 128×128 断言 → uint8/255 → 每图 z-score（pixels>0，无偏 std，eps=1e-6）
```

- 全部实现为可微算子（`torch.where`），对抗攻击可以反传到原始像素。
- 与官方流程在 10 张真实图像上最大误差 **1.19e-6**（float32 舍入级）。
- 全零/单非零像素样本会显式报错，不静默产生 NaN。

### 3.2 数据划分（协议 A，已确认）

| 划分 | 来源 | 每人张数 | 总数 | 用途 |
|---|---|---|---|---|
| Gallery | session1 | 10 | 6000 | 注册模板 |
| Validation | session2，第 1–5 张 | 5 | 3000 | **仅**阈值/超参数选择 |
| Test | session2，第 6–10 张 | 5 | 3000 | **仅**最终评估 |

- 三个划分图像零重叠；覆盖全部 600 个身份。
- 路径使用相对项目根目录的 POSIX 形式；`split_meta.json` 记录每个 CSV 的 SHA256。
- `TongjiDataset` 只返回原始 `[0,1]` 张量与标签，不做归一化。

### 3.3 模型适配器（`models/`）

统一契约（CompNet / CCNet 相同）：

| 方法 | 输出 |
|---|---|
| `preprocess_raw(x)` | 官方可微预处理，`[B,1,128,128]` |
| `extract_features(x)` | `[B,D]`，L2 归一化（D=512 / 2048） |
| `get_model_logits(x)` | `[B,600]`，已验证 `== 30·cos(feature, ArcFace weight)` |
| `get_similarity(f1,f2)` | 余弦相似度矩阵 |

- 全程 `eval()` + `requires_grad_(False)`；24 项测试中已验证反向传播后**参数完全不变、无 `.grad`**。
- checkpoint 加载前校验 SHA256；过滤 `module.` 前缀。
- CCNet 仅忽略配置中声明的 36 个 `att_*` 键，**且断言数量必须为 36**；其余 59 个张量严格加载，任何缺失/多余键立即报错。
- 官方模型文件通过绝对路径 `importlib` 加载，不会被本地 `models` 包遮蔽。

### 3.4 Gallery 与统一身份分类器

严格按规范：

```python
z_i = normalize(f(x_i))                 # 每张注册图先归一化
prototype_j = normalize(mean_i z_i)     # 再平均、再归一化
scores = normalize(query) @ prototypes.T
logits = temperature_scale * scores      # 默认 30，与 ArcFace 数值尺度一致
```

`temperature_scale` 写在 `configs/dataset.yaml::gallery.temperature_scale`，可配置。选 30 的理由：官方 ArcFace logits 就是 `30·cos`，攻击损失在相同数值尺度上更稳定；不是照搬 CLIP 的参数。

### 3.5 指标

- Rank-1 / Rank-5：多列相似度取 top-k，与 gallery 标签比较。
- EER：在**每个真实 genuine 分数**上精确计算 TPR/FPR（非分箱近似），取 `|FAR−FRR|` 最小点。
- TAR@FAR=1% / 0.1%。
- 验证协议：全部 probe × gallery **图像对**（3000×6000 = 1800 万对），与 Phase 0 审计口径一致。

## 4. 运行命令

```bash
PY="D:/code/research/envs/pr/python.exe"

# 生成/校验划分（已生成，可重复执行）
$PY scripts/build_splits.py

# 单元测试（24 项）
$PY -m unittest discover -s tests -v

# 环境 / 数据 / 权重 / 模型自检
$PY scripts/check_environment.py

# Phase 1 全量验收（600 人）
$PY scripts/verify_phase1.py --model all

# 小规模冒烟（100 人，符合“先小后大”）
$PY scripts/verify_phase1.py --model all --max-ids 100 \
    --out results/phase1_smoke_100ids.json
```

## 5. 实际测试结果

### 5.1 单元测试

```
Ran 24 tests in 1.749s
OK
```

覆盖：预处理逐位对齐、零像素、尺寸保护、梯度；标签映射、协议计数、无泄漏、存取代往返；适配器形状/归一化/logits 关系/输入梯度/参数冻结；prototype、classifier、Rank-k、EER、pairwise 计数。

### 5.2 环境自检

`scripts/check_environment.py`：**ALL CHECKS PASSED**（连续运行 2 次），包括 checkpoint SHA256 与两个 adapter 的随机输入形状/归一化检查。

### 5.3 全量 600 人（`results/phase1_baseline.json`）

| 模型 | 划分 | Rank-1 | Rank-5 | EER | TAR@FAR=1% | ArcFace Acc |
|---|---|---|---|---|---|---|
| CompNet | Validation | 100.000% | 100.000% | 0.0136% | 100% | 100.000% |
| CompNet | Test | 100.000% | 100.000% | 0.0223% | 100% | 100.000% |
| CCNet | Validation | 100.000% | 100.000% | 0.0000% | 100% | 100.000% |
| CCNet | Test | 100.000% | 100.000% | 0.0000% | 100% | 100.000% |

耗时（全量）：
- CompNet：gallery 6.39 s；每个划分特征提取 ≈3.1 s；验证打分 ≈0.33 s。
- CCNet：gallery 12.23 s；每个划分特征提取 ≈6.1 s；验证打分 ≈0.40 s。

### 5.4 小规模 100 人（`results/phase1_smoke_100ids.json`）

| 模型 | Validation Rank-1 | Test Rank-1 | Validation EER | Test EER |
|---|---|---|---|---|
| CompNet | 100% | 100% | 0.0195% | 0.0004% |
| CCNet | 100% | 100% | 0.0000% | 0.0000% |

“先小后大”的入口已验证可用。

## 6. 尚未解决的问题 / 限制

1. **验证集与测试集同 session、同身份**（协议 A 的固有性质）。验证集只用于阈值与超参数，图像与测试集零重叠；若审稿要求身份隔离，可切换协议 B（配置已支持 `split.protocol` 但尚未实现 B 的选择逻辑）。
2. **`temperature_scale=30` 是工程选择**，不是从 CSR 原文继承的最优值；Phase 4/9 会把它纳入敏感性分析（若需要）。
3. **Gallery 特征缓存函数已实现但尚未接入** `verify_phase1.py`；Phase 2 起会接入，避免每个实验重复提取 6000 张注册图。
4. **Windows 偶发单次大分配 OOM**：已加重试（3 次，2 s 间隔）。仍建议长时间实验前确认 `nvidia-smi` 空闲显存 > 6 GiB。
5. **autoattack 已安装但未使用**：Phase 3 会先做接口适配与交叉验证；自实现 APGD 仍是主路径。
6. 指标目前只覆盖 clean 的 Rank/EER；ASR、Recovery、检测 AUC 等将在 Phase 3–5 补充。

## 7. 下一阶段（Phase 2）

实现并测试：

- `spectral/fft_utils.py`：rfft2/irfft2、频率坐标、往返一致性。
- `spectral/gaussian_filter.py`：径向 Gaussian LPF（sigma 越小越强）。
- `spectral/frequency_masks.py`：低/中/高互斥且覆盖 [0,1] 的径向掩码。
- `spectral/spectral_consistency.py`：`C(x,σ)=cos(z(x), z(LPF(x)))`。
- `spectral/sgm.py`：幅值变量反传、2D 热力图、径向 profile、频带能量。
- `spectral/band_projection.py`：Dykstra 交替投影 + 残差/频带外能量报告。

先用 100 人真实数据验证，再全量。
