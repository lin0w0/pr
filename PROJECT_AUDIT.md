# PROJECT_AUDIT — PalmPrint-CSR Phase 0 审计报告

审计日期：2026-11-03  
审计对象：`D:\code\palmprint_csr`  
审计方式：逐文件阅读 + 真实 checkpoint 加载 + 真实同济 ROI 数据前向/反向验证。所有结论均有可复现脚本支撑，未使用任何假设。

---

## 0. 结论摘要

| 项目 | 状态 | 依据 |
|---|---|---|
| CompNet 网络定义 | ✅ 可用 | 官方代码原样加载，strict=True 无缺失/多余键 |
| CompNet checkpoint | ✅ 可用 | SHA256 校验，5,301,137 参数，600 类 |
| CCNet 网络定义 | ✅ 可用（需过滤 36 个未使用键） | 59 个活跃参数 strict=True 加载成功 |
| CCNet checkpoint | ✅ 可用 | 过滤 `att_*` 后 strict 加载；全量 Rank-1=100% 实证有效 |
| 同济 ROI 数据 | ✅ 完整 | 2 sessions × 6000 张，600 人 × 10 张，128×128 灰度，无损坏/重复 |
| 标签映射 | ✅ 明确 | `label = (int(filename[:5]) - 1) // 10 ∈ [0,599]` |
| 预处理 | ✅ 已精确复刻 | torch 实现与官方 PIL 流程最大误差 1.19e-6 |
| 输入梯度 | ✅ 可求 | CompNet / CCNet 均有限、非零 |
| 频谱幅值梯度（SGM 链路） | ✅ 可求 | rfft2 → 幅值变量 → irfft2 → 模型 → CE，梯度有限 |
| 模型冻结 | ✅ 已验证 | 反向传播后参数 `.grad` 数量为 0 |
| 现有攻击代码 | ❌ 不存在 | 仓库内 grep `pgd/attack/fgsm/apgd/cw/autoattack` 无结果 |
| 现有防御/CSR 代码 | ❌ 不存在 | 需从零实现 |
| 数据划分（gallery/val/test） | ⚠️ 需用户确认 | 官方只有 session1=注册、session2=查询；无独立验证集 |
| 依赖完整性 | ⚠️ 缺 4 个 | `pandas`、`openpyxl`、`autoattack`、`seaborn` 未安装 |

**总体判断：模型与数据接口无误，可以进入 Phase 1。唯一需要用户决策的是验证集划分协议与实验规模。**

---

## 1. 项目结构

```
D:\code\palmprint_csr\
├── .gitignore              # 忽略 data/ weights/ .vscode/
├── data\TongjI\ROI\
│   ├── session1\           # 6000 张 BMP（未纳入 git）
│   └── session2\           # 6000 张 BMP（未纳入 git）
├── weights\
│   ├── CompNet\compnet.pth # 21.2 MB
│   └── CCNet\ccnet.pth     # 254.3 MB
├── official_project\
│   ├── compnet\CompNet\    # 官方 CompNet 代码（含 data/for_reference/genText_tongji.py）
│   └── CCNet\             # 官方 CCNet 代码（train/inference/getEER/getGI/loss/utils）
└── scripts\                # 本次审计脚本（新增）
    ├── audit_probe.py
    ├── audit_probe2.py
    ├── audit_baseline.py
    ├── audit_probe_result.json
    ├── audit_probe2_result.json
    └── audit_baseline_result.json
```

项目根目录原本**没有任何自有代码**，只有官方参考实现、数据与权重。因此 CSR 工程需要新建（官方代码保持不动，作为原模型实现的唯一依据）。

> 说明：本机另存在一个不在本仓库内的历史工程 `D:\code\Palm-CSR`。本次审计**没有复用其代码**；仅用它作为“CCNet 多出 36 个注意力张量”这一已知线索的旁证，随后在本仓库用官方 `ccnet.py` + 官方 `inference.py` 的 `strict=False` 行为 + 真实数据全量实验独立验证。是否在未来复用该工程由用户决定（见 §12 问题 3）。

---

## 2. 运行环境

| 项 | 值 |
|---|---|
| Python | 3.10.22（conda env `pr`，路径 `D:\code\research\envs\pr\python.exe`） |
| PyTorch | 2.5.1+cu121 |
| CUDA | 12.1，`torch.cuda.is_available()=True` |
| GPU | NVIDIA GeForce RTX 4070 Laptop，显存 **8.0 GiB 可用**（8.6 GB 标称） |
| NumPy | 2.2.6 |
| 已安装 | scipy 1.15.3、scikit-learn 1.7.2、matplotlib 3.10.9、opencv 5.0.0、h5py 3.16.0、PyYAML 6.0.3、tqdm 4.70.1 |
| 2026-11-03 新增 | **openpyxl 3.1.5**、**autoattack 0.1**（GitHub `fra31/auto-attack`，PyPI 无该包） |
| 仍缺失（不影响当前阶段） | pandas、seaborn |

> 官方 `requirements.txt` 固定了 torch 1.7/1.10 时代的旧版本（如 matplotlib==3.5.3），**不要**在当前环境执行 `pip install -r`，否则会破坏 `pr` 环境。现有 torch 2.5.1 已验证可正常加载与推理。

---

## 3. 数据集

### 3.1 实际结构

```
data/TongjI/ROI/session1/00001.bmp ... 06000.bmp   (6000)
data/TongjI/ROI/session2/00001.bmp ... 06000.bmp   (6000)
```

- 格式：BMP，灰度（mode `L`），**全部为 128×128**。
- 规模：600 个身份 × 每身份 10 张 × 2 sessions。
- 文件名即样本 ID：`00001`–`06000`，两个 session 文件名完全一致。
- 标签映射（由官方 `genText_tongji.py` 确定并经验证）：
  `label = (int(filename[:5]) - 1) // 10`，即 `00001–00010 → 0`，…，`05991–06000 → 599`。
- 像素：uint8，原始采样范围约 `[0.118, 0.973]`（归一化到 [0,1] 后）；**图像没有纯黑背景**（整图 `pixel > 0` 恒成立，采样最小值为 30/255），因此官方 `NormSingleROI` 中 `tensor > 0` 掩码实际覆盖整幅图像。
- 完整性：12000 张全部可解码、尺寸/模式一致；session 内无内容重复；**跨 session 无完全相同的图像**（MD5 交集为 0）。

### 3.2 现有划分

官方 `data/train_Tongji.txt`（6000 行）= session1 全部；`data/test_Tongji.txt`（6000 行）= session2 全部。  
即官方协议：**注册 Gallery = session1 每人 10 张；查询 Probe = session2 每人 10 张**。

官方 txt 的路径问题（不影响新工程，但记录）：
- CompNet 的 `data/train.txt/test.txt` 是 Linux 绝对路径 `/home/sunny/datasets/...`，Windows 下不可用。
- CCNet 的 `data/train_Tongji.txt` 使用 `./Tongji/session1/...`，需以 `data/` 为工作目录才可用。
- 新工程将**自行生成划分清单**（相对路径 + 标签），不修改官方 txt。

### 3.3 基线复现（全量 600 人，session1 → session2）

| 模型 | Rank-1 | Rank-5 | EER | TAR@FAR=1% | TAR@FAR=0.1% |
|---|---|---|---|---|---|
| CompNet | **100.000%** | 100.000% | **0.0175%** | 100% | 100% |
| CCNet | **100.000%** | 100.000% | **<0.001%** | 100% | 100% |

- Gallery 模板：同一身份 10 张注册特征 L2 归一化后取均值，再 L2 归一化。
- 验证 EER：6000×6000 真实图对（60000 genuine / 35,940,000 impostor）。
- 特征提取耗时：CompNet 12000 张 10.1 s；CCNet 12000 张 22.6 s（batch 256 / 64）。
- 结论：两个权重在同济上训练充分，clean 场景无余量（100%），后续 ASR 分母即为全部 6000 张查询图。

---

## 4. CompNet 接口（`official_project/compnet/CompNet/models/compnet.py`）

```python
from models.compnet import compnet
net = compnet(num_classes=600)
net.load_state_dict(torch.load("weights/CompNet/compnet.pth", weights_only=True), strict=True)
net.eval()
```

| 项 | 实测值 |
|---|---|
| 输入 | `[B, 1, 128, 128]`，float32 |
| 特征接口 | `getFeatureCode(x) -> [B, 512]`，已 L2 归一化（`torch.norm` 后） |
| 分类接口 | `forward(x, None) -> [B, 600]`（eval 模式下 `y` 必须为 `None`） |
| logits 关系 | `logits == 30 * cosine(getFeatureCode(x), arclayer.weight)`，实测最大误差 **0.0** |
| 参数 | 5,301,137（约 5.3M） |
| 分类器 | `arclayer.weight` shape `[600, 512]` |
| 随机性 | eval 模式下 dropout 关闭，forward 确定 |
| 输入梯度 | 有限、非零（实测 `absmax≈0.027`） |
| 参数梯度 | 冻结后为 0 |

> `forward` 的训练模式会强制 `label is not None`（ArcFace margin），CSR 全流程保持 `eval()`。

---

## 5. CCNet 接口（`official_project/CCNet/models/ccnet.py`）

```python
from models.ccnet import ccnet
net = ccnet(num_classes=600, weight=0.8)   # weight=0.8 只影响竞争分支权重，不影响 state_dict 形状
net.load_state_dict(filtered_state_dict, strict=True)
net.eval()
```

### 5.1 checkpoint 与网络定义不一致（已解决，必须记录）

CCNet checkpoint 含 **95 个张量**，其中 **36 个** 是官方 `ccnet.py` 中不存在的：

```
cb{1,2,3}.att_{1,2}.{conv0, conv_spatial, conv1}.{weight,bias}
```

- 官方 `inference.py` 本身使用 `load_state_dict(..., strict=False)`，说明发布权重本就允许这些键存在。
- 去掉这 36 个键后，官方网络定义的 **59 个活跃参数全部存在且形状完全匹配**；`missing=[]`、`unexpected=[]`、`strict=True` 通过。
- 经验验证决定性地排除了“注意力分支参与前向”的可能：
  - 过滤后模型在 600 人全量数据上 Rank-1 = **100%**、EER < 0.001%；
  - 若注意力参与前向而被忽略，特征不可能在全量闭集上达到 100%。
- 结论：这 36 个键是该 checkpoint 的训练代码遗留、**不参与官方 forward**。新适配器只加载 59 个活跃参数，并在加载时断言“除这 36 个已知键外无任何多余/缺失键”。

### 5.2 接口

| 项 | 实测值 |
|---|---|
| 输入 | `[B, 1, 128, 128]`，float32 |
| 特征接口 | `getFeatureCode(x) -> [B, 2048]`，已 L2 归一化 |
| 分类接口 | `forward(x, None) -> (logits [B,600], fe [B,6144])` |
| logits 关系 | `logits == 30 * cosine(getFeatureCode(x), arclayer_.weight)`，实测最大误差 **0.0** |
| 特征选择 | 本项目统一使用 **getFeatureCode 的 2048 维**（官方 inference/verification 使用的就是它；forward 返回的 6144 维是 `cat(fc,fc1)`，不用于识别评价） |
| 参数 | 63,554,636（含 36 个未使用张量共 11,232 个参数；**活跃参数 63,543,404**） |
| 分类器 | `arclayer_.weight` shape `[600, 2048]` |
| 输入梯度 | 有限、非零（实测 `absmax≈0.048`） |
| 参数梯度 | 冻结后为 0 |

> 官方 `models/dataset.py` 第 27 行 `if c is not 1:` 在 Python 3.10/3.12 会触发 `SyntaxWarning`；新工程不使用官方 dataset，自行实现 loader。

---

## 6. 预处理（必须逐位复刻）

官方推理链（`MyDataset(train=False)`）：

1. `Image.open(path).convert("L")` —— 灰度
2. `T.Resize(128)` —— 由于原图已是 128×128，**是恒等变换**
3. `T.ToTensor()` —— uint8 → float32 / 255，范围 `[0,1]`
4. `NormSingleROI` —— 对该样本所有 `>0` 像素做 **per-image z-score**：
   - `m = mean(pixels>0)`，`s = std(pixels>0)`（PyTorch 默认无偏，除以 `n-1`）
   - `x_norm = (x - m) / (s + 1e-6)`
   - 由于本数据集无零像素，等价于对整幅图标准化。

复刻精度：torch 实现 vs 官方 PIL 流程，20 张图最大绝对误差 **1.19e-6**（float32 舍入级）。

对抗攻击/频谱实验的重要含义：
- 攻击预算 `0 <= x_adv <= 1` 定义在 **ToTensor 后的原始像素空间**；随后必须把同一套 per-image z-score 放进计算图（可微），否则梯度无效。
- 该 z-score 对全图仿射变化（亮度/对比度）具有不变性，会影响“全局亮度/对比度”边界实验的解释；需在报告中单独说明。
- 频谱实验应在**原始 [0,1] 图像**上做 FFT，再送入含 z-score 的可微预处理，保证与 clean 推理链路完全一致。

---

## 7. 频谱链路验证

| 检查 | 结果 |
|---|---|
| `irfft2(rfft2(x), s=(128,128))` 往返误差 | `1.79e-7` |
| 幅值 + 相位重建误差 | `1.79e-7` |
| `rfft2` 频谱形状 | `[B, 1, 128, 65]` |
| 归一化径向频率 `rho` 最大值 | `0.99999994`（定义正确覆盖 [0,1]） |
| 幅值变量反传 | CompNet `absmax≈0.077`、CCNet `absmax≈0.220`，全部有限、无 0 梯度异常 |
| 重建图 `requires_grad` | True |

使用的频率坐标：`fy = fftfreq(128)`（周期/像素，范围 [-0.5,0.5)），`fx = rfftfreq(128)`（[0,0.5]），  
`rho = sqrt(fy² + fx²) / sqrt(0.5² + 0.5²)`。Gaussian 与频带掩码都将在**未 shift 的 rfft2 排列**上构建，IFFT 使用同一排列。

---

## 8. 模型冻结验证

对两个模型执行：

```python
model.eval()
for p in model.parameters():
    p.requires_grad_(False)
```

在求输入梯度、频谱幅值梯度、以及完整攻击/CSR 目标反传后：

- 两个模型参数 `.grad` 数量均为 **0**；
- 输入/频谱梯度路径正常（未使用 `no_grad`/`inference_mode`）。

---

## 9. 现有攻击代码 / 缺失组件

### 9.1 已有

- 无任何 PGD / APGD / CW / FGSM / AutoAttack / 对抗训练代码。
- 无检测、CSR、频谱分析、评价指标代码。
- 无测试、配置文件、实验入口。
- `autoattack` 包未安装。

### 9.2 需要新建（对应目标工程结构）

Phase 1：`datasets/`（loader、split、gallery、preprocessing）、`models/`（adapter）、`evaluation/`（identification/verification metrics）。  
Phase 2：`spectral/`（fft_utils、gaussian_filter、frequency_masks、sgm、spectral_consistency、band_projection）。  
Phase 3：`attacks/`（pgd、apgd_ce、apgd_dlr、cw、targeted、feature、band_limited）。  
Phase 4：`defenses/`（detector、csr、greedy_selection、pipeline）。  
Phase 5：`experiments/`、`visualization/`。  
Phase 6：`scripts/run_all.py`、报告与图表。

### 9.3 依赖策略

- **APGD**：`autoattack` 未安装，且其标准接口针对普通分类 logits；我们的 logits 是 Gallery 余弦相似度构造的，虽然可适配，但为避免隐藏假设，建议**按 Croce & Hein 算法自行实现 APGD-CE / APGD-DLR**，并用单元测试验证步长、震荡检查、最佳候选保存。若用户希望安装 `autoattack` 做交叉验证，可另行安装。
- **表格输出**：CSV 用标准库即可；`final_summary.xlsx` 需要 `openpyxl`（或 pandas）。建议在 Phase 6 前安装 `openpyxl`，不安装 pandas 也能完成。
- **绘图**：matplotlib 已足够，不需要 seaborn。

---

## 10. 性能与显存实测（RTX 4070 Laptop 8GB）

| 模型 | 批量 | 前向+反向耗时 | 峰值显存 |
|---|---|---|---|
| CompNet | 128 | 56.9 ms | 0.47 GB |
| CCNet | 32 | 74.3 ms | 0.56 GB |
| CCNet | 64 | 137.1 ms | 0.78 GB |
| CCNet | 512 | **OOM** | — |

纯推理参考：batch=4 时 CompNet 3.3 ms、CCNet 14.2 ms/批（含每批固定开销）。

规划建议：
- 推理批量：CompNet 256、CCNet 64（保守），可尝试 CCNet 128。
- 攻击批量：CompNet 128、CCNet 32–64。
- 单次 PGD-50，6000 张查询图的粗略耗时：CompNet ≈ 2–3 分钟；CCNet ≈ 11–13 分钟。CW 因内层优化会明显更慢。
- 全量实验矩阵（多攻击 × 多防御 × 多消融 × 2 模型）预计需要数小时到十几小时，必须支持断点续跑与分批保存。

---

## 11. 风险清单

| # | 风险 | 影响 | 处理 |
|---|---|---|---|
| R1 | CCNet checkpoint 多 36 个键 | 若直接 strict 加载会失败；若误以为注意力参与前向会导致特征错误 | 已实证：过滤并 strict 加载，全量 Rank-1=100% |
| R2 | 每图 z-score 预处理 | 若攻击在已标准化张量上做、或漏掉预处理求导，攻击结论无效 | 预处理实现为可微模块，单测对齐官方流程（误差 1e-6） |
| R3 | 8GB 显存 | CCNet 大 batch OOM | 分批 + 显存上限配置；默认 batch 保守 |
| R4 | clean 准确率 100% | 无“干净错误”缓冲，任何攻击都会直接体现在 ASR 上；指标定义需严格 | ASR 分母固定为 6000；同时报告无条件错误率 |
| R5 | 验证集缺失 | 阈值/超参数若用测试集选择会污染结论 | 见 §12 问题 1，先定划分再实现 |
| R6 | Dykstra 频带投影收敛 | 像素裁剪会破坏频带约束 | 保存每次迭代残差、频带外能量比、是否达容差；未收敛不得标记为严格满足 |
| R7 | GPU 非确定性 | 复现性 | 固定 seed + `torch.backends.cudnn.deterministic`；记录 GPU 型号与版本 |
| R8 | 官方 data txt 路径失效 | 直接运行官方脚本失败 | 新工程自生成相对路径清单 |
| R9 | 官方 dataset 的 `is not 1` | SyntaxWarning | 新 loader 不复用该文件 |
| R10 | CCNet 特征 2048 vs 6144 | 混用会破坏可比性 | 统一 `getFeatureCode` 2048 维，写入 adapter |
| R11 | 全局亮度/对比度实验与 z-score 相消 | 可能测得“无影响”并被误读 | 单独报告，明确预处理不变性 |
| R12 | 频带定义与掩码覆盖 | 低/中/高不重叠且覆盖 [0,1] | 掩码构造单测：互斥、并集覆盖、边界 |
| R13 | 攻击预算浮点解析 | YAML 中 `1/255` 若当字符串会出错 | 配置解析显式转换为 float 并断言范围 |
| R14 | Clean 样本几乎没有零像素 | 对抗裁剪到 0 时 z-score 掩码会变化 | 与官方行为一致；记录被掩码像素数量 |
| R15 | 实验规模大 | 单次全流程耗时长 | 断点续跑、逐样本 CSV、结果去重、分批保存 |
| R16 | Windows/WDDM 偶发 206 MiB 单次分配 OOM | CCNet 加载可能失败，尽管显存充足 | 已实现 `_move_to_device` 三次重试（`gc` + `empty_cache` + 2s 等待）；Phase 1 连续 4 次加载/自检均通过。注：`expandable_segments` 在该 Windows 构建不支持，不要依赖 |

---

## 12. 已确认决策（2026-11-03）

用户已确认以下四项，后续实现以此为准：

1. **数据划分：方案 A**
   - Gallery：session1 全部 10 张/人（6000 张）
   - Validation：session2 第 1–5 张/人（3000 张），仅用于阈值与超参数选择
   - Test：session2 第 6–10 张/人（3000 张），仅用于最终评估
2. **实验规模：先小后大** —— 先用 100–150 个真实身份跑通全部流程，确认后再全量 600 人；代码始终支持全量。
3. **不复用** `D:\code\Palm-CSR`，在当前仓库从零实现并逐模块测试。
4. **依赖**：允许在 `pr` 环境安装 `openpyxl` 与 `autoattack`；APGD 仍以自实现为主，`autoattack` 用于交叉验证。

## 12b. 需要用户确认的问题（历史记录）

### 问题 1：验证集 / 测试集划分（必须先定）

官方只有 session1（注册）和 session2（查询）。CSR 需要独立验证集来选择 sigma、lambda、steps、epsilon_rec 和检测阈值，且测试集不能被使用。推荐方案：

**方案 A（推荐）**
- Gallery：session1 全部 10 张/人（6000 张）
- Validation：session2 第 01–05 张/人（3000 张）—— 只用于阈值与超参数选择
- Test：session2 第 06–10 张/人（3000 张）—— 只在最终评估使用
- 优点：同 session 分布、600 人全覆盖、图像完全不重叠、实现简单

**方案 B（身份隔离更严格）**
- Gallery：session1 全部 10 张/人
- Validation：固定 100 个身份（seed 固定）的 session2 第 01–05 张（500 张）
- Test：全部 600 人的 session2 第 06–10 张（3000 张）
- 优点：验证与测试身份不重叠，超参数选择更独立；缺点：阈值只在 100 人上校准

**方案 C**：先用方案 A 跑通全流程，最终报告前再用方案 B 复核阈值稳定性。

### 问题 2：实验规模

是否先按“小规模真实子集（如 100–150 人）跑通并验证全流程”，再启动 600 人全量？还是直接全量？

### 问题 3：是否复用本机历史工程

`D:\code\Palm-CSR` 不在本仓库内。建议**不复用**，在当前仓库从零实现并经单元测试验证；如需复用其攻击/CSR 实现，请明确告知。

### 问题 4：依赖安装

是否允许在 `pr` 环境安装 `openpyxl`（生成 xlsx）？`autoattack` 是否安装用于 APGD 交叉验证？（默认：只装 openpyxl，APGD 自行实现）

---

## 13. Phase 1 实现计划（待确认后开始）

Phase 1 交付物（只做基础接口，不写大规模实验）：

1. `configs/dataset.yaml`：数据根目录、session、split、seed。
2. `datasets/tongji_dataset.py`：返回原始 `[0,1]` 张量与标签；**不**做归一化。
3. `datasets/preprocessing.py`：可微的 `resize(no-op)+to_tensor+zscore`，单测对齐官方 1e-6。
4. `datasets/dataset_split.py`：按确认的方案生成并固化 `splits/*.csv`（含文件 hash）。
5. `datasets/gallery.py`：prototype 构建、1:N 打分、1:1 配对。
6. `models/model_adapter.py` + `compnet_adapter.py` + `ccnet_adapter.py`：统一 `extract_features -> [B,D]`、`normalize_features`、`get_similarity`、`get_logits`；CCNet 的 36 个已知键处理写入 adapter。
7. `evaluation/identification_metrics.py`、`verification_metrics.py`：Rank-1/5、EER、TAR@FAR。
8. `tests/`：接口与指标单测；模型冻结测试；clean 全量基线复现测试。
9. `scripts/check_environment.py`：环境/权重/数据自检。

**Phase 1 验收门槛**：两个 adapter 在确认的 test 划分上复现 Rank-1=100%，EER 与本次审计一致；所有测试通过；不修改任何官方代码；不写入 `data/` 或 `weights/`。

---

## 14. 审计脚本与原始结果

| 脚本 | 作用 | 结果文件 |
|---|---|---|
| `scripts/audit_probe.py` | 环境、数据、checkpoint 元信息、FFT 链路 | `audit_probe_result.json` |
| `scripts/audit_probe2.py` | 严格加载、预处理对齐、前向/反向、100 人 Rank-1 | `audit_probe2_result.json` |
| `scripts/audit_baseline.py` | 600 人全量 Rank-1/5、EER、耗时 | `audit_baseline_result.json` |

权重 SHA256：
- CompNet：`e27c86179dad2a69e9d9c7d4365eae5a31c3ccb4770a891a4ea3c1844d9f8a9a`
- CCNet：`288678c178f33675e0f15231dd0d1134131cde6f3960955dab5fce92a8cd3955`
