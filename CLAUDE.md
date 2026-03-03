# GNSS-RO 大气剖面反演系统 — 开发指南

## 项目架构

```
ro_retrieval/          # 核心包
  config.py            # 全局配置 (路径/超参数/设备)
  model/unet.py        # ConditionalUNet1D (legacy) + EnhancedConditionalUNet1D
  model/diffusion.py   # DiffusionSchedule, ddpm_sample(), ddim_sample()
  training/trainer.py  # Trainer 类
  evaluation/metrics.py # EvaluationReport, RMSE/Bias/CC
  inference/predict.py # run_inference(), run_inference_ddim()
  app/streamlit_app.py # Streamlit 前端
  data/                # 数据处理子包
    dataset.py         # RODataset, ROMultiVarDataset
    process_enhanced.py # 预处理流水线 (QC + 插值 + 标准化)
    fy3d_process.py    # FY-3D GNOS 数据处理

src/                   # 入口脚本
  process_data.py      # 数据预处理 (--source cosmic/fy3d)
  train.py             # 训练 (--mode single/multi --model legacy/enhanced)
  evaluate.py          # 批量评估 (--sampler ddim/ddpm)
  run_pipeline.py      # 端到端流水线 (--all)
  ablation_study.py    # 消融实验
  compare_cdaac.py     # CDAAC 产品对比
```

## 数据格式

- 输入 X: 弯曲角剖面 (N, 301)，log10(|BA|+1e-6) 变换后 Z-Score 标准化
- 标签 Y: [温度, 气压, 湿度] (N, 3, 301)，Z-Score 标准化
- 高度网格: 0–60 km, 301 点 (np.linspace)
- 划分: train 70% / val 15% / test 15%
- 存储: Data/Processed/ 下 train_x.npy, train_y.npy 等

## 模型关键超参数

- DDPM: T=1000, β ∈ [1e-4, 0.02]
- DDIM: 50 步加速采样, η=0.0
- U-Net: base_dim=64, 交叉注意力, 3 级编解码
- 训练: batch=64, lr=1e-4, patience=20 (early stopping)
- 模型参数量: 1,115,651

## 训练结果

| 项目 | 详情 |
|------|------|
| 模型 | EnhancedConditionalUNet1D (多变量, 3 通道) |
| 数据源 | COSMIC-2 atmPrf (Day 001) |
| 样本数 | 1830 (QC 通过率 55.3%, 原始 3310) |
| 划分 | train 1281 / val 274 / test 275 |
| 最优 val_loss | 0.013806 (Epoch 64) |
| 早停轮次 | Epoch 84 (patience=20) |
| 训练时长 | ~0.8 分钟 (GPU) |
| 权重文件 | `enhanced_ro_diffusion_best.pth` (4.3 MB) |

### 数据统计 (标准化前)

| 变量 | 均值 | 标准差 | 备注 |
|------|------|--------|------|
| 温度 | 237.6 K | 27.9 K | 正常范围 |
| 气压 | 132.4 hPa | 241.7 hPa | 正常范围 |
| 湿度 | 0 | 0 | 全零 (无 wetPf2 数据) |

## 数据来源

- COSMIC-2: atmPrf (弯曲角) + wetPf2 (温/压/湿)，CDAAC netCDF 格式
- **FY-3D GNOS L2 ATP 数据** (已处理完成)
  - 时间范围: 2025-01-01 ~ 2025-01-31 (整月)
  - 文件数量: 15,317 个 L2 ATP 文件
  - 数据量: 2.2 GB
  - 存储位置: `utils/atp/`
  - 处理后数据: `Data/Processed_ATP/`
  - 成功处理: 13,088 个廓线 (通过率 85.4%)
  - 训练集: 9,161 / 验证集: 1,963 / 测试集: 1,964
- FY-3D GNOS L1 数据 (原始观测，已弃用)
  - 文件数量: 30,794 个
  - 存储位置: `utils/down/`
  - 问题: L1 数据需要自行推导弯曲角，技术难度大

## FY-3D GNOS ATP 数据处理 (已完成)

### L2 ATP 数据格式

FY-3D GNOS L2 ATP (大气廓线) 数据是已处理的产品，包含：
- **弯曲角**: `Opt_Bend_ang` (优化后的弯曲角, rad)
- **冲击参数**: `Opt_Impact_parm` (地心距离, km)
- **温度**: `Temp` (K)
- **气压**: `Pres` (mb)
- **密度**: `Dens` (g/m³)
- **高度**: `MSL_alt` (海拔高度, km)

### 数据处理流程

```bash
# 处理 ATP 数据
python ro_retrieval/data/atp_process.py --atp-dir utils/atp --output-dir Data/Processed_ATP

# 测试处理 (前 100 个文件)
python ro_retrieval/data/atp_process.py --atp-dir utils/atp --output-dir Data/Processed_ATP --max-files 100
```

### 处理结果

| 项目 | 详情 |
|------|------|
| 原始文件数 | 15,317 个 |
| 成功处理 | 13,088 个廓线 |
| 失败 | 2,229 个 |
| 通过率 | 85.4% |
| 训练集 | 9,161 个 |
| 验证集 | 1,963 个 |
| 测试集 | 1,964 个 |

### 数据统计 (标准化前)

| 变量 | 均值 | 标准差 | 备注 |
|------|------|--------|------|
| 弯曲角 (log10) | -3.582 | 1.063 | log10 变换后 |
| 温度 | 240.8 K | 22.8 K | 正常范围 |
| 气压 | 101.4 mb | 193.5 mb | 正常范围 |

**注意**: ATP 数据不包含湿度信息，因此输出为 2 通道（温度+气压）。

### 关键技术点

1. **高度坐标转换**: 冲击参数 (Impact_parm) 是地心距离，需转换为海拔高度
   ```python
   msl_alt = impact_parm - curv  # curv 为曲率半径
   ```

2. **质量控制**:
   - qc=100 的文件才处理
   - 物理合理性检查（温度 150-350 K，气压 0.01-1100 mb）
   - 气压单调递减检查

3. **插值**: 线性插值到 0-60 km 标准高度网格 (301 点)

## FY-3D GNOS L1 数据分析 (进行中)

### 数据结构

FY-3D GNOS L1 级 netCDF 文件与 COSMIC atmPrf **完全不同**。L1 数据为原始观测量，不含弯曲角剖面，需自行推导。

#### 文件分类

| 前缀 | 数量 | 采样率 | 切线高度范围 | 用途 |
|------|------|--------|-------------|------|
| AEG | 15,580 | 50 Hz (dt=0.02s) | -275 ~ 113 km | **大气层掩星** (开环追踪) |
| IEG | 14,288 | 1 Hz (dt=1.0s) | 78 ~ 843 km | 电离层掩星 (闭环追踪) |
| POD | 926 | — | — | 精密定轨 (.SP3 格式, 非 netCDF) |

> **结论**: 只有 **AEG 文件** 包含对流层/平流层信号，可用于大气反演。

#### netCDF 变量

| 变量 | 单位 | 说明 |
|------|------|------|
| `time` | s | 掩星开始后的秒数 |
| `exL1` | m | L1 频率附加相位 (excess phase) |
| `exL2` | m | L2 频率附加相位 |
| `caL1Snr`, `caL2Snr`, `pL2Snr` | V/V | 信噪比 |
| `xLeo`, `yLeo`, `zLeo` | km | LEO 卫星位置 (ECI 坐标系) |
| `xdLeo`, `ydLeo`, `zdLeo` | km/s | LEO 卫星速度 |
| `xGnss`, `yGnss`, `zGnss` | km | GNSS 卫星位置 |
| `xdGnss`, `ydGnss`, `zdGnss` | km/s | GNSS 卫星速度 |

#### 全局属性

| 属性 | 示例值 | 说明 |
|------|--------|------|
| `setting` | 0 或 1 | 0=上升掩星, 1=下降掩星 |
| `gnssName` | GPS | GNSS 系统 |
| `coordinate` | ECI | 坐标系 |
| `occsatId` | 30 | GNSS 卫星 PRN 号 |

### 弯曲角推导思路

L1 数据需要经过以下处理才能得到弯曲角剖面:

```
excess phase (L1, L2)
  → 电离层校正: Lc = (f1²·L1 - f2²·L2) / (f1² - f2²)
  → Doppler: dLc/dt
  → 几何光学反演: Doppler + 轨道数据 → 冲击参数 a + 弯曲角 α
  → 插值到标准高度网格
```

### 当前技术障碍

**弯曲角推导结果异常**: 使用几何光学方法 (Newton 迭代求解冲击参数) 得到的弯曲角为 **负值** (~-480 mrad)，与预期正值 (~5-20 mrad at 30 km) 严重不符。

可能原因:
1. **excess phase 的符号约定不明确** — NSMC L1 数据中 `exL1` 的定义可能与标准 RO 处理软件不同
2. **Doppler 符号/公式错误** — D_obs = D_sl + dLc/dt 中的加减号需确认 (取决于 excess phase 的定义)
3. **坐标系问题** — 数据使用 ECI 坐标系，LEO 和 GNSS 速度在掩星平面的投影方向需仔细处理
4. **开环追踪数据的特殊性** — AEG 文件使用 50 Hz 开环追踪，可能需要特殊的 Canonical Transform / Phase Matching 处理

### 后续计划

1. **方案 A**: 修正弯曲角推导公式 (调试符号约定、验证 D_sl 计算)
2. **方案 B**: 寻找 FY-3D L2 级数据 (已含弯曲角剖面) 替代 L1 级
3. **标签数据**: 无论采用哪种方案，均需 ERA5 再分析数据作为温度/气压/湿度真值标签
4. **时空匹配**: 根据掩星事件的经纬度/时间，匹配最近的 ERA5 格点

## FY-3D ATP 模型训练 (进行中)

### 训练配置

```bash
python src/train.py --mode multi --model enhanced \
  --data_dir Data/Processed_ATP --epochs 100 --batch_size 64 --patience 20
```

| 项目 | 详情 |
|------|------|
| 模型 | EnhancedConditionalUNet1D (交叉注意力) |
| 参数量 | 1,115,330 |
| 数据集 | FY-3D GNOS L2 ATP (2025-01) |
| 训练集 | 9,161 个廓线 |
| 验证集 | 1,963 个廓线 |
| 测试集 | 1,964 个廓线 |
| 输出通道 | 2 (温度 + 气压) |
| 设备 | Tesla V100-PCIE-32GB |
| Batch Size | 64 |
| Learning Rate | 1e-4 |
| Early Stopping | patience=20 |

### 训练状态

训练已启动，日志保存在 `training.log`。

## 已知问题

- ~~湿度通道全零~~: ATP 数据不包含湿度，改为 2 通道输出（温度+气压）
- ~~训练数据较少~~: 已使用 FY-3D ATP 数据（13,088 样本），数据量充足
- ~~FY-3D GNOS L1 数据弯曲角推导失败~~: 改用 L2 ATP 数据，已包含处理好的弯曲角和大气廓线
- ~~FY-3D 数据缺少标签~~: L2 ATP 数据自带温度/气压标签
- **推理结果质量差**: 模型在低时间步（t=0-200）性能不足，导致推理产生物理上不可能的值（详见下方）

## 开发约定

- 每次修改后同步更新 readme.md
- Python ≥ 3.9, PyTorch ≥ 2.0
- 使用中文交流
