# GNSS-RO 大气剖面反演系统

基于 **条件扩散模型**（Conditional Diffusion Model）的 GNSS 无线电掩星（Radio Occultation）大气剖面反演系统。

---

## 项目概述

利用 COSMIC-2 弯曲角观测数据作为输入条件，通过条件扩散模型生成大气温度 / 气压 / 湿度剖面，以 ERA5 再分析数据和 CDAAC wetPf2 产品作为训练真值。

### 核心特性

| 特性 | 说明 |
|------|------|
| 条件扩散模型 | DDPM（1000 步）+ DDIM（50 步加速采样） |
| 增强版 U-Net | 交叉注意力机制 + 残差块 + 正弦时间嵌入 |
| 多变量输出 | 温度 + 气压 + 湿度三通道同时反演 |
| ERA5 时空匹配 | 根据掩星观测的经纬度/时间自动匹配 ERA5 格点 |
| 多级质量控制 | 穿透高度 / 有效点数 / 物理范围 / 递减率 / 单调性 |
| 系统化训练 | 验证集监控 + Early Stopping + 训练日志 |
| 完整评估 | RMSE / Bias / CC · 逐高度层分析 · JSON 报告 |
| 交互式前端 | Streamlit 可视化应用 |

---

## 项目结构

```
.
├── ro_retrieval/              # 核心 Python 包
│   ├── __init__.py            # 包入口, 版本号
│   ├── config.py              # 全局配置 (路径·超参数·设备)
│   ├── data/                  # 数据处理子包
│   ├── model/                 # 模型子包 (U-Net + 扩散调度)
│   ├── training/              # 训练子包
│   ├── evaluation/            # 评估子包
│   ├── inference/             # 推理子包
│   └── app/                   # Streamlit 交互式应用
│
├── src/                       # 入口脚本
│   ├── process_data.py        # 数据预处理入口
│   ├── train.py               # 训练入口
│   ├── evaluate.py            # 批量评估入口
│   ├── run_pipeline.py        # 端到端闭环流水线
│   └── legacy/                # 早期独立脚本 (已归档)
│
├── utils/                     # 辅助工具
│   ├── download_era5_sample.py
│   └── visualize.py
│
├── Data/                      # 数据目录 (gitignore)
│   ├── Sample/                # 原始数据样本
│   └── Processed/             # 预处理后的标准化数据
│
├── checkpoints/               # 模型权重 (gitignore)
├── outputs/                   # 输出目录 (gitignore)
│   ├── logs/                  # 训练日志
│   ├── figures/               # 图片输出
│   └── evaluation/            # 评估结果
│
├── docs/                      # 项目文档
│   ├── proposal/              # 开题相关
│   ├── midterm/               # 中期相关
│   └── defense/               # 答辩相关
│
└── requirements.txt           # Python 依赖
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 数据预处理

```bash
# 使用 wetPf2 作为标签 (默认)
python src/process_data.py

# 使用 ERA5 作为标签
python src/process_data.py --mode era5

# 关闭严格 QC / 关闭数据划分
python src/process_data.py --no-strict-qc --no-split
```

### 3. 训练模型

```bash
# 单变量 (温度) + 原始 U-Net
python src/train.py --mode single --model legacy --epochs 100

# 多变量 (温/压/湿) + 增强版 U-Net (推荐)
python src/train.py --mode multi --model enhanced --epochs 100 --patience 20
```

### 4. 批量评估

```bash
# DDIM 快速评估
python src/evaluate.py --sampler ddim --n_samples 50

# 指定模型权重
python src/evaluate.py --model_path enhanced_ro_diffusion_best.pth --model_type enhanced --out_channels 3
```

### 5. 端到端流水线 (一键运行)

```bash
# 全部阶段: 数据处理 → 训练 → 评估
python src/run_pipeline.py --all

# 仅训练 + 评估
python src/run_pipeline.py --train --evaluate

# 指定参数
python src/run_pipeline.py --all --model_type enhanced --var_mode multi --epochs 50
```

### 6. 启动交互式界面

```bash
streamlit run ro_retrieval/app/streamlit_app.py
```

---

## 模型架构

### 条件扩散模型 (DDPM)

- **前向过程**: 逐步向大气剖面添加高斯噪声（$T=1000$ 步）
- **反向过程**: 以弯曲角剖面为条件，U-Net 预测噪声并迭代去噪
- **DDIM 加速**: 确定性跳步采样，50 步即可生成高质量剖面

### 增强版 U-Net (`EnhancedConditionalUNet1D`)

```
输入: x_t (噪声剖面) + t (时间步) + condition (弯曲角)
  ↓
正弦时间嵌入 (SinusoidalTimeEmbedding)
  ↓
编码器: ResBlock1D × 2 + CrossAttention1D × 2 + MaxPool
  ↓
瓶颈层: ResBlock1D + CrossAttention1D
  ↓
解码器: ConvTranspose + Skip Connection + ResBlock1D × 2
  ↓
输出: 预测噪声 ε (channels = 1 或 3)
```

### 原始 U-Net (`ConditionalUNet1D`)

保留用于兼容早期训练的 `.pth` 权重，结构更简单（无注意力 / 无残差块）。

---

## 数据流水线

```
COSMIC atmPrf (弯曲角)          COSMIC wetPf2 (温/压/湿) 或 ERA5
       │                                    │
       ▼                                    ▼
  QC: 有效点数 / 穿透高度          QC: 温度范围 / 气压单调性
       │                           / 湿度非负 / 递减率
       ▼                                    ▼
  插值到标准高度网格              插值到标准高度网格
  (0–60 km, 301 点)              (0–60 km, 301 点)
       │                                    │
       ▼                                    ▼
  log10(|BA| + ε) → x_vec       [T, P, q] stack → y_vec (3, 301)
       │                                    │
       └──────────── 配对 ──────────────────┘
                     │
                     ▼
            Z-Score 标准化
                     │
                     ▼
        train / val / test (70 / 15 / 15)
```

---

## 评估指标

| 指标 | 公式 | 说明 |
|------|------|------|
| RMSE | $\sqrt{\frac{1}{n}\sum(y_{pred}-y_{true})^2}$ | 均方根误差 |
| Bias | $\frac{1}{n}\sum(y_{pred}-y_{true})$ | 系统偏差 |
| CC   | $\text{corr}(y_{pred},\, y_{true})$ | 相关系数 |
| MAE  | $\frac{1}{n}\sum|y_{pred}-y_{true}|$ | 平均绝对误差 |

---

## 数据来源

- **COSMIC-2**: [CDAAC](https://data.cosmic.ucar.edu/) — `atmPrf` (弯曲角) + `wetPf2` (温/压/湿)
- **ERA5**: [ECMWF](https://cds.climate.copernicus.eu/) — 37 层气压面再分析数据 (温度 / 比湿 / 位势高度)

---

## 依赖

- Python ≥ 3.9
- PyTorch ≥ 2.0
- NumPy, SciPy, Matplotlib, xarray, netCDF4, tqdm, Streamlit
