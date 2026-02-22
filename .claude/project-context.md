# 项目上下文文档

> 本文档供 AI 助手快速了解项目背景和当前状态

## 项目基本信息

- **项目名称**: 基于掩星数据的气象要素反演系统
- **学生**: 林逸飞 (220110814)
- **技术栈**: PyTorch, DDPM/DDIM 扩散模型, Streamlit
- **项目路径**: `D:\02_Study\01_Schoolwork\Graduation Design`

## 项目结构

```
├── ro_retrieval/              # 核心包
│   ├── config.py              # 全局配置（超参数、路径）
│   ├── data/
│   │   ├── dataset.py         # RODataset, ROMultiVarDataset
│   │   ├── process_enhanced.py # 数据处理流水线 ⚠️ 2026-02-22 已修复湿度提取
│   │   ├── quality_control.py # 质量控制
│   │   └── era5_matching.py   # ERA5 时空匹配
│   ├── model/
│   │   ├── unet.py            # ConditionalUNet1D, EnhancedConditionalUNet1D
│   │   └── diffusion.py       # DiffusionSchedule, ddpm_sample, ddim_sample
│   ├── training/
│   │   └── trainer.py         # Trainer 类（训练+测试集评估）
│   ├── inference/
│   │   └── predict.py         # 推理接口
│   ├── evaluation/
│   │   └── metrics.py         # RMSE, Bias, CC, EvaluationReport
│   └── app/
│       └── streamlit_app.py   # Web 可视化界面
├── src/                       # 入口脚本
│   ├── process_data.py        # 数据处理入口
│   ├── train.py               # 训练入口
│   ├── evaluate.py            # 评估入口
│   └── run_pipeline.py        # 端到端流水线
├── Data/
│   ├── Sample/                # 原始 COSMIC-2 数据
│   └── Processed/             # 处理后的 npy 文件
└── *.pth                      # 训练好的模型权重
```

## 核心概念

### 数据流
```
COSMIC-2 atmPrf (弯曲角) + wetPf2 (温度/压力/湿度)
    ↓ process_enhanced.py
标准化数据 train_x.npy (N, 301), train_y.npy (N, 3, 301)
    ↓ trainer.py
训练好的模型 enhanced_ro_diffusion_best.pth
    ↓ predict.py / evaluate.py
反演结果 + 评估报告
```

### 模型架构
- **输入**: 弯曲角剖面 (1, 301) - 经 log10 变换和 Z-Score 标准化
- **输出**: 温度/压力/湿度剖面 (3, 301)
- **条件机制**: 交叉注意力 (Query=主特征, Key/Value=弯曲角)
- **采样**: DDIM 50步（比 DDPM 1000步快 20 倍）

### 关键配置 (config.py)
```python
STD_HEIGHT = 301          # 高度层数 (0-60 km)
TIMESTEPS = 1000          # 扩散步数
DDIM_STEPS = 50           # DDIM 采样步数
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
```

## 当前状态 (2026-02-22)

### 已完成
- ✅ 数据处理流水线（含 QC）
- ✅ 增强版 U-Net 模型（交叉注意力）
- ✅ 训练器（Early Stopping, 检查点）
- ✅ DDPM/DDIM 采样
- ✅ 评估指标体系
- ✅ Streamlit 可视化界面
- ✅ 数据集划分 (train/val/test)

### 最近修复 (2026-02-22)
**湿度变量提取问题**:
- 文件: `ro_retrieval/data/process_enhanced.py`
- 问题: 变量名不匹配 (`Vap_pres`/`Shum` vs 实际的 `Vp`/`sph`)
- 修复: 支持多种变量名 + g/kg → kg/kg 单位转换

### 待完成
- ⬜ 重新处理数据（应用湿度修复）
- ⬜ 重新训练模型
- ⬜ 完整多变量评估
- ⬜ 论文撰写

## 常用命令

```bash
# 数据处理
python src/process_data.py --mode wet --split

# 训练
python src/train.py --model_type enhanced --mode multi --epochs 100

# 评估
python src/evaluate.py --model_path enhanced_ro_diffusion_best.pth \
    --model_type enhanced --out_channels 3 --n_samples 100

# 启动 Web 界面
streamlit run ro_retrieval/app/streamlit_app.py

# 测试集评估（使用 Trainer）
python -c "
from ro_retrieval.training.trainer import Trainer
trainer = Trainer(model_type='enhanced', mode='multi')
trainer.evaluate_test()
"
```

## 数据文件说明

| 文件 | 形状 | 说明 |
|------|------|------|
| train_x.npy | (1493, 301) | 训练集弯曲角 |
| train_y.npy | (1493, 3, 301) | 训练集标签 [T, P, Q] |
| val_x.npy | (319, 301) | 验证集弯曲角 |
| val_y.npy | (319, 3, 301) | 验证集标签 |
| test_x.npy | (321, 301) | 测试集弯曲角 |
| test_y.npy | (321, 3, 301) | 测试集标签 |

**注意**: 当前数据中湿度全为零，需重新处理！

## 模型文件说明

| 文件 | 说明 |
|------|------|
| enhanced_ro_diffusion_best.pth | 增强模型最佳权重 |
| enhanced_ro_diffusion_epoch_*.pth | 增强模型检查点 |
| ro_diffusion_epoch_*.pth | 原始模型检查点 |

## 评估结果摘要

**温度反演** (321 测试样本):
- RMSE: 8.94 K
- 相关系数: 0.934
- R²: 0.866

**压力反演**:
- RMSE: 64.14 hPa
- 相关系数: 0.959
- R²: 0.918

## 已知问题

1. **湿度数据全零** - 已修复代码，需重新处理数据
2. **评估报告只有温度** - 运行评估时需指定 `--out_channels 3`

## 相关文档

- `README.md` - 项目说明
- `项目完成度评估报告.md` - 详细评估报告
- `开题内容/林逸飞-220110814-本科毕业设计开题报告.docx` - 开题报告

## 快速调试

```python
# 检查数据
import numpy as np
y = np.load('Data/Processed/test_y.npy')
print(f'形状: {y.shape}')
for i, name in enumerate(['温度', '压力', '湿度']):
    print(f'{name}: min={y[:,i,:].min():.2f}, max={y[:,i,:].max():.2f}')

# 检查模型
import torch
from ro_retrieval.model.unet import EnhancedConditionalUNet1D
model = EnhancedConditionalUNet1D(in_channels=3, cond_channels=1, out_channels=3)
print(f'参数量: {sum(p.numel() for p in model.parameters()):,}')
```
