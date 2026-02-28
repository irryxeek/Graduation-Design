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
  data/                # 数据处理子包 (部分模块待实现)

src/                   # 入口脚本
  process_data.py      # 数据预处理
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

## 数据来源

- 当前: COSMIC-2 atmPrf (弯曲角) + wetPf2 (温/压/湿)，CDAAC netCDF 格式
- 计划迁移: FY-3D GNOS 掩星数据 (NSMC HDF5 格式)，待下载样本后适配
- 备选标签: ERA5 再分析数据 (37 层气压面)

## 已知问题

- ro_retrieval/data/ 下 dataset.py, process_enhanced.py 等被引用但未实现
- 湿度数据曾出现全零问题，代码已修复但可能需重新处理

## 开发约定

- 每次修改后同步更新 readme.md
- Python ≥ 3.9, PyTorch ≥ 2.0
- 使用中文交流
