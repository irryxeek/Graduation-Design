# FY-3D ATP+WAP 论文主线总结

## 项目定位

当前项目主线与毕业论文一致：

- 数据源：FY-3D GNOS L2 ATP + WAP
- 输入：ATP `Opt_Bend_ang`
- 标签：WAP `Temp`、`Pres`、`Shum`
- 任务：温度 / 气压 / 湿度三变量联合反演
- 模型：`EnhancedConditionalUNet1D`

## 当前数据规模

- 时间范围：2025-01-01 至 2025-06-30
- 原始 WAP 文件：74,334
- 成功配对 ATP：69,376
- 有效样本：64,116
- 训练 / 验证 / 测试：44,881 / 9,617 / 9,618

处理后主数据目录：

- `Data/Processed_ATP_WAP_2025`

## 当前推荐训练配置

- `mode=multi`
- `model=enhanced`
- `epochs=50`
- `batch_size=64`
- `lr=1e-4`
- `patience=15`
- `var_weights=[1,1,4]`
- `monitor_target=humidity`
- `humidity_grad_weight=0.05`
- `humidity_cc_weight=0.0`

## 代表性实验结果

### DDPM 全测试集

| 变量 | RMSE | Bias | CC |
|------|-----:|-----:|---:|
| 温度 | 0.6267 | 0.0105 | 0.7820 |
| 气压 | 0.0756 | 0.0099 | 0.9990 |
| 湿度 | 0.7996 | 0.0808 | 0.6960 |

目录：

- `experiments/atp_wap_2025_hw4_hmon_g005_eval_fulltest`

### DDIM 50 步全测试集

| 变量 | RMSE | Bias | CC |
|------|-----:|-----:|---:|
| 温度 | 0.6261 | 0.0103 | 0.7820 |
| 气压 | 0.0756 | 0.0099 | 0.9990 |
| 湿度 | 0.7974 | 0.0793 | 0.6967 |

目录：

- `experiments/atp_wap_2025_hw4_hmon_g005_ddim50_eval_fulltest`

## 当前项目状态

- `README.md` 已切换到论文主线说明
- `src/process_data.py` 默认入口已切换到 `fy3d_atp_wap`
- `src/train.py` 默认训练配置已切换到 ATP+WAP 三变量训练
- `src/evaluate.py` 已支持按论文口径在标准化空间评估，并默认沿用主实验平滑口径
- `src/run_pipeline.py` 已改为 ATP+WAP 主流程

## 仍保留的历史内容

仓库中仍保留以下历史代码与实验，用于回溯和兼容：

- COSMIC-2 / ERA5 / wetPf2 早期流程
- FY-3D ATP-only / Q1 数据阶段实验
- `src/legacy/` 下的早期训练与评估脚本

这些内容不再是默认主入口。
