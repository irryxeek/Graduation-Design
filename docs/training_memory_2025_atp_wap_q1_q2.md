# 2025年1–6月 ATP+WAP 数据集训练记忆

## 1. 任务背景

- 时间：`2026-04-09`
- 目标：基于 `2025-01-01` 至 `2025-06-30` 的 FY-3D GNOS `ATP+WAP` 配对数据，完成预处理、训练与评估
- 训练目标：提升湿度反演效果，同时保持温度、气压通道可用

## 2. 数据集版本

### 2.1 原始数据范围

- 数据范围：`2025-01-01` 至 `2025-06-30`
- 原始统一目录：
  - `Data/FY-3/ATP_WAP_2025_RAW/ATP`
  - `Data/FY-3/ATP_WAP_2025_RAW/WAP`
- 原始汇总文件：`Data/FY-3/ATP_WAP_2025_RAW/summary.json`

### 2.2 预处理后数据集

- 最终数据目录：`Data/Processed_ATP_WAP_2025`
- 数据摘要文件：`Data/Processed_ATP_WAP_2025/summary.json`

### 2.3 数据统计

- `total_wap`: `74334`
- `paired`: `69376`
- `missing_pairs`: `4958`
- `failed`: `5260`
- `processed`: `64116`
- 划分结果：
  - `train`: `44881`
  - `val`: `9617`
  - `test`: `9618`

### 2.4 数据标准化统计量

- `x_mean`: `-3.4699220239836093`
- `x_std`: `1.119969342932032`
- `y_mean`:
  - temperature: `238.58773476166033`
  - pressure: `1.1202732584230624`
  - humidity: `0.3926786999367507`
- `y_std`:
  - temperature: `22.177633119957484`
  - pressure: `1.0927786762726766`
  - humidity: `1.6639531176671474`

## 3. 预处理流程说明

### 3.1 采用的最终方案

- 原始单次全量预处理在“最终合并/保存阶段”存在不稳定退出问题
- 最终改为：
  1. 分块处理原始 `ATP+WAP` 文件
  2. 保存每个块的 `x.npy / y.npy / meta.json`
  3. 基于块结果进行统计
  4. 再按块直接写入最终 `train/val/test`

### 3.2 关键脚本

- 分块处理脚本：`utils/process_atp_wap_chunked.py`
- 处理器实现：`ro_retrieval/data/atp_wap_process.py`

### 3.3 中间结果目录

- 分块中间目录：`Data/Processed_ATP_WAP_2025_work`
- 共完成 `15` 个块

## 4. 本次训练配置

### 4.1 训练命令等价配置

- 训练入口：`src/train.py`
- 数据目录：`Data/Processed_ATP_WAP_2025`
- 输出目录：`experiments/atp_wap_2025_hw4_hmon_g005`

### 4.2 主要超参数

- `mode`: `multi`
- `model`: `enhanced`
- `epochs`: `50`
- `batch_size`: `64`
- `lr`: `1e-4`
- `patience`: `15`
- `var_weights`: `1,1,4`
- `monitor_target`: `humidity`
- `humidity_grad_weight`: `0.05`
- `humidity_cc_weight`: `0.0`

### 4.3 设备

- GPU: `Tesla V100-PCIE-32GB`

## 5. 训练结果

### 5.1 训练产物

- 最佳模型：`experiments/atp_wap_2025_hw4_hmon_g005/enhanced_ro_diffusion_best.pth`
- 最终轮模型：`experiments/atp_wap_2025_hw4_hmon_g005/enhanced_ro_diffusion_epoch_50.pth`
- 训练日志：`experiments/atp_wap_2025_hw4_hmon_g005/enhanced_ro_diffusion_training_log.json`
- 损失历史：`experiments/atp_wap_2025_hw4_hmon_g005/enhanced_ro_diffusion_loss_history.npy`

### 5.2 训练摘要

- `epochs_trained`: `50`
- `best_val_loss`: `0.02270949887888913`
- `best_monitor_value (humidity)`: `0.009187384826186674`
- 总训练耗时：约 `21.9` 分钟

### 5.3 收敛特征

- 初始训练损失较高，前几轮快速下降
- 后期训练损失稳定在 `0.02x`
- 验证损失稳定在 `0.02x`
- 湿度监控指标在后半程稳定改善

## 6. 评估配置

- 评估入口：`src/evaluate.py`
- 评估模型：`experiments/atp_wap_2025_hw4_hmon_g005/enhanced_ro_diffusion_best.pth`
- 评估输出目录：`experiments/atp_wap_2025_hw4_hmon_g005_eval`
- 评估方式：
  - `sampler`: `ddim`
  - `ddim_steps`: `50`
  - `n_samples`: `100`
  - `out_channels`: `3`

## 7. 本次评估结果

评估结果文件：`experiments/atp_wap_2025_hw4_hmon_g005_eval/evaluation_report.json`

### 7.1 Temperature

- `RMSE`: `0.6343408046662807`
- `Bias`: `0.08372102498309687`
- `CC`: `0.803889784700258`

### 7.2 Pressure

- `RMSE`: `0.08391555089503527`
- `Bias`: `0.02106663252285216`
- `CC`: `0.998969473200535`

### 7.3 Humidity

- `RMSE`: `0.715326806306839`
- `Bias`: `0.13775630528107285`
- `CC`: `0.7235299499907434`

## 8. 与旧版结果对比

旧版对比文件：`experiments/atp_wap_hw4_hmon_g005_eval/evaluation_report.json`

### 8.1 旧版结果

- Temperature:
  - `RMSE=0.6481530076265335`
  - `CC=0.7682601629649775`
- Pressure:
  - `RMSE=0.048554166723042726`
  - `CC=0.9995650800939099`
- Humidity:
  - `RMSE=0.8689866772294045`
  - `CC=0.5321323946520607`

### 8.2 新旧对比结论

- 温度：
  - `RMSE` 小幅改善：`0.6482 -> 0.6343`
  - `CC` 提升：`0.7683 -> 0.8039`
- 气压：
  - `RMSE` 变差：`0.0486 -> 0.0839`
  - `CC` 略降但仍很高：`0.9996 -> 0.9990`
- 湿度：
  - `RMSE` 明显改善：`0.8690 -> 0.7153`
  - `CC` 明显提升：`0.5321 -> 0.7235`

### 8.3 当前判断

- 本次扩展到 `2025年1–6月` 数据后，湿度通道收益明显
- 温度通道同步提升
- 气压通道有轻微退化，需要后续继续平衡多变量损失设计

## 9. 当前最值得记住的结论

- 当前数据集已经是 `2025年1–6月` 整合数据集，不再是仅 `Q1`
- 新数据集训练后，湿度结果显著优于旧版基线
- 当前最优实验可记为：
  - 数据：`Data/Processed_ATP_WAP_2025`
  - 训练目录：`experiments/atp_wap_2025_hw4_hmon_g005`
  - 评估目录：`experiments/atp_wap_2025_hw4_hmon_g005_eval`
  - 关键配置：`var_weights=1,1,4`，`monitor_target=humidity`，`humidity_grad_weight=0.05`

## 10. 后续建议

- 保留本次模型作为新的湿度主基线
- 下一步优先考虑：
  1. 在保持湿度收益的前提下恢复气压精度
  2. 做新旧数据集对比表，写入论文正文
  3. 做按月份或按湿度强弱的分组评估

