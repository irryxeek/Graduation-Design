# 项目记忆

> 供后续 AI 助手快速建立上下文。以下内容已按 2026-04-13 的本地仓库状态核对；当文档结论、实验摘要与本地代码不一致时，以当前仓库实际文件为准。

## 项目概况

- 项目名称: 基于掩星数据的气象要素反演系统
- 学生: 林逸飞（220110814）
- 工作区: `D:\02_Study\01_Schoolwork\Graduation Design`
- 主要语言: Python
- 主要框架: PyTorch, Streamlit
- 当前仓库不只是代码仓库，还包含论文、开题/中期/答辩材料、下载与文档处理脚本

## 当前两条工作主线

1. 默认主线: 基于 `Data/Processed/` 的 GNSS-RO/COSMIC 风格标准化数据，使用条件扩散模型从弯曲角剖面反演三变量大气剖面。
2. 实验主线: 基于 FY-3D GNOS `ATP+WAP` 的 2025 年上半年扩展实验，相关代码、实验结果和论文草稿已写入仓库，但对应数据目录当前本地缺失，无法直接复现。

## 代码结构

- `ro_retrieval/config.py`: 全局路径、扩散超参数、评估参数
- `ro_retrieval/data/process_enhanced.py`: 默认主线数据处理
- `ro_retrieval/data/atp_process.py`: ATP 双变量处理流程
- `ro_retrieval/data/atp_wap_process.py`: ATP+WAP 三变量处理流程
- `ro_retrieval/data/fy3d_process.py`: FY-3D 相关处理代码，统计量键名使用 `y_means/y_stds`
- `ro_retrieval/data/dataset.py`: `RODataset` 与 `ROMultiVarDataset`，当前没有 `get_stats()`
- `ro_retrieval/model/unet.py`: `ConditionalUNet1D` 与 `EnhancedConditionalUNet1D`
- `ro_retrieval/model/diffusion.py`: `DiffusionSchedule`、`ddpm_sample()`、`ddim_sample()`
- `ro_retrieval/training/trainer.py`: 训练与测试集评估入口
- `ro_retrieval/inference/predict.py`: 通用推理接口
- `ro_retrieval/app/streamlit_app.py`: Streamlit 演示界面
- `src/`: 主脚本与实验脚本；既有默认主线入口，也有 ATP、Q1、ATP+WAP 专项脚本
- `src/legacy/`: 早期单变量/旧评估脚本，除非明确需要，不要优先基于这里做新开发

## 文档与非代码目录

- `docs/proposal/`: 开题相关材料
- `docs/midterm/`: 中期答辩材料，现已整理为 `figures/`、`presentation/`、`reports/`、`scripts/`、`workspace/`
- `docs/defense/`: 毕设答辩 PPT 与模板
- `docs/thesis/`: 论文活跃工作目录，当前包含 `draft.md`、`outline.md`、`post_midterm_progress.md` 和 `figures/`
- `experiments/`: 近期 ATP+WAP 实验权重、训练日志、评估结果
- `utils/`: 下载、ATP/WAP 原始数据整理、文档处理、画图和调试脚本

## 当前本地真实存在的数据与产物

- 本地存在 `Data/Processed/`
- 本地存在 `Data/cdaac/`、`Data/FY-3/`、`Data/Sample/`
- 本地不存在 `Data/Processed_ATP/`
- 本地不存在 `Data/Processed_ATP_Q1/`
- 本地不存在 `Data/Processed_ATP_WAP/`
- 本地不存在 `Data/Processed_ATP_WAP_2025/`
- 本地不存在 `Data/Processed_ATP_WAP_2025_work/`
- 本地不存在 `Data/FY-3/ATP_WAP_2025_RAW/`
- 根目录存在 `enhanced_ro_diffusion_best.pth` 及多个根目录检查点
- `checkpoints/` 目录下同时存在 `legacy` 和 `enhanced` 两套检查点
- 根目录存在 `evaluation_results_ddim_enhanced/`
- `samples/` 下存在 ATP+WAP 演示上传样例：
  - `demo_upload_atp_wap_2025_16.npz`
  - `demo_upload_atp_wap_2025_16_x_only.npz`
  - `demo_upload_atp_wap_2025_16_README.json`

## `Data/Processed/` 当前状态

来自本地目录和 `processing_report.json`：

- 数据文件:
  - `train_x.npy` / `train_y.npy`
  - `val_x.npy` / `val_y.npy`
  - `test_x.npy` / `test_y.npy`
  - `processing_report.json`
  - `split_meta.json`
- 总文件数: 5766
- 处理成功: 5090
- QC 过滤: 676
- 通过率: 88.3%
- 训练集: 3563
- 验证集: 763
- 测试集: 764
- 标签形状: `(N, 3, 301)`
- `era5_used`: `false`
- 当前目录里没有 `norm_params.npz`
- 当前目录里没有 `stats.npy`
- 已确认 `train_x.npy` / `val_x.npy` / `test_x.npy` 与 `train_y.npy` / `val_y.npy` / `test_y.npy` 是标准化后的数据，任何脚本都不应再假设它们是原始物理量

## ATP+WAP 相关实验产物

- 本地存在实验目录 `experiments/atp_wap_2025_hw4_hmon_g005/`
- 本地存在 `experiments/atp_wap_2025_hw4_hmon_g005/enhanced_ro_diffusion_best.pth`
- 本地存在 `experiments/atp_wap_2025_hw4_hmon_g005/enhanced_ro_diffusion_training_log.json`
- 本地存在 `experiments/atp_wap_2025_hw4_hmon_g005/enhanced_ro_diffusion_loss_history.npy`
- 本地不存在 `experiments/atp_wap_2025_hw4_hmon_g005/enhanced_ro_diffusion_epoch_50.pth`
- 训练日志记录的关键值:
  - `epochs_trained = 50`
  - `best_val_loss = 0.02270949887888913`
  - `best_monitor_value = 0.009187384826186674`
  - `var_weights = [1.0, 1.0, 4.0]`
  - `monitor_target = humidity`
  - `humidity_grad_weight = 0.05`
- 本地存在评估目录:
  - `experiments/atp_wap_2025_hw4_hmon_g005_eval/`
  - `experiments/atp_wap_2025_hw4_hmon_g005_eval_fulltest/`
  - `experiments/atp_wap_2025_hw4_hmon_g005_ddim50_eval_fulltest/`

## ATP+WAP 报告中的结果摘要

来自现有 `evaluation_report.json`，仅记录为“仓库内已有结果”，不要自动视为物理量空间下完全可信的最终结论：

- `experiments/atp_wap_2025_hw4_hmon_g005_eval/evaluation_report.json`
  - `n_samples = 100`
  - temperature: `RMSE = 0.6343`, `CC = 0.8039`
  - pressure: `RMSE = 0.0839`, `CC = 0.9990`
  - humidity: `RMSE = 0.7153`, `CC = 0.7235`
- `experiments/atp_wap_2025_hw4_hmon_g005_eval_fulltest/evaluation_report.json`
  - `n_samples = 9618`
  - temperature: `RMSE = 0.6267`, `CC = 0.7820`
  - pressure: `RMSE = 0.0756`, `CC = 0.9990`
  - humidity: `RMSE = 0.7996`, `CC = 0.6960`
- `experiments/atp_wap_2025_hw4_hmon_g005_ddim50_eval_fulltest/evaluation_report.json`
  - `n_samples = 9618`
  - metadata 记录 `sampler = ddim`, `ddim_steps = 50`, `batch_size = 64`, `device = cuda`
  - temperature: `RMSE = 0.6261`, `CC = 0.7820`
  - pressure: `RMSE = 0.0756`, `CC = 0.9990`
  - humidity: `RMSE = 0.7974`, `CC = 0.6967`

## 统计量与数据契约

- 默认主线 `process_enhanced.py` 产出 `norm_params.npz`，键名使用 `x_mean` / `x_std` / `y_means` / `y_stds`
- `fy3d_process.py` 也使用 `y_means` / `y_stds`
- ATP、ATP+WAP、Q1 相关流程通常产出 `stats.npy`，键名使用 `x_mean` / `x_std` / `y_mean` / `y_std`
- ATP/Q1 流程还会用到 `pressure_log_transformed`
- 后续开发不要再假设“所有流程都共享同一套统计量文件格式”

## 当前代码层面的高优先级风险

1. `src/evaluate.py` 当前仍会对已经标准化的 `test_x.npy` 再做一次标准化，并用标准化后的 `train_y.npy` 重新计算 `y_mean/y_std` 做“反标准化”。
   这会导致评估输入被二次变换，输出也没有真正回到物理量空间。

2. `ro_retrieval/app/streamlit_app.py` 在缺少 `stats.npy` 时，会直接从标准化后的训练集重新计算 `x_mean/x_std/y_mean/y_std`，并依赖 `y_mean/y_std` 键名。
   对默认主线 `Data/Processed/` 而言，这会把展示结果建立在伪统计量之上。

3. `ro_retrieval/training/trainer.py` 的 `evaluate_test()` 仍调用 `test_dataset.get_stats()`，但 `ro_retrieval/data/dataset.py` 中的 `RODataset` 和 `ROMultiVarDataset` 当前并没有 `get_stats()`。
   这条评估路径按现状会运行时报错。

4. `ro_retrieval/inference/predict.py` 仍依赖 `y_mean/y_std` 键名。
   它与 `process_enhanced.py`、`fy3d_process.py` 的 `y_means/y_stds` 契约不一致。

5. 多个 ATP/Q1/ATP+WAP 专项脚本硬编码依赖缺失目录，例如 `Data/Processed_ATP/`、`Data/Processed_ATP_Q1/`、`Data/Processed_ATP_WAP_2025/`。
   当前本地环境下，除非先恢复数据，否则这些脚本不能直接运行成功。

6. 仓库中文档、论文草稿和实验摘要已把 ATP+WAP H1 结果与 `DDIM-50` 结果当作重要结论使用。
   但这些结果所依赖的 `src/evaluate.py` 路径目前仍有标准化/反标准化问题，因此 RMSE、Bias 以及物理空间解释都需要谨慎复核。

## 关于 DDIM 的当前判断

- 旧文档里有“DDIM 完全失败”的表述
- 新实验摘要与 `experiments/atp_wap_2025_hw4_hmon_g005_ddim50_eval_fulltest/` 则记录了 `DDIM-50` 的完整测试结果
- 更准确的当前记忆应为:
  - 早期版本曾出现 DDIM 失稳
  - 当前仓库内已经保留了一份 `DDIM-50` 全测试集评估结果
  - 但由于公共评估脚本仍存在统计量与反归一化问题，不能把这些结果直接当作“已完全验证的物理量空间结论”

## 论文与答辩写作状态

- `docs/thesis/draft.md` 是当前论文初稿的主参考文件；后续凡涉及“当前论文怎么写了”“论文现阶段结论是什么”，优先以此文件为准
- `docs/thesis/outline.md` 给出了毕业论文章节结构与模板要求
- `docs/thesis/post_midterm_progress.md` 记录了中期答辩后从 ATP-only 扩展到 ATP+WAP、从单月扩展到半年、以及湿度加权训练的总结
- `docs/training_memory_2025_atp_wap_q1_q2.md` 与 `docs/experiment_summary_2025_h1_atp_wap.md` 总结了 ATP+WAP H1 数据、训练配置与实验结果
- 如果继续写论文，必须区分“仓库中已有实验摘要”与“当前代码链路已经完全可信”这两件事

## 当前论文初稿的关键信息

- 标题已写为“基于掩星数据的气象要素反演系统设计与实现”
- 当前初稿已经覆盖摘要、Abstract、第 1 章到第 6 章、参考文献、致谢和附录框架
- 初稿中 ATP+WAP 路线已经被写成论文主线，而不是备选实验
- 初稿明确写入了以下结论性表述：
  - 基于 2025 年 1 到 6 月 FY-3D GNOS `ATP+WAP` 数据构建 `64,116` 条有效样本
  - 全测试集样本数写为 `9,618`
  - `DDPM 1000` 步与 `DDIM 50` 步结果“基本一致”
  - `DDIM` 将推理迭代降低为 `1/20`
- 初稿第 5 章已经明确说明：表中的 `RMSE` 和 `Bias` 均在标准化空间下计算
- 因此如果后续润色论文，需要避免把这些指标直接表述成原始物理单位下的最终业务精度

## 当前工作区状态

- 仓库是脏工作区
- 至少以下文件或目录有未提交变更:
  - `.claude/project-context.md`
  - `.claude/settings.json`
  - `.claude/settings.local.json`
  - `docs/thesis/draft.md`
  - `docs/` 下多份中期/论文相关文档
  - `utils/` 下新增下载日志与脚本
- `docs/midterm/` 旧的散落文件正在被替换为新的分目录结构
- 不要回滚与当前任务无关的变更

## 下次接手建议

1. 如果目标是修代码并保证本地可复现，优先走默认主线，先从 `Data/Processed/` 和默认权重开始，不要直接跳到 ATP/WAP 数据目录。
2. 如果目标是接 ATP/WAP 实验，第一步先确认或恢复 `Data/Processed_ATP_WAP_2025/` 等缺失目录，否则不要假设实验脚本能直接运行。
3. 如果目标是修评估可信度，优先处理 `src/evaluate.py`、`ro_retrieval/app/streamlit_app.py`、`Trainer.evaluate_test()`、`ro_retrieval/inference/predict.py` 的统计量契约与反归一化问题。
4. 如果目标是继续写论文，引用 ATP+WAP 指标时要明确出处路径和日期，并在必要时补一句“当前代码链路仍需复核评估口径”。
