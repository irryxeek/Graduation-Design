# 基于掩星数据的气象要素反演系统

基于条件扩散模型的 GNSS 掩星一维大气廓线反演项目。当前仓库主线与论文一致，重点场景为：

- 输入：FY-3D GNOS L2 ATP 弯曲角廓线
- 标签：FY-3D GNOS L2 WAP 温度 / 气压 / 湿度廓线
- 模型：`EnhancedConditionalUNet1D` + DDPM / DDIM
- 任务：三变量联合反演

## 当前主线

论文与当前实验主线使用 2025 年 1 月到 6 月 FY-3D ATP+WAP 配对数据。

- 原始 WAP 文件：74,334
- 成功配对 ATP：69,376
- 有效样本：64,116
- 数据划分：train 44,881 / val 9,617 / test 9,618
- 默认处理后数据目录：`Data/Processed_ATP_WAP_2025`

代表性实验目录：

- `experiments/atp_wap_2025_hw4_hmon_g005`
- `experiments/atp_wap_2025_hw4_hmon_g005_eval_fulltest`
- `experiments/atp_wap_2025_hw4_hmon_g005_ddim50_eval_fulltest`

## 仓库结构

```text
.
├── ro_retrieval/              核心包
│   ├── data/                  数据处理
│   ├── model/                 U-Net 与扩散调度
│   ├── training/              训练器
│   ├── evaluation/            指标与报告
│   ├── inference/             推理接口
│   └── app/                   Streamlit 前端
├── src/                       主入口脚本
├── Data/                      原始数据与处理后数据
├── experiments/               训练与评估产物
├── docs/thesis/               论文草稿、图件、正式 PDF
└── utils/                     下载、分块处理、文档辅助脚本
```

## 关键入口

### 1. ATP+WAP 数据预处理

```bash
python src/process_data.py \
  --source fy3d_atp_wap \
  --atp_dir Data/FY-3/ATP_WAP_2025_RAW/ATP \
  --wap_dir Data/FY-3/ATP_WAP_2025_RAW/WAP \
  --output_dir Data/Processed_ATP_WAP_2025 \
  --qc-threshold 100
```

### 2. 按论文配置训练

```bash
python src/train.py \
  --mode multi \
  --model enhanced \
  --data_dir Data/Processed_ATP_WAP_2025 \
  --epochs 50 \
  --batch_size 64 \
  --patience 15 \
  --var_weights 1,1,4 \
  --monitor_target humidity \
  --humidity_grad_weight 0.05
```

### 3. 按论文口径评估

默认在标准化空间统计指标；`n_samples=0` 表示评估全部测试集样本。
当前默认会沿用论文主实验的 Savitzky-Golay 平滑口径；如需关闭，可追加 `--no_smooth`。

```bash
python src/evaluate.py \
  --model_path experiments/atp_wap_2025_hw4_hmon_g005/enhanced_ro_diffusion_best.pth \
  --model_type enhanced \
  --data_dir Data/Processed_ATP_WAP_2025 \
  --out_channels 3 \
  --sampler ddim \
  --ddim_steps 50 \
  --n_samples 0 \
  --batch_size 64 \
  --metric_space standardized
```

如需物理空间指标：

```bash
python src/evaluate.py ... --metric_space physical
```

### 4. 一键流水线

```bash
python src/run_pipeline.py --all
```

默认行为：

- 数据源：`fy3d_atp_wap`
- 训练模式：`multi`
- 模型：`enhanced`
- 变量权重：`[1, 1, 4]`
- 监控目标：`humidity`
- 湿度梯度约束：`0.05`
- 评估空间：`standardized`

### 5. 启动前端

```bash
streamlit run ro_retrieval/app/streamlit_app.py
```

前端优先发现以下数据集：

- `Processed_ATP_WAP_2025`
- `Processed_ATP_WAP`
- `Processed_ATP_Q1`

并支持上传 `.npy` / `.npz` / `.csv` 做批量推理与指标汇总。

## 当前代表性结果

全测试集 DDPM 结果：

| 变量 | RMSE | Bias | CC |
|------|-----:|-----:|---:|
| 温度 | 0.6267 | 0.0105 | 0.7820 |
| 气压 | 0.0756 | 0.0099 | 0.9990 |
| 湿度 | 0.7996 | 0.0808 | 0.6960 |

对应目录：

- `experiments/atp_wap_2025_hw4_hmon_g005_eval_fulltest`

全测试集 DDIM 50 步结果：

| 变量 | RMSE | Bias | CC |
|------|-----:|-----:|---:|
| 温度 | 0.6261 | 0.0103 | 0.7820 |
| 气压 | 0.0756 | 0.0099 | 0.9990 |
| 湿度 | 0.7974 | 0.0793 | 0.6967 |

对应目录：

- `experiments/atp_wap_2025_hw4_hmon_g005_ddim50_eval_fulltest`

## 工程规范

仓库已补充最小化工程规范约定，见：

- `docs/engineering/工程规范说明.md`

推荐只将以下内容视为默认主线：

- 核心包：`ro_retrieval/`
- 命令行入口：`src/process_data.py`、`src/train.py`、`src/evaluate.py`、`src/run_pipeline.py`
- 论文主线实验：`experiments/atp_wap_2025_hw4_hmon_g005*`

`src/legacy/` 和其他历史脚本保留用于回溯，不建议继续作为默认入口。

## 快速自检

仓库根目录提供 `Makefile`，可用于执行最基础的工程自检和常用命令：

```bash
make test
make eval-paper
make app
```

其中 `make test` 会运行轻量级单元测试，主要覆盖统计量兼容和主线参数解析等基础逻辑。

## 说明

- `README`、`src/run_pipeline.py`、`src/evaluate.py` 已按论文主线对齐。
- 仓库中仍保留 COSMIC / FY-3D GNOS 的早期处理代码，用于回溯和兼容，不再是默认主入口。
- 正式论文 PDF 位于 `docs/thesis/reports/220110814-林逸飞-基于掩星数据的气象要素反演系统设计与实现.pdf`。
