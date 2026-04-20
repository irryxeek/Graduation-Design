# 项目记忆

> 已按 2026-04-20 本地仓库状态核对。后续若文档、旧记忆、实验摘要与当前代码不一致，以当前仓库实际文件为准。

## 项目概况

- 项目名称: 基于掩星数据的气象要素反演系统设计与实现
- 学生: 林逸飞（220110814）
- 仓库根目录: `/root/autodl-tmp/Graduation-Design`
- 主要语言与框架: Python, PyTorch, Streamlit
- 仓库同时包含代码、论文、开题/中期/答辩材料、实验结果、数据处理与下载脚本

## 当前真实主线

- 当前论文主线和当前已验证代码主线，都是 FY-3D GNOS `ATP+WAP` 2025 年上半年三变量反演路径，不再是早期 `Data/Processed/` 演示主线。
- 论文主线数据目录:
  - `Data/Processed_ATP_WAP_2025`
- 当前第三轮主实验模型目录:
  - `experiments/main_rerun_current_standard_20260419T121928Z`
- 当前第三轮主实验物理空间评估:
  - `experiments/main_rerun_current_standard_20260419T121928Z/eval_ddpm_fulltest_physical/evaluation_report.json`
  - `experiments/main_rerun_current_standard_20260419T121928Z/eval_ddim50_fulltest_physical/evaluation_report.json`
- 当前第三轮判别式基线汇总:
  - `experiments/baseline_summary_20260419T224628Z.json`
- 当前第三轮 DDIM 诊断汇总:
  - `experiments/ddim_diagnosis_20260419T225333Z/summary.json`
- 当前论文 PDF 提取文本:
  - `docs/thesis/pdf_extracted_text.txt`
- 当前第三轮论文修订稿:
  - `docs/thesis/论文修订稿_第三轮结果版.md`
- 旧修订稿仍保留，但不应再作为最新结论来源:
  - `docs/thesis/论文修订稿_基于当前项目结果.md`

## 关键数据契约

- `Data/Processed_ATP_WAP_2025` 当前本地真实存在，且包含 `stats.npy`。
- 该目录中的 `train_x.npy/train_y.npy/val_*/test_*` 已是标准化后的磁盘数组。
- `stats.npy` 保存的是物理空间统计量，不能再把它当成“磁盘数组仍未标准化”的证据。
- 兼容逻辑必须同时处理两类键名:
  - `y_mean/y_std`
  - `y_means/y_stds`
- `Data/Processed/` 这条早期默认主线仍存在，但它更偏演示/旧主线，不应再作为论文主线理解。

## 当前已修复并验证的代码路径

- `src/evaluate.py`
  - 已恢复到与历史论文结果兼容的评估口径。
  - 默认 `data_dir` 指向论文主线 `PAPER_PROCESSED_DIR`。
  - 默认 `model_type=enhanced`、`out_channels=3`。
  - `n_samples=0` 表示评估整个测试集。
  - `metric_space=standardized` 走历史兼容评估口径。
  - `metric_space=physical` 使用 `stats.npy` 做物理空间反标准化。
  - 物理空间评估会先按历史 `legacy_y_mean/std` 做预测校准，再恢复到真实物理空间。
  - 气压通道会从 `log10(P)` 反变换回 `hPa`。
  - 已加入固定随机种子，保证评估可复现。
  - 已支持高度分层统计，默认分段为 `0-5km`、`5-20km`、`20-60km`。
  - 默认开启 Savitzky-Golay 平滑，关闭需显式加 `--no_smooth`。
- `ro_retrieval/evaluation/metrics.py`
  - 已补充物理空间统计和高度分段结果所需逻辑。
- `ro_retrieval/model/baselines.py`
  - 新增最小判别式基线实现，当前包含 `MLP` 和 `1D-CNN`。
- `src/run_baselines.py`
  - 新增基线训练与评估入口，用于回答“为什么要用扩散模型”。
- `src/run_ddim_diagnosis.py`
  - 新增 DDIM 诊断入口，用于对 `step` 与 `eta` 做补充分析。
- `src/run_pipeline.py`
  - 评估阶段已经委托给 `src.evaluate.main`，与论文主链路一致。
- `ro_retrieval/stats_utils.py`
  - 当前是统计量兼容层，负责加载 `stats.npy`、统一键名、必要时做回退。
- `ro_retrieval/inference/predict.py`
  - 已修复多通道反标准化广播问题，并改为复用共享统计量逻辑。
- `ro_retrieval/training/trainer.py`
  - `evaluate_test()` 已不再依赖不存在的 `get_stats()`，而是直接调用 `src.evaluate` 的兼容评估链路。
- `ro_retrieval/app/streamlit_app.py`
  - 已接入共享统计量逻辑。
  - 前端 `use_container_width` 弃用参数已清理为 `width=...`。
  - `torch.load(..., weights_only=True)` 已显式化。
  - 当前环境下中文字体缺失产生的 matplotlib glyph 告警已做降噪处理。

## 当前已验证结论

- `src/evaluate.py` 的论文兼容链路和物理空间链路都已经实际跑通。
- 当前论文定稿应优先参考第三轮结果，而不是更早期的历史兼容报告。
- 第三轮主实验物理空间结果:
  - `DDPM`
    - 温度: `RMSE=17.0831 K`, `Bias=-1.6330 K`, `CC=0.7254`
    - 气压: `RMSE=14.7016 hPa`, `Bias=4.8111 hPa`, `CC=0.9993`
    - 湿度: `RMSE=1.3948 g/kg`, `Bias=-0.0035 g/kg`, `CC=0.8279`
  - `DDIM-50`
    - 温度: `RMSE=14.2091 K`, `Bias=1.6815 K`, `CC=0.7773`
    - 气压: `RMSE=15.7442 hPa`, `Bias=0.4174 hPa`, `CC=0.9987`
    - 湿度: `RMSE=1.3676 g/kg`, `Bias=0.1138 g/kg`, `CC=0.6587`
- 第三轮主实验高度分层结果表明:
  - 气压整廓线高 `CC` 需要谨慎解释，容易受单调垂直背景影响。
  - 湿度整廓线 `CC=0.8279` 不能直接解释为结构恢复良好。
  - `DDPM` 湿度分层 `CC`:
    - `0-5km`: `0.2313`
    - `5-20km`: `0.0708`
    - `20-60km`: `0.0168`
- 第三轮判别式基线结果:
  - `MLP`
    - 温度: `RMSE=8.8193 K`, `CC=0.8501`
    - 气压: `RMSE=14.8478 hPa`, `CC=0.9997`
    - 湿度: `RMSE=0.6544 g/kg`, `CC=0.8540`
  - `1D-CNN`
    - 温度: `RMSE=11.6336 K`, `CC=0.8070`
    - 气压: `RMSE=30.2511 hPa`, `CC=0.9897`
    - 湿度: `RMSE=0.8438 g/kg`, `CC=0.8382`
- 当前关键论文结论必须更新为:
  - 扩散模型已经证明可行，并完成了完整系统原型。
  - 但在当前 `ATP -> WAP` 监督设定下，`MLP` 基线整体强于扩散主模型。
  - 因此论文不应再宣称“扩散模型当前优于简单监督基线”。
- DDIM 诊断结论:
  - `DDIM-100 eta=0.0` 湿度 `CC=0.6542`
  - `DDIM-200 eta=0.0` 湿度 `CC=0.6472`
  - `DDIM-100 eta=0.5` 出现 `NaN`
  - 这说明当前 DDIM 湿度退化并不能简单归因于“50 步太少”。
- `Trainer.evaluate_test()` 已做 smoke 验证:
  - 输出目录: `experiments/trainer_eval_smoke_v2`
- Streamlit 前端已做两层验证:
  - 真实 headless 启动成功，HTTP `200`
  - `streamlit.testing.v1.AppTest` 下，上传模式与本地模式都能跑通
  - 本地模式点击“开始反演”后，确认出现“反演结果”和“全测试集评估报告”区块
  - 当前验证结果为 `has_exception=False`, `has_error=False`, `has_warning=False`
- 表 5-4 已按当前代码与训练配置做严格复跑:
  - 输出目录: `experiments/table5_4_ablation_rerun_20260418T225500Z`
  - 汇总文件:
    - `experiments/table5_4_ablation_rerun_20260418T225500Z/summary.json`
    - `experiments/table5_4_ablation_rerun_20260418T225500Z/summary.md`
  - 当前严格复跑结论:
    - 变量加权损失对湿度通道提升明确
    - 梯度约束在当前代码版本下未稳定带来湿度 RMSE 的进一步下降
    - 当前复跑结果不能直接视为对 PDF 原表 5-4 的严格复现
- 已新增第三轮论文修订稿:
  - `docs/thesis/论文修订稿_第三轮结果版.md`
  - 该稿已切换到物理空间主实验、分层统计、判别式基线和 DDIM 诊断结论

## 当前论文与文档理解

- 当前论文草稿主文件:
  - `docs/thesis/draft.md`
- 当前论文正式判断应优先依据:
  - `docs/thesis/reports/220110814-林逸飞-基于掩星数据的气象要素反演系统设计与实现.pdf`
  - `docs/thesis/pdf_extracted_text.txt`
- 当前论文实验修订优先依据:
  - `docs/thesis/论文修订稿_第三轮结果版.md`
- 当前项目技术上下文摘要:
  - `README.md`
  - `SUMMARY.md`
  - `docs/项目技术文档.md`
  - `docs/入门指南.md`
  - `docs/thesis/post_midterm_progress.md`
- 需要记住:
  - 旧文档里“DDIM 完全失败”的说法只适用于更早期阶段，不能再当成当前主结论。
  - 当前更准确的说法是:
    - 早期版本出现过 DDIM 失稳
    - 当前仓库保留了 `DDIM-50` 的完整评估结果
    - 第三轮结果显示 `DDIM` 温度更优，但湿度结构一致性明显差于 `DDPM`
    - 增加 `DDIM` 步数到 `100/200` 也未恢复湿度 `CC`
- 论文里若提到 RMSE/Bias，当前定稿应优先采用物理空间数值，并明确单位。
- 论文里若提到湿度或气压相关系数，必须结合高度分层统计解释，不能只看整廓线 `CC`。
- 当前更稳妥的论文边界表述:
  - 已完成 ATP->WAP 监督下的训练、评估、DDIM 对比、消融复跑、基线对比、前端原型
  - 尚未完成可直接用于定稿背书的独立外部验证闭环
  - 不能把当前结果直接表述成“对真实大气状态的独立精度验证”
  - 不能把当前结果直接表述成“扩散模型优于简单监督基线”

## 当前工程规范状态

- 仓库根目录已增加统一命令入口:
  - `Makefile`
- 已增加工程规范文档:
  - `docs/engineering/工程规范说明.md`
- 已增加最小单元测试:
  - `tests/test_stats_utils.py`
  - `tests/test_run_pipeline.py`
- 当前已实际验证:
  - `make test` 通过
  - `python src/evaluate.py --help` 可正常运行
  - `python src/run_pipeline.py --help` 可正常运行
- 当前工程化结论:
  - 本科毕业设计层面已经具备较完整闭环
  - 主要短板在独立外部验证闭环、仓库收口与结论表述边界，不在主线功能缺失

## 当前连接与工作区注意事项

- 当前远端 IDE 在大工作区下偶发 `reconnect`，更适合用终端 `sed/grep` 分段读文件并用补丁直接修改。
- 已新增工作区配置:
  - `.vscode/settings.json`
- 其作用是排除 `Data/` 和 `experiments/` 的文件监控与全文搜索，减轻远端连接压力。
- 当前机器本地资源基本正常，但 `nvidia-smi` 一度返回过 `No devices were found`，若继续跑 GPU 任务需先确认设备挂载状态。

## 重要路径

- 上传演示样例:
  - `samples/demo_upload_atp_wap_2025_16.npz`
  - `samples/demo_upload_atp_wap_2025_16_README.json`
- 当前第三轮主实验目录:
  - `experiments/main_rerun_current_standard_20260419T121928Z`
- 当前第三轮 DDPM 物理空间报告:
  - `experiments/main_rerun_current_standard_20260419T121928Z/eval_ddpm_fulltest_physical/evaluation_report.json`
- 当前第三轮 DDIM 物理空间报告:
  - `experiments/main_rerun_current_standard_20260419T121928Z/eval_ddim50_fulltest_physical/evaluation_report.json`
- 当前基线汇总:
  - `experiments/baseline_summary_20260419T224628Z.json`
- 当前 DDIM 诊断汇总:
  - `experiments/ddim_diagnosis_20260419T225333Z/summary.json`
- 当前表 5-4 复跑汇总:
  - `experiments/table5_4_ablation_rerun_20260418T225500Z/summary.json`
- 当前工程规范文档:
  - `docs/engineering/工程规范说明.md`
- 当前论文修订稿:
  - `docs/thesis/论文修订稿_第三轮结果版.md`
- 当前 PDF 提取文本:
  - `docs/thesis/pdf_extracted_text.txt`

## 当前工作区注意事项

- 当前仓库是脏工作区，含有大量未提交源码变更、文档更新、实验输出目录和未跟踪结果。
- 不要回滚与当前任务无关的改动。
- `experiments/*`、`evaluation_results_*` 下有大量实验产物，默认不要自动删除。
- 若继续做“清理环境”，优先删缓存、日志、`__pycache__`，不要误删实验结果和论文材料。
- 当前若继续写论文，优先从第三轮修订稿继续，而不是回到旧修订稿。

## 下次接手建议

1. 若目标是继续复核论文主线，优先从 `Data/Processed_ATP_WAP_2025`、`src/evaluate.py`、`experiments/main_rerun_current_standard_20260419T121928Z` 开始。
2. 若目标是继续做前端/演示，优先检查 `ro_retrieval/app/streamlit_app.py` 与 `samples/demo_upload_atp_wap_2025_16.npz`。
3. 若目标是继续清理代码/文档，优先信任当前仓库真实状态，不要照搬旧记忆里“数据目录缺失”“评估链路未修”的结论。
4. 若目标是继续写论文，优先看 PDF 与 `docs/thesis/pdf_extracted_text.txt`，再参考 `docs/thesis/论文修订稿_第三轮结果版.md`。
5. 若目标是继续补实验，应优先做外部独立验证、传统链路对比和更多超参数敏感性分析，而不是重复旧口径主实验。
6. 若目标是继续做工程规范收口，先跑 `make test`，再从 `Makefile`、`docs/engineering/工程规范说明.md` 和 `tests/` 开始。
