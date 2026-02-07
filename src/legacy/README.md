# Legacy Scripts (早期独立脚本)

此目录保存项目早期开发阶段的独立脚本，仅供参考和历史回溯。

**请勿在新流程中使用这些脚本**，它们包含内联的模型定义和硬编码路径，
已被 `ro_retrieval` 包 + `src/` 入口脚本完全取代。

| 旧脚本 | 替代方案 |
|--------|----------|
| `train_standalone.py` | `src/train.py` (使用 `ro_retrieval.training.Trainer`) |
| `process_batch.py` | `src/process_data.py` (使用 `ro_retrieval.data.process_enhanced`) |
| `inference_ddpm.py` | `src/run_pipeline.py --evaluate --sampler ddpm` |
| `inference_ddim.py` | `src/run_pipeline.py --evaluate --sampler ddim` |
| `evaluate_ddpm_batch.py` | `src/evaluate.py --sampler ddpm` |
| `evaluate_ddim_batch.py` | `src/evaluate.py --sampler ddim` |
