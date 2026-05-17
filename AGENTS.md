# Codex Project Memory

> Last updated: 2026-05-11. This file records only the current project mainline.
> If it conflicts with current code, data, or experiment artifacts, trust the repository state first.

## Project Overview

- Project: 基于掩星数据的气象要素反演系统设计与实现
- Author: 林逸飞, 220110814
- Repository root: `/root/autodl-tmp/Graduation-Design`
- Stack: Python, PyTorch, Streamlit
- Core package: `ro_retrieval/`
- Main entry scripts: `src/train.py`, `src/evaluate.py`, `src/run_pipeline.py`

## Current Mainline

- Current thesis and verified code mainline: FY-3D GNOS `ATP+WAP` 2025 three-variable retrieval.
- Use only the current ATP+WAP 2025 three-variable path for thesis conclusions and project decisions.
- Main processed data:
  - `Data/Processed_ATP_WAP_2025`
- Main experiment:
  - `experiments/main_rerun_current_standard_20260419T121928Z`
- Main physical-space reports:
  - `experiments/main_rerun_current_standard_20260419T121928Z/eval_ddpm_fulltest_physical/evaluation_report.json`
  - `experiments/main_rerun_current_standard_20260419T121928Z/eval_ddim50_fulltest_physical/evaluation_report.json`
- Baseline summary:
  - `experiments/baseline_summary_20260419T224628Z.json`
- DDIM diagnosis summary:
  - `experiments/ddim_diagnosis_20260419T225333Z/summary.json`
- Table 5-4 rerun summary:
  - `experiments/table5_4_ablation_rerun_20260418T225500Z/summary.json`
- Current thesis materials:
  - `docs/thesis/reports/220110814-林逸飞-基于掩星数据的气象要素反演系统设计与实现.pdf`
  - `docs/thesis/pdf_extracted_text.txt`
  - `docs/thesis/论文修订稿_第三轮结果版.md`

## Data Contract

- `Data/Processed_ATP_WAP_2025` contains standardized disk arrays and `stats.npy`.
- `train_x.npy`, `train_y.npy`, `val_*`, and `test_*` are already standardized on disk.
- `stats.npy` stores physical-space statistics. Do not treat it as proof that disk arrays are unstandardized.
- Statistics compatibility must support both key styles:
  - `y_mean` / `y_std`
  - `y_means` / `y_stds`
- Input bending-angle profiles use shape `(N, 301)`, generally `log10(|BA| + 1e-6)` then Z-score standardized.
- Targets use channel-first shape `(N, 3, 301)` for `[temperature, pressure, humidity]`.
- Standard height grid is `0-60 km` with `301` levels.
- Pressure may be stored/standardized in log space and must be restored from `log10(P)` to hPa for physical metrics.

## Architecture

```text
ro_retrieval/
  config.py
  stats_utils.py
  data/
    dataset.py
    atp_wap_process.py
  model/
    diffusion.py
    unet.py
    baselines.py
  training/
    trainer.py
  evaluation/
    metrics.py
  inference/
    predict.py
  app/
    streamlit_app.py

src/
  train.py
  evaluate.py
  run_pipeline.py
  run_baselines.py
  run_ddim_diagnosis.py
  run_uncertainty_probe.py
```

- Diffusion mainline model: `EnhancedConditionalUNet1D`.
- Legacy compatibility exists via `ConditionalUNet1D`, but it is not the current thesis mainline.
- DDPM/DDIM sampling lives in `ro_retrieval/model/diffusion.py`.
- Shared statistics handling belongs in `ro_retrieval/stats_utils.py`.
- Typical diffusion setup: `T=1000`, beta schedule `[1e-4, 0.02]`; DDIM commonly uses 50 steps with `eta=0.0`.
- Local root checkpoint files have been moved under `checkpoints/`; do not reintroduce root-level `.pth` files.

## Verified Code Paths

- `src/evaluate.py`
  - Defaults to thesis data via `PAPER_PROCESSED_DIR`.
  - Defaults to `model_type=enhanced` and `out_channels=3`.
  - `n_samples=0` means evaluate the full test set.
  - `metric_space=standardized` uses the standardized-space evaluation path.
  - `metric_space=physical` restores values using `stats.npy`.
  - Physical-space evaluation first applies legacy prediction calibration, then restores physical units.
  - Pressure is transformed from `log10(P)` back to hPa.
  - Uses fixed random seed for reproducible evaluation.
  - Supports height-band metrics, default bands: `0-5km`, `5-20km`, `20-60km`.
  - Savitzky-Golay smoothing is on by default; disable with `--no_smooth`.
- `ro_retrieval/evaluation/metrics.py`
  - Provides physical-space and height-band metrics.
- `ro_retrieval/model/baselines.py`
  - Contains MLP and 1D-CNN discriminative baselines.
- `src/run_baselines.py`
  - Trains/evaluates baselines for the "why diffusion" comparison.
- `src/run_ddim_diagnosis.py`
  - Diagnoses DDIM step and eta behavior.
- `src/run_pipeline.py`
  - Delegates evaluation to `src.evaluate.main` for consistency.
- `ro_retrieval/inference/predict.py`
  - Reuses shared stats logic and handles multi-channel denormalization broadcasting.
- `ro_retrieval/training/trainer.py`
  - `evaluate_test()` calls the compatible `src.evaluate` path.
- `ro_retrieval/app/streamlit_app.py`
  - Uses shared stats logic.
  - Uses `width=...` instead of deprecated `use_container_width`.
  - Uses explicit `torch.load(..., weights_only=True)` where applicable.

## Current Results

- Third-round physical-space DDPM:
  - Temperature: `RMSE=17.0831 K`, `Bias=-1.6330 K`, `CC=0.7254`
  - Pressure: `RMSE=14.7016 hPa`, `Bias=4.8111 hPa`, `CC=0.9993`
  - Humidity: `RMSE=1.3948 g/kg`, `Bias=-0.0035 g/kg`, `CC=0.8279`
- Third-round physical-space DDIM-50:
  - Temperature: `RMSE=14.2091 K`, `Bias=1.6815 K`, `CC=0.7773`
  - Pressure: `RMSE=15.7442 hPa`, `Bias=0.4174 hPa`, `CC=0.9987`
  - Humidity: `RMSE=1.3676 g/kg`, `Bias=0.1138 g/kg`, `CC=0.6587`
- Third-round DDPM humidity height-band CC:
  - `0-5km`: `0.2313`
  - `5-20km`: `0.0708`
  - `20-60km`: `0.0168`
- Third-round discriminative baselines:
  - MLP: temperature `RMSE=8.8193 K`, `CC=0.8501`; pressure `RMSE=14.8478 hPa`, `CC=0.9997`; humidity `RMSE=0.6544 g/kg`, `CC=0.8540`
  - 1D-CNN: temperature `RMSE=11.6336 K`, `CC=0.8070`; pressure `RMSE=30.2511 hPa`, `CC=0.9897`; humidity `RMSE=0.8438 g/kg`, `CC=0.8382`
- DDIM diagnosis:
  - `DDIM-100 eta=0.0` humidity `CC=0.6542`
  - `DDIM-200 eta=0.0` humidity `CC=0.6472`
  - `DDIM-100 eta=0.5` produced `NaN`

## Thesis Boundaries

- Current conclusion: diffusion retrieval is feasible and a complete system prototype exists.
- In the current `ATP -> WAP` supervised setup, the MLP baseline is overall stronger than the diffusion main model.
- Do not claim that the current diffusion model outperforms simple supervised baselines.
- Do not describe the current results as independent verification against true atmospheric state unless an external validation loop has been completed.
- Whole-profile pressure CC is easy to inflate because of monotonic vertical background structure; interpret pressure CC with height-band metrics.
- Whole-profile humidity CC must also be interpreted with height-band metrics; the overall `CC=0.8279` does not prove strong low-level structure recovery.
- Current DDIM conclusion: DDIM-50 is fully evaluated; it improves temperature but weakens humidity structural consistency relative to DDPM. Increasing steps to 100/200 did not recover humidity CC.

## Engineering Status

- Root command entry:
  - `Makefile`
- Engineering specification:
  - `docs/engineering/工程规范说明.md`
- Minimal tests:
  - `tests/test_stats_utils.py`
  - `tests/test_run_pipeline.py`
- Previously verified:
  - `make test`
  - `python src/evaluate.py --help`
  - `python src/run_pipeline.py --help`
- Demo upload sample:
  - `samples/demo_upload_atp_wap_2025_16.npz`
  - `samples/demo_upload_atp_wap_2025_16_README.json`
- Streamlit app:
  - `ro_retrieval/app/streamlit_app.py`

## Working Rules

- Keep the working tree clean and avoid adding generated artifacts by default.
- Preserve unrelated user changes. Do not reset or revert unrelated dirty files.
- Delete caches, logs, and `__pycache__` first during cleanup.
- Do not delete `Data/`, `Results/`, `experiments/`, `evaluation_results_*`, model weights, or thesis materials unless explicitly requested.
- Academic writing must keep physical interpretation precise and avoid overclaiming.
- Before commits, provide a clear Git commit message for confirmation.
- New local experiment outputs are ignored by default; force-add intentional archival artifacts with `git add -f`.
