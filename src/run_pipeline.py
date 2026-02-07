"""
端到端闭环流水线
================
一键执行: 数据处理 → 训练 → 推理 → 评估

用法:
  # 全流程 (从原始数据开始)
  python src/run_pipeline.py --all

  # 仅训练 + 评估 (已有处理好的数据)
  python src/run_pipeline.py --train --evaluate

  # 仅评估已有模型
  python src/run_pipeline.py --evaluate --model_path enhanced_ro_diffusion_best.pth

  # 自定义各阶段参数
  python src/run_pipeline.py --all --mode era5 --model_type enhanced --epochs 50
"""

import os
import sys
import argparse
import time
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ro_retrieval.config import (
    DEVICE, PROCESSED_DIR, PROJECT_ROOT,
    TIMESTEPS, BATCH_SIZE, EPOCHS, LEARNING_RATE,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="RO-Retrieval 端到端闭环流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 阶段选择
    stage = p.add_argument_group("流程阶段")
    stage.add_argument("--all", action="store_true", help="执行全部阶段")
    stage.add_argument("--process", action="store_true", help="执行数据处理")
    stage.add_argument("--train", action="store_true", help="执行训练")
    stage.add_argument("--evaluate", action="store_true", help="执行评估")

    # 数据处理参数
    data = p.add_argument_group("数据处理")
    data.add_argument("--atm_dir", type=str,
                      default=os.path.join(PROJECT_ROOT, "Data", "Sample",
                                           "atmPrf_nrt_2026_001"))
    data.add_argument("--wet_dir", type=str,
                      default=os.path.join(PROJECT_ROOT, "Data", "Sample",
                                           "wetPf2_nrt_2026_001"))
    data.add_argument("--era5_dir", type=str,
                      default=os.path.join(PROJECT_ROOT, "Data", "Sample"))
    data.add_argument("--data_dir", type=str, default=PROCESSED_DIR,
                      help="处理后数据存放目录")
    data.add_argument("--mode", choices=["wet", "era5"], default="wet",
                      help="真值来源: wet=wetPf2, era5=ERA5再分析")
    data.add_argument("--no-strict-qc", action="store_true")

    # 训练参数
    train = p.add_argument_group("训练")
    train.add_argument("--model_type", choices=["legacy", "enhanced"],
                       default="enhanced")
    train.add_argument("--var_mode", choices=["single", "multi"],
                       default="single", help="single=仅温度, multi=温度+压力+湿度")
    train.add_argument("--epochs", type=int, default=EPOCHS)
    train.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    train.add_argument("--lr", type=float, default=LEARNING_RATE)
    train.add_argument("--patience", type=int, default=20,
                       help="Early Stopping 容忍轮数")

    # 评估参数
    evl = p.add_argument_group("评估")
    evl.add_argument("--model_path", type=str, default=None,
                     help="模型权重路径 (不指定则使用训练产生的 best 模型)")
    evl.add_argument("--sampler", choices=["ddpm", "ddim"], default="ddim")
    evl.add_argument("--ddim_steps", type=int, default=50)
    evl.add_argument("--n_eval_samples", type=int, default=50)

    return p.parse_args()


# =====================================================================
# Stage 1: 数据处理
# =====================================================================
def stage_process(args):
    print("\n" + "=" * 60)
    print("  Stage 1: 数据处理")
    print("=" * 60)

    from ro_retrieval.data.process_enhanced import run_enhanced_pipeline

    result = run_enhanced_pipeline(
        atm_root=args.atm_dir,
        wet_root=args.wet_dir,
        output_dir=args.data_dir,
        era5_root=args.era5_dir if args.mode == "era5" else None,
        strict_qc=not getattr(args, 'no_strict_qc', False),
        do_split=True,
    )

    if result is None:
        print("数据处理失败!")
        return False

    X, Y, report = result
    print(f"数据处理完成: X={X.shape}, Y={Y.shape}")
    return True


# =====================================================================
# Stage 2: 训练
# =====================================================================
def stage_train(args):
    print("\n" + "=" * 60)
    print("  Stage 2: 模型训练")
    print("=" * 60)

    from ro_retrieval.training.trainer import Trainer

    trainer = Trainer(
        data_dir=args.data_dir,
        model_type=args.model_type,
        mode=args.var_mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_dir=PROJECT_ROOT,
        patience=args.patience,
    )

    model = trainer.train()

    # 记录最佳模型路径供评估阶段使用
    prefix = "enhanced_ro_diffusion" if args.model_type == "enhanced" else "ro_diffusion"
    best_path = os.path.join(PROJECT_ROOT, f"{prefix}_best.pth")
    args._best_model_path = best_path
    return True


# =====================================================================
# Stage 3: 评估
# =====================================================================
def stage_evaluate(args):
    print("\n" + "=" * 60)
    print("  Stage 3: 模型评估")
    print("=" * 60)

    import torch
    import numpy as np
    from scipy.signal import savgol_filter
    from tqdm import tqdm

    from ro_retrieval.model.unet import ConditionalUNet1D, EnhancedConditionalUNet1D
    from ro_retrieval.model.diffusion import DiffusionSchedule, ddpm_sample, ddim_sample
    from ro_retrieval.evaluation.metrics import (
        EvaluationReport, compute_rmse_profile, compute_bias_profile,
    )
    from ro_retrieval.config import SAVGOL_WINDOW, SAVGOL_POLYORDER

    # 确定模型路径
    model_path = args.model_path
    if model_path is None:
        model_path = getattr(args, '_best_model_path', None)
    if model_path is None:
        # 自动搜索最新的 best 模型
        for prefix in ["enhanced_ro_diffusion", "ro_diffusion"]:
            candidate = os.path.join(PROJECT_ROOT, f"{prefix}_best.pth")
            if os.path.exists(candidate):
                model_path = candidate
                break
    if model_path and not os.path.isabs(model_path):
        model_path = os.path.join(PROJECT_ROOT, model_path)

    if model_path is None or not os.path.exists(model_path):
        print(f"未找到模型权重: {model_path}")
        return False

    print(f"使用模型: {model_path}")

    # 加载数据 — 优先使用 test 集
    test_x_path = os.path.join(args.data_dir, "test_x.npy")
    test_y_path = os.path.join(args.data_dir, "test_y.npy")
    if os.path.exists(test_x_path):
        print("使用 test 集进行评估")
        raw_x = np.load(test_x_path).astype(np.float32)
        raw_y = np.load(test_y_path).astype(np.float32)
    else:
        print("未检测到 test 集, 使用 train 集")
        raw_x = np.load(os.path.join(args.data_dir, "train_x.npy")).astype(np.float32)
        raw_y = np.load(os.path.join(args.data_dir, "train_y.npy")).astype(np.float32)

    # 需要从训练集计算统计量 (标准化必须与训练时一致)
    train_x = np.load(os.path.join(args.data_dir, "train_x.npy")).astype(np.float32)
    train_y = np.load(os.path.join(args.data_dir, "train_y.npy")).astype(np.float32)
    x_mean = np.mean(train_x, axis=0)
    x_std = np.std(train_x, axis=0) + 1e-6
    y_mean_np = np.mean(train_y, axis=0)
    y_std_np = np.std(train_y, axis=0) + 1e-6
    y_mean = torch.tensor(y_mean_np).float().to(DEVICE)
    y_std = torch.tensor(y_std_np).float().to(DEVICE)

    # 判断通道数
    if raw_y.ndim == 3:
        out_ch = raw_y.shape[1]
    else:
        out_ch = 1

    # 加载模型
    state_dict = torch.load(model_path, map_location=DEVICE)
    is_enhanced = any(k.startswith("time_embed.") for k in state_dict.keys())

    if is_enhanced:
        model = EnhancedConditionalUNet1D(
            in_channels=out_ch, cond_channels=1, out_channels=out_ch,
            use_cross_attention=True,
        ).to(DEVICE)
    else:
        model = ConditionalUNet1D(
            in_channels=out_ch, cond_channels=1, out_channels=out_ch,
        ).to(DEVICE)

    model.load_state_dict(state_dict)
    model.eval()

    schedule = DiffusionSchedule(TIMESTEPS, device=DEVICE)

    # 采样评估
    np.random.seed(42)
    n_total = len(raw_x)
    n_eval = min(args.n_eval_samples, n_total)
    indices = np.random.choice(n_total, n_eval, replace=False)

    var_names = ["temperature", "pressure", "humidity"][:out_ch]
    report = EvaluationReport(variable_names=var_names)

    save_dir = os.path.join(PROJECT_ROOT, f"evaluation_results_{args.sampler}_pipeline")
    os.makedirs(save_dir, exist_ok=True)

    all_preds, all_truths = [], []

    print(f"开始评估 ({args.sampler.upper()}, {n_eval} 样本)...")
    for idx in tqdm(indices):
        input_ba = raw_x[idx]
        true_vals = raw_y[idx]

        input_norm = (input_ba - x_mean) / x_std
        cond = torch.tensor(input_norm).float().unsqueeze(0).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            if args.sampler == "ddim":
                gen = ddim_sample(model, cond, shape=(1, out_ch, 301),
                                  schedule=schedule, ddim_steps=args.ddim_steps)
            else:
                gen = ddpm_sample(model, cond, shape=(1, out_ch, 301),
                                  schedule=schedule)

        pred = gen.squeeze(0).cpu() * y_std.cpu() + y_mean.cpu()
        pred = pred.numpy()

        if pred.ndim == 1:
            pred = savgol_filter(pred, SAVGOL_WINDOW, SAVGOL_POLYORDER)
        else:
            for i in range(pred.shape[0]):
                try:
                    pred[i] = savgol_filter(pred[i], SAVGOL_WINDOW, SAVGOL_POLYORDER)
                except Exception:
                    pass

        if true_vals.ndim == 1 and pred.ndim == 2:
            true_vals = true_vals[np.newaxis, :]

        all_preds.append(pred)
        all_truths.append(true_vals)

        report.add_sample(
            pred=pred if pred.ndim > 1 else pred[np.newaxis, :],
            truth=true_vals if true_vals.ndim > 1 else true_vals[np.newaxis, :],
            sample_idx=int(idx),
            input_ba=input_ba,
        )

    report.print_report()
    report.save_json(os.path.join(save_dir, "evaluation_report.json"))

    # 绘图: RMSE / Bias 随高度分布
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    heights = np.linspace(0, 60, 301)

    if len(all_preds) > 0:
        preds_arr = np.array([p.flatten()[:301] for p in all_preds])
        truths_arr = np.array([t.flatten()[:301] for t in all_truths])

        rmse_prof = compute_rmse_profile(preds_arr, truths_arr, heights)
        bias_prof = compute_bias_profile(preds_arr, truths_arr, heights)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
        ax1.plot(rmse_prof, heights, 'r-', linewidth=2)
        ax1.set_xlabel('RMSE')
        ax1.set_ylabel('Height (km)')
        ax1.set_title('RMSE Profile')
        ax1.grid(True, alpha=0.3)

        ax2.plot(bias_prof, heights, 'b-', linewidth=2)
        ax2.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Bias')
        ax2.set_title('Bias Profile')
        ax2.grid(True, alpha=0.3)

        plt.suptitle(f'Height-resolved Metrics ({args.sampler.upper()})', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "rmse_bias_profile.png"), dpi=150)
        plt.close()

    # Best / Median / Worst 对比图
    for var_name in var_names:
        best, median, worst = report.get_sorted_results(variable=var_name, metric="rmse")
        if best is None:
            continue

        cases = [("Best", best), ("Median", median), ("Worst", worst)]
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        fig.suptitle(f"{var_name.upper()} Retrieval ({args.sampler.upper()})", fontsize=14)

        var_idx = var_names.index(var_name)
        for ax, (label, data) in zip(axes, cases):
            m = data["per_var"][var_name]
            match_idx = next(
                i for i, r in enumerate(report.results) if r["idx"] == data["idx"]
            )
            pred_v = all_preds[match_idx]
            true_v = all_truths[match_idx]

            if pred_v.ndim > 1 and var_idx < pred_v.shape[0]:
                pv, tv = pred_v[var_idx], true_v[var_idx] if true_v.ndim > 1 else true_v
            else:
                pv, tv = pred_v.flatten(), true_v.flatten()

            ax.plot(tv, heights, 'k-', label='Truth', linewidth=2)
            ax.plot(pv, heights, 'r--',
                    label=f'Pred (RMSE={m["rmse"]:.2f})', linewidth=2)
            ax.set_title(f'{label} (#{data["idx"]})')
            ax.set_xlabel(var_name)
            if ax == axes[0]:
                ax.set_ylabel('Height (km)')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{var_name}_comparison.png"), dpi=150)
        plt.close()

    print(f"\n评估完成! 结果保存在: {save_dir}")
    return True


# =====================================================================
# 主流程
# =====================================================================
def main():
    args = parse_args()

    # 如果没有指定任何阶段, 默认 --all
    if not (args.all or args.process or args.train or args.evaluate):
        args.all = True

    do_process = args.all or args.process
    do_train = args.all or args.train
    do_evaluate = args.all or args.evaluate

    print("=" * 60)
    print("  RO-Retrieval 端到端闭环流水线")
    print("=" * 60)
    print(f"  阶段: {'处理' if do_process else '-'} → "
          f"{'训练' if do_train else '-'} → "
          f"{'评估' if do_evaluate else '-'}")
    print(f"  设备: {DEVICE}")

    t0 = time.time()

    # Stage 1
    if do_process:
        if not stage_process(args):
            print("流水线在数据处理阶段终止")
            return

    # Stage 2
    if do_train:
        if not stage_train(args):
            print("流水线在训练阶段终止")
            return

    # Stage 3
    if do_evaluate:
        if not stage_evaluate(args):
            print("流水线在评估阶段终止")
            return

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  流水线全部完成! 总耗时: {elapsed / 60:.1f} 分钟")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
