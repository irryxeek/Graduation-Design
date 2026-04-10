"""
评估入口脚本
============
使用 ro_retrieval 包进行批量评估。

用法:
  python src/evaluate.py --sampler ddim --n_samples 50
  python src/evaluate.py --model_path enhanced_ro_diffusion_best.pth --model_type enhanced --out_channels 3

支持:
  - RMSE / Bias / CC 完整指标体系
  - 单变量和多变量评估
  - DDPM / DDIM 两种采样方式
  - 自动生成评估报告 (JSON + 图表)
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import savgol_filter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ro_retrieval.config import (
    DEVICE, PROCESSED_DIR, PROJECT_ROOT,
    TIMESTEPS, SAVGOL_WINDOW, SAVGOL_POLYORDER,
)
from ro_retrieval.model.unet import ConditionalUNet1D, EnhancedConditionalUNet1D
from ro_retrieval.model.diffusion import DiffusionSchedule, ddpm_sample, ddim_sample
from ro_retrieval.evaluation.metrics import (
    EvaluationReport, compute_rmse_profile, compute_bias_profile,
)


def parse_args():
    parser = argparse.ArgumentParser(description="批量评估掩星反演模型")
    parser.add_argument("--model_path", type=str,
                        default=os.path.join(PROJECT_ROOT, "ro_diffusion_epoch_100.pth"))
    parser.add_argument("--model_type", choices=["legacy", "enhanced"], default="legacy")
    parser.add_argument("--sampler", choices=["ddpm", "ddim"], default="ddim")
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1,
                        help="批量评估时的推理 batch size; 建议 GPU 下使用 16/32/64")
    parser.add_argument("--out_channels", type=int, default=1,
                        help="输出通道数: 1=温度, 2=温度+气压, 3=温度+气压+湿度")
    parser.add_argument("--data_dir", type=str, default=PROCESSED_DIR)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.save_dir is None:
        suffix = "ddim" if args.sampler == "ddim" else "ddpm"
        args.save_dir = os.path.join(PROJECT_ROOT, f"evaluation_results_{suffix}_enhanced")
    os.makedirs(args.save_dir, exist_ok=True)

    # 1. 加载数据 - 优先使用测试集
    test_x_path = os.path.join(args.data_dir, "test_x.npy")
    test_y_path = os.path.join(args.data_dir, "test_y.npy")
    train_x_path = os.path.join(args.data_dir, "train_x.npy")
    train_y_path = os.path.join(args.data_dir, "train_y.npy")

    # 优先使用测试集进行评估
    if os.path.exists(test_x_path) and os.path.exists(test_y_path):
        print("使用测试集进行评估")
        raw_x = np.load(test_x_path).astype(np.float32)
        raw_y = np.load(test_y_path).astype(np.float32)
        # 使用训练集的统计量进行标准化 (保持一致性)
        train_x = np.load(train_x_path).astype(np.float32)
        train_y = np.load(train_y_path).astype(np.float32)
        x_mean = np.mean(train_x, axis=0)
        x_std = np.std(train_x, axis=0) + 1e-6
        if train_y.ndim == 3:
            y_mean_np = np.mean(train_y, axis=0)
            y_std_np = np.std(train_y, axis=0) + 1e-6
        else:
            y_mean_np = np.mean(train_y, axis=0)
            y_std_np = np.std(train_y, axis=0) + 1e-6
    else:
        print("警告: 未找到测试集，使用训练集进行评估 (不推荐)")
        raw_x = np.load(train_x_path).astype(np.float32)
        raw_y = np.load(train_y_path).astype(np.float32)
        x_mean = np.mean(raw_x, axis=0)
        x_std = np.std(raw_x, axis=0) + 1e-6
        if raw_y.ndim == 3:
            y_mean_np = np.mean(raw_y, axis=0)
            y_std_np = np.std(raw_y, axis=0) + 1e-6
        else:
            y_mean_np = np.mean(raw_y, axis=0)
            y_std_np = np.std(raw_y, axis=0) + 1e-6

    print(f"评估数据: X={raw_x.shape}, Y={raw_y.shape}")

    y_mean = torch.tensor(y_mean_np).float().to(DEVICE)
    y_std = torch.tensor(y_std_np).float().to(DEVICE)

    # 2. 加载模型
    out_ch = args.out_channels
    if args.model_type == "enhanced":
        model = EnhancedConditionalUNet1D(
            in_channels=out_ch, cond_channels=1, out_channels=out_ch,
            use_cross_attention=True,
        ).to(DEVICE)
    else:
        model = ConditionalUNet1D(
            in_channels=out_ch, cond_channels=1, out_channels=out_ch,
        ).to(DEVICE)

    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
        print(f"模型加载成功: {args.model_path}")
    else:
        print(f"模型未找到: {args.model_path}")
        return

    model.eval()
    schedule = DiffusionSchedule(TIMESTEPS, device=DEVICE)

    # 3. 随机抽样
    np.random.seed(args.seed)
    n_total = len(raw_x)
    indices = np.random.choice(n_total, min(args.n_samples, n_total), replace=False)

    # 确定变量名
    if out_ch == 3:
        var_names = ["temperature", "pressure", "humidity"]
    elif out_ch == 2:
        var_names = ["temperature", "pressure"]
    else:
        var_names = ["temperature"]

    report = EvaluationReport(variable_names=var_names)

    # 4. 逐样本推理与评估
    batch_size = max(1, args.batch_size)
    print(f"\n开始批量评估 ({args.sampler.upper()}, {len(indices)} 样本, batch_size={batch_size})...")
    all_preds = []
    all_truths = []

    for start in tqdm(range(0, len(indices), batch_size)):
        batch_indices = indices[start:start + batch_size]
        input_ba = raw_x[batch_indices]
        true_vals = raw_y[batch_indices]

        # 标准化
        input_norm = (input_ba - x_mean) / x_std
        cond = torch.tensor(input_norm).float().unsqueeze(1).to(DEVICE)

        # 推理
        with torch.no_grad():
            if args.sampler == "ddim":
                gen = ddim_sample(
                    model,
                    cond,
                    shape=(len(batch_indices), out_ch, 301),
                    schedule=schedule,
                    ddim_steps=args.ddim_steps,
                )
            else:
                gen = ddpm_sample(
                    model,
                    cond,
                    shape=(len(batch_indices), out_ch, 301),
                    schedule=schedule,
                )

        # 反归一化
        preds = gen.cpu() * y_std.cpu().unsqueeze(0) + y_mean.cpu().unsqueeze(0)
        preds = preds.numpy()  # (B, out_ch, 301)

        # 平滑并写入报告
        for local_idx, sample_idx in enumerate(batch_indices):
            pred = preds[local_idx]
            truth = true_vals[local_idx]
            ba = input_ba[local_idx]

            if pred.ndim == 1:
                pred = savgol_filter(pred, SAVGOL_WINDOW, SAVGOL_POLYORDER)
            else:
                for i in range(pred.shape[0]):
                    try:
                        pred[i] = savgol_filter(pred[i], SAVGOL_WINDOW, SAVGOL_POLYORDER)
                    except Exception:
                        pass

            if truth.ndim == 1 and pred.ndim == 2:
                truth = truth[np.newaxis, :]

            all_preds.append(pred)
            all_truths.append(truth)

            report.add_sample(
                pred=pred if pred.ndim > 1 else pred[np.newaxis, :],
                truth=truth if truth.ndim > 1 else truth[np.newaxis, :],
                sample_idx=int(sample_idx),
                input_ba=ba,
            )

    # 5. 输出评估报告
    report.print_report()
    report.save_json(os.path.join(args.save_dir, "evaluation_report.json"))

    # 6. 绘图
    heights = np.linspace(0, 60, 301)

    # 6a. Best / Median / Worst 对比图
    for var_name in var_names:
        best, median, worst = report.get_sorted_results(variable=var_name, metric="rmse")
        if best is None:
            continue

        cases = [("Best", best), ("Median", median), ("Worst", worst)]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        fig.suptitle(f"{var_name.upper()} Retrieval Evaluation ({args.sampler.upper()})",
                     fontsize=14)

        for ax, (label, data) in zip(axes, cases):
            var_idx = var_names.index(var_name)
            m = data["per_var"][var_name]

            # 查找对应的预测和真值
            match_idx = next(
                i for i, r in enumerate(report.results) if r["idx"] == data["idx"]
            )
            pred_v = all_preds[match_idx]
            true_v = all_truths[match_idx]

            if pred_v.ndim > 1 and var_idx < pred_v.shape[0]:
                pv = pred_v[var_idx]
                tv = true_v[var_idx] if true_v.ndim > 1 else true_v
            else:
                pv = pred_v.flatten()
                tv = true_v.flatten()

            ax.plot(tv, heights, 'k-', label='Truth', linewidth=2)
            ax.plot(pv, heights, 'r--',
                    label=f'Pred (RMSE={m["rmse"]:.2f}, Bias={m["bias"]:.2f}, CC={m["cc"]:.3f})',
                    linewidth=2)
            ax.set_title(f'{label} Case (#{data["idx"]})')
            ax.set_xlabel(f'{var_name}')
            if ax == axes[0]:
                ax.set_ylabel('Height (km)')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(args.save_dir, f"{var_name}_comparison.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"已保存: {save_path}")

    # 6b. RMSE Profile 图 (逐高度层)
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
        plt.savefig(os.path.join(args.save_dir, "rmse_bias_profile.png"), dpi=150)
        plt.close()
        print(f"已保存: rmse_bias_profile.png")

    print(f"\n评估完成! 结果保存在: {args.save_dir}")


if __name__ == "__main__":
    main()
