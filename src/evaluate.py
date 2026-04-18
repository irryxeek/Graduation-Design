"""
评估入口脚本
============
批量评估 ATP+WAP / 其他掩星反演模型。

默认行为与历史实验报告保持兼容:
  - 优先使用 test 集
  - 默认评估全部测试样本
  - standardized 模式复用历史评估口径
  - physical 模式使用 stats.npy 还原到物理空间
"""

import argparse
import os
import sys
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import savgol_filter
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ro_retrieval.config import (
    DEVICE,
    PAPER_MODEL_PATH,
    PAPER_PROCESSED_DIR,
    PROJECT_ROOT,
    SAVGOL_POLYORDER,
    SAVGOL_WINDOW,
    TIMESTEPS,
)
from ro_retrieval.evaluation.metrics import (
    EvaluationReport,
    compute_bias_profile,
    compute_rmse_profile,
)
from ro_retrieval.model.diffusion import DiffusionSchedule, ddim_sample, ddpm_sample
from ro_retrieval.model.unet import ConditionalUNet1D, EnhancedConditionalUNet1D


def parse_args():
    parser = argparse.ArgumentParser(description="批量评估掩星反演模型")
    parser.add_argument(
        "--model_path",
        type=str,
        default=PAPER_MODEL_PATH,
    )
    parser.add_argument("--model_type", choices=["legacy", "enhanced"], default="enhanced")
    parser.add_argument("--sampler", choices=["ddpm", "ddim"], default="ddim")
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument(
        "--n_samples",
        type=int,
        default=0,
        help="评估样本数; 0 表示评估全部 test 样本",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="批量评估时的推理 batch size",
    )
    parser.add_argument(
        "--out_channels",
        type=int,
        default=3,
        help="输出通道数: 1=温度, 2=温度+气压, 3=温度+气压+湿度",
    )
    parser.add_argument("--data_dir", type=str, default=PAPER_PROCESSED_DIR)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--metric_space",
        choices=["standardized", "physical"],
        default="standardized",
        help="指标统计空间: standardized=历史兼容口径, physical=物理空间",
    )
    parser.add_argument(
        "--no_smooth",
        action="store_true",
        help="关闭默认的 Savitzky-Golay 平滑",
    )
    return parser.parse_args()


def _load_stats(data_dir):
    stats_path = os.path.join(data_dir, "stats.npy")
    if not os.path.exists(stats_path):
        return None

    stats = np.load(stats_path, allow_pickle=True)
    if isinstance(stats, np.ndarray) and stats.shape == ():
        stats = stats.item()
    return stats


def _load_split_arrays(data_dir):
    test_x_path = os.path.join(data_dir, "test_x.npy")
    test_y_path = os.path.join(data_dir, "test_y.npy")
    train_x_path = os.path.join(data_dir, "train_x.npy")
    train_y_path = os.path.join(data_dir, "train_y.npy")

    if os.path.exists(test_x_path) and os.path.exists(test_y_path):
        print("使用测试集进行评估")
        eval_x = np.load(test_x_path).astype(np.float32)
        eval_y = np.load(test_y_path).astype(np.float32)
    else:
        print("警告: 未找到测试集，使用训练集进行评估")
        eval_x = np.load(train_x_path).astype(np.float32)
        eval_y = np.load(train_y_path).astype(np.float32)

    train_x = np.load(train_x_path).astype(np.float32)
    train_y = np.load(train_y_path).astype(np.float32)
    return eval_x, eval_y, train_x, train_y


def _build_model(args):
    if args.model_type == "enhanced":
        return EnhancedConditionalUNet1D(
            in_channels=args.out_channels,
            cond_channels=1,
            out_channels=args.out_channels,
            use_cross_attention=True,
        ).to(DEVICE)
    return ConditionalUNet1D(
        in_channels=args.out_channels,
        cond_channels=1,
        out_channels=args.out_channels,
    ).to(DEVICE)


def _maybe_smooth(pred, enabled):
    if not enabled:
        return pred

    pred = np.array(pred, copy=True)
    if pred.ndim == 1:
        return savgol_filter(pred, SAVGOL_WINDOW, SAVGOL_POLYORDER)

    for i in range(pred.shape[0]):
        try:
            pred[i] = savgol_filter(pred[i], SAVGOL_WINDOW, SAVGOL_POLYORDER)
        except Exception:
            pass
    return pred


def main(args=None):
    if args is None:
        args = parse_args()

    if not hasattr(args, "smooth"):
        args.smooth = not getattr(args, "no_smooth", False)
    elif hasattr(args, "no_smooth") and args.no_smooth:
        args.smooth = False

    if args.save_dir is None:
        suffix = "ddim" if args.sampler == "ddim" else "ddpm"
        args.save_dir = os.path.join(PROJECT_ROOT, f"evaluation_results_{suffix}_enhanced")
    os.makedirs(args.save_dir, exist_ok=True)

    raw_x, raw_y, train_x, train_y = _load_split_arrays(args.data_dir)
    print(f"评估数据: X={raw_x.shape}, Y={raw_y.shape}")

    if raw_y.ndim == 2:
        raw_y = raw_y[:, np.newaxis, :]
    if train_y.ndim == 2:
        train_y = train_y[:, np.newaxis, :]

    args.out_channels = min(args.out_channels, raw_y.shape[1])
    raw_y = raw_y[:, :args.out_channels, :]
    train_y = train_y[:, :args.out_channels, :]

    # 历史兼容口径: 磁盘数组已标准化，但历史报告在评估时会基于 train split
    # 重新计算每高度层均值/方差，并仅对预测值执行该变换。
    x_mean = np.mean(train_x, axis=0)
    x_std = np.std(train_x, axis=0) + 1e-6
    legacy_y_mean = np.mean(train_y, axis=0)
    legacy_y_std = np.std(train_y, axis=0) + 1e-6

    physical_stats = _load_stats(args.data_dir)
    if args.metric_space == "physical":
        if physical_stats is None:
            raise FileNotFoundError(
                f"{args.data_dir} 缺少 stats.npy，无法恢复 physical 指标空间"
            )
        physical_y_mean = np.asarray(physical_stats["y_mean"], dtype=np.float32)[:args.out_channels]
        physical_y_std = np.asarray(physical_stats["y_std"], dtype=np.float32)[:args.out_channels]
    else:
        physical_y_mean = None
        physical_y_std = None

    model = _build_model(args)
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=DEVICE, weights_only=True))
        print(f"模型加载成功: {args.model_path}")
    else:
        raise FileNotFoundError(f"模型未找到: {args.model_path}")

    model.eval()
    schedule = DiffusionSchedule(TIMESTEPS, device=DEVICE)

    np.random.seed(args.seed)
    n_total = len(raw_x)
    n_eval = n_total if args.n_samples <= 0 else min(args.n_samples, n_total)
    indices = np.random.choice(n_total, n_eval, replace=False)

    var_names = ["temperature", "pressure", "humidity"][:args.out_channels]
    report = EvaluationReport(variable_names=var_names)

    batch_size = max(1, args.batch_size)
    print(
        f"\n开始批量评估 ({args.sampler.upper()}, {len(indices)} 样本, "
        f"batch_size={batch_size}, metric_space={args.metric_space})..."
    )
    all_preds = []
    all_truths = []

    for start in tqdm(range(0, len(indices), batch_size)):
        batch_indices = indices[start:start + batch_size]
        input_ba = raw_x[batch_indices]
        true_vals = raw_y[batch_indices]

        input_norm = (input_ba - x_mean) / x_std
        cond = torch.tensor(input_norm, dtype=torch.float32, device=DEVICE).unsqueeze(1)

        with torch.no_grad():
            if args.sampler == "ddim":
                gen = ddim_sample(
                    model,
                    cond,
                    shape=(len(batch_indices), args.out_channels, 301),
                    schedule=schedule,
                    ddim_steps=args.ddim_steps,
                )
            else:
                gen = ddpm_sample(
                    model,
                    cond,
                    shape=(len(batch_indices), args.out_channels, 301),
                    schedule=schedule,
                )

        preds = gen.cpu().numpy()
        if args.metric_space == "physical":
            preds_metric = preds * physical_y_std.reshape(1, -1, 1) + physical_y_mean.reshape(1, -1, 1)
            truths_metric = true_vals * physical_y_std.reshape(1, -1, 1) + physical_y_mean.reshape(1, -1, 1)
        else:
            preds_metric = preds * legacy_y_std[np.newaxis, :, :] + legacy_y_mean[np.newaxis, :, :]
            truths_metric = true_vals

        for local_idx, sample_idx in enumerate(batch_indices):
            pred = _maybe_smooth(preds_metric[local_idx], args.smooth)
            truth = truths_metric[local_idx]
            ba = input_ba[local_idx]

            all_preds.append(pred)
            all_truths.append(truth)
            report.add_sample(
                pred=pred,
                truth=truth,
                sample_idx=int(sample_idx),
                input_ba=ba,
            )

    report.print_report()
    report.save_json(
        os.path.join(args.save_dir, "evaluation_report.json"),
        metadata={
            "sampler": args.sampler,
            "ddim_steps": args.ddim_steps if args.sampler == "ddim" else None,
            "batch_size": batch_size,
            "model_path": os.path.relpath(args.model_path, PROJECT_ROOT)
            if os.path.isabs(args.model_path)
            else args.model_path,
            "model_type": args.model_type,
            "out_channels": args.out_channels,
            "data_dir": os.path.relpath(args.data_dir, PROJECT_ROOT)
            if os.path.isabs(args.data_dir)
            else args.data_dir,
            "metric_space": args.metric_space,
            "smooth": args.smooth,
            "device": str(DEVICE),
            "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
            "seed": args.seed,
        },
    )

    heights = np.linspace(0, 60, 301)

    for var_name in var_names:
        best, median, worst = report.get_sorted_results(variable=var_name, metric="rmse")
        if best is None:
            continue

        cases = [("Best", best), ("Median", median), ("Worst", worst)]
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        fig.suptitle(
            f"{var_name.upper()} Retrieval Evaluation ({args.sampler.upper()}, {args.metric_space})",
            fontsize=14,
        )

        for ax, (label, data) in zip(axes, cases):
            var_idx = var_names.index(var_name)
            metrics = data["per_var"][var_name]
            match_idx = next(i for i, item in enumerate(report.results) if item["idx"] == data["idx"])
            pred_v = all_preds[match_idx][var_idx]
            true_v = all_truths[match_idx][var_idx]

            ax.plot(true_v, heights, "k-", label="Truth", linewidth=2)
            ax.plot(
                pred_v,
                heights,
                "r--",
                label=f"Pred (RMSE={metrics['rmse']:.3f}, CC={metrics['cc']:.3f})",
                linewidth=2,
            )
            ax.set_title(f"{label} Case (#{data['idx']})")
            ax.set_xlabel(var_name)
            if ax == axes[0]:
                ax.set_ylabel("Height (km)")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(args.save_dir, f"{var_name}_comparison.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"已保存: {save_path}")

    preds_arr = np.array([pred[0] if pred.ndim == 2 and pred.shape[0] == 1 else pred.flatten()[:301] for pred in all_preds])
    truths_arr = np.array([truth[0] if truth.ndim == 2 and truth.shape[0] == 1 else truth.flatten()[:301] for truth in all_truths])
    if preds_arr.ndim == 3:
        preds_arr = preds_arr[:, 0, :]
        truths_arr = truths_arr[:, 0, :]

    rmse_prof = compute_rmse_profile(preds_arr, truths_arr, heights)
    bias_prof = compute_bias_profile(preds_arr, truths_arr, heights)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    ax1.plot(rmse_prof, heights, "r-", linewidth=2)
    ax1.set_xlabel("RMSE")
    ax1.set_ylabel("Height (km)")
    ax1.set_title("RMSE Profile")
    ax1.grid(True, alpha=0.3)

    ax2.plot(bias_prof, heights, "b-", linewidth=2)
    ax2.axvline(x=0, color="k", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Bias")
    ax2.set_title("Bias Profile")
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f"Height-resolved Metrics ({args.sampler.upper()}, {args.metric_space})", fontsize=14)
    plt.tight_layout()
    profile_path = os.path.join(args.save_dir, "rmse_bias_profile.png")
    plt.savefig(profile_path, dpi=150)
    plt.close()
    print(f"已保存: {profile_path}")

    print(f"\n评估完成! 结果保存在: {args.save_dir}")


if __name__ == "__main__":
    main()
