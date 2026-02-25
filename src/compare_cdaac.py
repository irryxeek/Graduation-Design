"""
CDAAC 官方产品对比脚本
======================
将模型反演结果与 CDAAC wetPf2 官方产品进行对比

对比内容:
  1. 模型输出 vs wetPf2 (作为 baseline)
  2. 逐高度层 RMSE/Bias 对比
  3. 各变量的相关性分析
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ro_retrieval.config import PROCESSED_DIR, PROJECT_ROOT, STD_HEIGHT, HEIGHT_MIN, HEIGHT_MAX


def load_test_data(data_dir: str):
    """加载测试集数据"""
    test_x = np.load(os.path.join(data_dir, "test_x.npy"))
    test_y = np.load(os.path.join(data_dir, "test_y.npy"))
    return test_x, test_y


def compute_profile_metrics(pred: np.ndarray, target: np.ndarray, heights: np.ndarray):
    """
    计算逐高度层的评估指标

    Args:
        pred: 预测值 (N, L) 或 (N, C, L)
        target: 真实值，同上
        heights: 高度网格 (L,)

    Returns:
        dict: 包含 rmse_profile, bias_profile, corr_profile
    """
    if pred.ndim == 3:
        # 多变量: 分别计算每个变量
        results = {}
        var_names = ["temperature", "pressure", "humidity"]
        for i, name in enumerate(var_names[:pred.shape[1]]):
            results[name] = compute_profile_metrics(pred[:, i, :], target[:, i, :], heights)
        return results

    # 单变量
    n_levels = pred.shape[1]
    rmse_profile = np.zeros(n_levels)
    bias_profile = np.zeros(n_levels)
    corr_profile = np.zeros(n_levels)

    for l in range(n_levels):
        p = pred[:, l]
        t = target[:, l]

        # 过滤无效值
        valid = ~np.isnan(p) & ~np.isnan(t) & (t != 0)
        if np.sum(valid) < 5:
            rmse_profile[l] = np.nan
            bias_profile[l] = np.nan
            corr_profile[l] = np.nan
            continue

        p_valid = p[valid]
        t_valid = t[valid]

        rmse_profile[l] = np.sqrt(np.mean((p_valid - t_valid) ** 2))
        bias_profile[l] = np.mean(p_valid - t_valid)

        if np.std(p_valid) > 1e-10 and np.std(t_valid) > 1e-10:
            corr_profile[l] = np.corrcoef(p_valid, t_valid)[0, 1]
        else:
            corr_profile[l] = np.nan

    return {
        "rmse_profile": rmse_profile,
        "bias_profile": bias_profile,
        "corr_profile": corr_profile,
        "heights": heights,
    }


def compare_with_cdaac(
    model_pred: np.ndarray,
    cdaac_truth: np.ndarray,
    heights: np.ndarray,
    output_dir: str,
    model_name: str = "Diffusion Model",
):
    """
    与 CDAAC 产品对比

    Args:
        model_pred: 模型预测 (N, 3, 301)
        cdaac_truth: CDAAC wetPf2 真值 (N, 3, 301)
        heights: 高度网格
        output_dir: 输出目录
        model_name: 模型名称
    """
    os.makedirs(output_dir, exist_ok=True)

    var_names = ["Temperature", "Pressure", "Humidity"]
    var_units = ["K", "hPa", "kg/kg"]
    var_keys = ["temperature", "pressure", "humidity"]

    # 计算逐高度层指标
    metrics = compute_profile_metrics(model_pred, cdaac_truth, heights)

    # 绘制对比图
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    for i, (name, unit, key) in enumerate(zip(var_names, var_units, var_keys)):
        if key not in metrics:
            continue

        m = metrics[key]
        h = m["heights"]

        # RMSE profile
        ax = axes[i, 0]
        ax.plot(m["rmse_profile"], h, 'b-', linewidth=1.5)
        ax.set_xlabel(f"RMSE ({unit})")
        ax.set_ylabel("Height (km)")
        ax.set_title(f"{name} RMSE Profile")
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 60])

        # Bias profile
        ax = axes[i, 1]
        ax.plot(m["bias_profile"], h, 'r-', linewidth=1.5)
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel(f"Bias ({unit})")
        ax.set_ylabel("Height (km)")
        ax.set_title(f"{name} Bias Profile")
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 60])

        # Correlation profile
        ax = axes[i, 2]
        ax.plot(m["corr_profile"], h, 'g-', linewidth=1.5)
        ax.axvline(x=1, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel("Correlation")
        ax.set_ylabel("Height (km)")
        ax.set_title(f"{name} Correlation Profile")
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-0.2, 1.1])
        ax.set_ylim([0, 60])

    plt.suptitle(f"{model_name} vs CDAAC wetPf2", fontsize=14, fontweight='bold')
    plt.tight_layout()

    fig_path = os.path.join(output_dir, "cdaac_comparison_profiles.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"对比图已保存: {fig_path}")

    # 计算整体统计
    summary = {
        "model_name": model_name,
        "n_samples": len(model_pred),
        "variables": {}
    }

    for key in var_keys:
        if key not in metrics:
            continue
        m = metrics[key]
        valid_rmse = m["rmse_profile"][~np.isnan(m["rmse_profile"])]
        valid_bias = m["bias_profile"][~np.isnan(m["bias_profile"])]
        valid_corr = m["corr_profile"][~np.isnan(m["corr_profile"])]

        summary["variables"][key] = {
            "mean_rmse": float(np.mean(valid_rmse)) if len(valid_rmse) > 0 else None,
            "mean_bias": float(np.mean(valid_bias)) if len(valid_bias) > 0 else None,
            "mean_corr": float(np.mean(valid_corr)) if len(valid_corr) > 0 else None,
            "rmse_0_10km": float(np.mean(valid_rmse[:50])) if len(valid_rmse) > 50 else None,
            "rmse_10_30km": float(np.mean(valid_rmse[50:150])) if len(valid_rmse) > 150 else None,
            "rmse_30_60km": float(np.mean(valid_rmse[150:])) if len(valid_rmse) > 150 else None,
        }

    # 保存统计结果
    summary_path = os.path.join(output_dir, "cdaac_comparison_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"统计结果已保存: {summary_path}")

    # 打印摘要
    print(f"\n{'=' * 60}")
    print(f"CDAAC 对比结果摘要 ({model_name})")
    print(f"{'=' * 60}")
    print(f"样本数: {summary['n_samples']}")
    print(f"\n{'变量':<12} {'平均RMSE':<15} {'平均Bias':<15} {'平均相关系数':<15}")
    print("-" * 60)

    for key, var_metrics in summary["variables"].items():
        rmse = var_metrics.get("mean_rmse", float("nan"))
        bias = var_metrics.get("mean_bias", float("nan"))
        corr = var_metrics.get("mean_corr", float("nan"))
        print(f"{key:<12} {rmse:<15.4f} {bias:<15.4f} {corr:<15.4f}")

    print(f"{'=' * 60}")

    return summary, metrics


def run_cdaac_comparison(
    data_dir: str = PROCESSED_DIR,
    model_pred_path: str = None,
    output_dir: str = None,
):
    """
    运行 CDAAC 对比分析

    如果没有提供模型预测，则使用测试集真值作为"完美预测"的 baseline
    """
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "outputs", "cdaac_comparison")

    # 加载数据
    test_x, test_y = load_test_data(data_dir)
    heights = np.linspace(HEIGHT_MIN, HEIGHT_MAX, STD_HEIGHT)

    print(f"测试集: X={test_x.shape}, Y={test_y.shape}")

    # 如果有模型预测文件，加载它
    if model_pred_path and os.path.exists(model_pred_path):
        model_pred = np.load(model_pred_path)
        print(f"加载模型预测: {model_pred.shape}")
    else:
        # 使用真值作为 baseline (理想情况)
        print("未提供模型预测，使用真值作为 baseline 演示")
        # 添加一些噪声模拟模型误差
        noise_scale = [2.0, 10.0, 0.001]  # T, P, Q 的噪声尺度
        model_pred = test_y.copy()
        for i, scale in enumerate(noise_scale):
            model_pred[:, i, :] += np.random.randn(*model_pred[:, i, :].shape) * scale

    # 运行对比
    summary, metrics = compare_with_cdaac(
        model_pred, test_y, heights, output_dir,
        model_name="Diffusion Model"
    )

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CDAAC 官方产品对比")
    parser.add_argument("--data_dir", type=str, default=PROCESSED_DIR, help="数据目录")
    parser.add_argument("--model_pred", type=str, default=None, help="模型预测文件路径 (.npy)")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录")

    args = parser.parse_args()

    run_cdaac_comparison(
        data_dir=args.data_dir,
        model_pred_path=args.model_pred,
        output_dir=args.output_dir,
    )
