"""运行 MLP + MC Dropout 不确定性评估。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import savgol_filter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ro_retrieval.config import DEVICE, PAPER_PROCESSED_DIR, PROJECT_ROOT, SAVGOL_POLYORDER, SAVGOL_WINDOW
from ro_retrieval.model.baselines import MLPBaseline1D
from ro_retrieval.stats_utils import load_stats_from_dir


def parse_args():
    parser = argparse.ArgumentParser(description="运行 MLP + MC Dropout 不确定性评估")
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.path.join(
            PROJECT_ROOT,
            "experiments",
            "baseline_mlp_20260419T224628Z",
            "mlp_best.pth",
        ),
    )
    parser.add_argument("--data_dir", type=str, default=PAPER_PROCESSED_DIR)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--n_cases", type=int, default=50)
    parser.add_argument("--n_repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_root", type=str, default=os.path.join(PROJECT_ROOT, "experiments"))
    parser.add_argument("--case_indices", type=str, default="")
    parser.add_argument("--smooth", action="store_true", default=False)
    return parser.parse_args()


def _load_arrays(data_dir, split):
    x = np.load(os.path.join(data_dir, f"{split}_x.npy"), mmap_mode="r")
    y = np.load(os.path.join(data_dir, f"{split}_y.npy"), mmap_mode="r")
    if y.ndim == 2:
        y = y[:, np.newaxis, :]
    return x, y


def _maybe_smooth_batch(preds, enabled):
    if not enabled:
        return preds
    preds = np.array(preds, copy=True)
    for sample_idx in range(preds.shape[0]):
        for var_idx in range(preds.shape[1]):
            try:
                preds[sample_idx, var_idx] = savgol_filter(
                    preds[sample_idx, var_idx], SAVGOL_WINDOW, SAVGOL_POLYORDER
                )
            except Exception:
                pass
    return preds


def _restore_physical(values, stats):
    restored = values * stats["y_std"].reshape(1, -1, 1) + stats["y_mean"].reshape(1, -1, 1)
    if stats.get("pressure_log_transformed", False) and restored.shape[1] >= 2:
        restored[:, 1, :] = np.power(10.0, restored[:, 1, :], dtype=np.float32)
    return restored


def _select_indices(args, n_total):
    if args.case_indices.strip():
        return [int(item.strip()) for item in args.case_indices.split(",") if item.strip()]
    rng = np.random.default_rng(args.seed)
    return rng.choice(n_total, size=min(args.n_cases, n_total), replace=False).tolist()


def _compute_band_metrics(mean_preds, lower, upper, truths, heights, band_specs, var_names):
    result = {}
    for label, low, high in band_specs:
        mask = (heights >= low) & (heights <= high if np.isclose(high, heights.max()) else heights < high)
        band = {}
        for var_idx, var_name in enumerate(var_names):
            truth = truths[:, var_idx, :][:, mask]
            lo = lower[:, var_idx, :][:, mask]
            hi = upper[:, var_idx, :][:, mask]
            mean_pred = mean_preds[:, var_idx, :][:, mask]
            covered = ((truth >= lo) & (truth <= hi)).astype(np.float32)
            band[var_name] = {
                "coverage_95": float(np.mean(covered)),
                "mean_interval_width": float(np.mean(hi - lo)),
                "mean_abs_error": float(np.mean(np.abs(mean_pred - truth))),
            }
        result[label] = band
    return result


def _write_summary_md(path, summary):
    lines = [
        "# MLP + MC Dropout 不确定性摘要",
        "",
        f"- 样本数: `{summary['metadata']['n_cases']}`",
        f"- 每样本重复采样次数: `{summary['metadata']['n_repeats']}`",
        f"- 模型: `{summary['metadata']['model_name']}`",
        f"- 指标空间: `{summary['metadata']['metric_space']}`",
        "",
        "## 整体结果",
        "",
        "| 变量 | 95%区间覆盖率 | 平均区间宽度 | 均值预测MAE | 单位 |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for var_name, item in summary["global_summary"].items():
        lines.append(
            f"| {var_name} | {item['coverage_95']:.4f} | {item['mean_interval_width']:.4f} | "
            f"{item['mean_abs_error']:.4f} | {item['unit']} |"
        )
    lines.extend(["", "## 高度分层结果", ""])
    for band, band_metrics in summary["height_band_summary"].items():
        lines.append(f"### {band}")
        lines.append("")
        lines.append("| 变量 | 95%区间覆盖率 | 平均区间宽度 | 均值预测MAE |")
        lines.append("| --- | ---: | ---: | ---: |")
        for var_name, item in band_metrics.items():
            lines.append(
                f"| {var_name} | {item['coverage_95']:.4f} | {item['mean_interval_width']:.4f} | {item['mean_abs_error']:.4f} |"
            )
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _enable_dropout_only(model: nn.Module):
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def main():
    args = parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    save_dir = os.path.join(args.save_root, f"baseline_mlp_mcdo_{args.split}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    print("加载数据...")
    test_x, test_y = _load_arrays(args.data_dir, args.split)
    stats = load_stats_from_dir(args.data_dir)
    heights = np.asarray(stats["target_heights"], dtype=np.float32)
    var_names = ["temperature", "pressure", "humidity"][:test_y.shape[1]]
    units = {"temperature": "K", "pressure": "hPa", "humidity": "g/kg"}

    case_indices = _select_indices(args, len(test_x))
    print(f"选取样本: {case_indices}")
    x_batch = np.asarray(test_x[case_indices], dtype=np.float32)
    truths_std = np.asarray(test_y[case_indices], dtype=np.float32)
    truths = _restore_physical(truths_std, stats)

    print(f"加载模型: {args.model_path}")
    model = MLPBaseline1D(input_length=x_batch.shape[-1], out_channels=truths_std.shape[1]).to(DEVICE)
    state = torch.load(args.model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    x_tensor = torch.from_numpy(x_batch).to(DEVICE)

    all_preds = []
    for repeat_idx in range(args.n_repeats):
        seed = args.seed + repeat_idx
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        _enable_dropout_only(model)
        print(f"[repeat {repeat_idx + 1}/{args.n_repeats}] seed={seed}")
        with torch.no_grad():
            preds_std = model(x_tensor).cpu().numpy()
        preds_phys = _restore_physical(preds_std, stats)
        preds_phys = _maybe_smooth_batch(preds_phys, args.smooth)
        all_preds.append(preds_phys)

    all_preds = np.asarray(all_preds, dtype=np.float32)
    mean_preds = np.mean(all_preds, axis=0)
    std_preds = np.std(all_preds, axis=0)
    lower = np.percentile(all_preds, 2.5, axis=0)
    upper = np.percentile(all_preds, 97.5, axis=0)

    global_summary = {}
    for var_idx, var_name in enumerate(var_names):
        truth = truths[:, var_idx, :]
        lo = lower[:, var_idx, :]
        hi = upper[:, var_idx, :]
        mean_pred = mean_preds[:, var_idx, :]
        covered = ((truth >= lo) & (truth <= hi)).astype(np.float32)
        global_summary[var_name] = {
            "coverage_95": float(np.mean(covered)),
            "mean_interval_width": float(np.mean(hi - lo)),
            "mean_abs_error": float(np.mean(np.abs(mean_pred - truth))),
            "mean_predictive_std": float(np.mean(std_preds[:, var_idx, :])),
            "unit": units[var_name],
        }

    band_specs = [("0-5km", 0.0, 5.0), ("5-20km", 5.0, 20.0), ("20-60km", 20.0, 60.0)]
    height_band_summary = _compute_band_metrics(
        mean_preds=mean_preds,
        lower=lower,
        upper=upper,
        truths=truths,
        heights=heights,
        band_specs=band_specs,
        var_names=var_names,
    )

    fig, axes = plt.subplots(1, len(var_names), figsize=(5 * len(var_names), 6), sharey=True)
    if len(var_names) == 1:
        axes = [axes]
    for var_idx, var_name in enumerate(var_names):
        ax = axes[var_idx]
        ax.plot(np.mean(std_preds[:, var_idx, :], axis=0), heights, linewidth=2)
        ax.set_title(f"{var_name} predictive std")
        ax.set_xlabel(units[var_name])
        if var_idx == 0:
            ax.set_ylabel("Height (km)")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "predictive_std_profile.png"), dpi=150)
    plt.close()

    max_plot_cases = min(3, len(case_indices))
    for local_idx in range(max_plot_cases):
        fig, axes = plt.subplots(1, len(var_names), figsize=(5 * len(var_names), 6), sharey=True)
        if len(var_names) == 1:
            axes = [axes]
        for var_idx, var_name in enumerate(var_names):
            ax = axes[var_idx]
            ax.plot(truths[local_idx, var_idx], heights, "k-", linewidth=2, label="Truth")
            ax.plot(mean_preds[local_idx, var_idx], heights, "r--", linewidth=2, label="Mean pred")
            ax.fill_betweenx(
                heights,
                lower[local_idx, var_idx],
                upper[local_idx, var_idx],
                color="#8ecae6",
                alpha=0.35,
                label="95% interval",
            )
            ax.set_title(f"{var_name} (case #{case_indices[local_idx]})")
            ax.set_xlabel(units[var_name])
            if var_idx == 0:
                ax.set_ylabel("Height (km)")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"case_{case_indices[local_idx]}_interval.png"), dpi=150)
        plt.close()

    np.savez_compressed(
        os.path.join(save_dir, "interval_payload.npz"),
        mean_preds=mean_preds.astype(np.float32),
        lower=lower.astype(np.float32),
        upper=upper.astype(np.float32),
        truths=truths.astype(np.float32),
        heights=heights.astype(np.float32),
        case_indices=np.asarray(case_indices, dtype=np.int32),
    )

    summary = {
        "metadata": {
            "model_name": "mlp_mc_dropout",
            "metric_space": "physical",
            "model_path": os.path.relpath(args.model_path, PROJECT_ROOT)
            if os.path.isabs(args.model_path)
            else args.model_path,
            "data_dir": os.path.relpath(args.data_dir, PROJECT_ROOT)
            if os.path.isabs(args.data_dir)
            else args.data_dir,
            "n_cases": len(case_indices),
            "n_repeats": args.n_repeats,
            "case_indices": case_indices,
            "split": args.split,
            "device": str(DEVICE),
            "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
            "seed": args.seed,
            "smooth": args.smooth,
        },
        "global_summary": global_summary,
        "height_band_summary": height_band_summary,
    }

    summary_json_path = os.path.join(save_dir, "summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    summary_md_path = os.path.join(save_dir, "summary.md")
    _write_summary_md(summary_md_path, summary)

    print(f"\nMLP + MC Dropout 不确定性评估完成: {save_dir}")
    print(f"summary.json: {summary_json_path}")
    print(f"summary.md: {summary_md_path}")


if __name__ == "__main__":
    main()
