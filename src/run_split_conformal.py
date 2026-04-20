"""对区间预测结果运行 split conformal 后校准。"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="运行 split conformal 区间后校准")
    parser.add_argument("--calib_npz", type=str, required=True)
    parser.add_argument("--test_npz", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser.parse_args()


def _load_payload(path):
    data = np.load(path)
    return {
        "mean_preds": data["mean_preds"].astype(np.float32),
        "lower": data["lower"].astype(np.float32),
        "upper": data["upper"].astype(np.float32),
        "truths": data["truths"].astype(np.float32),
        "heights": data["heights"].astype(np.float32),
        "case_indices": data["case_indices"].astype(np.int32),
    }


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    scores = np.sort(scores.reshape(-1))
    n = len(scores)
    rank = int(np.ceil((n + 1) * (1 - alpha))) - 1
    rank = min(max(rank, 0), n - 1)
    return float(scores[rank])


def _compute_scalar_scores(lower, upper, truths):
    return np.maximum.reduce([lower - truths, truths - upper, np.zeros_like(truths)])


def _evaluate(lower, upper, truths, mean_preds, heights):
    var_names = ["temperature", "pressure", "humidity"][:truths.shape[1]]
    units = {"temperature": "K", "pressure": "hPa", "humidity": "g/kg"}
    band_specs = [("0-5km", 0.0, 5.0), ("5-20km", 5.0, 20.0), ("20-60km", 20.0, 60.0)]

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
            "unit": units[var_name],
        }

    height_band_summary = {}
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
        height_band_summary[label] = band

    return global_summary, height_band_summary


def _write_summary_md(path, summary):
    lines = [
        "# Split Conformal 校准摘要",
        "",
        f"- alpha: `{summary['metadata']['alpha']}`",
        f"- 校准集: `{summary['metadata']['calib_npz']}`",
        f"- 测试集: `{summary['metadata']['test_npz']}`",
        "",
        "## 全局缩放量",
        "",
        "| 变量 | q_hat | 单位 |",
        "| --- | ---: | --- |",
    ]
    for var_name, item in summary["q_hat"].items():
        lines.append(f"| {var_name} | {item['q_hat']:.4f} | {item['unit']} |")
    lines.extend(["", "## 校准后整体结果", "", "| 变量 | 覆盖率 | 平均区间宽度 | 均值预测MAE | 单位 |", "| --- | ---: | ---: | ---: | --- |"])
    for var_name, item in summary["global_summary"].items():
        lines.append(
            f"| {var_name} | {item['coverage_95']:.4f} | {item['mean_interval_width']:.4f} | {item['mean_abs_error']:.4f} | {item['unit']} |"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    calib = _load_payload(args.calib_npz)
    test = _load_payload(args.test_npz)
    var_names = ["temperature", "pressure", "humidity"][:calib["truths"].shape[1]]
    units = {"temperature": "K", "pressure": "hPa", "humidity": "g/kg"}

    q_hat = {}
    calibrated_lower = np.array(test["lower"], copy=True)
    calibrated_upper = np.array(test["upper"], copy=True)
    for var_idx, var_name in enumerate(var_names):
        scores = _compute_scalar_scores(
            calib["lower"][:, var_idx, :],
            calib["upper"][:, var_idx, :],
            calib["truths"][:, var_idx, :],
        )
        q = _conformal_quantile(scores, alpha=args.alpha)
        q_hat[var_name] = {"q_hat": q, "unit": units[var_name]}
        calibrated_lower[:, var_idx, :] -= q
        calibrated_upper[:, var_idx, :] += q

    global_summary, height_band_summary = _evaluate(
        lower=calibrated_lower,
        upper=calibrated_upper,
        truths=test["truths"],
        mean_preds=test["mean_preds"],
        heights=test["heights"],
    )

    summary = {
        "metadata": {
            "alpha": args.alpha,
            "calib_npz": args.calib_npz,
            "test_npz": args.test_npz,
            "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "q_hat": q_hat,
        "global_summary": global_summary,
        "height_band_summary": height_band_summary,
    }

    summary_json = os.path.join(args.save_dir, "summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    _write_summary_md(os.path.join(args.save_dir, "summary.md"), summary)
    print(f"split conformal summary saved: {summary_json}")


if __name__ == "__main__":
    main()
