"""
评估指标模块
============
包含: RMSE / Bias / 相关系数 (CC) / 综合评估报告
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import json
import os


def compute_rmse(pred, truth):
    """
    计算 RMSE (Root Mean Square Error)

    Parameters
    ----------
    pred, truth : np.ndarray, shape=(L,) or (N, L)

    Returns
    -------
    float or np.ndarray
    """
    diff = pred - truth
    if diff.ndim == 1:
        return float(np.sqrt(np.mean(diff ** 2)))
    else:
        return np.sqrt(np.mean(diff ** 2, axis=-1))


def compute_bias(pred, truth):
    """
    计算 Bias (Mean Error)
    Bias > 0: 模型偏高; Bias < 0: 模型偏低

    Parameters
    ----------
    pred, truth : np.ndarray

    Returns
    -------
    float or np.ndarray
    """
    diff = pred - truth
    if diff.ndim == 1:
        return float(np.mean(diff))
    else:
        return np.mean(diff, axis=-1)


def compute_correlation(pred, truth):
    """
    计算 Pearson 相关系数 (CC)

    Parameters
    ----------
    pred, truth : np.ndarray, shape=(L,)

    Returns
    -------
    float
        相关系数 [-1, 1]
    """
    if pred.ndim > 1:
        # 逐样本计算
        ccs = []
        for p, t in zip(pred, truth):
            ccs.append(_single_cc(p, t))
        return np.array(ccs)

    return _single_cc(pred, truth)


def _single_cc(pred, truth):
    """单条廓线的相关系数"""
    p_mean = np.mean(pred)
    t_mean = np.mean(truth)
    p_std = np.std(pred) + 1e-10
    t_std = np.std(truth) + 1e-10
    cc = np.mean((pred - p_mean) * (truth - t_mean)) / (p_std * t_std)
    return float(np.clip(cc, -1, 1))


def compute_rmse_profile(pred, truth, heights):
    """
    逐高度层 RMSE (用于画 RMSE profile 图)

    Parameters
    ----------
    pred : np.ndarray (N, L)
    truth : np.ndarray (N, L)
    heights : np.ndarray (L,)

    Returns
    -------
    np.ndarray (L,)  每个高度层的 RMSE
    """
    diff = pred - truth   # (N, L)
    return np.sqrt(np.mean(diff ** 2, axis=0))


def compute_bias_profile(pred, truth, heights):
    """逐高度层 Bias"""
    diff = pred - truth
    return np.mean(diff, axis=0)


def evaluate_profile(pred, truth, variable_name="temperature"):
    """
    综合评估单条或批量廓线

    Parameters
    ----------
    pred : np.ndarray
    truth : np.ndarray
    variable_name : str

    Returns
    -------
    dict
    """
    rmse = compute_rmse(pred, truth)
    bias = compute_bias(pred, truth)
    cc = compute_correlation(pred, truth)

    mae = float(np.mean(np.abs(pred - truth)))

    return {
        "variable": variable_name,
        "rmse": rmse,
        "bias": bias,
        "cc": cc,
        "mae": mae,
    }


@dataclass
class EvaluationReport:
    """
    综合评估报告
    ============
    收集多条廓线的评估结果, 生成统计摘要
    """
    variable_names: List[str] = field(default_factory=lambda: ["temperature", "pressure", "humidity"])
    results: List[Dict] = field(default_factory=list)

    def add_sample(self, pred, truth, sample_idx=None, input_ba=None):
        """
        添加一个样本的评估结果

        Parameters
        ----------
        pred : np.ndarray
            预测值, shape=(L,) 单变量 或 (num_vars, L) 多变量
        truth : np.ndarray
            真值, 同 pred 形状
        sample_idx : int, optional
        input_ba : np.ndarray, optional
        """
        if pred.ndim == 1:
            pred = pred[np.newaxis, :]
            truth = truth[np.newaxis, :]

        entry = {"idx": sample_idx, "per_var": {}}
        if input_ba is not None:
            entry["input"] = input_ba

        for i, var_name in enumerate(self.variable_names):
            if i >= pred.shape[0]:
                break
            metrics = evaluate_profile(pred[i], truth[i], var_name)
            entry["per_var"][var_name] = metrics

        self.results.append(entry)

    def summary(self):
        """
        生成统计摘要

        Returns
        -------
        dict  包含每个变量的 avg/min/max RMSE, Bias, CC
        """
        summary = {}
        for var_name in self.variable_names:
            rmses, biases, ccs = [], [], []
            for r in self.results:
                if var_name in r["per_var"]:
                    m = r["per_var"][var_name]
                    rmses.append(m["rmse"])
                    biases.append(m["bias"])
                    ccs.append(m["cc"])

            if not rmses:
                continue

            summary[var_name] = {
                "count": len(rmses),
                "rmse_mean": float(np.mean(rmses)),
                "rmse_std": float(np.std(rmses)),
                "rmse_min": float(np.min(rmses)),
                "rmse_max": float(np.max(rmses)),
                "bias_mean": float(np.mean(biases)),
                "bias_std": float(np.std(biases)),
                "cc_mean": float(np.mean(ccs)),
                "cc_min": float(np.min(ccs)),
                "cc_max": float(np.max(ccs)),
            }

        return summary

    def print_report(self):
        """打印格式化评估报告"""
        s = self.summary()
        print("\n" + "=" * 60)
        print("           综合评估报告 (Evaluation Report)")
        print("=" * 60)

        for var_name, m in s.items():
            unit = "K" if var_name == "temperature" else (
                "hPa" if var_name == "pressure" else "kg/kg"
            )
            print(f"\n  [{var_name.upper()}]  (N={m['count']})")
            print(f"    RMSE  : {m['rmse_mean']:.4f} ± {m['rmse_std']:.4f} {unit}")
            print(f"            (best={m['rmse_min']:.4f}, worst={m['rmse_max']:.4f})")
            print(f"    Bias  : {m['bias_mean']:.4f} ± {m['bias_std']:.4f} {unit}")
            print(f"    CC    : {m['cc_mean']:.4f} (min={m['cc_min']:.4f}, max={m['cc_max']:.4f})")

        print("\n" + "=" * 60)

    def save_json(self, filepath):
        """保存评估结果到 JSON"""
        output = {
            "summary": self.summary(),
            "n_samples": len(self.results),
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"评估报告已保存: {filepath}")

    def get_sorted_results(self, variable="temperature", metric="rmse"):
        """按指定指标排序, 返回 (best, median, worst)"""
        filtered = [
            r for r in self.results
            if variable in r["per_var"]
        ]
        filtered.sort(key=lambda x: x["per_var"][variable][metric])

        if len(filtered) == 0:
            return None, None, None

        best = filtered[0]
        median = filtered[len(filtered) // 2]
        worst = filtered[-1]
        return best, median, worst
