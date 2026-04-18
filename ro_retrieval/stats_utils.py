"""统计量加载与兼容辅助函数。"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np


def canonicalize_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    """统一不同流程产出的统计量键名。"""
    result = dict(stats)

    if "x_mean" in result:
        result["x_mean"] = np.asarray(result["x_mean"], dtype=np.float32)
    if "x_std" in result:
        result["x_std"] = np.asarray(result["x_std"], dtype=np.float32)

    if "y_mean" not in result and "y_means" in result:
        result["y_mean"] = result["y_means"]
    if "y_std" not in result and "y_stds" in result:
        result["y_std"] = result["y_stds"]

    if "y_mean" in result:
        result["y_mean"] = np.asarray(result["y_mean"], dtype=np.float32)
    if "y_std" in result:
        result["y_std"] = np.asarray(result["y_std"], dtype=np.float32)

    if "target_heights" in result:
        result["target_heights"] = np.asarray(result["target_heights"], dtype=np.float32)

    if "stats_space" not in result:
        result["stats_space"] = "physical"
    return result


def infer_normalized_only_stats(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """为仅有标准化数组、缺少原始 stats 文件的目录构造保守统计量。"""
    y_arr = np.asarray(y, dtype=np.float32)
    if y_arr.ndim == 2:
        out_channels = 1
    else:
        out_channels = y_arr.shape[1]

    return {
        "x_mean": np.float32(0.0),
        "x_std": np.float32(1.0),
        "y_mean": np.zeros(out_channels, dtype=np.float32),
        "y_std": np.ones(out_channels, dtype=np.float32),
        "target_heights": np.linspace(0, 60, x.shape[-1], dtype=np.float32),
        "stats_space": "normalized",
    }


def compute_fallback_stats(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """缺少 stats 文件时，根据数组状态做保守回退。"""
    x_arr = np.asarray(x, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.float32)

    x_mean = float(np.mean(x_arr))
    x_std = float(np.std(x_arr))
    y_mean = float(np.mean(y_arr))
    y_std = float(np.std(y_arr))

    normalized_like = (
        abs(x_mean) < 0.1
        and abs(x_std - 1.0) < 0.2
        and (
            (abs(y_mean) < 0.1 and abs(y_std - 1.0) < 0.2)
            or y_std < 1e-5
        )
    )
    if normalized_like:
        return infer_normalized_only_stats(x_arr, y_arr)

    if y_arr.ndim == 3:
        out_y_mean = np.mean(y_arr, axis=(0, 2)).astype(np.float32)
        out_y_std = (np.std(y_arr, axis=(0, 2)) + 1e-6).astype(np.float32)
    else:
        out_y_mean = np.asarray(np.mean(y_arr, axis=0), dtype=np.float32)
        out_y_std = np.asarray(np.std(y_arr, axis=0) + 1e-6, dtype=np.float32)

    return {
        "x_mean": np.asarray(np.mean(x_arr), dtype=np.float32),
        "x_std": np.asarray(np.std(x_arr) + 1e-6, dtype=np.float32),
        "y_mean": out_y_mean,
        "y_std": out_y_std,
        "target_heights": np.linspace(0, 60, x_arr.shape[-1], dtype=np.float32),
        "stats_space": "physical",
    }


def load_stats_from_dir(data_dir: str, x_fallback: Optional[np.ndarray] = None, y_fallback: Optional[np.ndarray] = None) -> Optional[Dict[str, Any]]:
    """从数据目录加载统计量；若缺失则按回退规则生成。"""
    stats_path = os.path.join(data_dir, "stats.npy")
    if os.path.exists(stats_path):
        stats = np.load(stats_path, allow_pickle=True)
        if isinstance(stats, np.ndarray) and stats.shape == ():
            stats = stats.item()
        return canonicalize_stats(stats)

    if x_fallback is None or y_fallback is None:
        return None
    return compute_fallback_stats(x_fallback, y_fallback)
