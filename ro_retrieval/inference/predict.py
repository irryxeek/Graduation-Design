"""
推理管线
========
提供统一的 DDPM / DDIM 推理接口
"""

import torch
import numpy as np
from scipy.signal import savgol_filter

from ro_retrieval.config import DEVICE, SAVGOL_WINDOW, SAVGOL_POLYORDER


def run_inference(model, condition_raw, stats, schedule, device=DEVICE,
                  out_channels=1, smooth=True):
    """
    DDPM 推理单条样本

    Parameters
    ----------
    model : nn.Module
    condition_raw : np.ndarray (301,) 未标准化的弯曲角
    stats : dict  包含 x_mean, x_std, y_mean, y_std
    schedule : DiffusionSchedule
    out_channels : int
    smooth : bool

    Returns
    -------
    np.ndarray  预测结果 shape=(301,) 或 (num_vars, 301)
    """
    from ro_retrieval.model.diffusion import ddpm_sample

    # 标准化输入
    x_norm = (condition_raw - stats["x_mean"]) / stats["x_std"]
    cond = torch.tensor(x_norm).float().unsqueeze(0).unsqueeze(0).to(device)  # (1,1,301)

    # 采样
    gen = ddpm_sample(model, cond, shape=(1, out_channels, 301),
                      schedule=schedule, device=device)

    # 反归一化
    pred = gen.squeeze(0).cpu().numpy()  # (out_ch, 301)
    y_mean = stats["y_mean"]
    y_std = stats["y_std"]

    if pred.ndim == 1:
        pred = pred * y_std + y_mean
    else:
        pred = pred * y_std + y_mean

    # 平滑
    if smooth:
        if pred.ndim == 1:
            pred = savgol_filter(pred, SAVGOL_WINDOW, SAVGOL_POLYORDER)
        else:
            for i in range(pred.shape[0]):
                pred[i] = savgol_filter(pred[i], SAVGOL_WINDOW, SAVGOL_POLYORDER)

    return pred


def run_inference_ddim(model, condition_raw, stats, schedule, device=DEVICE,
                       out_channels=1, ddim_steps=50, smooth=True):
    """
    DDIM 加速推理单条样本
    """
    from ro_retrieval.model.diffusion import ddim_sample

    x_norm = (condition_raw - stats["x_mean"]) / stats["x_std"]
    cond = torch.tensor(x_norm).float().unsqueeze(0).unsqueeze(0).to(device)

    gen = ddim_sample(model, cond, shape=(1, out_channels, 301),
                      schedule=schedule, ddim_steps=ddim_steps, device=device)

    pred = gen.squeeze(0).cpu().numpy()
    y_mean = stats["y_mean"]
    y_std = stats["y_std"]

    if pred.ndim == 1:
        pred = pred * y_std + y_mean
    else:
        pred = pred * y_std + y_mean

    if smooth:
        if pred.ndim == 1:
            pred = savgol_filter(pred, SAVGOL_WINDOW, SAVGOL_POLYORDER)
        else:
            for i in range(pred.shape[0]):
                pred[i] = savgol_filter(pred[i], SAVGOL_WINDOW, SAVGOL_POLYORDER)

    return pred
