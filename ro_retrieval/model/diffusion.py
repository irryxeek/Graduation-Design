"""
扩散过程
========
包含:
  - Beta Schedule 与预计算系数
  - DDPM 采样
  - DDIM 加速采样
"""

import torch
import torch.nn as nn
import numpy as np

from ro_retrieval.config import TIMESTEPS, BETA_START, BETA_END, DDIM_STEPS, DDIM_ETA


class DiffusionSchedule:
    """扩散过程的 beta schedule 与预计算系数"""

    def __init__(self, timesteps=TIMESTEPS, beta_start=BETA_START, beta_end=BETA_END,
                 device=None):
        self.timesteps = timesteps
        self.device = device or torch.device("cpu")

        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0]).to(self.device), self.alphas_cumprod[:-1]]
        )

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def q_sample(self, x_0, t, noise=None):
        """
        前向加噪: x_t = sqrt(alpha_bar) * x_0 + sqrt(1-alpha_bar) * noise
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)

        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise

    def to(self, device):
        self.device = device
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        return self


# =====================================================================
# DDPM 采样
# =====================================================================
@torch.no_grad()
def _p_sample(model, x, t, t_index, condition, schedule):
    """单步反向去噪 (DDPM)"""
    betas_t = schedule.betas[t_index]
    sqrt_one_minus = torch.sqrt(1.0 - schedule.alphas_cumprod[t_index])
    sqrt_recip = torch.sqrt(1.0 / schedule.alphas[t_index])

    noise_pred = model(x, t, condition)
    model_mean = sqrt_recip * (x - betas_t * noise_pred / sqrt_one_minus)

    if t_index == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(schedule.posterior_variance[t_index]) * noise


@torch.no_grad()
def ddpm_sample(model, condition, shape, schedule, device=None):
    """
    DDPM 完整采样

    Parameters
    ----------
    model : nn.Module
    condition : Tensor (B, cond_ch, L)
    shape : tuple (B, out_ch, L)
    schedule : DiffusionSchedule
    """
    device = device or condition.device
    b = shape[0]
    img = torch.randn(shape, device=device)

    for i in reversed(range(schedule.timesteps)):
        t = torch.full((b, 1), i, device=device, dtype=torch.long)
        img = _p_sample(model, img, t, i, condition, schedule)

    return img


# =====================================================================
# DDIM 加速采样
# =====================================================================
@torch.no_grad()
def ddim_sample(model, condition, shape, schedule,
                ddim_steps=DDIM_STEPS, eta=DDIM_ETA, device=None):
    """
    DDIM 加速采样

    Parameters
    ----------
    ddim_steps : int
        采样步数 (默认50, 比 DDPM 的 1000 步快 20 倍)
    eta : float
        随机性控制: 0=纯确定性, 1=等价 DDPM
    """
    device = device or condition.device
    b = shape[0]
    total_timesteps = schedule.timesteps

    # 生成子序列时间步
    times = torch.linspace(0, total_timesteps - 1, steps=ddim_steps + 1).long().to(device)
    time_pairs = list(zip(reversed(times[:-1]), reversed(times[1:])))

    img = torch.randn(shape, device=device)

    for t_curr, t_prev in time_pairs:
        t_batch = torch.full((b, 1), t_curr.item(), device=device).long()

        alpha_t = schedule.alphas_cumprod[t_curr]
        alpha_t_prev = schedule.alphas_cumprod[t_prev] if t_prev >= 0 \
            else torch.tensor(1.0).to(device)

        noise_pred = model(img, t_batch, condition)

        # 预测 x0
        pred_x0 = (img - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)

        # 方向项
        sigma_t = eta * torch.sqrt(
            (1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev)
        ) if eta > 0 else 0

        dir_xt = torch.sqrt(1 - alpha_t_prev - sigma_t ** 2) * noise_pred
        noise = sigma_t * torch.randn_like(img) if eta > 0 else 0

        img = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt + noise

    return img
