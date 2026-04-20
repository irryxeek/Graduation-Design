"""轻量判别式基线模型。"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLPBaseline1D(nn.Module):
    """将整条弯曲角廓线展平后直接回归三变量剖面。"""

    def __init__(self, input_length: int = 301, out_channels: int = 3, hidden_dim: int = 512):
        super().__init__()
        output_dim = out_channels * input_length
        self.net = nn.Sequential(
            nn.Linear(input_length, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )
        self.out_channels = out_channels
        self.input_length = input_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.squeeze(1)
        out = self.net(x)
        return out.view(x.shape[0], self.out_channels, self.input_length)


class CNNBaseline1D(nn.Module):
    """轻量一维卷积回归网络。"""

    def __init__(self, out_channels: int = 3, base_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, base_dim, kernel_size=7, padding=3),
            nn.SiLU(),
            nn.Conv1d(base_dim, base_dim, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(base_dim, base_dim * 2, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(base_dim * 2, base_dim * 2, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Conv1d(base_dim * 2, base_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(base_dim, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        feat = self.encoder(x)
        return self.head(feat)


def build_baseline(name: str, input_length: int = 301, out_channels: int = 3) -> nn.Module:
    name = name.lower()
    if name == "mlp":
        return MLPBaseline1D(input_length=input_length, out_channels=out_channels)
    if name == "cnn":
        return CNNBaseline1D(out_channels=out_channels)
    raise ValueError(f"未知基线模型: {name}")


def load_baseline_checkpoint(
    name: str,
    checkpoint_path: str,
    input_length: int = 301,
    out_channels: int = 3,
    device: torch.device | None = None,
) -> nn.Module:
    """加载判别式基线权重并切换到 eval 模式。"""
    model = build_baseline(name, input_length=input_length, out_channels=out_channels)
    state_dict = torch.load(checkpoint_path, map_location=device or "cpu", weights_only=True)
    model.load_state_dict(state_dict)
    if device is not None:
        model = model.to(device)
    model.eval()
    return model
