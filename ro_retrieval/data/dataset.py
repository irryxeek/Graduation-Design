"""
PyTorch 数据集类
================
RODataset:        单变量 (温度)，y shape = (301,)
ROMultiVarDataset: 多变量 (温度+气压+湿度)，y shape = (C, 301)
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class RODataset(Dataset):
    """单变量掩星数据集 (弯曲角 → 温度)"""

    def __init__(self, x_path: str, y_path: str):
        self.X = np.load(x_path).astype(np.float32)   # (N, 301)
        Y = np.load(y_path).astype(np.float32)
        # 如果 Y 是多变量 (N, C, 301)，只取第一个通道 (温度)
        if Y.ndim == 3:
            Y = Y[:, 0, :]
        self.Y = Y  # (N, 301)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])      # (301,) — condition
        y = torch.from_numpy(self.Y[idx])      # (301,) — target
        # 添加通道维度: (1, 301)
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        return x, y


class ROMultiVarDataset(Dataset):
    """多变量掩星数据集 (弯曲角 → 温度+气压+湿度)"""

    def __init__(self, x_path: str, y_path: str):
        self.X = np.load(x_path).astype(np.float32)   # (N, 301)
        self.Y = np.load(y_path).astype(np.float32)   # (N, C, 301) 或 (N, 301)

        # 确保 Y 有通道维度
        if self.Y.ndim == 2:
            self.Y = self.Y[:, np.newaxis, :]          # (N, 1, 301)

        self.num_vars = self.Y.shape[1]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])      # (301,) — condition
        y = torch.from_numpy(self.Y[idx])      # (C, 301) — target
        # condition 添加通道维度: (1, 301)
        x = x.unsqueeze(0)
        return x, y
