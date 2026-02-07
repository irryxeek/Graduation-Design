"""
增强版训练脚本
==============
使用 ro_retrieval 包模块化训练
支持:
  - 单变量 (兼容旧模型) 和多变量 (温度+压力+湿度) 训练模式
  - 增强版 U-Net (交叉注意力机制)
  - 完整的训练日志
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# 将项目根目录加入 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ro_retrieval.config import (
    DEVICE, BATCH_SIZE, EPOCHS, LEARNING_RATE,
    TIMESTEPS, PROCESSED_DIR, PROJECT_ROOT,
)
from ro_retrieval.data.dataset import RODataset, ROMultiVarDataset
from ro_retrieval.model.unet import ConditionalUNet1D, EnhancedConditionalUNet1D
from ro_retrieval.model.diffusion import DiffusionSchedule


def parse_args():
    parser = argparse.ArgumentParser(description="训练掩星反演扩散模型")
    parser.add_argument("--mode", choices=["single", "multi"], default="single",
                        help="single=单变量(温度), multi=多变量(温度+压力+湿度)")
    parser.add_argument("--model", choices=["legacy", "enhanced"], default="legacy",
                        help="legacy=原始U-Net, enhanced=交叉注意力增强版")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--data_dir", type=str, default=PROCESSED_DIR)
    parser.add_argument("--save_dir", type=str, default=PROJECT_ROOT)
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. 加载数据
    x_path = os.path.join(args.data_dir, "train_x.npy")
    y_path = os.path.join(args.data_dir, "train_y.npy")

    if not os.path.exists(x_path):
        print(f"错误: 找不到数据文件 {x_path}")
        return

    if args.mode == "multi":
        dataset = ROMultiVarDataset(x_path, y_path)
        out_channels = dataset.num_vars
    else:
        dataset = RODataset(x_path, y_path)
        out_channels = 1

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # 2. 初始化模型
    if args.model == "enhanced":
        model = EnhancedConditionalUNet1D(
            in_channels=out_channels,
            cond_channels=1,
            out_channels=out_channels,
            use_cross_attention=True,
        ).to(DEVICE)
        model_prefix = "enhanced_ro_diffusion"
        print(f"使用增强版模型 (交叉注意力 + {out_channels} 通道输出)")
    else:
        model = ConditionalUNet1D(
            in_channels=out_channels,
            cond_channels=1,
            out_channels=out_channels,
        ).to(DEVICE)
        model_prefix = "ro_diffusion"
        print(f"使用原始模型 ({out_channels} 通道输出)")

    # 统计模型参数量
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {n_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    schedule = DiffusionSchedule(TIMESTEPS, device=DEVICE)

    # 3. 训练循环
    print(f"\n开始训练 on {DEVICE}...")
    print(f"  Epochs: {args.epochs}, BatchSize: {args.batch_size}, LR: {args.lr}")
    print(f"  模式: {args.mode}, 输出通道数: {out_channels}")

    loss_history = []

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        epoch_loss = 0

        for condition, x_0 in pbar:
            condition = condition.to(DEVICE)
            x_0 = x_0.to(DEVICE)
            batch_size_real = x_0.shape[0]

            # 随机时间步
            t = torch.randint(0, TIMESTEPS, (batch_size_real, 1), device=DEVICE).long()

            # 生成噪声
            noise = torch.randn_like(x_0)

            # 前向加噪
            x_t = schedule.q_sample(x_0, t, noise)

            # 预测噪声
            noise_pred = model(x_t, t, condition)

            # 计算损失
            loss = loss_fn(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.6f}"})

        mean_loss = epoch_loss / len(dataloader)
        loss_history.append(mean_loss)
        print(f"Epoch {epoch + 1} Mean Loss: {mean_loss:.6f}")

        # 每 10 轮保存
        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(
                args.save_dir, f"{model_prefix}_epoch_{epoch + 1}.pth"
            )
            torch.save(model.state_dict(), save_path)
            print(f"模型已保存: {save_path}")

    # 保存训练损失曲线
    np.save(os.path.join(args.save_dir, f"{model_prefix}_loss_history.npy"),
            np.array(loss_history))
    print("\n训练完成!")


if __name__ == "__main__":
    main()
