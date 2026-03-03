"""
改进的训练脚本 - 使用时间步加权损失
重点训练低时间步（t<200）以改善推理质量
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import sys
sys.path.append('/root/autodl-tmp/Graduation-Design')

from ro_retrieval.model.unet import EnhancedConditionalUNet1D
from ro_retrieval.model.diffusion import DiffusionSchedule
from ro_retrieval.data.dataset import ROMultiVarDataset


def get_timestep_weight(t, timesteps=1000):
    """
    计算时间步权重
    低时间步（t<200）获得更高权重
    """
    # 线性权重：t=0 权重为 5.0，t=999 权重为 1.0
    weight = 5.0 - 4.0 * (t / timesteps)
    return weight


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}\n")

    # 超参数
    epochs = 150
    batch_size = 64
    lr = 5e-5  # 降低学习率以更好地学习低时间步
    timesteps = 1000

    print("=== 训练配置 ===")
    print(f"轮数: {epochs}")
    print(f"批大小: {batch_size}")
    print(f"学习率: {lr}")
    print(f"时间步加权: 是 (t=0 权重 5.0, t=999 权重 1.0)\n")

    # 加载数据
    print("=== 加载数据 ===")
    train_dataset = ROMultiVarDataset(
        'Data/Processed_ATP/train_x.npy',
        'Data/Processed_ATP/train_y.npy'
    )
    val_dataset = ROMultiVarDataset(
        'Data/Processed_ATP/val_x.npy',
        'Data/Processed_ATP/val_y.npy'
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(val_dataset)} 样本\n")

    # 创建模型
    print("=== 创建模型 ===")
    model = EnhancedConditionalUNet1D(
        in_channels=2,
        cond_channels=1,
        out_channels=2,
        base_dim=64,
        time_dim=128,
        num_heads=4
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {n_params:,}\n")

    # 优化器和调度器
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    schedule = DiffusionSchedule(timesteps=timesteps, beta_start=1e-4, beta_end=0.02, device=device)

    # 训练
    best_val_loss = float('inf')
    patience = 30
    epochs_no_improve = 0

    print("=== 开始训练 ===\n")

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_weighted_loss = 0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for condition, x_0 in pbar:
            condition = condition.to(device)
            x_0 = x_0.to(device)
            b = x_0.shape[0]

            # 随机采样时间步
            t = torch.randint(0, timesteps, (b, 1), device=device).long()
            noise = torch.randn_like(x_0)
            x_t = schedule.q_sample(x_0, t, noise)

            # 模型预测
            noise_pred = model(x_t, t, condition)

            # 计算时间步权重
            t_weights = torch.tensor([get_timestep_weight(t_val.item(), timesteps)
                                     for t_val in t.squeeze()], device=device)
            t_weights = t_weights.view(-1, 1, 1)  # (B, 1, 1)

            # 加权损失
            loss_per_sample = torch.mean((noise_pred - noise) ** 2, dim=(1, 2))  # (B,)
            weighted_loss = torch.mean(t_weights.squeeze() * loss_per_sample)

            optimizer.zero_grad()
            weighted_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_weighted_loss += weighted_loss.item()
            n_batches += 1
            pbar.set_postfix({"loss": f"{weighted_loss.item():.6f}"})

        avg_train_loss = train_weighted_loss / n_batches

        # 验证阶段
        model.eval()
        val_loss = 0
        val_batches = 0

        with torch.no_grad():
            for condition, x_0 in val_loader:
                condition = condition.to(device)
                x_0 = x_0.to(device)
                b = x_0.shape[0]

                t = torch.randint(0, timesteps, (b, 1), device=device).long()
                noise = torch.randn_like(x_0)
                x_t = schedule.q_sample(x_0, t, noise)

                noise_pred = model(x_t, t, condition)
                loss = nn.functional.mse_loss(noise_pred, noise)

                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'enhanced_ro_diffusion_weighted_best.pth')
            print(f"  ✓ 保存最佳模型 (val_loss={best_val_loss:.6f})")
        else:
            epochs_no_improve += 1
            print(f"  验证损失未改善 ({epochs_no_improve}/{patience})")

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    print(f"\n=== 训练完成 ===")
    print(f"最佳验证损失: {best_val_loss:.6f}")
    print(f"模型已保存: enhanced_ro_diffusion_weighted_best.pth")


if __name__ == '__main__':
    main()
