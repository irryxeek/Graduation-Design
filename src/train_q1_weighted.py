"""训练模型 - 使用损失函数加权改进温度预测"""
import sys
sys.path.append('/root/autodl-tmp/Graduation-Design')

import torch
import torch.nn as nn
from pathlib import Path
import numpy as np

from ro_retrieval.model.unet import EnhancedConditionalUNet1D
from ro_retrieval.model.diffusion import DiffusionSchedule
from ro_retrieval.training.trainer import Trainer
from ro_retrieval.config import DEVICE

def main():
    print("="*60)
    print("训练模型 - Q1 数据 + 损失加权")
    print("="*60)

    # 加载数据
    data_dir = Path('Data/Processed_ATP_Q1')
    print(f"\n加载数据: {data_dir}")

    train_x = np.load(data_dir / 'train_x.npy')
    train_y = np.load(data_dir / 'train_y.npy')
    val_x = np.load(data_dir / 'val_x.npy')
    val_y = np.load(data_dir / 'val_y.npy')

    print(f"训练集: {len(train_x)} 样本")
    print(f"验证集: {len(val_x)} 样本")

    # 创建模型
    print("\n创建模型...")
    model = EnhancedConditionalUNet1D(
        in_channels=2,      # 温度 + 气压
        cond_channels=1,    # 弯曲角
        out_channels=2,     # 温度 + 气压
        base_dim=64,
        time_dim=128,
        num_heads=4
    ).to(DEVICE)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    # 扩散调度
    schedule = DiffusionSchedule(
        timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        device=DEVICE
    )

    # 训练器
    print("\n初始化训练器...")
    print(f"  温度损失权重: 2.0")
    print(f"  气压损失权重: 1.0")

    trainer = Trainer(
        data_dir='Data/Processed_ATP_Q1',
        model_type='enhanced',
        mode='multi',
        epochs=100,
        batch_size=64,
        lr=1e-4,
        patience=20,
        device=DEVICE,
        var_weights=[2.0, 1.0]  # [温度权重, 气压权重]
    )

    # 训练
    print("\n开始训练...")
    print(f"  Epochs: 100")
    print(f"  Batch Size: 64")
    print(f"  Learning Rate: 1e-4")
    print(f"  Early Stopping Patience: 20")
    print(f"  设备: {DEVICE}")

    trainer.train()

    print("\n训练完成！")
    print(f"最佳模型已保存")


if __name__ == '__main__':
    main()
