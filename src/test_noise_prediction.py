"""
测试模型的噪声预测能力
直接验证模型是否能正确预测添加的噪声
"""
import torch
import numpy as np
import sys
sys.path.append('/root/autodl-tmp/Graduation-Design')

from ro_retrieval.model.unet import EnhancedConditionalUNet1D
from ro_retrieval.model.diffusion import DiffusionSchedule


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}\n")

    # 加载测试数据
    test_x = np.load('Data/Processed_ATP/test_x.npy')[:10]  # 只测试 10 个样本
    test_y = np.load('Data/Processed_ATP/test_y.npy')[:10]

    print(f"测试样本数: {len(test_x)}")
    print(f"输入形状: {test_x.shape}")
    print(f"标签形状: {test_y.shape}\n")

    # 加载模型
    print("=== 加载模型 ===")
    model = EnhancedConditionalUNet1D(
        in_channels=2,
        cond_channels=1,
        out_channels=2,
        base_dim=64,
        time_dim=128,
        num_heads=4
    ).to(device)

    checkpoint = torch.load('enhanced_ro_diffusion_best.pth', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint)
    model.eval()
    print("模型加载成功\n")

    # 初始化扩散调度器
    schedule = DiffusionSchedule(timesteps=1000, beta_start=1e-4, beta_end=0.02, device=device)

    # 测试不同时间步的噪声预测
    print("=== 测试噪声预测准确性 ===")
    timesteps_to_test = [0, 100, 500, 999]  # 测试不同的时间步

    with torch.no_grad():
        for t_val in timesteps_to_test:
            print(f"\n时间步 t={t_val}:")

            # 准备数据
            condition = torch.from_numpy(test_x).float().to(device)
            condition = condition.unsqueeze(1)  # (10, 1, 301)

            x_0 = torch.from_numpy(test_y).float().to(device)  # (10, 2, 301)

            # 添加噪声
            t = torch.full((len(test_x), 1), t_val, device=device).long()
            noise_true = torch.randn_like(x_0)
            x_t = schedule.q_sample(x_0, t, noise_true)

            # 模型预测噪声
            noise_pred = model(x_t, t, condition)

            # 计算预测误差
            mse = torch.mean((noise_pred - noise_true) ** 2).item()
            mae = torch.mean(torch.abs(noise_pred - noise_true)).item()

            # 统计信息
            noise_true_mean = noise_true.mean().item()
            noise_true_std = noise_true.std().item()
            noise_pred_mean = noise_pred.mean().item()
            noise_pred_std = noise_pred.std().item()

            print(f"  真实噪声: mean={noise_true_mean:.4f}, std={noise_true_std:.4f}")
            print(f"  预测噪声: mean={noise_pred_mean:.4f}, std={noise_pred_std:.4f}")
            print(f"  MSE: {mse:.6f}")
            print(f"  MAE: {mae:.6f}")

            # 计算相关系数
            noise_true_flat = noise_true.cpu().numpy().flatten()
            noise_pred_flat = noise_pred.cpu().numpy().flatten()
            corr = np.corrcoef(noise_true_flat, noise_pred_flat)[0, 1]
            print(f"  相关系数: {corr:.4f}")

    print("\n=== 测试完成 ===")
    print("\n解释:")
    print("- MSE 应该接近验证损失 (0.015)")
    print("- 相关系数应该接近 1.0")
    print("- 如果这些指标很差，说明模型没有学好")
    print("- 如果这些指标很好，说明问题出在采样过程")


if __name__ == '__main__':
    main()
