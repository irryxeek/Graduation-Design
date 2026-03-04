"""
测试 DDPM 完整采样（1000 步）
"""
import torch
import numpy as np
from pathlib import Path
import sys
sys.path.append('/root/autodl-tmp/Graduation-Design')

from ro_retrieval.model.unet import EnhancedConditionalUNet1D
from ro_retrieval.model.diffusion import DiffusionSchedule, ddpm_sample


def denormalize(data, mean, std):
    """反标准化"""
    return data * std + mean


def calculate_metrics(pred, true):
    """计算评估指标"""
    rmse = np.sqrt(np.mean((pred - true) ** 2))
    bias = np.mean(pred - true)
    corr = np.corrcoef(pred.flatten(), true.flatten())[0, 1]
    return {'rmse': rmse, 'bias': bias, 'corr': corr}


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载数据（只测试前 100 个样本以节省时间）
    print("\n=== 加载测试数据 ===")
    test_x = np.load('Data/Processed_ATP/test_x.npy')[:100]
    test_y = np.load('Data/Processed_ATP/test_y.npy')[:100]
    stats = np.load('Data/Processed_ATP/stats.npy', allow_pickle=True).item()

    print(f"测试样本数: {len(test_x)}")

    # 加载模型
    print("\n=== 加载模型 ===")
    model = EnhancedConditionalUNet1D(
        in_channels=2,
        cond_channels=1,
        out_channels=2,
        base_dim=64,
        time_dim=128,
        num_heads=4
    ).to(device)

    checkpoint = torch.load('enhanced_ro_diffusion_best.pth', map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    print("模型加载成功")

    # 初始化扩散调度器
    schedule = DiffusionSchedule(timesteps=1000, beta_start=1e-4, beta_end=0.02, device=device)

    # DDPM 采样（1000 步）
    print("\n=== DDPM 采样 (1000 步) ===")
    predictions = []

    with torch.no_grad():
        for i in range(len(test_x)):
            if i % 10 == 0:
                print(f"处理样本 {i}/{len(test_x)}")

            condition = torch.from_numpy(test_x[i:i+1]).float().to(device)
            condition = condition.unsqueeze(1)  # (1, 1, 301)

            # DDPM 采样
            pred = ddpm_sample(
                model=model,
                condition=condition,
                shape=(1, 2, 301),
                schedule=schedule,
                device=device
            )

            predictions.append(pred.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)
    print(f"采样完成，生成 {len(predictions)} 个预测")

    # 反标准化
    print("\n=== 反标准化 ===")
    pred_denorm = np.zeros_like(predictions)
    true_denorm = np.zeros_like(test_y)

    for i in range(2):
        pred_denorm[:, i, :] = denormalize(predictions[:, i, :], stats['y_mean'][i], stats['y_std'][i])
        true_denorm[:, i, :] = denormalize(test_y[:, i, :], stats['y_mean'][i], stats['y_std'][i])

    # 计算指标
    print("\n=== 评估指标 (DDPM 1000 步) ===")
    temp_metrics = calculate_metrics(pred_denorm[:, 0, :], true_denorm[:, 0, :])
    pres_metrics = calculate_metrics(pred_denorm[:, 1, :], true_denorm[:, 1, :])

    print(f"温度 - RMSE: {temp_metrics['rmse']:.2f} K, Bias: {temp_metrics['bias']:.2f} K, Corr: {temp_metrics['corr']:.4f}")
    print(f"气压 - RMSE: {pres_metrics['rmse']:.2f} mb, Bias: {pres_metrics['bias']:.2f} mb, Corr: {pres_metrics['corr']:.4f}")

    # 检查标准化空间的统计
    print("\n=== 标准化空间统计 ===")
    print(f"预测 - 温度: mean={predictions[:, 0, :].mean():.4f}, std={predictions[:, 0, :].std():.4f}")
    print(f"预测 - 气压: mean={predictions[:, 1, :].mean():.4f}, std={predictions[:, 1, :].std():.4f}")
    print(f"真值 - 温度: mean={test_y[:, 0, :].mean():.4f}, std={test_y[:, 0, :].std():.4f}")
    print(f"真值 - 气压: mean={test_y[:, 1, :].mean():.4f}, std={test_y[:, 1, :].std():.4f}")


if __name__ == '__main__':
    main()
