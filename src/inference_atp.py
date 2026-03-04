"""
ATP 数据推理脚本
使用训练好的扩散模型对测试集进行推理
"""
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
sys.path.append('/root/autodl-tmp/Graduation-Design')

from ro_retrieval.model.unet import EnhancedConditionalUNet1D
from ro_retrieval.model.diffusion import DiffusionSchedule, ddim_sample


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

    # 加载数据
    print("\n=== 加载测试数据 ===")
    test_x = np.load('Data/Processed_ATP/test_x.npy')
    test_y = np.load('Data/Processed_ATP/test_y.npy')
    stats = np.load('Data/Processed_ATP/stats.npy', allow_pickle=True).item()

    print(f"测试集大小: {len(test_x)} 个样本")
    print(f"输入形状: {test_x.shape}")  # (N, 301)
    print(f"标签形状: {test_y.shape}")  # (N, 2, 301)

    # 加载模型
    print("\n=== 加载模型 ===")
    model = EnhancedConditionalUNet1D(
        in_channels=2,  # 温度 + 气压
        cond_channels=1,  # 弯曲角
        out_channels=2,
        base_dim=64,
        time_dim=128,
        num_heads=4
    ).to(device)

    checkpoint = torch.load('enhanced_ro_diffusion_best.pth', map_location=device)
    model.load_state_dict(checkpoint)  # 直接加载 state_dict
    model.eval()
    print("模型加载成功")

    # 初始化扩散调度器
    schedule = DiffusionSchedule(timesteps=1000, beta_start=1e-4, beta_end=0.02, device=device)

    # 推理
    print("\n=== 开始推理 (DDIM 50 步) ===")
    predictions = []
    batch_size = 32

    with torch.no_grad():
        for i in tqdm(range(0, len(test_x), batch_size), desc="推理进度"):
            batch_x = test_x[i:i+batch_size]

            # 转换为 tensor
            condition = torch.from_numpy(batch_x).float().to(device)
            condition = condition.unsqueeze(1)  # (B, 1, 301)

            # DDIM 采样
            pred = ddim_sample(
                model=model,
                condition=condition,
                schedule=schedule,
                shape=(len(batch_x), 2, 301),
                ddim_steps=50,
                eta=0.0,
                device=device
            )

            predictions.append(pred.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)
    print(f"推理完成，生成 {len(predictions)} 个预测")

    # 反标准化
    print("\n=== 反标准化 ===")
    pred_denorm = np.zeros_like(predictions)
    true_denorm = np.zeros_like(test_y)

    for i in range(2):  # 温度和气压
        pred_denorm[:, i, :] = denormalize(predictions[:, i, :], stats['y_mean'][i], stats['y_std'][i])
        true_denorm[:, i, :] = denormalize(test_y[:, i, :], stats['y_mean'][i], stats['y_std'][i])

    # 气压从 log10 空间转回
    if stats.get('pressure_log_transformed', False):
        pred_denorm[:, 1, :] = 10 ** pred_denorm[:, 1, :]
        true_denorm[:, 1, :] = 10 ** true_denorm[:, 1, :]

    # 计算指标
    print("\n=== 评估指标 ===")
    temp_metrics = calculate_metrics(pred_denorm[:, 0, :], true_denorm[:, 0, :])
    pres_metrics = calculate_metrics(pred_denorm[:, 1, :], true_denorm[:, 1, :])

    print(f"温度 - RMSE: {temp_metrics['rmse']:.2f} K, Bias: {temp_metrics['bias']:.2f} K, Corr: {temp_metrics['corr']:.4f}")
    print(f"气压 - RMSE: {pres_metrics['rmse']:.2f} mb, Bias: {pres_metrics['bias']:.2f} mb, Corr: {pres_metrics['corr']:.4f}")

    # 保存结果
    print("\n=== 保存结果 ===")
    output_dir = Path('Results/ATP_Inference')
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / 'predictions.npy', pred_denorm)
    np.save(output_dir / 'ground_truth.npy', true_denorm)

    # 保存指标
    metrics = {
        'temperature': temp_metrics,
        'pressure': pres_metrics
    }
    np.save(output_dir / 'metrics.npy', metrics)

    # 可视化
    print("\n=== 生成可视化 ===")
    heights = np.linspace(0, 60, 301)

    # 选择 5 个样本进行可视化
    sample_indices = np.random.choice(len(predictions), 5, replace=False)

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))

    for idx, sample_idx in enumerate(sample_indices):
        # 温度
        axes[0, idx].plot(true_denorm[sample_idx, 0, :], heights, 'b-', label='True', linewidth=2)
        axes[0, idx].plot(pred_denorm[sample_idx, 0, :], heights, 'r--', label='Pred', linewidth=2)
        axes[0, idx].set_xlabel('Temperature (K)')
        axes[0, idx].set_ylabel('Height (km)')
        axes[0, idx].set_title(f'Sample {sample_idx}')
        axes[0, idx].legend()
        axes[0, idx].grid(True, alpha=0.3)

        # 气压
        axes[1, idx].plot(true_denorm[sample_idx, 1, :], heights, 'b-', label='True', linewidth=2)
        axes[1, idx].plot(pred_denorm[sample_idx, 1, :], heights, 'r--', label='Pred', linewidth=2)
        axes[1, idx].set_xlabel('Pressure (mb)')
        axes[1, idx].set_ylabel('Height (km)')
        axes[1, idx].set_xscale('log')
        axes[1, idx].legend()
        axes[1, idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'sample_predictions.png', dpi=150, bbox_inches='tight')
    print(f"可视化已保存到 {output_dir / 'sample_predictions.png'}")

    # 散点图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 温度散点图
    axes[0].scatter(true_denorm[:, 0, :].flatten(), pred_denorm[:, 0, :].flatten(),
                   alpha=0.1, s=1)
    axes[0].plot([150, 350], [150, 350], 'r--', linewidth=2)
    axes[0].set_xlabel('True Temperature (K)')
    axes[0].set_ylabel('Predicted Temperature (K)')
    axes[0].set_title(f'Temperature (Corr={temp_metrics["corr"]:.4f})')
    axes[0].grid(True, alpha=0.3)

    # 气压散点图
    axes[1].scatter(true_denorm[:, 1, :].flatten(), pred_denorm[:, 1, :].flatten(),
                   alpha=0.1, s=1)
    axes[1].plot([0.01, 1100], [0.01, 1100], 'r--', linewidth=2)
    axes[1].set_xlabel('True Pressure (mb)')
    axes[1].set_ylabel('Predicted Pressure (mb)')
    axes[1].set_title(f'Pressure (Corr={pres_metrics["corr"]:.4f})')
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'scatter_plots.png', dpi=150, bbox_inches='tight')
    print(f"散点图已保存到 {output_dir / 'scatter_plots.png'}")

    print("\n=== 推理完成 ===")
    print(f"结果保存在: {output_dir}")


if __name__ == '__main__':
    main()
