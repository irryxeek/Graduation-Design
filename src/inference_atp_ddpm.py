"""ATP 数据推理 - 使用 DDPM 完整采样"""
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
sys.path.append('/root/autodl-tmp/Graduation-Design')

from ro_retrieval.model.unet import EnhancedConditionalUNet1D
from ro_retrieval.model.diffusion import DiffusionSchedule, ddpm_sample


def denormalize(data, mean, std):
    return data * std + mean


def calculate_metrics(pred, true):
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

    print(f"测试集: {len(test_x)} 样本")

    # 加载模型
    print("\n=== 加载模型 ===")
    model = EnhancedConditionalUNet1D(
        in_channels=2, cond_channels=1, out_channels=2,
        base_dim=64, time_dim=128, num_heads=4
    ).to(device)
    model.load_state_dict(torch.load('enhanced_ro_diffusion_best.pth', map_location=device))
    model.eval()
    print("模型加载成功")

    schedule = DiffusionSchedule(timesteps=1000, beta_start=1e-4, beta_end=0.02, device=device)

    # 推理
    print("\n=== DDPM 采样 (1000 步) ===")
    predictions = []
    batch_size = 16

    with torch.no_grad():
        for i in tqdm(range(0, len(test_x), batch_size), desc="推理"):
            batch_x = test_x[i:i+batch_size]
            condition = torch.from_numpy(batch_x).float().to(device).unsqueeze(1)
            pred = ddpm_sample(model, condition, (len(batch_x), 2, 301), schedule, device)
            predictions.append(pred.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)
    print(f"推理完成: {len(predictions)} 个预测")

    # 反标准化
    print("\n=== 反标准化 ===")
    pred_denorm = np.zeros_like(predictions)
    true_denorm = np.zeros_like(test_y)

    for i in range(2):
        pred_denorm[:, i, :] = denormalize(predictions[:, i, :], stats['y_mean'][i], stats['y_std'][i])
        true_denorm[:, i, :] = denormalize(test_y[:, i, :], stats['y_mean'][i], stats['y_std'][i])

    # 气压从 log10 转回
    if stats.get('pressure_log_transformed', False):
        pred_denorm[:, 1, :] = 10 ** pred_denorm[:, 1, :]
        true_denorm[:, 1, :] = 10 ** true_denorm[:, 1, :]

    # 评估
    print("\n=== 评估指标 ===")
    temp_metrics = calculate_metrics(pred_denorm[:, 0, :], true_denorm[:, 0, :])
    pres_metrics = calculate_metrics(pred_denorm[:, 1, :], true_denorm[:, 1, :])

    print(f"温度 - RMSE: {temp_metrics['rmse']:.2f} K, Bias: {temp_metrics['bias']:.2f} K, Corr: {temp_metrics['corr']:.4f}")
    print(f"气压 - RMSE: {pres_metrics['rmse']:.2f} mb, Bias: {pres_metrics['bias']:.2f} mb, Corr: {pres_metrics['corr']:.4f}")

    # 保存
    output_dir = Path('Results/ATP_DDPM')
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / 'predictions.npy', pred_denorm)
    np.save(output_dir / 'ground_truth.npy', true_denorm)
    np.save(output_dir / 'metrics.npy', {'temperature': temp_metrics, 'pressure': pres_metrics})

    print(f"\n结果保存至: {output_dir}")


if __name__ == '__main__':
    main()
