"""ATP 数据推理 - 调试 DDIM 采样（测试不同步数）"""
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
sys.path.append('/root/autodl-tmp/Graduation-Design')

from ro_retrieval.model.unet import EnhancedConditionalUNet1D
from ro_retrieval.model.diffusion import DiffusionSchedule, ddim_sample, ddpm_sample


def denormalize(data, mean, std):
    return data * std + mean


def calculate_metrics(pred, true):
    rmse = np.sqrt(np.mean((pred - true) ** 2))
    bias = np.mean(pred - true)
    corr = np.corrcoef(pred.flatten(), true.flatten())[0, 1]
    return {'rmse': rmse, 'bias': bias, 'corr': corr}


def run_inference(model, test_x, schedule, device, sampler='ddim', ddim_steps=50, eta=0.0):
    """运行推理"""
    predictions = []
    batch_size = 16

    with torch.no_grad():
        for i in tqdm(range(0, len(test_x), batch_size), desc=f"{sampler.upper()} {ddim_steps if sampler=='ddim' else 1000} 步"):
            batch_x = test_x[i:i+batch_size]
            condition = torch.from_numpy(batch_x).float().to(device).unsqueeze(1)

            if sampler == 'ddim':
                pred = ddim_sample(model, condition, (len(batch_x), 2, 301),
                                 schedule, ddim_steps=ddim_steps, eta=eta, device=device)
            else:
                pred = ddpm_sample(model, condition, (len(batch_x), 2, 301),
                                 schedule, device=device)

            predictions.append(pred.cpu().numpy())

    return np.concatenate(predictions, axis=0)


def evaluate_predictions(predictions, test_y, stats):
    """评估预测结果"""
    # 反标准化
    pred_denorm = np.zeros_like(predictions)
    true_denorm = np.zeros_like(test_y)

    for i in range(2):
        pred_denorm[:, i, :] = denormalize(predictions[:, i, :], stats['y_mean'][i], stats['y_std'][i])
        true_denorm[:, i, :] = denormalize(test_y[:, i, :], stats['y_mean'][i], stats['y_std'][i])

    # 气压从 log10 转回
    if stats.get('pressure_log_transformed', False):
        pred_denorm[:, 1, :] = 10 ** pred_denorm[:, 1, :]
        true_denorm[:, 1, :] = 10 ** true_denorm[:, 1, :]

    # 计算指标
    temp_metrics = calculate_metrics(pred_denorm[:, 0, :], true_denorm[:, 0, :])
    pres_metrics = calculate_metrics(pred_denorm[:, 1, :], true_denorm[:, 1, :])

    return temp_metrics, pres_metrics, pred_denorm, true_denorm


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载数据（使用少量样本快速测试）
    print("\n=== 加载测试数据 ===")
    test_x = np.load('Data/Processed_ATP/test_x.npy')
    test_y = np.load('Data/Processed_ATP/test_y.npy')
    stats = np.load('Data/Processed_ATP/stats.npy', allow_pickle=True).item()

    # 使用前 50 个样本快速测试
    test_x = test_x[:50]
    test_y = test_y[:50]
    print(f"测试集: {len(test_x)} 样本")

    if device.type == 'cpu':
        print("警告: 未检测到 GPU，推理速度会很慢")

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

    # 测试不同配置
    configs = [
        {'sampler': 'ddpm', 'ddim_steps': None, 'eta': None, 'name': 'DDPM-1000'},
        {'sampler': 'ddim', 'ddim_steps': 50, 'eta': 0.0, 'name': 'DDIM-50-eta0.0'},
        {'sampler': 'ddim', 'ddim_steps': 100, 'eta': 0.0, 'name': 'DDIM-100-eta0.0'},
        {'sampler': 'ddim', 'ddim_steps': 200, 'eta': 0.0, 'name': 'DDIM-200-eta0.0'},
        {'sampler': 'ddim', 'ddim_steps': 100, 'eta': 0.5, 'name': 'DDIM-100-eta0.5'},
        {'sampler': 'ddim', 'ddim_steps': 200, 'eta': 0.5, 'name': 'DDIM-200-eta0.5'},
    ]

    results = {}

    for config in configs:
        print(f"\n{'='*60}")
        print(f"测试配置: {config['name']}")
        print(f"{'='*60}")

        # 运行推理
        predictions = run_inference(
            model, test_x, schedule, device,
            sampler=config['sampler'],
            ddim_steps=config['ddim_steps'] if config['ddim_steps'] else 50,
            eta=config['eta'] if config['eta'] else 0.0
        )

        # 评估
        temp_metrics, pres_metrics, pred_denorm, true_denorm = evaluate_predictions(
            predictions, test_y, stats
        )

        print(f"\n温度 - RMSE: {temp_metrics['rmse']:.2f} K, Bias: {temp_metrics['bias']:.2f} K, Corr: {temp_metrics['corr']:.4f}")
        print(f"气压 - RMSE: {pres_metrics['rmse']:.2f} mb, Bias: {pres_metrics['bias']:.2f} mb, Corr: {pres_metrics['corr']:.4f}")

        results[config['name']] = {
            'temp': temp_metrics,
            'pres': pres_metrics,
            'predictions': pred_denorm,
            'config': config
        }

    # 保存结果
    output_dir = Path('Results/DDIM_Debug')
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存指标对比
    print(f"\n{'='*60}")
    print("结果汇总")
    print(f"{'='*60}")
    print(f"{'配置':<20} {'温度CC':>10} {'气压CC':>10} {'温度RMSE':>12} {'气压RMSE':>12}")
    print("-" * 60)

    for name, result in results.items():
        print(f"{name:<20} {result['temp']['corr']:>10.4f} {result['pres']['corr']:>10.4f} "
              f"{result['temp']['rmse']:>12.2f} {result['pres']['rmse']:>12.2f}")

    # 保存详细结果
    np.save(output_dir / 'results.npy', results)
    np.save(output_dir / 'ground_truth.npy', true_denorm)

    # 生成对比图
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for idx, (name, result) in enumerate(results.items()):
        if idx >= 6:
            break

        row = idx // 3
        col = idx % 3
        ax = axes[row, col]

        # 绘制温度散点图
        pred_temp = result['predictions'][:, 0, :].flatten()
        true_temp = true_denorm[:, 0, :].flatten()

        ax.scatter(true_temp, pred_temp, alpha=0.3, s=1)
        ax.plot([150, 350], [150, 350], 'r--', lw=2)
        ax.set_xlabel('True Temperature (K)')
        ax.set_ylabel('Predicted Temperature (K)')
        ax.set_title(f"{name}\nCorr: {result['temp']['corr']:.4f}")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n结果保存至: {output_dir}")


if __name__ == '__main__':
    main()
