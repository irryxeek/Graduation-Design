"""Q1 模型推理 - 对比 DDPM 和 DDIM 采样"""
import sys
sys.path.append('/root/autodl-tmp/Graduation-Design')

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

from ro_retrieval.model.unet import EnhancedConditionalUNet1D
from ro_retrieval.model.diffusion import DiffusionSchedule, ddpm_sample, ddim_sample


def denormalize(data, mean, std):
    return data * std + mean


def calculate_metrics(pred, true):
    rmse = np.sqrt(np.mean((pred - true) ** 2))
    bias = np.mean(pred - true)
    corr = np.corrcoef(pred.flatten(), true.flatten())[0, 1]
    return {'rmse': rmse, 'bias': bias, 'corr': corr}


def run_inference(model, test_x, schedule, device, sampler='ddpm', ddim_steps=50, batch_size=16):
    """运行推理"""
    predictions = []

    with torch.no_grad():
        for i in tqdm(range(0, len(test_x), batch_size), desc=f"{sampler.upper()} 推理"):
            batch_x = test_x[i:i+batch_size]
            condition = torch.from_numpy(batch_x).float().to(device).unsqueeze(1)

            if sampler == 'ddpm':
                pred = ddpm_sample(model, condition, (len(batch_x), 2, 301), schedule, device)
            else:  # ddim
                pred = ddim_sample(model, condition, (len(batch_x), 2, 301), schedule, ddim_steps=ddim_steps, device=device)

            predictions.append(pred.cpu().numpy())

    return np.concatenate(predictions, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sampler', choices=['ddpm', 'ddim', 'both'], default='both')
    parser.add_argument('--ddim-steps', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--model-path', default='enhanced_ro_diffusion_best.pth')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载数据
    print("\n=== 加载测试数据 ===")
    data_dir = Path('Data/Processed_ATP_Q1')
    test_x = np.load(data_dir / 'test_x.npy')
    test_y = np.load(data_dir / 'test_y.npy')
    stats = np.load(data_dir / 'stats.npy', allow_pickle=True).item()
    print(f"测试集: {len(test_x)} 样本")

    # 加载模型
    print("\n=== 加载模型 ===")
    model = EnhancedConditionalUNet1D(
        in_channels=2, cond_channels=1, out_channels=2,
        base_dim=64, time_dim=128, num_heads=4
    ).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print(f"模型加载成功: {args.model_path}")

    schedule = DiffusionSchedule(timesteps=1000, beta_start=1e-4, beta_end=0.02, device=device)

    # 运行推理
    samplers = ['ddpm', 'ddim'] if args.sampler == 'both' else [args.sampler]
    results = {}

    for sampler in samplers:
        print(f"\n{'='*60}")
        print(f"=== {sampler.upper()} 采样 ===")
        print(f"{'='*60}")

        if sampler == 'ddpm':
            print("采样步数: 1000")
        else:
            print(f"采样步数: {args.ddim_steps}")

        # 推理
        predictions = run_inference(
            model, test_x, schedule, device,
            sampler=sampler, ddim_steps=args.ddim_steps,
            batch_size=args.batch_size
        )

        # 反标准化
        pred_denorm = np.zeros_like(predictions)
        true_denorm = np.zeros_like(test_y)

        for i in range(2):
            pred_denorm[:, i, :] = denormalize(predictions[:, i, :], stats['y_mean'][i], stats['y_std'][i])
            true_denorm[:, i, :] = denormalize(test_y[:, i, :], stats['y_mean'][i], stats['y_std'][i])

        # 气压从 log10 转回
        if stats.get('pressure_log_transformed', True):
            pred_denorm[:, 1, :] = 10 ** pred_denorm[:, 1, :]
            true_denorm[:, 1, :] = 10 ** true_denorm[:, 1, :]

        # 评估
        temp_metrics = calculate_metrics(pred_denorm[:, 0, :], true_denorm[:, 0, :])
        pres_metrics = calculate_metrics(pred_denorm[:, 1, :], true_denorm[:, 1, :])

        print(f"\n温度:")
        print(f"  RMSE: {temp_metrics['rmse']:.2f} K")
        print(f"  Bias: {temp_metrics['bias']:.2f} K")
        print(f"  相关系数: {temp_metrics['corr']:.4f}")

        print(f"\n气压:")
        print(f"  RMSE: {pres_metrics['rmse']:.2f} mb")
        print(f"  Bias: {pres_metrics['bias']:.2f} mb")
        print(f"  相关系数: {pres_metrics['corr']:.4f}")

        # 保存
        output_dir = Path(f'Results/Q1_{sampler.upper()}')
        output_dir.mkdir(parents=True, exist_ok=True)

        np.save(output_dir / 'predictions.npy', pred_denorm)
        np.save(output_dir / 'ground_truth.npy', true_denorm)
        np.save(output_dir / 'metrics.npy', {'temperature': temp_metrics, 'pressure': pres_metrics})

        print(f"\n结果保存至: {output_dir}")

        results[sampler] = {
            'temperature': temp_metrics,
            'pressure': pres_metrics
        }

    # 对比结果
    if len(samplers) == 2:
        print(f"\n{'='*60}")
        print("=== DDPM vs DDIM 对比 ===")
        print(f"{'='*60}")
        print(f"{'指标':<20} {'DDPM':<15} {'DDIM':<15} {'差异':<10}")
        print("-" * 60)

        ddpm_temp_cc = results['ddpm']['temperature']['corr']
        ddim_temp_cc = results['ddim']['temperature']['corr']
        ddpm_pres_cc = results['ddpm']['pressure']['corr']
        ddim_pres_cc = results['ddim']['pressure']['corr']

        print(f"{'温度 CC':<20} {ddpm_temp_cc:<15.4f} {ddim_temp_cc:<15.4f} {ddim_temp_cc-ddpm_temp_cc:>9.4f}")
        print(f"{'气压 CC':<20} {ddpm_pres_cc:<15.4f} {ddim_pres_cc:<15.4f} {ddim_pres_cc-ddpm_pres_cc:>9.4f}")

        ddpm_temp_rmse = results['ddpm']['temperature']['rmse']
        ddim_temp_rmse = results['ddim']['temperature']['rmse']
        ddpm_pres_rmse = results['ddpm']['pressure']['rmse']
        ddim_pres_rmse = results['ddim']['pressure']['rmse']

        print(f"{'温度 RMSE (K)':<20} {ddpm_temp_rmse:<15.2f} {ddim_temp_rmse:<15.2f} {ddim_temp_rmse-ddpm_temp_rmse:>9.2f}")
        print(f"{'气压 RMSE (mb)':<20} {ddpm_pres_rmse:<15.2f} {ddim_pres_rmse:<15.2f} {ddim_pres_rmse-ddpm_pres_rmse:>9.2f}")

    # 与之前模型对比
    print(f"\n{'='*60}")
    print("=== 与之前模型对比 (13k样本) ===")
    print(f"{'='*60}")

    best_sampler = 'ddpm' if 'ddpm' in results else samplers[0]
    best_results = results[best_sampler]

    print(f"使用 {best_sampler.upper()} 采样结果对比:")
    print(f"{'指标':<20} {'之前模型':<15} {'Q1模型':<15} {'提升':<10}")
    print("-" * 60)

    temp_cc = best_results['temperature']['corr']
    pres_cc = best_results['pressure']['corr']
    temp_rmse = best_results['temperature']['rmse']
    pres_rmse = best_results['pressure']['rmse']

    print(f"{'温度 CC':<20} {'0.23':<15} {temp_cc:<15.4f} {(temp_cc-0.23)/0.23*100:>8.1f}%")
    print(f"{'气压 CC':<20} {'0.75':<15} {pres_cc:<15.4f} {(pres_cc-0.75)/0.75*100:>8.1f}%")
    print(f"{'温度 RMSE':<20} {'25.79 K':<15} {f'{temp_rmse:.2f} K':<15} {(25.79-temp_rmse)/25.79*100:>8.1f}%")
    print(f"{'气压 RMSE':<20} {'146.82 mb':<15} {f'{pres_rmse:.2f} mb':<15} {(146.82-pres_rmse)/146.82*100:>8.1f}%")


if __name__ == '__main__':
    main()
