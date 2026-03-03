"""
测试加权模型的噪声预测能力
对比原始模型和加权模型在不同时间步的表现
"""
import torch
import numpy as np
import sys
sys.path.append('/root/autodl-tmp/Graduation-Design')

from ro_retrieval.model.unet import EnhancedConditionalUNet1D
from ro_retrieval.model.diffusion import DiffusionSchedule


def test_model(model_path, model_name):
    """测试单个模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载测试数据
    test_x = np.load('Data/Processed_ATP/test_x.npy')[:10]
    test_y = np.load('Data/Processed_ATP/test_y.npy')[:10]

    # 加载模型
    model = EnhancedConditionalUNet1D(
        in_channels=2,
        cond_channels=1,
        out_channels=2,
        base_dim=64,
        time_dim=128,
        num_heads=4
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint)
    model.eval()

    # 初始化扩散调度器
    schedule = DiffusionSchedule(timesteps=1000, beta_start=1e-4, beta_end=0.02, device=device)

    # 测试不同时间步
    timesteps_to_test = [0, 50, 100, 200, 500, 999]
    results = {}

    with torch.no_grad():
        for t_val in timesteps_to_test:
            condition = torch.from_numpy(test_x).float().to(device)
            condition = condition.unsqueeze(1)
            x_0 = torch.from_numpy(test_y).float().to(device)

            t = torch.full((len(test_x), 1), t_val, device=device).long()
            noise_true = torch.randn_like(x_0)
            x_t = schedule.q_sample(x_0, t, noise_true)

            noise_pred = model(x_t, t, condition)

            mse = torch.mean((noise_pred - noise_true) ** 2).item()
            corr_flat = np.corrcoef(
                noise_true.cpu().numpy().flatten(),
                noise_pred.cpu().numpy().flatten()
            )[0, 1]

            results[t_val] = {'mse': mse, 'corr': corr_flat}

    return results


def main():
    print("=== 对比原始模型 vs 加权模型 ===\n")

    # 测试原始模型
    print("测试原始模型...")
    original_results = test_model('enhanced_ro_diffusion_best.pth', '原始模型')

    # 测试加权模型
    print("测试加权模型...")
    weighted_results = test_model('enhanced_ro_diffusion_weighted_best.pth', '加权模型')

    # 对比结果
    print("\n" + "="*70)
    print(f"{'时间步':<10} {'原始模型 MSE':<15} {'加权模型 MSE':<15} {'改善':<10}")
    print(f"{'时间步':<10} {'原始模型 Corr':<15} {'加权模型 Corr':<15} {'改善':<10}")
    print("="*70)

    for t in [0, 50, 100, 200, 500, 999]:
        orig_mse = original_results[t]['mse']
        weight_mse = weighted_results[t]['mse']
        mse_improve = ((orig_mse - weight_mse) / orig_mse * 100) if orig_mse > 0 else 0

        orig_corr = original_results[t]['corr']
        weight_corr = weighted_results[t]['corr']
        corr_improve = ((weight_corr - orig_corr) / (1 - orig_corr) * 100) if orig_corr < 1 else 0

        print(f"\nt={t:<8}")
        print(f"  MSE:  {orig_mse:>12.6f}  {weight_mse:>12.6f}  {mse_improve:>8.1f}%")
        print(f"  Corr: {orig_corr:>12.4f}  {weight_corr:>12.4f}  {corr_improve:>8.1f}%")

    # 重点关注低时间步
    print("\n" + "="*70)
    print("关键发现：")
    print("="*70)

    low_t_avg_orig = np.mean([original_results[t]['mse'] for t in [0, 50, 100, 200]])
    low_t_avg_weight = np.mean([weighted_results[t]['mse'] for t in [0, 50, 100, 200]])
    low_t_improve = (low_t_avg_orig - low_t_avg_weight) / low_t_avg_orig * 100

    print(f"低时间步 (t=0-200) 平均 MSE:")
    print(f"  原始模型: {low_t_avg_orig:.6f}")
    print(f"  加权模型: {low_t_avg_weight:.6f}")
    print(f"  改善: {low_t_improve:.1f}%")

    if low_t_improve > 10:
        print("\n✓ 加权训练显著改善了低时间步性能！")
        print("  建议使用加权模型进行推理。")
    elif low_t_improve > 0:
        print("\n✓ 加权训练略微改善了低时间步性能。")
        print("  可以尝试使用加权模型进行推理。")
    else:
        print("\n✗ 加权训练未能改善低时间步性能。")
        print("  可能需要更长的训练时间或调整权重策略。")


if __name__ == '__main__':
    main()
