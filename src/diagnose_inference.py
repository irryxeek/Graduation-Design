"""
诊断推理问题
"""
import numpy as np
import matplotlib.pyplot as plt

# 加载数据
print("=== 加载数据 ===")
predictions = np.load('Results/ATP_Inference/predictions.npy')
ground_truth = np.load('Results/ATP_Inference/ground_truth.npy')
test_x = np.load('Data/Processed_ATP/test_x.npy')
test_y = np.load('Data/Processed_ATP/test_y.npy')
stats = np.load('Data/Processed_ATP/stats.npy', allow_pickle=True).item()

print(f"预测形状: {predictions.shape}")
print(f"真值形状: {ground_truth.shape}")
print(f"测试输入形状: {test_x.shape}")
print(f"测试标签形状: {test_y.shape}")

print("\n=== 数据统计 ===")
print(f"预测 - 温度: mean={predictions[:, 0, :].mean():.2f}, std={predictions[:, 0, :].std():.2f}, min={predictions[:, 0, :].min():.2f}, max={predictions[:, 0, :].max():.2f}")
print(f"真值 - 温度: mean={ground_truth[:, 0, :].mean():.2f}, std={ground_truth[:, 0, :].std():.2f}, min={ground_truth[:, 0, :].min():.2f}, max={ground_truth[:, 0, :].max():.2f}")
print(f"预测 - 气压: mean={predictions[:, 1, :].mean():.2f}, std={predictions[:, 1, :].std():.2f}, min={predictions[:, 1, :].min():.2f}, max={predictions[:, 1, :].max():.2f}")
print(f"真值 - 气压: mean={ground_truth[:, 1, :].mean():.2f}, std={ground_truth[:, 1, :].std():.2f}, min={ground_truth[:, 1, :].min():.2f}, max={ground_truth[:, 1, :].max():.2f}")

print("\n=== 标准化统计 ===")
print(f"y_mean: {stats['y_mean']}")
print(f"y_std: {stats['y_std']}")

print("\n=== 检查标准化前的预测 ===")
# 重新标准化预测，看看标准化空间的值
pred_norm_temp = (predictions[:, 0, :] - stats['y_mean'][0]) / stats['y_std'][0]
pred_norm_pres = (predictions[:, 1, :] - stats['y_mean'][1]) / stats['y_std'][1]
print(f"标准化后预测 - 温度: mean={pred_norm_temp.mean():.4f}, std={pred_norm_temp.std():.4f}")
print(f"标准化后预测 - 气压: mean={pred_norm_pres.mean():.4f}, std={pred_norm_pres.std():.4f}")
print(f"标准化后真值 - 温度: mean={test_y[:, 0, :].mean():.4f}, std={test_y[:, 0, :].std():.4f}")
print(f"标准化后真值 - 气压: mean={test_y[:, 1, :].mean():.4f}, std={test_y[:, 1, :].std():.4f}")

print("\n=== 检查单个样本 ===")
idx = 0
print(f"样本 {idx}:")
print(f"  输入 (弯曲角): mean={test_x[idx].mean():.4f}, std={test_x[idx].std():.4f}")
print(f"  真值温度: {ground_truth[idx, 0, :5]} ... {ground_truth[idx, 0, -5:]}")
print(f"  预测温度: {predictions[idx, 0, :5]} ... {predictions[idx, 0, -5:]}")
print(f"  真值气压: {ground_truth[idx, 1, :5]} ... {ground_truth[idx, 1, -5:]}")
print(f"  预测气压: {predictions[idx, 1, :5]} ... {predictions[idx, 1, -5:]}")

# 检查是否所有预测都相同
print("\n=== 检查预测多样性 ===")
pred_var_temp = np.var(predictions[:, 0, :], axis=0).mean()
pred_var_pres = np.var(predictions[:, 1, :], axis=0).mean()
print(f"预测温度的样本间方差: {pred_var_temp:.2f}")
print(f"预测气压的样本间方差: {pred_var_pres:.2f}")
print(f"真值温度的样本间方差: {np.var(ground_truth[:, 0, :], axis=0).mean():.2f}")
print(f"真值气压的样本间方差: {np.var(ground_truth[:, 1, :], axis=0).mean():.2f}")

# 可视化第一个样本
heights = np.linspace(0, 60, 301)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(ground_truth[0, 0, :], heights, 'b-', label='True', linewidth=2)
axes[0].plot(predictions[0, 0, :], heights, 'r--', label='Pred', linewidth=2)
axes[0].set_xlabel('Temperature (K)')
axes[0].set_ylabel('Height (km)')
axes[0].set_title('Sample 0 - Temperature')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(ground_truth[0, 1, :], heights, 'b-', label='True', linewidth=2)
axes[1].plot(predictions[0, 1, :], heights, 'r--', label='Pred', linewidth=2)
axes[1].set_xlabel('Pressure (mb)')
axes[1].set_ylabel('Height (km)')
axes[1].set_title('Sample 0 - Pressure')
axes[1].set_xscale('log')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Results/ATP_Inference/diagnostic_sample0.png', dpi=150)
print("\n诊断图已保存到 Results/ATP_Inference/diagnostic_sample0.png")
