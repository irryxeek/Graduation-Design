"""修复气压标准化问题 - 使用对数变换"""
import numpy as np
from pathlib import Path

data_dir = Path('Data/Processed_ATP')

# 加载数据
train_x = np.load(data_dir / 'train_x.npy')
train_y = np.load(data_dir / 'train_y.npy')
val_x = np.load(data_dir / 'val_x.npy')
val_y = np.load(data_dir / 'val_y.npy')
test_x = np.load(data_dir / 'test_x.npy')
test_y = np.load(data_dir / 'test_y.npy')
stats = np.load(data_dir / 'stats.npy', allow_pickle=True).item()

print(f"原始数据: train {train_y.shape}, val {val_y.shape}, test {test_y.shape}")

# 反标准化到物理空间
train_y_phys = train_y * stats['y_std'][:, None] + stats['y_mean'][:, None]
val_y_phys = val_y * stats['y_std'][:, None] + stats['y_mean'][:, None]
test_y_phys = test_y * stats['y_std'][:, None] + stats['y_mean'][:, None]

# 对气压应用 log10 变换
train_y_phys[:, 1, :] = np.log10(train_y_phys[:, 1, :] + 1e-6)
val_y_phys[:, 1, :] = np.log10(val_y_phys[:, 1, :] + 1e-6)
test_y_phys[:, 1, :] = np.log10(test_y_phys[:, 1, :] + 1e-6)

# 重新计算统计信息
y_mean = np.array([train_y_phys[:, 0, :].mean(), train_y_phys[:, 1, :].mean()])
y_std = np.array([train_y_phys[:, 0, :].std(), train_y_phys[:, 1, :].std()])

print(f"\n新统计信息:")
print(f"  温度: mean={y_mean[0]:.2f} K, std={y_std[0]:.2f} K")
print(f"  气压(log10): mean={y_mean[1]:.4f}, std={y_std[1]:.4f}")

# 重新标准化
train_y_new = (train_y_phys - y_mean[:, None]) / y_std[:, None]
val_y_new = (val_y_phys - y_mean[:, None]) / y_std[:, None]
test_y_new = (test_y_phys - y_mean[:, None]) / y_std[:, None]

print(f"\n标准化后范围:")
print(f"  温度: [{train_y_new[:, 0, :].min():.3f}, {train_y_new[:, 0, :].max():.3f}]")
print(f"  气压: [{train_y_new[:, 1, :].min():.3f}, {train_y_new[:, 1, :].max():.3f}]")

# 保存
np.save(data_dir / 'train_y.npy', train_y_new.astype(np.float32))
np.save(data_dir / 'val_y.npy', val_y_new.astype(np.float32))
np.save(data_dir / 'test_y.npy', test_y_new.astype(np.float32))

new_stats = {
    'x_mean': stats['x_mean'],
    'x_std': stats['x_std'],
    'y_mean': y_mean,
    'y_std': y_std,
    'pressure_log_transformed': True
}
np.save(data_dir / 'stats.npy', new_stats)

print(f"\n✓ 数据已更新，气压已转换为 log10 空间")
