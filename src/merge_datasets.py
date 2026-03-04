"""
合并1月和2月的 ATP 数据
"""
import numpy as np
from pathlib import Path

def merge_datasets():
    print("=== 合并 1月 和 2月 ATP 数据 ===\n")

    # 加载1月数据（原始 ATP 数据）
    jan_dir = Path('Data/Processed_ATP')
    print("加载1月数据...")
    jan_train_x = np.load(jan_dir / 'train_x.npy')
    jan_train_y = np.load(jan_dir / 'train_y.npy')
    jan_val_x = np.load(jan_dir / 'val_x.npy')
    jan_val_y = np.load(jan_dir / 'val_y.npy')
    jan_test_x = np.load(jan_dir / 'test_x.npy')
    jan_test_y = np.load(jan_dir / 'test_y.npy')
    jan_stats = np.load(jan_dir / 'stats.npy', allow_pickle=True).item()

    print(f"  训练集: {len(jan_train_x)}")
    print(f"  验证集: {len(jan_val_x)}")
    print(f"  测试集: {len(jan_test_x)}")

    # 加载2月数据
    feb_dir = Path('Data/Processed_ATP_Feb')
    print("\n加载2月数据...")
    feb_train_x = np.load(feb_dir / 'train_x.npy')
    feb_train_y = np.load(feb_dir / 'train_y.npy')
    feb_val_x = np.load(feb_dir / 'val_x.npy')
    feb_val_y = np.load(feb_dir / 'val_y.npy')
    feb_test_x = np.load(feb_dir / 'test_x.npy')
    feb_test_y = np.load(feb_dir / 'test_y.npy')

    print(f"  训练集: {len(feb_train_x)}")
    print(f"  验证集: {len(feb_val_x)}")
    print(f"  测试集: {len(feb_test_x)}")

    # 合并数据
    print("\n合并数据...")
    merged_train_x = np.concatenate([jan_train_x, feb_train_x], axis=0)
    merged_train_y = np.concatenate([jan_train_y, feb_train_y], axis=0)
    merged_val_x = np.concatenate([jan_val_x, feb_val_x], axis=0)
    merged_val_y = np.concatenate([jan_val_y, feb_val_y], axis=0)
    merged_test_x = np.concatenate([jan_test_x, feb_test_x], axis=0)
    merged_test_y = np.concatenate([jan_test_y, feb_test_y], axis=0)

    print(f"\n合并后数据量:")
    print(f"  训练集: {len(merged_train_x)} ({len(jan_train_x)} + {len(feb_train_x)})")
    print(f"  验证集: {len(merged_val_x)} ({len(jan_val_x)} + {len(feb_val_x)})")
    print(f"  测试集: {len(merged_test_x)} ({len(jan_test_x)} + {len(feb_test_x)})")
    print(f"  总计: {len(merged_train_x) + len(merged_val_x) + len(merged_test_x)}")

    # 重新计算统计信息（基于合并后的训练集）
    print("\n重新计算标准化统计...")
    x_mean = merged_train_x.mean()
    x_std = merged_train_x.std()
    y_mean = merged_train_y.mean(axis=(0, 2))  # (2,) for temp and pressure
    y_std = merged_train_y.std(axis=(0, 2))

    merged_stats = {
        'x_mean': x_mean,
        'x_std': x_std,
        'y_mean': y_mean,
        'y_std': y_std
    }

    print(f"  x_mean: {x_mean:.4f}, x_std: {x_std:.4f}")
    print(f"  y_mean: {y_mean}")
    print(f"  y_std: {y_std}")

    # 保存合并后的数据
    output_dir = Path('Data/Processed_ATP_Merged')
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n保存到 {output_dir}...")
    np.save(output_dir / 'train_x.npy', merged_train_x)
    np.save(output_dir / 'train_y.npy', merged_train_y)
    np.save(output_dir / 'val_x.npy', merged_val_x)
    np.save(output_dir / 'val_y.npy', merged_val_y)
    np.save(output_dir / 'test_x.npy', merged_test_x)
    np.save(output_dir / 'test_y.npy', merged_test_y)
    np.save(output_dir / 'stats.npy', merged_stats)

    print("\n=== 合并完成 ===")
    return output_dir


if __name__ == '__main__':
    merge_datasets()
