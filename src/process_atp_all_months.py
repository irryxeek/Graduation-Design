"""处理 1-3 月 FY-3D ATP 数据并合并"""
import sys
sys.path.append('/root/autodl-tmp/Graduation-Design')

import numpy as np
from pathlib import Path
from tqdm import tqdm
import netCDF4 as nc
from ro_retrieval.data.atp_process import ATPProcessor

def main():
    # 数据目录
    atp_dirs = [
        'utils/Jan_atp',
        'utils/Feb_atp',
        'utils/Mar_atp'
    ]

    output_dir = Path('Data/Processed_ATP_Q1')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("处理 FY-3D ATP 数据 (2025年1-3月)")
    print("="*60)

    # 初始化处理器
    processor = ATPProcessor()

    all_x = []
    all_y = []
    total_files = 0
    success_count = 0

    # 处理每个月的数据
    for atp_dir in atp_dirs:
        atp_path = Path(atp_dir)
        if not atp_path.exists():
            print(f"警告: {atp_dir} 不存在，跳过")
            continue

        atp_files = sorted(list(atp_path.glob('*.NC')) + list(atp_path.glob('*.nc')))
        month_name = atp_path.name.replace('_atp', '')

        print(f"\n处理 {month_name}: {len(atp_files)} 个文件")

        month_x = []
        month_y = []
        month_success = 0

        for atp_file in tqdm(atp_files, desc=f"{month_name}"):
            try:
                # 读取文件
                data = processor.read_atp_file(atp_file)
                if data is None:
                    continue

                # 处理单个廓线
                result = processor.process_single_profile(data)
                if result is None:
                    continue

                x, y = result
                month_x.append(x)
                month_y.append(y)
                month_success += 1
            except Exception as e:
                continue

        if month_x:
            all_x.extend(month_x)
            all_y.extend(month_y)
            success_count += month_success
            print(f"{month_name} 成功处理: {month_success}/{len(atp_files)} ({month_success/len(atp_files)*100:.1f}%)")

        total_files += len(atp_files)

    # 转换为数组
    print(f"\n总计: 成功处理 {success_count}/{total_files} ({success_count/total_files*100:.1f}%)")

    if not all_x:
        print("错误: 没有成功处理任何文件")
        return

    all_x = np.array(all_x)
    all_y = np.array(all_y)

    print(f"\n数据形状:")
    print(f"  X (弯曲角): {all_x.shape}")
    print(f"  Y (温度+气压): {all_y.shape}")

    # 计算统计量（用于标准化）
    print("\n计算标准化统计量...")

    # 弯曲角已经是 log10 变换后的
    x_mean = np.mean(all_x)
    x_std = np.std(all_x)

    # 温度和气压分别计算
    y_mean = np.mean(all_y, axis=(0, 2))  # (2,)
    y_std = np.std(all_y, axis=(0, 2))

    print(f"\n标准化统计:")
    print(f"  弯曲角 (log10): mean={x_mean:.4f}, std={x_std:.4f}")
    print(f"  温度: mean={y_mean[0]:.2f} K, std={y_std[0]:.2f} K")
    print(f"  气压 (log10): mean={y_mean[1]:.4f}, std={y_std[1]:.4f}")

    # 标准化
    x_norm = (all_x - x_mean) / x_std
    y_norm = np.zeros_like(all_y)
    for i in range(2):
        y_norm[:, i, :] = (all_y[:, i, :] - y_mean[i]) / y_std[i]

    # 划分数据集 (70% train, 15% val, 15% test)
    n_samples = len(x_norm)
    indices = np.random.permutation(n_samples)

    n_train = int(0.7 * n_samples)
    n_val = int(0.15 * n_samples)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]

    print(f"\n数据集划分:")
    print(f"  训练集: {len(train_idx)} ({len(train_idx)/n_samples*100:.1f}%)")
    print(f"  验证集: {len(val_idx)} ({len(val_idx)/n_samples*100:.1f}%)")
    print(f"  测试集: {len(test_idx)} ({len(test_idx)/n_samples*100:.1f}%)")

    # 保存
    print(f"\n保存到 {output_dir}/")
    np.save(output_dir / 'train_x.npy', x_norm[train_idx])
    np.save(output_dir / 'train_y.npy', y_norm[train_idx])
    np.save(output_dir / 'val_x.npy', x_norm[val_idx])
    np.save(output_dir / 'val_y.npy', y_norm[val_idx])
    np.save(output_dir / 'test_x.npy', x_norm[test_idx])
    np.save(output_dir / 'test_y.npy', y_norm[test_idx])

    # 保存统计量
    stats = {
        'x_mean': x_mean,
        'x_std': x_std,
        'y_mean': y_mean,
        'y_std': y_std,
        'pressure_log_transformed': True,
        'n_samples': n_samples,
        'n_train': len(train_idx),
        'n_val': len(val_idx),
        'n_test': len(test_idx)
    }
    np.save(output_dir / 'stats.npy', stats)

    print("\n处理完成！")
    print(f"数据已保存到: {output_dir}")


if __name__ == '__main__':
    main()
