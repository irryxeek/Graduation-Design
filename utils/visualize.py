"""
数据管线可视化验证工具
======================
读取 COSMIC 弯曲角 + ERA5 温度/湿度样本, 绘制对比图,
用于快速验证数据预处理流水线的正确性。

用法:
  python utils/visualize.py
  python utils/visualize.py --cosmic <atmPrf_file> --era5 <era5_file>
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from scipy.interpolate import interp1d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ro_retrieval.config import PROJECT_ROOT, PROCESSED_DIR, STD_HEIGHT


def parse_args():
    parser = argparse.ArgumentParser(description="数据管线可视化验证")
    parser.add_argument(
        "--cosmic", type=str,
        default=os.path.join(
            PROJECT_ROOT, "Data", "Sample", "atmPrf_nrt_2026_001",
            "atmPrf_C2E1.2026.001.00.00.G29_0001.0001_nc",
        ),
        help="COSMIC atmPrf 文件路径",
    )
    parser.add_argument(
        "--era5", type=str,
        default=os.path.join(PROJECT_ROOT, "era5_sample.nc"),
        help="ERA5 NetCDF 文件路径",
    )
    parser.add_argument("--processed", action="store_true",
                        help="可视化已处理的 train_x / train_y 数据")
    return parser.parse_args()


def visualize_raw(cosmic_file, era5_file):
    """可视化原始 COSMIC + ERA5 数据"""
    if not os.path.exists(cosmic_file):
        print(f"COSMIC 文件不存在: {cosmic_file}")
        return
    if not os.path.exists(era5_file):
        print(f"ERA5 文件不存在: {era5_file}")
        return

    ds_cosmic = xr.open_dataset(cosmic_file)
    ds_era5 = xr.open_dataset(era5_file)

    # 自适应维度名
    time_dim = 'valid_time' if 'valid_time' in ds_era5.dims else 'time'
    lat_dim = 'latitude' if 'latitude' in ds_era5.dims else 'lat'
    lon_dim = 'longitude' if 'longitude' in ds_era5.dims else 'lon'

    era_t = ds_era5['t'].mean(dim=[time_dim, lat_dim, lon_dim]).values
    era_q = ds_era5['q'].mean(dim=[time_dim, lat_dim, lon_dim]).values
    era_z = ds_era5['z'].mean(dim=[time_dim, lat_dim, lon_dim]).values

    g = 9.80665
    era_height_km = era_z / g / 1000.0

    cosmic_ba = ds_cosmic['Bend_ang'].values
    cosmic_height = (
        ds_cosmic['Impact_height'].values
        if 'Impact_height' in ds_cosmic
        else ds_cosmic['MSL_alt'].values
    )

    if era_height_km[0] > era_height_km[-1]:
        era_height_km = era_height_km[::-1]
        era_t = era_t[::-1]
        era_q = era_q[::-1]

    f_temp = interp1d(era_height_km, era_t, kind='linear', fill_value="extrapolate")
    era_t_interp = f_temp(cosmic_height)

    fig, ax = plt.subplots(1, 3, figsize=(15, 6), sharey=True)

    ax[0].plot(cosmic_ba, cosmic_height, 'b-', linewidth=1.5)
    ax[0].set_title("Input: Bending Angle\n(COSMIC-2)")
    ax[0].set_xlabel("Rad")
    ax[0].set_ylabel("Height (km)")
    ax[0].grid(True, linestyle='--', alpha=0.6)
    ax[0].set_ylim(0, 40)

    ax[1].plot(era_t, era_height_km, 'ro', label='ERA5 Raw', markersize=4, alpha=0.6)
    ax[1].plot(era_t_interp, cosmic_height, 'k--', label='Interpolated', linewidth=1)
    ax[1].set_title("Label: Temperature")
    ax[1].set_xlabel("Kelvin (K)")
    ax[1].grid(True, linestyle='--', alpha=0.6)
    ax[1].legend()

    ax[2].plot(era_q * 1000, era_height_km, 'g-', linewidth=1.5)
    ax[2].set_title("Label: Specific Humidity")
    ax[2].set_xlabel("g/kg")
    ax[2].grid(True, linestyle='--', alpha=0.6)

    plt.suptitle("Data Pipeline Verification", fontsize=16)
    plt.tight_layout()
    plt.show()
    print("可视化完成。")

    ds_cosmic.close()
    ds_era5.close()


def visualize_processed():
    """可视化已处理的标准化数据"""
    heights = np.linspace(0, 60, STD_HEIGHT)

    for split_name in ["train", "val", "test"]:
        x_path = os.path.join(PROCESSED_DIR, f"{split_name}_x.npy")
        y_path = os.path.join(PROCESSED_DIR, f"{split_name}_y.npy")
        if not os.path.exists(x_path):
            continue

        X = np.load(x_path)
        Y = np.load(y_path)
        print(f"[{split_name}] X={X.shape}, Y={Y.shape}")

        # 随机抽取 3 个样本展示
        n = min(3, len(X))
        indices = np.random.choice(len(X), n, replace=False)

        fig, axes = plt.subplots(n, 2, figsize=(12, 4 * n), squeeze=False)
        fig.suptitle(f"{split_name.upper()} Set Samples", fontsize=14)

        for row, idx in enumerate(indices):
            axes[row, 0].plot(X[idx], heights, 'b-', linewidth=1.5)
            axes[row, 0].set_xlabel("log10(BA)")
            axes[row, 0].set_ylabel("Height (km)")
            axes[row, 0].set_title(f"Sample #{idx} - Bending Angle")
            axes[row, 0].grid(True, alpha=0.3)

            if Y.ndim == 3:
                var_labels = ["Temp (K)", "Pres (hPa)", "Humidity (kg/kg)"]
                for v in range(min(Y.shape[1], 3)):
                    axes[row, 1].plot(Y[idx, v], heights, label=var_labels[v])
                axes[row, 1].legend(fontsize=8)
            else:
                axes[row, 1].plot(Y[idx], heights, 'r-', linewidth=1.5)
            axes[row, 1].set_xlabel("Value")
            axes[row, 1].set_title(f"Sample #{idx} - Labels")
            axes[row, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


def main():
    args = parse_args()

    if args.processed:
        visualize_processed()
    else:
        visualize_raw(args.cosmic, args.era5)


if __name__ == "__main__":
    main()
