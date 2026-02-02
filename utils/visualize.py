import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os

# ==========================================
# 1. 文件路径配置
# ==========================================
# ⚠️⚠️⚠️ 请在这里填入你真实的 COSMIC 文件路径
cosmic_file = r"D:\02_Study\01_Schoolwork\Graduation Design\Data\Sample\atmPrf_nrt_2026_001\atmPrf_C2E1.2026.001.00.00.G29_0001.0001_nc"  # <--- 请替换这里
era5_file = "era5_sample.nc"

try:
    # -----------------------------------------------------
    # 步骤 1: 读取文件
    # -----------------------------------------------------
    print(f"正在读取文件: {era5_file} ...")
    if not os.path.exists(era5_file):
        raise FileNotFoundError(f"找不到文件: {era5_file}，请检查路径是否正确！")

    ds_cosmic = xr.open_dataset(cosmic_file)
    ds_era5 = xr.open_dataset(era5_file)
    print("文件读取成功，正在处理数据...")

    # -----------------------------------------------------
    # 步骤 2: 智能识别 ERA5 维度名 (解决报错的关键)
    # -----------------------------------------------------
    # 检查时间维度
    if 'valid_time' in ds_era5.dims:
        time_dim = 'valid_time'
    else:
        time_dim = 'time'
    
    # 检查经纬度维度
    lat_dim = 'latitude' if 'latitude' in ds_era5.dims else 'lat'
    lon_dim = 'longitude' if 'longitude' in ds_era5.dims else 'lon'

    print(f"识别到维度名称: 时间=[{time_dim}], 纬度=[{lat_dim}], 经度=[{lon_dim}]")

    # -----------------------------------------------------
    # 步骤 3: 提取变量并取平均 (降维)
    # -----------------------------------------------------
    # t=温度, q=比湿, z=位势
    era_t = ds_era5['t'].mean(dim=[time_dim, lat_dim, lon_dim])
    era_q = ds_era5['q'].mean(dim=[time_dim, lat_dim, lon_dim])
    era_z = ds_era5['z'].mean(dim=[time_dim, lat_dim, lon_dim])

    # 转换高度: 位势 -> 几何高度 (km)
    g = 9.80665
    era_height_km = era_z / g / 1000.0

    print(f"ERA5 高度范围: {era_height_km.min().item():.2f} km ~ {era_height_km.max().item():.2f} km")

    # -----------------------------------------------------
    # 步骤 4: 准备 COSMIC 数据
    # -----------------------------------------------------
    cosmic_ba = ds_cosmic['Bend_ang']
    
    if 'Impact_height' in ds_cosmic:
        cosmic_height = ds_cosmic['Impact_height']
    else:
        cosmic_height = ds_cosmic['MSL_alt']

    # -----------------------------------------------------
    # 步骤 5: 插值对齐 (Interpolation)
    # -----------------------------------------------------
    # 确保 ERA5 高度是单调递增的
    if era_height_km[0] > era_height_km[-1]:
        era_height_km = era_height_km[::-1]
        era_t = era_t[::-1]
        era_q = era_q[::-1]

    # 插值函数: 用 ERA5 的高度和温度，构建映射函数
    f_temp = interp1d(era_height_km, era_t, kind='linear', fill_value="extrapolate")
    
    # 计算 COSMIC 高度上对应的 ERA5 温度
    era_t_interp = f_temp(cosmic_height)

    # -----------------------------------------------------
    # 步骤 6: 画图验证
    # -----------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(15, 6), sharey=True)

    # [左图] 弯曲角
    ax[0].plot(cosmic_ba, cosmic_height, 'b-', linewidth=1.5)
    ax[0].set_title("Input: Bending Angle\n(COSMIC-2)")
    ax[0].set_xlabel("Rad")
    ax[0].set_ylabel("Impact Height (km)")
    ax[0].grid(True, linestyle='--', alpha=0.6)
    ax[0].set_ylim(0, 40)
    ax[0].set_xlim(-0.002, 0.03)

    # [中图] 温度对比
    ax[1].plot(era_t, era_height_km, 'ro', label='ERA5 Raw', markersize=4, alpha=0.6)
    ax[1].plot(era_t_interp, cosmic_height, 'k--', label='Interpolated', linewidth=1)
    ax[1].set_title("Label: Temperature")
    ax[1].set_xlabel("Kelvin (K)")
    ax[1].grid(True, linestyle='--', alpha=0.6)
    ax[1].legend()

    # [右图] 湿度
    ax[2].plot(era_q * 1000, era_height_km, 'g-', linewidth=1.5)
    ax[2].set_title("Label: Specific Humidity")
    ax[2].set_xlabel("g/kg")
    ax[2].grid(True, linestyle='--', alpha=0.6)

    plt.suptitle("Data Pipeline Verification", fontsize=16)
    plt.tight_layout()
    
    print("正在显示图像...")
    plt.show()
    print("✅ 验证成功！数据预处理逻辑已跑通。")

except Exception as e:
    print(f"\n❌ 出错啦: {e}")
    import traceback
    traceback.print_exc()