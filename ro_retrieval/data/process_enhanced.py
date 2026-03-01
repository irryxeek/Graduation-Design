"""
增强预处理流水线
================
从 COSMIC atmPrf netCDF 文件中提取弯曲角、温度、气压剖面，
执行质量控制、插值到标准高度网格、Z-Score 标准化，
并划分 train / val / test。

支持两种模式:
  - 仅 atmPrf (弯曲角 → 温度 + 气压，无湿度)
  - atmPrf + wetPf2 (弯曲角 → 温度 + 气压 + 湿度)
"""

import os
import glob
import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm

from ro_retrieval.config import (
    STD_HEIGHT, HEIGHT_MIN, HEIGHT_MAX,
    MIN_PENETRATION_HEIGHT, MIN_VALID_POINTS,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
)


def _load_nc(path):
    """兼容 xarray 或 netCDF4 打开 netCDF 文件"""
    try:
        import xarray as xr
        ds = xr.open_dataset(path)
        return ds
    except Exception:
        pass
    try:
        import netCDF4 as nc
        ds = nc.Dataset(path)
        return ds
    except Exception:
        return None


def _get_var(ds, names):
    """从 dataset 中尝试多个变量名，返回 numpy array"""
    for name in names:
        try:
            if hasattr(ds, 'variables'):
                if name in ds.variables:
                    v = ds.variables[name][:]
                    if hasattr(v, 'values'):
                        return v.values.astype(np.float64)
                    return np.asarray(v, dtype=np.float64)
            if hasattr(ds, '__getitem__'):
                v = ds[name]
                if hasattr(v, 'values'):
                    return v.values.astype(np.float64)
                return np.asarray(v, dtype=np.float64)
        except (KeyError, IndexError):
            continue
    return None


def process_single_atmprf(atm_path, std_height):
    """
    处理单个 atmPrf 文件，提取弯曲角 (输入) + 温度/气压 (标签)

    Returns:
        (x_vec, y_vec) 或 None (如果 QC 失败)
        x_vec: (301,)  — log10(|BA| + ε) 弯曲角
        y_vec: (2, 301) — [温度(K), 气压(hPa)]
    """
    ds = _load_nc(atm_path)
    if ds is None:
        return None

    try:
        h = _get_var(ds, ['MSL_alt', 'msl_alt', 'altitude', 'Alt'])
        ba = _get_var(ds, ['Bend_ang', 'bend_ang', 'ba', 'bendingAngle'])
        temp = _get_var(ds, ['Temp', 'temp', 'T', 'temperature'])
        pres = _get_var(ds, ['Pres', 'pres', 'P', 'pressure'])
    except Exception:
        return None
    finally:
        if hasattr(ds, 'close'):
            ds.close()

    if h is None or ba is None or temp is None or pres is None:
        return None

    # --- 掩码：去除 NaN 和 masked 数据 ---
    h = np.ma.filled(h, np.nan) if hasattr(h, 'filled') else h
    ba = np.ma.filled(ba, np.nan) if hasattr(ba, 'filled') else ba
    temp = np.ma.filled(temp, np.nan) if hasattr(temp, 'filled') else temp
    pres = np.ma.filled(pres, np.nan) if hasattr(pres, 'filled') else pres

    # --- QC: 弯曲角 ---
    mask_ba = np.isfinite(ba) & np.isfinite(h) & (ba > 0)
    h_ba = h[mask_ba]
    ba_clean = ba[mask_ba]

    if len(ba_clean) < MIN_VALID_POINTS:
        return None

    # 穿透高度 QC
    if h_ba.min() > MIN_PENETRATION_HEIGHT:
        return None

    # --- QC: 温度 ---
    mask_t = np.isfinite(temp) & np.isfinite(h)
    h_t = h[mask_t]
    temp_clean = temp[mask_t]

    if len(temp_clean) < MIN_VALID_POINTS:
        return None

    # 温度单位：如果小于 100 可能是摄氏度，转为开尔文
    if np.nanmax(temp_clean) < 100:
        temp_clean = temp_clean + 273.15

    # 温度物理范围 QC: 150K ~ 350K
    valid_t = (temp_clean >= 150) & (temp_clean <= 350)
    if valid_t.sum() < MIN_VALID_POINTS:
        return None
    h_t = h_t[valid_t]
    temp_clean = temp_clean[valid_t]

    # --- QC: 气压 ---
    mask_p = np.isfinite(pres) & np.isfinite(h) & (pres > 0)
    h_p = h[mask_p]
    pres_clean = pres[mask_p]

    if len(pres_clean) < MIN_VALID_POINTS:
        return None

    # --- 插值到标准高度网格 ---
    try:
        f_ba = interp1d(h_ba, ba_clean, kind='linear',
                        bounds_error=False, fill_value=0)
        x_vec = f_ba(std_height)
        x_vec = np.log10(np.abs(x_vec) + 1e-6)

        # 温度: 边界用最近有效值填充
        f_temp = interp1d(h_t, temp_clean, kind='linear',
                          bounds_error=False,
                          fill_value=(temp_clean[np.argmin(h_t)],
                                      temp_clean[np.argmax(h_t)]))
        temp_vec = f_temp(std_height)
        # 温度物理范围裁剪
        temp_vec = np.clip(temp_vec, 150, 350)

        # 气压: 边界用最近有效值填充
        f_pres = interp1d(h_p, pres_clean, kind='linear',
                          bounds_error=False,
                          fill_value=(pres_clean[np.argmin(h_p)],
                                      pres_clean[np.argmax(h_p)]))
        pres_vec = f_pres(std_height)
        # 气压物理范围裁剪
        pres_vec = np.clip(pres_vec, 0.01, 1200)

    except Exception:
        return None

    # 检查插值结果有效性
    if np.any(np.isnan(x_vec)) or np.any(np.isnan(temp_vec)):
        return None

    # y_vec: (2, 301) — [温度, 气压]
    y_vec = np.stack([temp_vec, pres_vec], axis=0)

    return x_vec.astype(np.float32), y_vec.astype(np.float32)


def run_enhanced_pipeline(
    atm_root: str,
    wet_root: str = None,
    output_dir: str = None,
    era5_root: str = None,
    strict_qc: bool = True,
    do_split: bool = True,
):
    """
    增强预处理流水线主函数

    Args:
        atm_root: atmPrf 数据目录
        wet_root: wetPf2 数据目录 (可选, 当前未使用)
        output_dir: 输出目录
        era5_root: ERA5 数据目录 (可选, 当前未使用)
        strict_qc: 是否使用严格 QC
        do_split: 是否划分 train/val/test

    Returns:
        (X, Y, report) 或 None
    """
    os.makedirs(output_dir, exist_ok=True)

    # 标准高度网格
    std_height = np.linspace(HEIGHT_MIN, HEIGHT_MAX, STD_HEIGHT)

    # 搜索 atmPrf 文件
    atm_files = []
    for pattern in ['*_nc', '*.nc', '*.NC']:
        atm_files.extend(glob.glob(os.path.join(atm_root, '**', pattern), recursive=True))

    # 去重
    atm_files = list(set(atm_files))
    atm_files.sort()

    print(f"找到 {len(atm_files)} 个 atmPrf 文件")

    if len(atm_files) == 0:
        print("错误: 未找到任何 atmPrf 文件!")
        return None

    # --- 逐文件处理 ---
    X_list = []
    Y_list = []
    n_fail = 0

    for fpath in tqdm(atm_files, desc="处理 atmPrf 文件"):
        result = process_single_atmprf(fpath, std_height)
        if result is not None:
            x_vec, y_vec = result
            X_list.append(x_vec)
            Y_list.append(y_vec)
        else:
            n_fail += 1

    if len(X_list) == 0:
        print("错误: 所有文件处理失败!")
        return None

    X = np.array(X_list)  # (N, 301)
    Y = np.array(Y_list)  # (N, 2, 301) — [温度, 气压]

    # 添加 0 填充的湿度通道 → (N, 3, 301)
    humidity = np.zeros((len(X), 1, STD_HEIGHT), dtype=np.float32)
    Y = np.concatenate([Y, humidity], axis=1)  # (N, 3, 301)

    print(f"\n处理完成: 成功 {len(X)} / 总 {len(atm_files)} "
          f"(失败 {n_fail}, 成功率 {len(X)/len(atm_files)*100:.1f}%)")
    print(f"  X shape: {X.shape}")
    print(f"  Y shape: {Y.shape}")

    # --- Z-Score 标准化 ---
    print("\n正在 Z-Score 标准化...")

    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0) + 1e-8
    X_norm = (X - x_mean) / x_std

    # 分通道标准化 Y
    y_means = []
    y_stds = []
    Y_norm = np.zeros_like(Y)
    channel_names = ['温度(K)', '气压(hPa)', '湿度']

    for c in range(Y.shape[1]):
        ch = Y[:, c, :]
        m = ch.mean()
        s = ch.std() + 1e-8
        y_means.append(float(m))
        y_stds.append(float(s))
        Y_norm[:, c, :] = (ch - m) / s
        print(f"  {channel_names[c]}: mean={m:.4f}, std={s:.4f}")

    # 保存标准化参数
    norm_params = {
        'x_mean': x_mean, 'x_std': x_std,
        'y_means': np.array(y_means), 'y_stds': np.array(y_stds),
    }
    np.savez(os.path.join(output_dir, 'norm_params.npz'), **norm_params)

    report = {
        'total_files': len(atm_files),
        'success': len(X),
        'failed': n_fail,
        'qc_pass_rate': f"{len(X)/len(atm_files)*100:.1f}%",
        'x_shape': list(X_norm.shape),
        'y_shape': list(Y_norm.shape),
    }

    # --- 数据划分 ---
    if do_split:
        n = len(X_norm)
        indices = np.random.RandomState(42).permutation(n)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        np.save(os.path.join(output_dir, 'train_x.npy'), X_norm[train_idx])
        np.save(os.path.join(output_dir, 'train_y.npy'), Y_norm[train_idx])
        np.save(os.path.join(output_dir, 'val_x.npy'), X_norm[val_idx])
        np.save(os.path.join(output_dir, 'val_y.npy'), Y_norm[val_idx])
        np.save(os.path.join(output_dir, 'test_x.npy'), X_norm[test_idx])
        np.save(os.path.join(output_dir, 'test_y.npy'), Y_norm[test_idx])

        print(f"\n数据划分完成:")
        print(f"  训练集: {len(train_idx)}")
        print(f"  验证集: {len(val_idx)}")
        print(f"  测试集: {len(test_idx)}")
    else:
        np.save(os.path.join(output_dir, 'train_x.npy'), X_norm)
        np.save(os.path.join(output_dir, 'train_y.npy'), Y_norm)

    return X_norm, Y_norm, report
