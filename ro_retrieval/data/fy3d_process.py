"""
FY-3D GNOS L1 掩星数据预处理
==============================
从 FY-3D GNOS L1 级 netCDF (AEG) 文件中推导弯曲角剖面。

处理流程:
  excess phase (exLC) → Doppler (dexLC/dt) → 几何光学反演 → 弯曲角 α(h)

注意:
  - 仅处理 AEG 文件 (大气层 GPS 掩星, 50 Hz 开环追踪)
  - IEG (电离层) 和 POD (精密定轨) 文件会被跳过
  - L1 数据不含温度/气压/湿度, 标签需由 ERA5 再分析数据提供
"""

import os
import glob
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from tqdm import tqdm

from ro_retrieval.config import (
    STD_HEIGHT, HEIGHT_MIN, HEIGHT_MAX,
    MIN_VALID_POINTS, TRAIN_RATIO, VAL_RATIO,
)

# ==================== 物理常数 ====================
C_LIGHT = 299792458.0      # 光速 (m/s)
F_GPS_L1 = 1575.42e6       # GPS L1 频率 (Hz)
R_EARTH = 6371.0           # 地球平均半径 (km)
LAMBDA_L1 = C_LIGHT / F_GPS_L1  # L1 波长 (m)

# ==================== QC 阈值 ====================
FILL_VALUE = -990.0        # NC 文件中的缺失值标记 (实际为 -999)
MIN_VALID_EX_FRAC = 0.3    # exLC 有效数据最低比例
MIN_PROFILE_POINTS = 50    # 弯曲角剖面最低有效点数
BA_MAX_MRAD = 100.0        # 弯曲角上限 (mrad), 超出视为异常
BA_MIN_MRAD = -1.0         # 弯曲角下限 (mrad), 允许微小负值 (噪声)
MAX_BA_JUMP = 10.0         # 相邻点弯曲角最大跳变 (mrad)
SMOOTH_WINDOW = 51         # Savitzky-Golay 平滑窗口 (50 Hz ≈ 1s)
SMOOTH_ORDER = 3           # 多项式阶数


def _load_nc(path):
    """打开 netCDF 文件"""
    try:
        import netCDF4 as nc
        return nc.Dataset(path, 'r')
    except Exception:
        return None


def _read_var(ds, name):
    """读取 netCDF 变量为 float64 数组"""
    if name not in ds.variables:
        return None
    data = np.array(ds.variables[name][:], dtype=np.float64)
    return data


def _is_aeg_file(filepath):
    """判断是否为 AEG (大气层掩星) 文件"""
    basename = os.path.basename(filepath)
    return 'AEG' in basename and basename.endswith('.NC')


def compute_bending_angle_profile(ds):
    """
    从单个 FY-3D AEG 文件推导弯曲角剖面。

    算法:
      1. 读取 exLC (电离层校正后附加相位) 和卫星轨道数据
      2. 平滑 exLC 并求导得到 excess Doppler
      3. 由轨道几何计算直线 Doppler 和切线高度
      4. 利用几何光学公式从 excess Doppler 推导弯曲角

    Args:
        ds: netCDF4.Dataset 对象

    Returns:
        (heights_km, alpha_mrad): 切线高度数组和弯曲角数组 (mrad)
        或 None (QC 失败)
    """
    # --- 读取变量 ---
    time = _read_var(ds, 'time')
    exLC = _read_var(ds, 'exLC')

    if time is None or exLC is None:
        return None

    # 卫星位置 (km) 和速度 (km/s) — ECI 坐标系
    rL = np.array([_read_var(ds, v) for v in ['xLeo', 'yLeo', 'zLeo']])
    vL = np.array([_read_var(ds, v) for v in ['xdLeo', 'ydLeo', 'zdLeo']])
    rG = np.array([_read_var(ds, v) for v in ['xGnss', 'yGnss', 'zGnss']])
    vG = np.array([_read_var(ds, v) for v in ['xdGnss', 'ydGnss', 'zdGnss']])

    if any(v is None for v in [rL, vL, rG, vG]):
        return None
    if any(v.shape[0] != 3 for v in [rL, vL, rG, vG]):
        return None

    # --- QC: 检查 exLC 有效数据 ---
    valid_mask = exLC > FILL_VALUE
    n_valid = valid_mask.sum()
    n_total = len(exLC)

    if n_valid < MIN_PROFILE_POINTS:
        return None
    if n_valid / n_total < MIN_VALID_EX_FRAC:
        return None

    # 确定有效数据范围
    idx_valid = np.where(valid_mask)[0]
    i_start = idx_valid[0]
    i_end = idx_valid[-1]

    if i_end - i_start < MIN_PROFILE_POINTS:
        return None

    # --- 在有效范围内插值 exLC ---
    idx_sub = np.arange(i_start, i_end + 1)
    exLC_interp = np.interp(idx_sub, idx_valid, exLC[idx_valid])

    # 采样间隔
    dt = np.median(np.diff(time[i_start:i_end + 1]))
    if dt <= 0 or not np.isfinite(dt):
        return None

    # --- Savitzky-Golay 平滑 + 求导 → excess Doppler ---
    n_sub = len(exLC_interp)
    window = min(SMOOTH_WINDOW, n_sub // 2 * 2 - 1)
    if window < 5:
        window = 5
    if window % 2 == 0:
        window -= 1

    # dexLC/dt (m/s)
    doppler_excess_ms = savgol_filter(
        exLC_interp, window, SMOOTH_ORDER, deriv=1, delta=dt
    )
    # 转为 Hz
    D_excess_hz = doppler_excess_ms * F_GPS_L1 / C_LIGHT

    # --- 轨道几何 (有效范围) ---
    rL_sub = rL[:, i_start:i_end + 1]
    vL_sub = vL[:, i_start:i_end + 1]
    rG_sub = rG[:, i_start:i_end + 1]
    vG_sub = vG[:, i_start:i_end + 1]

    # 星地距离和单位矢量
    r_vec = rG_sub - rL_sub
    r_dist = np.sqrt(np.sum(r_vec ** 2, axis=0))
    r_hat = r_vec / r_dist

    # 卫星到地心距离
    rL_mag = np.sqrt(np.sum(rL_sub ** 2, axis=0))
    rG_mag = np.sqrt(np.sum(rG_sub ** 2, axis=0))

    # LEO-GNSS 夹角 (地心角)
    cos_gamma = np.clip(
        np.sum(rL_sub * rG_sub, axis=0) / (rL_mag * rG_mag), -1, 1
    )
    gamma = np.arccos(cos_gamma)

    # 直线切线高度 (近似)
    a_sl = rL_mag * rG_mag * np.sin(gamma) / r_dist
    h_tp = a_sl - R_EARTH

    # --- 弯曲角推导 ---
    # LEO/GNSS 速度在垂直于连线方向的分量 (km/s)
    cross_L = np.cross(vL_sub.T, r_hat.T)  # (N, 3)
    v_perp_L = np.sqrt(np.sum(cross_L ** 2, axis=1)) * 1e3  # → m/s

    cross_G = np.cross(vG_sub.T, r_hat.T)
    v_perp_G = np.sqrt(np.sum(cross_G ** 2, axis=1)) * 1e3

    # 几何光学小角近似:
    #   D_excess ≈ (f/c) · α · (v_perp_L + v_perp_G · a/rG)
    # 解出 α:
    denominator = v_perp_L + v_perp_G * (a_sl / rG_mag)

    # 避免除零
    safe_mask = np.abs(denominator) > 1.0  # m/s
    alpha_rad = np.full(len(denominator), np.nan)
    alpha_rad[safe_mask] = (
        D_excess_hz[safe_mask] * C_LIGHT / F_GPS_L1
    ) / denominator[safe_mask]

    alpha_mrad = alpha_rad * 1e3  # → milliradians

    # --- QC: 弯曲角范围检查 ---
    valid_ba = (
        np.isfinite(alpha_mrad)
        & np.isfinite(h_tp)
        & (alpha_mrad > BA_MIN_MRAD)
        & (alpha_mrad < BA_MAX_MRAD)
        & (h_tp > -5)
        & (h_tp < 80)
    )

    h_valid = h_tp[valid_ba]
    alpha_valid = alpha_mrad[valid_ba]

    if len(alpha_valid) < MIN_PROFILE_POINTS:
        return None

    # 按高度排序 (升序)
    sort_idx = np.argsort(h_valid)
    h_valid = h_valid[sort_idx]
    alpha_valid = alpha_valid[sort_idx]

    # 去除跳变异常点
    if len(alpha_valid) > 2:
        diff = np.abs(np.diff(alpha_valid))
        good = np.concatenate([[True], diff < MAX_BA_JUMP])
        h_valid = h_valid[good]
        alpha_valid = alpha_valid[good]

    if len(alpha_valid) < MIN_PROFILE_POINTS:
        return None

    return h_valid, alpha_valid


def process_single_fy3d(nc_path, std_height):
    """
    处理单个 FY-3D AEG 文件, 提取弯曲角剖面并插值到标准高度网格。

    Args:
        nc_path: AEG NC 文件路径
        std_height: 标准高度网格 (301,)

    Returns:
        x_vec (301,) 或 None
        x_vec 为 log10(|BA| + 1e-6) 变换后的弯曲角剖面
    """
    ds = _load_nc(nc_path)
    if ds is None:
        return None

    try:
        # QC: 检查全局属性
        if hasattr(ds, 'bad') and int(ds.bad) != 0:
            return None

        result = compute_bending_angle_profile(ds)
        if result is None:
            return None

        h_km, alpha_mrad = result

        # 检查穿透高度
        if h_km.min() > 5.0:
            return None

        # 检查高度覆盖范围
        if h_km.max() < 30.0:
            return None

        # --- 插值到标准高度网格 ---
        # 弯曲角转弧度后取绝对值
        alpha_rad = np.abs(alpha_mrad * 1e-3)

        f_ba = interp1d(
            h_km, alpha_rad, kind='linear',
            bounds_error=False,
            fill_value=(alpha_rad[0], alpha_rad[-1])
        )
        ba_interp = f_ba(std_height)

        # log10 变换 (与 COSMIC 一致)
        x_vec = np.log10(ba_interp + 1e-6)

        # 最终有效性检查
        if np.any(np.isnan(x_vec)) or np.any(np.isinf(x_vec)):
            return None

        return x_vec.astype(np.float32)

    except Exception:
        return None
    finally:
        if hasattr(ds, 'close'):
            ds.close()


def explore_fy3d_directory(data_dir, n_samples=3):
    """
    探索 FY-3D 数据目录, 输出文件统计和样本分析。

    Args:
        data_dir: 数据目录路径
        n_samples: 详细打印的样本文件数
    """
    print("=" * 60)
    print("  FY-3D GNOS 数据探索")
    print("=" * 60)

    # 查找所有 NC 文件
    nc_files = glob.glob(os.path.join(data_dir, '**', '*.NC'), recursive=True)
    print(f"\n总文件数: {len(nc_files)}")

    if len(nc_files) == 0:
        print(f"错误: 目录 {data_dir} 中未找到 .NC 文件!")
        return

    # 分类统计
    aeg_files = [f for f in nc_files if 'AEG' in os.path.basename(f)]
    ieg_files = [f for f in nc_files if 'IEG' in os.path.basename(f)]
    other = len(nc_files) - len(aeg_files) - len(ieg_files)

    print(f"  AEG (大气层, 可用): {len(aeg_files)}")
    print(f"  IEG (电离层, 跳过): {len(ieg_files)}")
    if other > 0:
        print(f"  其他: {other}")

    # 详细样本分析
    std_height = np.linspace(HEIGHT_MIN, HEIGHT_MAX, STD_HEIGHT)

    print(f"\n--- 样本文件分析 (前 {n_samples} 个 AEG 文件) ---")
    n_success = 0
    for fpath in sorted(aeg_files)[:n_samples]:
        fname = os.path.basename(fpath)
        print(f"\n{'='*50}")
        print(f"文件: {fname}")

        ds = _load_nc(fpath)
        if ds is None:
            print("  打开失败!")
            continue

        # 全局属性
        print(f"  全局属性:")
        for attr in ['satName', 'gnssName', 'setting', 'duration',
                      'exL1qc', 'exL2qc', 'bad', 'occsatId']:
            if hasattr(ds, attr):
                print(f"    {attr} = {getattr(ds, attr)}")

        # 变量概览
        print(f"  变量数: {len(ds.variables)}")
        exLC = _read_var(ds, 'exLC')
        if exLC is not None:
            valid = exLC > FILL_VALUE
            print(f"  exLC: {valid.sum()}/{len(exLC)} 有效 "
                  f"({100*valid.sum()/len(exLC):.1f}%)")

        # 尝试推导弯曲角
        result = compute_bending_angle_profile(ds)
        if result is not None:
            h_km, alpha_mrad = result
            print(f"  弯曲角推导成功!")
            print(f"    高度范围: {h_km.min():.1f} ~ {h_km.max():.1f} km")
            print(f"    弯曲角范围: {alpha_mrad.min():.3f} ~ "
                  f"{alpha_mrad.max():.3f} mrad")
            print(f"    有效点数: {len(h_km)}")

            # 尝试完整处理
            x_vec = process_single_fy3d(fpath, std_height)
            if x_vec is not None:
                print(f"    标准化后 x 范围: [{x_vec.min():.3f}, {x_vec.max():.3f}]")
                n_success += 1
        else:
            print(f"  弯曲角推导失败")

        ds.close()

    print(f"\n--- 探索总结 ---")
    print(f"可用 AEG 文件数: {len(aeg_files)}")
    print(f"样本测试成功: {n_success}/{min(n_samples, len(aeg_files))}")


def run_fy3d_pipeline(
    data_root,
    output_dir=None,
    strict_qc=True,
    do_split=True,
    use_refractivity=False,
    max_files=None,
):
    """
    FY-3D GNOS 数据预处理流水线。

    从 AEG L1 文件中推导弯曲角剖面, 执行 QC, 插值到标准高度网格,
    Z-Score 标准化, 并划分 train/val/test。

    注意: FY-3D L1 数据不含温度/气压/湿度标签,
         Y 数组将全部填零 (需后续通过 ERA5 匹配补充)。

    Args:
        data_root: FY-3D NC 文件目录
        output_dir: 输出目录
        strict_qc: 是否启用严格 QC
        do_split: 是否划分 train/val/test
        use_refractivity: 保留接口, 当前未使用
        max_files: 最大处理文件数 (调试用)

    Returns:
        (X, Y, report) 或 None
    """
    if output_dir is None:
        from ro_retrieval.config import PROCESSED_DIR
        output_dir = os.path.join(PROCESSED_DIR, 'fy3d')

    os.makedirs(output_dir, exist_ok=True)

    # 标准高度网格
    std_height = np.linspace(HEIGHT_MIN, HEIGHT_MAX, STD_HEIGHT)

    # --- 查找 AEG 文件 ---
    all_nc = glob.glob(os.path.join(data_root, '**', '*.NC'), recursive=True)
    aeg_files = sorted([f for f in all_nc if _is_aeg_file(f)])

    print(f"找到 {len(all_nc)} 个 NC 文件, 其中 {len(aeg_files)} 个 AEG 文件")

    if len(aeg_files) == 0:
        print("错误: 未找到 AEG 文件!")
        return None

    if max_files is not None:
        aeg_files = aeg_files[:max_files]
        print(f"调试模式: 仅处理前 {max_files} 个文件")

    # --- 逐文件处理 ---
    X_list = []
    n_fail = 0
    fail_reasons = {
        'open_error': 0, 'bad_flag': 0,
        'insufficient_data': 0, 'ba_derivation': 0,
        'interpolation': 0,
    }

    for fpath in tqdm(aeg_files, desc="处理 FY-3D AEG 文件"):
        x_vec = process_single_fy3d(fpath, std_height)
        if x_vec is not None:
            X_list.append(x_vec)
        else:
            n_fail += 1

    if len(X_list) == 0:
        print("错误: 所有文件处理失败!")
        return None

    X = np.array(X_list)  # (N, 301)

    # Y 填零 (FY-3D L1 无标签数据)
    Y = np.zeros((len(X), 3, STD_HEIGHT), dtype=np.float32)

    print(f"\n处理完成: 成功 {len(X)} / 总 {len(aeg_files)} "
          f"(失败 {n_fail}, 成功率 {len(X)/len(aeg_files)*100:.1f}%)")
    print(f"  X shape: {X.shape}")
    print(f"  Y shape: {Y.shape} (全零占位, 需 ERA5 标签)")

    # --- Z-Score 标准化 ---
    print("\n正在 Z-Score 标准化...")

    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0) + 1e-8
    X_norm = (X - x_mean) / x_std

    # Y 全零不需要标准化, 但保持接口一致
    y_means = [0.0, 0.0, 0.0]
    y_stds = [1.0, 1.0, 1.0]
    Y_norm = Y.copy()

    print(f"  X (弯曲角): mean={X.mean():.4f}, std={X.std():.4f}")

    # 保存标准化参数
    norm_params = {
        'x_mean': x_mean, 'x_std': x_std,
        'y_means': np.array(y_means), 'y_stds': np.array(y_stds),
        'data_source': 'fy3d',
        'label_status': 'placeholder_zeros',
    }
    np.savez(os.path.join(output_dir, 'norm_params.npz'), **norm_params)

    report = {
        'data_source': 'FY-3D GNOS L1',
        'total_files': len(aeg_files),
        'aeg_files': len(aeg_files),
        'success': len(X),
        'failed': n_fail,
        'qc_pass_rate': f"{len(X)/len(aeg_files)*100:.1f}%",
        'x_shape': list(X_norm.shape),
        'y_shape': list(Y_norm.shape),
        'label_note': 'Y 为全零占位, 需通过 ERA5 再分析数据匹配补充标签',
    }

    # 保存报告
    import json
    with open(os.path.join(output_dir, 'processing_report.json'), 'w') as fp:
        json.dump(report, fp, indent=2, ensure_ascii=False)

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

        # 同时保存未标准化的 X (便于后续与 ERA5 匹配)
        np.save(os.path.join(output_dir, 'X_raw.npy'), X)

        print(f"\n数据划分完成:")
        print(f"  训练集: {len(train_idx)}")
        print(f"  验证集: {len(val_idx)}")
        print(f"  测试集: {len(test_idx)}")
        print(f"\n数据已保存至: {output_dir}")
    else:
        np.save(os.path.join(output_dir, 'train_x.npy'), X_norm)
        np.save(os.path.join(output_dir, 'train_y.npy'), Y_norm)
        np.save(os.path.join(output_dir, 'X_raw.npy'), X)

    return X_norm, Y_norm, report
