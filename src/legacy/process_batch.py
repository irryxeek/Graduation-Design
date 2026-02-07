import os
import glob
import numpy as np
import xarray as xr
from scipy.interpolate import interp1d
from tqdm import tqdm

# ==========================================
# 1. 路径配置
# ==========================================
# 请确认这两个路径是你解压后的根目录
ATM_ROOT = r"D:\02_Study\01_Schoolwork\Graduation Design\Data\Sample\atmPrf_nrt_2026_001"
WET_ROOT = r"D:\02_Study\01_Schoolwork\Graduation Design\Data\Sample\wetPf2_nrt_2026_001"

OUTPUT_DIR = r"D:\02_Study\01_Schoolwork\Graduation Design\Data\Processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

STD_HEIGHT = np.linspace(0, 60, 301)

# ==========================================
# 2. 核心处理函数
# ==========================================
def process_pair_v4(atm_path, wet_path):
    try:
        with xr.open_dataset(atm_path) as ds_atm, xr.open_dataset(wet_path) as ds_wet:
            # --- Input ---
            # 兼容变量名: Bend_ang 或 ba
            if 'Bend_ang' in ds_atm:
                ba = ds_atm['Bend_ang'].values
            elif 'ba' in ds_atm:
                ba = ds_atm['ba'].values
            else:
                return None
                
            h_atm = ds_atm['MSL_alt'].values
            
            mask_atm = ~np.isnan(ba) & ~np.isnan(h_atm)
            ba, h_atm = ba[mask_atm], h_atm[mask_atm]
            if len(ba) < 10: return None

            f_ba = interp1d(h_atm, ba, kind='linear', bounds_error=False, fill_value=0)
            x_vec = f_ba(STD_HEIGHT)
            x_vec = np.log10(np.abs(x_vec) + 1e-6)

            # --- Label ---
            if 'Temp' in ds_wet:
                temp = ds_wet['Temp'].values
            elif 'T' in ds_wet:
                temp = ds_wet['T'].values
            else:
                return None
            
            h_wet = ds_wet['MSL_alt'].values
            
            mask_wet = ~np.isnan(temp) & ~np.isnan(h_wet)
            temp, h_wet = temp[mask_wet], h_wet[mask_wet]
            if len(temp) < 10: return None

            f_temp = interp1d(h_wet, temp, kind='linear', bounds_error=False, fill_value=0)
            y_vec = f_temp(STD_HEIGHT)
            
            # 摄氏度转开尔文
            if np.min(y_vec[y_vec != 0]) < 0 and np.max(y_vec) < 100:
                y_vec = y_vec + 273.15

            return x_vec, y_vec
    except Exception:
        return None

# ==========================================
# 3. 主程序 (针对 _nc 后缀修正版)
# ==========================================
if __name__ == "__main__":
    print(f"扫描 ATM 目录: {ATM_ROOT}")
    
    # 🔍 关键修改：搜索 *_nc 而不仅仅是 *.nc
    # 为了保险，我们搜索所有包含 'nc' 的文件，然后再过滤
    atm_files = []
    # 搜索模式 1: 标准 .nc
    atm_files.extend(glob.glob(os.path.join(ATM_ROOT, "**", "*.nc"), recursive=True))
    # 搜索模式 2: CDAAC 特有的 _nc
    atm_files.extend(glob.glob(os.path.join(ATM_ROOT, "**", "*_nc"), recursive=True))
    
    print(f"扫描 WET 目录: {WET_ROOT}")
    wet_files = []
    wet_files.extend(glob.glob(os.path.join(WET_ROOT, "**", "*.nc"), recursive=True))
    wet_files.extend(glob.glob(os.path.join(WET_ROOT, "**", "*_nc"), recursive=True))

    print(f"✅ 扫描结果: 找到 {len(atm_files)} 个 ATM 文件, {len(wet_files)} 个 WET 文件")
    
    if len(atm_files) == 0:
        print("❌ 绝望了：依然没找到文件。请截图你的文件夹内容给我。")
        exit()

    # --- 构建索引 ---
    print("正在构建文件索引...")
    wet_map = {}
    for f in wet_files:
        fname = os.path.basename(f)
        wet_map[fname] = f
    
    data_x_list = []
    data_y_list = []
    success_count = 0
    
    print("开始配对处理...")
    for atm_f in tqdm(atm_files):
        atm_fname = os.path.basename(atm_f)
        
        # 智能匹配逻辑
        # 1. 尝试直接替换前缀 (atmPrf -> wetPf2)
        target_name_1 = atm_fname.replace('atmPrf', 'wetPf2')
        
        # 2. 尝试添加前缀 (如果原文件名没有atmPrf)
        target_name_2 = "wetPf2_" + atm_fname

        target_path = None
        
        if target_name_1 in wet_map:
            target_path = wet_map[target_name_1]
        elif target_name_2 in wet_map:
            target_path = wet_map[target_name_2]
        elif atm_fname in wet_map: # 完全同名
            target_path = wet_map[atm_fname]
            
        if target_path:
            res = process_pair_v4(atm_f, target_path)
            if res:
                data_x_list.append(res[0])
                data_y_list.append(res[1])
                success_count += 1
    
    if success_count > 0:
        X = np.array(data_x_list)
        Y = np.array(data_y_list)
        print(f"\n🎉 成功处理! 样本数: {len(X)}")
        print(f"保存路径: {OUTPUT_DIR}")
        
        np.save(os.path.join(OUTPUT_DIR, "train_x.npy"), X)
        np.save(os.path.join(OUTPUT_DIR, "train_y.npy"), Y)
    else:
        print("\n❌ 找到了文件，但配对数为 0。")
        print(f"ATM 示例: {os.path.basename(atm_files[0])}")
        if len(wet_files) > 0:
            print(f"WET 示例: {os.path.basename(wet_files[0])}")
        print("请检查 atmPrf 和 wetPf2 文件名除了前缀外，剩下的部分是否一致？")