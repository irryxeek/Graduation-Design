import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from scipy.signal import savgol_filter
# ✅ 修正：只导入模型类、采样函数和配置常量，不导入统计量
from inference_ddim import ConditionalUNet1D, ddim_sample, DATA_DIR, MODEL_PATH, DEVICE

# =================配置=================
SAVE_DIR = "evaluation_results_ddim"
os.makedirs(SAVE_DIR, exist_ok=True)
TEST_SAMPLES = 50 

# =================主程序=================
if __name__ == "__main__":
    print(f"Loading Model for DDIM Batch Eval...")
    
    # 1. 加载数据
    # 确保路径存在
    x_path = os.path.join(DATA_DIR, "train_x.npy")
    y_path = os.path.join(DATA_DIR, "train_y.npy")
    if not os.path.exists(x_path):
        print(f"❌ 错误：找不到数据文件 {x_path}")
        exit()

    raw_x = np.load(x_path).astype(np.float32)
    raw_y = np.load(y_path).astype(np.float32)
    
    # ✅ 修正：在这里重新计算统计量 (Mean/Std)
    print("正在计算数据统计量...")
    y_mean = torch.tensor(np.mean(raw_y, axis=0)).to(DEVICE)
    y_std = torch.tensor(np.std(raw_y, axis=0) + 1e-6).to(DEVICE)
    x_mean = np.mean(raw_x, axis=0)
    x_std = np.std(raw_x, axis=0) + 1e-6
    
    # 2. 加载模型
    model = ConditionalUNet1D().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    else:
        print(f"❌ 找不到模型权重: {MODEL_PATH}")
        exit()
    model.eval()

    # 3. 随机抽取样本
    # 固定随机种子，保证每次跑的结果一样，方便写报告
    np.random.seed(42) 
    indices = np.random.choice(len(raw_x), TEST_SAMPLES, replace=False)
    
    results = [] 

    print(f"🚀 Start DDIM Batch Evaluation ({TEST_SAMPLES} samples)...")
    for idx in tqdm(indices):
        input_ba = raw_x[idx]
        true_temp = raw_y[idx]
        
        # 标准化
        input_norm = (input_ba - x_mean) / x_std
        cond_tensor = torch.tensor(input_norm).unsqueeze(0).unsqueeze(0).to(DEVICE)
        
        # DDIM 推理 (50步)
        # 注意：这里调用的是 inference_ddim 里定义的 ddim_sample
        gen = ddim_sample(model, cond_tensor, shape=(1, 1, 301))
        
        # 反归一化
        pred_temp = gen.squeeze().cpu() * y_std.cpu() + y_mean.cpu()
        
        # 平滑
        try:
            pred_smooth = savgol_filter(pred_temp.numpy(), window_length=31, polyorder=3)
        except:
            pred_smooth = pred_temp.numpy()
            
        # 计算 RMSE
        rmse = np.sqrt(np.mean((pred_smooth - true_temp)**2))
        
        results.append({
            "rmse": rmse,
            "idx": idx,
            "true": true_temp,
            "pred": pred_smooth,
            "input": input_ba
        })

    # 4. 统计
    rmses = [r['rmse'] for r in results]
    avg_rmse = np.mean(rmses)
    min_rmse = np.min(rmses)
    max_rmse = np.max(rmses)
    
    print(f"\n======== DDIM 评估报告 ========")
    print(f"平均 RMSE: {avg_rmse:.4f} K")
    print(f"最好 RMSE: {min_rmse:.4f} K")
    print(f"最差 RMSE: {max_rmse:.4f} K")
    
    # 5. 保存对比图
    results.sort(key=lambda x: x['rmse'])
    cases = [("Best", results[0]), ("Median", results[len(results)//2]), ("Worst", results[-1])]
    
    heights = np.linspace(0, 60, 301)
    for label, data in cases:
        plt.figure(figsize=(10, 6))
        plt.plot(data['true'], heights, 'k-', label='Truth', linewidth=2)
        plt.plot(data['pred'], heights, 'r--', label=f'DDIM (RMSE={data["rmse"]:.2f})', linewidth=2)
        plt.title(f"DDIM {label} Case (Sample {data['idx']})")
        plt.xlabel("Temperature (K)")
        plt.ylabel("Height (km)")
        plt.legend()
        plt.grid(True)
        save_path = os.path.join(SAVE_DIR, f"DDIM_{label}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"已保存: {save_path}")