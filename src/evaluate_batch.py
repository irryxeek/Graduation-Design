import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from scipy.signal import savgol_filter
from inference import ConditionalUNet1D, p_sample, betas, alphas, alphas_cumprod, posterior_variance, TIMESTEPS

# =================配置=================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = r"D:\02_Study\01_Schoolwork\Graduation Design\Data\Processed"
MODEL_PATH = "ro_diffusion_epoch_100.pth"
SAVE_DIR = "evaluation_results"
os.makedirs(SAVE_DIR, exist_ok=True)
TEST_SAMPLES = 50  # 测试多少个样本

# =================采样函数 (保持不变)=================
@torch.no_grad()
def sample_one(model, condition):
    b = 1
    img = torch.randn((1, 1, 301), device=DEVICE)
    for i in reversed(range(0, TIMESTEPS)):
        t = torch.full((b, 1), i, device=DEVICE, dtype=torch.long)
        img = p_sample(model, img, t, i, condition)
    return img

# =================主程序=================
if __name__ == "__main__":
    print(f"Loading Data & Model on {DEVICE}...")
    
    # 1. 加载数据
    raw_x = np.load(os.path.join(DATA_DIR, "train_x.npy")).astype(np.float32)
    raw_y = np.load(os.path.join(DATA_DIR, "train_y.npy")).astype(np.float32)
    
    y_mean = torch.tensor(np.mean(raw_y, axis=0)).to(DEVICE)
    y_std = torch.tensor(np.std(raw_y, axis=0) + 1e-6).to(DEVICE)
    x_mean = np.mean(raw_x, axis=0)
    x_std = np.std(raw_x, axis=0) + 1e-6
    
    # 2. 加载模型
    model = ConditionalUNet1D().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 3. 随机抽取样本进行测试
    indices = np.random.choice(len(raw_x), TEST_SAMPLES, replace=False)
    
    results = [] # 存储 (rmse, index, true_temp, pred_temp_smooth)

    print(f"Start Batch Evaluation ({TEST_SAMPLES} samples)...")
    for idx in tqdm(indices):
        # 准备数据
        input_ba = raw_x[idx]
        true_temp = raw_y[idx]
        
        # 标准化
        input_norm = (input_ba - x_mean) / x_std
        cond_tensor = torch.tensor(input_norm).unsqueeze(0).unsqueeze(0).to(DEVICE)
        
        # 推理
        gen = sample_one(model, cond_tensor)
        
        # 反归一化
        pred_temp = gen.squeeze().cpu() * y_std.cpu() + y_mean.cpu()
        
        # 平滑
        try:
            pred_smooth = savgol_filter(pred_temp.numpy(), window_length=31, polyorder=3)
        except:
            pred_smooth = pred_temp.numpy() # Fallback
            
        # 计算 RMSE
        rmse = np.sqrt(np.mean((pred_smooth - true_temp)**2))
        
        results.append({
            "rmse": rmse,
            "idx": idx,
            "true": true_temp,
            "pred": pred_smooth,
            "input": input_ba
        })

    # 4. 统计分析
    rmses = [r['rmse'] for r in results]
    avg_rmse = np.mean(rmses)
    min_rmse = np.min(rmses)
    max_rmse = np.max(rmses)
    
    print(f"\n======== 评估报告 ========")
    print(f"测试样本数: {TEST_SAMPLES}")
    print(f"平均 RMSE: {avg_rmse:.4f} K")
    print(f"最好 RMSE: {min_rmse:.4f} K")
    print(f"最差 RMSE: {max_rmse:.4f} K")
    
    # 5. 排序并画图
    results.sort(key=lambda x: x['rmse'])
    
    # 挑选 3 个代表性样本
    cases = [
        ("Best", results[0]),                # 最好
        ("Median", results[len(results)//2]), # 中位数
        ("Worst", results[-1])               # 最差
    ]
    
    heights = np.linspace(0, 60, 301)
    
    for label, data in cases:
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        
        # Input
        ax[0].plot(data['input'], heights, 'b-')
        ax[0].set_title(f"{label} Case (Sample {data['idx']})\nInput Bending Angle")
        ax[0].set_ylabel("Height (km)")
        ax[0].grid(True)
        
        # Output
        ax[1].plot(data['true'], heights, 'k-', label='ERA5 Truth', linewidth=2)
        ax[1].plot(data['pred'], heights, 'r--', label=f'AI Pred (RMSE={data["rmse"]:.2f}K)', linewidth=2)
        ax[1].set_title(f"Retrieval Result ({label})")
        ax[1].legend()
        ax[1].grid(True)
        
        save_path = os.path.join(SAVE_DIR, f"{label}_RMSE_{data['rmse']:.2f}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"已保存图像: {save_path}")

    print(f"\n✅ 批量评估完成！请去 {SAVE_DIR} 文件夹查看图片。")
    print(f"建议：把 'Median'（中等）的图放入中期报告的【结果展示】部分。")
    print(f"建议：把 'Worst'（最差）的图放入中期报告的【困难与挑战】部分。")