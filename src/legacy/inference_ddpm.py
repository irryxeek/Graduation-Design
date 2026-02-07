import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

# =================配置区域=================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = r"D:\02_Study\01_Schoolwork\Graduation Design\Data\Processed"
MODEL_PATH = "ro_diffusion_epoch_100.pth" # 加载你刚刚训练好的权重

# 必须与训练时完全一致的参数
TIMESTEPS = 1000
betas = torch.linspace(1e-4, 0.02, TIMESTEPS).to(DEVICE)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
# 预计算一些反向过程需要的系数
alphas_cumprod_prev = torch.cat([torch.tensor([1.]).to(DEVICE), alphas_cumprod[:-1]])
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

# =================1. 模型定义 (必须复制 train.py 中的定义)=================
# 为了方便，这里重新定义一遍，保证结构一致
class ConditionalUNet1D(nn.Module):
    def __init__(self, in_channels=1, cond_channels=1, out_channels=1):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.Linear(1, 32), nn.SiLU(), nn.Linear(32, 32))
        self.down1 = nn.Conv1d(in_channels + cond_channels, 32, 3, padding=1)
        self.down2 = nn.Conv1d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.bot1 = nn.Conv1d(64, 128, 3, padding=1)
        self.up2 = nn.ConvTranspose1d(128, 64, 2, stride=2)
        self.up1 = nn.ConvTranspose1d(64 + 64, 32, 2, stride=2)
        self.out = nn.Conv1d(32 + 32, out_channels, 3, padding=1)
        self.act = nn.SiLU()

    def forward(self, x, t, condition):
        x_in = torch.cat([x, condition], dim=1)
        t_emb = self.time_mlp(t.float()).unsqueeze(-1)
        d1 = self.act(self.down1(x_in))
        d1 = d1 + t_emb
        p1 = self.pool(d1)
        d2 = self.act(self.down2(p1))
        p2 = self.pool(d2)
        b = self.act(self.bot1(p2))
        u2 = self.up2(b)
        if u2.shape[2] != d2.shape[2]: u2 = nn.functional.pad(u2, (0, 1))
        u2 = torch.cat([u2, d2], dim=1)
        u1 = self.up1(u2)
        if u1.shape[2] != d1.shape[2]: u1 = nn.functional.pad(u1, (0, 1))
        u1 = torch.cat([u1, d1], dim=1)
        output = self.out(u1)
        return output

# =================2. 采样核心函数 (Reverse Diffusion)=================
@torch.no_grad()
def p_sample(model, x, t, t_index, condition):
    # 预测噪声
    betas_t = betas[t_index]
    sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1. - alphas_cumprod[t_index])
    sqrt_recip_alphas_t = torch.sqrt(1. / alphas[t_index])
    
    # model_mean = 1/sqrt(alpha) * (x - beta/sqrt(1-alpha_bar) * noise_pred)
    noise_pred = model(x, t, condition)
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * noise_pred / sqrt_one_minus_alphas_cumprod_t
    )
    
    if t_index == 0:
        return model_mean
    else:
        # 添加方差 (Langevin Dynamics)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance[t_index]) * noise

@torch.no_grad()
def sample(model, condition, shape):
    b = shape[0]
    # 从纯高斯噪声开始
    img = torch.randn(shape, device=DEVICE)
    
    # 逐步去噪: T -> T-1 -> ... -> 0
    for i in reversed(range(0, TIMESTEPS)):
        t = torch.full((b, 1), i, device=DEVICE, dtype=torch.long)
        img = p_sample(model, img, t, i, condition)
        
    return img

# =================3. 主程序=================
if __name__ == "__main__":
    # A. 准备数据统计量 (用于反归一化)
    try:
        raw_x = np.load(os.path.join(DATA_DIR, "train_x.npy")).astype(np.float32)
        raw_y = np.load(os.path.join(DATA_DIR, "train_y.npy")).astype(np.float32)
    except FileNotFoundError:
        print("❌ 找不到 train_x.npy 或 train_y.npy，请检查 DATA_DIR")
        exit()

    # 重新计算 Mean/Std (必须和训练时一样)
    y_mean = torch.tensor(np.mean(raw_y, axis=0)).to(DEVICE)
    y_std = torch.tensor(np.std(raw_y, axis=0) + 1e-6).to(DEVICE)
    x_mean = np.mean(raw_x, axis=0)
    x_std = np.std(raw_x, axis=0) + 1e-6
    
    # B. 加载模型
    model = ConditionalUNet1D().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"成功加载模型权重: {MODEL_PATH}")
    else:
        print(f"❌ 找不到模型权重: {MODEL_PATH}")
        exit()
        
    model.eval()

    # C. 随机挑选一个样本进行测试
    sample_idx = 748 # np.random.randint(0, len(raw_x))
    print(f"正在测试样本 ID: {sample_idx}")
    
    # 准备输入条件 (标准化)
    input_ba = raw_x[sample_idx]
    input_ba_norm = (input_ba - x_mean) / x_std
    cond_tensor = torch.tensor(input_ba_norm).unsqueeze(0).unsqueeze(0).to(DEVICE) # (1, 1, 301)
    
    # D. 执行生成 (Sampling)
    generated_norm = sample(model, cond_tensor, shape=(1, 1, 301))
    
    # E. 反归一化 (还原为真实温度 K)
    pred_temp = generated_norm.squeeze().cpu() * y_std.cpu() + y_mean.cpu()
    true_temp = raw_y[sample_idx]
    
    # F. 计算误差
    rmse = np.sqrt(np.mean((pred_temp.numpy() - true_temp)**2))
    print(f"预测完成, RMSE: {rmse:.2f} K")

# ... (前面的代码保持不变) ...
    
    # ==========================================
    # 新增: 后处理平滑 (Savitzky-Golay Filter)
    # ==========================================
    from scipy.signal import savgol_filter
    
    # window_length: 窗口长度 (必须是奇数)，越大越平滑，但可能丢失细节
    # polyorder: 多项式阶数，通常选 2 或 3
    pred_temp_smooth = savgol_filter(pred_temp.numpy(), window_length=31, polyorder=3)
    
    # 重新计算平滑后的 RMSE
    rmse_smooth = np.sqrt(np.mean((pred_temp_smooth - true_temp)**2))
    print(f"平滑后 RMSE: {rmse_smooth:.2f} K (原值: {rmse:.2f} K)")

    # ==========================================
    # G. 画图对比 (升级版)
    # ==========================================
    heights = np.linspace(0, 60, 301)
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图: 输入弯曲角
    ax[0].plot(input_ba, heights, 'b-', linewidth=1.5)
    ax[0].set_title(f"Input: Bending Angle (Log10)\nSample {sample_idx}")
    ax[0].set_ylabel("Height (km)")
    ax[0].set_xlabel("Log10(BA)")
    ax[0].grid(True, linestyle='--', alpha=0.5)
    
    # 右图: 温度对比
    # 1. 画真值 (黑线)
    ax[1].plot(true_temp, heights, 'k-', label='ERA5 (Truth)', linewidth=2.5, alpha=0.8)
    
    # 2. 画原始AI输出 (浅红色，作为背景对比)
    ax[1].plot(pred_temp, heights, color='red', linestyle='-', linewidth=0.5, alpha=0.3, label='AI Raw (Noisy)')
    
    # 3. 画平滑后的AI输出 (深红色虚线，作为最终结果)
    ax[1].plot(pred_temp_smooth, heights, color='red', linestyle='--', linewidth=2, label=f'AI Smoothed (RMSE={rmse_smooth:.2f}K)')
    
    ax[1].set_title(f"Retrieval Result Comparison")
    ax[1].set_xlabel("Temperature (K)")
    ax[1].legend(loc='upper right')
    ax[1].grid(True, linestyle='--', alpha=0.5)
    
    # 设置一下X轴范围，让图好看点
    ax[1].set_xlim(180, 320)
    
    plt.tight_layout()
    plt.show()