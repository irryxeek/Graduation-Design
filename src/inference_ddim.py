import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter

# =================配置区域=================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = r"D:\02_Study\01_Schoolwork\Graduation Design\Data\Processed"
MODEL_PATH = "ro_diffusion_epoch_100.pth"

# ⚠️ 关键参数：DDIM 采样步数
# 训练时是 1000 步，这里我们只用 50 步！(速度提升20倍)
DDIM_STEPS = 50 
TOTAL_TIMESTEPS = 1000
ETA = 0.0  # eta=0 代表纯确定性采样 (DDIM)，eta=1 代表 DDPM

# 预计算系数 (必须与训练一致)
betas = torch.linspace(1e-4, 0.02, TOTAL_TIMESTEPS).to(DEVICE)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

# =================1. 模型定义 (保持不变)=================
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

# =================2. DDIM 核心采样逻辑=================
@torch.no_grad()
def ddim_sample(model, condition, shape):
    b = shape[0]
    device = condition.device
    
    # 1. 生成时间步序列 (例如: [0, 20, 40, ..., 980])
    # 我们只采样 50 个点，而不是 1000 个
    times = torch.linspace(0, TOTAL_TIMESTEPS - 1, steps=DDIM_STEPS + 1).long().to(device)
    # 反转时间: 980 -> ... -> 20 -> 0
    time_pairs = list(zip(reversed(times[:-1]), reversed(times[1:])))
    
    # 2. 初始噪声 x_T
    img = torch.randn(shape, device=device)
    
    print(f"🚀 开始 DDIM 加速采样 ({DDIM_STEPS} steps)...")
    
    for i, (t_curr, t_prev) in enumerate(time_pairs):
        # t_curr: 当前时间步 (例如 999)
        # t_prev: 下一步时间步 (例如 979)
        
        # 构造 batch 时间输入
        t_batch = torch.full((b, 1), t_curr.item(), device=device).long()
        
        # 获取对应的 alpha_cumprod
        alpha_t = alphas_cumprod[t_curr]
        alpha_t_prev = alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0).to(device)
        
        # A. 预测噪声 epsilon_theta
        noise_pred = model(img, t_batch, condition)
        
        # B. 预测 x0 (Denoised)
        # x0 = (xt - sqrt(1-alpha_t) * eps) / sqrt(alpha_t)
        pred_x0 = (img - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
        
        # C. 施加 DDIM 公式指向 x_{t-1}
        #sigma_t = ETA * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev))
        sigma_t = 0 # 强制确定性采样
        
        # 方向项 (指向 x_t)
        dir_xt = torch.sqrt(1 - alpha_t_prev - sigma_t**2) * noise_pred
        
        # 随机项 (DDIM 中通常为 0)
        noise = sigma_t * torch.randn_like(img)
        
        # 更新 img
        img = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt + noise
        
    return img

# =================3. 主程序=================
if __name__ == "__main__":
    # 加载数据统计量
    raw_x = np.load(os.path.join(DATA_DIR, "train_x.npy")).astype(np.float32)
    raw_y = np.load(os.path.join(DATA_DIR, "train_y.npy")).astype(np.float32)
    
    y_mean = torch.tensor(np.mean(raw_y, axis=0)).to(DEVICE)
    y_std = torch.tensor(np.std(raw_y, axis=0) + 1e-6).to(DEVICE)
    x_mean = np.mean(raw_x, axis=0)
    x_std = np.std(raw_x, axis=0) + 1e-6
    
    # 加载模型
    model = ConditionalUNet1D().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("✅ 模型加载成功")
    else:
        print("❌ 模型未找到")
        exit()
    model.eval()

    # 随机测试
    idx = np.random.randint(0, len(raw_x))
    # 或者指定之前那个“坏掉”的样本 ID 来测试修复效果
    # idx = 748 
    
    input_ba = raw_x[idx]
    true_temp = raw_y[idx]
    
    # 归一化输入
    input_norm = (input_ba - x_mean) / x_std
    cond_tensor = torch.tensor(input_norm).unsqueeze(0).unsqueeze(0).to(DEVICE)
    
    # === 执行 DDIM 采样 ===
    # 注意: 这里速度会比之前快很多
    gen = ddim_sample(model, cond_tensor, shape=(1, 1, 301))
    
    # 反归一化
    pred_temp = gen.squeeze().cpu() * y_std.cpu() + y_mean.cpu()
    
    # (可选) 依然可以保留平滑，虽然 DDIM 自身的噪声已经很小了
    pred_smooth = savgol_filter(pred_temp.numpy(), window_length=31, polyorder=3)
    
    # 计算 RMSE
    rmse = np.sqrt(np.mean((pred_smooth - true_temp)**2))
    print(f"DDIM 采样完成! RMSE: {rmse:.2f} K")
    
    # 画图
    heights = np.linspace(0, 60, 301)
    plt.figure(figsize=(8, 6))
    plt.plot(true_temp, heights, 'k-', label='ERA5 Truth', linewidth=2)
    plt.plot(pred_temp.numpy(), heights, 'r--', label='DDIM Raw (No Smoothing)', alpha=0.5)
    plt.plot(pred_smooth, heights, 'r-', label=f'DDIM Smoothed (RMSE={rmse:.2f}K)', linewidth=2)
    plt.title(f"DDIM Retrieval (Steps={DDIM_STEPS})")
    plt.xlabel("Temperature (K)")
    plt.ylabel("Height (km)")
    plt.legend()
    plt.grid(True)
    plt.show()