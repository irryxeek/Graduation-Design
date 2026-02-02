import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# =================配置区域=================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-4
DATA_DIR = r"D:\02_Study\01_Schoolwork\Graduation Design\Data\Processed"

# 扩散模型参数
TIMESTEPS = 1000  # 扩散步数
# 简单的线性 Beta Schedule
beta_start = 1e-4
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, TIMESTEPS).to(DEVICE)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# =================1. 数据集定义=================
class RODataset(Dataset):
    def __init__(self, x_path, y_path):
        # 读取数据 (N, 301)
        self.x = np.load(x_path).astype(np.float32)
        self.y = np.load(y_path).astype(np.float32)
        
        # 简单的数据标准化 (Z-Score Normalization)
        # 这一步对扩散模型收敛至关重要！
        self.x_mean = np.mean(self.x, axis=0)
        self.x_std = np.std(self.x, axis=0) + 1e-6
        self.y_mean = np.mean(self.y, axis=0)
        self.y_std = np.std(self.y, axis=0) + 1e-6
        
        self.x_norm = (self.x - self.x_mean) / self.x_std
        self.y_norm = (self.y - self.y_mean) / self.y_std
        
        print(f"Dataset Loaded. Shape: {self.x.shape}")
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # 增加 Channel 维度 (Length) -> (1, Length)
        cond = torch.tensor(self.x_norm[idx]).unsqueeze(0) 
        target = torch.tensor(self.y_norm[idx]).unsqueeze(0)
        return cond, target

# =================2. 模型定义 (Conditional U-Net)=================
class ConditionalUNet1D(nn.Module):
    def __init__(self, in_channels=1, cond_channels=1, out_channels=1):
        super().__init__()
        
        # 简单的 Time Embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 32),
            nn.SiLU(),
            nn.Linear(32, 32)
        )
        
        # Downsample
        self.down1 = nn.Conv1d(in_channels + cond_channels, 32, 3, padding=1)
        self.down2 = nn.Conv1d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool1d(2)
        
        # Bottleneck
        self.bot1 = nn.Conv1d(64, 128, 3, padding=1)
        
        # Upsample
        self.up2 = nn.ConvTranspose1d(128, 64, 2, stride=2)
        self.up1 = nn.ConvTranspose1d(64 + 64, 32, 2, stride=2) # Skip connection
        
        # Output
        self.out = nn.Conv1d(32 + 32, out_channels, 3, padding=1)
        
        self.act = nn.SiLU()

    def forward(self, x, t, condition):
        # x: Noisy Target (Batch, 1, 301)
        # t: Timestep (Batch, 1)
        # condition: Bending Angle (Batch, 1, 301)
        
        # 1. 融合条件信息 (简单拼接)
        # 将条件直接拼接到输入上，作为强引导
        x_in = torch.cat([x, condition], dim=1) 
        
        # 2. Time Embedding
        t_emb = self.time_mlp(t.float()).unsqueeze(-1) # (Batch, 32, 1)
        
        # --- Encoder ---
        d1 = self.act(self.down1(x_in)) # (B, 32, 301)
        # 融合时间信息
        d1 = d1 + t_emb 
        
        p1 = self.pool(d1) # (B, 32, 150)
        
        d2 = self.act(self.down2(p1)) # (B, 64, 150)
        p2 = self.pool(d2) # (B, 64, 75)
        
        # --- Bottleneck ---
        b = self.act(self.bot1(p2)) # (B, 128, 75)
        
        # --- Decoder ---
        u2 = self.up2(b) # (B, 64, 150)
        # 处理池化导致的尺寸不匹配 (padding)
        if u2.shape[2] != d2.shape[2]:
            u2 = nn.functional.pad(u2, (0, 1))
            
        u2 = torch.cat([u2, d2], dim=1) # Skip Connection
        
        u1 = self.up1(u2) # (B, 32, 300)
        if u1.shape[2] != d1.shape[2]:
             u1 = nn.functional.pad(u1, (0, 1))
             
        u1 = torch.cat([u1, d1], dim=1)
        
        output = self.out(u1) # (B, 1, 301)
        return output

# =================3. 训练主循环=================
def main():
    # 1. 加载数据
    x_path = os.path.join(DATA_DIR, "train_x.npy")
    y_path = os.path.join(DATA_DIR, "train_y.npy")
    
    if not os.path.exists(x_path):
        print("错误：找不到数据文件！请检查 DATA_DIR")
        return

    dataset = RODataset(x_path, y_path)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 2. 初始化模型
    model = ConditionalUNet1D().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    
    print(f"Start Training on {DEVICE}...")
    
    # 3. 训练循环
    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        epoch_loss = 0
        
        for condition, x_0 in pbar:
            condition = condition.to(DEVICE) # 弯曲角
            x_0 = x_0.to(DEVICE)             # 真实大气廓线 (Target)
            
            batch_size = x_0.shape[0]
            
            # A. 随机采样时间步 t
            t = torch.randint(0, TIMESTEPS, (batch_size, 1), device=DEVICE).long()
            
            # B. 生成噪声 epsilon
            noise = torch.randn_like(x_0)
            
            # C. 前向加噪 (Forward Diffusion)
            # x_t = sqrt(alpha_bar) * x_0 + sqrt(1-alpha_bar) * epsilon
            sqrt_alpha_t = sqrt_alphas_cumprod[t].view(-1, 1, 1)
            sqrt_one_minus_alpha_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
            
            x_t = sqrt_alpha_t * x_0 + sqrt_one_minus_alpha_t * noise
            
            # D. 模型预测噪声 (Predict Noise)
            # 输入: x_t, t, condition
            noise_pred = model(x_t, t, condition)
            
            # E. 计算损失 (Loss)
            loss = loss_fn(noise_pred, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'Loss': loss.item()})
            
        print(f"Epoch {epoch+1} Mean Loss: {epoch_loss / len(dataloader):.6f}")
        
        # 每10轮保存一次模型
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"ro_diffusion_epoch_{epoch+1}.pth")
            print(f"Model saved: ro_diffusion_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    main()