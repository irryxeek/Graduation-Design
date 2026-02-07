"""
条件 U-Net 模型
================
包含:
  1. ConditionalUNet1D       : 原始版本 (向后兼容已训练权重)
  2. EnhancedConditionalUNet1D : 增强版 (交叉注意力 + 多变量输出)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# =====================================================================
# 原始模型 (兼容已有 .pth 权重)
# =====================================================================
class ConditionalUNet1D(nn.Module):
    """原始条件 U-Net (保持与训练权重兼容)"""

    def __init__(self, in_channels=1, cond_channels=1, out_channels=1):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 32), nn.SiLU(), nn.Linear(32, 32)
        )
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
        if u2.shape[2] != d2.shape[2]:
            u2 = F.pad(u2, (0, 1))
        u2 = torch.cat([u2, d2], dim=1)
        u1 = self.up1(u2)
        if u1.shape[2] != d1.shape[2]:
            u1 = F.pad(u1, (0, 1))
        u1 = torch.cat([u1, d1], dim=1)
        return self.out(u1)


# =====================================================================
# 辅助模块
# =====================================================================
class SinusoidalTimeEmbedding(nn.Module):
    """正弦时间嵌入 (更强的时间步表示)"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.float().unsqueeze(-1) * emb.unsqueeze(0)  # (B, half_dim)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # (B, dim)
        return emb


class CrossAttention1D(nn.Module):
    """
    一维交叉注意力模块
    ====================
    Query 来自输入特征, Key/Value 来自条件 (弯曲角)
    实现报告中要求的 "交叉注意力条件机制"
    """

    def __init__(self, feature_dim, cond_dim, num_heads=4, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        assert feature_dim % num_heads == 0

        self.q_proj = nn.Conv1d(feature_dim, feature_dim, 1)
        self.k_proj = nn.Conv1d(cond_dim, feature_dim, 1)
        self.v_proj = nn.Conv1d(cond_dim, feature_dim, 1)
        self.out_proj = nn.Conv1d(feature_dim, feature_dim, 1)
        self.norm = nn.GroupNorm(1, feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x, cond):
        """
        x    : (B, C, L)  主特征
        cond : (B, C_cond, L) 条件特征
        """
        B, C, L = x.shape

        residual = x
        x = self.norm(x)

        q = self.q_proj(x)    # (B, C, L)
        k = self.k_proj(cond)  # (B, C, L)
        v = self.v_proj(cond)  # (B, C, L)

        # Reshape for multi-head attention
        q = q.view(B, self.num_heads, self.head_dim, L)
        k = k.view(B, self.num_heads, self.head_dim, L)
        v = v.view(B, self.num_heads, self.head_dim, L)

        # (B, heads, head_dim, L) @ (B, heads, L, head_dim) -> (B, heads, head_dim, head_dim)
        # 更改为: attention over sequence dimension
        # q: (B, heads, L, head_dim) @ k^T: (B, heads, head_dim, L) -> (B, heads, L, L)
        q = q.permute(0, 1, 3, 2)  # (B, heads, L, head_dim)
        k = k.permute(0, 1, 3, 2)  # (B, heads, L, head_dim)
        v = v.permute(0, 1, 3, 2)  # (B, heads, L, head_dim)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, heads, L, L)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (B, heads, L, head_dim)
        out = out.permute(0, 1, 3, 2).contiguous().view(B, C, L)

        out = self.out_proj(out)
        return out + residual


class ResBlock1D(nn.Module):
    """残差卷积块 (含时间嵌入注入)"""

    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm1 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.act = nn.SiLU()
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.act(self.norm1(self.conv1(x)))
        h = h + self.time_proj(t_emb).unsqueeze(-1)
        h = self.act(self.norm2(self.conv2(h)))
        return h + self.skip(x)


# =====================================================================
# 增强版模型 (交叉注意力 + 多变量)
# =====================================================================
class EnhancedConditionalUNet1D(nn.Module):
    """
    增强版条件 U-Net
    =====================
    改进:
    1. 正弦时间嵌入 (替代简单 MLP)
    2. 交叉注意力条件机制 (替代简单拼接)
    3. 残差块与 GroupNorm
    4. 多变量输出 (out_channels=3: 温度/压力/湿度)
    """

    def __init__(self, in_channels=1, cond_channels=1, out_channels=3,
                 base_dim=64, time_dim=128, num_heads=4,
                 use_cross_attention=True):
        super().__init__()
        self.use_cross_attention = use_cross_attention

        # === 时间嵌入 ===
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # === 条件编码器 (将弯曲角映射到 base_dim) ===
        self.cond_encoder = nn.Sequential(
            nn.Conv1d(cond_channels, base_dim, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(base_dim, base_dim, 3, padding=1),
            nn.SiLU(),
        )

        # === 编码器 ===
        ch1, ch2, ch3 = base_dim, base_dim * 2, base_dim * 4

        self.enc1 = ResBlock1D(in_channels, ch1, time_dim)
        self.cross_attn1 = CrossAttention1D(ch1, base_dim, num_heads) if use_cross_attention else None
        self.pool1 = nn.MaxPool1d(2)

        self.enc2 = ResBlock1D(ch1, ch2, time_dim)
        self.cross_attn2 = CrossAttention1D(ch2, base_dim, num_heads) if use_cross_attention else None
        self.pool2 = nn.MaxPool1d(2)

        # === Bottleneck ===
        self.bottleneck = ResBlock1D(ch2, ch3, time_dim)
        self.cross_attn_bot = CrossAttention1D(ch3, base_dim, num_heads) if use_cross_attention else None

        # === 解码器 ===
        self.up2 = nn.ConvTranspose1d(ch3, ch2, 2, stride=2)
        self.dec2 = ResBlock1D(ch2 + ch2, ch2, time_dim)  # skip connection

        self.up1 = nn.ConvTranspose1d(ch2, ch1, 2, stride=2)
        self.dec1 = ResBlock1D(ch1 + ch1, ch1, time_dim)

        # === 输出头 ===
        self.out_conv = nn.Sequential(
            nn.Conv1d(ch1, ch1, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(ch1, out_channels, 1),
        )

        # 条件下采样 (用于多尺度交叉注意力)
        self.cond_down1 = nn.MaxPool1d(2)
        self.cond_down2 = nn.MaxPool1d(2)
        self.cond_proj2 = nn.Conv1d(base_dim, base_dim, 1)  # 保持 base_dim
        self.cond_proj_bot = nn.Conv1d(base_dim, base_dim, 1)

    def forward(self, x, t, condition):
        """
        x         : (B, out_channels, L)  带噪目标
        t         : (B, 1)                时间步
        condition : (B, cond_channels, L) 条件输入 (弯曲角)
        """
        # 时间嵌入
        t_emb = self.time_embed(t.squeeze(-1))  # (B, time_dim)

        # 条件编码
        cond_feat = self.cond_encoder(condition)  # (B, base_dim, L)

        # === 编码器 ===
        d1 = self.enc1(x, t_emb)                   # (B, ch1, L)
        if self.use_cross_attention and self.cross_attn1 is not None:
            d1 = self.cross_attn1(d1, cond_feat)

        p1 = self.pool1(d1)                         # (B, ch1, L//2)

        # 下采样条件
        cond_down1 = self.cond_down1(cond_feat)      # (B, base_dim, L//2)

        d2 = self.enc2(p1, t_emb)                   # (B, ch2, L//2)
        if self.use_cross_attention and self.cross_attn2 is not None:
            cond_for_attn2 = self.cond_proj2(cond_down1)
            # 需要扩展 cond 到 ch2 维度用于 K/V
            d2 = self.cross_attn2(d2, cond_for_attn2)

        p2 = self.pool2(d2)                          # (B, ch2, L//4)

        # 下采样条件
        cond_down2 = self.cond_down2(cond_down1)

        # === Bottleneck ===
        b = self.bottleneck(p2, t_emb)               # (B, ch3, L//4)
        if self.use_cross_attention and self.cross_attn_bot is not None:
            cond_for_bot = self.cond_proj_bot(cond_down2)
            b = self.cross_attn_bot(b, cond_for_bot)

        # === 解码器 ===
        u2 = self.up2(b)                              # (B, ch2, L//2)
        if u2.shape[2] != d2.shape[2]:
            u2 = F.pad(u2, (0, d2.shape[2] - u2.shape[2]))
        u2 = torch.cat([u2, d2], dim=1)               # skip
        u2 = self.dec2(u2, t_emb)

        u1 = self.up1(u2)                              # (B, ch1, L)
        if u1.shape[2] != d1.shape[2]:
            u1 = F.pad(u1, (0, d1.shape[2] - u1.shape[2]))
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.dec1(u1, t_emb)

        return self.out_conv(u1)
