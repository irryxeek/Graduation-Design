# `unet.py` 模块代码详解

## 1. 文件定位与作用

文件路径：

- `ro_retrieval/model/unet.py`

这个文件定义了项目中的两套一维条件 U-Net：

- `ConditionalUNet1D`：原始版本，主要用于兼容旧权重
- `EnhancedConditionalUNet1D`：增强版本，是当前主线模型

它们都服务于扩散模型中的“去噪网络”角色。  
在训练时，它们接收带噪目标 `x_t`、时间步 `t` 和条件输入 `condition`，输出对噪声的预测 `noise_pred`。

在你的项目里可以把它理解成：

- 输入条件：弯曲角廓线
- 主分支输入：带噪的温度/气压廓线
- 输出：当前时间步的噪声估计

---

## 2. 文件整体结构

`unet.py` 可以分成 5 个部分：

1. 原始条件 U-Net：`ConditionalUNet1D`
2. 时间步嵌入：`SinusoidalTimeEmbedding`
3. 交叉注意力：`CrossAttention1D`
4. 残差块：`ResBlock1D`
5. 增强版条件 U-Net：`EnhancedConditionalUNet1D`

代码的组织逻辑是：

- 先保留一个简单、兼容旧权重的版本
- 再定义增强版所需要的辅助模块
- 最后组装成当前实际使用的增强版网络

---

## 3. 先理解这类 U-Net 在项目中的输入输出

在扩散模型里，`unet.py` 中网络的输入输出不是“原始物理量直接进出”，而是扩散过程里的中间状态。

### 3.1 输入

增强版 `forward(x, t, condition)` 中：

- `x`：带噪目标，形状为 `(B, out_channels, L)`
- `t`：时间步，形状为 `(B, 1)`
- `condition`：条件输入，形状为 `(B, cond_channels, L)`

在项目当前主线中通常是：

- `condition`：弯曲角，`cond_channels = 1`
- `x`：带噪的温度/气压目标，`out_channels = 2` 或 `3`
- `L = 301`：统一高度层数

### 3.2 输出

- 输出仍是 `(B, out_channels, L)`
- 但它不是直接输出温度/气压，而是预测“当前时间步加入的噪声”

这点非常关键，因为扩散模型训练目标是噪声预测，而不是直接回归物理量。

---

## 4. 原始版本：`ConditionalUNet1D`

对应代码开头的 `ConditionalUNet1D` 类。

## 4.1 结构概览

这个版本比较简单，特点是：

- 条件输入和主输入直接在通道维拼接
- 时间步用一个很浅的 `MLP` 编码
- 整体结构是经典的编码器 - 瓶颈 - 解码器

主要层如下：

- `time_mlp`：将时间步从标量映射到 32 维
- `down1`：第一层卷积
- `down2`：第二层卷积
- `pool`：下采样
- `bot1`：瓶颈卷积
- `up2`、`up1`：反卷积上采样
- `out`：输出卷积

## 4.2 前向流程

### 第一步：拼接条件输入

```python
x_in = torch.cat([x, condition], dim=1)
```

这里把：

- 主输入 `x`
- 条件输入 `condition`

直接在通道维拼接。  
这是一种最简单的条件注入方式。

### 第二步：时间步编码

```python
t_emb = self.time_mlp(t.float()).unsqueeze(-1)
```

作用是把扩散时间步编码成可参与特征计算的向量。

### 第三步：编码

```python
d1 = self.act(self.down1(x_in))
d1 = d1 + t_emb
p1 = self.pool(d1)
d2 = self.act(self.down2(p1))
p2 = self.pool(d2)
```

逻辑是：

- 先提取局部特征
- 注入时间信息
- 再下采样获取更高层特征

### 第四步：瓶颈

```python
b = self.act(self.bot1(p2))
```

这里是编码后的最深层表示。

### 第五步：解码与跳跃连接

```python
u2 = self.up2(b)
u2 = torch.cat([u2, d2], dim=1)
u1 = self.up1(u2)
u1 = torch.cat([u1, d1], dim=1)
return self.out(u1)
```

这里的核心是 U-Net 的 skip connection：

- 上采样恢复分辨率
- 与编码器对应层特征拼接
- 兼顾高层语义信息和底层细节

### 第六步：长度对齐

```python
if u2.shape[2] != d2.shape[2]:
    u2 = F.pad(u2, (0, 1))
```

因为长度 `301` 不是 2 的整次幂，经过池化和反卷积后可能会出现长度差 1 的情况，所以需要补齐。

## 4.3 这个版本的局限

原始版本能工作，但有 3 个明显局限：

- 条件信息只靠简单拼接，不够灵活
- 时间步表示较弱，只是浅层 MLP
- 没有残差块和更规范的归一化设计

这也是后面增强版出现的原因。

---

## 5. 时间步嵌入：`SinusoidalTimeEmbedding`

这个模块的作用是把离散时间步 `t` 转成连续、高维、可表达不同频率信息的向量。

## 5.1 为什么需要它

扩散模型不同时间步的去噪难度不同：

- 早期噪声很强
- 后期噪声较弱

因此网络必须知道“当前是第几步”，否则无法针对不同噪声水平采取不同策略。

## 5.2 代码逻辑

核心过程：

```python
half_dim = self.dim // 2
emb = math.log(10000) / (half_dim - 1)
emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
emb = t.float().unsqueeze(-1) * emb.unsqueeze(0)
emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
```

可以理解为：

- 先构造一组不同频率
- 用 `sin` 和 `cos` 编码时间步
- 得到长度为 `dim` 的向量

这和 Transformer 的位置编码思路很像。

## 5.3 输出含义

如果 `dim = 128`，那么输出形状就是：

- `(B, 128)`

后续会再送入线性层和激活函数，形成更强的时间特征表示。

---

## 6. 交叉注意力：`CrossAttention1D`

这是增强版中最关键的模块之一。

## 6.1 它解决什么问题

原始版本里，条件信息只是直接拼接到输入上。  
增强版希望做到：

- 在不同尺度层
- 动态参考弯曲角条件
- 按“当前主特征需要什么”去取用条件信息

这就是交叉注意力。

## 6.2 自注意力和交叉注意力的区别

### 自注意力

- `Q、K、V` 都来自同一条特征

### 交叉注意力

- `Q` 来自主特征
- `K、V` 来自条件特征

在这里：

- 主特征是当前正在去噪的目标特征
- 条件特征是弯曲角编码后的特征

## 6.3 初始化部分在做什么

```python
self.q_proj = nn.Conv1d(feature_dim, feature_dim, 1)
self.k_proj = nn.Conv1d(cond_dim, feature_dim, 1)
self.v_proj = nn.Conv1d(cond_dim, feature_dim, 1)
self.out_proj = nn.Conv1d(feature_dim, feature_dim, 1)
```

含义是：

- 用 `1x1 Conv1d` 做线性投影
- 把主特征映射成 `Q`
- 把条件特征映射成 `K` 和 `V`

这里用 `Conv1d(kernel_size=1)` 本质上就相当于逐位置线性变换。

## 6.4 `GroupNorm` 和残差

```python
self.norm = nn.GroupNorm(1, feature_dim)
```

先归一化再做注意力，是为了稳定训练。

```python
residual = x
...
return out + residual
```

这是残差连接，用于保留原特征并改善梯度传播。

## 6.5 前向过程详细解释

### 第一步：输入

```python
x    : (B, C, L)
cond : (B, C_cond, L)
```

这里：

- `x` 是主特征
- `cond` 是条件特征

### 第二步：投影成 Q/K/V

```python
q = self.q_proj(x)
k = self.k_proj(cond)
v = self.v_proj(cond)
```

得到的三者形状都先统一成 `(B, C, L)`。

### 第三步：多头拆分

```python
q = q.view(B, self.num_heads, self.head_dim, L)
```

例如如果：

- `feature_dim = 64`
- `num_heads = 4`

那么每个头就是 `16` 维。

### 第四步：转置后按序列维做注意力

```python
q = q.permute(0, 1, 3, 2)  # (B, heads, L, head_dim)
```

这样做是为了在高度序列维度 `L` 上计算注意力矩阵。

最终：

```python
attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
```

得到的是：

- `(B, heads, L, L)`

表示每个高度位置对其他高度位置的条件相关性。

### 第五步：加权聚合条件信息

```python
out = torch.matmul(attn, v)
```

这一步相当于：

- 用注意力权重从条件特征中提取与当前位置最相关的信息

### 第六步：恢复形状并输出

```python
out = out.permute(0, 1, 3, 2).contiguous().view(B, C, L)
out = self.out_proj(out)
return out + residual
```

把多头结果拼回原通道维，再加残差输出。

## 6.6 这个模块在项目里的意义

它使模型不再只是“静态拿到弯曲角条件”，而是：

- 在去噪过程中
- 结合当前主特征状态
- 动态参考弯曲角廓线

这比简单拼接更适合“条件反演”任务。

---

## 7. 残差块：`ResBlock1D`

这是增强版的基本构件。

## 7.1 结构组成

```python
self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
self.time_proj = nn.Linear(time_dim, out_ch)
self.norm1 = nn.GroupNorm(min(8, out_ch), out_ch)
self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
```

包含：

- 两层卷积
- 两次归一化
- 一次时间嵌入注入
- 一条残差支路

## 7.2 前向逻辑

```python
h = self.act(self.norm1(self.conv1(x)))
h = h + self.time_proj(t_emb).unsqueeze(-1)
h = self.act(self.norm2(self.conv2(h)))
return h + self.skip(x)
```

解释如下：

### 第一步：卷积提取特征

- 用第一层卷积提取局部模式
- 再经过归一化和 `SiLU`

### 第二步：注入时间步信息

```python
h = h + self.time_proj(t_emb).unsqueeze(-1)
```

把时间步从 `(B, time_dim)` 映射到 `(B, out_ch)`，再扩展成 `(B, out_ch, 1)` 与序列特征逐位置相加。

这说明：

- 每个卷积块都能感知扩散时间步

### 第三步：第二次卷积变换

- 进一步提取特征

### 第四步：残差输出

```python
return h + self.skip(x)
```

如果输入输出通道不同，就先用 `1x1 Conv` 对齐通道数，再相加。

## 7.3 残差块在项目中的作用

- 提高深层网络训练稳定性
- 保留原特征信息
- 让网络学习增量修正，而不是完全重建

---

## 8. 增强版主模型：`EnhancedConditionalUNet1D`

这是当前最重要的类。

## 8.1 增强点概览

相对原始版本，它有 4 个主要增强：

1. 用正弦时间嵌入替代浅层时间 MLP
2. 用交叉注意力替代简单条件拼接
3. 用残差块和 `GroupNorm` 提升表达和稳定性
4. 支持多变量输出

## 8.2 初始化参数含义

```python
def __init__(self, in_channels=1, cond_channels=1, out_channels=3,
             base_dim=64, time_dim=128, num_heads=4,
             use_cross_attention=True):
```

关键参数解释：

- `in_channels`：主输入通道数，即带噪目标通道数
- `cond_channels`：条件输入通道数，当前弯曲角为 1
- `out_channels`：最终输出通道数，比如温度/气压/湿度
- `base_dim`：基础通道宽度，决定整体模型容量
- `time_dim`：时间嵌入维度
- `num_heads`：多头注意力头数
- `use_cross_attention`：是否启用交叉注意力

---

## 9. 增强版的各模块解析

## 9.1 时间嵌入模块

```python
self.time_embed = nn.Sequential(
    SinusoidalTimeEmbedding(time_dim),
    nn.Linear(time_dim, time_dim),
    nn.SiLU(),
    nn.Linear(time_dim, time_dim),
)
```

这里相当于：

- 先用正弦编码表示时间步
- 再用 MLP 做非线性映射

使时间步信息更适合送入后续残差块。

## 9.2 条件编码器

```python
self.cond_encoder = nn.Sequential(
    nn.Conv1d(cond_channels, base_dim, 3, padding=1),
    nn.SiLU(),
    nn.Conv1d(base_dim, base_dim, 3, padding=1),
    nn.SiLU(),
)
```

作用：

- 把原始弯曲角序列编码成更高维的条件特征

输入：

- `(B, 1, 301)`

输出：

- `(B, 64, 301)`，如果 `base_dim = 64`

## 9.3 编码器主干

```python
ch1, ch2, ch3 = base_dim, base_dim * 2, base_dim * 4
```

所以通道数依次是：

- `64`
- `128`
- `256`

对应模块：

- `enc1`：第一层残差块
- `cross_attn1`：第一层交叉注意力
- `pool1`：下采样
- `enc2`：第二层残差块
- `cross_attn2`：第二层交叉注意力
- `pool2`：下采样

## 9.4 瓶颈层

```python
self.bottleneck = ResBlock1D(ch2, ch3, time_dim)
self.cross_attn_bot = CrossAttention1D(ch3, base_dim, num_heads)
```

瓶颈层位于最低分辨率、最高语义层级。

它承担两个职责：

- 汇聚编码器得到的高层特征
- 在最抽象层面引入条件约束

## 9.5 解码器

```python
self.up2 = nn.ConvTranspose1d(ch3, ch2, 2, stride=2)
self.dec2 = ResBlock1D(ch2 + ch2, ch2, time_dim)

self.up1 = nn.ConvTranspose1d(ch2, ch1, 2, stride=2)
self.dec1 = ResBlock1D(ch1 + ch1, ch1, time_dim)
```

逻辑和标准 U-Net 一致：

- 先上采样
- 与编码器对应层特征拼接
- 再用残差块融合

## 9.6 输出头

```python
self.out_conv = nn.Sequential(
    nn.Conv1d(ch1, ch1, 3, padding=1),
    nn.SiLU(),
    nn.Conv1d(ch1, out_channels, 1),
)
```

作用：

- 将最后的高分辨率特征映射到目标输出通道数

如果当前任务输出温度和气压，那么最终输出就是：

- `(B, 2, 301)`

---

## 10. 为什么还要对条件分支做下采样

代码中有：

```python
self.cond_down1 = nn.MaxPool1d(2)
self.cond_down2 = nn.MaxPool1d(2)
```

原因是：

- 编码器每下采样一次，主特征长度就减半
- 如果条件特征仍保持原长度，就无法与对应尺度主特征做注意力

所以需要对条件特征也同步降采样到：

- `L`
- `L/2`
- `L/4`

这就是“多尺度条件融合”的基础。

---

## 11. 增强版 `forward()` 详细流程

下面按执行顺序解释。

## 11.1 时间步编码

```python
t_emb = self.time_embed(t.squeeze(-1))
```

输入：

- `t` 形状为 `(B, 1)`

先压成 `(B,)`，再编码成：

- `(B, 128)`

这个 `t_emb` 后面会被送进每个 `ResBlock1D`。

## 11.2 条件编码

```python
cond_feat = self.cond_encoder(condition)
```

把原始弯曲角序列编码成高维特征。

## 11.3 编码器第一层

```python
d1 = self.enc1(x, t_emb)
d1 = self.cross_attn1(d1, cond_feat)
```

逻辑是：

1. 先用残差块提取主特征
2. 再让主特征通过交叉注意力参考弯曲角条件

这一层保留的是最高分辨率的局部细节。

## 11.4 第一层下采样

```python
p1 = self.pool1(d1)
cond_down1 = self.cond_down1(cond_feat)
```

主特征和条件特征同时降采样，为下一层做准备。

## 11.5 编码器第二层

```python
d2 = self.enc2(p1, t_emb)
cond_for_attn2 = self.cond_proj2(cond_down1)
d2 = self.cross_attn2(d2, cond_for_attn2)
```

这里的 `cond_proj2` 是一个 `1x1 Conv`，作用是把条件特征投影到更适合当前注意力层使用的表示空间。

## 11.6 第二层下采样

```python
p2 = self.pool2(d2)
cond_down2 = self.cond_down2(cond_down1)
```

继续进入更深层表示。

## 11.7 瓶颈层

```python
b = self.bottleneck(p2, t_emb)
cond_for_bot = self.cond_proj_bot(cond_down2)
b = self.cross_attn_bot(b, cond_for_bot)
```

这是最深层的语义融合位置。

你可以把它理解成：

- 在最抽象的大尺度表示上，再次利用弯曲角条件引导去噪方向

## 11.8 解码器第一阶段

```python
u2 = self.up2(b)
u2 = torch.cat([u2, d2], dim=1)
u2 = self.dec2(u2, t_emb)
```

逻辑：

1. 先上采样恢复长度
2. 与编码器第二层特征拼接
3. 用残差块融合信息

## 11.9 解码器第二阶段

```python
u1 = self.up1(u2)
u1 = torch.cat([u1, d1], dim=1)
u1 = self.dec1(u1, t_emb)
```

继续恢复到原始长度。

## 11.10 最终输出

```python
return self.out_conv(u1)
```

输出与 `x` 形状一致：

- `(B, out_channels, L)`

在扩散训练里，这个输出会被拿去和真实噪声做损失计算。

---

## 12. 为什么编码器里用了交叉注意力，解码器里没继续用

当前实现中：

- 交叉注意力主要放在编码端和瓶颈层
- 解码器主要依赖 skip connection 和已融合好的编码器特征

这样设计的好处是：

- 条件融合位置清晰
- 计算量相对可控
- 已经能实现多尺度条件约束

当然，这也意味着：

- 当前版本并没有在解码阶段进一步细化条件注意力
- 这是后续可能的增强点

---

## 13. 长度不一致时为什么要 `pad`

代码中多次出现：

```python
if u2.shape[2] != d2.shape[2]:
    u2 = F.pad(u2, (0, d2.shape[2] - u2.shape[2]))
```

原因是：

- 序列长度 `301` 经过 `MaxPool1d(2)` 后会变成奇偶不整齐的长度
- 反卷积上采样回来后，长度可能和 skip connection 对应层差 1

为了顺利拼接，就必须手动补齐。

这是处理非 2 的幂长度序列时很常见的工程细节。

---

## 14. 这份代码体现的设计思想

可以概括成下面 5 点：

1. **一维化建模**  
   输入输出都是垂直廓线，所以全网采用 `Conv1d`

2. **多尺度特征提取**  
   编码器 - 解码器结构提取局部与全局结构

3. **时间感知去噪**  
   每个残差块都注入时间步嵌入

4. **条件动态引导**  
   通过交叉注意力把弯曲角条件引入不同尺度层

5. **稳定训练**  
   使用 `ResBlock + GroupNorm + 残差连接`

---

## 15. 从答辩角度，如何概括这份 `unet.py`

最推荐的说法：

> `unet.py` 定义了扩散模型中的核心去噪网络。原始版本采用简单的条件拼接方式，主要用于兼容旧权重；增强版则引入了正弦时间嵌入、残差块、GroupNorm 和多尺度交叉注意力，使模型能够在不同去噪阶段、不同特征尺度上动态利用弯曲角条件信息，从而更有效地重建温度和气压廓线。

如果老师继续追问“它具体怎么工作”，你就按下面这个顺序答：

1. 输入带噪目标 `x_t`、时间步 `t`、弯曲角条件 `condition`
2. 时间步先编码成高维向量
3. 弯曲角先编码成条件特征
4. 主分支经过编码器、瓶颈、解码器提取多尺度特征
5. 在编码器和瓶颈层用交叉注意力融合条件信息
6. 最后输出当前时间步的噪声预测

---

## 16. 这份代码你最该记住的 6 个点

1. `ConditionalUNet1D` 是旧版，`EnhancedConditionalUNet1D` 是主线
2. 网络输出的是噪声预测，不是直接物理量
3. 时间步通过 `SinusoidalTimeEmbedding` 编码
4. 条件弯曲角先经过 `cond_encoder` 编码
5. 交叉注意力在编码器多尺度层和瓶颈层实现
6. 解码器通过 skip connection 恢复细节

---

## 17. 可继续阅读的关联文件

如果你想把 `unet.py` 放到整条链路中理解，建议继续读：

- `ro_retrieval/model/diffusion.py`：扩散前向加噪和反向采样
- `ro_retrieval/training/trainer.py`：训练时如何调用 U-Net 预测噪声
- `ro_retrieval/config.py`：模型超参数配置

这样就能把：

- U-Net 结构
- 扩散过程
- 训练损失

三者连起来理解。
