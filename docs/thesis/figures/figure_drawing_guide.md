# 论文图例绘制说明

本文档用于汇总 `draft.md` 中所有“图 x 建议”的绘制方式，区分：

1. 可直接使用 Banana 进行 AI 绘图的示意图
2. 更适合手工绘制的流程图/结构图
3. 需要从外部获取官方资料或基于真实实验数据绘制的图

## 一、Banana 通用风格设置

建议在所有 AI 绘图 prompt 末尾追加如下统一风格描述：

```text
clean scientific infographic, flat vector style, white background, blue and teal color palette, minimal academic layout, high resolution, no watermark, no photo style, no clutter, with clear English labels automatically placed near each component
```

建议事项：

- 若需要自动填充标签，优先使用英文标签，生成稳定性通常高于中文。
- AI 图中的中文文字、公式和复杂箭头说明仍建议后期手动微调。
- 如果 Banana 支持负面提示，可加入 `no dense text, no 3d rendering, no photorealism, no unnecessary decoration`。
- 原理示意图建议输出横版，便于插入论文。

## 二、适合使用 Banana 直接绘制的图

### 图 A1 GNSS 无线电掩星观测原理图

对应正文位置：

- 第 2 章 GNSS-RO 原理部分
- 第 3 章 FY-3D/GNOS 数据来源部分可复用

建议用途：

- 展示 GNSS 卫星、LEO 卫星、地球、切点、弯曲路径、弯曲角、冲击参数

Banana prompt：

```text
Scientific schematic of GNSS radio occultation in Earth's atmosphere, show one GNSS satellite, one LEO satellite, Earth curvature, tangent point, bent signal ray path through the atmosphere, bending angle, impact parameter, atmospheric layers, automatically place clear English labels for GNSS satellite, LEO satellite, tangent point, bent ray path, bending angle, impact parameter, atmosphere, Earth surface, clean academic vector infographic, flat design, white background, blue and teal palette, precise geometry, clean scientific infographic, flat vector style, white background, blue and teal color palette, minimal academic layout, high resolution, no watermark, no photo style, no clutter, with clear English labels automatically placed near each component
```

建议标签：

- GNSS 卫星
- LEO 卫星
- 切点
- 弯曲射线路径
- 弯曲角
- 冲击参数

### 图 A2 传统 GNSS-RO 反演链路示意图

建议用途：

- 展示“相位观测/多普勒频移 -> 弯曲角 -> 折射率 -> 温度/气压/湿度”的处理链路

Banana prompt：

```text
Flowchart infographic of traditional GNSS radio occultation retrieval chain, from phase observation to Doppler shift, bending angle, refractivity, then temperature pressure humidity retrieval, include arrows and layered boxes, automatically place clear English labels for Phase observation, Doppler shift, Bending angle, Refractivity, Temperature, Pressure, Humidity, optional Background constraint, clean scientific vector diagram, minimal academic style, white background, blue gray palette, clean scientific infographic, flat vector style, white background, blue and teal color palette, minimal academic layout, high resolution, no watermark, no photo style, no clutter, with clear English labels automatically placed near each component
```

建议标签：

- 相位观测
- 多普勒频移
- 弯曲角
- 折射率
- 温度/气压/湿度
- 背景场约束（可选）

### 图 A3 DDPM 前向加噪与反向去噪过程示意图

建议用途：

- 用于第 2 章扩散模型原理部分

Banana prompt：

```text
Scientific diffusion model infographic, left-to-right forward noising process and right-to-left reverse denoising process for 1D atmospheric profile retrieval, show real sample, noisy sample, predicted noise, conditional bending-angle profile, symmetric process arrows, automatically place clear English labels for Real sample, Noisy sample, Predicted noise, Conditional bending-angle profile, Forward diffusion, Reverse denoising, clean flat vector academic diagram, white background, blue teal palette, minimal style, clean scientific infographic, flat vector style, white background, blue and teal color palette, minimal academic layout, high resolution, no watermark, no photo style, no clutter, with clear English labels automatically placed near each component
```

建议标签：

- 真实样本
- 带噪样本
- 噪声预测网络输出
- 条件输入弯曲角廓线
- 前向扩散
- 反向去噪

### 图 A4 GNSS 掩星观测几何补充图

适用场景：

- 如果第 3 章想单独放一张更简化的观测几何图，可直接复用图 A1 的 prompt

可选简化版 prompt：

```text
Minimal scientific geometry diagram of GNSS radio occultation, show GNSS satellite, LEO satellite, Earth limb, tangent point and bent ray path, automatically place clear English labels for GNSS satellite, LEO satellite, Earth limb, tangent point, bent ray path, simple vector layout, white background, blue palette, academic textbook style
```

## 三、建议从外部获取或参考官方资料的图

### 图 B1 FY-3D 卫星平台与 GNOS 载荷示意图

建议方式：

- 优先从国家卫星气象中心 NSMC 官方页面、FY-3D 官方介绍页、GNOS 产品说明文档中查找
- 如果只能找到复杂原图，建议基于官方图进行简化重绘

原因：

- 该图涉及真实卫星平台结构和载荷位置，AI 容易画错
- 论文中使用真实任务平台示意图更稳妥

推荐来源：

- FY-3D 官方页面
- 国家卫星气象中心产品介绍页
- GNOS 任务介绍 PDF 或公开论文中的 mission 图

使用建议：

- 若直接引用外部图，需按学校要求标注来源
- 若参考后自行重绘，可在图注写“根据某某资料整理绘制”

## 四、不建议 AI 绘制，建议手工绘制的流程图/结构图

这类图的关键在于“逻辑准确”，优先使用 draw.io、ProcessOn、Visio 或 PPT 绘制。

### 图 C1 一维条件 U-Net 结构示意图

原因：

- 编码器/解码器/跳跃连接/交叉注意力位置必须准确
- AI 容易把网络层次和连接关系画错

推荐布局：

```text
输入 x_t(3x301) -> 编码器 Block1 -> 下采样 -> 编码器 Block2 -> 下采样 -> 编码器 Block3
                                     |                             |
                                     |<------ 条件交叉注意力 ------|

时间步 t -> 正弦时间嵌入 -> MLP -> 注入各残差块
条件 c(弯曲角) -> 条件编码器 -> 多尺度条件特征 -> 交叉注意力模块

瓶颈层 -> 解码器 Block3 -> 上采样 -> 解码器 Block2 -> 上采样 -> 解码器 Block1 -> 输出(3x301)
并在对应层之间画跳跃连接
```

图例颜色建议：

- 主干特征流：蓝色
- 条件特征流：橙色
- 时间嵌入：绿色

### 图 C2 系统需求到模块功能的对应关系图

推荐布局：

```text
左侧：需求
- 数据处理需求
- 模型训练需求
- 推理反演需求
- 评估可视化需求

右侧：模块
- 数据处理模块
- 训练模块
- 推理模块
- 评估与前端模块

中间用箭头对应输入输出
```

### 图 C3 样本构建流程图

推荐布局：

```text
原始 WAP 文件
-> 匹配 ATP 文件
-> 读取变量
-> 质量控制
-> 高度插值
-> 对数变换
-> 标准化
-> 划分训练/验证/测试集
-> 保存 npz/统计量
```

### 图 C4 数据处理流水线图

推荐布局：

```text
文件配对 -> 变量提取 -> 质量控制 -> 统一高度网格 -> 对数变换 -> 非负裁剪 -> Z-Score 标准化 -> 数据集划分
```

### 图 C5 数据处理实现流程图

推荐布局：

```text
原始 ATP/WAP 文件 -> 配对 -> 插值 -> 质控 -> 标准化 -> 数据集保存
```

图形类型建议：

- 文件：圆角矩形
- 处理操作：普通矩形
- 输出结果：带底纹矩形

### 图 C6 模型训练流程图

推荐布局：

```text
读取 batch
-> 采样时间步 t
-> 添加噪声得到 x_t
-> 输入条件 c 和时间步 t
-> U-Net 预测噪声
-> 计算加权损失和梯度约束
-> 反向传播
-> 参数更新
-> 验证集监控
-> 保存最佳权重
```

### 图 C7 DDPM 与 DDIM 推理流程对比图

推荐布局：

```text
左列 DDPM：
高斯噪声 -> 1000 步反向去噪 -> 输出廓线

右列 DDIM：
高斯噪声 -> 50 步跳步去噪 -> 输出廓线

底部增加对比项：
采样步数 / 是否随机 / 推理耗时 / 精度变化
```

## 五、必须基于真实数据或真实实验结果绘制的图

以下图不建议使用 AI 生成。

### 图 D1 ATP 弯曲角样例廓线图

建议方式：

- 从真实 ATP 样本中读取一条典型弯曲角廓线
- 使用 Matplotlib 绘制“横轴弯曲角、纵轴高度”的折线图

### 图 D2 WAP 温度、气压、湿度典型廓线图

建议方式：

- 从真实 WAP 样本中选取一条典型剖面
- 分三幅子图绘制温度、气压、湿度随高度变化曲线

### 图 D3 训练损失与验证损失曲线图

建议方式：

- 根据训练日志绘制真实曲线
- 若可行，可叠加总验证损失与湿度分量验证损失

### 图 D4 三变量主实验结果可视化图

建议方式：

- 根据表 5-1 绘制柱状图

### 图 D5 旧基线与最终实验结果对比图

建议方式：

- 根据表 5-2 绘制柱状图或雷达图
- 若重点突出湿度改进，可单独画湿度 CC 和 RMSE 对比

### 图 D6 采样方法精度-效率对比图

建议方式：

- 横轴：推理时间
- 纵轴：湿度 CC 或综合指标
- 至少包含 DDPM 1000 步与 DDIM 50 步两个点

### 图 D7 损失函数消融结果图

建议方式：

- 根据表 5-4 绘制柱状图
- 推荐两张并排子图：湿度 CC、湿度 RMSE

### 图 D8 湿度廓线平滑效果对比图

建议方式：

- 选取同一样本，对比“无梯度约束”和“最终模型”的湿度曲线

### 图 D9 典型样本剖面对比图

建议方式：

- 选择“效果较好”“中等”“较差”三类样本
- 每类样本展示弯曲角输入、预测曲线和标签曲线

## 六、推荐执行顺序

建议按以下顺序准备图：

1. 先做原理类图：图 A1、图 A2、图 A3
2. 再找外部官方图：图 B1
3. 然后手工画结构图和流程图：图 C1 到图 C7
4. 最后统一用真实数据出实验图：图 D1 到图 D9

## 七、后续可继续补充的内容

如果后续需要，可以继续补充：

- 每一张数据图对应的 Matplotlib 脚本模板
- 每一张手工流程图的编号和建议图名
- 外部图片的标准图注写法
