# 基于掩星数据的气象要素反演系统设计与实现

学生：林逸飞（220110814）

专业：计算机科学与技术

说明：本文档为毕业论文初稿，依据 `docs/thesis/outline.md`、项目技术文档和当前实验结果整理而成。文中图表编号、参考文献格式和部分外部文献条目信息后续需按学校模板统一核对。

---

# 摘要

全球导航卫星系统无线电掩星（Global Navigation Satellite System Radio Occultation, GNSS-RO）能够通过接收穿越大气层的 GNSS 信号，获得具有较高垂直分辨率和全天候观测能力的大气廓线信息，是数值天气预报、气候监测和大气科学研究中的重要观测手段。传统 GNSS-RO 大气反演通常采用 Abel 逆变换、静力平衡方程和一维变分同化等多步骤流程，将弯曲角观测逐级转换为折射率、温度、气压和湿度等气象要素。该类方法具有明确的物理基础，但在实际应用中容易受到球对称假设、误差逐级传递、湿项与干项耦合等因素影响，尤其在低对流层湿度反演中存在较大不确定性。

针对上述问题，本文设计并实现了一套基于条件扩散模型的 GNSS-RO 气象要素端到端反演系统。系统以 FY-3D GNOS L2 ATP 产品中的弯曲角廓线作为条件输入，以 WAP 湿大气产品中的温度、气压和湿度廓线作为监督标签，构建了覆盖 2025 年 1 月至 6 月的 ATP+WAP 配对数据集。经过文件配对、质量控制、高度网格插值、非线性变换和标准化处理后，共获得 64,116 条有效大气廓线样本。模型方面，本文采用去噪扩散概率模型（Denoising Diffusion Probabilistic Model, DDPM）作为生成式反演框架，设计增强版一维条件 U-Net 作为噪声预测网络，并引入正弦时间嵌入、残差卷积块、GroupNorm 和交叉注意力机制，实现弯曲角条件信息对温度、气压和湿度三变量生成过程的多尺度约束。

在 Tesla V100-PCIE-32GB 设备上的实验结果表明，该模型可以完成三变量联合反演任务。在全部 9,618 个测试样本上，温度、气压和湿度的平均相关系数分别为 0.7820、0.9990 和 0.6960，标准化空间 RMSE 分别为 0.6267、0.0756 和 0.7996。与较小规模旧基线相比，湿度相关系数由 0.5321 提升至 0.6960，湿度 RMSE 由 0.8690 降低至 0.7996，表明扩充数据规模并针对湿度通道进行加权训练有助于改善湿度反演质量。本文还基于 Streamlit 实现了交互式可视化原型，支持样本选择、一键反演和多变量剖面对比展示，验证了条件扩散模型用于 GNSS-RO 多变量大气剖面反演的可行性。

关键词：GNSS 无线电掩星；大气剖面反演；条件扩散模型；一维 U-Net；FY-3D GNOS；湿度反演

---

# Abstract

Global Navigation Satellite System Radio Occultation (GNSS-RO) is an important atmospheric remote sensing technique with global coverage, high vertical resolution and all-weather observation capability. It provides valuable profile information for numerical weather prediction, climate monitoring and atmospheric research. Conventional GNSS-RO retrieval systems usually rely on a cascade of physical procedures, including Abel inversion, hydrostatic integration and one-dimensional variational retrieval. Although these methods are physically interpretable, their performance can be affected by spherical symmetry assumptions, error propagation and the coupling between dry and wet atmospheric components, especially for humidity retrieval in the lower troposphere.

To address these issues, this thesis designs and implements an end-to-end meteorological profile retrieval system based on a conditional diffusion model. The system uses bending angle profiles from FY-3D GNOS L2 ATP products as conditional inputs and temperature, pressure and humidity profiles from WAP wet atmosphere products as supervised labels. A paired ATP+WAP dataset covering January to June 2025 is constructed through file matching, quality control, height-grid interpolation, nonlinear transformation and standardization. Finally, 64,116 valid atmospheric profile samples are obtained. In terms of modeling, a denoising diffusion probabilistic model (DDPM) is adopted as the generative retrieval framework, and an enhanced one-dimensional conditional U-Net is designed as the noise prediction network. Sinusoidal time embedding, residual convolution blocks, GroupNorm and cross-attention modules are introduced to inject bending angle conditions into the multi-variable denoising process.

Experimental results show that the proposed model can perform joint retrieval of temperature, pressure and humidity profiles. Evaluated on all 9,618 test samples from the first half of 2025, the model achieves mean correlation coefficients of 0.7820, 0.9990 and 0.6960 for temperature, pressure and humidity, respectively, with normalized RMSE values of 0.6267, 0.0756 and 0.7996. Compared with the previous baseline trained on a smaller dataset, the humidity correlation coefficient increases from 0.5321 to 0.6960, and the humidity RMSE decreases from 0.8690 to 0.7996. These results indicate that expanding the paired dataset and applying humidity-aware weighted training can improve humidity retrieval performance. In addition, a Streamlit-based interactive prototype is developed to support sample selection, one-click retrieval and profile visualization. The study demonstrates the feasibility of applying conditional diffusion models to GNSS-RO multi-variable atmospheric profile retrieval.

Keywords: GNSS radio occultation; atmospheric profile retrieval; conditional diffusion model; one-dimensional U-Net; FY-3D GNOS; humidity retrieval

---

# 第1章 绪论

## 1.1 研究背景与意义

大气温度、气压和湿度垂直廓线是描述大气热力结构与水汽分布的重要基础变量。准确获取这些气象要素，对于数值天气预报初始场构建、极端天气监测、气候变化分析以及卫星遥感反演产品验证均具有重要意义。传统大气探测主要依赖探空、飞机观测、地基雷达、微波辐射计以及多源卫星遥感等手段。其中，探空气球能够提供较高垂直分辨率的原位观测，但其空间覆盖受站点分布限制，海洋、极地和高原等区域观测稀疏；被动卫星遥感具有较好的水平覆盖能力，但容易受到云雨、地表发射率和反演先验假设影响，垂直分辨率通常有限。因此，发展具有全球覆盖、高垂直分辨率和较强稳定性的观测与反演方法，一直是大气探测领域的重要方向。

GNSS 无线电掩星技术利用低轨卫星接收 GNSS 卫星发射的无线电信号。当信号穿越地球大气并到达低轨卫星接收机时，受大气折射率垂直变化影响，信号传播路径会发生弯曲，同时相位、频率和幅度也会发生变化。通过精密轨道和相位观测，可以反推出信号的弯曲角廓线，并进一步获得大气折射率以及温度、气压、水汽等变量。与传统观测方式相比，GNSS-RO 的优势在于：GNSS 星座与低轨卫星的组合可形成全球尺度观测，弥补海洋和偏远地区探测不足；掩星观测依赖相位延迟，长期稳定性好、仪器漂移小；无线电信号穿透云层的能力较强，具备全天候观测特性；垂直分辨率高，适合刻画对流层顶和平流层结构等关键层结特征。

然而，从 GNSS-RO 原始观测到最终气象要素产品并非直接过程。传统反演通常包含由相位到多普勒频移、由多普勒频移到弯曲角、由弯曲角到折射率、由折射率到温度和气压，以及在低层结合背景场进行湿大气反演等多个步骤。每一步都可能引入误差或依赖特定假设。例如，Abel 逆变换通常基于局地球对称假设，当水平梯度较强时会产生结构性误差；低对流层中折射率同时受干空气密度和水汽影响，仅依赖弯曲角难以唯一确定温度与湿度，形成常见的“温湿模糊”问题；一维变分方法依赖背景场和误差协方差设置，其结果受先验信息质量影响较大。因此，探索能够直接从弯曲角观测学习复杂非线性映射关系的智能反演方法，具有较强的研究意义。

近年来，深度学习方法在遥感反演、气象预测和科学数据建模中得到广泛应用。卷积神经网络、循环神经网络、Transformer 以及生成模型等方法能够从大规模数据中学习复杂的非线性特征表示，为传统物理反演提供了新的补充。与确定性回归模型相比，扩散模型作为近年来发展迅速的生成式模型，通过逐步加噪和反向去噪过程建模数据分布，具有训练稳定、生成质量高、可结合条件信息等特点。将条件扩散模型用于 GNSS-RO 大气剖面反演时，弯曲角廓线作为条件输入，温度、气压和湿度廓线作为生成目标，模型学习的是从观测空间到大气状态空间的概率映射。这一框架在实现端到端反演的同时，也为不确定性估计、多次采样和多源条件融合留下了接口。

基于上述背景，本文围绕”基于掩星数据的气象要素反演系统设计与实现”展开研究，重点探索条件扩散模型在 FY-3D GNOS 掩星数据三变量联合反演任务中的可行性。

## 1.2 国内外研究现状

GNSS-RO 技术自二十世纪末逐步应用于大气探测以来，已经形成较为成熟的传统反演体系。早期研究主要围绕 GPS/MET、CHAMP、COSMIC 等任务开展，证明了掩星观测在温度廓线反演和数值天气预报同化中的价值。传统 GNSS-RO 反演方法通常以几何光学或波动光学处理为基础，通过相位观测恢复弯曲角，再利用 Abel 逆变换获得折射率廓线。在较高高度层，大气水汽含量较低，可在静力平衡假设下反演干温度和气压；在低对流层，水汽影响显著，通常需要引入背景场，通过一维变分方法同时估计温度、气压和湿度。该类方法具有清晰的物理解释，是目前业务化产品的重要基础。

随着机器学习和深度学习的发展，越来越多研究开始尝试使用数据驱动方法改进大气反演。对于遥感反演任务，深度神经网络能够处理高维观测与目标变量之间的非线性关系，在温度反演、水汽反演、降水估计和云参数反演等场景中取得了较好效果。在 GNSS-RO 领域，已有研究尝试使用多层感知机、卷积网络或端到端模型从弯曲角、折射率或其他中间产品直接预测大气变量。这类方法可以减少传统流程中的中间误差传播，但也面临训练数据质量、物理约束不足和泛化能力等问题。

生成式模型的发展为大气反演提供了新的技术方向。去噪扩散概率模型最初在图像生成任务中取得优异效果，随后逐步扩展到语音、时间序列、医学影像、遥感影像和气象场生成等领域。扩散模型通过模拟数据逐步退化为高斯噪声的前向过程，并学习从噪声恢复数据的反向过程，能够对复杂分布进行稳定建模。在气象和遥感场景中，扩散模型已被用于降尺度、缺测补全、概率预报、云图生成以及多源遥感融合等任务。由于大气廓线本质上是一维连续物理场，具有明显的垂直结构和变量耦合关系，扩散模型有潜力学习其分布规律并结合观测条件生成合理剖面。

综合来看，传统 GNSS-RO 反演方法成熟可靠，但端到端智能反演仍处于探索阶段。确定性深度学习模型虽然具备处理非线性映射的能力，却较难表达反演不确定性。扩散模型在气象场景中已有初步应用，但针对 GNSS-RO 三变量大气廓线反演的研究尚不多见。本文尝试将条件扩散模型引入 FY-3D GNOS ATP+WAP 数据反演任务，从数据处理、模型设计、训练策略和系统实现等方面开展实验。

## 1.3 研究目标与主要内容

本文的总体目标是设计并实现一套基于 FY-3D GNOS 掩星数据的气象要素反演系统，利用条件扩散模型从弯曲角廓线端到端生成温度、气压和湿度垂直廓线，并通过实验验证该方法在多变量联合反演任务中的可行性。

围绕上述目标，本文主要研究内容包括以下几个方面。

第一，构建 ATP+WAP 配对数据处理流水线。本文以 FY-3D GNOS L2 ATP 产品中的优化弯曲角作为输入，以 WAP 产品中的温度、气压和比湿作为标签，通过文件名配对、质量控制、物理合理性检查、统一高度网格插值和标准化处理，构建适用于深度学习训练的数据集。

第二，设计条件扩散反演模型。本文采用 DDPM 框架，将大气廓线反演建模为条件生成任务。模型以带噪目标剖面、时间步和弯曲角条件作为输入，通过增强版一维 U-Net 预测噪声，并在反向采样过程中逐步生成温度、气压和湿度三变量剖面。

第三，优化多变量训练策略。针对湿度通道学习难度较高、不同变量尺度和损失贡献不平衡等问题，本文在训练中引入变量加权损失和湿度梯度约束，并将模型监控目标调整为湿度分量，以增强湿度反演能力。

第四，实现完整反演系统。系统包括数据处理、模型训练、推理采样、评估分析和 Streamlit 可视化展示等模块，能够支持从原始 NetCDF 数据到模型反演结果展示的完整流程。

第五，开展实验评估与结果分析。本文基于 2025 年 1 月至 6 月 FY-3D GNOS 数据构建数据集，评估温度、气压和湿度反演结果，并与旧基线进行对比，分析数据规模、损失设计和采样方法对结果的影响。

## 1.4 论文组织结构

本文共分为六章。第1章介绍研究背景、意义、国内外研究现状以及本文研究内容。第2章阐述 GNSS 无线电掩星基本原理、传统反演流程、去噪扩散概率模型和一维条件 U-Net 相关技术。第3章从需求分析、总体架构、数据处理、模型设计和评估体系等方面给出系统总体设计。第4章介绍系统各模块的具体实现，包括数据处理、模型训练、推理采样和可视化前端。第5章给出实验设置、主实验结果、对比实验和结果分析。第6章总结全文工作，归纳创新点与不足，并展望后续改进方向。

---

# 第2章 相关理论与技术基础

## 2.1 GNSS 无线电掩星原理

GNSS 无线电掩星是一种利用 GNSS 卫星与低轨卫星之间的相对运动进行大气探测的方法。当 GNSS 卫星从低轨卫星视线方向上升起或落下时，其发射的 L 波段无线电信号会掠过地球大气边缘。由于大气折射率随高度变化，信号传播路径不再是直线，而会发生弯曲并产生相位延迟。低轨卫星接收机记录信号的相位和幅度变化，结合卫星精密轨道信息，可以恢复信号传播路径上的弯曲角廓线。

在理想球对称大气假设下，弯曲角与折射率之间存在积分关系。通常以冲击参数表示射线距离地心的最近距离，用弯曲角描述信号传播方向相对于直线传播的偏转程度。通过 Abel 逆变换，可以由弯曲角廓线反演得到折射率廓线。大气折射率与温度、气压和水汽压之间存在经验关系，常用形式为：

$$
N = 77.6 \frac{P}{T} + 3.73 \times 10^5 \frac{e}{T^2}
$$

其中，$N$ 为折射率，$P$ 为气压，$T$ 为温度，$e$ 为水汽压。第一项通常称为干项，主要由干空气密度决定；第二项为湿项，与水汽含量密切相关。中高层大气水汽较少时，湿项可以忽略，反演相对稳定；低对流层水汽含量较高时，温度和湿度对折射率的贡献相互耦合，导致仅凭折射率难以唯一确定温湿结构。

传统 GNSS-RO 产品生产通常包括以下流程：首先由原始相位观测解算弯曲角；其次基于 Abel 逆变换获得折射率；然后在干大气假设或辅助背景场约束下反演温度、气压和湿度。该流程具有坚实的物理基础，但多步骤处理可能导致误差逐级累积。此外，球对称假设在强水平梯度、复杂天气系统和低层水汽丰富区域可能不完全成立。因此，本文尝试以弯曲角为条件输入，通过数据驱动模型直接学习其与目标气象要素之间的映射关系。

## 2.2 去噪扩散概率模型

去噪扩散概率模型是一类基于马尔可夫链的生成模型，其核心思想是先将真实数据逐步加入高斯噪声，使其最终接近标准正态分布；再训练神经网络学习反向去噪过程，从随机噪声中逐步恢复数据样本。

设真实大气廓线为 $x_0$，前向扩散过程在每个时间步向数据中加入高斯噪声：

$$
q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)
$$

其中，$\beta_t$ 为第 $t$ 步噪声调度参数。令 $\alpha_t=1-\beta_t$，$\bar{\alpha}_t=\prod_{s=1}^{t}\alpha_s$，则可以直接从 $x_0$ 采样任意时间步的带噪样本：

$$
x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon,\quad \epsilon \sim \mathcal{N}(0,I)
$$

训练阶段，模型接收带噪样本 $x_t$、时间步 $t$ 和条件输入 $c$，学习预测加入的噪声 $\epsilon$。本文中的条件输入 $c$ 为弯曲角廓线。模型训练目标采用简化噪声预测均方误差：

$$
L = \mathbb{E}_{x_0,t,\epsilon}\left[\|\epsilon - \epsilon_\theta(x_t,t,c)\|^2\right]
$$

推理阶段，从与目标廓线形状相同的高斯噪声开始，利用训练好的模型逐步预测并去除噪声，最终得到生成的大气廓线。本文采用 DDPM 完整采样流程，扩散步数设置为 1000，噪声调度采用线性调度，$\beta$ 从 $1\times 10^{-4}$ 增长到 $0.02$。由于完整 DDPM 采样需要较多迭代，本文也尝试使用 DDIM 进行加速采样，但实验发现当前配置下 50 步 DDIM 结果不稳定，因此主实验仍采用 DDPM 1000 步采样。

## 2.3 U-Net 架构与条件机制

U-Net 是一种典型的编码器-解码器网络结构，最初常用于图像分割任务。其核心特点是通过编码器逐步提取高层语义特征，通过解码器恢复空间分辨率，并利用跳跃连接将浅层细节特征传递到解码阶段。对于大气垂直廓线任务，输入和输出均为一维序列，因此本文采用一维卷积构建 U-Net 结构。

本文使用的增强版一维条件 U-Net 在标准编解码结构基础上，引入了时间步嵌入、条件编码器、交叉注意力和残差归一化等组件。

扩散模型在不同时间步面对不同程度的噪声污染，因此需要时间步信息来调整去噪策略。本文采用正弦位置编码生成时间嵌入，通过多层感知机映射到统一维度后注入残差卷积块。弯曲角廓线作为反演任务的观测条件，由条件编码器（一维卷积）映射到高维特征空间，为交叉注意力模块提供 Key 和 Value。

交叉注意力是实现条件注入的核心机制。主干网络当前特征作为 Query，弯曲角条件特征作为 Key 和 Value，计算形式为：

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

相比简单拼接，交叉注意力允许模型在不同高度位置动态关注弯曲角信息，更适合建模目标剖面与观测条件之间的非局部依赖关系。网络中还使用了残差连接和 GroupNorm 以稳定训练过程，其中 GroupNorm 对批量大小不敏感，适合 GPU 显存受限的场景。模型最终输出三通道序列，分别对应温度、气压和湿度的噪声预测。

## 2.4 本章小结

本章梳理了 GNSS-RO 反演原理、DDPM 生成框架和条件 U-Net 架构三方面的理论基础。传统反演的多步骤流程和湿度反演不确定性构成了本文采用数据驱动方法的出发点，DDPM 的条件生成能力和 U-Net 的多尺度特征提取能力则为系统设计提供了技术支撑。

---

# 第3章 系统总体设计

## 3.1 系统需求分析

本文系统的设计目标是围绕 FY-3D GNOS 掩星数据实现气象要素端到端反演。根据研究任务和工程实现需求，系统需求可分为功能需求与性能需求。

功能需求方面，系统需覆盖四项核心功能：（1）数据处理——读取 FY-3D GNOS L2 ATP 和 WAP NetCDF 文件，完成文件配对、变量提取、质量控制、插值、标准化和数据集划分；（2）模型训练——加载数据集，构建条件扩散模型，执行噪声采样、前向传播、损失计算、反向传播和模型保存；（3）推理反演——从测试样本弯曲角条件出发，通过 DDPM 反向采样生成温度、气压和湿度廓线；（4）评估与可视化——计算 RMSE、Bias 和相关系数等指标，并以图表形式展示预测剖面与标签剖面对比。

性能需求方面，系统应保证反演结果具有一定精度，尤其需要在气压通道保持高相关性，并尽可能提高温度和湿度反演质量。数据处理流程应能处理 2025 年上半年数万级 NetCDF 文件，避免一次性加载过多数据导致内存不足。训练流程应能够在单张 Tesla V100 GPU 上稳定运行，支持早停、梯度裁剪和训练日志记录。推理流程虽然采用 1000 步 DDPM 采样，速度相对较慢，但应保证结果稳定可用。

## 3.2 系统架构设计

系统采用分层架构设计，可分为数据层、模型层、训练评估层、推理服务层和展示层。

数据层负责原始数据读取与预处理。该层以 FY-3D GNOS ATP 和 WAP 文件为输入，输出标准化后的 `train_x.npy`、`train_y.npy`、`val_x.npy`、`val_y.npy`、`test_x.npy`、`test_y.npy` 以及统计量文件。数据层的核心模块为 `ro_retrieval/data/atp_wap_process.py`，其中 `ATPWAPProcessor` 类封装了 ATP/WAP 配对、物理检查和数据保存逻辑。

模型层负责条件扩散模型的定义。该层包含 `ro_retrieval/model/diffusion.py` 和 `ro_retrieval/model/unet.py`。前者实现扩散调度、前向加噪和 DDPM/DDIM 采样；后者实现条件 U-Net 和增强版条件 U-Net。主实验采用 `EnhancedConditionalUNet1D`，支持三变量联合输出。

训练评估层负责模型优化和性能计算。训练模块加载标准化数据集，随机采样扩散时间步，生成带噪目标廓线，并以噪声预测误差作为训练目标。评估模块计算逐样本和总体 RMSE、Bias、CC 等指标，并输出 JSON 报告和可视化图像。

推理服务层负责调用训练好的模型进行反演。给定弯曲角条件后，系统从随机噪声初始化目标廓线，通过 DDPM 反向去噪获得预测结果，并进行反标准化和必要后处理。

展示层基于 Streamlit 实现交互式页面，提供样本选择、结果展示和剖面对比功能。该层主要面向系统演示和答辩展示，使用户能够直观看到输入弯曲角与输出气象要素的关系。

## 3.3 数据处理流水线设计

本文数据来源为 FY-3D GNOS L2 ATP 和 WAP 产品。ATP 产品提供优化弯曲角、冲击参数、质量标志等信息；WAP 产品提供温度、气压和比湿等湿大气廓线。由于本文目标是从弯曲角直接反演温度、气压和湿度，因此将 ATP 作为输入源，将 WAP 作为监督标签源。

ATP 与 WAP 文件配对依赖文件名规则。两类产品对应同一掩星事件时，文件名中的 `_L2_ATP_` 和 `_L2_WAP_` 字段不同，其余部分基本一致。因此，系统通过将 WAP 文件名替换为对应 ATP 文件名完成配对。配对成功后，系统分别读取 ATP 中的 `Opt_Bend_ang` 和 `Opt_Impact_parm`，以及 WAP 中的 `Temp`、`Pres`、`Shum` 和 `MSL_alt`。

质量控制方面，系统仅保留 ATP 质量标志 `qc=100` 的样本，并对温度、气压、弯曲角和湿度进行物理范围检查，过滤异常样本。具体阈值设置见第4章。

由于原始廓线高度层不完全一致，本文将所有变量线性插值到 0–60 km 的统一高度网格（301 点）。弯曲角和气压动态范围跨越多个数量级，分别进行对数变换后再做 Z-Score 标准化：

$$
x_{\mathrm{BA}} = \log_{10}(|\mathrm{BA}| + 10^{-6}), \quad y_{\mathrm{P}} = \log_{10}(\max(P, 10^{-4}))
$$

湿度裁剪到非负范围后同样进行 Z-Score 标准化。数据集按 70%/15%/15% 划分为训练集、验证集和测试集。

本文最终处理结果如下：2025 年 1 月至 6 月数据中，成功配对 69,376 个 ATP+WAP 文件，经过质量控制和处理后获得 64,116 个有效样本，其中训练集 44,881 个，验证集 9,617 个，测试集 9,618 个。

## 3.4 模型设计

本文将 GNSS-RO 大气反演建模为条件生成任务。设弯曲角廓线为条件 $c$，目标气象要素廓线为 $x_0$，其中 $x_0$ 包含温度、气压和湿度三个通道。扩散模型通过学习条件分布 $p_\theta(x_0|c)$，从随机噪声中生成与弯曲角条件一致的大气剖面。

模型整体由扩散调度器和噪声预测网络组成。扩散调度器 `DiffusionSchedule` 负责生成线性 $\beta$ 序列，并预计算 $\alpha_t$、$\bar{\alpha}_t$、后验方差等采样所需参数。噪声预测网络采用 `EnhancedConditionalUNet1D`，输入为带噪目标廓线 $x_t$、时间步 $t$ 和条件弯曲角 $c$，输出为预测噪声 $\hat{\epsilon}$。

增强版条件 U-Net 的基础通道数为 64，时间嵌入维度为 128，采用 3 级编解码结构。编码器逐步提取大气廓线局部和全局特征，瓶颈层整合高层语义信息，解码器通过转置卷积恢复序列长度，并使用跳跃连接保留浅层细节。模型在编码器和瓶颈层引入交叉注意力，使弯曲角条件能够在不同尺度参与去噪过程。最终输出通道数为 3，对应温度、气压和湿度三变量噪声预测。当前模型可训练参数量约为 1,115,651，权重文件约 4.3 MB。

## 3.5 评估体系设计

为评价反演结果，本文采用 RMSE、Bias 和相关系数 CC 三类指标。由于不同变量的物理量纲和数值范围差异较大（温度以 K 为单位，气压经对数变换后为无量纲值，湿度以 g/kg 为单位），本文统一在 Z-Score 标准化空间下计算 RMSE 和 Bias，以便于跨变量横向比较。RMSE 衡量预测值与标签值之间的均方根误差；Bias 衡量预测结果的平均偏差，用于判断是否存在系统性高估或低估；CC 衡量预测廓线与真实廓线的形态相关程度，适合评价垂直结构的一致性。

RMSE 定义为：

$$
\mathrm{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
$$

Bias 定义为：

$$
\mathrm{Bias} = \frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i-y_i)
$$

相关系数定义为：

$$
\mathrm{CC} = \frac{\sum_i(\hat{y}_i-\bar{\hat{y}})(y_i-\bar{y})}{\sqrt{\sum_i(\hat{y}_i-\bar{\hat{y}})^2}\sqrt{\sum_i(y_i-\bar{y})^2}}
$$

除总体指标外，系统还支持逐高度层 RMSE 和 Bias 计算，用于分析模型在不同高度区域的误差分布。由于低对流层湿度变化剧烈，中高层弯曲角信号较弱，不同高度区域的误差特征可能存在明显差异，因此逐高度层分析对理解模型表现具有重要意义。

## 3.6 本章小结

本章确定了系统的分层架构和各模块职责。需要说明的是，数据处理流水线的设计直接影响训练数据质量，其中对数变换和 Z-Score 标准化的选择对后续模型收敛和评估指标的可比性至关重要，具体实现将在第4章展开。

---

# 第4章 系统实现

## 4.1 开发环境与技术栈

本文系统使用 Python 作为主要开发语言，核心深度学习框架为 PyTorch。数据处理依赖 NumPy、netCDF4、tqdm 等库，评估与可视化依赖 Matplotlib、JSON 和 Streamlit 等工具。模型训练主要在 Tesla V100-PCIE-32GB GPU 上完成，能够满足 64 批量大小和 1000 步扩散模型训练需求。

系统代码采用模块化组织方式。`ro_retrieval/data` 目录负责数据读取和预处理，`ro_retrieval/model` 目录负责模型结构和扩散过程，`ro_retrieval/training` 目录负责训练逻辑，`ro_retrieval/evaluation` 目录负责指标计算，`ro_retrieval/inference` 目录负责模型推理，`ro_retrieval/app` 目录负责 Streamlit 前端展示，`src` 目录则提供训练、评估和对比实验入口脚本。

## 4.2 数据处理实现

ATP+WAP 数据处理由 `ATPWAPProcessor` 类实现。该类在初始化时设置目标高度网格、质量控制阈值和随机种子。目标高度网格默认为 `np.linspace(0, 60, 301)`，即从 0 km 到 60 km 共 301 个高度层。

在文件读取阶段，处理器首先根据 WAP 文件名构造对应 ATP 文件名。若 ATP 文件不存在，则记为缺失配对；若文件存在，则分别读取 ATP 和 WAP 中的关键变量。ATP 文件中的 `qc` 属性用于初步质量筛选，`curv` 属性用于将冲击参数转换为海拔高度：

$$
h = a - r_c
$$

其中，$a$ 为冲击参数，$r_c$ 为曲率半径。WAP 文件中的 `MSL_alt` 直接作为温度、气压和湿度标签的高度坐标。

在插值阶段，系统先过滤 NaN 和无效值，对高度进行排序并去除重复高度，然后使用线性插值将变量映射到统一高度网格。如果有效点少于 10 个，或插值结果仍包含 NaN，则丢弃该样本。

在物理合理性检查阶段，系统依次检查温度范围、气压范围、弯曲角范围、湿度范围和气压单调性。气压单调性检查允许少量局部波动，但若气压随高度异常增加的点数过多，或局部增加幅度过大，则认为样本异常。

在保存阶段，系统计算弯曲角、温度、气压和湿度的统计量，对训练集、验证集和测试集分别标准化并保存为 `.npy` 文件，同时生成 `summary.json`，记录总文件数、配对数、失败数、最终样本数和数据集划分结果。该设计便于后续训练脚本直接加载标准化数组，也便于复现实验。

为解决大规模数据处理过程中的内存和中断问题，项目还实现了分块处理脚本，将数万文件拆分为多个数据块进行处理，支持中间结果缓存和断点续跑。该策略提高了 2025 年上半年 ATP+WAP 数据处理的稳定性。

## 4.3 模型训练实现

训练阶段，系统使用 PyTorch `DataLoader` 按批加载弯曲角条件和目标气象要素廓线。对于每个批次，首先随机采样时间步 $t$，再从标准正态分布采样噪声 $\epsilon$，利用扩散调度器生成带噪目标 $x_t$。模型输入为 $x_t$、$t$ 和弯曲角条件，输出为预测噪声 $\hat{\epsilon}$。

基础损失函数为噪声预测 MSE。由于本文为三变量联合反演任务，不同变量的学习难度和重要性不同，尤其湿度变量在低层变化强烈且样本分布不均，容易在总损失中被气压等较稳定通道掩盖。因此，最终训练配置采用变量加权损失：

$$
L_{\mathrm{var}} = w_T L_T + w_P L_P + w_Q L_Q
$$

其中，$w_T:w_P:w_Q = 1:1:4$，分别对应温度、气压和湿度通道。湿度通道权重提高后，模型能够获得更强的湿度学习信号。

此外，为增强湿度廓线的垂直连续性，系统引入湿度梯度约束。该约束通过惩罚湿度预测在高度方向上的异常梯度变化，使生成结果更符合大气湿度随高度变化的物理直觉。最终配置中湿度梯度约束权重为 0.05。

训练采用 AdamW 优化器，学习率为 $1\times 10^{-4}$，批量大小为 64，最大训练轮数为 50。为避免梯度爆炸，系统使用梯度裁剪；为防止过拟合，系统设置 Early Stopping，耐心值为 15。与中期阶段主要监控总验证损失不同，最终训练将监控目标调整为湿度分量验证损失，从而优先保存湿度表现较好的模型权重。

最终模型训练耗时约 21.9 分钟，最佳验证损失为 0.022709，最佳湿度监控指标为 0.009187。训练损失在前期快速下降，后期稳定收敛，说明当前数据规模和训练配置能够支持模型较稳定地学习三变量反演任务。

## 4.4 推理模块实现

推理阶段，系统首先读取测试样本弯曲角条件和训练好的模型权重，然后创建与目标输出形状一致的高斯随机噪声。对于每个反向时间步，模型根据当前带噪状态、时间步和条件输入预测噪声，扩散调度器根据 DDPM 后验公式更新到上一时间步，直至生成 $x_0$。

本文主实验采用 DDPM 1000 步完整采样。该方法采样速度相对较慢，但结果较稳定。项目中也实现了 DDIM 50 步加速采样，用于探索减少推理时间的可能性。实验发现，当前数据和模型配置下 DDIM 50 步采样结果不稳定，生成廓线容易发散，相关系数接近不可用水平。因此，本文实验结果以 DDPM 完整采样为主，DDIM 仅作为问题分析和后续优化方向。

生成的标准化预测结果需根据训练集统计量进行反标准化。对于气压通道，由于训练标签采用 $\log_{10}(P)$，反标准化后仍处于对数空间；如果需要物理单位气压值，可进一步进行指数变换。对于展示结果，系统可对预测廓线进行平滑处理，以减小采样带来的局部噪声波动。

## 4.5 评估与可视化实现

评估模块由 `ro_retrieval/evaluation/metrics.py` 实现，提供 RMSE、Bias、CC、MAE 以及逐高度层误差计算函数。`EvaluationReport` 类用于收集多样本评估结果，生成变量级统计摘要，并保存为 JSON 文件。系统还支持按 RMSE 对样本排序，以选取最佳、中位和最差案例进行可视化展示。

可视化前端基于 Streamlit 实现。页面主要包括样本选择、模型推理、剖面对比和指标展示等功能。用户可以选择测试样本并触发反演，系统展示温度、气压和湿度预测剖面与标签剖面对比图，同时可展示弯曲角输入曲线。该前端作为原型系统，可以支撑论文答辩和系统演示，展示从数据输入到结果输出的完整流程。

## 4.6 本章小结

本章给出了各模块的实现细节。实现过程中遇到的主要工程问题包括：大规模 NetCDF 文件处理的内存控制（通过分块处理解决）、多变量损失平衡（通过湿度加权和梯度约束解决）以及 DDIM 采样不稳定（最终回退到 DDPM 完整采样）。这些问题及其解决方案将在第5章实验中进一步验证。

---

# 第5章 实验与结果分析

## 5.1 实验设置

本文实验数据为 FY-3D GNOS 2025 年 1 月至 6 月 L2 ATP 与 WAP 配对产品。经过配对和质量控制后，最终获得 64,116 个有效样本。数据按 70%、15%、15% 随机划分为训练集、验证集和测试集，其中训练集 44,881 个样本，验证集 9,617 个样本，测试集 9,618 个样本。

模型采用 `EnhancedConditionalUNet1D`，基础通道数为 64，时间嵌入维度为 128，注意力头数为 4，输出通道数为 3。扩散步数为 1000，$\beta$ 线性范围为 $1\times 10^{-4}$ 至 0.02。训练批量大小为 64，学习率为 $1\times 10^{-4}$，训练轮数为 50，变量权重为 `[1, 1, 4]`，湿度梯度约束权重为 0.05，监控目标为湿度分量验证损失。

评估采用 DDPM 1000 步采样，对全部 9,618 个测试样本进行反演并统计指标。表中 RMSE 和 Bias 均在标准化空间下计算。

## 5.2 主实验结果

最终模型在全测试集上的评估结果如表 5-1 所示。

表 5-1 三变量反演主实验结果（标准化空间）

| 变量 | RMSE | Bias | CC |
| --- | ---: | ---: | ---: |
| 温度 | 0.6267 | 0.0105 | 0.7820 |
| 气压 | 0.0756 | 0.0099 | 0.9990 |
| 湿度 | 0.7996 | 0.0808 | 0.6960 |

气压通道表现最好，相关系数达到 0.9990，接近完美恢复。气压与弯曲角之间物理关联较强，加上对数变换使分布更加平滑，模型容易学习其变化规律。

温度通道相关系数为 0.7820。相比中期 ATP-only 阶段，引入 WAP 配对数据和更大样本规模后温度反演有明显改善。温度受大气层结、湿度耦合和不同高度信号强度影响，反演难度高于气压，但模型已能捕捉垂直廓线的主要变化趋势。

湿度通道相关系数为 0.6960，在三变量中最低。湿度集中在低对流层，垂直变化剧烈，且与温度、气压存在非线性耦合，是反演难度最大的变量。本文通过增加数据规模、提高湿度损失权重和加入梯度约束，使湿度反演达到初步可用水平，但仍有较大改进空间。

## 5.3 数据规模影响分析

为分析数据规模对模型性能的影响，本文将最终 2025 年上半年 ATP+WAP 数据实验与旧基线实验进行对比。旧基线使用较小规模 Q1 数据，最终实验则扩展至 H1 数据。对比结果如表 5-2 所示。

表 5-2 旧基线与最终实验结果对比

| 变量 | 指标 | 旧基线 | 最终实验 | 变化 |
| --- | --- | ---: | ---: | ---: |
| 温度 | RMSE | 0.6482 | 0.6267 | -3.3% |
| 温度 | CC | 0.7683 | 0.7820 | +1.8% |
| 气压 | RMSE | 0.0486 | 0.0756 | +55.6% |
| 气压 | CC | 0.9996 | 0.9990 | -0.06% |
| 湿度 | RMSE | 0.8690 | 0.7996 | -8.0% |
| 湿度 | CC | 0.5321 | 0.6960 | +30.8% |

扩充数据规模对湿度通道收益最为明显。湿度 CC 从 0.5321 提升至 0.6960，RMSE 下降 8.0%。湿度分布具有明显的空间和季节差异，较小数据集难以覆盖足够多样的大气状态；扩大到半年数据后，样本多样性增强，模型得以更充分地学习湿度垂直结构与弯曲角之间的统计关系。

温度通道 CC 从 0.7683 提升至 0.7820，RMSE 下降 3.3%，说明扩大数据规模对热力结构学习也有帮助。

气压通道 CC 仍保持在 0.9990 的高水平，但 RMSE 相比旧基线上升了 55.6%。这一退化与训练目标向湿度通道倾斜有关：最终配置将湿度权重提高到 4，并以湿度分量作为早停监控指标，模型优化重心偏向湿度，气压误差有所牺牲。不过气压相关系数几乎不变，垂直结构仍被准确恢复，该 trade-off 在当前阶段是可接受的。

## 5.4 采样方法对比与问题分析

扩散模型推理速度与采样步数密切相关。DDPM 1000 步采样结果稳定，但计算成本较高。为提高推理效率，本文实现了 DDIM 50 步加速采样。理论上，DDIM 可以通过非马尔可夫确定性采样减少步数，在图像生成任务中常用于加速推理。

然而，实验发现 DDIM 50 步在当前任务中效果不理想。可能的原因有多方面：大气廓线虽然是一维序列，但具有强物理约束和复杂垂直耦合关系，数据流形对大步长反向更新较为敏感；当前模型训练目标基于标准 DDPM 随机时间步噪声预测，并未针对少步采样做蒸馏或重加权优化；线性噪声调度在大跨度跳步时容易积累反向过程误差；此外，不同变量经过标准化和对数变换后对采样误差的响应不同，少步采样可能放大某些通道的不稳定性。

因此，本文正式实验采用 DDPM 完整采样作为主结果。后续可尝试逐步增加 DDIM 步数、采用 DPM-Solver 等高阶求解器、重新设计噪声调度，或通过一致性蒸馏训练专门的快速采样模型。

## 5.5 与传统方法和产品的对比讨论

本文使用 WAP 产品作为监督标签，因此严格意义上模型学习的是从 ATP 弯曲角到 WAP 反演产品的映射。WAP 产品本身通常基于传统物理反演和背景场信息生成，具有较高业务参考价值。与传统方法相比，本文方法的优势在于能够将多步骤反演过程压缩为端到端模型推理，并通过数据驱动方式学习复杂非线性关系。系统训练完成后，对单个样本的反演只需输入弯曲角廓线即可生成三变量结果，省去了传统流程中逐级转换的中间步骤。

不过，本文模型目前仍依赖 WAP 产品作为训练标签，因此其结果不能完全替代传统物理反演。模型性能上限受到标签质量和样本分布影响，且缺少显式物理约束可能导致某些极端样本或分布外样本生成不合理结果。此外，当前模型主要验证了与 WAP 标签的一致性，尚未充分评估其相对于独立探空、ERA5 或其他业务产品的真实性能。正式论文后续可增加与 CDAAC 或其他外部产品的对比分析，以更全面评价反演质量。

## 5.6 本章小结

本章给出了基于 2025 年上半年 FY-3D GNOS ATP+WAP 数据的实验结果。在 9,618 个测试样本上，温度、气压和湿度的平均相关系数分别为 0.7820、0.9990 和 0.6960。与旧基线相比，湿度 CC 提升 30.8%，验证了扩大数据规模和湿度加权策略的有效性。DDIM 加速采样在当前配置下效果不理想，有待后续改进。

---

# 第6章 总结与展望

## 6.1 工作总结

本文围绕“基于掩星数据的气象要素反演系统设计与实现”开展研究，设计并实现了一套基于条件扩散模型的 GNSS-RO 端到端气象要素反演系统。系统以 FY-3D GNOS L2 ATP 弯曲角廓线作为输入条件，以 WAP 湿大气产品中的温度、气压和湿度作为监督标签，构建了覆盖 2025 年上半年的 ATP+WAP 配对数据集。

在数据层面，本文实现了完整的数据处理流水线，包括 NetCDF 文件读取、ATP/WAP 文件配对、质量控制、物理合理性检查、统一高度网格插值、对数变换、Z-Score 标准化和数据集划分。最终获得 64,116 个有效样本，其中训练集 44,881 个，验证集 9,617 个，测试集 9,618 个。

在模型层面，本文采用 DDPM 作为生成式反演框架，设计了增强版一维条件 U-Net 噪声预测网络。模型引入正弦时间嵌入、残差卷积块、GroupNorm 和交叉注意力机制，使弯曲角条件能够在多尺度特征层参与去噪过程，并支持温度、气压和湿度三变量联合输出。

在训练策略方面，本文针对湿度通道反演难度较高的问题，引入变量加权损失和湿度梯度约束，将变量权重设置为 `[1,1,4]`，并将监控目标调整为湿度分量验证损失。该策略有效改善了湿度学习效果。

在实验结果方面，最终模型在全部 9,618 个测试样本上取得温度 CC=0.7820、气压 CC=0.9990、湿度 CC=0.6960 的结果。与旧基线相比，湿度 CC 由 0.5321 提升至 0.6960，湿度 RMSE 由 0.8690 降低至 0.7996，验证了 ATP+WAP 联合建模路线和数据规模扩展的有效性。

在系统实现方面，本文完成了从数据处理、模型训练、推理评估到 Streamlit 可视化前端的完整原型系统，具备一定的工程完整性和展示价值。

## 6.2 创新点

本文主要创新点体现在以下几个方面。

第一，将条件扩散模型引入 GNSS-RO 大气三变量联合反演任务。不同于传统多步骤物理反演和确定性深度回归模型，本文将弯曲角到气象要素的映射建模为条件生成过程，探索了 DDPM 在一维大气廓线反演中的可行性。

第二，设计了面向掩星弯曲角条件的一维交叉注意力 U-Net。模型通过正弦时间嵌入表示扩散时间步，通过条件编码器提取弯曲角特征，并在多尺度编码层和瓶颈层使用交叉注意力注入条件信息，提高了条件约束能力。

第三，构建了 FY-3D GNOS ATP+WAP 大规模配对数据处理流程。本文从 2025 年上半年数据中获得 64,116 个有效样本，实现了从 ATP 弯曲角到 WAP 温度、气压和湿度标签的配对建模，为后续类似研究提供了工程参考。

第四，提出了面向湿度通道的加权训练策略。通过湿度加权损失、湿度梯度约束和湿度监控目标设置，模型湿度反演性能得到显著提升，缓解了三变量联合训练中湿度学习不足的问题。

## 6.3 不足与展望

本文工作仍存在以下不足，也是后续改进的方向。

在评估方面，当前主结果已基于全部 9,618 个测试样本统计，但评估维度仍可丰富——例如按月份、纬度带或高度层分别统计，以揭示模型在不同条件下的表现差异。同时，当前模型以 WAP 产品为标签，缺少与独立观测资料（探空、ERA5 再分析等）的交叉验证，后续应引入外部数据进一步检验反演结果的可靠性。

在模型层面，物理约束仍然较弱。当前方法主要依赖数据驱动学习，虽引入了湿度梯度约束，但尚未显式加入静力平衡、折射率一致性、湿度非负性和气压单调性等约束。设计物理一致性损失函数是一个值得探索的方向。

在推理效率方面，DDPM 1000 步采样速度较慢，不适合大规模业务应用。后续可尝试改进噪声调度、使用 DPM-Solver 等高阶求解器、渐进蒸馏或一致性模型来提高推理速度。

在输入信息方面，本文仅使用弯曲角廓线作为条件。如果能融合经纬度、时间、掩星类型、背景场等辅助信息，模型有望更好地处理不同区域和天气条件下的大气状态差异。

## 6.4 本章小结

本文验证了条件扩散模型用于 FY-3D GNOS 掩星数据三变量大气剖面反演的可行性，其中气压反演效果优异，温度反演可用，湿度反演仍有较大改进空间。物理约束引入、快速采样适配和独立数据验证是后续最值得投入的三个方向。

---

# 参考文献

说明：以下为初稿参考文献清单，后续需按学校要求统一 GB/T 7714 格式，并核对作者、题名、期刊、卷期页码和 DOI。

[1] Kursinski E R, Hajj G A, Schofield J T, et al. Observing Earth's atmosphere with radio occultation measurements using the Global Positioning System[J]. Journal of Geophysical Research: Atmospheres, 1997.

[2] Rocken C, Anthes R, Exner M, et al. Analysis and validation of GPS/MET data in the neutral atmosphere[J]. Journal of Geophysical Research: Atmospheres, 1997.

[3] Anthes R A. Exploring Earth's atmosphere with radio occultation: contributions to weather, climate and space weather[J]. Atmospheric Measurement Techniques, 2011.

[4] Healy S B, Thépaut J N. Assimilation experiments with CHAMP GPS radio occultation measurements[J]. Quarterly Journal of the Royal Meteorological Society, 2006.

[5] Hajj G A, Kursinski E R, Romans L J, et al. A technical description of atmospheric sounding by GPS occultation[J]. Journal of Atmospheric and Solar-Terrestrial Physics, 2002.

[6] Kuo Y H, Sokolovskiy S V, Anthes R A, et al. Assimilation of GPS radio occultation data for numerical weather prediction[J]. Terrestrial, Atmospheric and Oceanic Sciences, 2000.

[7] Ho J, Jain A, Abbeel P. Denoising diffusion probabilistic models[C]. Advances in Neural Information Processing Systems, 2020.

[8] Song J, Meng C, Ermon S. Denoising diffusion implicit models[C]. International Conference on Learning Representations, 2021.

[9] Sohl-Dickstein J, Weiss E, Maheswaranathan N, et al. Deep unsupervised learning using nonequilibrium thermodynamics[C]. International Conference on Machine Learning, 2015.

[10] Nichol A Q, Dhariwal P. Improved denoising diffusion probabilistic models[C]. International Conference on Machine Learning, 2021.

[11] Ronneberger O, Fischer P, Brox T. U-Net: Convolutional networks for biomedical image segmentation[C]. Medical Image Computing and Computer-Assisted Intervention, 2015.

[12] Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need[C]. Advances in Neural Information Processing Systems, 2017.

[13] Wu Y, He K. Group normalization[C]. European Conference on Computer Vision, 2018.

[14] Rombach R, Blattmann A, Lorenz D, et al. High-resolution image synthesis with latent diffusion models[C]. IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022.

[15] Kingma D P, Ba J. Adam: A method for stochastic optimization[C]. International Conference on Learning Representations, 2015.

[16] Goodfellow I, Bengio Y, Courville A. Deep Learning[M]. Cambridge: MIT Press, 2016.

[17] Ware R, Exner M, Feng D, et al. GPS sounding of the atmosphere from low Earth orbit: preliminary results[J]. Bulletin of the American Meteorological Society, 1996.

[18] Schreiner W S, Sokolovskiy S, Hunt D, et al. Quality assessment of COSMIC/FORMOSAT-3 GPS radio occultation data[J]. Atmospheric Measurement Techniques, 2011.

[19] Liou Y A, Pavelyev A G, Liu S F, et al. FORMOSAT-3/COSMIC GPS radio occultation mission: preliminary results[J]. IEEE Transactions on Geoscience and Remote Sensing, 2007.

[20] 中国气象局国家卫星气象中心. 风云三号卫星 GNOS 产品相关资料[EB/OL]. 后续需补充访问地址和访问日期。

---

# 致谢

本论文工作从选题、系统设计、数据处理到实验分析，得到了指导老师和同学们的帮助与支持。在此，首先感谢指导老师在毕业设计过程中给予的方向指导和修改建议，使本人能够逐步明确研究问题并完善技术路线。感谢实验室和课程学习过程中各位老师对计算机基础、深度学习和遥感应用知识的讲授，为本文工作提供了必要的理论基础。

同时，感谢项目开发过程中同学和朋友在环境配置、论文写作和答辩准备方面提供的帮助。感谢开源社区提供的 PyTorch、NumPy、Streamlit 等工具，使本文系统能够较高效地完成实现与验证。最后，感谢家人在本科阶段学习生活中给予的理解和支持。

---

# 附录

## 附录 A 系统核心代码说明

建议后续在正式论文中选取以下核心代码片段或流程图放入附录：

1. `ro_retrieval/model/unet.py` 中 `EnhancedConditionalUNet1D` 的网络结构定义。
2. `ro_retrieval/model/diffusion.py` 中 `DiffusionSchedule` 与 `ddpm_sample` 的核心实现。
3. `ro_retrieval/data/atp_wap_process.py` 中 ATP+WAP 配对和质量控制逻辑。

## 附录 B 数据处理关键流程

ATP+WAP 数据处理主要流程如下：

1. 遍历 WAP 文件并构造对应 ATP 文件名。
2. 读取 ATP 弯曲角、冲击参数和质量标志。
3. 读取 WAP 温度、气压、湿度和高度。
4. 进行质量控制和物理范围检查。
5. 插值到 0-60 km、301 点统一高度网格。
6. 对弯曲角和气压进行对数变换。
7. 对输入和标签进行 Z-Score 标准化。
8. 按 70/15/15 比例划分训练集、验证集和测试集。

## 附录 C 待补充图表清单

1. 系统总体架构图。
2. ATP+WAP 数据处理流程图。
3. EnhancedConditionalUNet1D 网络结构图。
4. DDPM 训练与采样流程图。
5. 训练损失曲线。
6. 温度、气压、湿度典型样本剖面对比图。
7. 逐高度层 RMSE/Bias 曲线。
8. Streamlit 系统界面截图。
