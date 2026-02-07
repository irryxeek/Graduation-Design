"""
RO-Retrieval: 基于条件扩散模型的掩星大气廓线反演系统
=======================================================

模块结构:
    - ro_retrieval.data       : 数据处理、质量控制、ERA5匹配
    - ro_retrieval.model      : 条件扩散模型 (U-Net + 交叉注意力)
    - ro_retrieval.evaluation : 评估指标 (RMSE / Bias / CC)
    - ro_retrieval.inference  : 推理管线 (DDPM / DDIM)

作者: Graduation Design Project
"""

__version__ = "1.0.0"
__all__ = ["data", "model", "evaluation", "inference"]
