"""
全局配置文件
============
集中管理所有超参数与路径配置
"""

import os
import torch

# ==================== 路径配置 ====================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "Data")
PROCESSED_DIR = os.path.join(DATA_DIR, "Processed")
SAMPLE_DIR = os.path.join(DATA_DIR, "Sample")
MODEL_DIR = os.path.join(PROJECT_ROOT, "checkpoints")  # 模型权重目录
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")     # 输出目录
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")            # 训练日志
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")      # 图片输出
EVAL_DIR = os.path.join(OUTPUT_DIR, "evaluation")      # 评估结果

# ==================== 设备 ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== 数据处理参数 ====================
STD_HEIGHT = 301              # 标准化高度层数 (0~60km, 301个点)
HEIGHT_MIN = 0.0              # 最低高度 (km)
HEIGHT_MAX = 60.0             # 最高高度 (km)
MIN_PENETRATION_HEIGHT = 0.5  # 最小穿透高度 QC 阈值 (km)
MIN_VALID_POINTS = 10         # 最低有效数据点数
TRAIN_RATIO = 0.7             # 训练集比例
VAL_RATIO = 0.15              # 验证集比例
TEST_RATIO = 0.15             # 测试集比例

# ==================== 变量名映射 ====================
# 输出变量: 温度(K), 压力(hPa), 比湿(kg/kg)
OUTPUT_VARIABLES = ["temperature", "pressure", "humidity"]
NUM_OUTPUT_VARS = len(OUTPUT_VARIABLES)  # 3

# ==================== 模型超参数 ====================
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 1e-4

# 扩散模型参数
TIMESTEPS = 1000
BETA_START = 1e-4
BETA_END = 0.02

# DDIM 参数
DDIM_STEPS = 50
DDIM_ETA = 0.0

# U-Net 参数
UNET_BASE_DIM = 64           # 基础通道数 (升级)
UNET_USE_CROSS_ATTENTION = True  # 是否使用交叉注意力

# ==================== 评估参数 ====================
TEST_SAMPLES = 50
SAVGOL_WINDOW = 31
SAVGOL_POLYORDER = 3
