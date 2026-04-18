"""
Streamlit 交互式掩星反演可视化应用
===================================
基于条件扩散模型的 GNSS-RO 大气剖面端到端反演系统
"""

import os
import sys
import json
import io
import csv
import warnings
from pathlib import Path
import torch
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal import savgol_filter

# ─────────────────── 中文字体 & 绘图风格 ───────────────────
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings(
    "ignore",
    message=r".*Glyph.*missing from current font.*",
    category=UserWarning,
)

# 项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from ro_retrieval.config import (
    DEVICE, TIMESTEPS,
    SAVGOL_WINDOW, SAVGOL_POLYORDER,
)
from ro_retrieval.model.unet import ConditionalUNet1D, EnhancedConditionalUNet1D
from ro_retrieval.model.diffusion import DiffusionSchedule, ddpm_sample, ddim_sample
from ro_retrieval.evaluation.metrics import evaluate_profile
from ro_retrieval.stats_utils import canonicalize_stats, load_stats_from_dir

# ─────────────────── 颜色系统 ───────────────────
COLORS = {
    'bg_dark':     '#0a0e1a',
    'bg_card':     '#111827',
    'bg_surface':  '#1a2035',
    'border':      '#2a3555',
    'border_glow': '#3b82f6',
    'text':        '#e2e8f0',
    'text_dim':    '#94a3b8',
    'text_muted':  '#64748b',
    'accent_blue': '#3b82f6',
    'accent_cyan': '#06b6d4',
    'accent_teal': '#14b8a6',
    'temp_color':  '#f59e0b',
    'pres_color':  '#3b82f6',
    'hum_color':   '#10b981',
    'truth_color': '#e2e8f0',
    'pred_color':  '#f43f5e',
    'good':        '#10b981',
    'warn':        '#f59e0b',
    'bad':         '#ef4444',
}

VAR_META = [
    {'name': '温度', 'unit': 'K',    'color': COLORS['temp_color'], 'icon': '🌡'},
    {'name': '气压', 'unit': 'hPa',  'color': COLORS['pres_color'], 'icon': '📊'},
    {'name': '湿度', 'unit': 'g/kg', 'color': COLORS['hum_color'],  'icon': '💧'},
]

DATASET_PREFERENCE = [
    "Processed_ATP_WAP_2025",
    "Processed_ATP_WAP",
    "Processed_ATP_Q1",
    "Processed_ATP_Merged",
    "Processed_ATP",
    "Processed",
]

# ─────────────────── 页面配置 ───────────────────
st.set_page_config(
    page_title="GNSS-RO 大气剖面反演系统",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────── 自定义样式 ───────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@300;400;500;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── 全局基底 ── */
.stApp {
    background: linear-gradient(165deg, #0a0e1a 0%, #0f1729 40%, #111d35 100%);
    color: #e2e8f0;
    font-family: 'Noto Sans SC', sans-serif;
}

/* ── 侧栏 ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1321 0%, #111827 100%);
    border-right: 1px solid #1e293b;
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown span,
section[data-testid="stSidebar"] label {
    color: #cbd5e1 !important;
    font-family: 'Noto Sans SC', sans-serif;
}
section[data-testid="stSidebar"] .stSelectbox > div > div,
section[data-testid="stSidebar"] .stRadio > div {
    background-color: #1a2035;
    border-color: #2a3555;
    border-radius: 8px;
}

/* ── 主标题区 ── */
.hero-title {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #60a5fa, #06b6d4, #14b8a6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.02em;
    margin-bottom: 0.2rem;
    font-family: 'Noto Sans SC', sans-serif;
}
.hero-subtitle {
    font-size: 0.95rem;
    color: #64748b;
    font-weight: 300;
    letter-spacing: 0.05em;
    margin-bottom: 1.5rem;
    font-family: 'Noto Sans SC', sans-serif;
}

/* ── 卡片 ── */
.card {
    background: linear-gradient(145deg, #111827 0%, #0f1524 100%);
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 1.25rem;
    margin-bottom: 1rem;
    transition: border-color 0.3s ease;
}
.card:hover { border-color: #2a3a5c; }

.card-header {
    font-size: 0.75rem;
    font-weight: 500;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
}

/* ── 指标卡片 ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.75rem;
}
.metric-card {
    background: #0d1321;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
    transition: all 0.3s ease;
}
.metric-card:hover {
    border-color: #3b82f6;
    box-shadow: 0 0 20px rgba(59, 130, 246, 0.08);
}
.metric-label {
    font-size: 0.7rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: 0.35rem;
    font-family: 'JetBrains Mono', monospace;
}
.metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: -0.02em;
}
.metric-good { color: #10b981; }
.metric-warn { color: #f59e0b; }
.metric-bad  { color: #ef4444; }
.metric-neutral { color: #94a3b8; }

/* ── 变量标签 ── */
.var-tag {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 500;
    border: 1px solid;
    font-family: 'Noto Sans SC', sans-serif;
}

/* ── 分隔线 ── */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #2a3555, transparent);
    margin: 1.5rem 0;
}

/* ── 状态指示 ── */
.status-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 6px;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

/* ── 隐藏 Streamlit 默认装饰 ── */
header[data-testid="stHeader"] { background: transparent; }
.stDeployButton { display: none; }
div[data-testid="stDecoration"] { display: none; }

/* ── 按钮 ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
    border: 1px solid #3b82f6;
    color: white;
    border-radius: 8px;
    font-weight: 500;
    transition: all 0.3s ease;
    font-family: 'Noto Sans SC', sans-serif;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 0 25px rgba(59, 130, 246, 0.3);
    border-color: #60a5fa;
}

/* ── Streamlit metric 覆写 ── */
div[data-testid="stMetric"] {
    background: #0d1321;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 0.75rem 1rem;
}
div[data-testid="stMetric"] label {
    color: #64748b !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 0.12em;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.5rem !important;
    color: #e2e8f0 !important;
}

/* ── 数据信息栏 ── */
.info-bar {
    display: flex;
    gap: 2rem;
    padding: 0.75rem 1.25rem;
    background: rgba(59, 130, 246, 0.06);
    border: 1px solid rgba(59, 130, 246, 0.15);
    border-radius: 8px;
    margin-bottom: 1rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: #94a3b8;
    flex-wrap: wrap;
}
.info-item { display: flex; align-items: center; gap: 6px; }
.info-val { color: #e2e8f0; font-weight: 500; }

/* ── 节标题 ── */
.section-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #e2e8f0;
    margin: 1.5rem 0 0.75rem 0;
    display: flex;
    align-items: center;
    gap: 10px;
    font-family: 'Noto Sans SC', sans-serif;
}
.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, #2a3555, transparent);
}

/* ── 图表容器 ── */
.plot-container {
    background: #0d1321;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 0.5rem;
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────── 绘图配置 ───────────────────
def get_plot_style():
    """返回统一的深色科学绘图参数"""
    return {
        'figure.facecolor':  '#0d1321',
        'axes.facecolor':    '#0d1321',
        'axes.edgecolor':    '#2a3555',
        'axes.labelcolor':   '#94a3b8',
        'text.color':        '#94a3b8',
        'xtick.color':       '#64748b',
        'ytick.color':       '#64748b',
        'grid.color':        '#1e293b',
        'grid.alpha':        0.6,
        'axes.grid':         True,
        'font.size':         10,
        'axes.titlesize':    11,
        'axes.titleweight':  '500',
        'legend.facecolor':  '#111827',
        'legend.edgecolor':  '#2a3555',
        'legend.fontsize':   9,
    }


def cc_quality_class(cc):
    """根据 CC 值返回质量等级"""
    cc = abs(cc)
    if cc >= 0.95:  return 'metric-good', '优秀'
    if cc >= 0.80:  return 'metric-good', '良好'
    if cc >= 0.60:  return 'metric-warn', '中等'
    return 'metric-bad', '待改进'


# ─────────────────── 数据/模型加载 ───────────────────
def discover_datasets(require_training_data=True):
    """扫描可用数据目录，支持仅统计量来源或完整数据集。"""
    dataset_root = Path(PROJECT_ROOT) / "Data"
    if not dataset_root.is_dir():
        return []

    discovered = []
    for child in dataset_root.iterdir():
        if not child.is_dir():
            continue
        train_x = child / "train_x.npy"
        train_y = child / "train_y.npy"
        stats_path = child / "stats.npy"
        if require_training_data and (not train_x.exists() or not train_y.exists()):
            continue
        if not require_training_data and not (stats_path.exists() or (train_x.exists() and train_y.exists())):
            continue

        summary_path = child / "summary.json"
        summary = None
        if summary_path.exists():
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    summary = json.load(f)
            except (OSError, json.JSONDecodeError):
                summary = None

        discovered.append({
            "name": child.name,
            "path": str(child),
            "summary": summary,
            "has_training_data": train_x.exists() and train_y.exists(),
            "has_stats": stats_path.exists(),
        })

    def dataset_rank(item):
        try:
            rank = DATASET_PREFERENCE.index(item["name"])
        except ValueError:
            rank = len(DATASET_PREFERENCE)
        return (rank, item["name"])

    return sorted(discovered, key=dataset_rank)


@st.cache_resource
def load_stats_bundle(data_dir):
    """仅加载统计量与数据摘要，用于上传推理。"""
    data_path = Path(data_dir)
    stats_path = data_path / "stats.npy"
    summary_path = data_path / "summary.json"

    summary = None
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

    if stats_path.exists():
        stats = canonicalize_stats(np.load(stats_path, allow_pickle=True).item())
    else:
        result = load_data(data_dir)
        if result is None:
            return None
        _, stats, summary_from_data, heights = result
        if summary is None:
            summary = summary_from_data
        return stats, summary, heights

    heights = np.asarray(
        stats.get("target_heights", np.linspace(0, 60, 301)),
        dtype=np.float32,
    )
    return stats, summary, heights


@st.cache_resource
def load_data(data_dir):
    """加载指定数据目录及其统计量。"""
    data_path = Path(data_dir)
    data = {}
    for split in ['train', 'val', 'test']:
        xp = data_path / f"{split}_x.npy"
        yp = data_path / f"{split}_y.npy"
        if xp.exists() and yp.exists():
            data[split] = {
                'x': np.load(xp).astype(np.float32),
                'y': np.load(yp).astype(np.float32),
            }
    if 'train' not in data:
        return None

    stats = load_stats_from_dir(
        str(data_path),
        x_fallback=data['train']['x'],
        y_fallback=data['train']['y'],
    )

    summary = None
    summary_path = data_path / "summary.json"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

    heights = np.asarray(
        stats.get("target_heights", np.linspace(0, 60, data["train"]["x"].shape[-1])),
        dtype=np.float32,
    )
    return data, stats, summary, heights


@st.cache_resource
def load_model(path, model_type, out_ch):
    """加载模型权重, 自动检测架构"""
    state_dict = torch.load(path, map_location=DEVICE, weights_only=True)
    detected = "enhanced" if any(k.startswith("time_embed.") for k in state_dict) else "legacy"
    if model_type == "auto":
        model_type = detected

    if model_type == "enhanced":
        m = EnhancedConditionalUNet1D(
            in_channels=out_ch, cond_channels=1, out_channels=out_ch,
            use_cross_attention=True,
        )
    else:
        m = ConditionalUNet1D(in_channels=out_ch, cond_channels=1, out_channels=out_ch)

    m.load_state_dict(state_dict)
    m.to(DEVICE).eval()
    return m


def find_model_files():
    """扫描模型权重，优先推荐最新 best 权重。"""
    candidates = []
    search_roots = [
        Path(PROJECT_ROOT),
        Path(PROJECT_ROOT) / 'checkpoints',
        Path(PROJECT_ROOT) / 'experiments',
    ]

    for root in search_roots:
        if not root.is_dir():
            continue

        if root.name == "experiments":
            for path in root.glob("*/*.pth"):
                candidates.append(path)
        else:
            for path in root.glob("*.pth"):
                candidates.append(path)

    unique_candidates = []
    seen = set()
    for path in candidates:
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_candidates.append(path)

    def model_rank(path):
        name = path.name
        path_str = str(path)
        best_rank = 0 if "best" in name else 1
        atp_wap_2025_rank = 0 if "atp_wap_2025" in path_str else 1
        return (
            best_rank,
            atp_wap_2025_rank,
            -path.stat().st_mtime,
            path_str,
        )

    return [str(path) for path in sorted(unique_candidates, key=model_rank)]


def load_eval_report():
    """加载最近更新的评估报告。"""
    exp_dir = os.path.join(PROJECT_ROOT, 'experiments')
    if not os.path.isdir(exp_dir):
        return None

    reports = []
    for name in os.listdir(exp_dir):
        jp = os.path.join(exp_dir, name, 'evaluation_report.json')
        if os.path.exists(jp):
            reports.append((os.path.getmtime(jp), name, jp))

    for _, name, jp in sorted(reports, reverse=True):
        with open(jp, encoding="utf-8") as f:
            return json.load(f), name
    return None


def get_temperature_stats(stats):
    """单通道温度输出使用温度通道统计量。"""
    y_mean = np.asarray(stats["y_mean"], dtype=np.float32)
    y_std = np.asarray(stats["y_std"], dtype=np.float32)
    if y_mean.ndim == 0:
        return y_mean, y_std
    return np.float32(y_mean[0]), np.float32(y_std[0])


def denormalize_prediction(pred, stats, out_ch):
    """根据输出通道数做反标准化。"""
    pred = pred.squeeze(0).cpu().numpy()

    if pred.ndim == 1 or out_ch == 1:
        temp_mean, temp_std = get_temperature_stats(stats)
        return pred.reshape(-1) * temp_std + temp_mean

    y_mean = np.asarray(stats["y_mean"], dtype=np.float32)
    y_std = np.asarray(stats["y_std"], dtype=np.float32)
    return pred * y_std[:, None] + y_mean[:, None]


def denormalize_input_profiles(x_array, stats):
    """将标准化后的弯曲角恢复到展示空间。"""
    x_mean = np.asarray(stats["x_mean"], dtype=np.float32)
    x_std = np.asarray(stats["x_std"], dtype=np.float32)
    return x_array.astype(np.float32) * x_std + x_mean


def normalize_input_profiles(x_array, stats):
    """使用训练统计量标准化输入弯曲角。"""
    x_mean = np.asarray(stats["x_mean"], dtype=np.float32)
    x_std = np.asarray(stats["x_std"], dtype=np.float32)
    return (x_array.astype(np.float32) - x_mean) / (x_std + 1e-8)


def denormalize_target_profiles(y_array, stats):
    """将标准化标签恢复到展示空间。"""
    arr = np.asarray(y_array, dtype=np.float32)
    if stats.get("stats_space") == "normalized":
        return arr

    if arr.ndim == 1:
        temp_mean, temp_std = get_temperature_stats(stats)
        return arr * temp_std + temp_mean

    y_mean = np.asarray(stats["y_mean"], dtype=np.float32)
    y_std = np.asarray(stats["y_std"], dtype=np.float32)
    if arr.ndim == 2:
        channels = min(arr.shape[0], y_mean.shape[0])
        return arr[:channels] * y_std[:channels, None] + y_mean[:channels, None]
    if arr.ndim == 3:
        channels = min(arr.shape[1], y_mean.shape[0])
        return arr[:, :channels] * y_std[:channels][None, :, None] + y_mean[:channels][None, :, None]
    raise ValueError(f"不支持的标签维度 {arr.ndim}")


def ensure_x_shape(array):
    """标准化输入形状为 (N, 301)。"""
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 1:
        if arr.shape[0] != 301:
            raise ValueError(f"输入长度应为 301，当前为 {arr.shape[0]}")
        arr = arr[None, :]
    elif arr.ndim == 2:
        if arr.shape[1] != 301:
            raise ValueError(f"输入形状应为 (N, 301)，当前为 {arr.shape}")
    else:
        raise ValueError(f"不支持的输入维度 {arr.ndim}，请上传 (301,) 或 (N, 301)")
    return arr


def ensure_y_shape(array):
    """标准化标签形状为 (N, C, 301)。"""
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 1:
        if arr.shape[0] != 301:
            raise ValueError(f"单样本标签长度应为 301，当前为 {arr.shape[0]}")
        arr = arr[None, None, :]
    elif arr.ndim == 2:
        if arr.shape[1] != 301:
            raise ValueError(f"标签形状应为 (N, 301) 或 (N, C, 301)，当前为 {arr.shape}")
        arr = arr[:, None, :]
    elif arr.ndim == 3:
        if arr.shape[2] != 301:
            raise ValueError(f"标签最后一维应为 301，当前为 {arr.shape}")
    else:
        raise ValueError(f"不支持的标签维度 {arr.ndim}，请上传 (N, 301) 或 (N, C, 301)")
    return arr


def load_array_from_upload(uploaded_file):
    """从上传文件读取数组，支持 npy / npz / csv。"""
    if uploaded_file is None:
        return None, None

    suffix = Path(uploaded_file.name).suffix.lower()
    raw = uploaded_file.getvalue()

    if suffix == ".npy":
        return np.load(io.BytesIO(raw), allow_pickle=False), uploaded_file.name

    if suffix == ".npz":
        with np.load(io.BytesIO(raw), allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        return arrays, uploaded_file.name

    if suffix == ".csv":
        text = raw.decode("utf-8-sig")
        reader = csv.reader(io.StringIO(text))
        rows = []
        for row in reader:
            if not row:
                continue
            try:
                rows.append([float(v) for v in row])
            except ValueError:
                if rows:
                    raise ValueError("CSV 中包含无法解析的非数值内容")
                continue
        return np.asarray(rows, dtype=np.float32), uploaded_file.name

    raise ValueError(f"不支持的文件格式: {uploaded_file.name}")


def extract_upload_arrays(x_payload, y_payload=None):
    """从上传内容提取 x/y 数组。"""
    x_array = None
    y_array = None

    if isinstance(x_payload, dict):
        for key in ("x", "X", "inputs", "input", "features"):
            if key in x_payload:
                x_array = x_payload[key]
                break
        if x_array is None and len(x_payload) == 1:
            x_array = next(iter(x_payload.values()))
        if x_array is None:
            raise ValueError("上传的 npz 未找到 `x` 数组")
        for key in ("y", "Y", "labels", "targets", "target"):
            if key in x_payload:
                y_array = x_payload[key]
                break
    else:
        x_array = x_payload

    if y_payload is not None:
        if isinstance(y_payload, dict):
            for key in ("y", "Y", "labels", "targets", "target"):
                if key in y_payload:
                    y_array = y_payload[key]
                    break
            if y_array is None and len(y_payload) == 1:
                y_array = next(iter(y_payload.values()))
        else:
            y_array = y_payload

    x_array = ensure_x_shape(x_array)
    if y_array is not None:
        y_array = np.asarray(y_array, dtype=np.float32)
        if y_array.ndim == 2 and y_array.shape[1] == 301 and x_array.shape[0] == 1 and y_array.shape[0] in (1, 3):
            y_array = y_array[None, :, :]
        y_array = ensure_y_shape(y_array)
        if len(y_array) != len(x_array):
            raise ValueError(f"x/y 样本数不一致: {len(x_array)} vs {len(y_array)}")
    return x_array, y_array


def run_uploaded_inference(model, x_array, stats, sampler, schedule, out_ch, ddim_steps, smooth):
    """对上传的样本逐条执行推理。"""
    preds = []
    normalized_x = normalize_input_profiles(x_array, stats)
    total = len(normalized_x)

    progress_bar = st.progress(0, text="准备开始批量分析...")
    status = st.empty()

    for idx, x_norm in enumerate(normalized_x):
        cond = torch.tensor(x_norm).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            if sampler == "DDIM":
                gen = ddim_sample(
                    model,
                    cond,
                    shape=(1, out_ch, 301),
                    schedule=schedule,
                    ddim_steps=ddim_steps,
                )
            else:
                gen = ddpm_sample(
                    model,
                    cond,
                    shape=(1, out_ch, 301),
                    schedule=schedule,
                )

        pred_np = denormalize_prediction(gen, stats, out_ch)
        if smooth:
            if pred_np.ndim == 1:
                pred_np = savgol_filter(pred_np, SAVGOL_WINDOW, SAVGOL_POLYORDER)
            else:
                for vi in range(pred_np.shape[0]):
                    pred_np[vi] = savgol_filter(pred_np[vi], SAVGOL_WINDOW, SAVGOL_POLYORDER)

        preds.append(pred_np.astype(np.float32))
        pct = int((idx + 1) / total * 100)
        progress_bar.progress(pct, text=f"正在分析第 {idx + 1}/{total} 个样本...")
        status.caption(f"已完成 {idx + 1}/{total} 个样本")

    progress_bar.empty()
    status.empty()
    return np.stack(preds, axis=0)


def summarize_uploaded_metrics(preds, labels):
    """汇总上传数据集的评估指标。"""
    if labels is None:
        return None

    labels = ensure_y_shape(labels)
    if preds.ndim == 2:
        preds = preds[:, None, :]

    n_vars = min(preds.shape[1], labels.shape[1], len(VAR_META))
    summary = {}
    for vi in range(n_vars):
        metrics_list = [
            evaluate_profile(preds[i, vi], labels[i, vi])
            for i in range(len(preds))
        ]
        summary[vi] = {
            "rmse_mean": float(np.mean([m["rmse"] for m in metrics_list])),
            "bias_mean": float(np.mean([m["bias"] for m in metrics_list])),
            "cc_mean": float(np.mean([m["cc"] for m in metrics_list])),
            "mae_mean": float(np.mean([m["mae"] for m in metrics_list])),
            "count": len(metrics_list),
        }
    return summary


def build_prediction_download(preds):
    """构造可下载的预测结果 npz。"""
    buffer = io.BytesIO()
    if preds.ndim == 2:
        np.savez_compressed(buffer, prediction=preds)
    else:
        payload = {"prediction": preds}
        for vi in range(min(preds.shape[1], len(VAR_META))):
            payload[f"prediction_{vi}_{VAR_META[vi]['name']}"] = preds[:, vi, :]
        np.savez_compressed(buffer, **payload)
    return buffer.getvalue()


# ─────────────────── 绘图函数 ───────────────────
def plot_bending_angle(ba, heights):
    """绘制弯曲角输入剖面"""
    with mpl.rc_context(get_plot_style()):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(ba, heights, color=COLORS['accent_cyan'], linewidth=1.8, alpha=0.9)
        ax.fill_betweenx(heights, ba, alpha=0.08, color=COLORS['accent_cyan'])
        ax.set_xlabel('log₁₀(弯曲角 / rad)')
        ax.set_ylabel('高度 (km)')
        ax.set_title('输入：弯曲角剖面', pad=12)
        ax.set_ylim(0, 60)
        fig.tight_layout()
    return fig


def plot_profile_comparison(pred, truth, heights, var_idx):
    """绘制单变量预测-真值对比图"""
    meta = VAR_META[var_idx]
    with mpl.rc_context(get_plot_style()):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(truth, heights, color=COLORS['truth_color'], linewidth=2,
                label='真值', alpha=0.85, zorder=3)
        ax.plot(pred, heights, color=meta['color'], linewidth=2,
                label='反演', linestyle='--', alpha=0.9, zorder=4)
        # 误差填充
        ax.fill_betweenx(heights, truth, pred, alpha=0.06, color=meta['color'])
        ax.set_xlabel(f"{meta['name']} ({meta['unit']})")
        ax.set_ylabel('高度 (km)')
        ax.set_title(f"{meta['icon']} {meta['name']}剖面对比", pad=12)
        ax.set_ylim(0, 60)
        ax.legend(loc='upper right', framealpha=0.8)
        fig.tight_layout()
    return fig


def plot_error_profile(pred, truth, heights, var_idx):
    """绘制逐高度误差剖面"""
    meta = VAR_META[var_idx]
    error = pred - truth
    with mpl.rc_context(get_plot_style()):
        fig, ax = plt.subplots(figsize=(3.5, 5))
        ax.barh(heights, error, height=0.22, color=meta['color'], alpha=0.5)
        ax.axvline(0, color='#475569', linewidth=0.8, linestyle='-')
        ax.set_xlabel('误差')
        ax.set_ylabel('高度 (km)')
        ax.set_title('逐高度误差', pad=12)
        ax.set_ylim(0, 60)
        fig.tight_layout()
    return fig


def render_truth_preview(truth, heights):
    """渲染真值预览。"""
    if truth is None:
        st.info("未提供标签文件，系统将只展示预测结果，不计算误差指标。")
        return

    if truth.ndim > 1 and truth.shape[0] >= 3:
        with mpl.rc_context(get_plot_style()):
            fig, ax = plt.subplots(figsize=(5, 5))
            for vi in range(min(truth.shape[0], 3)):
                meta = VAR_META[vi]
                tv = truth[vi]
                tv_norm = (tv - tv.min()) / (tv.max() - tv.min() + 1e-8)
                ax.plot(tv_norm, heights, color=meta['color'], linewidth=1.5,
                        label=meta['name'], alpha=0.8)
            ax.set_xlabel('归一化值')
            ax.set_ylabel('高度 (km)')
            ax.set_title('真值：三变量廓线概览', pad=12)
            ax.set_ylim(0, 60)
            ax.legend(loc='upper right', framealpha=0.8)
            fig.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)
    elif truth.ndim == 1:
        with mpl.rc_context(get_plot_style()):
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.plot(truth, heights, color=COLORS['truth_color'], linewidth=2)
            ax.set_xlabel('温度 (K)')
            ax.set_ylabel('高度 (km)')
            ax.set_title('真值：温度剖面', pad=12)
            ax.set_ylim(0, 60)
            fig.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)


def render_prediction_results(pred_np, truth, heights):
    """渲染单样本预测结果。"""
    st.markdown('<div class="section-title">反演结果</div>', unsafe_allow_html=True)

    if pred_np.ndim == 1:
        col1, col2 = st.columns([2, 1])
        with col1:
            if truth is not None:
                fig = plot_profile_comparison(pred_np, truth.flatten(), heights, 0)
            else:
                with mpl.rc_context(get_plot_style()):
                    fig, ax = plt.subplots(figsize=(5, 5))
                    ax.plot(pred_np, heights, color=VAR_META[0]['color'], linewidth=2)
                    ax.set_xlabel('温度 (K)')
                    ax.set_ylabel('高度 (km)')
                    ax.set_title('🌡 温度反演结果', pad=12)
                    ax.set_ylim(0, 60)
                    fig.tight_layout()
            st.pyplot(fig, width="stretch")
            plt.close(fig)
        with col2:
            if truth is not None:
                metrics = evaluate_profile(pred_np, truth.flatten())
                _render_metric_card("RMSE", metrics['rmse'], ".4f", "neutral")
                _render_metric_card("Bias", metrics['bias'], ".4f", "neutral")
                cc_cls, cc_label = cc_quality_class(metrics['cc'])
                _render_metric_card(f"CC ({cc_label})", metrics['cc'], ".4f", cc_cls)
            else:
                st.caption("未上传真值标签，无法计算 RMSE / Bias / CC。")
    else:
        n_vars = min(pred_np.shape[0], 3)
        all_metrics = []
        if truth is not None:
            for vi in range(n_vars):
                tv = truth[vi] if truth.ndim > 1 and vi < truth.shape[0] else truth
                all_metrics.append(evaluate_profile(pred_np[vi], tv))

        metric_cols = st.columns(n_vars)
        for vi in range(n_vars):
            meta = VAR_META[vi]
            with metric_cols[vi]:
                if truth is not None:
                    metric = all_metrics[vi]
                    cc_cls, _ = cc_quality_class(metric['cc'])
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size: 0.85rem; color: {meta['color']}; font-weight: 600;
                                    margin-bottom: 0.5rem; font-family: 'Noto Sans SC', sans-serif;">
                            {meta['icon']} {meta['name']}
                        </div>
                        <div style="display: flex; justify-content: space-around; gap: 0.5rem;">
                            <div>
                                <div class="metric-label">CC</div>
                                <div class="metric-value {cc_cls}">{metric['cc']:.4f}</div>
                            </div>
                            <div>
                                <div class="metric-label">RMSE</div>
                                <div class="metric-value metric-neutral">{metric['rmse']:.4f}</div>
                            </div>
                            <div>
                                <div class="metric-label">Bias</div>
                                <div class="metric-value metric-neutral">{metric['bias']:.4f}</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size: 0.85rem; color: {meta['color']}; font-weight: 600;
                                    margin-bottom: 0.5rem; font-family: 'Noto Sans SC', sans-serif;">
                            {meta['icon']} {meta['name']}
                        </div>
                        <div class="metric-label">Prediction Only</div>
                    </div>
                    """, unsafe_allow_html=True)

        plot_cols = st.columns(n_vars)
        for vi in range(n_vars):
            with plot_cols[vi]:
                if truth is not None:
                    tv = truth[vi] if truth.ndim > 1 and vi < truth.shape[0] else truth
                    fig = plot_profile_comparison(pred_np[vi], tv, heights, vi)
                else:
                    with mpl.rc_context(get_plot_style()):
                        fig, ax = plt.subplots(figsize=(5, 5))
                        ax.plot(pred_np[vi], heights, color=VAR_META[vi]['color'], linewidth=2)
                        ax.set_xlabel(f"{VAR_META[vi]['name']} ({VAR_META[vi]['unit']})")
                        ax.set_ylabel('高度 (km)')
                        ax.set_title(f"{VAR_META[vi]['icon']} {VAR_META[vi]['name']}反演结果", pad=12)
                        ax.set_ylim(0, 60)
                        fig.tight_layout()
                st.pyplot(fig, width="stretch")
                plt.close(fig)


# ─────────────────── 主界面 ───────────────────
def main():
    # ── 标题 ──
    st.markdown("""
    <div style="padding: 0.5rem 0 0.25rem 0;">
        <div class="hero-title">GNSS-RO 大气剖面反演系统</div>
        <div class="hero-subtitle">
            基于条件扩散模型 · 多数据源切换演示 · 温度 / 气压 / 湿度三变量联合反演
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 侧栏 ──
    with st.sidebar:
        local_datasets = discover_datasets(require_training_data=True)
        stats_sources = discover_datasets(require_training_data=False)

        st.markdown("""
        <div style="padding: 0.5rem 0; margin-bottom: 0.5rem;">
            <div style="font-size: 1.05rem; font-weight: 600; color: #e2e8f0;
                        font-family: 'Noto Sans SC', sans-serif;">
                ⚙️ 模型配置
            </div>
        </div>
        """, unsafe_allow_html=True)

        mode_options = ["upload"]
        if local_datasets:
            mode_options.append("local")
        demo_mode = st.radio(
            "演示模式",
            mode_options,
            index=0,
            format_func=lambda x: {
                "upload": "📤 上传分析",
                "local": "🗂️ 本地数据集",
            }[x],
            help="推荐使用上传分析模式，适合没有本地完整数据集的场景。",
        )

        def format_source_label(item):
            if item.get("summary"):
                return (
                    f"{item['name']} · "
                    f"{item['summary'].get('train', 0):,}/{item['summary'].get('val', 0):,}/{item['summary'].get('test', 0):,}"
                )
            if item.get("has_stats"):
                return f"{item['name']} · stats only"
            return item["name"]

        selected_data_dir = None
        selected_stats_dir = None
        if demo_mode == "local":
            selected_data_dir = st.selectbox(
                "本地数据源",
                [item["path"] for item in local_datasets],
                index=0,
                format_func=lambda path: format_source_label(
                    next(item for item in local_datasets if item["path"] == path)
                ),
            )
        else:
            if not stats_sources:
                st.error("未找到可用于推理的统计量目录（至少需要 `stats.npy`）")
                return
            selected_stats_dir = st.selectbox(
                "统计量来源",
                [item["path"] for item in stats_sources],
                index=0,
                format_func=lambda path: format_source_label(
                    next(item for item in stats_sources if item["path"] == path)
                ),
                help="上传分析只依赖训练统计量和模型权重，不需要本地完整数据集。",
            )

        pth_files = find_model_files()
        if not pth_files:
            st.error("未找到模型权重文件")
            return

        def format_model_path(path):
            rel_path = os.path.relpath(path, PROJECT_ROOT)
            if rel_path == os.path.basename(path):
                return rel_path
            return f"{os.path.basename(path)} · {os.path.dirname(rel_path)}"

        sel_idx = st.selectbox(
            "模型权重",
            range(len(pth_files)),
            format_func=lambda i: format_model_path(pth_files[i]),
            index=0,
        )
        model_path = pth_files[sel_idx]

        model_type = st.radio(
            "架构",
            ["auto", "enhanced", "legacy"],
            format_func=lambda x: {
                "auto": "自动检测",
                "enhanced": "增强 U-Net (交叉注意力)",
                "legacy": "原始 U-Net",
            }[x],
        )

        out_ch = st.selectbox(
            "输出通道",
            [3, 1],
            format_func=lambda x: f"{x} 通道 — {'温度+气压+湿度' if x == 3 else '仅温度'}",
        )

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size: 0.85rem; font-weight: 500; color: #94a3b8;
                    font-family: 'Noto Sans SC', sans-serif; margin-bottom: 0.5rem;">
            采样设置
        </div>
        """, unsafe_allow_html=True)

        sampler = st.radio(
            "采样方法",
            ["DDPM", "DDIM"],
            format_func=lambda x: (
                "🔬 DDPM — 1000 步完整采样" if x == "DDPM" else "⚡ DDIM — 快速采样"
            ),
        )

        ddim_steps = 50
        if sampler == "DDIM":
            ddim_steps = st.slider("DDIM 步数", 10, 200, 50, step=10)

        smooth = st.checkbox("Savitzky-Golay 平滑", value=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        dev_str = str(DEVICE)
        dev_icon = "🟢" if "cuda" in dev_str else "🟡"
        st.markdown(f"""
        <div style="font-size: 0.75rem; color: #64748b; font-family: 'JetBrains Mono', monospace;">
            {dev_icon} 设备: {dev_str}<br>
            📐 高度网格: 0–60 km, 301 层<br>
            🔢 扩散步数: {TIMESTEPS}
        </div>
        """, unsafe_allow_html=True)

    if demo_mode == "upload":
        stats_bundle = load_stats_bundle(selected_stats_dir)
        if stats_bundle is None:
            st.error(f"统计量未找到: {selected_stats_dir}")
            return
        stats, summary, heights = stats_bundle
        source_name = os.path.basename(selected_stats_dir)

        st.markdown(f"""
        <div class="info-bar">
            <div class="info-item">📤 模式 <span class="info-val">上传分析</span></div>
            <div class="info-item">📏 统计量 <span class="info-val">{source_name}</span></div>
            <div class="info-item">🤖 模型 <span class="info-val">{os.path.basename(model_path)}</span></div>
            <div class="info-item">🎯 输出 <span class="info-val">{out_ch} 通道</span></div>
        </div>
        """, unsafe_allow_html=True)

        if summary:
            summary_parts = []
            for key in ("train", "val", "test", "processed"):
                if key in summary:
                    summary_parts.append(f"{key}={summary[key]:,}")
            if summary_parts:
                st.caption(f"{source_name} · {' · '.join(summary_parts)}")

        st.markdown('<div class="section-title">上传数据</div>', unsafe_allow_html=True)
        st.caption("支持 `npy` / `npz` / `csv`。输入应为 `(301,)` 或 `(N, 301)`；标签可选，为 `(N, 301)` 或 `(N, C, 301)`。")

        col_upload_x, col_upload_y = st.columns(2)
        with col_upload_x:
            uploaded_x_file = st.file_uploader(
                "上传输入弯曲角",
                type=["npy", "npz", "csv"],
                key="uploaded_x_file",
                help="推荐 npz 中使用键名 `x`。",
            )
        with col_upload_y:
            uploaded_y_file = st.file_uploader(
                "上传真值标签（可选）",
                type=["npy", "npz", "csv"],
                key="uploaded_y_file",
                help="若上传标签，可自动计算误差指标。",
            )

        uploaded_x = None
        uploaded_y = None
        if uploaded_x_file is not None:
            try:
                x_payload, _ = load_array_from_upload(uploaded_x_file)
                y_payload = None
                if uploaded_y_file is not None:
                    y_payload, _ = load_array_from_upload(uploaded_y_file)
                uploaded_x, uploaded_y = extract_upload_arrays(x_payload, y_payload)
            except Exception as exc:
                st.error(f"上传文件解析失败: {exc}")
                uploaded_x = None
                uploaded_y = None

        if uploaded_x is not None:
            total_uploaded = len(uploaded_x)
            st.markdown(f"""
            <div class="info-bar">
                <div class="info-item">📥 上传样本 <span class="info-val">{total_uploaded:,}</span></div>
                <div class="info-item">📐 输入形状 <span class="info-val">{tuple(uploaded_x.shape)}</span></div>
                <div class="info-item">🏷️ 标签 <span class="info-val">{'已提供' if uploaded_y is not None else '未提供'}</span></div>
            </div>
            """, unsafe_allow_html=True)

            col_limit, col_preview = st.columns([1, 1])
            with col_limit:
                default_limit = min(total_uploaded, 16 if sampler == "DDPM" else 64)
                analysis_limit = st.number_input(
                    "本次分析样本数",
                    min_value=1,
                    max_value=total_uploaded,
                    value=default_limit,
                    help="DDPM 较慢，建议现场控制在较小样本数。",
                )
            with col_preview:
                preview_idx = st.number_input(
                    "预览样本索引",
                    min_value=0,
                    max_value=total_uploaded - 1,
                    value=0,
                )

            preview_truth = uploaded_y[preview_idx] if uploaded_y is not None else None
            col_ba, col_truth = st.columns(2)
            with col_ba:
                fig = plot_bending_angle(uploaded_x[preview_idx], heights)
                st.pyplot(fig, width="stretch")
                plt.close(fig)
            with col_truth:
                render_truth_preview(preview_truth, heights)

            st.markdown('<div class="section-title">运行分析</div>', unsafe_allow_html=True)
            st.caption("系统将对选定数量的上传样本逐条推理，并提供结果下载。")

            if st.button("开始分析上传数据", type="primary"):
                try:
                    model = load_model(model_path, model_type, out_ch)
                except Exception as exc:
                    st.error(f"模型加载失败: {exc}")
                    return

                schedule = DiffusionSchedule(TIMESTEPS, device=DEVICE)
                x_batch = uploaded_x[:analysis_limit]
                y_batch = uploaded_y[:analysis_limit] if uploaded_y is not None else None
                preds = run_uploaded_inference(
                    model,
                    x_batch,
                    stats,
                    sampler,
                    schedule,
                    out_ch,
                    ddim_steps,
                    smooth,
                )

                metric_summary = summarize_uploaded_metrics(preds, y_batch)
                st.success(f"上传数据分析完成，共处理 {len(preds)} 个样本。")

                if metric_summary:
                    st.markdown('<div class="section-title">批量评估汇总</div>', unsafe_allow_html=True)
                    summary_cols = st.columns(min(len(metric_summary), 3))
                    metric_export = {}
                    for vi, column in enumerate(summary_cols):
                        metric = metric_summary[vi]
                        metric_export[VAR_META[vi]["name"]] = metric
                        cc_cls, _ = cc_quality_class(metric["cc_mean"])
                        with column:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div style="font-size: 0.85rem; color: {VAR_META[vi]['color']}; font-weight: 600;
                                            margin-bottom: 0.5rem; font-family: 'Noto Sans SC', sans-serif;">
                                    {VAR_META[vi]['icon']} {VAR_META[vi]['name']}
                                </div>
                                <div style="display: flex; justify-content: space-around; gap: 0.5rem;">
                                    <div>
                                        <div class="metric-label">Mean CC</div>
                                        <div class="metric-value {cc_cls}">{metric['cc_mean']:.4f}</div>
                                    </div>
                                    <div>
                                        <div class="metric-label">Mean RMSE</div>
                                        <div class="metric-value metric-neutral">{metric['rmse_mean']:.4f}</div>
                                    </div>
                                    <div>
                                        <div class="metric-label">Mean Bias</div>
                                        <div class="metric-value metric-neutral">{metric['bias_mean']:.4f}</div>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                    st.download_button(
                        "下载批量预测结果 (.npz)",
                        data=build_prediction_download(preds),
                        file_name="uploaded_predictions.npz",
                        mime="application/octet-stream",
                    )
                    st.download_button(
                        "下载评估摘要 (.json)",
                        data=json.dumps(metric_export, ensure_ascii=False, indent=2),
                        file_name="uploaded_metrics_summary.json",
                        mime="application/json",
                    )
                else:
                    st.download_button(
                        "下载批量预测结果 (.npz)",
                        data=build_prediction_download(preds),
                        file_name="uploaded_predictions.npz",
                        mime="application/octet-stream",
                    )

                result_preview_idx = min(preview_idx, len(preds) - 1)
                preview_label = y_batch[result_preview_idx] if y_batch is not None else None
                render_prediction_results(preds[result_preview_idx], preview_label, heights)

        else:
            st.info("请先上传输入弯曲角文件，再开始分析。")

    else:
        result = load_data(selected_data_dir)
        if result is None:
            st.error(f"数据未找到: {selected_data_dir}")
            return

        data, stats, summary, heights = result
        dataset_name = os.path.basename(selected_data_dir)
        if os.path.exists(os.path.join(selected_data_dir, "stats.npy")):
            stats_source = "stats.npy"
        elif stats.get("stats_space") == "normalized":
            stats_source = "normalized fallback"
        else:
            stats_source = "train split fallback"

        train_n = len(data['train']['x']) if 'train' in data else 0
        val_n = len(data['val']['x']) if 'val' in data else 0
        test_n = len(data['test']['x']) if 'test' in data else 0

        st.markdown(f"""
        <div class="info-bar">
            <div class="info-item">🗂️ 数据源 <span class="info-val">{dataset_name}</span></div>
            <div class="info-item">📦 训练集 <span class="info-val">{train_n:,}</span></div>
            <div class="info-item">📋 验证集 <span class="info-val">{val_n:,}</span></div>
            <div class="info-item">🧪 测试集 <span class="info-val">{test_n:,}</span></div>
            <div class="info-item">📐 总样本 <span class="info-val">{train_n + val_n + test_n:,}</span></div>
            <div class="info-item">📏 统计量 <span class="info-val">{stats_source}</span></div>
        </div>
        """, unsafe_allow_html=True)

        if summary:
            paired = summary.get("paired")
            processed = summary.get("processed")
            extra_parts = []
            if paired is not None:
                extra_parts.append(f"配对 {paired:,}")
            if processed is not None:
                extra_parts.append(f"处理成功 {processed:,}")
            if extra_parts:
                st.caption(f"{dataset_name} · {' · '.join(extra_parts)}")
        elif dataset_name == "Processed":
            st.warning("当前数据源是早期 `COSMIC/FY-3D` 演示数据，不建议用于论文结果展示。")

        st.markdown('<div class="section-title">样本选择</div>', unsafe_allow_html=True)
        col_ds, col_idx, col_btn = st.columns([1, 1, 0.6])
        with col_ds:
            available_splits = [s for s in ['train', 'val', 'test'] if s in data]
            split_labels = {'train': '训练集', 'val': '验证集', 'test': '测试集'}
            split = st.selectbox(
                "数据集",
                available_splits,
                format_func=lambda x: f"{split_labels[x]} ({len(data[x]['x']):,} 样本)",
                index=available_splits.index('test') if 'test' in available_splits else 0,
            )
        split_x = data[split]['x']
        split_y = data[split]['y']
        n_split = len(split_x)

        with col_idx:
            sample_idx = st.number_input("样本索引", 0, n_split - 1, value=min(42, n_split - 1))
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🎲 随机", width="stretch"):
                st.session_state['_rand_idx'] = int(np.random.randint(0, n_split))
                st.rerun()
        if '_rand_idx' in st.session_state:
            sample_idx = st.session_state.pop('_rand_idx')

        input_ba_std = split_x[sample_idx]
        truth_std = split_y[sample_idx]
        input_ba = denormalize_input_profiles(input_ba_std, stats)
        truth = denormalize_target_profiles(truth_std, stats)

        st.markdown('<div class="section-title">输入数据</div>', unsafe_allow_html=True)
        col_ba, col_truth = st.columns(2)
        with col_ba:
            fig = plot_bending_angle(input_ba, heights)
            st.pyplot(fig, width="stretch")
            plt.close(fig)
        with col_truth:
            render_truth_preview(truth, heights)

        st.markdown('<div class="section-title">模型反演</div>', unsafe_allow_html=True)
        if st.button("开始反演", type="primary", width="content"):
            try:
                model = load_model(model_path, model_type, out_ch)
            except Exception as exc:
                st.error(f"模型加载失败: {exc}")
                return

            schedule = DiffusionSchedule(TIMESTEPS, device=DEVICE)
            # 本地数据集中的磁盘数组已经标准化，可直接作为模型输入。
            cond_np = input_ba_std.astype(np.float32)
            cond = torch.tensor(cond_np).float().unsqueeze(0).unsqueeze(0).to(DEVICE)

            progress_bar = st.progress(0, text="初始化扩散采样...")
            with torch.no_grad():
                if sampler == "DDIM":
                    progress_bar.progress(10, text=f"DDIM {ddim_steps} 步采样中...")
                    gen = ddim_sample(model, cond, shape=(1, out_ch, 301), schedule=schedule, ddim_steps=ddim_steps)
                else:
                    progress_bar.progress(10, text="DDPM 1000 步采样中...")
                    gen = ddpm_sample(model, cond, shape=(1, out_ch, 301), schedule=schedule)

            pred_np = denormalize_prediction(gen, stats, out_ch)
            if smooth:
                if pred_np.ndim == 1:
                    pred_np = savgol_filter(pred_np, SAVGOL_WINDOW, SAVGOL_POLYORDER)
                else:
                    for i in range(pred_np.shape[0]):
                        pred_np[i] = savgol_filter(pred_np[i], SAVGOL_WINDOW, SAVGOL_POLYORDER)
            progress_bar.progress(100, text="反演完成")
            render_prediction_results(pred_np, truth, heights)

    # ── 全测试集评估报告 ──
    eval_result = load_eval_report()
    if eval_result:
        report, exp_name = eval_result
        st.markdown('<div class="section-title">全测试集评估报告</div>', unsafe_allow_html=True)

        n_eval = report.get('n_samples', '—')
        st.markdown(f"""
        <div class="info-bar">
            <div class="info-item">📁 实验 <span class="info-val">{exp_name}</span></div>
            <div class="info-item">🧪 评估样本 <span class="info-val">{n_eval:,}</span></div>
        </div>
        """, unsafe_allow_html=True)

        summary = report.get('summary', {})
        var_keys = ['temperature', 'pressure', 'humidity']
        var_cn = {'temperature': '温度', 'pressure': '气压', 'humidity': '湿度'}

        ecols = st.columns(3)
        for i, vk in enumerate(var_keys):
            if vk not in summary:
                continue
            s = summary[vk]
            meta = VAR_META[i]
            cc_cls, cc_label = cc_quality_class(s.get('cc_mean', 0))
            with ecols[i]:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 0.85rem; color: {meta['color']}; font-weight: 600;
                                margin-bottom: 0.5rem; font-family: 'Noto Sans SC', sans-serif;">
                        {meta['icon']} {var_cn[vk]}
                        <span style="font-size: 0.7rem; color: #64748b; margin-left: 0.5rem;">
                            n={s.get('count', '—'):,}
                        </span>
                    </div>
                    <div style="display: flex; justify-content: space-around; gap: 0.5rem;">
                        <div>
                            <div class="metric-label">Mean CC</div>
                            <div class="metric-value {cc_cls}">{s.get('cc_mean', 0):.4f}</div>
                        </div>
                        <div>
                            <div class="metric-label">Mean RMSE</div>
                            <div class="metric-value metric-neutral">{s.get('rmse_mean', 0):.4f}</div>
                        </div>
                        <div>
                            <div class="metric-label">Mean Bias</div>
                            <div class="metric-value metric-neutral">{s.get('bias_mean', 0):.4f}</div>
                        </div>
                    </div>
                    <div style="margin-top: 0.5rem; font-size: 0.7rem; color: #475569;
                                font-family: 'JetBrains Mono', monospace;">
                        CC range: [{s.get('cc_min', 0):.3f}, {s.get('cc_max', 0):.3f}] &nbsp;|&nbsp;
                        RMSE range: [{s.get('rmse_min', 0):.3f}, {s.get('rmse_max', 0):.3f}]
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # 评估图片
        exp_path = os.path.join(PROJECT_ROOT, 'experiments', exp_name)
        pngs = [f for f in os.listdir(exp_path) if f.endswith('.png')] if os.path.isdir(exp_path) else []
        if pngs:
            with st.expander("📊 查看评估图表", expanded=False):
                img_cols = st.columns(min(len(pngs), 3))
                for i, png in enumerate(sorted(pngs)):
                    with img_cols[i % 3]:
                        st.image(
                            os.path.join(exp_path, png),
                            caption=png.replace('.png', ''),
                            width="stretch",
                        )

    # ── 底部 ──
    st.markdown("""
    <div style="margin-top: 3rem; padding: 1rem 0; border-top: 1px solid #1e293b;
                text-align: center; font-size: 0.75rem; color: #475569;
                font-family: 'JetBrains Mono', monospace;">
        GNSS-RO Atmospheric Profile Retrieval System &nbsp;·&nbsp;
        Conditional Diffusion Model &nbsp;·&nbsp; FY-3D GNOS
    </div>
    """, unsafe_allow_html=True)


def _render_metric_card(label, value, fmt, css_class):
    """渲染单个指标卡片"""
    st.markdown(f"""
    <div class="metric-card" style="margin-bottom: 0.5rem;">
        <div class="metric-label">{label}</div>
        <div class="metric-value {css_class}">{value:{fmt}}</div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
