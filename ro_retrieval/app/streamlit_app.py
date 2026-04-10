"""
Streamlit 交互式掩星反演可视化应用
===================================
基于条件扩散模型的 GNSS-RO 大气剖面端到端反演系统
"""

import os
import sys
import json
import torch
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal import savgol_filter

# ─────────────────── 中文字体 & 绘图风格 ───────────────────
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False

# 项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from ro_retrieval.config import (
    DEVICE, PROCESSED_DIR, TIMESTEPS,
    SAVGOL_WINDOW, SAVGOL_POLYORDER,
)
from ro_retrieval.model.unet import ConditionalUNet1D, EnhancedConditionalUNet1D
from ro_retrieval.model.diffusion import DiffusionSchedule, ddpm_sample, ddim_sample
from ro_retrieval.evaluation.metrics import evaluate_profile

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
@st.cache_resource
def load_data():
    """加载预处理数据与统计量"""
    data = {}
    for split in ['train', 'val', 'test']:
        xp = os.path.join(PROCESSED_DIR, f"{split}_x.npy")
        yp = os.path.join(PROCESSED_DIR, f"{split}_y.npy")
        if os.path.exists(xp) and os.path.exists(yp):
            data[split] = {
                'x': np.load(xp).astype(np.float32),
                'y': np.load(yp).astype(np.float32),
            }
    if 'train' not in data:
        return None
    # 统计量基于训练集
    raw_x, raw_y = data['train']['x'], data['train']['y']
    stats = {
        'x_mean': np.mean(raw_x, axis=0),
        'x_std':  np.std(raw_x, axis=0) + 1e-6,
        'y_mean': np.mean(raw_y, axis=0),
        'y_std':  np.std(raw_y, axis=0) + 1e-6,
    }
    return data, stats


@st.cache_resource
def load_model(path, model_type, out_ch):
    """加载模型权重, 自动检测架构"""
    state_dict = torch.load(path, map_location=DEVICE)
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
    """扫描模型权重"""
    pth = []
    for d in [PROJECT_ROOT, os.path.join(PROJECT_ROOT, 'checkpoints')]:
        if os.path.isdir(d):
            pth.extend(os.path.join(d, f) for f in os.listdir(d) if f.endswith('.pth'))
    return sorted(pth)


def load_eval_report():
    """加载最新评估报告"""
    exp_dir = os.path.join(PROJECT_ROOT, 'experiments')
    if not os.path.isdir(exp_dir):
        return None
    for name in sorted(os.listdir(exp_dir), reverse=True):
        jp = os.path.join(exp_dir, name, 'evaluation_report.json')
        if os.path.exists(jp):
            with open(jp) as f:
                return json.load(f), name
    return None


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


# ─────────────────── 主界面 ───────────────────
def main():
    # ── 标题 ──
    st.markdown("""
    <div style="padding: 0.5rem 0 0.25rem 0;">
        <div class="hero-title">GNSS-RO 大气剖面反演系统</div>
        <div class="hero-subtitle">
            基于条件扩散模型 · FY-3D GNOS 掩星数据 · 温度 / 气压 / 湿度三变量联合反演
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 侧栏 ──
    with st.sidebar:
        st.markdown("""
        <div style="padding: 0.5rem 0; margin-bottom: 0.5rem;">
            <div style="font-size: 1.05rem; font-weight: 600; color: #e2e8f0;
                        font-family: 'Noto Sans SC', sans-serif;">
                ⚙️ 模型配置
            </div>
        </div>
        """, unsafe_allow_html=True)

        pth_files = find_model_files()
        if not pth_files:
            st.error("未找到模型权重文件")
            return

        # 显示简短文件名
        short_names = [os.path.basename(p) for p in pth_files]
        sel_idx = st.selectbox("模型权重", range(len(pth_files)),
                               format_func=lambda i: short_names[i],
                               index=len(pth_files) - 1)
        model_path = pth_files[sel_idx]

        model_type = st.radio("架构", ["auto", "enhanced", "legacy"],
                              format_func=lambda x: {
                                  "auto": "自动检测",
                                  "enhanced": "增强 U-Net (交叉注意力)",
                                  "legacy": "原始 U-Net"
                              }[x])

        out_ch = st.selectbox("输出通道", [3, 1],
                              format_func=lambda x: f"{x} 通道 — {'温度+气压+湿度' if x==3 else '仅温度'}")

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size: 0.85rem; font-weight: 500; color: #94a3b8;
                    font-family: 'Noto Sans SC', sans-serif; margin-bottom: 0.5rem;">
            采样设置
        </div>
        """, unsafe_allow_html=True)

        sampler = st.radio("采样方法", ["DDPM", "DDIM"],
                           format_func=lambda x: f"{'🔬 DDPM — 1000 步完整采样' if x=='DDPM' else '⚡ DDIM — 快速采样'}")

        ddim_steps = 50
        if sampler == "DDIM":
            ddim_steps = st.slider("DDIM 步数", 10, 200, 50, step=10)

        smooth = st.checkbox("Savitzky-Golay 平滑", value=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # 设备信息
        dev_str = str(DEVICE)
        dev_icon = "🟢" if "cuda" in dev_str else "🟡"
        st.markdown(f"""
        <div style="font-size: 0.75rem; color: #64748b; font-family: 'JetBrains Mono', monospace;">
            {dev_icon} 设备: {dev_str}<br>
            📐 高度网格: 0–60 km, 301 层<br>
            🔢 扩散步数: {TIMESTEPS}
        </div>
        """, unsafe_allow_html=True)

    # ── 加载数据 ──
    result = load_data()
    if result is None:
        st.error(f"数据未找到: {PROCESSED_DIR}")
        return
    data, stats = result
    x_mean, x_std = stats['x_mean'], stats['x_std']
    y_mean, y_std = stats['y_mean'], stats['y_std']
    heights = np.linspace(0, 60, 301)

    # ── 数据概览 ──
    train_n = len(data['train']['x']) if 'train' in data else 0
    val_n = len(data['val']['x']) if 'val' in data else 0
    test_n = len(data['test']['x']) if 'test' in data else 0

    st.markdown(f"""
    <div class="info-bar">
        <div class="info-item">📦 训练集 <span class="info-val">{train_n:,}</span></div>
        <div class="info-item">📋 验证集 <span class="info-val">{val_n:,}</span></div>
        <div class="info-item">🧪 测试集 <span class="info-val">{test_n:,}</span></div>
        <div class="info-item">📐 总样本 <span class="info-val">{train_n+val_n+test_n:,}</span></div>
    </div>
    """, unsafe_allow_html=True)

    # ── 数据集选择与样本索引 ──
    st.markdown('<div class="section-title">样本选择</div>', unsafe_allow_html=True)

    col_ds, col_idx, col_btn = st.columns([1, 1, 0.6])
    with col_ds:
        available_splits = [s for s in ['train', 'val', 'test'] if s in data]
        split_labels = {'train': '训练集', 'val': '验证集', 'test': '测试集'}
        split = st.selectbox("数据集", available_splits,
                             format_func=lambda x: f"{split_labels[x]} ({len(data[x]['x']):,} 样本)",
                             index=available_splits.index('test') if 'test' in available_splits else 0)
    split_x = data[split]['x']
    split_y = data[split]['y']
    n_split = len(split_x)

    with col_idx:
        sample_idx = st.number_input("样本索引", 0, n_split - 1, value=min(42, n_split - 1))
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🎲 随机", use_container_width=True):
            st.session_state['_rand_idx'] = int(np.random.randint(0, n_split))
            st.rerun()
    if '_rand_idx' in st.session_state:
        sample_idx = st.session_state.pop('_rand_idx')

    input_ba = split_x[sample_idx]
    truth = split_y[sample_idx]

    # ── 输入展示 ──
    st.markdown('<div class="section-title">输入数据</div>', unsafe_allow_html=True)

    col_ba, col_truth = st.columns(2)
    with col_ba:
        fig = plot_bending_angle(input_ba, heights)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_truth:
        if truth.ndim > 1 and truth.shape[0] >= 3:
            # 真值概览 — 温度
            fig = plot_profile_comparison(truth[0], truth[0], heights, 0)
            # 重绘为纯真值展示
            with mpl.rc_context(get_plot_style()):
                fig, ax = plt.subplots(figsize=(5, 5))
                for vi in range(min(truth.shape[0], 3)):
                    m = VAR_META[vi]
                    # 归一化到 [0,1] 以便叠加展示
                    t = truth[vi]
                    t_norm = (t - t.min()) / (t.max() - t.min() + 1e-8)
                    ax.plot(t_norm, heights, color=m['color'], linewidth=1.5,
                            label=m['name'], alpha=0.8)
                ax.set_xlabel('归一化值')
                ax.set_ylabel('高度 (km)')
                ax.set_title('真值：三变量廓线概览', pad=12)
                ax.set_ylim(0, 60)
                ax.legend(loc='upper right', framealpha=0.8)
                fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
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
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    # ── 推理 ──
    st.markdown('<div class="section-title">模型反演</div>', unsafe_allow_html=True)

    if st.button("开始反演", type="primary", use_container_width=False):
        try:
            model = load_model(model_path, model_type, out_ch)
        except Exception as e:
            st.error(f"模型加载失败: {e}")
            return

        schedule = DiffusionSchedule(TIMESTEPS, device=DEVICE)
        cond_np = (input_ba - x_mean) / x_std
        cond = torch.tensor(cond_np).float().unsqueeze(0).unsqueeze(0).to(DEVICE)

        progress_bar = st.progress(0, text="初始化扩散采样...")

        with torch.no_grad():
            if sampler == "DDIM":
                progress_bar.progress(10, text=f"DDIM {ddim_steps} 步采样中...")
                gen = ddim_sample(model, cond, shape=(1, out_ch, 301),
                                  schedule=schedule, ddim_steps=ddim_steps)
            else:
                progress_bar.progress(10, text="DDPM 1000 步采样中...")
                gen = ddpm_sample(model, cond, shape=(1, out_ch, 301),
                                  schedule=schedule)
        progress_bar.progress(90, text="反标准化处理中...")

        # 反归一化
        pred = gen.squeeze(0).cpu()
        y_mean_t = torch.tensor(y_mean).float()
        y_std_t = torch.tensor(y_std).float()

        if pred.ndim == 1:
            pred = pred * y_std_t + y_mean_t
            pred_np = pred.numpy()
        else:
            if y_mean_t.ndim == 1:
                pred = pred[0] * y_std_t + y_mean_t
                pred_np = pred.numpy()
            else:
                pred = pred * y_std_t + y_mean_t
                pred_np = pred.numpy()

        # 平滑
        if smooth:
            if pred_np.ndim == 1:
                pred_np = savgol_filter(pred_np, SAVGOL_WINDOW, SAVGOL_POLYORDER)
            else:
                for i in range(pred_np.shape[0]):
                    pred_np[i] = savgol_filter(pred_np[i], SAVGOL_WINDOW, SAVGOL_POLYORDER)

        progress_bar.progress(100, text="反演完成")

        # ── 结果展示 ──
        st.markdown('<div class="section-title">反演结果</div>', unsafe_allow_html=True)

        if pred_np.ndim == 1:
            truth_flat = truth.flatten()
            m = evaluate_profile(pred_np, truth_flat)

            col1, col2 = st.columns([2, 1])
            with col1:
                fig = plot_profile_comparison(pred_np, truth_flat, heights, 0)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            with col2:
                _render_metric_card("RMSE", m['rmse'], ".4f", "neutral")
                _render_metric_card("Bias", m['bias'], ".4f", "neutral")
                cc_cls, cc_label = cc_quality_class(m['cc'])
                _render_metric_card(f"CC ({cc_label})", m['cc'], ".4f", cc_cls)
        else:
            n_vars = min(pred_np.shape[0], 3)

            # 指标汇总
            all_metrics = []
            for vi in range(n_vars):
                tv = truth[vi] if truth.ndim > 1 and vi < truth.shape[0] else truth
                all_metrics.append(evaluate_profile(pred_np[vi], tv))

            # 指标卡片行
            metric_cols = st.columns(n_vars)
            for vi in range(n_vars):
                m = all_metrics[vi]
                meta = VAR_META[vi]
                cc_cls, cc_label = cc_quality_class(m['cc'])
                with metric_cols[vi]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size: 0.85rem; color: {meta['color']}; font-weight: 600;
                                    margin-bottom: 0.5rem; font-family: 'Noto Sans SC', sans-serif;">
                            {meta['icon']} {meta['name']}
                        </div>
                        <div style="display: flex; justify-content: space-around; gap: 0.5rem;">
                            <div>
                                <div class="metric-label">CC</div>
                                <div class="metric-value {cc_cls}">{m['cc']:.4f}</div>
                            </div>
                            <div>
                                <div class="metric-label">RMSE</div>
                                <div class="metric-value metric-neutral">{m['rmse']:.4f}</div>
                            </div>
                            <div>
                                <div class="metric-label">Bias</div>
                                <div class="metric-value metric-neutral">{m['bias']:.4f}</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # 剖面对比图
            plot_cols = st.columns(n_vars)
            for vi in range(n_vars):
                tv = truth[vi] if truth.ndim > 1 and vi < truth.shape[0] else truth
                with plot_cols[vi]:
                    fig = plot_profile_comparison(pred_np[vi], tv, heights, vi)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

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
                        st.image(os.path.join(exp_path, png), caption=png.replace('.png', ''),
                                 use_container_width=True)

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
