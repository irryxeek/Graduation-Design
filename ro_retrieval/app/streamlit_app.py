"""
Streamlit 交互式掩星反演可视化应用
===================================
功能:
  1. 上传弯曲角数据 / 选择已有样本
  2. 选择模型权重 (原始/增强)
  3. 选择采样方式 (DDPM/DDIM)
  4. 运行推理并展示结果
  5. 多变量剖面对比可视化
  6. 评估指标面板
"""

import os
import sys
import json
import torch
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# 将项目根目录加入 path  (app -> ro_retrieval -> project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from ro_retrieval.config import (
    DEVICE, PROCESSED_DIR, TIMESTEPS,
    STD_HEIGHT, SAVGOL_WINDOW, SAVGOL_POLYORDER,
)
from ro_retrieval.model.unet import ConditionalUNet1D, EnhancedConditionalUNet1D
from ro_retrieval.model.diffusion import DiffusionSchedule, ddpm_sample, ddim_sample
from ro_retrieval.evaluation.metrics import evaluate_profile

# ─────────────────── 页面配置 ───────────────────
st.set_page_config(
    page_title="GNSS-RO 大气剖面反演系统",
    page_icon="🛰️",
    layout="wide",
)


# ─────────────────── 缓存加载 ───────────────────
@st.cache_resource
def load_data():
    """加载预处理数据与统计量"""
    x_path = os.path.join(PROCESSED_DIR, "train_x.npy")
    y_path = os.path.join(PROCESSED_DIR, "train_y.npy")
    if not os.path.exists(x_path):
        return None, None, None, None, None, None
    raw_x = np.load(x_path).astype(np.float32)
    raw_y = np.load(y_path).astype(np.float32)
    x_mean = np.mean(raw_x, axis=0)
    x_std = np.std(raw_x, axis=0) + 1e-6
    y_mean = np.mean(raw_y, axis=0)
    y_std = np.std(raw_y, axis=0) + 1e-6
    return raw_x, raw_y, x_mean, x_std, y_mean, y_std


@st.cache_resource
def load_model(path, model_type, out_ch):
    """缓存模型加载, 自动检测权重类型"""
    state_dict = torch.load(path, map_location=DEVICE)

    # 自动检测: 如果权重中有 "time_mlp" 说明是 legacy 模型
    detected = "legacy"
    if any(k.startswith("time_embed.") for k in state_dict.keys()):
        detected = "enhanced"

    if model_type == "auto":
        model_type = detected

    if model_type == "enhanced":
        m = EnhancedConditionalUNet1D(
            in_channels=out_ch, cond_channels=1, out_channels=out_ch,
            use_cross_attention=True,
        )
    else:
        m = ConditionalUNet1D(
            in_channels=out_ch, cond_channels=1, out_channels=out_ch,
        )

    m.load_state_dict(state_dict)
    m.to(DEVICE)
    m.eval()
    return m


def find_model_files():
    """扫描项目根目录中的 .pth 文件"""
    pth_files = []
    for f in os.listdir(PROJECT_ROOT):
        if f.endswith(".pth"):
            pth_files.append(f)
    return sorted(pth_files)


# ─────────────────── 主界面 ───────────────────
def main():
    st.title("🛰️ GNSS-RO 大气剖面反演系统")
    st.markdown(
        "基于**条件扩散模型**的 GNSS 无线电掩星观测反演 "
        "—— 从弯曲角剖面生成温度 / 气压 / 湿度大气剖面"
    )

    # ── 侧栏: 模型与参数 ──────
    with st.sidebar:
        st.header("⚙️ 参数设置")

        # 模型选择
        pth_files = find_model_files()
        if not pth_files:
            st.error("未找到模型权重文件 (.pth)")
            return

        model_file = st.selectbox("模型权重", pth_files,
                                  index=len(pth_files) - 1)
        model_type = st.radio("模型类型",
                              ["auto (自动检测)", "legacy (原始 U-Net)", "enhanced (交叉注意力)"],
                              index=0)
        model_type_key = "auto"
        if "legacy" in model_type:
            model_type_key = "legacy"
        elif "enhanced" in model_type:
            model_type_key = "enhanced"

        out_ch = st.selectbox("输出通道数", [1, 3], index=0,
                              help="1 = 仅温度; 3 = 温度+气压+湿度")

        st.divider()

        sampler = st.radio("采样方式", ["DDIM (快速)", "DDPM (完整)"], index=0)
        ddim_steps = 50
        if "DDIM" in sampler:
            ddim_steps = st.slider("DDIM 步数", 10, 200, 50, step=10)

        st.divider()
        smooth = st.checkbox("Savitzky-Golay 平滑", value=True)

    # ── 数据 ──────
    raw_x, raw_y, x_mean, x_std, y_mean, y_std = load_data()
    if raw_x is None:
        st.error(f"数据未找到: {PROCESSED_DIR}/train_x.npy")
        return

    n_total = len(raw_x)

    # ── 样本选择 ──────
    st.subheader("📊 选择输入样本")
    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        sample_idx = st.number_input("样本索引", 0, n_total - 1,
                                     value=min(748, n_total - 1))
    with col_sel2:
        if st.button("🎲 随机选择"):
            sample_idx = int(np.random.randint(0, n_total))
            st.rerun()

    # ── 输入弯曲角展示 ──────
    heights = np.linspace(0, 60, 301)
    input_ba = raw_x[sample_idx]
    truth = raw_y[sample_idx]

    col_inp, col_inp2 = st.columns(2)
    with col_inp:
        fig_ba, ax_ba = plt.subplots(figsize=(5, 4))
        ax_ba.plot(input_ba, heights, 'b-', linewidth=1.5)
        ax_ba.set_xlabel("log₁₀(弯曲角/rad)")
        ax_ba.set_ylabel("高度 (km)")
        ax_ba.set_title("输入: 弯曲角剖面")
        ax_ba.grid(True, alpha=0.3)
        st.pyplot(fig_ba)
        plt.close(fig_ba)

    with col_inp2:
        if truth.ndim == 1:
            fig_t, ax_t = plt.subplots(figsize=(5, 4))
            ax_t.plot(truth, heights, 'k-', linewidth=1.5)
            ax_t.set_xlabel("温度 (K)")
            ax_t.set_ylabel("高度 (km)")
            ax_t.set_title("真值: 温度剖面")
            ax_t.grid(True, alpha=0.3)
            st.pyplot(fig_t)
            plt.close(fig_t)

    # ── 推理 ──────
    st.subheader("🚀 运行推理")
    if st.button("开始反演", type="primary"):
        model_path = os.path.join(PROJECT_ROOT, model_file)
        try:
            model = load_model(model_path, model_type_key, out_ch)
        except Exception as e:
            st.error(f"模型加载失败: {e}")
            return

        schedule = DiffusionSchedule(TIMESTEPS, device=DEVICE)

        # 标准化输入
        cond_np = (input_ba - x_mean) / x_std
        cond = torch.tensor(cond_np).float().unsqueeze(0).unsqueeze(0).to(DEVICE)

        with st.spinner("正在扩散采样, 请稍候..."):
            with torch.no_grad():
                if "DDIM" in sampler:
                    gen = ddim_sample(model, cond, shape=(1, out_ch, 301),
                                      schedule=schedule, ddim_steps=ddim_steps)
                else:
                    gen = ddpm_sample(model, cond, shape=(1, out_ch, 301),
                                      schedule=schedule)

        # 反归一化
        y_mean_t = torch.tensor(y_mean).float().to(DEVICE)
        y_std_t = torch.tensor(y_std).float().to(DEVICE)
        pred = gen.squeeze(0).cpu()

        if pred.ndim == 1:
            pred = pred * torch.tensor(y_std).float() + torch.tensor(y_mean).float()
            pred_np = pred.numpy()
        else:
            # multi-var
            if y_mean_t.ndim == 1:
                pred = pred[0] * torch.tensor(y_std).float() + torch.tensor(y_mean).float()
                pred_np = pred.numpy()
            else:
                pred = pred * y_std_t.cpu() + y_mean_t.cpu()
                pred_np = pred.numpy()

        # 平滑
        if smooth:
            if pred_np.ndim == 1:
                pred_np = savgol_filter(pred_np, SAVGOL_WINDOW, SAVGOL_POLYORDER)
            else:
                for i in range(pred_np.shape[0]):
                    pred_np[i] = savgol_filter(pred_np[i], SAVGOL_WINDOW, SAVGOL_POLYORDER)

        # ── 结果可视化 ──────
        st.subheader("📈 反演结果")

        if pred_np.ndim == 1:
            truth_flat = truth.flatten()
            metrics = evaluate_profile(pred_np, truth_flat)

            col_r1, col_r2 = st.columns([2, 1])
            with col_r1:
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.plot(truth_flat, heights, 'k-', linewidth=2, label='真值')
                ax.plot(pred_np, heights, 'r--', linewidth=2, label='反演结果')
                ax.set_xlabel("温度 (K)")
                ax.set_ylabel("高度 (km)")
                ax.set_title("温度剖面对比")
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close(fig)

            with col_r2:
                st.metric("RMSE", f"{metrics['rmse']:.3f}")
                st.metric("Bias", f"{metrics['bias']:.3f}")
                st.metric("CC", f"{metrics['cc']:.4f}")
                st.metric("MAE", f"{metrics['mae']:.3f}")

        else:
            var_names = ["温度 (K)", "气压 (hPa)", "湿度 (g/kg)"]
            cols = st.columns(min(pred_np.shape[0], 3))
            for v_idx in range(min(pred_np.shape[0], 3)):
                with cols[v_idx]:
                    tv = truth[v_idx] if truth.ndim > 1 and v_idx < truth.shape[0] else truth
                    m = evaluate_profile(pred_np[v_idx], tv)

                    fig, ax = plt.subplots(figsize=(5, 4))
                    ax.plot(tv, heights, 'k-', lw=2, label='真值')
                    ax.plot(pred_np[v_idx], heights, 'r--', lw=2, label='反演')
                    ax.set_xlabel(var_names[v_idx] if v_idx < len(var_names) else f"变量{v_idx}")
                    ax.set_ylabel("高度 (km)")
                    ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    plt.close(fig)

                    st.metric("RMSE", f"{m['rmse']:.3f}")
                    st.metric("Bias", f"{m['bias']:.3f}")
                    st.metric("CC", f"{m['cc']:.4f}")

    # ── 底部: 历史评估结果 ──────
    st.divider()
    st.subheader("📋 历史评估报告")
    report_dirs = [
        os.path.join(PROJECT_ROOT, "evaluation_results"),
        os.path.join(PROJECT_ROOT, "evaluation_results_ddim"),
        os.path.join(PROJECT_ROOT, "evaluation_results_ddim_enhanced"),
    ]
    for rd in report_dirs:
        json_path = os.path.join(rd, "evaluation_report.json")
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                data = json.load(f)
            st.json(data, expanded=False)

    # 显示已有的评估图片
    for rd in report_dirs:
        if os.path.exists(rd):
            pngs = [f for f in os.listdir(rd) if f.endswith(".png")]
            if pngs:
                st.write(f"**{os.path.basename(rd)}** 评估图:")
                img_cols = st.columns(min(len(pngs), 3))
                for i, png in enumerate(pngs[:6]):
                    with img_cols[i % 3]:
                        st.image(os.path.join(rd, png), caption=png)


if __name__ == "__main__":
    main()
