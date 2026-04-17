"""生成更适合论文排版的一维条件 U-Net 结构示意图。"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe


matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False


OUT_DIR = Path("docs/thesis/figures")
PNG_PATH = OUT_DIR / "unet_conditional_architecture.png"
SVG_PATH = OUT_DIR / "unet_conditional_architecture.svg"

COLORS = {
    "bg": "#FCFBF8",
    "ink": "#1E2430",
    "muted": "#667085",
    "main_fill": "#E8F0F7",
    "main_edge": "#2E5E88",
    "cond_fill": "#FCEBD6",
    "cond_edge": "#C87A1C",
    "time_fill": "#E3F4E7",
    "time_edge": "#3E8E4F",
    "output_fill": "#F9DEE1",
    "output_edge": "#B64657",
    "skip": "#8C72B8",
    "arrow": "#414651",
    "section": "#EEF2F6",
}


def rounded_box(ax, x, y, w, h, title, subtitle="", fc="#FFFFFF", ec="#333333",
                title_size=11, subtitle_size=8.5, lw=1.6, radius=0.18, z=3):
    patch = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=z
    )
    patch.set_path_effects([
        pe.SimplePatchShadow(offset=(1.2, -1.2), alpha=0.10, rho=0.95),
        pe.Normal()
    ])
    ax.add_patch(patch)

    ax.text(
        x + w / 2, y + h * 0.63, title,
        ha="center", va="center", fontsize=title_size,
        fontweight="bold", color=COLORS["ink"], zorder=z + 1
    )
    if subtitle:
        ax.text(
            x + w / 2, y + h * 0.30, subtitle,
            ha="center", va="center", fontsize=subtitle_size,
            color=COLORS["muted"], zorder=z + 1
        )
    return patch


def arrow(ax, start, end, color, lw=1.7, style="-|>", mutation=12,
          connectionstyle="arc3,rad=0.0", linestyle="-", z=2):
    ax.annotate(
        "",
        xy=end, xytext=start,
        arrowprops=dict(
            arrowstyle=style,
            color=color,
            lw=lw,
            linestyle=linestyle,
            mutation_scale=mutation,
            connectionstyle=connectionstyle,
            shrinkA=5, shrinkB=5,
        ),
        zorder=z,
    )


def label(ax, x, y, text, color="#555", fc="white", size=8.2):
    ax.text(
        x, y, text,
        ha="center", va="center",
        fontsize=size, color=color, zorder=5,
        bbox=dict(boxstyle="round,pad=0.18", fc=fc, ec="none", alpha=0.94)
    )


def section_band(ax, x, y, w, h, text):
    band = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=0, facecolor=COLORS["section"], alpha=0.8, zorder=0
    )
    ax.add_patch(band)
    ax.text(
        x + 0.2, y + h / 2, text,
        ha="left", va="center", fontsize=9.5,
        color="#51606F", fontweight="bold", zorder=1
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    # 标题
    ax.text(
        8, 8.55, "增强版一维条件 U-Net 结构示意图",
        ha="center", va="center", fontsize=20,
        fontweight="bold", color=COLORS["ink"]
    )
    ax.text(
        8, 8.16,
        "Conditional encoder + multi-scale cross-attention + time-step embedding + skip connections",
        ha="center", va="center", fontsize=10.5, color=COLORS["muted"]
    )

    # 区域带
    section_band(ax, 0.55, 7.18, 14.9, 0.42, "条件分支 Condition Branch")
    section_band(ax, 0.55, 3.55, 14.9, 0.42, "主干去噪网络 Main U-Net Backbone")
    section_band(ax, 0.55, 0.82, 14.9, 0.42, "时间嵌入分支 Time Embedding Branch")

    # 主干坐标
    enc1 = (1.25, 4.65, 2.1, 1.15)
    enc2 = (4.25, 3.55, 2.2, 1.15)
    bottleneck = (7.15, 2.45, 2.35, 1.2)
    dec2 = (10.25, 3.55, 2.2, 1.15)
    dec1 = (13.15, 4.65, 2.1, 1.15)

    pool1 = (3.65, 4.08, 0.95, 0.66)
    pool2 = (6.72, 2.98, 0.95, 0.66)
    up2 = (9.75, 2.98, 0.95, 0.66)
    up1 = (12.78, 4.08, 0.95, 0.66)

    # 主干
    rounded_box(ax, *enc1, "编码器层 1", "ResBlock1D\n3 → 64, L=301",
                fc=COLORS["main_fill"], ec=COLORS["main_edge"])
    rounded_box(ax, *enc2, "编码器层 2", "ResBlock1D\n64 → 128, L/2",
                fc=COLORS["main_fill"], ec=COLORS["main_edge"])
    rounded_box(ax, *bottleneck, "瓶颈层", "ResBlock1D\n128 → 256, L/4",
                fc=COLORS["main_fill"], ec=COLORS["main_edge"])
    rounded_box(ax, *dec2, "解码器层 2", "Concat + ResBlock1D\n256 → 128, L/2",
                fc=COLORS["main_fill"], ec=COLORS["main_edge"], subtitle_size=8.1)
    rounded_box(ax, *dec1, "解码器层 1", "Concat + ResBlock1D\n128 → 64, L",
                fc=COLORS["main_fill"], ec=COLORS["main_edge"], subtitle_size=8.1)

    rounded_box(ax, *pool1, "池化 1", "MaxPool", fc="#F3F7FB", ec=COLORS["main_edge"],
                title_size=9.5, subtitle_size=7.8, radius=0.12)
    rounded_box(ax, *pool2, "池化 2", "MaxPool", fc="#F3F7FB", ec=COLORS["main_edge"],
                title_size=9.5, subtitle_size=7.8, radius=0.12)
    rounded_box(ax, *up2, "上采样 2", "ConvTranspose", fc="#F3F7FB", ec=COLORS["main_edge"],
                title_size=9.3, subtitle_size=7.5, radius=0.12)
    rounded_box(ax, *up1, "上采样 1", "ConvTranspose", fc="#F3F7FB", ec=COLORS["main_edge"],
                title_size=9.3, subtitle_size=7.5, radius=0.12)

    # 输入输出
    rounded_box(ax, 0.15, 4.65, 0.95, 1.15, "输入", "$x_t$\n(B, 3, 301)",
                fc=COLORS["main_fill"], ec=COLORS["main_edge"], title_size=11)
    rounded_box(ax, 14.0, 6.55, 1.1, 1.0, "输出头", "Conv1d + SiLU\nConv1d(64→3)",
                fc=COLORS["output_fill"], ec=COLORS["output_edge"], title_size=10.8, subtitle_size=7.8)
    rounded_box(ax, 14.15, 7.72, 0.8, 0.6, "输出", "$\\hat{\\epsilon}$",
                fc=COLORS["output_fill"], ec=COLORS["output_edge"], title_size=9.5, subtitle_size=7.2)

    # 条件分支
    rounded_box(ax, 0.35, 6.1, 1.2, 0.9, "条件输入", "$c$\n(B, 1, 301)",
                fc=COLORS["cond_fill"], ec=COLORS["cond_edge"], title_size=10.5)
    rounded_box(ax, 1.9, 6.02, 1.8, 1.05, "条件编码器", "Conv1d + SiLU\ncond_feat (64, L)",
                fc=COLORS["cond_fill"], ec=COLORS["cond_edge"], title_size=10.8, subtitle_size=7.9)
    rounded_box(ax, 4.52, 6.08, 1.55, 0.85, "Attn 条件 1", "原尺度特征",
                fc=COLORS["cond_fill"], ec=COLORS["cond_edge"], title_size=9.7, subtitle_size=7.6)
    rounded_box(ax, 7.25, 6.08, 1.55, 0.85, "Attn 条件 2", "MaxPool + 1×1 Conv",
                fc=COLORS["cond_fill"], ec=COLORS["cond_edge"], title_size=9.7, subtitle_size=7.4)
    rounded_box(ax, 10.0, 6.08, 1.72, 0.85, "Attn 条件 3", "再下采样 + 1×1 Conv",
                fc=COLORS["cond_fill"], ec=COLORS["cond_edge"], title_size=9.7, subtitle_size=7.1)

    rounded_box(ax, 4.85, 5.06, 0.9, 0.56, "Cross\nAttn", "",
                fc="#FFF6E8", ec=COLORS["cond_edge"], title_size=8.6, radius=0.1)
    rounded_box(ax, 7.6, 3.96, 0.9, 0.56, "Cross\nAttn", "",
                fc="#FFF6E8", ec=COLORS["cond_edge"], title_size=8.6, radius=0.1)
    rounded_box(ax, 10.45, 2.86, 0.9, 0.56, "Cross\nAttn", "",
                fc="#FFF6E8", ec=COLORS["cond_edge"], title_size=8.6, radius=0.1)

    # 时间分支
    rounded_box(ax, 0.45, 1.35, 1.05, 0.88, "时间步", "$t$",
                fc=COLORS["time_fill"], ec=COLORS["time_edge"], title_size=10.8)
    rounded_box(ax, 1.9, 1.28, 2.0, 1.02, "正弦时间嵌入", "Sinusoidal embedding",
                fc=COLORS["time_fill"], ec=COLORS["time_edge"], title_size=10.8, subtitle_size=7.8)
    rounded_box(ax, 4.35, 1.28, 1.95, 1.02, "MLP 映射", "$t_{emb}$ = 128 维",
                fc=COLORS["time_fill"], ec=COLORS["time_edge"], title_size=10.8, subtitle_size=7.8)

    # 主干箭头
    arrow(ax, (1.1, 5.22), (1.25, 5.22), COLORS["arrow"])
    arrow(ax, (3.35, 5.22), (3.65, 4.42), COLORS["arrow"], connectionstyle="arc3,rad=0.0")
    arrow(ax, (4.6, 4.2), (4.25, 4.12), COLORS["arrow"])
    arrow(ax, (5.35, 3.55), (6.72, 3.31), COLORS["arrow"], connectionstyle="arc3,rad=0.0")
    arrow(ax, (7.67, 3.22), (7.15, 3.05), COLORS["arrow"])
    arrow(ax, (9.5, 3.05), (9.75, 3.31), COLORS["arrow"])
    arrow(ax, (10.7, 4.2), (10.25, 4.12), COLORS["arrow"])
    arrow(ax, (12.45, 4.12), (12.78, 4.42), COLORS["arrow"])
    arrow(ax, (13.73, 5.22), (13.15, 5.22), COLORS["arrow"])
    arrow(ax, (14.2, 5.8), (14.55, 6.55), COLORS["arrow"])
    arrow(ax, (14.55, 7.55), (14.55, 7.72), COLORS["arrow"])

    # 条件分支箭头
    arrow(ax, (1.55, 6.55), (1.9, 6.55), COLORS["cond_edge"])
    arrow(ax, (3.7, 6.55), (4.52, 6.5), COLORS["cond_edge"])
    arrow(ax, (6.07, 6.5), (7.25, 6.5), COLORS["cond_edge"])
    arrow(ax, (8.8, 6.5), (10.0, 6.5), COLORS["cond_edge"])

    arrow(ax, (5.02, 6.08), (5.28, 5.62), COLORS["cond_edge"])
    arrow(ax, (8.03, 6.08), (8.03, 4.52), COLORS["cond_edge"])
    arrow(ax, (10.86, 6.08), (10.9, 3.42), COLORS["cond_edge"])

    arrow(ax, (5.3, 5.06), (4.85, 5.34), COLORS["cond_edge"])
    arrow(ax, (8.05, 3.96), (7.68, 4.14), COLORS["cond_edge"])
    arrow(ax, (10.8, 2.86), (10.3, 3.04), COLORS["cond_edge"])

    # 时间分支箭头与注入
    arrow(ax, (1.5, 1.78), (1.9, 1.78), COLORS["time_edge"])
    arrow(ax, (3.9, 1.78), (4.35, 1.78), COLORS["time_edge"])
    injection_targets = [
        (2.3, 4.65),
        (5.2, 3.55),
        (8.25, 2.45),
        (11.2, 3.55),
        (14.0, 4.65),
    ]
    for x, y in injection_targets:
        arrow(
            ax, (5.35, 2.3), (x, y),
            COLORS["time_edge"], lw=1.15, mutation=10,
            connectionstyle="arc3,rad=0.12", linestyle="--"
        )

    label(ax, 7.0, 1.92, "时间嵌入注入各残差块", COLORS["time_edge"], "#F4FBF5")

    # Skip 连接
    arrow(
        ax, (3.35, 5.22), (13.15, 5.22),
        COLORS["skip"], lw=1.5, mutation=11,
        connectionstyle="arc3,rad=0.18", linestyle="--"
    )
    arrow(
        ax, (6.45, 4.12), (10.25, 4.12),
        COLORS["skip"], lw=1.5, mutation=11,
        connectionstyle="arc3,rad=-0.05", linestyle="--"
    )
    label(ax, 8.55, 6.05, "Skip 1", COLORS["skip"], "#F7F4FC")
    label(ax, 8.25, 4.5, "Skip 2", COLORS["skip"], "#F7F4FC")

    # 说明文字
    ax.text(
        0.72, 0.32,
        "主干网络采用 3 级编解码结构；交叉注意力位于编码器 1、编码器 2 与瓶颈层；最终输出三通道噪声预测。",
        ha="left", va="center", fontsize=9.2, color=COLORS["muted"]
    )

    # 小图例
    legend_y = 0.58
    items = [
        ("主干特征流", COLORS["main_fill"], COLORS["main_edge"]),
        ("条件特征流", COLORS["cond_fill"], COLORS["cond_edge"]),
        ("时间步嵌入", COLORS["time_fill"], COLORS["time_edge"]),
        ("输出层", COLORS["output_fill"], COLORS["output_edge"]),
    ]
    x = 0.75
    for text, fc, ec in items:
        rect = patches.FancyBboxPatch(
            (x, legend_y), 0.38, 0.18,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=1.1, edgecolor=ec, facecolor=fc, zorder=3
        )
        ax.add_patch(rect)
        ax.text(x + 0.52, legend_y + 0.09, text, ha="left", va="center",
                fontsize=8.7, color=COLORS["muted"])
        x += 3.2

    plt.tight_layout()
    fig.savefig(PNG_PATH, dpi=240, bbox_inches="tight", facecolor=COLORS["bg"])
    fig.savefig(SVG_PATH, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)

    print(f"saved: {PNG_PATH}")
    print(f"saved: {SVG_PATH}")


if __name__ == "__main__":
    main()
