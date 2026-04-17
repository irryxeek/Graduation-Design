from pathlib import Path

import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt


matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False


OUT_DIR = Path("docs/thesis/figures")
PNG_PATH = OUT_DIR / "model_training_flow_formal.png"
SVG_PATH = OUT_DIR / "model_training_flow_formal.svg"
PDF_PATH = OUT_DIR / "model_training_flow_formal.pdf"

COLORS = {
    "bg": "#FFFFFF",
    "text": "#111111",
    "muted": "#555555",
    "line": "#333333",
    "fill": "#F7F7F7",
    "fill_alt": "#FCFCFC",
}


def add_box(ax, x, y, w, h, title, subtitle="", fill="white", lw=1.2, title_size=10.5, subtitle_size=8.4):
    box = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=lw,
        edgecolor=COLORS["line"],
        facecolor=fill,
        zorder=2,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h * 0.62,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=COLORS["text"],
        zorder=3,
    )
    if subtitle:
        ax.text(
            x + w / 2,
            y + h * 0.28,
            subtitle,
            ha="center",
            va="center",
            fontsize=subtitle_size,
            color=COLORS["muted"],
            linespacing=1.28,
            zorder=3,
        )


def add_arrow(ax, start, end):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(
            arrowstyle="-|>",
            lw=1.2,
            color=COLORS["line"],
            mutation_scale=12,
            shrinkA=4,
            shrinkB=4,
        ),
        zorder=1,
    )


def add_line(ax, start, end, lw=1.1):
    ax.plot([start[0], end[0]], [start[1], end[1]], color=COLORS["line"], lw=lw, zorder=1)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10.5, 13.2))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 19)
    ax.axis("off")

    add_box(
        ax,
        1.1,
        17.2,
        5.0,
        1.0,
        "训练集",
        "train_x.npy / train_y.npy",
        fill=COLORS["fill_alt"],
    )
    add_box(
        ax,
        8.9,
        17.2,
        5.0,
        1.0,
        "验证集",
        "val_x.npy / val_y.npy",
        fill=COLORS["fill_alt"],
    )

    add_box(
        ax,
        4.2,
        15.8,
        6.6,
        1.0,
        "ROMultiVarDataset 与 DataLoader",
        "batch_size = 64，训练集随机打乱，验证集顺序加载",
        fill=COLORS["fill"],
    )
    add_arrow(ax, (3.6, 17.2), (6.4, 16.8))
    add_arrow(ax, (11.4, 17.2), (8.6, 16.8))

    add_box(
        ax,
        4.2,
        14.35,
        6.6,
        1.0,
        "当前批次样本",
        "条件输入 c ∈ R^(1×301)，目标廓线 x0 ∈ R^(3×301)",
        fill=COLORS["fill"],
    )
    add_arrow(ax, (7.5, 15.8), (7.5, 15.35))

    add_box(
        ax,
        1.1,
        12.95,
        3.7,
        0.95,
        "随机采样时间步 t",
        "t ~ Uniform(0, T-1)",
        fill=COLORS["fill_alt"],
    )
    add_box(
        ax,
        10.2,
        12.95,
        3.7,
        0.95,
        "随机采样噪声 ε",
        "ε ~ N(0, I)",
        fill=COLORS["fill_alt"],
    )
    add_box(
        ax,
        4.2,
        12.65,
        6.6,
        1.1,
        "前向加噪",
        "利用 DiffusionSchedule 计算带噪样本 xt = q_sample(x0, t, ε)",
        fill=COLORS["fill"],
    )
    add_arrow(ax, (7.5, 14.35), (7.5, 13.75))
    add_arrow(ax, (4.8, 13.45), (4.35, 13.45))
    add_arrow(ax, (10.2, 13.45), (10.65, 13.45))

    add_box(
        ax,
        1.1,
        11.1,
        3.7,
        0.95,
        "条件输入分支",
        "弯曲角条件 c",
        fill=COLORS["fill_alt"],
    )
    add_line(ax, (7.5, 14.35), (7.5, 11.7))
    add_arrow(ax, (7.5, 11.7), (4.8, 11.55))

    add_box(
        ax,
        4.2,
        10.7,
        6.6,
        1.2,
        "增强版条件 U-Net",
        "输入 xt、t 与 c，预测噪声 eps_pred(xt, t, c)",
        fill=COLORS["fill"],
    )
    add_arrow(ax, (7.5, 12.65), (7.5, 11.9))

    add_box(
        ax,
        4.2,
        9.1,
        6.6,
        1.0,
        "损失计算",
        "变量加权 MSE（1:1:4）+ 湿度梯度约束（λ = 0.05）",
        fill=COLORS["fill"],
    )
    add_arrow(ax, (7.5, 10.7), (7.5, 10.1))

    add_box(
        ax,
        4.2,
        7.65,
        6.6,
        1.0,
        "反向传播与参数更新",
        "loss.backward → 梯度裁剪 → AdamW 更新模型参数",
        fill=COLORS["fill"],
    )
    add_arrow(ax, (7.5, 9.1), (7.5, 8.65))

    add_box(
        ax,
        4.2,
        6.2,
        6.6,
        1.0,
        "轮次结束后验证",
        "验证集执行相同前向过程，不更新参数",
        fill=COLORS["fill"],
    )
    add_arrow(ax, (7.5, 7.65), (7.5, 7.2))

    add_box(
        ax,
        4.2,
        4.75,
        6.6,
        1.0,
        "验证监控",
        "统计平均验证损失，并监控湿度分量验证损失",
        fill=COLORS["fill"],
    )
    add_arrow(ax, (7.5, 6.2), (7.5, 5.75))

    add_box(
        ax,
        1.1,
        3.15,
        4.1,
        1.05,
        "最佳模型保存",
        "根据 Early Stopping 保存最优权重",
        fill=COLORS["fill_alt"],
    )
    add_box(
        ax,
        5.45,
        3.15,
        4.1,
        1.05,
        "检查点保存",
        "每 10 轮保存一次训练检查点",
        fill=COLORS["fill_alt"],
    )
    add_box(
        ax,
        9.8,
        3.15,
        4.1,
        1.05,
        "训练终止条件",
        "patience = 15，连续无改善则早停",
        fill=COLORS["fill_alt"],
    )

    add_line(ax, (7.5, 4.75), (7.5, 4.45))
    add_line(ax, (3.15, 4.45), (11.85, 4.45))
    add_arrow(ax, (3.15, 4.45), (3.15, 4.2))
    add_arrow(ax, (7.5, 4.45), (7.5, 4.2))
    add_arrow(ax, (11.85, 4.45), (11.85, 4.2))

    plt.tight_layout()
    fig.savefig(PNG_PATH, dpi=320, bbox_inches="tight", facecolor=COLORS["bg"])
    fig.savefig(SVG_PATH, bbox_inches="tight", facecolor=COLORS["bg"])
    fig.savefig(PDF_PATH, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)

    print(f"saved: {PNG_PATH}")
    print(f"saved: {SVG_PATH}")
    print(f"saved: {PDF_PATH}")


if __name__ == "__main__":
    main()
