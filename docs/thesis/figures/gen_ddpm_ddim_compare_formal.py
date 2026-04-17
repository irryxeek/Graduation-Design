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
PNG_PATH = OUT_DIR / "ddpm_ddim_compare_formal.png"
SVG_PATH = OUT_DIR / "ddpm_ddim_compare_formal.svg"
PDF_PATH = OUT_DIR / "ddpm_ddim_compare_formal.pdf"

COLORS = {
    "bg": "#FFFFFF",
    "text": "#111111",
    "muted": "#555555",
    "line": "#333333",
    "fill": "#F7F7F7",
    "fill_alt": "#FCFCFC",
    "header": "#ECECEC",
}


def add_box(ax, x, y, w, h, title, subtitle="", fill="white", lw=1.2, title_size=10.2, subtitle_size=8.3):
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
        y + h * 0.63,
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


def draw_column(ax, x0, width, header, steps, y_start=15.9, step_h=1.1, gap=0.55):
    add_box(ax, x0, 17.2, width, 0.9, header, fill=COLORS["header"], title_size=11.0, subtitle_size=8.0)
    centers = []
    y = y_start
    for title, subtitle, fill in steps:
        add_box(ax, x0, y, width, step_h, title, subtitle, fill=fill)
        centers.append((x0 + width / 2, y + step_h / 2))
        y -= step_h + gap

    for idx in range(len(centers) - 1):
        cx, cy = centers[idx]
        nx, ny = centers[idx + 1]
        add_arrow(ax, (cx, cy - step_h / 2 + 0.02), (nx, ny + step_h / 2 - 0.02))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11.4, 12.2))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 19)
    ax.axis("off")

    ddpm_steps = [
        ("输入条件", "加载模型权重，输入弯曲角条件 c", COLORS["fill_alt"]),
        ("初始化噪声", "从高斯噪声 xT ~ N(0, I) 开始", COLORS["fill"]),
        ("完整时间步迭代", "t = T-1, T-2, ..., 0", COLORS["fill"]),
        ("单步噪声预测", "模型预测 eps_pred(xt, t, c)", COLORS["fill"]),
        ("反向去噪更新", "按 DDPM 公式计算 xt-1", COLORS["fill"]),
        ("加入随机噪声", "t > 0 时加入 sigma_t * z", COLORS["fill"]),
        ("输出结果", "得到标准化空间预测廓线", COLORS["fill_alt"]),
        ("采样特征", "1000 步，随机采样，推理耗时较高", COLORS["fill_alt"]),
    ]

    ddim_steps = [
        ("输入条件", "加载模型权重，输入弯曲角条件 c", COLORS["fill_alt"]),
        ("初始化噪声", "从高斯噪声 xT ~ N(0, I) 开始", COLORS["fill"]),
        ("子序列时间步", "均匀选取 50 个采样时间步", COLORS["fill"]),
        ("单步噪声预测", "模型预测 eps_pred(xt, t, c)", COLORS["fill"]),
        ("估计 pred_x0", "由 xt 与 eps_pred 反推出 x0", COLORS["fill"]),
        ("确定性跳步更新", "eta = 0 时直接更新到 t'", COLORS["fill"]),
        ("输出结果", "得到标准化空间预测廓线", COLORS["fill_alt"]),
        ("采样特征", "50 步，确定性采样，速度约快 20 倍", COLORS["fill_alt"]),
    ]

    draw_column(ax, 1.1, 5.8, "DDPM", ddpm_steps)
    draw_column(ax, 9.1, 5.8, "DDIM", ddim_steps)

    add_box(
        ax,
        5.3,
        0.95,
        5.4,
        1.0,
        "共同输出后处理",
        "反标准化得到温度、气压和湿度预测结果",
        fill=COLORS["header"],
    )
    add_arrow(ax, (4.0, 2.7), (7.0, 1.95))
    add_arrow(ax, (12.0, 2.7), (9.0, 1.95))

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
