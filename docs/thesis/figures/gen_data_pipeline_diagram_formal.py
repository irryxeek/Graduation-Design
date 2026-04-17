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
PNG_PATH = OUT_DIR / "data_processing_pipeline_formal.png"
SVG_PATH = OUT_DIR / "data_processing_pipeline_formal.svg"
PDF_PATH = OUT_DIR / "data_processing_pipeline_formal.pdf"

COLORS = {
    "bg": "#FFFFFF",
    "text": "#111111",
    "muted": "#555555",
    "line": "#333333",
    "header": "#222222",
    "fill": "#F6F6F6",
    "fill_alt": "#FBFBFB",
}


def add_box(ax, x, y, w, h, title, subtitle="", fill="white", lw=1.2, title_size=11, subtitle_size=8.7):
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
            linespacing=1.3,
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


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9.6, 11.2))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 16.8)
    ax.axis("off")

    add_box(
        ax,
        2.1,
        14.25,
        4.2,
        0.95,
        "ATP 原始文件",
        "弯曲角 / 冲击参数 / 质量标志",
        fill=COLORS["fill_alt"],
    )
    add_box(
        ax,
        8.7,
        14.25,
        4.2,
        0.95,
        "WAP 原始文件",
        "温度 / 气压 / 湿度 / 海拔高度",
        fill=COLORS["fill_alt"],
    )

    merge_x = 7.5
    ax.plot([4.2, 4.2], [14.25, 13.65], color=COLORS["line"], lw=1.1, zorder=1)
    ax.plot([10.8, 10.8], [14.25, 13.65], color=COLORS["line"], lw=1.1, zorder=1)
    ax.plot([4.2, merge_x], [13.65, 13.65], color=COLORS["line"], lw=1.1, zorder=1)
    ax.plot([10.8, merge_x], [13.65, 13.65], color=COLORS["line"], lw=1.1, zorder=1)
    add_arrow(ax, (merge_x, 13.65), (merge_x, 13.05))

    steps = [
        ("文件配对", "按掩星事件文件名匹配 ATP 与 WAP"),
        ("变量提取", "读取 BA、impact、Temp、Pres、Shum 与高度信息"),
        ("质量控制", "保留 qc = 100，并过滤异常值与无效样本"),
        ("统一高度网格插值", "线性插值到 0-60 km 的 301 层标准高度网格"),
        ("对数变换与物理约束", "对 BA 和气压做 log10 变换，湿度裁剪为非负"),
        ("Z-Score 标准化", "仅使用训练集均值与标准差进行归一化"),
        ("数据集划分", "按 70% / 15% / 15% 划分 train、val、test"),
    ]

    step_x = 4.0
    step_w = 7.0
    step_h = 0.9
    step_gap = 0.42
    start_y = 12.05
    centers = []

    for idx, (title, subtitle) in enumerate(steps):
        y = start_y - idx * (step_h + step_gap)
        add_box(
            ax,
            step_x,
            y,
            step_w,
            step_h,
            title,
            subtitle,
            fill=COLORS["fill"],
            title_size=10.5,
            subtitle_size=8.4,
        )
        centers.append((step_x + step_w / 2, y + step_h / 2))

    for idx in range(len(centers) - 1):
        cx, cy = centers[idx]
        nx, ny = centers[idx + 1]
        add_arrow(ax, (cx, cy - step_h / 2 + 0.02), (nx, ny + step_h / 2 - 0.02))

    last_cx, last_cy = centers[-1]
    add_arrow(ax, (last_cx, last_cy - step_h / 2 + 0.02), (last_cx, 2.55))

    add_box(
        ax,
        1.8,
        1.25,
        5.2,
        1.05,
        "标准化数据数组",
        "train_x.npy / train_y.npy\nval_x.npy / val_y.npy / test_x.npy / test_y.npy",
        fill=COLORS["fill_alt"],
        title_size=10.2,
        subtitle_size=8.1,
    )
    add_box(
        ax,
        8.0,
        1.25,
        5.2,
        1.05,
        "元数据与归一化参数",
        "summary.json / split_meta.json / norm_params.npz",
        fill=COLORS["fill_alt"],
        title_size=10.2,
        subtitle_size=8.1,
    )

    ax.plot([7.5, 7.5], [2.55, 2.25], color=COLORS["line"], lw=1.1, zorder=1)
    ax.plot([4.4, 10.6], [2.25, 2.25], color=COLORS["line"], lw=1.1, zorder=1)
    add_arrow(ax, (4.4, 2.25), (4.4, 2.3))
    add_arrow(ax, (10.6, 2.25), (10.6, 2.3))

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
