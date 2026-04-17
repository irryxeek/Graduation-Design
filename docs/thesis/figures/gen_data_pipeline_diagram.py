from pathlib import Path

import matplotlib
import matplotlib.patches as patches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt


matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False


OUT_DIR = Path("docs/thesis/figures")
PNG_PATH = OUT_DIR / "data_processing_pipeline.png"
SVG_PATH = OUT_DIR / "data_processing_pipeline.svg"

COLORS = {
    "bg": "#FFFFFF",
    "text": "#1F2937",
    "muted": "#667085",
    "arrow": "#5B6472",
    "section": "#F3F6FA",
    "input_fill": "#EAF2FB",
    "input_edge": "#5D87B8",
    "process_fill": "#F8FAFC",
    "process_edge": "#8A97A8",
    "highlight_fill": "#EAF7EF",
    "highlight_edge": "#4F9D69",
    "output_fill": "#FFF3E2",
    "output_edge": "#D28A22",
}


def add_box(
    ax,
    x,
    y,
    w,
    h,
    title,
    subtitle,
    facecolor,
    edgecolor,
    title_size=11,
    subtitle_size=8.5,
):
    box = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.03,rounding_size=0.10",
        linewidth=1.5,
        edgecolor=edgecolor,
        facecolor=facecolor,
        zorder=3,
    )
    box.set_path_effects(
        [
            pe.SimplePatchShadow(offset=(1.3, -1.3), alpha=0.08),
            pe.Normal(),
        ]
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h * 0.64,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=COLORS["text"],
        zorder=4,
    )
    ax.text(
        x + w / 2,
        y + h * 0.28,
        subtitle,
        ha="center",
        va="center",
        fontsize=subtitle_size,
        color=COLORS["muted"],
        zorder=4,
        linespacing=1.35,
    )


def add_arrow(ax, start, end, rad=0.0):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(
            arrowstyle="-|>",
            color=COLORS["arrow"],
            lw=1.7,
            mutation_scale=13,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=6,
            shrinkB=6,
        ),
        zorder=2,
    )


def add_section(ax, x, y, w, h, label):
    section = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.015,rounding_size=0.08",
        linewidth=0,
        facecolor=COLORS["section"],
        zorder=0,
    )
    ax.add_patch(section)
    ax.text(
        x + 0.16,
        y + h / 2,
        label,
        ha="left",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color=COLORS["muted"],
        zorder=1,
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(15.0, 6.2))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    ax.set_xlim(0, 15.0)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    add_section(ax, 0.50, 5.25, 2.05, 0.42, "输入数据")
    add_section(ax, 2.85, 5.25, 8.60, 0.42, "数据处理流水线")
    add_section(ax, 11.75, 5.25, 2.65, 0.42, "输出结果")

    add_box(
        ax,
        0.58,
        3.55,
        1.88,
        0.95,
        "ATP 原始文件",
        "弯曲角\n冲击参数 / 质量标志",
        COLORS["input_fill"],
        COLORS["input_edge"],
    )
    add_box(
        ax,
        0.58,
        2.05,
        1.88,
        0.95,
        "WAP 原始文件",
        "温度 / 气压 / 湿度\n高度坐标",
        COLORS["input_fill"],
        COLORS["input_edge"],
    )

    steps = [
        ("文件配对", "按文件名匹配\nATP 与 WAP 事件"),
        ("变量提取", "读取弯曲角、温度、\n气压、湿度与高度"),
        ("质量控制", "qc=100\n物理范围与有效点检查"),
        ("统一高度网格", "线性插值到\n0-60 km / 301 层"),
        ("变换与约束", "log10(BA)、log10(P)\n湿度非负裁剪"),
        ("标准化", "基于训练集统计量\n执行 Z-Score"),
        ("数据集划分", "train / val / test\n并保存统计量"),
    ]

    start_x = 2.98
    box_w = 1.12
    gap = 0.16
    box_y = 2.45
    box_h = 1.18
    highlight_idx = {2, 4, 5, 6}
    centers = []

    for idx, (title, subtitle) in enumerate(steps):
        x = start_x + idx * (box_w + gap)
        if idx in highlight_idx:
            fill = COLORS["highlight_fill"]
            edge = COLORS["highlight_edge"]
        else:
            fill = COLORS["process_fill"]
            edge = COLORS["process_edge"]
        add_box(
            ax,
            x,
            box_y,
            box_w,
            box_h,
            title,
            subtitle,
            fill,
            edge,
            title_size=10.0,
            subtitle_size=7.7,
        )
        centers.append((x + box_w / 2, box_y + box_h / 2))

    add_arrow(ax, (2.46, 4.05), (start_x, 3.16), rad=-0.08)
    add_arrow(ax, (2.46, 2.52), (start_x, 2.95), rad=0.08)

    for idx in range(len(centers) - 1):
        left_x, left_y = centers[idx]
        right_x, right_y = centers[idx + 1]
        add_arrow(
            ax,
            (left_x + box_w / 2 - 0.07, left_y),
            (right_x - box_w / 2 + 0.07, right_y),
        )

    add_box(
        ax,
        11.86,
        3.10,
        2.30,
        0.88,
        "标准化数组",
        "train_x / train_y\nval_x / val_y / test_x / test_y",
        COLORS["output_fill"],
        COLORS["output_edge"],
        title_size=10.4,
        subtitle_size=7.6,
    )
    add_box(
        ax,
        11.86,
        1.90,
        2.30,
        0.88,
        "元数据与统计量",
        "summary.json / split_meta.json\nstats.npy 或 norm_params.npz",
        COLORS["output_fill"],
        COLORS["output_edge"],
        title_size=10.2,
        subtitle_size=7.5,
    )

    right_start = start_x + (len(steps) - 1) * (box_w + gap) + box_w
    add_arrow(ax, (right_start, 3.18), (11.86, 3.55), rad=-0.03)
    add_arrow(ax, (right_start, 2.92), (11.86, 2.30), rad=0.03)

    ax.text(
        2.98,
        1.28,
        "图示说明：ATP 提供弯曲角条件输入，WAP 提供温度、气压和湿度标签；"
        "经过配对、质控、插值、对数变换与标准化后，形成可直接用于训练与评估的数据集。",
        ha="left",
        va="center",
        fontsize=8.8,
        color=COLORS["muted"],
    )

    legend_items = [
        ("输入文件", COLORS["input_fill"], COLORS["input_edge"]),
        ("常规处理", COLORS["process_fill"], COLORS["process_edge"]),
        ("关键处理", COLORS["highlight_fill"], COLORS["highlight_edge"]),
        ("输出产物", COLORS["output_fill"], COLORS["output_edge"]),
    ]
    legend_x = 0.72
    legend_y = 0.50
    for text, fill, edge in legend_items:
        chip = patches.FancyBboxPatch(
            (legend_x, legend_y),
            0.33,
            0.18,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            linewidth=1.0,
            edgecolor=edge,
            facecolor=fill,
            zorder=3,
        )
        ax.add_patch(chip)
        ax.text(
            legend_x + 0.45,
            legend_y + 0.09,
            text,
            ha="left",
            va="center",
            fontsize=8.4,
            color=COLORS["muted"],
        )
        legend_x += 2.55

    plt.tight_layout()
    fig.savefig(PNG_PATH, dpi=280, bbox_inches="tight", facecolor=COLORS["bg"])
    fig.savefig(SVG_PATH, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)

    print(f"saved: {PNG_PATH}")
    print(f"saved: {SVG_PATH}")


if __name__ == "__main__":
    main()
