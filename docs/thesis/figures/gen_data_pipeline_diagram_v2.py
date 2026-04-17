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
PNG_PATH = OUT_DIR / "data_processing_pipeline_v2.png"
SVG_PATH = OUT_DIR / "data_processing_pipeline_v2.svg"

COLORS = {
    "bg": "#FCFDFC",
    "panel": "#F4F7F6",
    "text": "#1E293B",
    "muted": "#607087",
    "arrow": "#51606F",
    "input_fill": "#EAF3FF",
    "input_edge": "#5C84B8",
    "process_fill": "#F7FAFC",
    "process_edge": "#8A96A3",
    "key_fill": "#E9F7EE",
    "key_edge": "#4E9667",
    "output_fill": "#FFF3E4",
    "output_edge": "#D18B2E",
}


def add_round_box(ax, x, y, w, h, facecolor, edgecolor, linewidth=1.5, radius=0.12, zorder=2):
    box = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.03,rounding_size={radius}",
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor,
        zorder=zorder,
    )
    box.set_path_effects(
        [
            pe.SimplePatchShadow(offset=(1.2, -1.2), alpha=0.07),
            pe.Normal(),
        ]
    )
    ax.add_patch(box)
    return box


def add_stage_label(ax, x, y, w, text):
    chip = patches.FancyBboxPatch(
        (x, y),
        w,
        0.46,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=0.0,
        edgecolor="none",
        facecolor=COLORS["panel"],
        zorder=0,
    )
    ax.add_patch(chip)
    ax.text(
        x + 0.16,
        y + 0.23,
        text,
        ha="left",
        va="center",
        fontsize=10.0,
        fontweight="bold",
        color=COLORS["muted"],
        zorder=1,
    )


def add_text_box(ax, x, y, w, h, title, subtitle, facecolor, edgecolor, title_size=11, subtitle_size=8.3):
    add_round_box(ax, x, y, w, h, facecolor, edgecolor)
    ax.text(
        x + w / 2,
        y + h * 0.66,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=COLORS["text"],
        zorder=3,
    )
    ax.text(
        x + w / 2,
        y + h * 0.28,
        subtitle,
        ha="center",
        va="center",
        fontsize=subtitle_size,
        color=COLORS["muted"],
        linespacing=1.34,
        zorder=3,
    )


def add_arrow(ax, start, end, rad=0.0):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(
            arrowstyle="-|>",
            color=COLORS["arrow"],
            lw=1.8,
            mutation_scale=13,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=7,
            shrinkB=7,
        ),
        zorder=1,
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(15.6, 7.1))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    ax.set_xlim(0, 15.6)
    ax.set_ylim(0, 7.1)
    ax.axis("off")

    ax.text(
        0.55,
        6.72,
        "ATP+WAP 数据处理流水线",
        fontsize=16,
        fontweight="bold",
        color=COLORS["text"],
        ha="left",
        va="center",
    )
    ax.text(
        0.55,
        6.34,
        "FY-3D GNOS 2025 年 1-6 月配对样本构建与标准化流程",
        fontsize=9.8,
        color=COLORS["muted"],
        ha="left",
        va="center",
    )

    add_stage_label(ax, 0.55, 5.60, 2.05, "输入数据")
    add_stage_label(ax, 2.86, 5.60, 8.95, "处理流水线")
    add_stage_label(ax, 12.08, 5.60, 2.95, "输出产物")

    add_text_box(
        ax,
        0.62,
        3.90,
        1.95,
        1.05,
        "ATP 原始文件",
        "弯曲角\n冲击参数 / qc 标志",
        COLORS["input_fill"],
        COLORS["input_edge"],
    )
    add_text_box(
        ax,
        0.62,
        2.35,
        1.95,
        1.05,
        "WAP 原始文件",
        "温度 / 气压 / 湿度\n海拔高度",
        COLORS["input_fill"],
        COLORS["input_edge"],
    )

    steps = [
        ("文件配对", "按掩星事件文件名\n匹配 ATP 与 WAP"),
        ("变量提取", "读取 BA、impact、Temp\nPres、Shum 与高度"),
        ("质量控制", "保留 qc=100\n并过滤异常值"),
        ("统一高度网格", "线性插值到\n0-60 km / 301 层"),
        ("非线性变换", "log10(BA)\nlog10(P)"),
        ("物理约束", "湿度裁剪到非负\n剔除无效样本"),
        ("Z-Score 标准化", "仅使用训练集\n均值与标准差"),
        ("数据集划分", "train / val / test\n70% / 15% / 15%"),
    ]

    start_x = 2.98
    box_y = 2.78
    box_w = 0.98
    gap = 0.14
    box_h = 1.28
    key_steps = {2, 3, 4, 5, 6, 7}
    centers = []

    for idx, (title, subtitle) in enumerate(steps):
        x = start_x + idx * (box_w + gap)
        facecolor = COLORS["key_fill"] if idx in key_steps else COLORS["process_fill"]
        edgecolor = COLORS["key_edge"] if idx in key_steps else COLORS["process_edge"]
        add_text_box(
            ax,
            x,
            box_y,
            box_w,
            box_h,
            title,
            subtitle,
            facecolor,
            edgecolor,
            title_size=9.5,
            subtitle_size=7.45,
        )
        centers.append((x + box_w / 2, box_y + box_h / 2))

    add_arrow(ax, (2.57, 4.35), (start_x, 3.55), rad=-0.06)
    add_arrow(ax, (2.57, 2.88), (start_x, 3.18), rad=0.06)

    for idx in range(len(centers) - 1):
        left_x, left_y = centers[idx]
        right_x, right_y = centers[idx + 1]
        add_arrow(ax, (left_x + box_w / 2 - 0.06, left_y), (right_x - box_w / 2 + 0.06, right_y))

    add_text_box(
        ax,
        12.16,
        3.65,
        2.55,
        0.96,
        "标准化数组",
        "train_x.npy / train_y.npy\nval_x.npy / val_y.npy\ntest_x.npy / test_y.npy",
        COLORS["output_fill"],
        COLORS["output_edge"],
        title_size=10.2,
        subtitle_size=7.5,
    )
    add_text_box(
        ax,
        12.16,
        2.10,
        2.55,
        0.96,
        "元数据与归一化参数",
        "summary.json / split_meta.json\nnorm_params.npz",
        COLORS["output_fill"],
        COLORS["output_edge"],
        title_size=10.0,
        subtitle_size=7.55,
    )

    right_start = start_x + (len(steps) - 1) * (box_w + gap) + box_w
    add_arrow(ax, (right_start, 3.63), (12.16, 4.10), rad=-0.03)
    add_arrow(ax, (right_start, 3.18), (12.16, 2.58), rad=0.03)

    ax.text(
        0.70,
        1.30,
        "说明：ATP 提供弯曲角条件输入，WAP 提供温度、气压和湿度监督标签；"
        "经过配对、质控、插值、变换、标准化与数据集划分后，得到可直接用于训练与评估的样本。",
        fontsize=8.9,
        color=COLORS["muted"],
        ha="left",
        va="center",
    )
    ax.text(
        0.70,
        0.90,
        "数据范围：FY-3D GNOS ATP+WAP，2025 年 1-6 月，共 64,116 个有效样本",
        fontsize=8.9,
        color=COLORS["muted"],
        ha="left",
        va="center",
    )

    legend_items = [
        ("输入文件", COLORS["input_fill"], COLORS["input_edge"]),
        ("常规处理", COLORS["process_fill"], COLORS["process_edge"]),
        ("关键处理", COLORS["key_fill"], COLORS["key_edge"]),
        ("输出产物", COLORS["output_fill"], COLORS["output_edge"]),
    ]

    legend_x = 0.72
    legend_y = 0.34
    for label, fill, edge in legend_items:
        chip = patches.FancyBboxPatch(
            (legend_x, legend_y),
            0.34,
            0.19,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            linewidth=1.0,
            edgecolor=edge,
            facecolor=fill,
            zorder=2,
        )
        ax.add_patch(chip)
        ax.text(
            legend_x + 0.46,
            legend_y + 0.095,
            label,
            ha="left",
            va="center",
            fontsize=8.4,
            color=COLORS["muted"],
            zorder=3,
        )
        legend_x += 2.72

    plt.tight_layout()
    fig.savefig(PNG_PATH, dpi=300, bbox_inches="tight", facecolor=COLORS["bg"])
    fig.savefig(SVG_PATH, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)

    print(f"saved: {PNG_PATH}")
    print(f"saved: {SVG_PATH}")


if __name__ == "__main__":
    main()
