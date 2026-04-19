from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False

OUT_DIR = Path("docs/thesis/figures")
PNG_PATH = OUT_DIR / "main_results_visual_formal.png"
SVG_PATH = OUT_DIR / "main_results_visual_formal.svg"
PDF_PATH = OUT_DIR / "main_results_visual_formal.pdf"

LATEST_RESULTS = {
    "temperature": {"label": "温度", "rmse": 0.7704, "bias": -0.0754, "cc": 0.7250},
    "pressure": {"label": "气压", "rmse": 0.0821, "bias": -0.0154, "cc": 0.9991},
    "humidity": {"label": "湿度", "rmse": 0.8482, "bias": 0.0010, "cc": 0.8270},
}

COLORS = {
    "temperature": "#C23B22",
    "pressure": "#2A6FBB",
    "humidity": "#2F8F5B",
    "grid": "#D9D9D9",
    "text": "#111111",
    "muted": "#555555",
    "spine": "#333333",
    "bg": "#FFFFFF",
}


def style_axis(ax):
    ax.set_facecolor(COLORS["bg"])
    ax.grid(True, axis="y", color=COLORS["grid"], linewidth=0.8, alpha=0.85)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(COLORS["spine"])
        spine.set_linewidth(1.0)
    ax.tick_params(axis="both", labelsize=9.5, colors=COLORS["text"])


def add_value_labels(ax, bars, fmt="{:.4f}", offset_ratio=0.025):
    y_min, y_max = ax.get_ylim()
    offset = (y_max - y_min) * offset_ratio
    for bar in bars:
        height = bar.get_height()
        va = "bottom" if height >= 0 else "top"
        y = height + offset if height >= 0 else height - offset
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            fmt.format(height),
            ha="center",
            va=va,
            fontsize=8.4,
            color=COLORS["muted"],
        )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    variables = [
        ("temperature", "温度"),
        ("pressure", "气压"),
        ("humidity", "湿度"),
    ]
    labels = [label for _, label in variables]
    x = np.arange(len(labels))
    bar_colors = [COLORS[key] for key, _ in variables]

    rmse = np.array([LATEST_RESULTS[key]["rmse"] for key, _ in variables], dtype=float)
    bias = np.array([LATEST_RESULTS[key]["bias"] for key, _ in variables], dtype=float)
    cc = np.array([LATEST_RESULTS[key]["cc"] for key, _ in variables], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 4.6))
    fig.patch.set_facecolor(COLORS["bg"])

    bias_limit = max(abs(float(np.min(bias))), abs(float(np.max(bias)))) * 1.35

    metrics = [
        ("RMSE", rmse, (0, max(rmse) * 1.18)),
        ("Bias", bias, (-bias_limit, bias_limit)),
        ("CC", cc, (0, 1.08)),
    ]

    for ax, (metric_name, values, ylim) in zip(axes, metrics):
        bars = ax.bar(
            x,
            values,
            width=0.56,
            color=bar_colors,
            edgecolor=COLORS["spine"],
            linewidth=0.8,
        )
        ax.set_xticks(x, labels)
        ax.set_ylim(*ylim)
        ax.set_ylabel(metric_name, fontsize=10.5, color=COLORS["text"])
        style_axis(ax)
        if metric_name == "Bias":
            ax.axhline(0, color=COLORS["spine"], linewidth=1.0)
        add_value_labels(ax, bars)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLORS[key], ec=COLORS["spine"], lw=0.8)
        for key, _ in variables
    ]
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=9.6,
        bbox_to_anchor=(0.5, 1.02),
    )

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(PNG_PATH, dpi=320, bbox_inches="tight", facecolor=COLORS["bg"])
    fig.savefig(SVG_PATH, bbox_inches="tight", facecolor=COLORS["bg"])
    fig.savefig(PDF_PATH, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)

    print(f"saved: {PNG_PATH}")
    print(f"saved: {SVG_PATH}")
    print(f"saved: {PDF_PATH}")


if __name__ == "__main__":
    main()
