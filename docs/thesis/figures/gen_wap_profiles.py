"""生成 WAP 温度、气压、湿度典型廓线图（仅导出 PNG）。"""

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
NPZ_PATH = Path("samples/demo_upload_atp_wap_2025_16.npz")
PNG_PATH = OUT_DIR / "wap_typical_profiles.png"


def select_representative_sample(x_log10: np.ndarray) -> int:
    rough = np.mean(np.abs(np.diff(x_log10, axis=1)), axis=1)
    surf = np.log10(10 ** x_log10[:, 0])
    rough_std = np.std(rough) if np.std(rough) > 0 else 1.0
    surf_std = np.std(surf) if np.std(surf) > 0 else 1.0
    score = np.abs(rough - np.median(rough)) / rough_std + np.abs(surf - np.median(surf)) / surf_std
    return int(np.argmin(score))


def style_axis(ax):
    ax.set_facecolor("white")
    ax.set_ylim(0, 60)
    ax.set_yticks(np.arange(0, 61, 10))
    ax.grid(True, which="major", color="#D8DEE8", linestyle="--", linewidth=0.8, alpha=0.9)
    ax.grid(True, which="minor", axis="x", color="#EEF2F6", linestyle=":", linewidth=0.6, alpha=0.9)
    for spine in ax.spines.values():
        spine.set_color("#98A2B3")
        spine.set_linewidth(0.95)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data = np.load(NPZ_PATH)
    x_log10 = data["x"]
    y = data["y"]
    heights = data["heights"]
    sample_indices = data["sample_indices"]

    sample_pos = select_representative_sample(x_log10)
    sample_id = int(sample_indices[sample_pos])

    temp = y[sample_pos, 0].astype(np.float64)
    pres_log10 = y[sample_pos, 1].astype(np.float64)
    shum = y[sample_pos, 2].astype(np.float64)
    pres = np.power(10.0, pres_log10)

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 5.1), sharey=True)
    fig.patch.set_facecolor("white")

    # 温度
    ax = axes[0]
    style_axis(ax)
    ax.plot(temp, heights, color="#C46B16", lw=2.2, solid_capstyle="round")
    ax.set_title("温度", fontsize=12.2, fontweight="bold", pad=8)
    ax.set_xlabel("K", fontsize=10.5)
    ax.set_ylabel("高度 / km", fontsize=11)
    ax.set_xlim(max(205, np.floor(np.nanmin(temp) - 2)), min(260, np.ceil(np.nanmax(temp) + 2)))
    ax.text(
        0.05, 0.05,
        f"{np.nanmin(temp):.1f} - {np.nanmax(temp):.1f} K",
        transform=ax.transAxes, ha="left", va="bottom",
        fontsize=8.2, color="#475467"
    )

    # 气压
    ax = axes[1]
    style_axis(ax)
    ax.plot(pres, heights, color="#1E4F7A", lw=2.2, solid_capstyle="round")
    ax.set_title("气压", fontsize=12.2, fontweight="bold", pad=8)
    ax.set_xlabel("hPa（对数坐标）", fontsize=10.5)
    ax.set_xscale("log")
    ax.set_xlim(1e-1, 2e3)
    ax.text(
        0.05, 0.05,
        f"{np.nanmax(pres):.1f} - {np.nanmin(pres):.2f} hPa",
        transform=ax.transAxes, ha="left", va="bottom",
        fontsize=8.2, color="#475467"
    )

    # 湿度
    ax = axes[2]
    style_axis(ax)
    ax.plot(shum, heights, color="#2C7A5A", lw=2.2, solid_capstyle="round")
    ax.set_title("湿度", fontsize=12.2, fontweight="bold", pad=8)
    ax.set_xlabel("g/kg", fontsize=10.5)
    ax.set_xlim(0, max(0.85, np.nanmax(shum) * 1.08))
    ax.text(
        0.05, 0.05,
        f"0 - {np.nanmax(shum):.3f} g/kg",
        transform=ax.transAxes, ha="left", va="bottom",
        fontsize=8.2, color="#475467"
    )

    plt.tight_layout(w_pad=1.5)
    fig.savefig(PNG_PATH, dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"saved: {PNG_PATH}")
    print(f"sample_id: {sample_id}")


if __name__ == "__main__":
    main()
