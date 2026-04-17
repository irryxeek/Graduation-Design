"""生成更简洁的 ATP 弯曲角样例廓线图（仅导出 PNG）。"""

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
PNG_PATH = OUT_DIR / "atp_bending_angle_profile_clean.png"


def select_representative_sample(x_log10: np.ndarray) -> int:
    """选择一个相对典型的样本。"""
    rough = np.mean(np.abs(np.diff(x_log10, axis=1)), axis=1)
    surf = np.log10(10 ** x_log10[:, 0])
    rough_std = np.std(rough) if np.std(rough) > 0 else 1.0
    surf_std = np.std(surf) if np.std(surf) > 0 else 1.0
    score = np.abs(rough - np.median(rough)) / rough_std + np.abs(surf - np.median(surf)) / surf_std
    return int(np.argmin(score))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data = np.load(NPZ_PATH)
    x_log10 = data["x"]          # log10 bending angle
    heights = data["heights"]    # km
    sample_indices = data["sample_indices"]

    sample_pos = select_representative_sample(x_log10)
    sample_id = int(sample_indices[sample_pos])
    bend_ang_rad = np.power(10.0, x_log10[sample_pos].astype(np.float64))

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # 主曲线
    ax.plot(
        bend_ang_rad,
        heights,
        color="#1E4F7A",
        lw=2.2,
        solid_capstyle="round",
        zorder=3,
    )

    # 轴与网格
    ax.set_xscale("log")
    ax.set_xlim(3e-6, 6e-2)
    ax.set_ylim(0, 60)
    ax.set_xlabel("弯曲角 / rad（对数坐标）", fontsize=11)
    ax.set_ylabel("高度 / km", fontsize=11)
    ax.set_yticks(np.arange(0, 61, 10))
    ax.grid(True, which="major", color="#D8DEE8", linestyle="--", linewidth=0.8, alpha=0.9)
    ax.grid(True, which="minor", axis="x", color="#EEF2F6", linestyle=":", linewidth=0.6, alpha=0.9)

    # 图内仅保留最少说明，避免与论文图题重复
    ax.text(
        0.98,
        0.96,
        f"sample index = {sample_id}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.3,
        color="#667085",
    )
    ax.text(
        0.03,
        0.08,
        f"0 km: {bend_ang_rad[0]:.2e} rad\n30 km: {bend_ang_rad[150]:.2e} rad\n60 km: {bend_ang_rad[-1]:.2e} rad",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.3,
        color="#475467",
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.9),
    )

    # 美化边框
    for spine in ax.spines.values():
        spine.set_color("#98A2B3")
        spine.set_linewidth(0.95)

    plt.tight_layout()
    fig.savefig(PNG_PATH, dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"saved: {PNG_PATH}")
    print(f"sample_id: {sample_id}")


if __name__ == "__main__":
    main()
