from pathlib import Path
import json

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


ROOT = Path("experiments/atp_wap_2025_hw4_hmon_g005")
LOG_PATH = ROOT / "enhanced_ro_diffusion_training_log.json"

OUT_DIR = Path("docs/thesis/figures")
PNG_PATH = OUT_DIR / "training_loss_curve_formal.png"
SVG_PATH = OUT_DIR / "training_loss_curve_formal.svg"
PDF_PATH = OUT_DIR / "training_loss_curve_formal.pdf"

COLORS = {
    "train": "#C23B22",
    "val": "#2A6FBB",
    "best": "#2A6FBB",
    "grid": "#D9D9D9",
    "spine": "#333333",
    "text": "#111111",
    "bg": "#FFFFFF",
}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log_data = json.loads(LOG_PATH.read_text(encoding="utf-8"))
    train_losses = np.array(log_data["train_losses"], dtype=float)
    val_losses = np.array(log_data["val_losses"], dtype=float)

    epochs = np.arange(1, len(train_losses) + 1)
    best_epoch = int(np.argmin(val_losses)) + 1
    best_val = float(np.min(val_losses))

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])

    ax.plot(
        epochs,
        train_losses,
        color=COLORS["train"],
        linewidth=1.7,
        label="训练损失",
    )
    ax.plot(
        epochs,
        val_losses,
        color=COLORS["val"],
        linewidth=1.7,
        linestyle="--",
        label="验证损失",
    )
    ax.scatter(
        [best_epoch],
        [best_val],
        s=28,
        facecolors="white",
        edgecolors=COLORS["best"],
        linewidths=1.2,
        zorder=4,
    )

    ax.set_xlim(1, len(epochs))
    ax.set_yscale("log")
    ax.set_xlabel("Epoch", fontsize=10.5, color=COLORS["text"])
    ax.set_ylabel("Loss", fontsize=10.5, color=COLORS["text"])

    ax.grid(True, which="major", axis="both", color=COLORS["grid"], linewidth=0.8, alpha=0.85)
    ax.grid(True, which="minor", axis="y", color=COLORS["grid"], linewidth=0.5, alpha=0.45)

    for spine in ax.spines.values():
        spine.set_color(COLORS["spine"])
        spine.set_linewidth(1.0)

    ax.tick_params(axis="both", labelsize=9.5, colors=COLORS["text"])
    ax.legend(frameon=False, fontsize=9.5, loc="upper right")

    plt.tight_layout()
    fig.savefig(PNG_PATH, dpi=320, bbox_inches="tight", facecolor=COLORS["bg"])
    fig.savefig(SVG_PATH, bbox_inches="tight", facecolor=COLORS["bg"])
    fig.savefig(PDF_PATH, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)

    print(f"best_epoch: {best_epoch}")
    print(f"best_val: {best_val:.6f}")
    print(f"saved: {PNG_PATH}")
    print(f"saved: {SVG_PATH}")
    print(f"saved: {PDF_PATH}")


if __name__ == "__main__":
    main()
