"""
严格补跑论文表 5-4 的损失函数消融实验。

实验口径:
  - 数据集: Data/Processed_ATP_WAP_2025
  - 模型: enhanced multi-output conditional U-Net
  - 训练: epochs=50, batch_size=64, lr=1e-4, patience=15
  - 监控: humidity 通道验证损失
  - 评估: DDPM 1000 步, 全测试集, standardized 空间

三组配置:
  1. 等权无约束              [1, 1, 1], grad=0.0
  2. 加权无约束              [1, 1, 4], grad=0.0
  3. 加权 + 梯度约束(最终)    [1, 1, 4], grad=0.05
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ro_retrieval.config import (  # noqa: E402
    BATCH_SIZE,
    LEARNING_RATE,
    PAPER_EPOCHS,
    PAPER_MONITOR_TARGET,
    PAPER_PATIENCE,
    PAPER_PROCESSED_DIR,
    PROJECT_ROOT,
)
from ro_retrieval.training.trainer import Trainer  # noqa: E402
from src.evaluate import main as evaluate_main  # noqa: E402


@dataclass(frozen=True)
class AblationConfig:
    name: str
    label: str
    var_weights: list[float]
    humidity_grad_weight: float


TABLE_54_TARGETS = {
    "equal_no_constraint": {
        "temperature_cc": 0.7856,
        "pressure_cc": 0.9992,
        "humidity_cc": 0.6104,
        "humidity_rmse": 0.8712,
    },
    "weighted_no_constraint": {
        "temperature_cc": 0.7831,
        "pressure_cc": 0.9990,
        "humidity_cc": 0.6783,
        "humidity_rmse": 0.8156,
    },
    "weighted_with_grad": {
        "temperature_cc": 0.7820,
        "pressure_cc": 0.9990,
        "humidity_cc": 0.6960,
        "humidity_rmse": 0.7996,
    },
}


EXPERIMENTS = [
    AblationConfig(
        name="equal_no_constraint",
        label="等权无约束",
        var_weights=[1.0, 1.0, 1.0],
        humidity_grad_weight=0.0,
    ),
    AblationConfig(
        name="weighted_no_constraint",
        label="加权无约束",
        var_weights=[1.0, 1.0, 4.0],
        humidity_grad_weight=0.0,
    ),
    AblationConfig(
        name="weighted_with_grad",
        label="加权+梯度约束（最终）",
        var_weights=[1.0, 1.0, 4.0],
        humidity_grad_weight=0.05,
    ),
]


def parse_args():
    parser = argparse.ArgumentParser(description="严格补跑论文表 5-4 消融实验")
    parser.add_argument("--data_dir", type=str, default=PAPER_PROCESSED_DIR)
    parser.add_argument("--epochs", type=int, default=PAPER_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--patience", type=int, default=PAPER_PATIENCE)
    parser.add_argument("--monitor_target", type=str, default=PAPER_MONITOR_TARGET)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="实验输出根目录; 默认写入 experiments/table5_4_ablation_<timestamp>",
    )
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _make_output_root(output_root: str | None) -> Path:
    if output_root:
        root = Path(output_root)
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        root = Path(PROJECT_ROOT) / "experiments" / f"table5_4_ablation_{timestamp}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _load_summary(report_path: Path) -> dict:
    data = json.loads(report_path.read_text(encoding="utf-8"))
    summary = data.get("summary", {})
    return {
        "temperature_cc": float(summary["temperature"]["cc_mean"]),
        "pressure_cc": float(summary["pressure"]["cc_mean"]),
        "humidity_cc": float(summary["humidity"]["cc_mean"]),
        "humidity_rmse": float(summary["humidity"]["rmse_mean"]),
        "report_path": str(report_path),
    }


def _build_markdown_table(rows: list[dict]) -> str:
    lines = [
        "# 表 5-4 消融实验补跑结果",
        "",
        "| 配置 | 变量权重 | 梯度约束 | 温度 CC | 气压 CC | 湿度 CC | 湿度 RMSE | 论文湿度 CC | 论文湿度 RMSE |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        target = TABLE_54_TARGETS[row["name"]]
        lines.append(
            "| {label} | {weights} | {grad:.2f} | {temperature_cc:.4f} | "
            "{pressure_cc:.4f} | {humidity_cc:.4f} | {humidity_rmse:.4f} | "
            "{target_hcc:.4f} | {target_hrmse:.4f} |".format(
                label=row["label"],
                weights=str(row["var_weights"]),
                grad=row["humidity_grad_weight"],
                temperature_cc=row["temperature_cc"],
                pressure_cc=row["pressure_cc"],
                humidity_cc=row["humidity_cc"],
                humidity_rmse=row["humidity_rmse"],
                target_hcc=target["humidity_cc"],
                target_hrmse=target["humidity_rmse"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def run_single_experiment(cfg: AblationConfig, args, root: Path) -> dict:
    exp_dir = root / cfg.name
    exp_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    print("\n" + "=" * 88)
    print(f"开始实验: {cfg.label} ({cfg.name})")
    print(
        f"weights={cfg.var_weights}, humidity_grad_weight={cfg.humidity_grad_weight}, "
        f"monitor_target={args.monitor_target}"
    )
    print("=" * 88)

    trainer = Trainer(
        data_dir=args.data_dir,
        model_type="enhanced",
        mode="multi",
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_dir=str(exp_dir),
        patience=args.patience,
        var_weights=list(cfg.var_weights),
        monitor_target=args.monitor_target,
        humidity_grad_weight=cfg.humidity_grad_weight,
        humidity_cc_weight=0.0,
    )
    trainer.train()

    model_path = exp_dir / "enhanced_ro_diffusion_best.pth"
    eval_dir = exp_dir / "eval_ddpm_fulltest"
    evaluate_main(
        SimpleNamespace(
            model_path=str(model_path),
            model_type="enhanced",
            sampler="ddpm",
            ddim_steps=50,
            n_samples=0,
            batch_size=args.batch_size,
            out_channels=3,
            data_dir=args.data_dir,
            save_dir=str(eval_dir),
            seed=args.seed,
            metric_space="standardized",
            smooth=True,
            no_smooth=False,
        )
    )

    result = {
        "name": cfg.name,
        "label": cfg.label,
        "var_weights": list(cfg.var_weights),
        "humidity_grad_weight": cfg.humidity_grad_weight,
        "seed": args.seed,
        "train_dir": str(exp_dir),
        "model_path": str(model_path),
    }
    result.update(_load_summary(eval_dir / "evaluation_report.json"))

    train_log_path = exp_dir / "enhanced_ro_diffusion_training_log.json"
    if train_log_path.exists():
        train_log = json.loads(train_log_path.read_text(encoding="utf-8"))
        result["best_val_loss"] = float(train_log["best_val_loss"])
        result["best_monitor_value"] = float(train_log["best_monitor_value"])
        result["epochs_trained"] = int(train_log["epochs_trained"])

    target = TABLE_54_TARGETS[cfg.name]
    result["paper_delta"] = {
        key: float(result[key] - target[key])
        for key in ["temperature_cc", "pressure_cc", "humidity_cc", "humidity_rmse"]
    }

    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def main():
    args = parse_args()
    root = _make_output_root(args.output_root)

    manifest = {
        "task": "paper_table_5_4_ablation_rerun",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": args.data_dir,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "patience": args.patience,
        "monitor_target": args.monitor_target,
        "seed": args.seed,
        "experiments": [asdict(item) for item in EXPERIMENTS],
        "paper_targets": TABLE_54_TARGETS,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    rows = []
    for cfg in EXPERIMENTS:
        rows.append(run_single_experiment(cfg, args, root))
        (root / "summary.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    markdown = _build_markdown_table(rows)
    (root / "summary.md").write_text(markdown, encoding="utf-8")

    print("\n" + "#" * 88)
    print(markdown)
    print("#" * 88)
    print(f"\n全部结果已写入: {root}")


if __name__ == "__main__":
    main()
