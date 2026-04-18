"""
训练入口脚本
============
使用 ro_retrieval.training.Trainer 进行系统化训练。

用法:
  python src/train.py --mode multi --model enhanced --data_dir Data/Processed_ATP_WAP_2025

支持:
  - 单变量 (single) / 多变量 (multi) 训练模式
  - legacy U-Net / enhanced U-Net (交叉注意力)
  - 验证集监控 + Early Stopping
  - 完整训练日志 (JSON + npy)
"""

import os
import sys
import argparse

# 将项目根目录加入 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ro_retrieval.config import (
    BATCH_SIZE, EPOCHS, LEARNING_RATE,
    PAPER_EPOCHS,
    PAPER_HUMIDITY_GRAD_WEIGHT, PAPER_MONITOR_TARGET,
    PAPER_PATIENCE, PAPER_VAR_WEIGHTS,
    PROCESSED_DIR, PROJECT_ROOT,
)


def parse_var_weights(value):
    """Parse comma-separated variable loss weights."""
    if value is None or value.strip() == "":
        return None

    try:
        weights = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--var_weights must be comma-separated numbers, e.g. 1,1,4"
        ) from exc

    if not weights:
        raise argparse.ArgumentTypeError("--var_weights cannot be empty")
    if any(weight <= 0 for weight in weights):
        raise argparse.ArgumentTypeError("--var_weights values must be positive")
    return weights


def parse_args():
    parser = argparse.ArgumentParser(description="训练掩星反演扩散模型")
    parser.add_argument("--mode", choices=["single", "multi"], default="multi",
                        help="single=单变量(温度), multi=多变量(温度+压力+湿度)")
    parser.add_argument("--model", choices=["legacy", "enhanced"], default="enhanced",
                        help="legacy=原始U-Net, enhanced=交叉注意力增强版")
    parser.add_argument("--epochs", type=int, default=PAPER_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "Data", "Processed_ATP_WAP_2025"),
    )
    parser.add_argument("--save_dir", type=str, default=PROJECT_ROOT)
    parser.add_argument("--patience", type=int, default=PAPER_PATIENCE,
                        help="Early Stopping 容忍轮数 (0=关闭)")
    parser.add_argument("--var_weights", type=parse_var_weights, default=list(PAPER_VAR_WEIGHTS),
                        help="多变量损失权重, 逗号分隔; 例如 1,1,4 表示温度/气压/湿度权重")
    parser.add_argument("--monitor_target",
                        choices=["loss", "temperature", "pressure", "humidity", "humidity_cc"],
                        default=PAPER_MONITOR_TARGET,
                        help="Early Stopping 监控目标, 可按湿度通道或湿度CC选最佳模型")
    parser.add_argument("--humidity_grad_weight", type=float, default=PAPER_HUMIDITY_GRAD_WEIGHT,
                        help="湿度廓线梯度约束权重; 例如 0.05")
    parser.add_argument("--humidity_cc_weight", type=float, default=0.0,
                        help="湿度相关性损失权重; 例如 0.1")
    return parser.parse_args()


def main():
    args = parse_args()

    from ro_retrieval.training.trainer import Trainer

    trainer = Trainer(
        data_dir=args.data_dir,
        model_type=args.model,
        mode=args.mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_dir=args.save_dir,
        patience=args.patience if args.patience > 0 else args.epochs,
        var_weights=args.var_weights,
        monitor_target=args.monitor_target,
        humidity_grad_weight=args.humidity_grad_weight,
        humidity_cc_weight=args.humidity_cc_weight,
    )

    trainer.train()


if __name__ == "__main__":
    main()
