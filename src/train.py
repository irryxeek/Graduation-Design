"""
训练入口脚本
============
使用 ro_retrieval.training.Trainer 进行系统化训练。

用法:
  python src/train.py --mode single --model legacy --epochs 100
  python src/train.py --mode multi  --model enhanced --epochs 50

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
    PROCESSED_DIR, PROJECT_ROOT,
)


def parse_args():
    parser = argparse.ArgumentParser(description="训练掩星反演扩散模型")
    parser.add_argument("--mode", choices=["single", "multi"], default="single",
                        help="single=单变量(温度), multi=多变量(温度+压力+湿度)")
    parser.add_argument("--model", choices=["legacy", "enhanced"], default="legacy",
                        help="legacy=原始U-Net, enhanced=交叉注意力增强版")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--data_dir", type=str, default=PROCESSED_DIR)
    parser.add_argument("--save_dir", type=str, default=PROJECT_ROOT)
    parser.add_argument("--patience", type=int, default=20,
                        help="Early Stopping 容忍轮数 (0=关闭)")
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
    )

    trainer.train()


if __name__ == "__main__":
    main()
