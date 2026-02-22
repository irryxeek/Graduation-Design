#!/usr/bin/env python
"""
多变量模型训练脚本
==================
训练 Enhanced U-Net 进行多变量联合反演 (温度 + 压力 + 湿度)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ro_retrieval.training.trainer import Trainer


def main():
    print("=" * 60)
    print("多变量联合反演模型训练")
    print("=" * 60)
    print("输入: 弯曲角 (1, 301)")
    print("输出: 温度 + 压力 + 比湿 (3, 301)")
    print("=" * 60)

    trainer = Trainer(
        model_type="enhanced",
        mode="multi",           # 多变量模式
        epochs=100,
        batch_size=16,
        lr=1e-4,
        patience=20,
        save_every=10,
    )

    # 训练
    trainer.train()

    # 测试集评估
    print("\n开始测试集评估...")
    results = trainer.evaluate_test(num_samples=3)

    print("\n训练与评估完成!")
    return results


if __name__ == "__main__":
    main()
