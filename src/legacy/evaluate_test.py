#!/usr/bin/env python
"""
测试集评估脚本
==============
对训练好的模型在独立测试集上进行评估
"""

import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ro_retrieval.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="测试集评估")
    parser.add_argument("--model", type=str, default=None,
                        help="模型权重路径 (默认使用 best 模型)")
    parser.add_argument("--model-type", type=str, default="enhanced",
                        choices=["legacy", "enhanced"],
                        help="模型类型")
    parser.add_argument("--mode", type=str, default="multi",
                        choices=["single", "multi"],
                        help="单变量/多变量模式")
    parser.add_argument("--samples", type=int, default=3,
                        help="扩散采样次数 (取平均)")
    args = parser.parse_args()

    print("=" * 60)
    print("测试集评估")
    print("=" * 60)
    print(f"模型类型: {args.model_type}")
    print(f"模式: {args.mode}")
    print(f"采样次数: {args.samples}")
    print("=" * 60)

    trainer = Trainer(
        model_type=args.model_type,
        mode=args.mode,
    )

    results = trainer.evaluate_test(
        model_path=args.model,
        num_samples=args.samples,
    )

    return results


if __name__ == "__main__":
    main()
