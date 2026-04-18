"""
消融实验脚本
============
对比原始 U-Net 与增强版 U-Net 的性能差异

说明:
  该脚本保留用于结构对比实验，不是论文主线默认入口。

实验设置:
  1. 相同数据集 (train/val/test)
  2. 相同超参数 (epochs, batch_size, lr)
  3. 对比模型: legacy vs enhanced
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ro_retrieval.config import PROCESSED_DIR, PROJECT_ROOT, DEVICE
from ro_retrieval.training.trainer import Trainer


def run_ablation_experiment(
    data_dir: str = PROCESSED_DIR,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-4,
    patience: int = 15,
    output_dir: str = None,
):
    """
    运行消融实验: 对比 legacy vs enhanced 模型

    Args:
        data_dir: 数据目录
        epochs: 训练轮数
        batch_size: 批大小
        lr: 学习率
        patience: Early Stopping 容忍轮数
        output_dir: 输出目录
    """
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "outputs", "ablation")
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "timestamp": timestamp,
        "config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "patience": patience,
            "data_dir": data_dir,
        },
        "experiments": {}
    }

    # 实验配置
    experiments = [
        {"name": "legacy_single", "model_type": "legacy", "mode": "single"},
        {"name": "enhanced_single", "model_type": "enhanced", "mode": "single"},
        {"name": "legacy_multi", "model_type": "legacy", "mode": "multi"},
        {"name": "enhanced_multi", "model_type": "enhanced", "mode": "multi"},
    ]

    for exp in experiments:
        print(f"\n{'=' * 70}")
        print(f"实验: {exp['name']}")
        print(f"  模型类型: {exp['model_type']}")
        print(f"  变量模式: {exp['mode']}")
        print(f"{'=' * 70}")

        exp_dir = os.path.join(output_dir, exp["name"])
        os.makedirs(exp_dir, exist_ok=True)

        try:
            # 训练
            trainer = Trainer(
                data_dir=data_dir,
                model_type=exp["model_type"],
                mode=exp["mode"],
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                save_dir=exp_dir,
                patience=patience,
            )
            trainer.train()

            # 测试集评估
            test_results = trainer.evaluate_test(num_samples=3)

            # 记录结果
            results["experiments"][exp["name"]] = {
                "model_type": exp["model_type"],
                "mode": exp["mode"],
                "best_val_loss": float(trainer.best_val_loss),
                "epochs_trained": len(trainer.train_losses),
                "test_metrics": test_results["metrics"],
            }

            print(f"\n[{exp['name']}] 完成")
            print(f"  最佳验证损失: {trainer.best_val_loss:.6f}")
            print(f"  测试 RMSE: {test_results['metrics']['rmse']:.4f}")
            print(f"  测试 R²: {test_results['metrics']['r2_mean']:.4f}")

        except Exception as e:
            print(f"\n[{exp['name']}] 失败: {e}")
            results["experiments"][exp["name"]] = {"error": str(e)}

    # 保存汇总结果
    summary_path = os.path.join(output_dir, f"ablation_summary_{timestamp}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=float)
    print(f"\n消融实验汇总已保存: {summary_path}")

    # 打印对比表格
    print_comparison_table(results)

    return results


def print_comparison_table(results: dict):
    """打印对比表格"""
    print(f"\n{'=' * 80}")
    print("消融实验结果对比")
    print(f"{'=' * 80}")
    print(f"{'实验名称':<20} {'模型':<10} {'模式':<8} {'Val Loss':<12} {'RMSE':<10} {'R²':<10}")
    print("-" * 80)

    for name, exp in results.get("experiments", {}).items():
        if "error" in exp:
            print(f"{name:<20} {'ERROR':<10}")
            continue

        model = exp.get("model_type", "?")
        mode = exp.get("mode", "?")
        val_loss = exp.get("best_val_loss", float("nan"))
        metrics = exp.get("test_metrics", {})
        rmse = metrics.get("rmse", float("nan"))
        r2 = metrics.get("r2_mean", float("nan"))

        print(f"{name:<20} {model:<10} {mode:<8} {val_loss:<12.6f} {rmse:<10.4f} {r2:<10.4f}")

    print(f"{'=' * 80}")

    # 多变量模式下的各变量对比
    multi_exps = {k: v for k, v in results.get("experiments", {}).items()
                  if v.get("mode") == "multi" and "error" not in v}

    if multi_exps:
        print(f"\n多变量模式各变量 RMSE 对比:")
        print(f"{'实验名称':<20} {'温度(K)':<12} {'气压(hPa)':<12} {'比湿(kg/kg)':<12}")
        print("-" * 60)

        for name, exp in multi_exps.items():
            rmse_per_var = exp.get("test_metrics", {}).get("rmse_per_var", [])
            if len(rmse_per_var) >= 3:
                print(f"{name:<20} {rmse_per_var[0]:<12.4f} {rmse_per_var[1]:<12.4f} {rmse_per_var[2]:<12.6f}")

        print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="消融实验: 对比 legacy vs enhanced 模型")
    parser.add_argument("--data_dir", type=str, default=PROCESSED_DIR, help="数据目录")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=64, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--patience", type=int, default=15, help="Early Stopping 容忍轮数")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录")

    args = parser.parse_args()

    run_ablation_experiment(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        output_dir=args.output_dir,
    )
