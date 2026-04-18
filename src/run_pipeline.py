"""
端到端流水线
============
默认按论文主线执行:
  FY-3D ATP+WAP 预处理 -> 多变量增强版训练 -> 全测试集评估
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ro_retrieval.config import BATCH_SIZE, DEVICE, EPOCHS, LEARNING_RATE, PROJECT_ROOT


DEFAULT_ATP_DIR = os.path.join(PROJECT_ROOT, "Data", "FY-3", "ATP_WAP_2025_RAW", "ATP")
DEFAULT_WAP_DIR = os.path.join(PROJECT_ROOT, "Data", "FY-3", "ATP_WAP_2025_RAW", "WAP")
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "Data", "Processed_ATP_WAP_2025")


def parse_var_weights(value):
    if value is None or value.strip() == "":
        return None
    weights = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not weights:
        raise argparse.ArgumentTypeError("--var_weights cannot be empty")
    if any(weight <= 0 for weight in weights):
        raise argparse.ArgumentTypeError("--var_weights values must be positive")
    return weights


def parse_args():
    parser = argparse.ArgumentParser(
        description="RO-Retrieval 论文主线流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    stage = parser.add_argument_group("流程阶段")
    stage.add_argument("--all", action="store_true", help="执行全部阶段")
    stage.add_argument("--process", action="store_true", help="执行数据处理")
    stage.add_argument("--train", action="store_true", help="执行训练")
    stage.add_argument("--evaluate", action="store_true", help="执行评估")

    data = parser.add_argument_group("数据处理")
    data.add_argument(
        "--source",
        choices=["fy3d_atp_wap", "fy3d", "cosmic"],
        default="fy3d_atp_wap",
        help="默认按论文主线使用 fy3d_atp_wap",
    )
    data.add_argument("--atp_dir", type=str, default=DEFAULT_ATP_DIR)
    data.add_argument("--wap_dir", type=str, default=DEFAULT_WAP_DIR)
    data.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    data.add_argument("--qc_threshold", type=int, default=100)
    data.add_argument("--max_files", type=int, default=None)
    data.add_argument("--atm_dir", type=str, default=os.path.join(PROJECT_ROOT, "Data", "Sample", "atmPrf_nrt_2026_001"))
    data.add_argument("--wet_dir", type=str, default=os.path.join(PROJECT_ROOT, "Data", "Sample", "wetPf2_nrt_2026_001"))
    data.add_argument("--era5_dir", type=str, default=os.path.join(PROJECT_ROOT, "Data", "Sample"))
    data.add_argument("--mode", choices=["wet", "era5"], default="wet")
    data.add_argument("--no-strict-qc", action="store_true")

    train = parser.add_argument_group("训练")
    train.add_argument("--model_type", choices=["legacy", "enhanced"], default="enhanced")
    train.add_argument("--var_mode", choices=["single", "multi"], default="multi")
    train.add_argument("--epochs", type=int, default=50)
    train.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    train.add_argument("--lr", type=float, default=LEARNING_RATE)
    train.add_argument("--patience", type=int, default=15)
    train.add_argument("--save_dir", type=str, default=os.path.join(PROJECT_ROOT, "experiments", "pipeline_run"))
    train.add_argument("--var_weights", type=parse_var_weights, default=[1.0, 1.0, 4.0])
    train.add_argument(
        "--monitor_target",
        choices=["loss", "temperature", "pressure", "humidity", "humidity_cc"],
        default="humidity",
    )
    train.add_argument("--humidity_grad_weight", type=float, default=0.05)
    train.add_argument("--humidity_cc_weight", type=float, default=0.0)

    evaluate = parser.add_argument_group("评估")
    evaluate.add_argument("--model_path", type=str, default=None)
    evaluate.add_argument("--sampler", choices=["ddpm", "ddim"], default="ddim")
    evaluate.add_argument("--ddim_steps", type=int, default=50)
    evaluate.add_argument("--n_eval_samples", type=int, default=0)
    evaluate.add_argument("--eval_batch_size", type=int, default=64)
    evaluate.add_argument("--eval_save_dir", type=str, default=None)
    evaluate.add_argument("--metric_space", choices=["standardized", "physical"], default="standardized")
    evaluate.add_argument("--no_smooth", action="store_true")

    return parser.parse_args()


def stage_process(args):
    print("\n" + "=" * 60)
    print("  Stage 1: 数据处理")
    print("=" * 60)

    if args.source == "fy3d_atp_wap":
        from pathlib import Path
        from ro_retrieval.data.atp_wap_process import ATPWAPProcessor

        processor = ATPWAPProcessor(qc_threshold=args.qc_threshold)
        result = processor.process_directory(
            atp_dir=Path(args.atp_dir),
            wap_dir=Path(args.wap_dir),
            output_dir=Path(args.data_dir),
            max_files=args.max_files,
        )
        print(
            "数据处理完成: "
            f"processed={result['processed']}, paired={result['paired']}, "
            f"train/val/test={result['train']}/{result['val']}/{result['test']}"
        )
        return True

    if args.source == "fy3d":
        from ro_retrieval.data.fy3d_process import run_fy3d_pipeline

        result = run_fy3d_pipeline(
            data_root=args.atp_dir,
            output_dir=args.data_dir,
            strict_qc=not args.no_strict_qc,
            do_split=True,
            use_refractivity=False,
            max_files=args.max_files,
        )
        if result is None:
            print("数据处理失败!")
            return False
        x_data, y_data, _report = result
        print(f"数据处理完成: X={x_data.shape}, Y={y_data.shape}")
        return True

    from ro_retrieval.data.process_enhanced import run_enhanced_pipeline

    result = run_enhanced_pipeline(
        atm_root=args.atm_dir,
        wet_root=args.wet_dir,
        output_dir=args.data_dir,
        era5_root=args.era5_dir if args.mode == "era5" else None,
        strict_qc=not args.no_strict_qc,
        do_split=True,
    )
    if result is None:
        print("数据处理失败!")
        return False
    x_data, y_data, _report = result
    print(f"数据处理完成: X={x_data.shape}, Y={y_data.shape}")
    return True


def stage_train(args):
    print("\n" + "=" * 60)
    print("  Stage 2: 模型训练")
    print("=" * 60)

    from ro_retrieval.training.trainer import Trainer

    os.makedirs(args.save_dir, exist_ok=True)
    trainer = Trainer(
        data_dir=args.data_dir,
        model_type=args.model_type,
        mode=args.var_mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_dir=args.save_dir,
        patience=args.patience,
        var_weights=args.var_weights,
        monitor_target=args.monitor_target,
        humidity_grad_weight=args.humidity_grad_weight,
        humidity_cc_weight=args.humidity_cc_weight,
    )
    trainer.train()

    prefix = "enhanced_ro_diffusion" if args.model_type == "enhanced" else "ro_diffusion"
    args._best_model_path = os.path.join(args.save_dir, f"{prefix}_best.pth")
    return True


def stage_evaluate(args):
    print("\n" + "=" * 60)
    print("  Stage 3: 模型评估")
    print("=" * 60)

    from types import SimpleNamespace
    from src.evaluate import main as evaluate_main

    model_path = args.model_path or getattr(args, "_best_model_path", None)
    if model_path is None:
        prefix = "enhanced_ro_diffusion" if args.model_type == "enhanced" else "ro_diffusion"
        candidate = os.path.join(args.save_dir, f"{prefix}_best.pth")
        if os.path.exists(candidate):
            model_path = candidate
    if model_path is None:
        print("未找到可评估的模型权重")
        return False

    save_dir = args.eval_save_dir
    if save_dir is None:
        suffix = "ddim" if args.sampler == "ddim" else "ddpm"
        save_dir = os.path.join(args.save_dir, f"eval_{suffix}")

    eval_args = SimpleNamespace(
        model_path=model_path,
        model_type=args.model_type,
        sampler=args.sampler,
        ddim_steps=args.ddim_steps,
        n_samples=args.n_eval_samples,
        batch_size=args.eval_batch_size,
        out_channels=3 if args.var_mode == "multi" else 1,
        data_dir=args.data_dir,
        save_dir=save_dir,
        seed=42,
        metric_space=args.metric_space,
        data_space="auto",
        smooth=not args.no_smooth,
        no_smooth=args.no_smooth,
    )
    evaluate_main(eval_args)
    return True


def main():
    args = parse_args()

    if not (args.all or args.process or args.train or args.evaluate):
        args.all = True

    do_process = args.all or args.process
    do_train = args.all or args.train
    do_evaluate = args.all or args.evaluate

    print("=" * 60)
    print("  RO-Retrieval 论文主线流水线")
    print("=" * 60)
    print(f"  source: {args.source}")
    print(f"  阶段: {'处理' if do_process else '-'} → {'训练' if do_train else '-'} → {'评估' if do_evaluate else '-'}")
    print(f"  设备: {DEVICE}")

    t0 = time.time()

    if do_process and not stage_process(args):
        print("流水线在数据处理阶段终止")
        return

    if do_train and not stage_train(args):
        print("流水线在训练阶段终止")
        return

    if do_evaluate and not stage_evaluate(args):
        print("流水线在评估阶段终止")
        return

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  流水线全部完成! 总耗时: {elapsed / 60:.1f} 分钟")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
