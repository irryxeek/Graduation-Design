"""
数据预处理入口脚本
==================
支持三种数据源:
  1. FY-3D ATP + WAP 配对数据 (论文主线, 默认)
  2. FY-3D GNOS 掩星数据 (单文件格式)
  3. COSMIC-2 掩星数据 (atmPrf + wetPf2 / ERA5)
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ro_retrieval.config import PROCESSED_DIR, PROJECT_ROOT


def parse_args():
    parser = argparse.ArgumentParser(description="掩星数据预处理")

    parser.add_argument(
        "--source",
        choices=["cosmic", "fy3d", "fy3d_atp_wap"],
        default="fy3d_atp_wap",
        help="数据源: cosmic=COSMIC-2, fy3d=FY-3D GNOS, fy3d_atp_wap=FY-3D ATP+WAP 配对",
    )

    parser.add_argument(
        "--atm_dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "Data", "Sample", "atmPrf_nrt_2026_001"),
        help="[COSMIC] ATM 弯曲角数据目录",
    )
    parser.add_argument(
        "--wet_dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "Data", "Sample", "wetPf2_nrt_2026_001"),
        help="[COSMIC] WET 温度 / 气压数据目录",
    )
    parser.add_argument(
        "--era5_dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "Data", "Sample"),
        help="[COSMIC] ERA5 再分析数据目录",
    )
    parser.add_argument(
        "--mode",
        choices=["wet", "era5"],
        default="wet",
        help="[COSMIC] wet=使用 wetPf2, era5=使用 ERA5",
    )

    parser.add_argument(
        "--fy3d_dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "Data", "FY-3", "raw"),
        help="[FY-3D] GNOS NC 文件目录",
    )
    parser.add_argument(
        "--use-refractivity",
        action="store_true",
        help="[FY-3D] 使用折射率代替弯曲角作为输入",
    )
    parser.add_argument(
        "--explore",
        action="store_true",
        help="[FY-3D] 仅探索数据格式, 不执行处理",
    )
    parser.add_argument("--max-files", type=int, default=None, help="最大处理文件数 (调试用)")

    parser.add_argument(
        "--atp_dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "Data", "FY-3", "ATP_WAP_2025_RAW", "ATP"),
        help="[FY-3D ATP+WAP] ATP 数据目录",
    )
    parser.add_argument(
        "--wap_dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "Data", "FY-3", "ATP_WAP_2025_RAW", "WAP"),
        help="[FY-3D ATP+WAP] WAP 数据目录",
    )
    parser.add_argument(
        "--qc-threshold",
        type=int,
        default=100,
        help="[FY-3D ATP+WAP] ATP 质量控制阈值",
    )

    parser.add_argument("--output_dir", type=str, default=PROCESSED_DIR)
    parser.add_argument("--no-strict-qc", action="store_true", help="关闭严格物理范围QC (放宽筛选)")
    parser.add_argument("--no-split", action="store_true", help="跳过 train/val/test 划分")
    return parser.parse_args()


def main_cosmic(args):
    """COSMIC-2 数据处理流程"""
    from ro_retrieval.data.process_enhanced import run_enhanced_pipeline

    print("=" * 60)
    print("  COSMIC-2 掩星数据预处理流水线")
    print("=" * 60)
    print(f"  ATM 目录   : {args.atm_dir}")
    print(f"  WET 目录   : {args.wet_dir}")
    print(f"  ERA5 目录  : {args.era5_dir}")
    print(f"  输出目录   : {args.output_dir}")
    print(f"  匹配模式   : {args.mode}")
    print(f"  严格QC     : {not args.no_strict_qc}")
    print(f"  数据划分   : {not args.no_split}")
    print()

    result = run_enhanced_pipeline(
        atm_root=args.atm_dir,
        wet_root=args.wet_dir,
        output_dir=args.output_dir,
        era5_root=args.era5_dir if args.mode == "era5" else None,
        strict_qc=not args.no_strict_qc,
        do_split=not args.no_split,
    )

    if result is None:
        print("预处理失败!")
        return 1

    x_data, y_data, report = result
    print("\n预处理完成!")
    print(f"  数据维度: X={x_data.shape}, Y={y_data.shape}")
    print(f"  成功率:   {report['qc_pass_rate']}")
    return 0


def main_fy3d(args):
    """FY-3D GNOS 数据处理流程"""
    from ro_retrieval.data.fy3d_process import run_fy3d_pipeline, explore_fy3d_directory

    print("=" * 60)
    print("  FY-3D GNOS 掩星数据预处理流水线")
    print("=" * 60)
    print(f"  数据目录   : {args.fy3d_dir}")
    print(f"  输出目录   : {args.output_dir}")
    print(f"  输入特征   : {'折射率' if args.use_refractivity else '弯曲角'}")
    print(f"  严格QC     : {not args.no_strict_qc}")
    print(f"  数据划分   : {not args.no_split}")
    if args.max_files:
        print(f"  最大文件数 : {args.max_files}")
    print()

    fy3d_output_dir = os.path.join(args.output_dir, "fy3d")

    if args.explore:
        explore_fy3d_directory(args.fy3d_dir, n_samples=3)
        return 0

    result = run_fy3d_pipeline(
        data_root=args.fy3d_dir,
        output_dir=fy3d_output_dir,
        strict_qc=not args.no_strict_qc,
        do_split=not args.no_split,
        use_refractivity=args.use_refractivity,
        max_files=args.max_files,
    )

    if result is None:
        print("预处理失败!")
        return 1

    x_data, y_data, report = result
    print("\n预处理完成!")
    print(f"  数据维度: X={x_data.shape}, Y={y_data.shape}")
    print(f"  成功率:   {report['qc_pass_rate']}")
    return 0


def main_fy3d_atp_wap(args):
    """FY-3D ATP + WAP 配对数据处理流程"""
    from pathlib import Path
    from ro_retrieval.data.atp_wap_process import ATPWAPProcessor

    print("=" * 60)
    print("  FY-3D ATP + WAP 配对预处理流水线")
    print("=" * 60)
    print(f"  ATP 目录   : {args.atp_dir}")
    print(f"  WAP 目录   : {args.wap_dir}")
    print(f"  输出目录   : {args.output_dir}")
    print(f"  QC 阈值    : {args.qc_threshold}")
    if args.max_files:
        print(f"  最大文件数 : {args.max_files}")
    print()

    processor = ATPWAPProcessor(qc_threshold=args.qc_threshold)
    result = processor.process_directory(
        atp_dir=Path(args.atp_dir),
        wap_dir=Path(args.wap_dir),
        output_dir=Path(args.output_dir),
        max_files=args.max_files,
    )

    print("\n预处理完成!")
    print(f"  成功处理   : {result['processed']}")
    print(f"  成功配对   : {result['paired']}")
    print(f"  缺失 ATP   : {result['missing_pairs']}")
    print(f"  训练/验证/测试: {result['train']} / {result['val']} / {result['test']}")
    return 0


def main():
    args = parse_args()

    if args.source == "fy3d":
        return main_fy3d(args)
    if args.source == "fy3d_atp_wap":
        return main_fy3d_atp_wap(args)
    return main_cosmic(args)


if __name__ == "__main__":
    raise SystemExit(main())
