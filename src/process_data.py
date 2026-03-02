"""
数据预处理入口脚本
==================
支持两种数据源:
  1. COSMIC-2 掩星数据 (atmPrf + wetPf2 / ERA5)
  2. FY-3D GNOS 掩星数据 (单文件格式)

用法:
  # COSMIC-2: 使用 wetPf2 (默认)
  python src/process_data.py

  # COSMIC-2: 使用 ERA5 作为参考真值
  python src/process_data.py --mode era5

  # FY-3D: 处理 GNOS 数据
  python src/process_data.py --source fy3d --fy3d_dir Data/FY-3/raw

  # FY-3D: 仅探索数据格式 (不处理)
  python src/process_data.py --source fy3d --fy3d_dir Data/FY-3/raw --explore

  # 关闭严格QC (处理更多数据, 但可能含噪声)
  python src/process_data.py --no-strict-qc
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ro_retrieval.config import PROCESSED_DIR, PROJECT_ROOT


def parse_args():
    parser = argparse.ArgumentParser(description="掩星数据预处理")

    # 数据源选择
    parser.add_argument("--source", choices=["cosmic", "fy3d"], default="cosmic",
                        help="数据源: cosmic=COSMIC-2, fy3d=FY-3D GNOS")

    # COSMIC-2 参数
    parser.add_argument("--atm_dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "Data", "Sample",
                                             "atmPrf_nrt_2026_001"),
                        help="[COSMIC] ATM 弯曲角数据目录")
    parser.add_argument("--wet_dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "Data", "Sample",
                                             "wetPf2_nrt_2026_001"),
                        help="[COSMIC] WET 温度 / 气压数据目录")
    parser.add_argument("--era5_dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "Data", "Sample"),
                        help="[COSMIC] ERA5 再分析数据目录")
    parser.add_argument("--mode", choices=["wet", "era5"], default="wet",
                        help="[COSMIC] wet=使用 wetPf2, era5=使用 ERA5")

    # FY-3D 参数
    parser.add_argument("--fy3d_dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "Data", "FY-3", "raw"),
                        help="[FY-3D] GNOS NC 文件目录")
    parser.add_argument("--use-refractivity", action="store_true",
                        help="[FY-3D] 使用折射率代替弯曲角作为输入")
    parser.add_argument("--explore", action="store_true",
                        help="[FY-3D] 仅探索数据格式, 不执行处理")
    parser.add_argument("--max-files", type=int, default=None,
                        help="[FY-3D] 最大处理文件数 (调试用)")

    # 通用参数
    parser.add_argument("--output_dir", type=str, default=PROCESSED_DIR)
    parser.add_argument("--no-strict-qc", action="store_true",
                        help="关闭严格物理范围QC (放宽筛选)")
    parser.add_argument("--no-split", action="store_true",
                        help="跳过 train/val/test 划分")
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

    X, Y, report = result
    print(f"\n预处理完成!")
    print(f"  数据维度: X={X.shape}, Y={Y.shape}")
    print(f"  成功率:   {report['qc_pass_rate']}")
    return 0


def main_fy3d(args):
    """FY-3D GNOS 数据处理流程"""
    from ro_retrieval.data.fy3d_process import (
        run_fy3d_pipeline, explore_fy3d_directory,
    )

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

    # 仅探索模式
    if args.explore:
        explore_fy3d_directory(args.fy3d_dir, n_samples=3)
        return 0

    result = run_fy3d_pipeline(
        data_root=args.fy3d_dir,
        output_dir=args.output_dir,
        strict_qc=not args.no_strict_qc,
        do_split=not args.no_split,
        use_refractivity=args.use_refractivity,
        max_files=args.max_files,
    )

    if result is None:
        print("预处理失败!")
        return 1

    X, Y, report = result
    print(f"\n预处理完成!")
    print(f"  数据维度: X={X.shape}, Y={Y.shape}")
    print(f"  成功率:   {report['qc_pass_rate']}")
    return 0


def main():
    args = parse_args()

    if args.source == "fy3d":
        return main_fy3d(args)
    else:
        return main_cosmic(args)


if __name__ == "__main__":
    exit(main())
