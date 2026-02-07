"""
数据预处理入口脚本
==================
调用 ro_retrieval.data 模块完成:
  1. 扫描 COSMIC atmPrf + wetPf2 文件
  2. ERA5 时空匹配 (如有)
  3. 多变量提取 (温度 + 压力 + 湿度)
  4. 多级质量控制
  5. train / val / test 划分 (70 / 15 / 15)

用法:
  # 仅使用 wetPf2 (默认)
  python src/process_data.py

  # 使用 ERA5 作为参考真值
  python src/process_data.py --mode era5

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
    parser.add_argument("--atm_dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "Data", "Sample",
                                             "atmPrf_nrt_2026_001"),
                        help="ATM 弯曲角数据目录")
    parser.add_argument("--wet_dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "Data", "Sample",
                                             "wetPf2_nrt_2026_001"),
                        help="WET 温度 / 气压数据目录")
    parser.add_argument("--era5_dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "Data", "Sample"),
                        help="ERA5 再分析数据目录 (包含 ERA5-*.nc 或子文件夹)")
    parser.add_argument("--output_dir", type=str, default=PROCESSED_DIR)
    parser.add_argument("--mode", choices=["wet", "era5"], default="wet",
                        help="wet=使用 wetPf2 作为真值, era5=使用 ERA5 作为真值")
    parser.add_argument("--no-strict-qc", action="store_true",
                        help="关闭严格物理范围QC (放宽筛选)")
    parser.add_argument("--no-split", action="store_true",
                        help="跳过 train/val/test 划分")
    return parser.parse_args()


def main():
    args = parse_args()

    from ro_retrieval.data.process_enhanced import run_enhanced_pipeline

    print("=" * 60)
    print("  掩星数据增强预处理流水线")
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


if __name__ == "__main__":
    exit(main())
