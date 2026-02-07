"""
增强版数据预处理入口脚本
========================
调用 ro_retrieval.data.process_enhanced 模块:
  - 支持 ERA5 匹配 / wetPf2 匹配两种模式
  - 质量控制过滤
  - 多变量提取 (温度 + 压力 + 湿度)
  - 训练集 / 验证集 / 测试集划分
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ro_retrieval.config import PROCESSED_DIR, PROJECT_ROOT, STD_HEIGHT
from ro_retrieval.data.process_enhanced import run_enhanced_pipeline, split_dataset


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
                        help="ERA5 再分析数据目录")
    parser.add_argument("--output_dir", type=str, default=PROCESSED_DIR)
    parser.add_argument("--mode", choices=["wet", "era5"], default="wet",
                        help="wet=使用 wetPf2 作为真值, era5=使用 ERA5 作为真值")
    parser.add_argument("--multi_var", action="store_true",
                        help="是否提取多变量 (温度+压力+湿度)")
    parser.add_argument("--skip_split", action="store_true",
                        help="跳过数据集划分")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("掩星数据增强预处理流水线")
    print("=" * 60)
    print(f"  ATM 目录  : {args.atm_dir}")
    print(f"  WET 目录  : {args.wet_dir}")
    print(f"  ERA5 目录 : {args.era5_dir}")
    print(f"  输出目录  : {args.output_dir}")
    print(f"  匹配模式  : {args.mode}")
    print(f"  多变量    : {args.multi_var}")
    print()

    # 1. 运行增强预处理
    result = run_enhanced_pipeline(
        atm_root=args.atm_dir,
        wet_root=args.wet_dir,
        output_dir=args.output_dir,
        era5_root=args.era5_dir if args.mode == "era5" else None,
    )

    if result is None:
        print("预处理失败!")
        return

    train_x, train_y, report = result
    print(f"\n预处理完成:")
    print(f"  train_x: {train_x.shape}")
    print(f"  train_y: {train_y.shape}")
    print(f"  总文件数: {report['total']}")
    print(f"  成功处理: {report['success']}")
    print(f"  质控过滤: {report['qc_filtered']}")

    # 2. 数据集划分
    if not args.skip_split:
        print("\n执行训练/验证/测试集划分 ...")
        splits = split_dataset(train_x, train_y)
        import numpy as _np
        for split_name, split_data in splits.items():
            _np.save(os.path.join(args.output_dir, f"{split_name}_x.npy"), split_data["x"])
            _np.save(os.path.join(args.output_dir, f"{split_name}_y.npy"), split_data["y"])
            print(f"  {split_name}: {len(split_data['x'])} 样本")
        print("划分完成!")

    print("\n全部完成!")


if __name__ == "__main__":
    main()
