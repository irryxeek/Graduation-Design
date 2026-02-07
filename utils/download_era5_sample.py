"""
ERA5 样本数据下载工具
=====================
使用 CDS API 从 Copernicus Climate Data Store 下载 ERA5 再分析数据。

前置条件:
  1. 安装 cdsapi:  pip install cdsapi
  2. 配置 ~/.cdsapirc (API key)
     参考: https://cds.climate.copernicus.eu/api-how-to

用法:
  python utils/download_era5_sample.py --lat -25.26 --lon -107.76 --date 2026-01-01 --time 00:00
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ro_retrieval.config import PROJECT_ROOT

# 37 层标准气压面
PRESSURE_LEVELS = [
    '1', '2', '3', '5', '7', '10',
    '20', '30', '50', '70', '100',
    '125', '150', '175', '200', '225',
    '250', '300', '350', '400', '450',
    '500', '550', '600', '650', '700',
    '750', '775', '800', '825', '850',
    '875', '900', '925', '950', '975',
    '1000',
]


def parse_args():
    parser = argparse.ArgumentParser(description="下载 ERA5 样本数据")
    parser.add_argument("--lat", type=float, default=-25.26, help="中心纬度")
    parser.add_argument("--lon", type=float, default=-107.76, help="中心经度")
    parser.add_argument("--date", type=str, default="2026-01-01",
                        help="日期, 格式: YYYY-MM-DD")
    parser.add_argument("--time", type=str, default="00:00",
                        help="时刻, 格式: HH:MM")
    parser.add_argument("--output", type=str, default=None,
                        help="输出文件路径 (默认: <PROJECT_ROOT>/era5_sample.nc)")
    parser.add_argument("--half_range", type=float, default=0.5,
                        help="经纬度半范围 (度)")
    return parser.parse_args()


def main():
    args = parse_args()

    import cdsapi

    output_path = args.output or os.path.join(PROJECT_ROOT, "era5_sample.nc")
    year, month, day = args.date.split("-")

    area = [
        args.lat + args.half_range,
        args.lon - args.half_range,
        args.lat - args.half_range,
        args.lon + args.half_range,
    ]

    print(f"正在请求 ERA5 数据...")
    print(f"  日期: {args.date} {args.time}")
    print(f"  中心: ({args.lat}, {args.lon})")
    print(f"  区域: {area}")
    print(f"  输出: {output_path}")

    c = cdsapi.Client()
    c.retrieve(
        'reanalysis-era5-pressure-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': ['temperature', 'specific_humidity', 'geopotential'],
            'pressure_level': PRESSURE_LEVELS,
            'year': year,
            'month': month,
            'day': day,
            'time': args.time,
            'area': area,
        },
        output_path,
    )

    print(f"下载完成: {output_path}")


if __name__ == "__main__":
    main()
