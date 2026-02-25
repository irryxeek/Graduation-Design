"""
CDAAC 数据批量下载脚本
======================
下载 COSMIC-2 掩星数据 (atmPrf + wetPf2)

用法:
    python utils/download_cdaac.py --year 2026 --start_day 1 --end_day 31
"""

import os
import sys
import argparse
import tarfile
import urllib.request
from pathlib import Path

# CDAAC 数据基础 URL
BASE_URL = "https://data.cosmic.ucar.edu/gnss-ro/cosmic2/nrt/level2"

# 数据类型
DATA_TYPES = ["atmPrf", "wetPf2"]


def download_file(url: str, output_path: str) -> bool:
    """使用 urllib 下载文件"""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        print(f"    正在下载...")
        urllib.request.urlretrieve(url, output_path)
        return os.path.exists(output_path) and os.path.getsize(output_path) > 0
    except Exception as e:
        print(f"    下载失败: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return False


def download_and_extract(url: str, output_dir: str, filename: str) -> bool:
    """下载并解压 tar 文件"""
    tar_path = os.path.join(output_dir, filename)
    extract_dir = os.path.join(output_dir, filename.replace(".tar.gz", ""))

    # 检查是否已存在
    if os.path.exists(extract_dir) and len(os.listdir(extract_dir)) > 0:
        print(f"  [跳过] {filename} (已存在)")
        return True

    # 下载
    print(f"  下载: {filename}")
    if not download_file(url, tar_path):
        print(f"  [失败] {filename}")
        return False

    # 解压 (使用Python tarfile)
    try:
        print(f"    正在解压...")
        os.makedirs(extract_dir, exist_ok=True)

        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=extract_dir)

        # 删除 tar 文件节省空间
        os.remove(tar_path)

        # 统计解压的文件数
        file_count = sum(1 for _ in Path(extract_dir).rglob("*") if _.is_file())
        print(f"  [完成] {filename} ({file_count} 文件)")
        return True

    except Exception as e:
        print(f"  [解压失败] {filename}: {e}")
        if os.path.exists(tar_path):
            os.remove(tar_path)
        return False


def download_day(year: int, day: int, output_dir: str) -> dict:
    """下载单日数据"""
    results = {"day": day, "atmPrf": False, "wetPf2": False}

    for data_type in DATA_TYPES:
        # 构建 URL 和文件名
        filename = f"{data_type}_nrt_{year}_{day:03d}.tar.gz"
        url = f"{BASE_URL}/{year}/{day:03d}/{filename}"

        success = download_and_extract(url, output_dir, filename)
        results[data_type] = success

    return results


def main():
    parser = argparse.ArgumentParser(description="CDAAC COSMIC-2 数据批量下载")
    parser.add_argument("--year", type=int, default=2026, help="年份")
    parser.add_argument("--start_day", type=int, default=1, help="起始日 (DOY)")
    parser.add_argument("--end_day", type=int, default=31, help="结束日 (DOY)")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录")

    args = parser.parse_args()

    if args.output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        args.output_dir = os.path.join(project_root, "Data", "Sample")

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"=" * 60)
    print(f"CDAAC COSMIC-2 数据下载")
    print(f"=" * 60)
    print(f"年份: {args.year}")
    print(f"日期范围: Day {args.start_day:03d} - Day {args.end_day:03d}")
    print(f"输出目录: {args.output_dir}")
    print(f"=" * 60)

    days = list(range(args.start_day, args.end_day + 1))
    success_count = 0
    fail_count = 0

    print(f"\n开始下载 {len(days)} 天的数据...\n")

    for day in days:
        print(f"\n[Day {day:03d}/{args.end_day:03d}]")
        result = download_day(args.year, day, args.output_dir)

        if result["atmPrf"] and result["wetPf2"]:
            success_count += 1
        else:
            fail_count += 1

    # 统计
    print(f"\n{'=' * 60}")
    print(f"下载完成!")
    print(f"  成功: {success_count} 天")
    print(f"  失败: {fail_count} 天")
    print(f"{'=' * 60}")

    # 统计文件数量
    total_atm = 0
    total_wet = 0
    for item in os.listdir(args.output_dir):
        item_path = os.path.join(args.output_dir, item)
        if os.path.isdir(item_path):
            if "atmPrf" in item:
                total_atm += len(list(Path(item_path).rglob("*")))
            elif "wetPf2" in item:
                total_wet += len(list(Path(item_path).rglob("*")))

    print(f"\n数据统计:")
    print(f"  atmPrf 文件: {total_atm}")
    print(f"  wetPf2 文件: {total_wet}")


if __name__ == "__main__":
    main()
