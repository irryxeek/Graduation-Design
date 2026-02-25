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
import time
import urllib.request
from pathlib import Path

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print("提示: 安装 requests 可获得更快的下载速度 (pip install requests)")

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# CDAAC 数据基础 URL
BASE_URL = "https://data.cosmic.ucar.edu/gnss-ro/cosmic2/nrt/level2"

# 数据类型
DATA_TYPES = ["atmPrf", "wetPf2"]

# 下载配置
MAX_RETRIES = 3
RETRY_DELAY = 5  # 秒
CHUNK_SIZE = 1024 * 1024  # 1MB 分块


class DownloadProgressBar:
    """下载进度条回调类"""
    def __init__(self, filename: str):
        self.pbar = None
        self.filename = filename
        self.last_percent = -1

    def __call__(self, block_num, block_size, total_size):
        if total_size <= 0:
            return

        downloaded = block_num * block_size
        percent = min(100, downloaded * 100 // total_size)

        if HAS_TQDM:
            if self.pbar is None:
                self.pbar = tqdm(
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=f"    下载",
                    ncols=70,
                    leave=False,
                    position=1,
                    bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{rate_fmt}]'
                )
            if downloaded < total_size:
                self.pbar.update(block_size)
            else:
                self.pbar.close()
        else:
            # 简单的百分比显示 (每10%更新一次避免刷屏)
            if percent != self.last_percent and percent % 10 == 0:
                self.last_percent = percent
                bar_len = 20
                filled = int(bar_len * percent // 100)
                bar = '█' * filled + '░' * (bar_len - filled)
                size_mb = total_size / 1024 / 1024
                sys.stdout.write(f"\r    [{bar}] {percent:3d}% ({size_mb:.1f}MB)")
                sys.stdout.flush()
                if percent >= 100:
                    sys.stdout.write("\n")


def download_file_requests(url: str, output_path: str, filename: str = "") -> bool:
    """使用 requests 分块下载大文件"""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))

        if HAS_TQDM:
            pbar = tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                desc=f"    {filename[:25]}",
                ncols=75,
                leave=False,
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{rate_fmt}, {remaining}]'
            )

        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
                    if HAS_TQDM:
                        pbar.update(len(chunk))

        if HAS_TQDM:
            pbar.close()

        return os.path.exists(output_path) and os.path.getsize(output_path) > 0

    except Exception as e:
        raise e


def download_file_urllib(url: str, output_path: str, filename: str = "") -> bool:
    """使用 urllib 下载文件 (备用)"""
    progress = DownloadProgressBar(filename or os.path.basename(output_path))
    urllib.request.urlretrieve(url, output_path, reporthook=progress)
    return os.path.exists(output_path) and os.path.getsize(output_path) > 0


def download_file(url: str, output_path: str, filename: str = "") -> bool:
    """下载文件，带进度显示和重试机制"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # 清理可能存在的不完整文件
            if os.path.exists(output_path):
                os.remove(output_path)

            # 优先使用 requests (更快更稳定)
            if HAS_REQUESTS:
                success = download_file_requests(url, output_path, filename)
            else:
                success = download_file_urllib(url, output_path, filename)

            if success:
                return True
            else:
                raise Exception("文件为空或不存在")

        except Exception as e:
            if attempt < MAX_RETRIES:
                print(f"\n    下载失败 (尝试 {attempt}/{MAX_RETRIES}): {e}")
                print(f"    {RETRY_DELAY}秒后重试...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"\n    下载失败 (已重试{MAX_RETRIES}次): {e}")
                if os.path.exists(output_path):
                    os.remove(output_path)
                return False

    return False


def download_and_extract(url: str, output_dir: str, filename: str) -> bool:
    """下载并解压 tar 文件"""
    tar_path = os.path.join(output_dir, filename)
    extract_dir = os.path.join(output_dir, filename.replace(".tar.gz", ""))

    # 检查是否已存在
    if os.path.exists(extract_dir) and len(os.listdir(extract_dir)) > 0:
        if not HAS_TQDM:
            print(f"  [跳过] {filename} (已存在)")
        return True

    # 下载
    if not HAS_TQDM:
        print(f"  下载: {filename}")
    if not download_file(url, tar_path, filename):
        print(f"  [失败] {filename}")
        return False

    # 解压 (使用Python tarfile)
    try:
        if not HAS_TQDM:
            print(f"    正在解压...")
        os.makedirs(extract_dir, exist_ok=True)

        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=extract_dir)

        # 删除 tar 文件节省空间
        os.remove(tar_path)

        # 统计解压的文件数
        file_count = sum(1 for _ in Path(extract_dir).rglob("*") if _.is_file())
        if not HAS_TQDM:
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
    parser.add_argument("--retries", type=int, default=3, help="下载失败重试次数")

    args = parser.parse_args()

    # 更新全局重试次数
    global MAX_RETRIES
    MAX_RETRIES = args.retries

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
    total_tasks = len(days) * len(DATA_TYPES)

    print(f"\n开始下载 {len(days)} 天的数据 (共 {total_tasks} 个文件)...\n")

    # 使用 tqdm 显示总体进度
    if HAS_TQDM:
        day_pbar = tqdm(days, desc="总进度", unit="天", ncols=70, position=0)
    else:
        day_pbar = days

    for day in day_pbar:
        if HAS_TQDM:
            day_pbar.set_postfix_str(f"Day {day:03d}")
        else:
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
