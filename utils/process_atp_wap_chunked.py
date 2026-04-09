#!/usr/bin/env python3
"""
按块处理 FY-3D ATP+WAP 原始数据，并在全部块完成后合并为最终训练数据。

用途:
1. 避免单次长进程中途退出导致全量重跑
2. 支持已完成块自动跳过，便于断线/重连后继续
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ro_retrieval.data.atp_wap_process import ATPWAPProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="分块处理 FY-3D ATP+WAP 数据")
    parser.add_argument("--atp-dir", required=True, help="ATP 数据目录")
    parser.add_argument("--wap-dir", required=True, help="WAP 数据目录")
    parser.add_argument("--output-dir", required=True, help="最终输出目录")
    parser.add_argument(
        "--work-dir",
        default=None,
        help="中间块目录，默认 output_dir/_chunks",
    )
    parser.add_argument("--chunk-size", type=int, default=5000, help="每块 WAP 文件数")
    parser.add_argument("--max-files", type=int, default=None, help="仅处理前 N 个 WAP 文件，调试用")
    parser.add_argument("--qc-threshold", type=int, default=100, help="ATP 质量控制阈值")
    parser.add_argument("--force-merge", action="store_true", help="即使块目录已存在也重新合并")
    return parser.parse_args()


def save_chunk(chunk_dir: Path, chunk_result: dict):
    chunk_dir.mkdir(parents=True, exist_ok=True)
    np.save(chunk_dir / "x.npy", chunk_result["x"])
    np.save(chunk_dir / "y.npy", chunk_result["y"])
    meta = {
        "total_wap": int(chunk_result["total_wap"]),
        "paired": int(chunk_result["paired"]),
        "missing_pairs": int(chunk_result["missing_pairs"]),
        "failed": int(chunk_result["failed"]),
        "processed": int(chunk_result["processed"]),
    }
    (chunk_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def load_chunk_meta(chunk_dir: Path) -> dict:
    return json.loads((chunk_dir / "meta.json").read_text(encoding="utf-8"))


def build_stats_and_counts(chunk_dirs: list[Path], target_heights: np.ndarray) -> tuple[dict, dict]:
    counts = {
        "total_wap": 0,
        "paired": 0,
        "missing_pairs": 0,
        "failed": 0,
        "processed": 0,
    }

    x_sum = 0.0
    x_sq_sum = 0.0
    x_count = 0
    y_sum = np.zeros(3, dtype=np.float64)
    y_sq_sum = np.zeros(3, dtype=np.float64)
    y_count = 0

    for chunk_dir in chunk_dirs:
        meta = load_chunk_meta(chunk_dir)
        for key in counts:
            counts[key] += int(meta[key])

        x = np.load(chunk_dir / "x.npy", mmap_mode="r")
        y = np.load(chunk_dir / "y.npy", mmap_mode="r")
        if len(x) == 0:
            continue

        x64 = np.asarray(x, dtype=np.float64)
        y64 = np.asarray(y, dtype=np.float64)
        x_sum += x64.sum()
        x_sq_sum += np.square(x64).sum()
        x_count += x64.size

        y_sum += y64.sum(axis=(0, 2))
        y_sq_sum += np.square(y64).sum(axis=(0, 2))
        y_count += y64.shape[0] * y64.shape[2]

    x_mean = x_sum / max(x_count, 1)
    x_var = max(x_sq_sum / max(x_count, 1) - x_mean ** 2, 0.0)
    y_mean = y_sum / max(y_count, 1)
    y_var = np.maximum(y_sq_sum / max(y_count, 1) - y_mean ** 2, 0.0)

    stats = {
        "x_mean": float(x_mean),
        "x_std": float(np.sqrt(x_var)),
        "y_mean": y_mean,
        "y_std": np.sqrt(y_var),
        "target_heights": target_heights,
    }
    return stats, counts


def write_final_splits(
    chunk_dirs: list[Path],
    output_dir: Path,
    processor: ATPWAPProcessor,
    stats: dict,
    counts: dict,
) -> dict:
    total_processed = int(counts["processed"])
    if total_processed == 0:
        raise ValueError("没有可写入的 processed 样本")

    train_idx, val_idx, test_idx = processor._split_indices(total_processed)
    split_ids = np.full(total_processed, 2, dtype=np.int8)
    split_ids[train_idx] = 0
    split_ids[val_idx] = 1

    train_n = len(train_idx)
    val_n = len(val_idx)
    test_n = len(test_idx)
    height_n = len(stats["target_heights"])

    output_dir.mkdir(parents=True, exist_ok=True)
    train_x = np.lib.format.open_memmap(output_dir / "train_x.npy", mode="w+", dtype=np.float32, shape=(train_n, height_n))
    train_y = np.lib.format.open_memmap(output_dir / "train_y.npy", mode="w+", dtype=np.float32, shape=(train_n, 3, height_n))
    val_x = np.lib.format.open_memmap(output_dir / "val_x.npy", mode="w+", dtype=np.float32, shape=(val_n, height_n))
    val_y = np.lib.format.open_memmap(output_dir / "val_y.npy", mode="w+", dtype=np.float32, shape=(val_n, 3, height_n))
    test_x = np.lib.format.open_memmap(output_dir / "test_x.npy", mode="w+", dtype=np.float32, shape=(test_n, height_n))
    test_y = np.lib.format.open_memmap(output_dir / "test_y.npy", mode="w+", dtype=np.float32, shape=(test_n, 3, height_n))

    split_buffers = {
        0: {"x": train_x, "y": train_y, "pos": 0},
        1: {"x": val_x, "y": val_y, "pos": 0},
        2: {"x": test_x, "y": test_y, "pos": 0},
    }

    x_mean = float(stats["x_mean"])
    x_std = float(stats["x_std"]) + 1e-8
    y_mean = np.asarray(stats["y_mean"], dtype=np.float32)
    y_std = np.asarray(stats["y_std"], dtype=np.float32) + 1e-8

    global_offset = 0
    for chunk_dir in chunk_dirs:
        x = np.load(chunk_dir / "x.npy", mmap_mode="r")
        y = np.load(chunk_dir / "y.npy", mmap_mode="r")
        n = len(x)
        if n == 0:
            continue

        chunk_split = split_ids[global_offset:global_offset + n]
        for split_value in (0, 1, 2):
            local_idx = np.flatnonzero(chunk_split == split_value)
            if local_idx.size == 0:
                continue

            x_sel = np.asarray(x[local_idx], dtype=np.float32)
            y_sel = np.asarray(y[local_idx], dtype=np.float32)
            x_norm = ((x_sel - x_mean) / x_std).astype(np.float32)
            y_norm = np.empty_like(y_sel, dtype=np.float32)
            for channel_idx in range(y_sel.shape[1]):
                y_norm[:, channel_idx, :] = (
                    y_sel[:, channel_idx, :] - y_mean[channel_idx]
                ) / y_std[channel_idx]

            buf = split_buffers[split_value]
            pos = buf["pos"]
            end = pos + local_idx.size
            buf["x"][pos:end] = x_norm
            buf["y"][pos:end] = y_norm
            buf["pos"] = end

        global_offset += n

    for split_value in (0, 1, 2):
        split_buffers[split_value]["x"].flush()
        split_buffers[split_value]["y"].flush()

    del train_x, train_y, val_x, val_y, test_x, test_y

    stats_to_save = {
        "x_mean": stats["x_mean"],
        "x_std": stats["x_std"],
        "y_mean": stats["y_mean"],
        "y_std": stats["y_std"],
        "target_heights": stats["target_heights"],
    }
    np.save(output_dir / "stats.npy", stats_to_save, allow_pickle=True)

    summary = {
        "total_wap": int(counts["total_wap"]),
        "paired": int(counts["paired"]),
        "missing_pairs": int(counts["missing_pairs"]),
        "failed": int(counts["failed"]),
        "processed": int(counts["processed"]),
        "train": int(train_n),
        "val": int(val_n),
        "test": int(test_n),
        "x_mean": float(stats["x_mean"]),
        "x_std": float(stats["x_std"]),
        "y_mean": [float(v) for v in stats["y_mean"]],
        "y_std": [float(v) for v in stats["y_std"]],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main():
    args = parse_args()
    atp_dir = Path(args.atp_dir)
    wap_dir = Path(args.wap_dir)
    output_dir = Path(args.output_dir)
    work_dir = Path(args.work_dir) if args.work_dir else output_dir / "_chunks"
    work_dir.mkdir(parents=True, exist_ok=True)

    processor = ATPWAPProcessor(qc_threshold=args.qc_threshold)
    wap_files = processor.list_wap_files(wap_dir)
    if args.max_files:
        wap_files = wap_files[:args.max_files]
    total_files = len(wap_files)
    print(f"总 WAP 文件数: {total_files}")
    print(f"分块大小: {args.chunk_size}")
    print(f"块目录: {work_dir}")

    chunk_dirs = []
    chunk_idx = 0
    for start in range(0, total_files, args.chunk_size):
        end = min(start + args.chunk_size, total_files)
        chunk_idx += 1
        chunk_dir = work_dir / f"chunk_{chunk_idx:03d}_{start:06d}_{end:06d}"
        chunk_dirs.append(chunk_dir)
        meta_path = chunk_dir / "meta.json"
        x_path = chunk_dir / "x.npy"
        y_path = chunk_dir / "y.npy"

        if meta_path.exists() and x_path.exists() and y_path.exists():
            meta = load_chunk_meta(chunk_dir)
            print(
                f"[跳过] chunk {chunk_idx:03d}: {start}-{end} "
                f"processed={meta['processed']} paired={meta['paired']} failed={meta['failed']}"
            )
            continue

        print(f"[开始] chunk {chunk_idx:03d}: {start}-{end}")
        chunk_result = processor.process_files(
            atp_dir=atp_dir,
            wap_files=wap_files[start:end],
        )
        save_chunk(chunk_dir, chunk_result)
        print(
            f"[完成] chunk {chunk_idx:03d}: processed={chunk_result['processed']} "
            f"paired={chunk_result['paired']} missing={chunk_result['missing_pairs']} failed={chunk_result['failed']}"
        )

    merge_marker = output_dir / "summary.json"
    if merge_marker.exists() and not args.force_merge:
        print(f"最终结果已存在，跳过合并: {merge_marker}")
        return

    print("开始统计所有块...")
    stats, counts = build_stats_and_counts(chunk_dirs, processor.target_heights)
    print(
        f"统计完成: processed={counts['processed']}, paired={counts['paired']}, "
        f"x_mean={stats['x_mean']:.3f}, x_std={stats['x_std']:.3f}"
    )

    print("开始按块写入最终 train/val/test ...")
    result = write_final_splits(
        chunk_dirs=chunk_dirs,
        output_dir=output_dir,
        processor=processor,
        stats=stats,
        counts=counts,
    )

    print("\n=== 分块预处理完成 ===")
    print(json.dumps(
        {
            "processed": result["processed"],
            "paired": result["paired"],
            "missing_pairs": result["missing_pairs"],
            "failed": result["failed"],
            "train": result["train"],
            "val": result["val"],
            "test": result["test"],
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
