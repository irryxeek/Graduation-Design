#!/usr/bin/env python3
"""
整理 FY-3D 2025 年 ATP/WAP 原始文件到统一目录。

默认将已有的 Q1 原始目录与新下载的 4-6 月 HTTP 目录合并为:
  Data/FY-3/ATP_WAP_2025_RAW/ATP
  Data/FY-3/ATP_WAP_2025_RAW/WAP

采用符号链接方式，避免重复拷贝大文件。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "Data" / "FY-3" / "ATP_WAP_2025_RAW"

ATP_SOURCES = [
    PROJECT_ROOT / "utils" / "down",
    Path("/root/autodl-tmp/Downloads/ATP_2025_04_06_HTTP"),
]

WAP_SOURCES = [
    PROJECT_ROOT / "utils" / "down",
    PROJECT_ROOT / "utils" / "WAP",
    Path("/root/autodl-tmp/Downloads/WAP_2025_04_06_HTTP"),
]


def iter_nc_files(paths: Iterable[Path], keyword: str):
    for src_dir in paths:
        if not src_dir.exists():
            continue
        for path in sorted(src_dir.glob("*.NC")):
            if keyword in path.name:
                yield path


def build_symlink_pool(paths: Iterable[Path], keyword: str, target_dir: Path):
    target_dir.mkdir(parents=True, exist_ok=True)
    linked = 0
    skipped = 0
    for src in iter_nc_files(paths, keyword):
        dst = target_dir / src.name
        if dst.exists() or dst.is_symlink():
            skipped += 1
            continue
        dst.symlink_to(src.resolve())
        linked += 1
    files = sorted(target_dir.glob("*.NC"))
    return {
        "linked_new": linked,
        "skipped_existing": skipped,
        "total_files": len(files),
        "first_file": files[0].name if files else None,
        "last_file": files[-1].name if files else None,
    }


def count_pairable(atp_dir: Path, wap_dir: Path):
    atp_names = {path.name for path in atp_dir.glob("*.NC")}
    wap_names = [path.name for path in wap_dir.glob("*.NC")]
    pairable = 0
    missing_samples = []

    for wap_name in wap_names:
        atp_name = wap_name.replace("_L2_WAP_", "_L2_ATP_")
        if atp_name in atp_names:
            pairable += 1
        elif len(missing_samples) < 20:
            missing_samples.append(
                {
                    "wap": wap_name,
                    "expected_atp": atp_name,
                }
            )

    return {
        "atp_total": len(atp_names),
        "wap_total": len(wap_names),
        "pairable": pairable,
        "missing_pairs": len(wap_names) - pairable,
        "missing_samples": missing_samples,
    }


def main():
    output_root = DEFAULT_OUTPUT_ROOT
    atp_dir = output_root / "ATP"
    wap_dir = output_root / "WAP"

    atp_summary = build_symlink_pool(ATP_SOURCES, "_L2_ATP_", atp_dir)
    wap_summary = build_symlink_pool(WAP_SOURCES, "_L2_WAP_", wap_dir)
    pair_summary = count_pairable(atp_dir, wap_dir)

    summary = {
        "output_root": str(output_root),
        "atp_dir": str(atp_dir),
        "wap_dir": str(wap_dir),
        "atp": atp_summary,
        "wap": wap_summary,
        "pairing": pair_summary,
        "sources": {
            "atp": [str(path) for path in ATP_SOURCES],
            "wap": [str(path) for path in WAP_SOURCES],
        },
    }

    summary_path = output_root / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
