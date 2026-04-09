"""
FY-3D GNOS ATP + WAP 配对预处理模块
将 ATP 弯曲角与 WAP 温度/气压/湿度剖面配对后转换为训练格式。
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import netCDF4 as nc
import numpy as np
from tqdm import tqdm


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ATPWAPProcessor:
    """ATP + WAP 配对数据处理器"""

    def __init__(
        self,
        target_heights: np.ndarray = np.linspace(0, 60, 301),
        qc_threshold: int = 100,
        random_seed: int = 42,
    ):
        self.target_heights = target_heights
        self.qc_threshold = qc_threshold
        self.random_seed = random_seed

    @staticmethod
    def _to_numpy(array) -> np.ndarray:
        arr = np.asarray(array)
        if np.ma.isMaskedArray(arr):
            arr = arr.filled(np.nan)
        return np.asarray(arr, dtype=np.float64)

    @staticmethod
    def wap_to_atp_name(wap_name: str) -> str:
        return wap_name.replace("_L2_WAP_", "_L2_ATP_")

    def read_pair(self, atp_path: Path, wap_path: Path) -> Optional[Dict]:
        try:
            with nc.Dataset(atp_path) as atp_ds:
                qc = int(atp_ds.getncattr("qc"))
                if qc < self.qc_threshold:
                    return None

                curv = float(atp_ds.getncattr("curv"))
                impact_parm = self._to_numpy(atp_ds.variables["Opt_Impact_parm"][:])
                bend_ang = self._to_numpy(atp_ds.variables["Opt_Bend_ang"][:])
                msl_alt_bend = impact_parm - curv

                atp_lat = float(atp_ds.getncattr("lat"))
                atp_lon = float(atp_ds.getncattr("lon"))
                setting = int(atp_ds.getncattr("setting"))

            with nc.Dataset(wap_path) as wap_ds:
                data = {
                    "bend_ang": bend_ang,
                    "msl_alt_bend": msl_alt_bend,
                    "msl_alt": self._to_numpy(wap_ds.variables["MSL_alt"][:]),
                    "temp": self._to_numpy(wap_ds.variables["Temp"][:]),
                    "pres": self._to_numpy(wap_ds.variables["Pres"][:]),
                    "shum": self._to_numpy(wap_ds.variables["Shum"][:]),
                    "lat": atp_lat,
                    "lon": atp_lon,
                    "setting": setting,
                }

            return data
        except Exception as exc:
            logger.warning("读取配对文件失败 %s / %s: %s", atp_path.name, wap_path.name, exc)
            return None

    def interpolate_to_grid(
        self,
        heights: np.ndarray,
        values: np.ndarray,
    ) -> Optional[np.ndarray]:
        try:
            valid_mask = np.isfinite(heights) & np.isfinite(values)
            if valid_mask.sum() < 10:
                return None

            heights = heights[valid_mask]
            values = values[valid_mask]

            sort_idx = np.argsort(heights)
            heights = heights[sort_idx]
            values = values[sort_idx]

            unique_heights, unique_indices = np.unique(heights, return_index=True)
            values = values[unique_indices]
            if unique_heights.size < 10:
                return None

            interp_values = np.interp(
                self.target_heights,
                unique_heights,
                values,
                left=values[0],
                right=values[-1],
            )
            if np.isnan(interp_values).any():
                return None
            return interp_values
        except Exception as exc:
            logger.warning("插值失败: %s", exc)
            return None

    def process_single_pair(self, data: Dict) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        bend_ang_interp = self.interpolate_to_grid(data["msl_alt_bend"], data["bend_ang"])
        temp_interp = self.interpolate_to_grid(data["msl_alt"], data["temp"])
        pres_interp = self.interpolate_to_grid(data["msl_alt"], data["pres"])
        shum_interp = self.interpolate_to_grid(data["msl_alt"], data["shum"])

        if any(item is None for item in [bend_ang_interp, temp_interp, pres_interp, shum_interp]):
            return None

        if not self._check_physical_validity(temp_interp, pres_interp, shum_interp, bend_ang_interp):
            return None

        bend_ang_log = np.log10(np.abs(bend_ang_interp) + 1e-6)
        pres_log = np.log10(np.clip(pres_interp, 1e-4, None))
        shum_clipped = np.clip(shum_interp, 0.0, None)

        x_vec = bend_ang_log.astype(np.float32)
        y_vec = np.stack([temp_interp, pres_log, shum_clipped], axis=0).astype(np.float32)
        return x_vec, y_vec

    def _check_physical_validity(
        self,
        temp: np.ndarray,
        pres: np.ndarray,
        shum: np.ndarray,
        bend_ang: np.ndarray,
    ) -> bool:
        if np.nanmin(temp) < 150 or np.nanmax(temp) > 350:
            return False

        if np.nanmin(pres) < 0.01 or np.nanmax(pres) > 1100:
            return False

        if np.nanmin(np.abs(bend_ang)) < 1e-6 or np.nanmax(np.abs(bend_ang)) > 0.1:
            return False

        if np.nanmin(shum) < -1e-6 or np.nanmax(shum) > 50:
            return False

        pres_valid = pres[~np.isnan(pres)]
        if len(pres_valid) > 10:
            pres_diff = np.diff(pres_valid)
            increasing_points = pres_diff > 0
            if increasing_points.sum() > len(pres_diff) * 0.2:
                return False
            if np.any(pres_diff > pres_valid[:-1] * 0.1):
                return False

        return True

    def _compute_statistics(self, x: np.ndarray, y: np.ndarray) -> Dict:
        stats = {
            "x_mean": np.nanmean(x),
            "x_std": np.nanstd(x),
            "y_mean": np.nanmean(y, axis=(0, 2)),
            "y_std": np.nanstd(y, axis=(0, 2)),
            "target_heights": self.target_heights,
        }

        logger.info("=== 数据统计 (标准化前) ===")
        logger.info("弯曲角 (log10): mean=%.3f, std=%.3f", stats["x_mean"], stats["x_std"])
        logger.info("温度: mean=%.1f K, std=%.1f K", stats["y_mean"][0], stats["y_std"][0])
        logger.info("气压(log10): mean=%.3f, std=%.3f", stats["y_mean"][1], stats["y_std"][1])
        logger.info("湿度: mean=%.3f g/kg, std=%.3f g/kg", stats["y_mean"][2], stats["y_std"][2])
        return stats

    @staticmethod
    def _normalize_data(x: np.ndarray, y: np.ndarray, stats: Dict) -> Tuple[np.ndarray, np.ndarray]:
        x_norm = (x - stats["x_mean"]) / (stats["x_std"] + 1e-8)
        y_norm = np.zeros_like(y)
        for i in range(y.shape[1]):
            y_norm[:, i, :] = (y[:, i, :] - stats["y_mean"][i]) / (stats["y_std"][i] + 1e-8)
        return x_norm, y_norm

    def _split_dataset(
        self,
        x: np.ndarray,
        y: np.ndarray,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        train_idx, val_idx, test_idx = self._split_indices(len(x), train_ratio, val_ratio)
        return (
            x[train_idx], y[train_idx],
            x[val_idx], y[val_idx],
            x[test_idx], y[test_idx],
        )

    def _split_indices(
        self,
        n: int,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.random_seed)
        indices = rng.permutation(n)

        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]
        return train_idx, val_idx, test_idx

    @staticmethod
    def list_wap_files(wap_dir: Path) -> List[Path]:
        return sorted(list(Path(wap_dir).glob("*.NC")) + list(Path(wap_dir).glob("*.nc")))

    def process_files(
        self,
        atp_dir: Path,
        wap_files: List[Path],
    ) -> Dict:
        x_list, y_list = [], []
        failed_count = 0
        missing_pairs = 0
        paired_count = 0

        for wap_path in tqdm(wap_files, desc="处理 ATP+WAP 数据"):
            atp_path = Path(atp_dir) / self.wap_to_atp_name(wap_path.name)
            if not atp_path.exists():
                missing_pairs += 1
                continue

            paired_count += 1
            data = self.read_pair(atp_path, wap_path)
            if data is None:
                failed_count += 1
                continue

            result = self.process_single_pair(data)
            if result is None:
                failed_count += 1
                continue

            x_vec, y_vec = result
            x_list.append(x_vec)
            y_list.append(y_vec)

        return {
            "x": np.array(x_list, dtype=np.float32) if x_list else np.empty((0, len(self.target_heights)), dtype=np.float32),
            "y": np.array(y_list, dtype=np.float32) if y_list else np.empty((0, 3, len(self.target_heights)), dtype=np.float32),
            "total_wap": len(wap_files),
            "paired": paired_count,
            "missing_pairs": missing_pairs,
            "failed": failed_count,
            "processed": len(x_list),
        }

    def finalize_processed_arrays(
        self,
        x_all: np.ndarray,
        y_all: np.ndarray,
        output_dir: Path,
        counts: Optional[Dict] = None,
    ) -> Dict:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if len(x_all) == 0:
            raise ValueError("没有成功处理任何 ATP+WAP 配对文件")

        logger.info("成功处理: %d 个廓线", len(x_all))
        if counts:
            logger.info("已配对: %d 个文件", counts.get("paired", 0))
            logger.info("缺失 ATP 配对: %d 个", counts.get("missing_pairs", 0))
            logger.info("处理失败: %d 个", counts.get("failed", 0))

        stats = self._compute_statistics(x_all, y_all)
        train_idx, val_idx, test_idx = self._split_indices(len(x_all))

        logger.info("开始按数据划分逐批标准化并保存...")

        def normalize_split(split_x: np.ndarray, split_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            split_x_norm = (split_x - stats["x_mean"]) / (stats["x_std"] + 1e-8)
            split_y_norm = np.empty_like(split_y, dtype=np.float32)
            for i in range(split_y.shape[1]):
                split_y_norm[:, i, :] = (
                    split_y[:, i, :] - stats["y_mean"][i]
                ) / (stats["y_std"][i] + 1e-8)
            return split_x_norm.astype(np.float32), split_y_norm.astype(np.float32)

        train_x, train_y = normalize_split(x_all[train_idx], y_all[train_idx])
        logger.info("保存训练集...")
        np.save(output_dir / "train_x.npy", train_x)
        np.save(output_dir / "train_y.npy", train_y)

        val_x, val_y = normalize_split(x_all[val_idx], y_all[val_idx])
        logger.info("保存验证集...")
        np.save(output_dir / "val_x.npy", val_x)
        np.save(output_dir / "val_y.npy", val_y)

        test_x, test_y = normalize_split(x_all[test_idx], y_all[test_idx])
        logger.info("保存测试集...")
        np.save(output_dir / "test_x.npy", test_x)
        np.save(output_dir / "test_y.npy", test_y)

        logger.info("保存统计量...")
        np.save(output_dir / "stats.npy", stats, allow_pickle=True)

        logger.info("数据已保存到 %s", output_dir)
        logger.info("训练集: %d, 验证集: %d, 测试集: %d", len(train_x), len(val_x), len(test_x))

        result = {
            "total_wap": counts["total_wap"] if counts else len(x_all),
            "paired": counts["paired"] if counts else len(x_all),
            "missing_pairs": counts["missing_pairs"] if counts else 0,
            "failed": counts["failed"] if counts else 0,
            "processed": len(x_all),
            "train": len(train_x),
            "val": len(val_x),
            "test": len(test_x),
            "stats": stats,
        }

        summary_path = output_dir / "summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "total_wap": result["total_wap"],
                    "paired": result["paired"],
                    "missing_pairs": result["missing_pairs"],
                    "failed": result["failed"],
                    "processed": result["processed"],
                    "train": result["train"],
                    "val": result["val"],
                    "test": result["test"],
                    "x_mean": float(stats["x_mean"]),
                    "x_std": float(stats["x_std"]),
                    "y_mean": [float(v) for v in stats["y_mean"]],
                    "y_std": [float(v) for v in stats["y_std"]],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return result

    def process_directory(
        self,
        atp_dir: Path,
        wap_dir: Path,
        output_dir: Path,
        max_files: Optional[int] = None,
    ) -> Dict:
        atp_dir = Path(atp_dir)
        wap_dir = Path(wap_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        wap_files = self.list_wap_files(wap_dir)
        if max_files:
            wap_files = wap_files[:max_files]

        logger.info("找到 %d 个 WAP 文件", len(wap_files))
        chunk_result = self.process_files(atp_dir=atp_dir, wap_files=wap_files)
        return self.finalize_processed_arrays(
            x_all=chunk_result["x"],
            y_all=chunk_result["y"],
            output_dir=output_dir,
            counts=chunk_result,
        )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="处理 FY-3D ATP + WAP 配对数据")
    parser.add_argument("--atp-dir", type=str, default="utils/down", help="ATP 数据目录")
    parser.add_argument("--wap-dir", type=str, default="utils/WAP", help="WAP 数据目录")
    parser.add_argument("--output-dir", type=str, default="Data/Processed_ATP_WAP", help="输出目录")
    parser.add_argument("--max-files", type=int, default=None, help="最大处理文件数 (调试用)")
    parser.add_argument("--qc-threshold", type=int, default=100, help="ATP 质量控制阈值")
    args = parser.parse_args()

    processor = ATPWAPProcessor(qc_threshold=args.qc_threshold)
    result = processor.process_directory(
        Path(args.atp_dir),
        Path(args.wap_dir),
        Path(args.output_dir),
        max_files=args.max_files,
    )

    print("\n=== 处理完成 ===")
    print(f"WAP 总数: {result['total_wap']}")
    print(f"成功配对: {result['paired']}")
    print(f"缺失 ATP: {result['missing_pairs']}")
    print(f"成功处理: {result['processed']}")
    print(f"训练集: {result['train']}")
    print(f"验证集: {result['val']}")
    print(f"测试集: {result['test']}")


if __name__ == "__main__":
    main()
