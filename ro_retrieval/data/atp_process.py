"""
FY-3D GNOS L2 ATP 数据处理模块
将 ATP 大气廓线数据转换为训练格式
"""
import netCDF4 as nc
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ATPProcessor:
    """ATP 数据处理器"""

    def __init__(
        self,
        target_heights: np.ndarray = np.linspace(0, 60, 301),
        qc_threshold: int = 100
    ):
        """
        Args:
            target_heights: 目标高度网格 (km)
            qc_threshold: 质量控制阈值
        """
        self.target_heights = target_heights
        self.qc_threshold = qc_threshold

    def read_atp_file(self, file_path: Path) -> Optional[Dict]:
        """读取单个 ATP 文件"""
        try:
            ds = nc.Dataset(file_path)

            # 质量控制
            qc = int(ds.getncattr('qc'))
            if qc < self.qc_threshold:
                ds.close()
                return None

            # 读取曲率半径 (用于转换冲击参数到海拔高度)
            curv = float(ds.getncattr('curv'))

            # 读取数据
            impact_parm = ds.variables['Opt_Impact_parm'][:]  # 冲击参数 (地心距离)
            msl_alt_bend = impact_parm - curv  # 转换为海拔高度

            data = {
                'bend_ang': ds.variables['Opt_Bend_ang'][:],  # 优化后的弯曲角
                'msl_alt_bend': msl_alt_bend,  # 弯曲角对应的海拔高度
                'msl_alt': ds.variables['MSL_alt'][:],  # 温度/气压对应的几何高度
                'temp': ds.variables['Temp'][:],  # 温度
                'pres': ds.variables['Pres'][:],  # 气压
                'dens': ds.variables['Dens'][:],  # 密度
                'lat': float(ds.getncattr('lat')),
                'lon': float(ds.getncattr('lon')),
                'setting': int(ds.getncattr('setting'))
            }

            ds.close()
            return data

        except Exception as e:
            logger.warning(f"读取文件失败 {file_path.name}: {e}")
            return None

    def interpolate_to_grid(
        self,
        heights: np.ndarray,
        values: np.ndarray,
        target_heights: np.ndarray
    ) -> Optional[np.ndarray]:
        """插值到标准高度网格"""
        try:
            # 移除 NaN 和无效值
            valid_mask = np.isfinite(heights) & np.isfinite(values)
            if valid_mask.sum() < 10:  # 至少需要 10 个有效点
                return None

            heights = heights[valid_mask]
            values = values[valid_mask]

            # 按高度排序
            sort_idx = np.argsort(heights)
            heights = heights[sort_idx]
            values = values[sort_idx]

            # 线性插值，使用边界值外推
            interp_values = np.interp(
                target_heights,
                heights,
                values,
                left=values[0],   # 使用最低高度的值填充
                right=values[-1]  # 使用最高高度的值填充
            )

            # 检查是否仍有 NaN（理论上不应该有）
            if np.isnan(interp_values).any():
                return None

            return interp_values

        except Exception as e:
            logger.warning(f"插值失败: {e}")
            return None

    def process_single_profile(self, data: Dict) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """处理单个廓线"""
        # 弯曲角插值 (使用海拔高度)
        bend_ang_interp = self.interpolate_to_grid(
            data['msl_alt_bend'],
            data['bend_ang'],
            self.target_heights
        )

        if bend_ang_interp is None:
            return None

        # 温度插值
        temp_interp = self.interpolate_to_grid(
            data['msl_alt'],
            data['temp'],
            self.target_heights
        )

        # 气压插值
        pres_interp = self.interpolate_to_grid(
            data['msl_alt'],
            data['pres'],
            self.target_heights
        )

        if temp_interp is None or pres_interp is None:
            return None

        # 质量控制: 检查物理合理性
        if not self._check_physical_validity(temp_interp, pres_interp, bend_ang_interp):
            return None

        # 标准化弯曲角 (log10 变换)
        bend_ang_log = np.log10(np.abs(bend_ang_interp) + 1e-6)

        # 组装输入和标签
        X = bend_ang_log  # (301,)
        Y = np.stack([temp_interp, pres_interp], axis=0)  # (2, 301) - 暂不包含湿度

        return X, Y

    def _check_physical_validity(
        self,
        temp: np.ndarray,
        pres: np.ndarray,
        bend_ang: np.ndarray
    ) -> bool:
        """检查物理合理性"""
        # 温度范围: 150-350 K
        if np.nanmin(temp) < 150 or np.nanmax(temp) > 350:
            return False

        # 气压范围: 0.01-1100 mb
        if np.nanmin(pres) < 0.01 or np.nanmax(pres) > 1100:
            return False

        # 弯曲角范围: 1e-6 - 0.1 rad
        if np.nanmin(np.abs(bend_ang)) < 1e-6 or np.nanmax(np.abs(bend_ang)) > 0.1:
            return False

        # 气压单调递减检查（放宽条件，允许小幅波动）
        pres_valid = pres[~np.isnan(pres)]
        if len(pres_valid) > 10:
            # 计算相邻点的差值
            pres_diff = np.diff(pres_valid)
            # 允许最多 20% 的点有小幅上升（< 10% 的相对变化）
            increasing_points = pres_diff > 0
            if increasing_points.sum() > len(pres_diff) * 0.2:
                return False
            # 检查是否有大幅上升
            if np.any(pres_diff > pres_valid[:-1] * 0.1):
                return False

        return True

    def process_directory(
        self,
        atp_dir: Path,
        output_dir: Path,
        max_files: Optional[int] = None
    ) -> Dict:
        """批量处理 ATP 数据"""
        atp_dir = Path(atp_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 获取所有 ATP 文件
        atp_files = sorted(atp_dir.glob("*.NC"))
        if max_files:
            atp_files = atp_files[:max_files]

        logger.info(f"找到 {len(atp_files)} 个 ATP 文件")

        X_list, Y_list = [], []
        failed_count = 0

        for file_path in tqdm(atp_files, desc="处理 ATP 数据"):
            # 读取文件
            data = self.read_atp_file(file_path)
            if data is None:
                failed_count += 1
                continue

            # 处理廓线
            result = self.process_single_profile(data)
            if result is None:
                failed_count += 1
                continue

            X, Y = result
            X_list.append(X)
            Y_list.append(Y)

        if len(X_list) == 0:
            raise ValueError("没有成功处理任何文件")

        # 转换为数组
        X_all = np.array(X_list)  # (N, 301)
        Y_all = np.array(Y_list)  # (N, 2, 301)

        logger.info(f"成功处理: {len(X_list)} 个廓线")
        logger.info(f"失败: {failed_count} 个廓线")
        logger.info(f"通过率: {len(X_list)/(len(X_list)+failed_count)*100:.1f}%")

        # 计算统计信息
        stats = self._compute_statistics(X_all, Y_all)

        # Z-Score 标准化
        X_norm, Y_norm = self._normalize_data(X_all, Y_all, stats)

        # 划分数据集
        train_x, train_y, val_x, val_y, test_x, test_y = self._split_dataset(
            X_norm, Y_norm
        )

        # 保存数据
        np.save(output_dir / 'train_x.npy', train_x)
        np.save(output_dir / 'train_y.npy', train_y)
        np.save(output_dir / 'val_x.npy', val_x)
        np.save(output_dir / 'val_y.npy', val_y)
        np.save(output_dir / 'test_x.npy', test_x)
        np.save(output_dir / 'test_y.npy', test_y)
        np.save(output_dir / 'stats.npy', stats)

        logger.info(f"数据已保存到 {output_dir}")
        logger.info(f"训练集: {len(train_x)}, 验证集: {len(val_x)}, 测试集: {len(test_x)}")

        return {
            'total': len(X_list),
            'failed': failed_count,
            'train': len(train_x),
            'val': len(val_x),
            'test': len(test_x),
            'stats': stats
        }

    def _compute_statistics(self, X: np.ndarray, Y: np.ndarray) -> Dict:
        """计算统计信息"""
        stats = {
            'x_mean': np.nanmean(X),
            'x_std': np.nanstd(X),
            'y_mean': np.nanmean(Y, axis=(0, 2)),  # (2,)
            'y_std': np.nanstd(Y, axis=(0, 2))
        }

        logger.info("=== 数据统计 (标准化前) ===")
        logger.info(f"弯曲角 (log10): mean={stats['x_mean']:.3f}, std={stats['x_std']:.3f}")
        logger.info(f"温度: mean={stats['y_mean'][0]:.1f} K, std={stats['y_std'][0]:.1f} K")
        logger.info(f"气压: mean={stats['y_mean'][1]:.1f} mb, std={stats['y_std'][1]:.1f} mb")

        return stats

    def _normalize_data(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        stats: Dict
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Z-Score 标准化"""
        X_norm = (X - stats['x_mean']) / (stats['x_std'] + 1e-8)

        Y_norm = np.zeros_like(Y)
        for i in range(Y.shape[1]):
            Y_norm[:, i, :] = (Y[:, i, :] - stats['y_mean'][i]) / (stats['y_std'][i] + 1e-8)

        return X_norm, Y_norm

    def _split_dataset(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
    ) -> Tuple:
        """划分数据集"""
        n = len(X)
        indices = np.random.permutation(n)

        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]

        return (
            X[train_idx], Y[train_idx],
            X[val_idx], Y[val_idx],
            X[test_idx], Y[test_idx]
        )


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='处理 FY-3D ATP 数据')
    parser.add_argument('--atp-dir', type=str, default='utils/atp',
                        help='ATP 数据目录')
    parser.add_argument('--output-dir', type=str, default='Data/Processed_ATP',
                        help='输出目录')
    parser.add_argument('--max-files', type=int, default=None,
                        help='最大处理文件数 (用于调试)')

    args = parser.parse_args()

    processor = ATPProcessor()
    result = processor.process_directory(
        Path(args.atp_dir),
        Path(args.output_dir),
        max_files=args.max_files
    )

    print("\n=== 处理完成 ===")
    print(f"总计: {result['total']} 个廓线")
    print(f"训练集: {result['train']} 个")
    print(f"验证集: {result['val']} 个")
    print(f"测试集: {result['test']} 个")


if __name__ == '__main__':
    main()
