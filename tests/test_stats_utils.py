import tempfile
import unittest
from pathlib import Path

import numpy as np

from ro_retrieval.stats_utils import (
    canonicalize_stats,
    compute_fallback_stats,
    load_stats_from_dir,
)


class StatsUtilsTest(unittest.TestCase):
    def test_canonicalize_stats_supports_legacy_keys(self):
        stats = canonicalize_stats(
            {
                "x_mean": 1.0,
                "x_std": 2.0,
                "y_means": [3.0, 4.0, 5.0],
                "y_stds": [0.1, 0.2, 0.3],
            }
        )
        self.assertIn("y_mean", stats)
        self.assertIn("y_std", stats)
        self.assertEqual(stats["stats_space"], "physical")
        np.testing.assert_allclose(stats["y_mean"], np.array([3.0, 4.0, 5.0], dtype=np.float32))

    def test_compute_fallback_stats_detects_normalized_arrays(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0.0, 1.0, size=(8, 301)).astype(np.float32)
        y = rng.normal(0.0, 1.0, size=(8, 3, 301)).astype(np.float32)
        stats = compute_fallback_stats(x, y)
        self.assertEqual(stats["stats_space"], "normalized")
        np.testing.assert_allclose(stats["y_std"], np.ones(3, dtype=np.float32))

    def test_load_stats_from_dir_prefers_stats_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "stats.npy"
            expected = {
                "x_mean": np.array(0.5, dtype=np.float32),
                "x_std": np.array(1.5, dtype=np.float32),
                "y_mean": np.array([1.0, 2.0, 3.0], dtype=np.float32),
                "y_std": np.array([0.1, 0.2, 0.3], dtype=np.float32),
                "target_heights": np.linspace(0, 60, 301, dtype=np.float32),
            }
            np.save(path, expected, allow_pickle=True)
            loaded = load_stats_from_dir(tmp_dir)
            np.testing.assert_allclose(loaded["y_mean"], expected["y_mean"])
            self.assertEqual(loaded["stats_space"], "physical")


if __name__ == "__main__":
    unittest.main()
