# Split Conformal 校准摘要

- alpha: `0.05`
- 校准集: `experiments/uncertainty_probe_val_20260420T131307Z/interval_payload.npz`
- 测试集: `experiments/uncertainty_probe_test_20260420T131003Z/interval_payload.npz`

## 全局缩放量

| 变量 | q_hat | 单位 |
| --- | ---: | --- |
| temperature | 22.2060 | K |
| pressure | 25.4557 | hPa |
| humidity | 1.1634 | g/kg |

## 校准后整体结果

| 变量 | 覆盖率 | 平均区间宽度 | 均值预测MAE | 单位 |
| --- | ---: | ---: | ---: | --- |
| temperature | 0.9533 | 73.0899 | 11.8652 | K |
| pressure | 0.9584 | 60.5226 | 7.3364 | hPa |
| humidity | 0.9570 | 3.2382 | 0.4287 | g/kg |