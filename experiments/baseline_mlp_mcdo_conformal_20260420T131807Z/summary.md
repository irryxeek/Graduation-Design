# Split Conformal 校准摘要

- alpha: `0.05`
- 校准集: `experiments/baseline_mlp_mcdo_val_20260420T130943Z/interval_payload.npz`
- 测试集: `experiments/baseline_mlp_mcdo_test_20260420T130918Z/interval_payload.npz`

## 全局缩放量

| 变量 | q_hat | 单位 |
| --- | ---: | --- |
| temperature | 22.8936 | K |
| pressure | 3.2154 | hPa |
| humidity | 0.9279 | g/kg |

## 校准后整体结果

| 变量 | 覆盖率 | 平均区间宽度 | 均值预测MAE | 单位 |
| --- | ---: | ---: | ---: | --- |
| temperature | 0.9397 | 50.1925 | 7.8508 | K |
| pressure | 0.9586 | 38.4130 | 7.3397 | hPa |
| humidity | 0.9523 | 2.0090 | 0.3074 | g/kg |