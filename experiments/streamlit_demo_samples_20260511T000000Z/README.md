# Streamlit 验收演示样本

- 来源数据集: `Data/Processed_ATP_WAP_2025`
- 总测试样本数: `9618`
- 选取样本数: `8`
- 推荐上传文件: `streamlit_demo_upload.npz`

## 选取原则
- 低弯曲角跨度样本：展示稳定输入场景。
- 中位复合样本：展示常规代表场景。
- 高湿度梯度样本：展示湿度变化明显场景。
- 高弯曲角跨度样本：展示输入变化幅度大的场景。

## 样本索引
8397, 3868, 6433, 8871, 9364, 5611, 7521, 8264

## 文件说明
- `streamlit_demo_upload.npz`: 含 `x` 和 `y`，可直接上传前端演示。
- `streamlit_demo_inputs_only.npz`: 仅含 `x`，用于只演示推理不展示真值的情况。
- `streamlit_demo_x.npy` / `streamlit_demo_y.npy`: 便于单独加载或二次处理。

## 单样本字段
- `x`: 弯曲角输入，shape `(N, 301)`。
- `y`: 真值标签，shape `(N, 3, 301)`，通道顺序为温度、气压、湿度。
