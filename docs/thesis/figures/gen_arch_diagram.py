"""生成系统架构图 (图 3-1)"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# 颜色方案
colors = {
    'data':    '#4A90D9',
    'model':   '#5BA55B',
    'train':   '#E8A838',
    'infer':   '#D96459',
    'display': '#8E6BBF',
    'arrow':   '#555555',
    'input':   '#888888',
}

# 层定义: (y_center, color_key, 层名, 描述)
layers = [
    (1.4, 'data',    '数据层',       '文件配对 · 质量控制 · 高度插值 · 对数变换 · Z-Score 标准化 · 数据集划分'),
    (3.2, 'model',   '模型层',       '扩散噪声调度 · 前向加噪 · DDPM/DDIM 反向采样 · 增强版条件 U-Net'),
    (5.0, 'train',   '训练评估层',   '噪声预测损失 · 变量加权 · 湿度梯度约束 · 早停 · RMSE/Bias/CC 评估'),
    (6.8, 'infer',   '推理服务层',   'DDPM 1000 步 / DDIM 50 步采样 · 反标准化 · 后处理'),
    (8.6, 'display', '展示层',       'Streamlit 交互前端 · 数据集浏览 · 上传分析 · 多变量剖面对比'),
]

box_w = 8.0
box_h = 1.15
box_x = 1.0

for y, ckey, name, desc in layers:
    # 圆角矩形
    rect = mpatches.FancyBboxPatch(
        (box_x, y - box_h / 2), box_w, box_h,
        boxstyle="round,pad=0.15",
        facecolor=colors[ckey], edgecolor='white',
        linewidth=2, alpha=0.88
    )
    ax.add_patch(rect)
    # 层名 (左侧加粗)
    ax.text(box_x + 0.5, y + 0.12, name,
            fontsize=14, fontweight='bold', color='white',
            va='center', ha='left')
    # 描述 (层名下方)
    ax.text(box_x + 0.5, y - 0.28, desc,
            fontsize=9.5, color='#F0F0F0',
            va='center', ha='left')

# 层间箭头 + 数据流标注
arrow_props = dict(
    arrowstyle='->', color=colors['arrow'],
    lw=1.8, connectionstyle='arc3,rad=0'
)
flow_labels = [
    (1.4, 3.2, 'train_x / train_y / val / test (.npy) + 统计量'),
    (3.2, 5.0, '模型结构 + 扩散参数'),
    (5.0, 6.8, '最优模型权重 (.pth) + 评估报告'),
    (6.8, 8.6, '预测廓线 + 评估指标'),
]

for y1, y2, label in flow_labels:
    y_start = y1 + box_h / 2 + 0.05
    y_end = y2 - box_h / 2 - 0.05
    ax.annotate('', xy=(5, y_end), xytext=(5, y_start), arrowprops=arrow_props)
    ax.text(5, (y_start + y_end) / 2, label,
            fontsize=8.5, color=colors['arrow'],
            va='center', ha='center',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.85))

# 外部输入标注 (数据层左侧)
ax.annotate('FY-3D GNOS\nATP + WAP\nNetCDF 文件',
            xy=(box_x, 1.4), xytext=(-0.6, 0.2),
            fontsize=9, color=colors['input'], ha='center', va='center',
            arrowprops=dict(arrowstyle='->', color=colors['input'], lw=1.2))

# 外部输出标注 (展示层右侧)
ax.annotate('用户交互\n(浏览器)',
            xy=(box_x + box_w, 8.6), xytext=(10.6, 9.6),
            fontsize=9, color=colors['input'], ha='center', va='center',
            arrowprops=dict(arrowstyle='->', color=colors['input'], lw=1.2))

plt.tight_layout()
plt.savefig('docs/thesis/figures/system_architecture.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print('saved: docs/thesis/figures/system_architecture.png')
