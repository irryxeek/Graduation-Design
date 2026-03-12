import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set up Chinese font
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(8, 12))
ax.axis('off')

boxes = [
    {
        "text": "原始 FY-3D GNOS L2 ATP 文件\n42,871 个 NetCDF 文件",
        "y": 0.9,
        "height": 0.08
    },
    {
        "text": "质量控制（7 项指标）\n① qc=100 官方质量标志筛选\n② 温度物理范围约束：150 K ≤ T ≤ 350 K\n③ 气压物理范围约束：0.01 mb ≤ P ≤ 1100 mb\n④ 气压廓线单调递减性验证\n⑤ 廓线有效高度覆盖检查（≥ 0 km）\n⑥ 弯曲角正值及量级合理性检查\n⑦ 插值有效点数验证（≥ 10 点）\n（通过率：86.7%）",
        "y": 0.65,
        "height": 0.22
    },
    {
        "text": "插值至标准垂直高度网格\n0 ~ 60 km 等间距 301 层（分辨率 0.2 km）",
        "y": 0.45,
        "height": 0.08
    },
    {
        "text": "非线性数值变换\n弯曲角：log₁₀(|BA| + 1×10⁻⁶)\n气压：log₁₀(P)  [解决4个数量级跨度问题]\n温度：直接使用原始值",
        "y": 0.28,
        "height": 0.12
    },
    {
        "text": "Z-Score 标准化\nx' = (x - μ) / σ\n（各变量独立计算均值和标准差）",
        "y": 0.12,
        "height": 0.1
    },
    {
        "text": "训练/验证/测试集划分（7:1.5:1.5）\n训练集：26,019   验证集：5,575   测试集：5,577",
        "y": -0.02,
        "height": 0.08
    }
]

# Draw boxes and text
box_width = 0.8
x_center = 0.5
for box in boxes:
    y_center = box["y"]
    h = box["height"]
    ax.add_patch(
        patches.FancyBboxPatch(
            (x_center - box_width/2, y_center - h/2),
            box_width, h,
            boxstyle="round,pad=0.03",
            ec="#1f77b4", fc="#f0f8ff", lw=2
        )
    )
    ax.text(x_center, y_center, box["text"], ha='center', va='center', fontsize=12, linespacing=1.6)

# Draw arrows
arrows = [
    (0.9 - 0.04 - 0.03, 0.65 + 0.11 + 0.03, ""),
    (0.65 - 0.11 - 0.03, 0.45 + 0.04 + 0.03, "37,171 条有效廓线"),
    (0.45 - 0.04 - 0.03, 0.28 + 0.06 + 0.03, ""),
    (0.28 - 0.06 - 0.03, 0.12 + 0.05 + 0.03, ""),
    (0.12 - 0.05 - 0.03, -0.02 + 0.04 + 0.03, "")
]

for y_start, y_end, label in arrows:
    ax.annotate(
        "",
        xy=(x_center, y_end), xycoords='data',
        xytext=(x_center, y_start), textcoords='data',
        arrowprops=dict(arrowstyle="->", color="gray", lw=2, shrinkA=0, shrinkB=0)
    )
    if label:
        ax.text(x_center + 0.05, (y_start + y_end)/2, label, ha='left', va='center', fontsize=11, color='#d62728', fontweight='bold')

plt.xlim(0, 1)
plt.ylim(-0.1, 1.0)
plt.tight_layout()
output_path = r'D:\02_Study\01_Schoolwork\Graduation Design\docs\midterm\数据处理流水线.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Image saved to {output_path}")
