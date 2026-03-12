import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set up Chinese font
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(6, 4))
ax.axis('off')

text = """质量控制（7 项指标）
① qc=100 官方质量标志筛选
② 温度物理范围约束：150 K ≤ T ≤ 350 K
③ 气压物理范围约束：0.01 mb ≤ P ≤ 1100 mb
④ 气压廓线单调递减性验证
⑤ 廓线有效高度覆盖检查（≥ 0 km）
⑥ 弯曲角正值及量级合理性检查
⑦ 插值有效点数验证（≥ 10 点）

通过率：86.7%"""

# Add a rounded rectangle
rect = patches.FancyBboxPatch(
    (0.05, 0.05), 0.9, 0.9,
    boxstyle="round,pad=0.05",
    edgecolor='#2c3e50',
    facecolor='#ecf0f1',
    linewidth=2
)
ax.add_patch(rect)

# Add text
ax.text(
    0.5, 0.5, text,
    ha='center', va='center',
    fontsize=14,
    linespacing=1.8,
    color='#34495e',
    fontweight='bold'
)

# Draw an arrow pointing downwards
ax.annotate(
    "",
    xy=(0.5, -0.15), xycoords='data',
    xytext=(0.5, 0.05), textcoords='data',
    arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=2, shrinkA=0, shrinkB=0)
)

plt.xlim(0, 1)
plt.ylim(-0.2, 1)
plt.tight_layout()

output_path = r'D:\02_Study\01_Schoolwork\Graduation Design\docs\midterm\质量控制模块_优化.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Image saved to {output_path}")
