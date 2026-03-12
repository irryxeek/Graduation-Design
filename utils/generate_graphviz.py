import os
from graphviz import Digraph

# Define graph attributes for a modern, tech-like aesthetic
dot = Digraph('QC_Pipeline', format='png')
dot.attr(rankdir='TB', nodesep='0.5', ranksep='0.6')
dot.attr('graph', bgcolor='#ffffff', pad='0.5')
dot.attr('node', fontname='Microsoft YaHei', fontsize='12', shape='box', style='rounded,filled', 
         fillcolor='#f8f9fa', color='#bdc3c7', penwidth='1.5')
dot.attr('edge', fontname='Microsoft YaHei', color='#34495e', penwidth='1.5', arrowsize='0.8')

# Input Node
dot.node('A', '输入数据\nFY-3D GNOS L2 ATP\n(42,871 个文件)', 
         shape='cylinder', fillcolor='#e1f5fe', color='#2980b9', fontcolor='#2c3e50', fontsize='14')

# QC Process Node (Record type for structural breakdown)
qc_label = '''<
<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6" COLOR="#bdc3c7">
  <TR><TD BGCOLOR="#3498db"><FONT COLOR="white" POINT-SIZE="14"><B>质量控制流水线 (7项物理指标)</B></FONT></TD></TR>
  <TR><TD ALIGN="LEFT" BGCOLOR="#ffffff">① qc=100 官方质量标志筛选</TD></TR>
  <TR><TD ALIGN="LEFT" BGCOLOR="#f9f9f9">② 150 K ≤ 温度(T) ≤ 350 K</TD></TR>
  <TR><TD ALIGN="LEFT" BGCOLOR="#ffffff">③ 0.01 mb ≤ 气压(P) ≤ 1100 mb</TD></TR>
  <TR><TD ALIGN="LEFT" BGCOLOR="#f9f9f9">④ 气压廓线单调递减性验证</TD></TR>
  <TR><TD ALIGN="LEFT" BGCOLOR="#ffffff">⑤ 廓线有效高度覆盖 (≥ 0 km)</TD></TR>
  <TR><TD ALIGN="LEFT" BGCOLOR="#f9f9f9">⑥ 弯曲角正值及量级合理性检查</TD></TR>
  <TR><TD ALIGN="LEFT" BGCOLOR="#ffffff">⑦ 插值有效点数验证 (≥ 10 点)</TD></TR>
</TABLE>
>'''

dot.node('B', label=qc_label, shape='none', margin='0')

# Output Node
dot.node('C', '有效观测廓线\n(共计 37,171 条)\n准备进入深度网络', 
         shape='cylinder', fillcolor='#e8f8f5', color='#27ae60', fontcolor='#145a32', fontsize='14')

# Edges
dot.edge('A', 'B', label=' 送入筛选')
dot.edge('B', 'C', label=' 总体通过率：86.7%', fontcolor='#c0392b', style='bold')

# Save and render
out_path = r'D:\02_Study\01_Schoolwork\Graduation Design\docs\midterm\质量控制流水线_Graphviz'
dot.render(out_path, cleanup=True)
print(f"Graphviz flowchart saved to {out_path}.png")
