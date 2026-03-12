import docx
import sys
import os

doc_path = r'D:\02_Study\01_Schoolwork\Graduation Design\docs\midterm\中期 - v2.0_fixed.docx'
doc = docx.Document(doc_path)

with open('table_previews.txt', 'w', encoding='utf-8') as f:
    for i, table in enumerate(doc.tables):
        f.write(f'--- Table {i} ---\n')
        f.write(f'Style: {table.style.name if table.style else "None"}\n')
        try:
            if table.rows and table.rows[0].cells:
                f.write(f'Preview: {table.rows[0].cells[0].text.replace(chr(10), " ")[:50]}\n')
        except Exception as e:
            f.write(f'Error: {e}\n')
