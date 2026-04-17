from pathlib import Path
import docx
import sys
import os

BASE_DIR = Path(__file__).resolve().parents[1]
doc_path = BASE_DIR / 'reports' / '中期 - v2.0_fixed.docx'
out_path = BASE_DIR / 'workspace' / 'tmp' / 'table_previews.txt'
doc = docx.Document(str(doc_path))

with open(out_path, 'w', encoding='utf-8') as f:
    for i, table in enumerate(doc.tables):
        f.write(f'--- Table {i} ---\n')
        f.write(f'Style: {table.style.name if table.style else "None"}\n')
        try:
            if table.rows and table.rows[0].cells:
                f.write(f'Preview: {table.rows[0].cells[0].text.replace(chr(10), " ")[:50]}\n')
        except Exception as e:
            f.write(f'Error: {e}\n')
