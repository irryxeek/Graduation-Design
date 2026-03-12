import docx
import json

docx_path = r"D:\02_Study\01_Schoolwork\Graduation Design\docs\midterm\2024 - 本科毕业设计中期模板(修改) - v2.0.docx"

try:
    doc = docx.Document(docx_path)
    tables_data = []
    
    for i, table in enumerate(doc.tables):
        table_data = []
        for row in table.rows:
            row_data = [cell.text.replace('\n', ' ').strip() for cell in row.cells]
            table_data.append(row_data)
        tables_data.append({"table_index": i, "data": table_data})
        
    print(json.dumps(tables_data, ensure_ascii=False, indent=2))
except Exception as e:
    print(f"Error: {e}")