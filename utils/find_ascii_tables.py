import docx

doc_path = r"D:\02_Study\01_Schoolwork\Graduation Design\docs\midterm\2024 - 本科毕业设计中期模板(修改) - v2.0.docx"
doc = docx.Document(doc_path)

for i, p in enumerate(doc.paragraphs):
    if '┌' in p.text:
        print(f"Line {i}: {p.text.strip()}")
