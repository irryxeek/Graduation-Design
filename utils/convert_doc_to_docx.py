import os
import win32com.client

doc_path = r"D:\02_Study\01_Schoolwork\Graduation Design\docs\midterm\2024 - 本科毕业设计中期模板(修改) - v2.0.doc"
docx_path = doc_path + "x"

word = win32com.client.Dispatch("Word.Application")
word.Visible = False
try:
    doc = word.Documents.Open(doc_path)
    doc.SaveAs2(docx_path, FileFormat=16) # 16 is wdFormatXMLDocument
    doc.Close()
    print(f"Converted successfully to {docx_path}")
except Exception as e:
    print(f"Error: {e}")
finally:
    word.Quit()
