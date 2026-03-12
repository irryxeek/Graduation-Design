import win32com.client
import os

doc_path = r"D:\02_Study\01_Schoolwork\Graduation Design\docs\midterm\2024 - 本科毕业设计中期模板(修改) - v2.0_表格优化.docx"
out_path = r"D:\02_Study\01_Schoolwork\Graduation Design\docs\midterm\2024 - 本科毕业设计中期模板(修改) - v2.0_公式优化.docx"
doc_path = os.path.abspath(doc_path)
out_path = os.path.abspath(out_path)

word = win32com.client.Dispatch("Word.Application")
word.Visible = False

try:
    doc = word.Documents.Open(doc_path)
    
    # Try finding \$*\$ and creating OMath
    # Word wildcards can't easily capture the inner group.
    # Instead, we do it paragraph by paragraph and use normal string operations.
    
    # We will search for exactly \$*\$
    find_obj = doc.Content.Find
    find_obj.ClearFormatting()
    find_obj.MatchWildcards = True
    find_obj.Text = "\$*\$"
    
    while find_obj.Execute():
        rng = doc.Range(find_obj.Parent.Start, find_obj.Parent.End)
        text = rng.Text
        # Remove the $ signs
        math_text = text[1:-1]
        rng.Text = math_text
        
        # Add OMath
        # In newer Word versions, inserting MathML or OMML string via Selection or Range is possible.
        # Let's see if Word natively supports building up LaTeX if it's just UnicodeMath.
        omath = rng.OMaths.Add(rng)
        omath.BuildUp()

    doc.SaveAs(out_path)
    print("Done")
except Exception as e:
    print("Error:", e)
finally:
    try:
        doc.Close(False)
    except:
        pass
    word.Quit()
