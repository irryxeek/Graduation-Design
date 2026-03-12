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
    
    # Iterate through paragraphs to find $...$
    # We do it from end to start to avoid index shifting problems, 
    # but paragraph objects might be safer to just iterate and find text.
    for i in range(1, doc.Paragraphs.Count + 1):
        p = doc.Paragraphs(i)
        text = p.Range.Text
        if text and '$' in text:
            # We found a paragraph with at least one $. 
            # We need to find all pairs of $
            # To do this safely, we can use the Find object on this specific paragraph's Range
            rng = p.Range
            find_obj = rng.Find
            find_obj.ClearFormatting()
            find_obj.MatchWildcards = False
            find_obj.Text = "$"
            
            # Find the first $
            while find_obj.Execute():
                start_pos = rng.Start
                # Find the second $
                rng_end = p.Range
                rng_end.Start = start_pos + 1
                find_end = rng_end.Find
                find_end.ClearFormatting()
                find_end.MatchWildcards = False
                find_end.Text = "$"
                
                if find_end.Execute():
                    end_pos = rng_end.End
                    
                    # Now we have a range from start_pos to end_pos containing $...$
                    math_rng = doc.Range(start_pos, end_pos)
                    math_text = math_rng.Text[1:-1] # strip $
                    
                    math_rng.Text = math_text
                    
                    # Convert to OMath
                    # Add OMath
                    omaths = math_rng.OMaths
                    if omaths.Count == 0:
                        new_math_rng = omaths.Add(math_rng)
                        # OMaths.Add returns a Range. The OMath object is inside it.
                        if new_math_rng.OMaths.Count > 0:
                            new_math_rng.OMaths(1).BuildUp()
                    
                    # Reset search range to continue after this math zone
                    rng.Start = math_rng.End
                    rng.End = p.Range.End
                else:
                    break

    doc.SaveAs(out_path)
    print("Done")
except Exception as e:
    import traceback
    traceback.print_exc()
    print("Error:", e)
finally:
    try:
        doc.Close(False)
    except:
        pass
    word.Quit()
