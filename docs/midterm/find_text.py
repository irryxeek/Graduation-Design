import docx

doc_path = r'D:\02_Study\01_Schoolwork\Graduation Design\docs\midterm\中期 - v2.0_fixed.docx'
doc = docx.Document(doc_path)

with open('paragraphs_233.txt', 'w', encoding='utf-8') as f:
    in_section = False
    for p in doc.paragraphs:
        if '2.3.3' in p.text and '数据规模对性能的影响' in p.text:
            in_section = True
            f.write(f"FOUND: {p.text}\n")
            continue
        
        if in_section:
            if '2.3.4' in p.text or '2.4' in p.text or '3 ' in p.text or ('总结' in p.text and len(p.text) < 10): # heuristic to stop
                break
            if p.text.strip():
                f.write(f"P: {p.text}\n")
