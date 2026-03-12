import docx

docx_path = r"D:\02_Study\01_Schoolwork\Graduation Design\docs\midterm\2024 - 本科毕业设计中期模板(修改) - v2.0.docx"

try:
    doc = docx.Document(docx_path)
    print("Potential formula paragraphs:")
    count = 0
    for p in doc.paragraphs:
        # Heuristic: looks for lines containing typical math symbols or equations
        text = p.text.strip()
        if any(char in text for char in ['=', '+', '-', '*', '/', '√', 'μ', 'σ', 'β', 'α', 'ε', 'θ', '∑', '∫', 'log', 'exp', '^', '_']) and len(text) > 0 and len(text) < 100:
            # Filter out generic text that just happens to have a hyphen
            if not text.startswith("202") and "图" not in text and "表" not in text and not text.endswith("报告") and "：" not in text and "-" in text and len(text.split("-")) < 3:
                # further refinement
                pass
            if sum(c.isalpha() for c in text) < len(text) * 0.8: # high density of non-alphabet means likely a formula
                print(f"[{count}] {text}")
                count += 1
                if count > 20:
                    break
except Exception as e:
    print(f"Error: {e}")