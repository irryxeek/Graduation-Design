from pathlib import Path
import docx

BASE_DIR = Path(__file__).resolve().parents[1]
doc_path = BASE_DIR / 'reports' / '中期 - v2.0_fixed.docx'
out_path = BASE_DIR / 'workspace' / 'tmp' / 'paragraphs_233.txt'
doc = docx.Document(str(doc_path))

with open(out_path, 'w', encoding='utf-8') as f:
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
