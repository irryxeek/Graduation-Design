import docx
import re

doc_path = r'D:\02_Study\01_Schoolwork\Graduation Design\docs\midterm\中期 - v2.0_fixed_final_v2.docx'
out_path = r'D:\02_Study\01_Schoolwork\Graduation Design\docs\midterm\中期 - v2.0_fixed_final.docx'
doc = docx.Document(doc_path)

# Track current counters for each level
h1_count = 0
h2_count = 0
h3_count = 0

def clean_title(text):
    # Remove existing numbers at the start like "1 ", "2.1 ", "3.2.1 "
    text = re.sub(r'^(\d+\.?)+(\s+|　)', '', text.strip())
    # Also handle things like "（1）" if they were wrongly used as section headers
    # text = re.sub(r'^（\d+）', '', text)
    return text

for p in doc.paragraphs:
    if p.style.name == 'Heading 1':
        h1_count += 1
        h2_count = 0
        h3_count = 0
        if "参考文献" in p.text:
            # References usually don't have a number, or just 6.
            p.text = f"{h1_count} {clean_title(p.text)}"
        else:
            p.text = f"{h1_count} {clean_title(p.text)}"
        print(f"Renumbered H1: {p.text}")
        
    elif p.style.name == 'Heading 2':
        h2_count += 1
        h3_count = 0
        p.text = f"{h1_count}.{h2_count} {clean_title(p.text)}"
        print(f"  Renumbered H2: {p.text}")
        
    elif p.style.name == 'Heading 3':
        h3_count += 1
        p.text = f"{h1_count}.{h2_count}.{h3_count} {clean_title(p.text)}"
        print(f"    Renumbered H3: {p.text}")

# Note: TOC in docx is a field. We've updated the source headings.
# The user will need to right-click the TOC in Word and select "Update Field" -> "Update entire table".
doc.save(out_path)
print(f"\nRenumbering complete. Saved to: {out_path}")
print("Note: To update the TOC, please open the document in Word, right-click the Table of Contents, and select 'Update Field' -> 'Update entire table'.")
