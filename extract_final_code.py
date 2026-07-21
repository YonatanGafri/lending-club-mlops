import json

with open('Lending_Club_Project_Final.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

code_cells = []
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        code_cells.append(f"\n# ==========================================\n# CODE CELL\n# ==========================================\n{source}\n")

with open('extracted_final_code.py', 'w', encoding='utf-8') as f:
    f.writelines(code_cells)
