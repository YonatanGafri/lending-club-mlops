import json
import re

notebook_path = 'Lending_Club_Project_Final.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

last_code_outputs = []

with open('forensic_audit_output.txt', 'w', encoding='utf-8') as out_f:
    for i, cell in enumerate(nb.get('cells', [])):
        if cell['cell_type'] == 'code':
            outputs = []
            for out in cell.get('outputs', []):
                if 'text' in out:
                    outputs.append("".join(out['text']))
                elif 'data' in out and 'text/plain' in out['data']:
                    outputs.append("".join(out['data']['text/plain']))
            if outputs:
                last_code_outputs = outputs
        elif cell['cell_type'] == 'markdown':
            source = "".join(cell.get('source', []))
            # Look for metric keywords or decimal numbers
            keywords = ['auc', 'brier', 'psi', 'threshold', 'profit', 'isotonic', 'woe', 'iv', 'roc', 'pr-auc', '0.']
            if any(kw in source.lower() for kw in keywords):
                out_f.write(f"--- Markdown Cell {i} ---\n")
                out_f.write(source + "\n")
                out_f.write("--- Preceding Code Outputs ---\n")
                for out in last_code_outputs:
                    out_f.write(out + "\n")
                out_f.write("="*80 + "\n")
