import json
import re

with open('Lending_Club_Project_Final.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

last_outputs = []

with open('audit_results.txt', 'w', encoding='utf-8') as out_f:
    out_f.write("==== FORENSIC AUDIT REPORT ====\n")
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            # collect printed text
            outs = []
            for out in cell.get('outputs', []):
                if out.get('name') == 'stdout':
                    text = out.get('text', [])
                    if isinstance(text, list):
                        outs.extend(text)
                    else:
                        outs.append(text)
                elif 'data' in out and 'text/plain' in out['data']:
                    text = out['data']['text/plain']
                    if isinstance(text, list):
                        outs.extend(text)
                    else:
                        outs.append(text)
            
            # Only keep lines that have numbers and keywords like AUC, Brier, Test, etc.
            filtered = []
            for line in "".join(outs).split('\n'):
                if any(kw in line.lower() for kw in ['auc', 'brier', 'f1', 'accuracy', 'precision', 'recall', 'threshold', 'profit', 'psi', '0.', 'score']):
                    filtered.append(line.strip())
            if filtered:
                last_outputs = filtered
        
        elif cell['cell_type'] == 'markdown':
            source = "".join(cell['source'])
            lines = source.split('\n')
            metric_lines = []
            for line in lines:
                 # Find numbers like 0.xxx or percentages or keywords
                 if re.search(r'0\.\d+', line) or re.search(r'\d+\.?\d*%', line) or any(kw in line.lower() for kw in ['auc', 'brier', 'psi', 'profit', 'isotonic', 'threshold', 'woe', 'iv']):
                     metric_lines.append(line.strip())
            
            if metric_lines:
                out_f.write("\n--- MARKDOWN CLAIMS ---\n")
                for m in metric_lines:
                    if len(m) > 0: out_f.write(m + "\n")
                out_f.write("--- PRECEDING CODE OUTPUTS (METRICS) ---\n")
                for o in last_outputs:
                    if len(o) > 0: out_f.write(o + "\n")
