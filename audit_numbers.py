import json
import re

with open('Lending_Club_Project_Final.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("==== Unverified Numbers Audit ====\n")

all_code_output = ""
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        for out in cell.get('outputs', []):
            if out.get('name') == 'stdout':
                text = out.get('text', [])
                if isinstance(text, list):
                    all_code_output += "".join(text)
                else:
                    all_code_output += text
            elif 'data' in out and 'text/plain' in out['data']:
                text = out['data']['text/plain']
                if isinstance(text, list):
                    all_code_output += "".join(text)
                else:
                    all_code_output += text
                    
# Now iterate markdown cells
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown':
        source = "".join(cell['source'])
        
        # Find all decimal numbers with 3 or 4 decimal places
        matches = set(re.findall(r'0\.\d{3,4}', source))
        
        # Also find percentages with 1 or 2 decimal places
        pct_matches = set(re.findall(r'\d{1,2}\.\d{1,2}%', source))
        
        unverified = []
        for match in matches:
            if match not in all_code_output:
                unverified.append(match)
                
        for pct in pct_matches:
            # check if pct is in output, or the pure number without %
            num = pct.replace('%', '')
            if pct not in all_code_output and num not in all_code_output:
                unverified.append(pct)
                
        if unverified:
             # Context of where it was claimed
             print(f"Cell {i} claims unverified metrics: {unverified}")
             # Print a snippet of the markdown around the unverified number
             lines = source.split('\n')
             for line in lines:
                 if any(u in line for u in unverified):
                     print(f"  -> {line.strip()}")
             print("-" * 40)
