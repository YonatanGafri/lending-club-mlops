import json
import re

def extract_notebook_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    cells = nb.get('cells', [])
    
    extracted_data = []
    
    for i, cell in enumerate(cells):
        cell_type = cell.get('cell_type')
        if cell_type == 'markdown':
            source = "".join(cell.get('source', []))
            
            # Find the closest preceding code cell with output
            prev_code_output = ""
            for j in range(i-1, -1, -1):
                prev_cell = cells[j]
                if prev_cell.get('cell_type') == 'code':
                    outputs = prev_cell.get('outputs', [])
                    output_text = []
                    for out in outputs:
                        if out.get('output_type') == 'stream':
                            output_text.append("".join(out.get('text', [])))
                        elif out.get('output_type') in ['display_data', 'execute_result']:
                            data = out.get('data', {})
                            if 'text/plain' in data:
                                output_text.append("".join(data['text/plain']))
                    prev_code_output = "\n".join(output_text)
                    if prev_code_output.strip() != "":
                        break
            
            extracted_data.append({
                'cell_index': i,
                'markdown': source,
                'prev_code_output': prev_code_output
            })
            
    with open('extracted_analysis.txt', 'w', encoding='utf-8') as f:
        for data in extracted_data:
            f.write(f"--- CELL INDEX: {data['cell_index']} ---\n")
            f.write(f"--- PRECEDING CODE OUTPUT ---\n")
            f.write(f"{data['prev_code_output']}\n")
            f.write(f"--- MARKDOWN TEXT ---\n")
            f.write(f"{data['markdown']}\n")
            f.write(f"="*80 + "\n\n")

if __name__ == '__main__':
    extract_notebook_data('Lending_Club_Project_Final.ipynb')
