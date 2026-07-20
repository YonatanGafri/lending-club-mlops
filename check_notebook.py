import json
import sys

def check_notebook(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    cells = nb.get('cells', [])
    
    print(f"Total cells: {len(cells)}")
    
    for i, cell in enumerate(cells):
        cell_type = cell.get('cell_type')
        if cell_type == 'markdown':
            source = "".join(cell.get('source', [])).strip()
            preview = source[:50].replace('\n', ' ') + '...' if source else '[EMPTY]'
            print(f"Cell {i} [Markdown]: {preview}")
        elif cell_type == 'code':
            exec_count = cell.get('execution_count')
            source = "".join(cell.get('source', [])).strip()
            preview = source[:50].replace('\n', ' ') + '...' if source else '[EMPTY]'
            outputs = cell.get('outputs', [])
            has_image = any('image/png' in out.get('data', {}) or 'image/jpeg' in out.get('data', {}) for out in outputs)
            has_output = len(outputs) > 0
            
            output_desc = []
            if has_output:
                output_desc.append(f"{len(outputs)} outputs")
            if has_image:
                output_desc.append("has image(s)")
                
            out_str = ", ".join(output_desc) if output_desc else "No output"
            print(f"Cell {i} [Code] (Exec: {exec_count}): {preview} -> {out_str}")

if __name__ == '__main__':
    check_notebook('Lending_Club_Project_Final.ipynb')
