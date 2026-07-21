import json
import re
import sys

def extract_audit_data(notebook_path, output_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    with open(output_path, 'w', encoding='utf-8') as out:
        for i, cell in enumerate(nb.get('cells', [])):
            cell_type = cell.get('cell_type')
            out.write(f"\\n{'='*40}\\n")
            out.write(f"CELL {i} [{cell_type.upper()}]\\n")
            out.write(f"{'='*40}\\n")
            
            if cell_type == 'markdown':
                source = "".join(cell.get('source', []))
                out.write(source + "\\n")
            elif cell_type == 'code':
                outputs = cell.get('outputs', [])
                for idx, output in enumerate(outputs):
                    output_type = output.get('output_type')
                    if output_type == 'stream':
                        out.write(f"-- Output {idx} (stream) --\\n")
                        out.write("".join(output.get('text', [])) + "\\n")
                    elif output_type in ['execute_result', 'display_data']:
                        data = output.get('data', {})
                        if 'text/plain' in data:
                            out.write(f"-- Output {idx} (text/plain) --\\n")
                            out.write("".join(data['text/plain']) + "\\n")
                        if 'text/html' in data:
                            html = "".join(data['text/html'])
                            # extract tables or numbers roughly
                            text = re.sub('<[^<]+>', ' ', html)
                            # Compress whitespace
                            text = re.sub(r'\\s+', ' ', text)
                            out.write(f"-- Output {idx} (html text) --\\n")
                            out.write(text.strip() + "\\n")
                        if 'application/json' in data:
                            out.write(f"-- Output {idx} (json) --\\n")
                            out.write(json.dumps(data['application/json']) + "\\n")

if __name__ == "__main__":
    extract_audit_data('Lending_Club_Project_Final.ipynb', 'audit_notebook_data.txt')
    print("Extraction complete. File saved to audit_notebook_data.txt")
