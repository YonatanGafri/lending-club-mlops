import json
import re

def filter_metrics(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    cells = nb.get('cells', [])
    keywords = ['auc', 'roc', 'brier', 'score', 'threshold', 'profit', 'isotonic', 'calibrat', 'precision', 'recall', 'f1', 'accuracy', 'psi', 'gini']
    
    with open('metrics_audit_report.txt', 'w', encoding='utf-8') as out:
        for i, cell in enumerate(cells):
            if cell.get('cell_type') == 'markdown':
                source = "".join(cell.get('source', []))
                
                # Check if markdown contains keywords
                lower_source = source.lower()
                if any(kw in lower_source for kw in keywords):
                    # Get previous 2 code cells' outputs
                    prev_outputs = []
                    for j in range(i-1, max(-1, i-5), -1):
                        if cells[j].get('cell_type') == 'code':
                            outputs = cells[j].get('outputs', [])
                            output_text = []
                            for out_item in outputs:
                                if out_item.get('output_type') == 'stream':
                                    output_text.append("".join(out_item.get('text', [])))
                                elif out_item.get('output_type') in ['display_data', 'execute_result']:
                                    data = out_item.get('data', {})
                                    if 'text/plain' in data:
                                        output_text.append("".join(data['text/plain']))
                            if output_text:
                                prev_outputs.insert(0, f"--- Code Cell {j} Output ---\n" + "\n".join(output_text))
                    
                    out.write(f"\n{'='*80}\n")
                    out.write(f"MARKDOWN CELL {i}:\n")
                    out.write(source.strip() + "\n")
                    out.write(f"\n--- PRECEDING CODE OUTPUTS ---\n")
                    if prev_outputs:
                        out.write("\n".join(prev_outputs))
                    else:
                        out.write("NO PRECEDING CODE OUTPUTS FOUND.\n")
                    out.write(f"{'='*80}\n")

if __name__ == '__main__':
    filter_metrics('Lending_Club_Project_Final.ipynb')
    print("Metrics audit report saved to metrics_audit_report.txt")
