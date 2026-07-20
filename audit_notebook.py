import json
import sys

notebook_path = "Lending_Club_Project_Final.ipynb"
try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
except Exception as e:
    print(f"Error reading notebook: {e}")
    sys.exit(1)

cells = nb.get('cells', [])
markdown_count = 0
code_count = 0
unexecuted_cells = 0
orphan_graphs = []

for i, cell in enumerate(cells):
    cell_type = cell['cell_type']
    if cell_type == 'markdown':
        markdown_count += 1
    elif cell_type == 'code':
        code_count += 1
        
        # Check execution state
        exec_count = cell.get('execution_count')
        source = "".join(cell.get('source', []))
        if source.strip() != "" and exec_count is None:
            unexecuted_cells += 1
            
        # Check for image/plot outputs
        has_graph = False
        outputs = cell.get('outputs', [])
        for out in outputs:
            if out.get('output_type') in ['display_data', 'execute_result']:
                data = out.get('data', {})
                if 'image/png' in data or 'image/jpeg' in data or 'image/svg+xml' in data:
                    has_graph = True
                    break
                    
        if has_graph:
            # Check if preceding cell is markdown
            prev_is_md = False
            if i > 0 and cells[i-1]['cell_type'] == 'markdown':
                prev_source = "".join(cells[i-1].get('source', [])).strip()
                if prev_source != "":
                    prev_is_md = True
                    
            # Check if following cell is markdown
            next_is_md = False
            if i + 1 < len(cells) and cells[i+1]['cell_type'] == 'markdown':
                next_source = "".join(cells[i+1].get('source', [])).strip()
                if next_source != "":
                    next_is_md = True
                    
            if not prev_is_md or not next_is_md:
                # Find the nearest section header to give context
                section = "Unknown Section"
                for j in range(i-1, -1, -1):
                    if cells[j]['cell_type'] == 'markdown':
                        src = "".join(cells[j].get('source', []))
                        for line in src.split('\n'):
                            if line.startswith('#'):
                                section = line.strip()
                                break
                        if section != "Unknown Section":
                            break
                
                issue = []
                if not prev_is_md:
                    issue.append("missing explanation before")
                if not next_is_md:
                    issue.append("missing explanation after")
                
                orphan_graphs.append({
                    "cell_index": i,
                    "section": section,
                    "issue": " and ".join(issue)
                })

print(f"Total cells: {len(cells)}")
print(f"Markdown cells: {markdown_count}")
print(f"Code cells: {code_count}")
print(f"Ratio Code/Total: {code_count/len(cells):.2f}")
print(f"Unexecuted code cells: {unexecuted_cells}")
print(f"Orphan graphs found: {len(orphan_graphs)}")
for og in orphan_graphs:
    print(f"SEVERE WARNING: Orphan Graph at cell index {og['cell_index']} in section '{og['section']}'. Issue: {og['issue']}")
