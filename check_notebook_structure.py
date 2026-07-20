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

unexecuted_cells = []
orphan_graphs = []
missing_sections = []

print(f"Total cells: {len(cells)}")

for i, cell in enumerate(cells):
    cell_type = cell['cell_type']
    
    if cell_type == 'code':
        # Check execution state
        exec_count = cell.get('execution_count')
        source = "".join(cell.get('source', []))
        if source.strip() != "":
            # Only complain about unexecuted non-empty cells
            if exec_count is None:
                unexecuted_cells.append(i)
                
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
            # Check if next cell is markdown
            if i + 1 < len(cells):
                next_cell = cells[i+1]
                if next_cell['cell_type'] != 'markdown':
                    orphan_graphs.append((i, "Followed by a non-markdown cell"))
                else:
                    # check if the markdown is not just empty
                    next_source = "".join(next_cell.get('source', [])).strip()
                    if next_source == "":
                        orphan_graphs.append((i, "Followed by an empty markdown cell"))
            else:
                orphan_graphs.append((i, "Last cell is a graph with no following markdown"))

print(f"Unexecuted non-empty code cells: {len(unexecuted_cells)}")
if unexecuted_cells:
    print(f"Indices: {unexecuted_cells[:10]}...")

print(f"Orphan graphs found: {len(orphan_graphs)}")
for og in orphan_graphs:
    print(f"Cell index {og[0]}: {og[1]}")
    
# check if required sections exist
text_content = ""
for cell in cells:
    if cell['cell_type'] == 'markdown':
        text_content += "".join(cell.get('source', [])) + "\n"

requirements = {
    "Optimal K Selection": "Silhouette",
    "Dimensionality Reduction": "t-SNE",
    "Cross-Validation": "TimeSeriesSplit",
    "Deep Learning Baseline": "Grinsztajn"
}

for req, keyword in requirements.items():
    if keyword.lower() not in text_content.lower():
        missing_sections.append(req)

print(f"Missing sections: {missing_sections}")
