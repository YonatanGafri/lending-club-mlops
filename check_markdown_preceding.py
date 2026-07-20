import json

def check_preceding_markdown(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            nb = json.load(f)
    except Exception as e:
        print(f"Error reading notebook: {e}")
        return

    cells = nb.get('cells', [])
    orphan_code_cells = []

    for i, cell in enumerate(cells):
        if cell['cell_type'] == 'code':
            if i == 0 or cells[i-1]['cell_type'] != 'markdown':
                orphan_code_cells.append(i)

    print(f"Total Code Cells: {len([c for c in cells if c['cell_type'] == 'code'])}")
    print(f"Code cells not preceded by markdown: {len(orphan_code_cells)}")
    if orphan_code_cells:
        for idx in orphan_code_cells:
            print(f"Orphan code cell at index {idx}")

if __name__ == '__main__':
    check_preceding_markdown('Lending_Club_Project_Final.ipynb')
