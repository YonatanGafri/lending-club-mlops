import json
import sys

def check_notebook(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            nb = json.load(f)
    except Exception as e:
        print(f"Error reading notebook: {e}")
        return

    cells = nb.get('cells', [])
    executed_code_cells = 0
    unexecuted_code_cells = 0
    image_outputs = []
    orphan_images = []

    for i, cell in enumerate(cells):
        if cell['cell_type'] == 'code':
            if cell.get('execution_count') is not None:
                executed_code_cells += 1
            else:
                unexecuted_code_cells += 1
            
            has_image = False
            for output in cell.get('outputs', []):
                if 'data' in output and 'image/png' in output['data']:
                    has_image = True
                    break
            
            if has_image:
                image_outputs.append(i)
                # check next cell
                if i + 1 < len(cells):
                    next_cell = cells[i+1]
                    if next_cell['cell_type'] != 'markdown':
                        orphan_images.append(i)
                else:
                    orphan_images.append(i)

    print(f"Total Code Cells: {executed_code_cells + unexecuted_code_cells}")
    print(f"Executed: {executed_code_cells}, Unexecuted: {unexecuted_code_cells}")
    print(f"Cells with images: {len(image_outputs)}")
    print(f"Orphan images (not followed by markdown): {len(orphan_images)}")
    if orphan_images:
        for idx in orphan_images:
            print(f"Orphan graph at cell index {idx}")

if __name__ == '__main__':
    check_notebook('Lending_Club_Project_Final.ipynb')
