import json

def extract_markdown():
    with open('Lending_Club_Project_Final.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)

    md_cells = []
    for cell in nb.get('cells', []):
        if cell['cell_type'] == 'markdown':
            md_cells.append("".join(cell['source']))
            md_cells.append("\n\n---\n\n")

    with open('final_notebook_text.cells.md', 'w', encoding='utf-8') as f:
        f.writelines(md_cells)
        
if __name__ == '__main__':
    extract_markdown()
