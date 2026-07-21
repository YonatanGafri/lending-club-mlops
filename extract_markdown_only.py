import json
def extract_md(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    with open('markdown_only.txt', 'w', encoding='utf-8') as out:
        for i, cell in enumerate(nb.get('cells', [])):
            if cell.get('cell_type') == 'markdown':
                out.write(f"--- CELL {i} ---\n")
                out.write("".join(cell.get('source', [])))
                out.write("\n\n")

if __name__ == "__main__":
    extract_md('Lending_Club_Project_Final.ipynb')
