import json
import ast
import glob
import re

def extract_imports(notebook_path):
    all_imports = set()
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
    except Exception as e:
        print(f"Error reading {notebook_path}: {e}")
        return set()
        
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            # Remove magics that break AST
            source_lines = cell.get('source', [])
            clean_lines = []
            for line in source_lines:
                # Basic cleaning of magics
                if not line.strip().startswith('%') and not line.strip().startswith('!'):
                    clean_lines.append(line)
            
            source_code = "".join(clean_lines)
            
            try:
                tree = ast.parse(source_code)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for n in node.names:
                            asname = f" as {n.asname}" if n.asname else ""
                            all_imports.add(f"import {n.name}{asname}")
                    elif isinstance(node, ast.ImportFrom):
                        module = node.module if node.module else ""
                        for n in node.names:
                            asname = f" as {n.asname}" if n.asname else ""
                            all_imports.add(f"from {module} import {n.name}{asname}")
            except SyntaxError:
                # Fallback to regex if AST fails (e.g., due to remaining magic or syntax errors)
                for line in clean_lines:
                    line = line.strip()
                    if line.startswith('import ') or line.startswith('from '):
                        all_imports.add(line)
    return all_imports

notebooks = glob.glob("c:/Users/yonat/.gemini/antigravity/scratch/lending_club_ds/**/*.ipynb", recursive=True)
print("Scanning notebooks...")
for nb_path in notebooks:
    if '.ipynb_checkpoints' in nb_path: continue
    imports = extract_imports(nb_path)
    if imports:
        print(f"\n--- Imports from {nb_path} ---")
        for imp in sorted(imports):
            print(imp)
