import nbformat

def main():
    nb_path = 'Lending_Club_Project_Final.ipynb'
    out_path = 'Lending_Club_Project_Final.ipynb'
    
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
        
    cells = nb.cells
    new_cells = []
    
    for c in cells:
        source = "".join(c.source)
        
        # Exact match for the duplicate cell
        if source.strip() == "### 4.1 Target Distribution & Temporal Drift":
            print("DELETED duplicate 4.1 header!")
            continue # skip adding to new_cells
            
        new_cells.append(c)
        
    nb.cells = new_cells
    
    with open(out_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
        
    print(f"Successfully processed {out_path}")

if __name__ == "__main__":
    main()
