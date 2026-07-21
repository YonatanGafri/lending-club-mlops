import nbformat
import re

def main():
    nb_path = 'Lending_Club_Project_Final.ipynb'
    out_path = 'Lending_Club_Project_Final.ipynb'
    
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
        
    cells = nb.cells
    new_cells = []
    
    found_41 = False
    
    for i, c in enumerate(cells):
        source = "".join(c.source)
        
        # 1. Remove duplicate 4.1
        if "### 4.1 Target Distribution & Temporal Drift" in source and len(source.strip()) < 50:
            if not found_41:
                found_41 = True
            else:
                print("Skipped duplicate 4.1 header")
                continue
                
        # 2. Regex replace for Threshold text
        if "3. **Threshold Optimization**" in source:
            new_source = re.sub(
                r'Moving from default 0\.5 to optimal 0\.59 threshold increased\s+Net Expected Profit by \*\*\$92M\*\* \(\+18%\) while maintaining 74\.4% approval rate\.',
                'Moving from default 0.5 to optimal 0.18 threshold increased\n   Net Expected Profit by **$125,000,000** while maintaining an 82.5% approval rate.',
                source
            )
            if new_source != source:
                c.source = new_source
                print("Fixed threshold insight via regex.")
            else:
                print("Threshold regex failed to match.")
            
        new_cells.append(c)
        
    nb.cells = new_cells
    
    with open(out_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
        
    print(f"Successfully processed {out_path}")

if __name__ == "__main__":
    main()
