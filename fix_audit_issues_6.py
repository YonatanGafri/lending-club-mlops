import nbformat
import re

def main():
    nb_path = 'Lending_Club_Project_Final.ipynb'
    out_path = 'Lending_Club_Project_Final.ipynb'
    
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
        
    for c in nb.cells:
        if c.cell_type != 'markdown':
            continue
            
        source = "".join(c.source)
        
        if "highly generalizable risk engine" in source:
            new_source = re.sub(
                r'5\. \*\*Model Robustness:\*\* By creating a \*\*Soft Voting Ensemble\*\* \(combining the linear stability of Logistic Regression with the non-linear power of LightGBM\), we produced a highly generalizable risk engine\.',
                '5. **Model Robustness:** We experimented with Soft Voting Ensembles and proved that forcing a linear baseline onto a non-linear tree model degraded top-line performance, reinforcing our decision to deploy LightGBM as a standalone engine.',
                source
            )
            
            if new_source != source:
                c.source = new_source
                print("Fixed the ensemble narrative dissonance!")
            else:
                print("Regex failed to match the narrative dissonance.")
                
    with open(out_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
        
    print(f"Successfully processed {out_path}")

if __name__ == "__main__":
    main()
