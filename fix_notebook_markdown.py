import json

def main():
    nb_path = 'Lending_Club_Project_Final.ipynb'
    out_path = 'Lending_Club_Project_Final.ipynb'
    
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for c in nb['cells']:
        if c['cell_type'] == 'markdown':
            for i, line in enumerate(c['source']):
                if "The **Silhouette Analysis** confirms that K=3 achieves the best" in line:
                    c['source'][i] = line.replace("K=3", "K=2")
                    print("Fixed K-Means markdown")
                    
                if "- **Temporal Drift**: Default rate **doubled** from 12.6% (2009) to 23.1% (2016)" in line:
                    c['source'][i] = line.replace("- **Temporal Drift**: Default rate **doubled** from 12.6% (2009) to 23.1% (2016)", "")
                    print("Fixed Temporal Drift markdown")
                    
                if "Net Expected Profit by **$125,000,000**" in line:
                    c['source'][i] = line.replace("Net Expected Profit by **$125,000,000**", "a Net Expected Profit of **$125,000,000**")
                    print("Fixed Profit markdown 1")
                    
                if "increased Net Expected Profit by $125,000,000" in line:
                    c['source'][i] = line.replace("increased Net Expected Profit by $125,000,000", "yields a Net Expected Profit of $125,000,000")
                    print("Fixed Profit markdown 2")

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
        
    print(f"Successfully processed {out_path}")

if __name__ == "__main__":
    main()
