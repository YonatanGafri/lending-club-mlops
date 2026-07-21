import nbformat

def main():
    nb_path = 'Lending_Club_Project_Final.ipynb'
    out_path = 'Lending_Club_Project_Final.ipynb'
    
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
        
    for c in nb.cells:
        if c.cell_type == 'markdown':
            source = "".join(c.source)
            
            # 1. Temporal Drift
            if "Default rate doubled from 12.6% (2009) to 23.1% (2016)" in source:
                source = source.replace("Default rate doubled from 12.6% (2009) to 23.1% (2016)", "Default rate increased significantly over the observed period")
                
            # 2. Structural Missingness
            if "We proved that Structural Missingness (e.g., missing employment length) is not noise, but a critical risk signal." in source:
                source = source.replace("We proved that Structural Missingness (e.g., missing employment length) is not noise, but a critical risk signal.", "Structural Missingness (e.g., missing employment length) served as a critical risk signal.")
                
            # 3. K-Means Clustering
            if "The Silhouette Analysis confirms that K=3 achieves the best cohesion-to-separation ratio." in source:
                source = source.replace("The Silhouette Analysis confirms that K=3 achieves the best cohesion-to-separation ratio.", "While Silhouette Analysis indicated K=2 was optimal, the Elbow Method supported K=3, which we selected for richer business interpretability.")
                
            # 4. Profit Optimization
            if "increased Net Expected Profit by $125,000,000" in source:
                source = source.replace("increased Net Expected Profit by $125,000,000", "achieved a total Net Expected Profit of $125,000,000")
                
            c.source = source
            
    with open(out_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
        
    print(f"Successfully processed {out_path}")

if __name__ == "__main__":
    main()
