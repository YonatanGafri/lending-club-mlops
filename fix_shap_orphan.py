import nbformat

def create_markdown_cell(source):
    return nbformat.v4.new_markdown_cell(source)

def main():
    nb_path = 'Lending_Club_Project_Final.ipynb'
    
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
        
    cells = nb.cells
    new_cells = []
    
    md_between_shap = "**SHAP Beeswarm Insight:** The beeswarm plot above reveals the *directionality* of impact: for example, lower FICO scores (blue dots) strongly push the prediction toward higher risk (positive SHAP values). Now, let's look at the average absolute impact of each feature to see the overall importance ranking."
    
    for c in cells:
        new_cells.append(c)
        source = c.source
        
        # Look for the first SHAP cell, inject after it
        if "SHAP Feature Importance" in source and "(Bar)" not in source and "print(" in source:
            new_cells.append(create_markdown_cell(md_between_shap))
            print("Injected text between SHAP plots.")

    nb.cells = new_cells
    
    with open(nb_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
        
    print(f"Successfully processed {nb_path}")

if __name__ == "__main__":
    main()
