import nbformat

def create_markdown_cell(source):
    return nbformat.v4.new_markdown_cell(source)

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
        if "### 4.1 Target Distribution & Temporal Drift" in source and len(source) < 50:
            if not found_41:
                found_41 = True
            else:
                print("Skipped duplicate 4.1 header")
                continue
                
        # 2. Fix the Threshold text in Section 7
        if "3. **Threshold Optimization**" in source and "0.59" in source:
            c.source = source.replace(
                "3. **Threshold Optimization**: Moving from default 0.5 to optimal 0.59 threshold increased\n   Net Expected Profit by **$92M** (+18%) while maintaining 74.4% approval rate.",
                "3. **Threshold Optimization**: Moving from default 0.5 to optimal 0.18 threshold increased\n   Net Expected Profit by **$125,000,000** while maintaining an 82.5% approval rate."
            )
            print("Fixed threshold insight.")
            
        # 3. Fix the Calibration text in Section 7
        if "4. **Calibration**" in source and "Isotonic" in source:
            c.source = source.replace(
                "4. **Calibration**: Isotonic Regression improved Brier Score by **21.1%**, making probability\n   estimates reliable for business decisions.",
                "4. **Calibration**: Probability calibration effectively aligns the predicted scores with actual default rates, reducing the Brier Score to 0.1450 and making probability estimates reliable for business decisions."
            )
            print("Fixed calibration insight.")
            
        new_cells.append(c)
        
        # 4. Insert Model Comparison Insight
        if "Model Comparison: ROC & PR Curves" in source and c.cell_type == 'code':
            new_cells.append(create_markdown_cell("**Model Comparison Insight:** The ROC curves clearly demonstrate LightGBM's superiority (0.745 AUC) over the Logistic Regression baseline (0.712 AUC) across all decision thresholds."))
            print("Injected Model Comparison insight.")

    nb.cells = new_cells
    
    with open(out_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
        
    print(f"Successfully processed {out_path}")

if __name__ == "__main__":
    main()
