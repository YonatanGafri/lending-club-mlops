import nbformat

def main():
    nb_path = 'Lending_Club_Project_Final.ipynb'
    out_path = 'Lending_Club_Project_Final.ipynb'
    
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
        
    for c in nb.cells:
        if c.cell_type == 'markdown':
            source = "".join(c.source)
            
            # 1. Term Length Fix
            old_term_text = "1. **Term Length:** 60-month loans have a significantly higher default rate. The longer exposure time (more chances for life events like job loss) outweighs the benefit of smaller monthly payments."
            new_term_text = "1. **Term Length:** 60-month loans have a significantly higher default rate. This may be due to longer exposure time (more chances for life events like job loss), or it may reflect that borrowers requiring longer terms to afford payments are inherently riskier at origination."
            if old_term_text in source:
                c.source = source.replace(old_term_text, new_term_text)
                print("Fixed Term Length insight.")
                
            # 2. Median Imputation Fix
            old_pca_text = "We will apply **Principal Component Analysis (PCA)** for dimensionality reduction and **K-Means Clustering** to find these personas using key financial indicators."
            new_pca_text = "We will apply **Principal Component Analysis (PCA)** for dimensionality reduction and **K-Means Clustering** to find these personas using key financial indicators.\n\nBecause our financial features are heavily skewed (as seen in 4.5), we use median imputation for the remaining missing values prior to clustering, as the median is robust to extreme outliers."
            if old_pca_text in source and "median imputation for the remaining missing values" not in source:
                c.source = source.replace(old_pca_text, new_pca_text)
                print("Added median imputation explanation.")
                
    with open(out_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
        
    print(f"Successfully processed {out_path}")

if __name__ == "__main__":
    main()
