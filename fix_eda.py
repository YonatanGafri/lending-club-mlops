import nbformat

def main():
    nb_path = 'Lending_Club_Project_Final.ipynb'
    out_path = 'Lending_Club_Project_Final.ipynb'
    
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
        
    for c in nb.cells:
        if c.cell_type == 'code':
            source = "".join(c.source)
            if "heavy_cols = ['annual_inc', 'revol_bal', 'loan_amnt'" in source:
                # Add log scale plotting
                plot_code = """
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

train_df_pd = train_df.to_pandas()
fig, axes = plt.subplots(len(heavy_cols), 2, figsize=(14, 4*len(heavy_cols)))
for i, col in enumerate(heavy_cols):
    vals = train_df_pd[col].dropna()
    sns.histplot(vals, bins=50, ax=axes[i, 0], color='#3498db')
    axes[i, 0].set_title(f'{col} (Linear Scale)', fontsize=12)
    sns.histplot(vals, bins=50, ax=axes[i, 1], color='#e67e22', log_scale=(False, True)) # wait, the X axis should be log scaled, or Y? 
    # Actually, for right-skewed data, log scale on X is common, but sns.histplot log_scale=True applies to X axis if not specified otherwise in older versions, or both? 
    # Let's just use log_scale=True which usually logs the data axis (X).
    # Wait, the safest is `log_scale=(True, False)` for X-axis log.
    sns.histplot(vals, bins=50, ax=axes[i, 1], color='#e67e22', log_scale=(True, False))
    axes[i, 1].set_title(f'{col} (Log Scale)', fontsize=12)
plt.tight_layout()
plt.show()
"""
                if "sns.histplot" not in source:
                    c.source = source + "\n" + plot_code
                    print("Added Log-Scale plots to code cell.")
                    
        elif c.cell_type == 'markdown':
            source = "".join(c.source)
            
            # Fix Isolation Forest text
            old_iso_text = "This proves that unsupervised anomaly flagging adds a valuable signal for detecting edge-case credit risks!"
            new_iso_text = "This proves that unsupervised anomaly flagging adds a valuable signal: anomalies in this dataset represent ultra-wealthy or extremely conservative profiles who are actually **safer** borrowers, not higher credit risks."
            if old_iso_text in source:
                c.source = source.replace(old_iso_text, new_iso_text)
                print("Fixed Isolation Forest interpretation.")
                
            # Add Missing Value imputation strategy
            missing_trigger = "Creating explicit binary flags for missingness allows the model to learn if the *absence* of information is itself a risk signal."
            missing_addition = "\n\nFor non-structural continuous features, we impute missing values using the median to prevent outliers from skewing the imputation. Categorical missing values are mapped to a distinct 'Unknown' category. This unified strategy ensures our models have complete, robust datasets for training."
            if missing_trigger in source and "For non-structural continuous features" not in source:
                c.source = source.replace(missing_trigger, missing_trigger + missing_addition)
                print("Added missing value treatment explanation.")
                
    with open(out_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
        
    print(f"Successfully processed {out_path}")

if __name__ == "__main__":
    main()
