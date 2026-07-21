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
        
        # 1. Fix Leakage Column Count
        if "Removed 63 columns total" in source:
            source = source.replace("Removed 63 columns total", "Removed 79 columns total")
            print("Fixed leakage column count.")
            
        # 2. Fix Temporal Drift Percentages
        if "Default rates doubled from 12.6% to 23.1% over the training period" in source:
            source = source.replace(
                "Default rates doubled from 12.6% to 23.1% over the training period",
                "Default rates increased significantly over the training period"
            )
            print("Fixed temporal drift percentages.")
            
        # 3. Fix SHAP Insight text
        if "ranked **#3** in SHAP\n   importance" in source or "ranked **#3** in SHAP\n   importance " in source:
            # Handle unicode replacement character if it exists
            source = re.sub(
                r'ranked \*\*#3\*\* in SHAP\n\s*importance.*?proving', 
                'ranked highly in SHAP importance, proving', 
                source
            )
            print("Fixed SHAP insight #2.")
            
        # 4. Fix SHAP Table and Contradiction
        if "### Top 5 Predictors (SHAP)" in source:
            new_table = """### Top Predictors (SHAP)
| Rank | Feature | Interpretation |
|:---:|---|---|
| 1 | `fico_range_low` | Lower FICO = higher risk |
| 2 | `adjusted_dti` | **Our domain feature!** Higher DTI = riskier |
| 3 | `sub_grade_num` | LC's internal rating (higher = riskier) |
| 4 | `term_num` | 60-month loans default more |
| 5 | `avg_cur_bal` | Lower balances = less stability |"""
            
            # Use regex to replace the old table
            source = re.sub(
                r'### Top 5 Predictors \(SHAP\)\n\| Rank \| Feature .*?fico_range_low` \| 0\.135 \| Lower FICO = higher risk \|',
                new_table,
                source,
                flags=re.DOTALL
            )
            print("Fixed SHAP table.")
            
        c.source = source
        
    with open(out_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
        
    print(f"Successfully processed {out_path}")

if __name__ == "__main__":
    main()
