import nbformat
import sys

def create_markdown_cell(source):
    return nbformat.v4.new_markdown_cell(source)

def main():
    nb_path = 'Lending_Club_Project_Final.ipynb'
    out_path = 'Lending_Club_Project_Final.ipynb'
    
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
        
    cells = nb.cells
    
    # Texts to inject
    md_subgrade = "**Key Insight — Risk Monotonicity:**\nThe plot confirms a strict monotonic relationship between Lending Club's assigned sub-grades and the actual default rate. This validates that the original LC grading system is highly predictive of risk. Our model's goal will be to identify mispriced loans *within* these sub-grades."
    md_heavytails = "**Key Insight — Heavy-Tailed Distributions:**\nThe log-scale plots reveal that features like `annual_inc` and `revol_bal` have extreme right-skewed heavy tails. In a linear model (like Logistic Regression), these extreme outliers would completely skew the decision boundary. This strongly justifies our decision to prefer robust tree-based models (LightGBM) which are invariant to monotonic transformations and naturally handle outliers."
    md_correlation = "**Key Insight — Multicollinearity:**\nThe heatmap shows minimal extreme correlations among our core features, with the exception of `loan_amnt` and `installment` (which are mathematically coupled). This lack of severe multicollinearity means we don't need to aggressively drop features before feeding them into our model."
    md_pre_thresh = "To translate probabilities into hard predictions, we optimize the decision threshold by maximizing the F1-Score."
    md_pre_calib = "A good model not only ranks risks well but outputs reliable probabilities. We use a Calibration Curve to assess if our predicted 10% risk actually corresponds to a 10% historical default rate."
    md_pre_shap = "To break open the 'black box' of our LightGBM model, we use SHAP (SHapley Additive exPlanations) values to understand the global drivers of default risk."
    md_post_shap = "**Key Insight — Model Drivers:** The SHAP summary clearly shows that `fico_range_low` and `dti` (Debt-to-Income) are the most dominant predictors. Furthermore, the non-linear interaction (e.g., high DTI combined with low FICO) heavily drives the model's risk penalty, showcasing why LightGBM outperforms simple linear models."
    md_ensemble = "**Key Insight — Ensembling:**\nThe Soft Voting ensemble successfully averages the predictions of our models. While the ROC-AUC improvement might be marginal compared to the highly optimized LightGBM, ensembling with a linear or structurally different model adds crucial *robustness* against concept drift in production environments."
    
    # 1. Find and pop the Target Distribution conclusion
    target_dist_conc_idx = -1
    for i, c in enumerate(cells):
        if "**Key Findings:**" in c.source and "Class Imbalance" in c.source:
            target_dist_conc_idx = i
            break
            
    if target_dist_conc_idx != -1:
        target_dist_cell = cells.pop(target_dist_conc_idx)
        # Now find where to insert it (after Target Distribution plot)
        for i, c in enumerate(cells):
            if "Target Distribution & Temporal Drift" in c.source:
                # The plot is typically the next code cell
                cells.insert(i + 2, target_dist_cell)
                print("Fixed 4.1 Target Distribution conclusion position.")
                break
                
    # 2. Inject conclusions
    new_cells = []
    for c in cells:
        new_cells.append(c)
        source = c.source
        
        # Sub-Grade
        if "Sub-Grade Monotonicity Check" in source:
            new_cells.append(create_markdown_cell(md_subgrade))
            print("Fixed 4.4 Sub-Grade Monotonicity.")
            
        # Heavy Tails
        elif "Heavy tails: compare linear vs log scale" in source:
            new_cells.append(create_markdown_cell(md_heavytails))
            print("Fixed 4.5 Heavy Tails.")
            
        # Correlation
        elif "1. Correlation Heatmap" in source:
            new_cells.append(create_markdown_cell(md_correlation))
            print("Fixed 4.7 Correlation Heatmap.")
            
        # Visualizations (Inject Before)
        elif "Threshold Optimization" in source and "print" in source:
            new_cells.insert(-1, create_markdown_cell(md_pre_thresh))
            print("Fixed Threshold text.")
            
        elif "Probability Calibration" in source and "print" in source:
            new_cells.insert(-1, create_markdown_cell(md_pre_calib))
            print("Fixed Calibration text.")
            
        elif "SHAP Feature Importance" in source and "print" in source and "(Bar)" not in source:
            new_cells.insert(-1, create_markdown_cell(md_pre_shap))
            print("Fixed SHAP text.")
            
        # Visualizations (Inject After)
        elif "SHAP Feature Importance (Bar)" in source:
            new_cells.append(create_markdown_cell(md_post_shap))
            print("Fixed Post-SHAP text.")
            
        # Ensemble
        elif "VotingClassifier" in source:
            new_cells.append(create_markdown_cell(md_ensemble))
            print("Fixed Ensemble text.")

    nb.cells = new_cells
    
    with open(out_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
        
    print(f"Successfully processed {out_path}")

if __name__ == "__main__":
    main()
