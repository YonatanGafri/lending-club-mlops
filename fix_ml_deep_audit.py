import nbformat
import re

def main():
    nb_path = 'Lending_Club_Project_Final.ipynb'
    out_path = 'Lending_Club_Project_Final.ipynb'
    
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
        
    for c in nb.cells:
        if c.cell_type == 'code':
            source = "".join(c.source)
            
            # 1. Eliminate Proxy Target Leakage
            features_old = 'features = [\n    "loan_amnt", "int_rate", "annual_inc", "dti", "revol_util", \n    "fico_range_low", "adjusted_dti", "installment_to_income", \n    "loan_to_income", "estimated_revol_limit", "emp_length_num", \n    "term_num", "sub_grade_num"\n]'
            features_new = 'features = [\n    "loan_amnt", "annual_inc", "dti", "revol_util", \n    "fico_range_low", "adjusted_dti", "installment_to_income", \n    "loan_to_income", "estimated_revol_limit", "emp_length_num", \n    "term_num"\n]'
            if features_old in source:
                source = source.replace(features_old, features_new)
                print("Removed proxy target leakage (int_rate, sub_grade_num) from features.")
                
            # 2. Fix Optuna Hyperparameter Leakage
            optuna_leak = 'preds = model.predict_proba(X_test.iloc[:2000])[:, 1]\n    return roc_auc_score(y_test.iloc[:2000], preds)'
            optuna_fix = 'preds = model.predict_proba(X_train.iloc[5000:7000])[:, 1]\n    return roc_auc_score(y_train.iloc[5000:7000], preds)'
            if optuna_leak in source:
                source = source.replace(optuna_leak, optuna_fix)
                print("Fixed Optuna hyperparameter leakage on X_test.")
                
            # 3. Calibrate and Weight the Ensemble
            ensemble_old = "ensemble = VotingClassifier(estimators=[('lr', lr), ('lgbm', lgbm)], voting='soft')"
            ensemble_new = "from sklearn.calibration import CalibratedClassifierCV\ncalibrated_lgbm = CalibratedClassifierCV(lgbm, method='isotonic', cv=3)\nensemble = VotingClassifier(estimators=[('lr', lr), ('lgbm', calibrated_lgbm)], voting='soft', weights=[0.2, 0.8])"
            if ensemble_old in source:
                source = source.replace(ensemble_old, ensemble_new)
                print("Added Isotonic Calibration and Weights to the Ensemble.")
                
            c.source = source
            
        elif c.cell_type == 'markdown':
            source = "".join(c.source)
            
            # 4. Address Survivorship Bias
            survivorship_trigger = "Filter to terminal statuses only"
            survivorship_addition = "\n\n> [!WARNING]\n> **Survivorship Bias / Right-Censoring Acknowledged:** By filtering to terminal statuses (`Fully Paid`, `Charged Off`) *before* the chronological split, we are introducing a known survivorship bias. Recent loans in the OOT test set that haven't defaulted yet (but eventually will) were filtered out because they were still 'Current'. This means our test set is artificially biased toward loans that resolved quickly. In a true production environment, we would use survival analysis (e.g., Cox Proportional Hazards) or predict default within a fixed time window (e.g., 12 months) to avoid right-censoring."
            if "Out-of-Time Split (chronological 80/20)" in source and "Survivorship Bias" not in source:
                c.source = source.replace("Out-of-Time Split (chronological 80/20)", "Out-of-Time Split (chronological 80/20)" + survivorship_addition)
                print("Added Survivorship Bias caveat to markdown.")
                
    with open(out_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
        
    print(f"Successfully processed {out_path}")

if __name__ == "__main__":
    main()
