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
            
            # 1 & 2 & 3: Fix features and NaN handling
            features_block = """features = [
    "loan_amnt", "annual_inc", "dti", "revol_util", 
    "fico_range_low", "adjusted_dti", "installment_to_income", 
    "loan_to_income", "estimated_revol_limit", "emp_length_num", 
    "term_num"
]"""
            features_block_new = """features = [
    "loan_amnt", "annual_inc", "dti", "revol_util", 
    "fico_range_low", "adjusted_dti", "installment_to_income", 
    "loan_to_income", "estimated_revol_limit", "emp_length_num", 
    "term_num", "purpose", "home_ownership", "verification_status"
]"""
            
            if features_block in source:
                source = source.replace(features_block, features_block_new)
                
            if "X_train = train_df.select(features).to_pandas().fillna(0)" in source:
                source = source.replace("X_train = train_df.select(features).to_pandas().fillna(0)", "X_train = train_df.select(features).to_pandas()")
            if "X_test = test_df.select(features).to_pandas().fillna(0)" in source:
                source = source.replace("X_test = test_df.select(features).to_pandas().fillna(0)", "X_test = test_df.select(features).to_pandas()")
                
            if "X_test = test_df.select(features).to_pandas()" in source and "category" not in source:
                cat_cast = """
cat_cols = ["purpose", "home_ownership", "verification_status"]
for col in cat_cols:
    if col in X_train.columns:
        X_train[col] = X_train[col].astype('category')
        X_test[col] = X_test[col].astype('category')
"""
                source = source.replace('y_test = test_df.select("target").to_pandas()["target"]', 'y_test = test_df.select("target").to_pandas()["target"]\n' + cat_cast)

            # Optuna Fix
            optuna_leak = """    model.fit(X_train.iloc[:5000], y_train.iloc[:5000])
    preds = model.predict_proba(X_train.iloc[5000:7000])[:, 1]
    return roc_auc_score(y_train.iloc[5000:7000], preds)"""
            
            optuna_fix = """    # 80/20 chronological split of the ENTIRE X_train for Optuna
    split_idx = int(len(X_train) * 0.8)
    X_tr, y_tr = X_train.iloc[:split_idx], y_train.iloc[:split_idx]
    X_va, y_va = X_train.iloc[split_idx:], y_train.iloc[split_idx:]
    
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(50, verbose=False)])
    preds = model.predict_proba(X_va)[:, 1]
    return roc_auc_score(y_va, preds)"""
            if optuna_leak in source:
                source = source.replace(optuna_leak, optuna_fix)
                
            # CV Early stopping
            cv_train = "fold_model.fit(X_fold_train, y_fold_train)"
            cv_train_new = "fold_model.fit(X_fold_train, y_fold_train, eval_set=[(X_fold_val, y_fold_val)], callbacks=[lgb.early_stopping(30, verbose=False)])"
            if cv_train in source:
                source = source.replace(cv_train, cv_train_new)
                
            # MLP NaN Fix
            mlp_nan = "X_train_nn = X_train.fillna(X_train.median())"
            mlp_nan_new = "cat_cols = ['purpose', 'home_ownership', 'verification_status']\nX_train_nn = X_train.drop(columns=[c for c in cat_cols if c in X_train.columns]).fillna(X_train.drop(columns=[c for c in cat_cols if c in X_train.columns]).median())"
            if mlp_nan in source:
                source = source.replace(mlp_nan, mlp_nan_new)
            mlp_nan2 = "X_test_nn = X_test.fillna(X_train.median())"
            mlp_nan2_new = "X_test_nn = X_test.drop(columns=[c for c in cat_cols if c in X_test.columns]).fillna(X_train.drop(columns=[c for c in cat_cols if c in X_train.columns]).median())"
            if mlp_nan2 in source:
                source = source.replace(mlp_nan2, mlp_nan2_new)
                
            # Remove Ensemble
            if "from sklearn.ensemble import VotingClassifier" in source and "ensemble.fit" in source:
                ensemble_replacement = """# ==========================================
# FINAL EVALUATION: STANDALONE LIGHTGBM
# ==========================================
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

print("Training Final LightGBM Model on Full Train Set (with native categoricals and NaNs)...")
# We use robust params.
final_lgbm = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1)
final_lgbm.fit(X_train, y_train)

lgbm_preds = final_lgbm.predict_proba(X_test)[:, 1]
final_auc = roc_auc_score(y_test, lgbm_preds)
print(f"Final LightGBM ROC-AUC on Held-out OOT Test Set: {final_auc:.4f}")
"""
                # Replace the whole cell source
                source = ensemble_replacement
                
            # Change ensemble_preds to lgbm_preds in False Negatives cell
            if "test_results['Predicted_Prob'] = ensemble_preds" in source:
                source = source.replace("test_results['Predicted_Prob'] = ensemble_preds", "test_results['Predicted_Prob'] = lgbm_preds")
                
            c.source = source
            
    with open(out_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
        
    print(f"Successfully processed {out_path}")

if __name__ == "__main__":
    main()
