import nbformat
import os

def main():
    nb_path = 'Lending_Club_Project_Final.ipynb'
    out_path = 'Lending_Club_Project_Final.ipynb'
    
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
        
    for c in nb.cells:
        if c.cell_type == 'code':
            source = "".join(c.source)
            
            # 1. Update Final LightGBM to use study.best_params
            old_lgb = "final_lgbm = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1)"
            new_lgb = """best_params = study.best_params if 'study' in globals() else {'n_estimators': 300, 'learning_rate': 0.05, 'num_leaves': 31}
final_lgbm = lgb.LGBMClassifier(**best_params, random_state=42, verbose=-1)"""
            if old_lgb in source:
                source = source.replace(old_lgb, new_lgb)
                
            # 2. Update CV LightGBM to use study.best_params
            old_cv_lgb = "fold_model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1)"
            new_cv_lgb = """cv_params = study.best_params if 'study' in globals() else {'n_estimators': 200, 'learning_rate': 0.05, 'num_leaves': 31}
    fold_model = lgb.LGBMClassifier(**cv_params, random_state=42, verbose=-1)"""
            if old_cv_lgb in source:
                source = source.replace(old_cv_lgb, new_cv_lgb)
                
            # 3. Update MLP Imputation to use pd.get_dummies
            old_mlp1 = "cat_cols = ['purpose', 'home_ownership', 'verification_status']\nX_train_nn = X_train.drop(columns=[c for c in cat_cols if c in X_train.columns]).fillna(X_train.drop(columns=[c for c in cat_cols if c in X_train.columns]).median())"
            old_mlp2 = "X_test_nn = X_test.drop(columns=[c for c in cat_cols if c in X_test.columns]).fillna(X_train.drop(columns=[c for c in cat_cols if c in X_train.columns]).median())"
            
            new_mlp_block = """cat_cols = ['purpose', 'home_ownership', 'verification_status']
import pandas as pd
X_train_nn = pd.get_dummies(X_train, columns=[c for c in cat_cols if c in X_train.columns])
X_test_nn = pd.get_dummies(X_test, columns=[c for c in cat_cols if c in X_test.columns])
X_train_nn, X_test_nn = X_train_nn.align(X_test_nn, join='left', axis=1, fill_value=0)
train_median = X_train_nn.median()
X_train_nn = X_train_nn.fillna(train_median)
X_test_nn = X_test_nn.fillna(train_median)"""
            
            if "cat_cols = ['purpose'" in source and "X_train_nn =" in source:
                # Instead of strict string replace, just replace the whole MLP data prep lines
                import re
                source = re.sub(r"cat_cols = \['purpose'.*?median\(\)\)", new_mlp_block, source, flags=re.DOTALL)
                
            c.source = source
            
    with open(out_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
        
    print(f"Successfully processed {out_path}")
    
    # 4. Delete ensemble_code.txt
    if os.path.exists('ensemble_code.txt'):
        os.remove('ensemble_code.txt')
        print("Deleted ensemble_code.txt")

if __name__ == "__main__":
    main()
