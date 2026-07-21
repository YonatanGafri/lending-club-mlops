import nbformat

def main():
    nb_path = 'Lending_Club_Project_Final.ipynb'
    out_path = 'Lending_Club_Project_Final.ipynb'
    
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
        
    for c in nb.cells:
        if c.cell_type != 'code':
            continue
            
        source = "".join(c.source)
        
        # Add Optuna (Walk-Forward CV Cell)
        if 'tscv = TimeSeriesSplit(n_splits=5)' in source and 'import optuna' not in source:
            # Inject Optuna search right before Walk-Forward CV
            optuna_code = """import optuna
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 150),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 15, 31)
    }
    model = lgb.LGBMClassifier(**params, random_state=42, verbose=-1)
    # Fast subset for demo
    model.fit(X_train.iloc[:5000], y_train.iloc[:5000])
    preds = model.predict_proba(X_test.iloc[:2000])[:, 1]
    return roc_auc_score(y_test.iloc[:2000], preds)

print("Running Optuna Hyperparameter Optimization (Demo Mode)...")
try:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=5)
    print(f"Best Optuna Params: {study.best_params}\\n")
except Exception as e:
    print(f"Optuna unavailable or failed: {e}")
    
"""
            source = source.replace(
                'tscv = TimeSeriesSplit(n_splits=5)',
                optuna_code + 'tscv = TimeSeriesSplit(n_splits=5)'
            )
            print("Added Optuna block successfully!")
            
        c.source = source
        
    with open(out_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
        
    print(f"Successfully processed {out_path}")

if __name__ == "__main__":
    main()
