# Phase 4 Audit Report: Supervised Learning & Predictive Modeling

**Project:** Lending Club Data Science (`c:\Users\yonat\.gemini\antigravity\scratch\lending_club_ds`)  
**Auditor:** Investigator Agent 4 (Phase 4 Specialist)  
**Audit Date:** July 27, 2026  
**Reference Standards:** `ds_review_guide.md` Section 4 & `ORIGINAL_REQUEST.md` R4  

---

## Executive Summary

This audit delivers an exhaustive forensic evaluation of Phase 4 (Supervised Learning & Predictive Modeling) of the Lending Club Data Science project. The evaluation covers time-based validation strategies, hyperparameter optimization via Optuna, model-data alignment, and categorical feature handling across tree-based models (LightGBM), neural network baselines (MLP), and linear models (Logistic Regression).

### Overall Assessment Grade: **92 / 100** (Solid Pass / Advanced Industry Standard with minor demo-mode caveats)

| Audit Criterion | Grade | Status | Key Highlights |
|---|---|---|---|
| **1. Time-Based Validation Strategy** | 95/100 | PASS | Explicitly rejects random K-Fold CV; implements `TimeSeriesSplit` (Walk-Forward CV); plots fold stability in `model_plots/timeseries_cv.png`. |
| **2. Professional Hyperparameter Tuning** | 88/100 | PASS (Minor Gaps) | Uses Optuna with chronological 80/20 train/val split & early stopping; downgraded slightly due to demo mode (`n_trials=5`) and narrow search space. |
| **3. Model-Data Fit & Categorical Handling** | 94/100 | PASS | LightGBM receives native pandas `category` dtypes; MLP receives OHE + Imputed + Scaled features (`StandardScaler`); backed by NeurIPS 2022 research. |

---

## Detailed Audit Criteria Findings

---

### Criterion 1: Time-Based Validation Strategy (`TimeSeriesSplit` / Walk-Forward CV)

#### Requirement (`ds_review_guide.md` Section 4.1)
*בדאטה פיננסי כרונולוגי אסור להשתמש ב-K-Fold אקראי. חפש שימוש ב-TimeSeriesSplit המבטיח שהמודל תמיד מתאמן על העבר ונבחן על העתיד.*

#### 1. Code & Text Investigation
- **File:** `Lending_Club_Project_Final.py`
  - **Lines 169–185 (Section 3):** Initial Out-of-Time (OOT) Split.
    ```python
    df = df.sort('issue_date')
    total_rows = df.shape[0]
    train_size = int(total_rows * 0.8)
    train_df = df.head(train_size)
    test_df = df.tail(total_rows - train_size)
    ```
    *Observation:* Training period spans earlier issue dates while test period spans later dates, enforcing temporal separation. An embargo period filtering out loans issued $\ge 2016$ is also applied (line 177) to eliminate survivorship bias from un-matured 36-month loans.
  - **Lines 902–914 (Section 6.2.1 Markdown Text):** Methodological Justification.
    ```text
    However, our data is chronologically ordered. Using standard K-Fold CV would shuffle the rows, 
    causing the model to train on 2017 data and predict 2012 data — a clear temporal leakage...
    Solution: We use TimeSeriesSplit (Walk-Forward Validation)...
    ```
    *Observation:* Explains temporal data leakage caused by row shuffling in standard K-Fold CV.
  - **Lines 919 & 973–1006 (Section 6.2.1 Code):** Walk-Forward Cross-Validation Execution.
    ```python
    from sklearn.model_selection import TimeSeriesSplit
    ...
    tscv = TimeSeriesSplit(n_splits=5)
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        fold_model = lgb.LGBMClassifier(**cv_params, random_state=42, verbose=-1)
        fold_model.fit(X_fold_train, y_fold_train, eval_set=[(X_fold_val, y_fold_val)], 
                       callbacks=[lgb.early_stopping(30, verbose=False)])
        ...
    ```
    *Observation:* `TimeSeriesSplit(n_splits=5)` splits `X_train` into expanding temporal windows (Fold 1: train 20%, val 20%; Fold 2: train 40%, val 20%; ...). Each fold fits LightGBM using fold-specific early stopping on `X_fold_val`.
  - **Lines 1012–1028:** Plot Generation & Export.
    ```python
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(cv_df['fold'], cv_df['roc_auc'], ...)
    ax.axhline(y=mean_auc, color='blue', linestyle='--', label=f'CV Mean: {mean_auc:.4f}')
    ax.axhline(y=oot_auc, color='red', linestyle='--', label=f'OOT Test: {oot_auc:.4f}')
    ...
    plt.savefig('model_plots/timeseries_cv.png', dpi=150, bbox_inches='tight')
    ```
- **Output Plot Verification:**
  - `model_plots/timeseries_cv.png`: File exists (Size: ~45 KB). Displays bar chart across 5 folds with mean CV line and OOT test line.
- **Notebook Text Verification:**
  - `final_notebook_text.md` lines 304–326: Matches script markdown explanations verbatim.

#### 2. Pass / Success Points
- **Strict Prohibition of Random K-Fold:** The code and narrative explicitly avoid standard `KFold` or `StratifiedKFold` on the time-indexed dataset.
- **Walk-Forward CV (`TimeSeriesSplit`):** Implemented with `n_splits=5`. Evaluates model stability across expanding chronological subsets.
- **Fold-Level Early Stopping:** `lgb.early_stopping(30)` is specified inside the fold loop (`line 986`), preventing overfitting on early, small training folds.
- **Clear Visualization:** `model_plots/timeseries_cv.png` presents fold-level AUC, mean CV AUC, and test AUC side-by-side.

#### 3. Flaws / Implementation Gaps
- **Implicit Order Dependence:** Line 934 extracts `X_train` via `train_df.select(features).to_pandas()`. While `df` was sorted by `issue_date` in Section 3 (line 180), there is no explicit assertion (e.g., `assert train_df['issue_date'].is_sorted()`) before calling `tscv.split(X_train)`. If any intermediate transformation alters row order, `TimeSeriesSplit` would operate on unsorted indices.

---

### Criterion 2: Professional Hyperparameter Optimization (Optuna)

#### Requirement (`ds_review_guide.md` Section 4.2)
*חפש סריקת היפר-פרמטרים המכילה Early Stopping וחלוקה כרונולוגית (80/20) למניעת Overfitting.*

#### 1. Code & Text Investigation
- **File:** `Lending_Club_Project_Final.py`
  - **Lines 946–971 (Optuna Study):**
    ```python
    import optuna
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 150),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'num_leaves': trial.suggest_int('num_leaves', 15, 31)
        }
        model = lgb.LGBMClassifier(**params, random_state=42, verbose=-1)
        # 80/20 chronological split of the ENTIRE X_train for Optuna
        split_idx = int(len(X_train) * 0.8)
        X_tr, y_tr = X_train.iloc[:split_idx], y_train.iloc[:split_idx]
        X_va, y_va = X_train.iloc[split_idx:], y_train.iloc[split_idx:]
        
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(50, verbose=False)])
        preds = model.predict_proba(X_va)[:, 1]
        return roc_auc_score(y_va, preds)

    print("Running Optuna Hyperparameter Optimization (Demo Mode)...")
    try:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=5)
        print(f"Best Optuna Params: {study.best_params}\n")
    except Exception as e:
        print(f"Optuna unavailable or failed: {e}")
    ```
  - **Lines 1097–1108 (MLP Early Stopping):**
    ```python
    early_stop = EarlyStopping(monitor='val_auc', mode='max', patience=5, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_auc', mode='max', factor=0.5, patience=2, verbose=1)
    history = mlp_model.fit(
        X_train_nn_scaled, y_train,
        validation_split=0.2,
        epochs=15,
        batch_size=2048,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    ```
- **Output Plot Verification:**
  - `model_plots/mlp_training.png`: File exists (Size: ~67 KB). Plots MLP loss curve and AUC curve vs epoch, showing early stopping effect and comparison to LightGBM OOT AUC line.

#### 2. Pass / Success Points
- **Chronological Internal Split for Tuning:** The Optuna objective function explicitly constructs a chronological 80/20 split (`split_idx = int(len(X_train) * 0.8)`, lines 956–958) so the model trains on `X_tr` (earlier 80%) and evaluates on `X_va` (later 20%).
- **Early Stopping Integration:** LightGBM trials use `lgb.early_stopping(50)` during Optuna fitting (line 960). MLP fitting uses Keras `EarlyStopping(monitor='val_auc', patience=5)` and `ReduceLROnPlateau` (lines 1097–1098).
- **Target Metric Alignment:** Optuna optimizes `roc_auc_score(y_va, preds)`, directly targeting discrimination on imbalanced default labels.

#### 3. Flaws / Implementation Gaps
- **Demo Mode Restriction (`n_trials=5`):** Line 968 executes `study.optimize(objective, n_trials=5)` under a "Demo Mode" label. Evaluating only 5 trials does not thoroughly explore the search space.
- **Narrow Parameter Search Space:** `params` dictionary (lines 949–951) only searches `n_estimators` [50, 150], `learning_rate` [0.01, 0.1], and `num_leaves` [15, 31]. Critical LightGBM regularization hyperparameters (`subsample`, `colsample_bytree`, `min_child_samples`, `reg_alpha`, `reg_lambda`, `max_depth`) are omitted.
- **Inconsistent Parameter Fallback Values:** Line 984 uses default fallback `{'n_estimators': 200, 'learning_rate': 0.05, 'num_leaves': 31}`, whereas line 1190 uses `{'n_estimators': 300, ...}`.
- **Implicit Keras Validation Split:** Line 1103 uses `validation_split=0.2` in `mlp_model.fit()`. While Keras takes the last 20% of `X_train_nn_scaled` (matching time order), explicit array passing (`validation_data=(X_va, y_va)`) is preferred to prevent any epoch-level shuffling ambiguity.

---

### Criterion 3: Model-Data Fit & Categorical Handling

#### Requirement (`ds_review_guide.md` Section 4.3)
*ודא שמודל העצים (LightGBM) מקבל קטגוריות טבעיות (category dtype), בעוד שמודל הרשת (MLP) מקבל קטגוריות לאחר קידוד מלא (One-Hot Encoding).*

#### 1. Code & Text Investigation
- **File:** `Lending_Club_Project_Final.py`
  - **Lines 808–830 (Section 5.3 Categorical Feature Encoding):**
    - `emp_length` mapped to numeric ordinal `emp_length_num` (lines 809–817).
    - `term` mapped to numeric `term_num` (lines 820–822).
    - `sub_grade` mapped to `sub_grade_num` (lines 825–829).
  - **Lines 939–944 (LightGBM Categorical Casting):**
    ```python
    cat_cols = ["purpose", "home_ownership", "verification_status"]
    for col in cat_cols:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype('category')
            X_test[col] = X_test[col].astype('category')
    ```
    *Observation:* Categorical columns are converted to pandas `category` dtype for native LightGBM processing.
  - **Lines 1078–1092 (MLP Feature Preparation - OHE, Imputation & Scaling):**
    ```python
    cat_cols = ['purpose', 'home_ownership', 'verification_status']
    import pandas as pd
    X_train_nn = pd.get_dummies(X_train, columns=[c for c in cat_cols if c in X_train.columns])
    X_test_nn = pd.get_dummies(X_test, columns=[c for c in cat_cols if c in X_test.columns])
    X_train_nn, X_test_nn = X_train_nn.align(X_test_nn, join='left', axis=1, fill_value=0)

    train_median = X_train_nn.median()
    X_train_nn = X_train_nn.fillna(train_median)
    X_test_nn = X_test_nn.fillna(train_median)

    from sklearn.preprocessing import StandardScaler
    scaler_nn = StandardScaler()
    X_train_nn_scaled = scaler_nn.fit_transform(X_train_nn)
    X_test_nn_scaled = scaler_nn.transform(X_test_nn)
    ```
    *Observation:* Neural Network preprocessing converts categorical features via One-Hot Encoding (`pd.get_dummies`), aligns train/test columns, imputes missing values using train medians, and scales features via `StandardScaler`.
  - **Lines 1150–1173 (Theoretical Synthesis & Literature Comparison):**
    - Section titled `# ### Why Tree-Based Models Outperform Deep Learning on Tabular Data` contains a tabular comparison citing Grinsztajn et al. (NeurIPS 2022) detailing Inductive Bias, Feature Interactions, Robustness to Noise, Data Efficiency, and Hyperparameter Sensitivity.
- **Output Plot Verification:**
  - `model_plots/model_comparison.png`: File exists (Size: ~105 KB). Displays ROC & PR curves for LightGBM.
  - `model_plots/mlp_training.png`: File exists (Size: ~67 KB). Compares MLP performance against LightGBM baseline.
  - `model_plots/calibration_curves.png`: File exists (Size: ~82 KB). Shows probability calibration curves.
  - `model_plots/threshold_optimization.png`: File exists (Size: ~88 KB). Shows F1-score optimization across decision thresholds.
  - `model_plots/shap_beeswarm.png` & `model_plots/shap_importance.png`: Files exist (~145 KB & ~42 KB). SHAP global explanations.

#### 2. Pass / Success Points
- **Native Categoricals for LightGBM:** Nominal categoricals (`purpose`, `home_ownership`, `verification_status`) are explicitly cast to `category` dtype (lines 941–943), enabling LightGBM's native histogram-based optimal categorical splits.
- **Native Missingness for LightGBM:** Raw NaNs are preserved for LightGBM, avoiding artificial imputation noise.
- **Strict Preprocessing Pipeline for MLP:**
  - One-Hot Encoding via `pd.get_dummies` (line 1080).
  - Train/test column alignment using `.align(join='left', fill_value=0)` (line 1082) preventing test set feature shape mismatches.
  - Median imputation computed strictly on training data (`train_median = X_train_nn.median()`, lines 1083–1085).
  - Standard scaling via `StandardScaler()` fit on train and transformed on test (lines 1089–1092).
- **Academic Rigor:** Excellent theoretical justification citing NeurIPS 2022 research explaining why tree models outperform deep learning architectures on tabular datasets.

#### 3. Flaws / Implementation Gaps
- **Missing Linear Model (Logistic Regression) Benchmark:** Lines 1174–1199 discuss Logistic Regression in the context of an ensemble, but decide to deploy standalone LightGBM to avoid imputation. However, Logistic Regression is not included in the model evaluation comparison table (lines 1143–1147) or ROC/PR curves plot (`model_plots/model_comparison.png`), leaving the linear baseline unmeasured alongside MLP and LightGBM.
- **Target Proxy Exclusion Verification:** `sub_grade` is ordinally encoded in Section 5.3 (line 828), but correctly excluded from the modeling `features` list in lines 928–933 to avoid target proxy data leakage.

---

## Complete Artifact & Plot Index

| Plot Filename | Line Ref in Script | Description / Status |
|---|---|---|
| `model_plots/model_comparison.png` | Line 852 | ROC & PR curves for LightGBM model. (Verified Present) |
| `model_plots/threshold_optimization.png` | Line 863 | F1-Score threshold tuning for binary default classification. (Verified Present) |
| `model_plots/calibration_curves.png` | Line 876 | Reliability diagram comparing predicted probabilities vs actual default rates. (Verified Present) |
| `model_plots/shap_beeswarm.png` | Line 888 | SHAP beeswarm plot illustrating feature impact directionality. (Verified Present) |
| `model_plots/shap_importance.png` | Line 897 | SHAP mean absolute feature importance ranking. (Verified Present) |
| `model_plots/timeseries_cv.png` | Line 1027 | 5-fold Walk-Forward `TimeSeriesSplit` stability plot. (Verified Present) |
| `model_plots/mlp_training.png` | Line 1140 | MLP Loss and AUC training curves vs epoch, compared to LightGBM. (Verified Present) |

---

## Actionable Recommendations to Reach 100/100

To elevate the Phase 4 modeling grade from **92/100** to a perfect **100/100**, implement the following concrete code enhancements:

### Recommendation 1: Defensive Date Verification for `TimeSeriesSplit`
**Target File:** `Lending_Club_Project_Final.py` (Line 938)  
**Issue:** `X_train` is converted to pandas without explicitly enforcing index date order.  
**Code Action:**
```python
# Replace lines 934-937 with defensive chronological ordering check:
if 'issue_date' in train_df.columns:
    train_df = train_df.sort('issue_date')
    test_df = test_df.sort('issue_date')

X_train = train_df.select(features).to_pandas()
y_train = train_df.select("target").to_pandas()["target"]
X_test = test_df.select(features).to_pandas()
y_test = test_df.select("target").to_pandas()["target"]
```

### Recommendation 2: Expand Optuna Hyperparameter Space & Trial Count
**Target File:** `Lending_Club_Project_Final.py` (Lines 948–968)  
**Issue:** Trial count is capped at 5 and missing key tree regularization hyperparameters.  
**Code Action:**
```python
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 15, 127),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 300),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True)
    }
    model = lgb.LGBMClassifier(**params, random_state=42, verbose=-1)
    split_idx = int(len(X_train) * 0.8)
    X_tr, y_tr = X_train.iloc[:split_idx], y_train.iloc[:split_idx]
    X_va, y_va = X_train.iloc[split_idx:], y_train.iloc[split_idx:]
    
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(50, verbose=False)])
    preds = model.predict_proba(X_va)[:, 1]
    return roc_auc_score(y_va, preds)

# Increase trials from 5 to 50
study.optimize(objective, n_trials=50)
```

### Recommendation 3: Add Logistic Regression Baseline to Final Model Comparison
**Target File:** `Lending_Club_Project_Final.py` (Lines 1143–1148)  
**Issue:** Linear model baseline is missing from the comparison table.  
**Code Action:**
```python
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
lr_model.fit(X_train_nn_scaled, y_train)
y_pred_lr = lr_model.predict_proba(X_test_nn_scaled)[:, 1]
lr_auc = roc_auc_score(y_test, y_pred_lr)

print("\n📊 Final Model Comparison:")
print(f"{'Model':<30} {'ROC-AUC':>10}")
print("-" * 42)
print(f"{'LightGBM + Optuna':<30} {oot_auc:>10.4f}")
print(f"{'Logistic Regression (L2)':<30} {lr_auc:>10.4f}")
print(f"{'MLP Neural Network':<30} {mlp_auc:>10.4f}")
```

### Recommendation 4: Explicit Chronological Validation Set in Keras MLP
**Target File:** `Lending_Club_Project_Final.py` (Lines 1101–1106)  
**Issue:** `validation_split=0.2` relies on implicit slicing.  
**Code Action:**
```python
split_nn_idx = int(len(X_train_nn_scaled) * 0.8)
X_tr_nn, X_va_nn = X_train_nn_scaled[:split_nn_idx], X_train_nn_scaled[split_nn_idx:]
y_tr_nn, y_va_nn = y_train.iloc[:split_nn_idx], y_train.iloc[split_nn_idx:]

history = mlp_model.fit(
    X_tr_nn, y_tr_nn,
    validation_data=(X_va_nn, y_va_nn),
    epochs=25,
    batch_size=2048,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)
```

---

## Conclusion

The Phase 4 Supervised Learning & Predictive Modeling implementation demonstrates high professional quality, meeting all core requirements of `ds_review_guide.md` Section 4. The explicit rejection of random K-Fold CV in favor of `TimeSeriesSplit`, the integration of Optuna tuning with early stopping, the appropriate handling of pandas `category` dtypes for LightGBM vs OHE/Scaling for MLP, and the theoretical backing from NeurIPS 2022 literature are major highlights. Resolving the demo-mode limitations and adding the Logistic Regression baseline will make this phase 100/100 compliant.
