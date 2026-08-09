# Lending Club Data Science Project QA Review Plan

## Objective
Execute a rigorous 5-phase collaborative code review of the Lending Club Data Science codebase (`Lending_Club_Project_Final.py`, `Lending_Club_Project_Final.ipynb`, `final_notebook_text.md`, saved plots, etc.) against `ds_review_guide.md` to help the student achieve a perfect 100 score.

## Review Architecture & Methodology
For each of the 5 phases, an Investigating Agent (`teamwork_preview_explorer`) will analyze the project artifacts, extract code snippets, line numbers, figures, and textual claims, and write an Investigation Report. Then a QA Counterpart Agent (`teamwork_preview_reviewer`) will independently audit the findings, verify line numbers and metrics, challenge discrepancies, and engage in cross-verification dialogue. Finally, Orchestrator will synthesize the dialogue into the master review report.

## 5 Review Phases
1. **Phase 1: Exploratory Data Analysis (EDA)**
   - Target presentation (`Fully Paid` vs. `Charged Off`), class imbalance (82%/18%).
   - Visualization aesthetics, labels, title, color usage.
   - Business context & Markdown interpretations under every major chart.

2. **Phase 2: Feature Engineering & Preprocessing**
   - Data leakage prevention (`total_pymnt`, `recoveries`, `int_rate`, `sub_grade`, etc.).
   - Survivorship bias & embargo period application (2016+ loan filtering).
   - Custom financial features (Joint application income handling, debt-to-income ratios).
   - NaN & categorical handling (avoiding blind `fillna(0)`, LightGBM categorical features, `_never` flags).

3. **Phase 3: Unsupervised Learning & Clustering**
   - PCA explained variance ratio & dimensionality reduction justification.
   - Mathematical justification for K-Means clusters (Silhouette Score, Elbow Method).
   - Business profiling of clusters (borrower personas & risk characterization).

4. **Phase 4: Supervised Learning & Predictive Modeling**
   - Validation strategy: `TimeSeriesSplit` / Walk-Forward CV vs. random K-Fold.
   - Hyperparameter tuning (Optuna, early stopping, chronological splits).
   - Model-specific categorical handling (LightGBM `category` dtype vs. MLP One-Hot Encoding).

5. **Phase 5: Forensic Evaluation & Explainability**
   - Metric selection: ROC-AUC and PR-AUC vs. misleading Accuracy.
   - **Forensic Consistency**: 100% strict cross-referencing between textual summary claims and actual printed outputs (unrounded, non-hallucinated).
   - Model Explainability: SHAP summary, dependence, and waterfall plots.

## Deliverables
- Detailed phase reports from Investigators & QA Reviewers with full dialogue transcripts.
- Comprehensive Master Synthesis Review Report with line-by-line code/figure references and actionable recommendations for a perfect score (100).
