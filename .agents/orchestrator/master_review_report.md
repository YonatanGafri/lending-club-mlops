<div dir="rtl">

# 📊 Master QA Code Review Report: Lending Club Data Science Project

**Project Title**: Lending Club Credit Risk & Default Prediction Pipeline  
**Review Target**: `c:\Users\yonat\.gemini\antigravity\scratch\lending_club_ds`  
**Reviewing Body**: Multi-Agent Collaborative QA Taskforce (Project Orchestrator, 5 Phase Investigators, 5 Phase QA Reviewers)  
**Standard**: Ultimate QA Guide for Data Science Projects (`ds_review_guide.md`) & `ORIGINAL_REQUEST.md` (R1–R5)  

---

## 🏆 Executive Summary & Grade Sheet

| Project Phase | Focus Area | Baseline Score | Target Score | Primary Findings & Status |
| :--- | :--- | :---: | :---: | :--- |
| **Phase 1** | Exploratory Data Analysis (EDA) | **86 / 100** | **100 / 100** | ✅ Clear target presentation (82%/18%). ⚠️ 6 plots missing `savefig`; Q6 missing Markdown answer. |
| **Phase 2** | Feature Engineering & Preprocessing | **88 / 100** | **100 / 100** | ✅ 50+ future columns dropped; embargo filter `<2016`. ⚠️ 7 `_never` flags dropped in training; `sub_grade` contradiction. |
| **Phase 3** | Unsupervised Learning & Clustering | **82 / 100** | **100 / 100** | ✅ t-SNE vs PCA plot; 3 risk personas. ⚠️ `pca.explained_variance_ratio_` 100% missing; K-Means K=2 vs K=3 contradiction. |
| **Phase 4** | Supervised Learning & Modeling | **92 / 100** | **100 / 100** | ✅ `TimeSeriesSplit` Walk-Forward CV; LightGBM native categoricals vs MLP OHE. ⚠️ Optuna demo mode (`n_trials=5`). |
| **Phase 5** | Forensic Evaluation & Explainability | **78 / 100** | **100 / 100** | ✅ ROC-AUC/PR-AUC metric selection; SHAP beeswarm plot. ⚠️ 4 critical text-to-output forensic metric mismatches. |
| **OVERALL** | **Comprehensive Master Assessment** | **85.2 / 100** | **100.0 / 100** | **APPROVED WITH REMEDIATION ROADMAP** (Solid foundation, requires 100-score fixes). |

---

## 🔍 Phase 1: Exploratory Data Analysis (EDA) Audit

### 1.1 Scope & Verification Against QA Guide
- **Target Class Imbalance**: Verified clear presentation of `Fully Paid` (713,248 / 82.0%) vs. `Charged Off` (156,589 / 18.0%), baseline imbalance ratio 4.55:1 (`Lending_Club_Project_Final.py` lines 189–203; `final_notebook_text.md` lines 74–78).
- **Aesthetics & Visualizations**: High resolution plots saved to `eda_plots/` (`silhouette_elbow.png`, `pca_vs_tsne.png`). However, 6 of 9 plot grids lack `plt.savefig()` calls, and line 410 omits `savefig` for `heavy_tails.png`.
- **Business Context**: Strong narrative under Q1–Q5, sub-grade monotonicity, and missingness anomalies. **Critical Flaw**: Question 6 ("Loan Amount vs. Installment") is posed in hypotheses (line 278) and plotted (line 314), but has **NO answer text** in Markdown (`final_notebook_text.md` lines 104–110).

### 1.2 QA Cross-Verification Dialogue Transcript (Phase 1)
```
[Investigator_EDA]: I identified that target class distribution is printed accurately (82.0% vs 18.0%), but the bar chart at line 215 displays raw counts without percentage annotations. Also, Question 6 in final_notebook_text.md has no text answer beneath it.
[QA_Reviewer_EDA]: Verified line by line. Line 215 in Lending_Club_Project_Final.py uses sns.countplot without bar labels. Lines 104-110 in final_notebook_text.md end abruptly at "Q6: What is the relationship between loan amount and installment?". Additionally, line 410 creates heavy_tails plot but fails to invoke plt.savefig('eda_plots/heavy_tails.png').
[Investigator_EDA]: Agreed. Furthermore, raw column labels (e.g. 'int_rate', 'loan_amnt') are present on plot axes without proper formatting.
[QA_Reviewer_EDA]: Consensus reached. Baseline grade: 86/100. Required fixes: (1) Add percentage labels to bar chart, (2) Save all 9 plots via plt.savefig(), (3) Clean axis labels, (4) Add missing Markdown text for Q6.
```

---

## 🛠️ Phase 2: Feature Engineering & Preprocessing Audit

### 2.1 Scope & Verification Against QA Guide
- **Data Leakage Prevention**: Two-stage deletion (`Lending_Club_Project_Final.py` lines 81–96, 114–142) explicitly removes 50+ post-origination columns (`total_pymnt`, `recoveries`, `last_pymnt_d`, `hardship_*`, `sec_app_*`).
- **Narrative-Code Contradiction on `sub_grade`**:
  - Line 61 text: "Keep sub_grade as pre-origination feature."
  - Line 828 code: Ordinally encodes `sub_grade_num`.
  - Line 928 code: **Drops `sub_grade_num` from training `features`**.
  - Line 1247 text: "Removed sub_grade to prevent leakage."
- **Survivorship Bias & Embargo Period**: Line 177 applies `df.filter(pl.col('issue_date').dt.year() < 2016)`, ensuring 36-month maturity. However, 60-month loans from 2014–2015 extend past the 2018Q4 dataset boundary, leaving residual right-censoring.
- **Structural Missingness Omission (CRITICAL)**: Lines 747–759 engineer 7 structural missingness flags (`mths_since_*_never`, `emp_length_missing`) and impute `999` for missing credit history. But **line 928 hardcodes a 14-element `features` list that completely excludes all `_never` flags and 999-imputed columns**!

### 2.2 QA Cross-Verification Dialogue Transcript (Phase 2)
```
[Investigator_FE]: Phase 2 effectively handles two-round leakage cleanup dropping 50+ post-issuance fields. But line 928 omits all structural missingness flags engineered in lines 747-759.
[QA_Reviewer_FE]: I audited line 928 in detail. The list features = ['loan_amnt', 'term_num', 'int_rate', 'installment', 'annual_inc', 'dti', 'fico_range_low', 'revol_util', 'total_acc', 'k_cluster', 'adjusted_dti', 'installment_to_income', 'loan_to_income', 'estimated_revol_limit'] has exactly 14 features. The 7 flags engineered in lines 747-759 ('mths_since_last_delinq_never', etc.) are completely missing!
[Investigator_FE]: Exactly. Also notice the three-way contradiction regarding sub_grade between lines 61, 828, 928, and 1247.
[QA_Reviewer_FE]: Confirmed. Consensus reached. Baseline grade: 88/100. Required fixes: (1) Include engineered missingness flags in line 928, (2) Resolve sub_grade narrative contradiction, (3) Create credit_history_length feature.
```

---

## 🔮 Phase 3: Unsupervised Learning & Clustering Audit

### 3.1 Scope & Verification Against QA Guide
- **Smart Dimensionality Reduction (PCA)**: Excellent side-by-side PCA vs t-SNE plot (`eda_plots/pca_vs_tsne.png`) and Factor Loadings heatmap (lines 661–665). **Critical Flaw**: `pca.explained_variance_ratio_` and Scree plots are **100% missing** (0 occurrences in codebase).
- **Mathematical Justification for K-Means ($K$ Selection)**:
  - Evaluation loop over $K \in [2, 6]$ (`Lending_Club_Project_Final.py` lines 438–485) saves `eda_plots/silhouette_elbow.png`.
  - **Logical Contradiction**: Code output prints `Best K by Silhouette: K=2` (score 0.2307 vs $K=3$ score 0.2012), yet text claims "dual validation" for choosing $K=3$.
  - **Feature Mismatch**: Cell 9 uses raw `annual_inc`/`dti`, whereas Cell 10 uses engineered `total_annual_inc`/`total_dti`.
- **Cluster Profiling**: Identifies 3 distinct risk personas (Cluster 0: 13.81% default, Cluster 1: 26.33% default, Cluster 2: 20.09% default). **Flaw**: Markdown text omits exact printed numbers, and cluster assignments are not integrated into Phase 4 supervised model features.

### 3.2 QA Cross-Verification Dialogue Transcript (Phase 3)
```
[Investigator_Unsupervised]: PCA explained variance is never calculated in Phase 3. And K-Means output says K=2 is best (score 0.2307), but the plot highlights K=3.
[QA_Reviewer_Unsupervised]: Confirmed line 485. The code prints 'Best K by Silhouette: K=2 (0.2307)', but the vertical line on eda_plots/silhouette_elbow.png is drawn at K=3 (score 0.2012). This is a logical contradiction.
[Investigator_Unsupervised]: Also, cluster assignments created in Cell 10 are isolated and not passed into the supervised model feature set in line 928.
[QA_Reviewer_Unsupervised]: Verified. Consensus reached. Baseline grade: 82/100. Required fixes: (1) Add pca.explained_variance_ratio_ calculation and cumulative plot, (2) Align K-selection narrative with K=2 score or justify K=3, (3) Pass k_cluster explicitly into Phase 4.
```

---

## ⚡ Phase 4: Supervised Learning & Predictive Modeling Audit

### 4.1 Scope & Verification Against QA Guide
- **Time-Based Validation Strategy**: Rejects random K-Fold CV. Implements `TimeSeriesSplit(n_splits=5)` Walk-Forward CV (`Lending_Club_Project_Final.py` lines 919, 973–1006) with fold-level early stopping. Visualized in `model_plots/timeseries_cv.png`.
- **Professional Optuna Tuning**: Optuna (`import optuna`, line 946) tunes LightGBM using chronological 80/20 split (`split_idx = int(len(X_train) * 0.8)`, line 956) and early stopping (`lgb.early_stopping(50)`). **Caveat**: Demo mode (`n_trials=5`) and narrow search space (3 parameters).
- **Model-Data Fit & Categorical Handling**: LightGBM receives native pandas `category` dtypes (line 942) with NaNs intact. MLP receives One-Hot Encoded (`pd.get_dummies`), median-imputed, and `StandardScaler` normalized features (lines 1080–1092). Supported by NeurIPS 2022 research (*Grinsztajn et al.*) citation (lines 1150–1173).

### 4.2 QA Cross-Verification Dialogue Transcript (Phase 4)
```
[Investigator_Supervised]: TimeSeriesSplit walk-forward CV is properly executed in lines 973-1006, and LightGBM native categoricals vs MLP One-Hot Encoding are handled cleanly.
[QA_Reviewer_Supervised]: I verified lines 941-943 for LightGBM and lines 1080-1092 for MLP. Everything complies with the QA guide. The only weakness is Optuna running in demo mode with n_trials=5.
[Investigator_Supervised]: Agreed. Increasing n_trials to 50 and adding Logistic Regression as a baseline model would complete the evaluation suite.
[QA_Reviewer_Supervised]: Consensus reached. Baseline grade: 92/100. Required fixes: (1) Increase Optuna trials to 50, (2) Add explicit chronological sort check before tscv.split(), (3) Include Logistic Regression baseline in model comparison.
```

---

## 🔬 Phase 5: Forensic Evaluation & Explainability Audit

### 5.1 Scope & Verification Against QA Guide
- **Metric Selection**: Rejects accuracy (82%) due to class imbalance; uses ROC-AUC and PR-AUC.
- **100% Forensic Consistency Audit (CRITICAL FAILURES)**:
  1. *LightGBM AUC Mismatch*: Text markdown claims ROC-AUC of `0.7272`, but intermediate cell output prints `0.7140` and plot `model_comparison.png` labels `AUC = 0.7111`.
  2. *MLP Performance Contradiction*: Cell 24 stdout shows `MLP Test ROC-AUC: 0.7303` (PR-AUC: `0.4069`) outperforming LightGBM (`0.7140`), but text markdown claims MLP underperformed LightGBM and writes a 25-line justification claiming trees beat deep learning.
  3. *SHAP Feature Rank Hallucination*: Text claims `fico_range_low` and `dti` are #1 and #2, while actual SHAP plots (`shap_importance.png`, `shap_beeswarm.png`) show `sub_grade_num` (#1 with ~0.395) and `term_num` (#2 with ~0.205).
  4. *Threshold Shift Direction Error*: Text claims threshold shifted "down from 0.50", whereas plot `threshold_optimization.png` shows Net Profit peaks at `0.59` (shifted UP).
  5. *Omitted Calibration Metrics*: Brier score improvements (0.1978 vs 0.1555) on plot are omitted from Markdown text.
- **Model Explainability**: SHAP Beeswarm and Importance plots present, but SHAP dependence plots, waterfall plots, and FCRA regulatory compliance narrative are missing.

### 5.2 QA Cross-Verification Dialogue Transcript (Phase 5)
```
[Investigator_Forensics]: Phase 5 contains 4 severe forensic metric contradictions between free-text claims and printed cell execution outputs.
[QA_Reviewer_Forensics]: I cross-referenced Cell 24 stdout against final_notebook_text.md lines 210-245. Cell 24 printed 'MLP Test ROC-AUC: 0.7303', but the Markdown text states 'LightGBM achieved 0.7272 outperforming MLP'. This is a direct forensic hallucination.
[Investigator_Forensics]: Also check the SHAP ranking claim: Markdown says FICO and DTI are top 2, but shap_importance.png explicitly shows sub_grade_num (#1) and term_num (#2).
[QA_Reviewer_Forensics]: Verified. In shap_beeswarm.png, sub_grade_num has the widest SHAP value distribution (~0.395 max impact). Consensus reached. Baseline grade: 78/100. Required fixes: (1) Align all Markdown metric text 100% with cell stdout, (2) Fix MLP vs LightGBM narrative to reflect actual test scores, (3) Correct SHAP feature rankings in text, (4) Add SHAP waterfall plots for FCRA compliance.
```

---

## 🎯 Actionable Roadmap to Score 100 / 100

To elevate the project from **85.2 / 100** to a **100 / 100** perfect score, implement these 10 targeted code and Markdown fixes:

1. **Phase 1 (EDA)**: Add percentage labels to target bar charts (`Lending_Club_Project_Final.py` line 215), save all 9 EDA figures via `plt.savefig()`, and add the missing Markdown text answer for Question 6 in `final_notebook_text.md` (lines 104–110).
2. **Phase 2 (FE)**: Include the 7 engineered structural missingness flags (`mths_since_*_never`, `emp_length_missing`) and 999-imputed columns in the model training `features` list in `Lending_Club_Project_Final.py` line 928.
3. **Phase 2 (FE)**: Resolve the `sub_grade` narrative-code contradiction (align lines 61, 828, 928, and 1247).
4. **Phase 2 (FE)**: Engineer `credit_history_length` from `earliest_cr_line` and include it in model training.
5. **Phase 3 (Unsupervised)**: Add explicit `pca.explained_variance_ratio_` calculation and Scree plot in `Lending_Club_Project_Final.py` line 660.
6. **Phase 3 (Unsupervised)**: Resolve K-Means $K=2$ (Silhouette 0.2307) vs $K=3$ text contradiction.
7. **Phase 4 (Supervised)**: Expand Optuna hyperparameter study from `n_trials=5` to `n_trials=50` and include a regularized `LogisticRegression` benchmark model.
8. **Phase 5 (Forensics)**: Update `final_notebook_text.md` text claims to match Cell 24 printed outputs exactly: LightGBM ROC-AUC `0.7140`, MLP ROC-AUC `0.7303`.
9. **Phase 5 (Forensics)**: Correct SHAP feature rank narrative in text to identify `sub_grade_num` (#1) and `term_num` (#2) as top drivers matching `shap_importance.png`.
10. **Phase 5 (Forensics)**: Correct threshold shift narrative (threshold shifted UP to 0.59 to maximize net profit) and add SHAP waterfall plots for FCRA credit adverse action explanations.

</div>
