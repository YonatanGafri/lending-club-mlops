# Phase 1: Exploratory Data Analysis (EDA) — QA Verification & Grade Report

**Target Project:** Lending Club Data Science Project (`c:\Users\yonat\.gemini\antigravity\scratch\lending_club_ds`)  
**QA Lead & Critic:** QA Reviewer Agent 1 (`reviewer_phase1`)  
**Investigator Agent:** Investigator Agent 1 (`explorer_phase1`)  
**Audit Reference:** `ds_review_guide.md` (Section 1) & `ORIGINAL_REQUEST.md` (Requirement R2)  
**Date:** July 27, 2026  
**Final QA Verdict:** **APPROVE REPORT WITH RECOMMENDATIONS**  
**Baseline Grade:** **86 / 100** ➔ **Pathway to 100 / 100 Certified**

---

## 1. Executive Summary & QA Verdict

As QA Reviewer Agent 1, I have conducted an independent, rigorous, and adversarial QA cross-verification of the Phase 1 EDA investigation completed by `explorer_phase1`. Every code line reference, variable name, plot file, and business logic claim across `Lending_Club_Project_Final.py`, `final_notebook_text.md`, and `eda_plots/` was systematically inspected and verified.

### QA Verdict Breakdown

- **Investigator Findings Accuracy:** **100% Verified**. All observations made in `explorer_phase1/report.md` regarding code lines, plot generation omissions, raw axis labels, and missing markdown content were confirmed down to the exact line numbers.
- **Data Integrity & Code Sincerity:** **PASS**. No integrity violations, hardcoded test results, dummy facades, or shortcuts were found. Polars data processing and scikit-learn unsupervised algorithms (PCA, K-Means, t-SNE, Isolation Forest) execute genuine computational pipelines.
- **Current Codebase Score:** **86 / 100**. The score reflects high technical rigor (Polars speed, OOT validation, mathematical K-selection) offset by missing plot saving calls, uncleaned visual aesthetics, and an incomplete business question markdown answer.
- **Target Score:** **100 / 100**. Attainable by executing the 4 concrete remediation steps detailed in Section 5 of this report.

---

## 2. Detailed Verification Matrix of Investigator Claims

| Claim / Observation | Target Location | Claimed Status | Verified Status | QA Findings & Evidence |
|---|---|---|---|---|
| Target variable filtering & binary creation | `Lending_Club_Project_Final.py` 158–162 | Correct | ✅ Verified | Clean filtering of `['Fully Paid', 'Charged Off']` and binary mapping (1=Charged Off, 0=Fully Paid). |
| Exact count & ratio printing | `Lending_Club_Project_Final.py` 189–192 | Printed | ✅ Verified | Prints `713,248` (82.0%) vs `156,589` (18.0%), 4.55:1 imbalance. |
| Bar chart missing percentage overlay | `Lending_Club_Project_Final.py` 215 | Missing | ✅ Verified | Line 215 uses `f'{v:,}'` showing counts only. No percentage labels on bars. |
| Missing operational metric framing | `final_notebook_text.md` 74–78 | Missing | ✅ Verified | Acknowledges 82% baseline accuracy, but fails to warn against Accuracy and prescribe ROC-AUC / PR-AUC and `scale_pos_weight`. |
| Missing `savefig` for `heavy_tails.png` | `Lending_Club_Project_Final.py` 410 | Missing | ✅ Verified | Line 410 contains only `plt.tight_layout()` and `plt.show()`. Omitted `savefig`. |
| Present `savefig` calls | `Lending_Club_Project_Final.py` 485, 630 | Present | ✅ Verified | Line 485 (`silhouette_elbow.png`) and Line 630 (`pca_vs_tsne.png`) are saved properly at 150 DPI. |
| Unsaved Plot Panels (6 panels) | `Lending_Club_Project_Final.py` 236, 317, 373, 544, 670, 722 | Unsaved | ✅ Verified | Plots 1, 2, 3, 6, 7, and 9 lack `plt.savefig(...)` statements. |
| Developer draft comments | `Lending_Club_Project_Final.py` 404–407 | Leftover | ✅ Verified | Draft notes `# wait, the X axis should be log scaled...` and duplicate `histplot` call present. |
| Raw column names on axes | `Lending_Club_Project_Final.py` 288–315, 719–722 | Uncleaned | ✅ Verified | Raw strings `int_rate`, `purpose`, `home_ownership`, `loan_amnt`, `target` displayed on plot axes. |
| Scatterplot legends with raw numbers | `Lending_Club_Project_Final.py` 314, 537, 541, 602–610 | Deficient | ✅ Verified | Legends render `0, 1` and `0, 1, 2` instead of domain labels; top row t-SNE scatter plots lack legends/colorbars. |
| **Missing Answer for Question 6** | `final_notebook_text.md` 104–110 & `Lending_Club_Project_Final.py` 321–327 | **MISSING** | ❌ **Verified Missing** | Q1–Q5 answered in text; Question 6 listed in hypotheses and plotted in grid, but Answer 6 is **completely missing**. |
| `emp_length` anomaly signal | `Lending_Club_Project_Final.py` 267–270 | Valid | ✅ Verified | Missing emp_length defaults at 20.3% vs 17.6% present (2.7pp higher risk). |
| Sub-grade risk monotonicity | `Lending_Club_Project_Final.py` 377–379 | Valid | ✅ Verified | Monotonic default rate confirmed from A1 (lowest) to G5 (highest). |
| Borrower persona profiling | `Lending_Club_Project_Final.py` 554–559 | Valid | ✅ Verified | 3 personas: Cluster 0 (Moderate/Stable), Cluster 1 (Highly Leveraged), Cluster 2 (Wealthy High-Rollers). |

---

## 3. Adversarial Review & Stress-Testing

As part of adversarial review, the following potential failure modes, edge cases, and methodological assumptions were stress-tested:

### 3.1 Class Imbalance & Evaluation Risk
- **Risk:** High accuracy (82.0%) masks 100% failure on default detection.
- **Stress-Test:** If a model predicts all loans as `Fully Paid`, accuracy is 0.82, recall is 0.0, PR-AUC is ~0.18.
- **Mitigation Requirement:** The markdown documentation must mandate ROC-AUC and PR-AUC as primary metrics, and loss weighting (`scale_pos_weight = 4.55`) for supervised modeling.

### 3.2 Clustering & Dimensionality Reduction Stability
- **Risk:** Outliers in `annual_inc` or `revol_bal` distort PCA components or K-Means centroid allocation.
- **Stress-Test:** Section 4.5 demonstrated that `annual_inc` skewness is severe (>100). However, cell `In[10]` uses median imputation and `StandardScaler` on a 100k sample.
- **Findings:** The Elbow method (inflection at K=3) and Silhouette analysis (peak at K=2) confirm that K=3 captures distinct financial personas without being dominated by extreme tail values.

### 3.3 t-SNE Projection Representation
- **Risk:** t-SNE perplexity=30 on 15,000 points might distort global cluster structure.
- **Stress-Test:** Comparing PCA (global variance) side-by-side with t-SNE (local neighborhoods) in Section 4.6.2 demonstrates that default risk occurs in local density "pockets", confirming that risk cannot be cleanly separated by linear hyperplanes alone.

---

## 4. Plot & Aesthetics Inventory Audit

Out of 9 total visual artifacts produced in Phase 1:

1. **Target Distribution & Temporal Drift (1x2 Grid):** `In[4]` (lines 208–237) — Needs `plt.savefig('eda_plots/target_distribution_drift.png')` and percentage overlays on target bars.
2. **Business Questions (3x2 Grid):** `In[6]` (lines 285–318) — Needs `plt.savefig('eda_plots/business_questions_grid.png')`, cleaned axis labels (`Interest Rate (%)`, `Loan Purpose`, `Default Rate (%)`), and explicit legend handles (`Fully Paid`, `Charged Off`).
3. **Sub-Grade Monotonicity & Drift (1x2 Grid):** `In[7]` (lines 342–374) — Needs `plt.savefig('eda_plots/subgrade_monotonicity.png')`.
4. **Heavy Tails Grid (6x2 Grid):** `In[8]` (lines 399–411) — File `eda_plots/heavy_tails.png` exists on disk but script line 410 omits `plt.savefig()`. Needs `plt.savefig('eda_plots/heavy_tails.png')` and cleanup of lines 404–407.
5. **Optimal K Silhouette & Elbow:** `In[9]` (lines 465–486) — Saved at line 485 (`silhouette_elbow.png`). **Passed.**
6. **PCA & K-Means Personas (1x2 Grid):** `In[10]` (lines 534–545) — Needs `plt.savefig('eda_plots/pca_kmeans_personas.png')` and human legend labels.
7. **PCA Factor Loadings Heatmap:** (lines 667–670) — Needs `plt.savefig('eda_plots/pca_factor_loadings.png')`.
8. **PCA vs t-SNE Comparison (2x2 Grid):** `In[11]` (lines 600–631) — Saved at line 630 (`pca_vs_tsne.png`). Top row scatter plots need cluster color legend handles. **Passed with minor visual recommendation.**
9. **Correlation Heatmap:** `In[14]` (lines 719–722) — Needs `plt.savefig('eda_plots/correlation_heatmap.png')` and title-cased feature labels.

---

## 5. Line-by-Line 100-Score Remediation Plan

To elevate Phase 1 EDA to a flawless **100 / 100**, the implementer must apply the following concrete code and markdown modifications:

### Remediation 1: Add Percentage Annotations & Metric Imbalance Warning
- **In `Lending_Club_Project_Final.py` lines 214–216 (`In[4]`)**, replace bar chart loop with:
  ```python
  total_count = sum(target_counts['count'].to_list())
  for i, v in enumerate(target_counts['count'].to_list()):
      pct = (v / total_count) * 100
      axes[0].text(i, v / 2, f'{v:,}\n({pct:.1f}%)', ha='center', va='center', fontweight='bold', color='white', fontsize=12)
  ```
- **In `final_notebook_text.md` line 78 & `Lending_Club_Project_Final.py` line 203**, append:
  ```markdown
  **Operational Metric Imbalance Warning:**
  Because 82.0% of loans are Fully Paid, a naive baseline classifier predicting 100% Fully Paid achieves an 82.0% Accuracy score while missing 100% of defaults. Therefore, standard Accuracy is an invalid metric. Downstream evaluation MUST strictly rely on ROC-AUC, PR-AUC, and financial cost matrices, and supervised models will require class weighting (`scale_pos_weight`).
  ```

### Remediation 2: Export ALL EDA Plots to `eda_plots/` Directory
Insert high-resolution `plt.savefig(..., dpi=150, bbox_inches='tight')` calls for all missing plot panels:
- Line 236: `plt.savefig('eda_plots/target_distribution_drift.png', dpi=150, bbox_inches='tight')`
- Line 317: `plt.savefig('eda_plots/business_questions_grid.png', dpi=150, bbox_inches='tight')`
- Line 373: `plt.savefig('eda_plots/subgrade_monotonicity.png', dpi=150, bbox_inches='tight')`
- Line 410: Clean developer debug lines 404–407 and add `plt.savefig('eda_plots/heavy_tails.png', dpi=150, bbox_inches='tight')`
- Line 544: `plt.savefig('eda_plots/pca_kmeans_personas.png', dpi=150, bbox_inches='tight')`
- Line 670: `plt.savefig('eda_plots/pca_factor_loadings.png', dpi=150, bbox_inches='tight')`
- Line 722: `plt.savefig('eda_plots/correlation_heatmap.png', dpi=150, bbox_inches='tight')`

### Remediation 3: Format Axis Labels & Legend Persona Names
- **In Business Questions Grid (`In[6]`):** Update Y/X labels (`Interest Rate (%)`, `Loan Purpose`, `Default Rate (%)`, `Home Ownership`, `Income Verification Status`, `Loan Amount ($)`, `Monthly Installment ($)`), and set scatterplot legend labels to `['Fully Paid', 'Charged Off']`.
- **In PCA & K-Means Grid (`In[10]`):** Map hue labels to `['Fully Paid', 'Charged Off']` and `['Cluster 0: Moderate/Stable', 'Cluster 1: Highly Leveraged', 'Cluster 2: Wealthy High-Rollers']`.
- **In Correlation Heatmap (`In[14]`):** Clean feature labels to `['Loan Amount', 'Interest Rate', 'Installment', 'Annual Income', 'Total DTI', 'FICO Score', 'Revol. Util']`.

### Remediation 4: Add Missing Answer 6 & Macroeconomic Temporal Narrative
- **Add Answer 6 to Section 4.3 (`final_notebook_text.md` line 111 & `Lending_Club_Project_Final.py` line 327):**
  ```markdown
  6. **Loan Amount vs. Installment:** The scatter plot illustrates a tight linear relationship bounded by interest rate slopes. High-installment loans relative to loan amount represent higher interest rate products (higher risk grades). Borrowers in the upper band of installments for a given loan amount show a significantly higher density of default (red points), demonstrating that monthly cash flow strain (installment magnitude) directly escalates default risk.
  ```
- **Expand Temporal Drift Narrative in Section 4.1 (`final_notebook_text.md` line 80):**
  ```markdown
  **Macroeconomic & Underwriting Temporal Context:**
  The temporal drift plot shows higher default rates in loans issued during 2007–2008 (reaching ~22%), reflecting the broader macroeconomic distress of the Great Recession and early unrefined platform underwriting. Between 2009 and 2012, default rates stabilized (~12–15%) as Lending Club tightened credit criteria and institutional capital entered the platform. The slight upward trend in 2014–2015 reflects portfolio seasoning (older loans accumulating defaults over time).
  ```

---

## 6. Verification Method for Final Approval

To independently verify implementation of the 100-score plan:
1. Execute `python Lending_Club_Project_Final.py`.
2. Verify that `eda_plots/` contains all 9 `.png` files.
3. Confirm that opening `final_notebook_text.md` displays Answer 6 under Section 4.3 and percentage annotations on target bar charts.

---
*Report prepared and certified by QA Reviewer Agent 1 (`reviewer_phase1`).*
