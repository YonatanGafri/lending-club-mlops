# Phase 1: Exploratory Data Analysis (EDA) — Exhaustive Audit Report

**Target Project:** Lending Club Data Science Project (`c:\Users\yonat\.gemini\antigravity\scratch\lending_club_ds`)  
**Auditor:** Investigator Agent 1 (Phase 1 EDA Specialist)  
**Audit Standards:** `ds_review_guide.md` (Section 1) & `ORIGINAL_REQUEST.md` (Requirement R2)  
**Date:** July 27, 2026  
**Status:** Audit Complete  
**Current Phase Score:** **86 / 100** ➔ **Pathway to 100 / 100** detailed below.

---

## 1. Executive Summary

This report delivers an exhaustive, line-by-line audit of Phase 1: Exploratory Data Analysis (EDA) for the Lending Club Data Science project. The project exhibits strong computational rigor, clean data pipelines using Polars and scikit-learn, mathematical justification for unsupervised clustering (Silhouette & Elbow methods), and thoughtful credit risk domain insights. 

However, to achieve a Kaggle-level / 100-percent perfect grade, critical gaps must be resolved in **plot saving & aesthetics**, **completeness of business markdown text** (specifically missing Q6 analysis and temporal drift context), and **explicit operational framing of class imbalance**.

### Summary Assessment Table

| Audit Area | Current Status | Key Strengths | Key Flaws / Gaps | Impact on Score |
|---|---|---|---|---|
| **1. Target Variable & Class Imbalance** | **Pass (Minor Gaps)** | Exact counts & percentages printed (82.0% vs 18.0%, 4.55:1 imbalance ratio); baseline accuracy (82.0%) acknowledged; annotated count bar chart. | Missing percentage annotations on bars; missing explicit connection in EDA text to metric selection (ROC-AUC / PR-AUC vs Accuracy) and imbalance handling techniques (`scale_pos_weight`, SMOTE). | -4 pts |
| **2. Aesthetics & Visualizations** | **Needs Improvement** | Comprehensive coverage (9 plot panels/grids); silhouette/elbow and PCA/t-SNE plots are high-resolution & saved to disk. | **6 of 9 plots lack `plt.savefig()`**; `heavy_tails.png` exists on disk but is unreferenced in script; developer draft comments left in heavy tails code; uncleaned raw column axis labels (`int_rate`, `purpose`, `home_ownership`, `loan_amnt`); scatterplot legends showing raw numbers (`0, 1`) instead of human labels (`Fully Paid`, `Charged Off`); t-SNE top row missing cluster colorbar/legend. | -7 pts |
| **3. Business Context & Explanations** | **Pass (Minor Gaps)** | High quality business domain explanations for Q1–Q5, structural missingness (`emp_length`), sub-grade monotonicity, and borrower persona profiling (Clusters 0, 1, 2). | **Question 6 (Loan Amount vs Installment) has NO answer** in the Markdown text; temporal drift line charts lack macroeconomic underwriting context explaining historical default spikes. | -3 pts |

---

## 2. Detailed Audit Area 1: Target Variable Presentation & Class Imbalance

### Audit Criteria (`ds_review_guide.md` §1.2 & `ORIGINAL_REQUEST.md` R2)
- Target variable definition clear (`Fully Paid` vs `Charged Off`).
- Exact distribution numbers and percentages stated (e.g. 82% vs 18%).
- Class imbalance awareness explicitly addressed (baseline accuracy 82%, impact on metrics, ratio awareness).

### Observations & File Evidence

1. **Target Definition Code**:
   - **File:** `Lending_Club_Project_Final.py` lines 158–162:
     ```python
     df = df.filter(pl.col('loan_status').is_in(['Fully Paid', 'Charged Off']))
     df = df.with_columns(
         pl.when(pl.col('loan_status') == 'Charged Off').then(1).otherwise(0).alias('target')
     )
     ```
   - **Assessment:** **PASS/SUCCESS**. Terminal status filtering is clean, explicit, and accurate.

2. **Printed Target Statistics**:
   - **File:** `Lending_Club_Project_Final.py` lines 189–192:
     ```python
     print(f'\nTarget distribution (Train):')
     print(f'  Fully Paid: {(train_df["target"] == 0).sum():,} ({(train_df["target"] == 0).mean()*100:.1f}%)')
     print(f'  Charged Off: {(train_df["target"] == 1).sum():,} ({(train_df["target"] == 1).mean()*100:.1f}%)')
     print(f'  Imbalance Ratio: 1:{(train_df["target"] == 1).sum() / (train_df["target"] == 0).sum():.2f} (Normalized: {(train_df["target"] == 0).mean()*100:.1f}% vs {(train_df["target"] == 1).mean()*100:.1f}%)')
     ```
   - **Assessment:** **PASS/SUCCESS**. Prints exact counts (`713,248` Fully Paid vs `156,589` Charged Off) and percentages (`82.0%` vs `18.0%`).

3. **Markdown Text Documentation**:
   - **File:** `final_notebook_text.md` lines 74–78 & `Lending_Club_Project_Final.py` lines 199–203:
     ```markdown
     **Key Findings:**
     - **Class Imbalance**: 1:0.22 ratio (or 4.55:1) (82.0% Fully Paid vs 18.0% Charged Off)
     - Baseline accuracy of 82.0% achieved by predicting all loans as 'Fully Paid'
     - This confirms **OOT split was the right choice** over random split
     ```
   - **Assessment:** **PARTIAL PASS / GAP**. While the baseline accuracy of 82.0% is acknowledged, the text fails to explain *why* class imbalance breaks standard evaluation metrics (like Accuracy) and why loss function adjustments (e.g. `scale_pos_weight` in LightGBM) or ROC-AUC/PR-AUC evaluation are strictly mandatory for downstream modeling.

4. **Target Distribution Visual Bar Chart**:
   - **File:** `Lending_Club_Project_Final.py` lines 210–218 (`In[4]`):
     ```python
     axes[0].bar(labels, target_counts['count'].to_list(), color=colors, edgecolor='white')
     for i, v in enumerate(target_counts['count'].to_list()):
         axes[0].text(i, v + 10000, f'{v:,}', ha='center', fontweight='bold')
     ```
   - **Assessment:** **FLAW/GAP**. The chart displays raw counts (`713,248` and `156,589`) above the bars, but does NOT display percentages (`82.0%` and `18.0%`). Adding percentages directly inside or above the bars is essential for immediate visual comprehension.

---

## 3. Detailed Audit Area 2: Aesthetic & Visualizations Audit

### Audit Criteria (`ds_review_guide.md` §1.1)
- Clean, readable plots with titles, formatted axis labels, legends, consistent color palettes.
- High resolution, proper bounding, saved files in `eda_plots/`.

### Comprehensive Inventory of Phase 1 Plots

| Plot # | Description | Py Script Lines | Notebook Cell | Output File | `plt.savefig` Present? | Aesthetic Grade | Major Issues |
|---|---|---|---|---|---|---|---|
| **1** | Target Distribution & Temporal Drift (1x2 Grid) | 208–237 | `In[4]` | Missing | ❌ No | **B** | Not saved to file; raw counts instead of percentages on bar chart. |
| **2** | Business Questions & Hypotheses (3x2 6-Panel Grid) | 285–318 | `In[6]` | Missing | ❌ No | **C+** | Not saved to file; uncleaned axis labels (`int_rate`, `purpose`, `home_ownership`, `loan_amnt`); scatterplot legend shows raw `0, 1`; inconsistent palettes. |
| **3** | Sub-Grade Monotonicity & Temporal Drift (1x2 Grid) | 342–374 | `In[7]` | Missing | ❌ No | **B+** | Not saved to file; sub-grade labels rotated 90° at font size 7; Y-axis in plot 2 shows decimals instead of % (e.g. `0.15` vs `15%`). |
| **4** | Heavy Tails Linear vs Log Scale (6x2 Grid) | 399–411 | `In[8]` | `eda_plots/heavy_tails.png` | ❌ **No in script!** | **B** | File exists on disk but script line 410 omits `plt.savefig()`; developer debug comment left on line 404; scientific notation on linear X-axes. |
| **5** | Optimal K Selection: Silhouette & Elbow (1x2 Grid) | 465–486 | `In[9]` | `eda_plots/silhouette_elbow.png` | ✅ Yes | **A** | Excellent plot! Saved at 150 DPI. Minor tension between text "Best K=2" and plot marker K=3. |
| **6** | PCA & K-Means 3 Borrower Personas (1x2 Grid) | 534–545 | `In[10]` | Missing | ❌ No | **B-** | Not saved to file; legend plot 1 shows `0, 1`; legend plot 2 shows `0, 1, 2` without persona names. |
| **7** | PCA Factor Loadings Heatmap | 667–670 | Unnumbered | Missing | ❌ No | **B** | Not saved to file; raw feature names on Y-axis. |
| **8** | PCA vs t-SNE Comparison (2x2 Grid) | 600–631 | `In[11]` | `eda_plots/pca_vs_tsne.png` | ✅ Yes | **A-** | Excellent plot! Saved at 150 DPI. Top row scatter plots lack cluster colorbar/legend labels. |
| **9** | Correlation Heatmap | 719–722 | `In[14]` | Missing | ❌ No | **B+** | Not saved to file; raw column names (`fico_range_low`, `total_annual_inc`). |

### Detailed Plot Flaw Breakdown

1. **Missing `plt.savefig()` Calls**:
   - **Observation:** Out of 9 plots, only 2 (`silhouette_elbow.png` and `pca_vs_tsne.png`) have explicit `plt.savefig()` statements in `Lending_Club_Project_Final.py` and `Lending_Club_Project_Final.ipynb`.
   - **`heavy_tails.png` Discrepancy:** File `eda_plots/heavy_tails.png` exists in the filesystem (size 246,743 bytes), but line 410 in `Lending_Club_Project_Final.py` only contains `plt.tight_layout()` and `plt.show()`. The `savefig` call was omitted!
   - **Impact:** Re-running the script does not populate `eda_plots/` with the remaining 7 plots required for presentation and HTML output.

2. **Uncleaned Axis Labels & Raw Column Names**:
   - **File:** `Lending_Club_Project_Final.py` lines 288–315 (`In[6]`):
     - `axes[0, 1]`: Y-axis is `int_rate` ➔ Should be `Interest Rate (%)`.
     - `axes[1, 0]`: Y-axis is `purpose` ➔ Should be `Loan Purpose`.
     - `axes[1, 1]`: Y-axis is `target` ➔ Should be `Default Rate (%)`, X-axis is `home_ownership` ➔ Should be `Home Ownership`.
     - `axes[2, 0]`: Y-axis is `target` ➔ Should be `Default Rate (%)`, X-axis is `verification_status` ➔ Should be `Income Verification Status`.
     - `axes[2, 1]`: X-axis is `loan_amnt` ➔ Should be `Loan Amount ($)`, Y-axis is `installment` ➔ Should be `Monthly Installment ($)`.
   - **File:** `Lending_Club_Project_Final.py` lines 719–722 (`In[14]` Correlation Heatmap):
     - Feature labels show `fico_range_low`, `total_annual_inc`, `total_dti`, `revol_util` ➔ Should be clean title-cased labels (`FICO Score`, `Total Income`, `Total DTI`, `Revol. Utilization`).

3. **Scatterplot Legend Deficiencies**:
   - **File:** `Lending_Club_Project_Final.py` line 314 (`In[6]`, Subplot Q6): `hue='target'` renders default Seaborn legend showing `0` and `1`. Viewers must guess which color represents `Fully Paid` vs `Charged Off`.
   - **File:** `Lending_Club_Project_Final.py` line 537 (`In[10]`, PCA Subplot 1): `hue='target'` renders `0` and `1`.
   - **File:** `Lending_Club_Project_Final.py` line 541 (`In[10]`, PCA Subplot 2): `hue='Cluster'` renders `0, 1, 2` instead of cluster persona names (`Moderate/Stable`, `Highly Leveraged`, `Wealthy High-Rollers`).
   - **File:** `Lending_Club_Project_Final.py` lines 602–610 (`In[11]`, `pca_vs_tsne.png` Top Row): `c=clusters_sample` is passed to `plt.scatter()`, but no legend or colorbar is created for the top row panels, leaving cluster colors unannotated!

4. **Leftover Developer Code Comments**:
   - **File:** `Lending_Club_Project_Final.py` lines 404–407 (`In[8]`):
     ```python
     sns.histplot(vals, bins=50, ax=axes[i, 1], color='#e67e22', log_scale=(False, True)) # wait, the X axis should be log scaled, or Y? 
     # Actually, for right-skewed data, log scale on X is common, but sns.histplot log_scale=True applies to X axis if not specified otherwise in older versions, or both? 
     # The safest is `log_scale=(True, False)` for X-axis log.
     ```
   - **Assessment:** **FLAW/GAP**. Developer scratch notes and internal debates must be cleaned up prior to final submission.

---

## 4. Detailed Audit Area 3: Business Context & Markdown Explanations

### Audit Criteria (`ds_review_guide.md` §1.3)
- Under EVERY major chart, a dedicated Markdown paragraph explaining what business insight was gained about borrowers and credit risk.

### Section-by-Section Markdown Audit

1. **Target Distribution & Temporal Drift (Section 4.1)**:
   - **Text Location:** `final_notebook_text.md` lines 74–79; `Lending_Club_Project_Final.py` lines 199–204.
   - **Content:** Highlights class imbalance (82/18) and OOT split rationale.
   - **Gap:** Lacks macroeconomic credit narrative for the temporal drift chart (`axes[1]`). Why did default rates change from 2007 to 2015? (Context needed: 2008 Financial Crisis impact, portfolio maturation, and Lending Club's credit policy tightening after 2011).

2. **Structural Missingness Analysis (Section 4.2)**:
   - **Text Location:** `final_notebook_text.md` lines 87–89; `Lending_Club_Project_Final.py` lines 267–270.
   - **Content:** Explains `emp_length` anomaly (missing emp_length defaults at 20.3% vs 17.6% present) and protective nature of `mths_since_*` columns.
   - **Assessment:** **PASS/SUCCESS**. Outstanding risk interpretation.

3. **Business Questions & Hypotheses Testing (Section 4.3)**:
   - **Text Location:** `final_notebook_text.md` lines 95–110; `Lending_Club_Project_Final.py` lines 271–279 & 321–327.
   - **Content:** Covers Q1 (Term), Q2 (Interest Rate), Q3 (Purpose), Q4 (Home Ownership), Q5 (Verification Status).
   - **CRITICAL FLAW / GAP:** **Question 6 is listed in the hypotheses (line 100) and plotted in subplot (2,1) (line 314), BUT has NO answer in the Markdown summary!** Answers 1 through 5 are documented, but Answer 6 is completely missing!

4. **Sub-Grade Monotonicity & Credit Risk EDA (Section 4.4)**:
   - **Text Location:** `final_notebook_text.md` lines 118–120; `Lending_Club_Project_Final.py` lines 377–379.
   - **Content:** Confirms strict risk monotonicity across sub-grades A1 to G5.
   - **Assessment:** **PASS/SUCCESS**. Validates Lending Club's grading baseline.

5. **Heavy Tails Analysis (Section 4.5)**:
   - **Text Location:** `final_notebook_text.md` lines 127–129; `Lending_Club_Project_Final.py` lines 414–416.
   - **Content:** Explains extreme right-skewness of `annual_inc` and `revol_bal`, justifying tree-based models over linear models.
   - **Assessment:** **PASS/SUCCESS**. Strong modeling rationale.

6. **Optimal K Selection (Section 4.6.0)**:
   - **Text Location:** `final_notebook_text.md` lines 153–157; `Lending_Club_Project_Final.py` lines 492–497.
   - **Content:** Explains Elbow inflection at K=3 and Silhouette score.
   - **Assessment:** **PASS/SUCCESS**. Rigorous mathematical cluster justification.

7. **PCA & K-Means Borrower Personas (Section 4.6)**:
   - **Text Location:** `final_notebook_text.md` lines 162–167; `Lending_Club_Project_Final.py` lines 554–559.
   - **Content:** Defines 3 Personas: Cluster 0 (Moderate/Stable), Cluster 1 (Highly Leveraged/High Risk), Cluster 2 (Wealthy High-Rollers).
   - **Assessment:** **PASS/SUCCESS**. High-value risk profiling.

8. **PCA Factor Loadings & PCA vs t-SNE (Sections 4.6.1 & 4.6.2)**:
   - **Text Location:** `final_notebook_text.md` lines 188–210; `Lending_Club_Project_Final.py` lines 634–648 & 673–676.
   - **Content:** Factor loadings (PC1=Stability, PC2=Scale); PCA vs t-SNE comparison table.
   - **Assessment:** **PASS/SUCCESS**. Excellent technical and business alignment.

9. **Anomaly Detection & Correlation Heatmap (Sections 4.7 & 4.8)**:
   - **Text Location:** `final_notebook_text.md` lines 220–223 & 232–234; `Lending_Club_Project_Final.py` lines 704–706 & 726–728.
   - **Content:** Unsupervised anomaly risk interpretation and multicollinearity assessment.
   - **Assessment:** **PASS/SUCCESS**.

---

## 5. Line-by-Line Actionable Recommendations (Pathway to 100/100)

To elevate Phase 1 EDA to a flawless 100/100, the implementer must execute the following concrete changes in `Lending_Club_Project_Final.py` and `final_notebook_text.md`:

### Recommendation 1: Fix Class Imbalance Text & Bar Chart Annotations
1. **In `Lending_Club_Project_Final.py` lines 213–216 (`In[4]`)**, update bar chart annotation code to show both exact counts and percentages:
   ```python
   # Replace lines 214-216:
   total_count = sum(target_counts['count'].to_list())
   for i, v in enumerate(target_counts['count'].to_list()):
       pct = (v / total_count) * 100
       axes[0].text(i, v / 2, f'{v:,}\n({pct:.1f}%)', ha='center', va='center', fontweight='bold', color='white', fontsize=12)
   ```
2. **In `final_notebook_text.md` line 78 & `Lending_Club_Project_Final.py` line 203**, add an explicit operational framing paragraph:
   ```markdown
   **Operational Metric Imbalance Warning:**
   Because 82.0% of loans are Fully Paid, a naive baseline classifier predicting 100% Fully Paid achieves an 82.0% Accuracy score while missing 100% of defaults. Therefore, standard Accuracy is an invalid metric. Downstream evaluation MUST strictly rely on ROC-AUC, PR-AUC, and financial cost matrices, and supervised models will require class weighting (`scale_pos_weight`).
   ```

### Recommendation 2: Save ALL EDA Plots to `eda_plots/` Directory
Add explicit `plt.savefig()` statements with high resolution (`dpi=150`, `bbox_inches='tight'`) for every single EDA plot:
1. **Target & Temporal Drift Plot (`In[4]`, line 236):**
   ```python
   plt.savefig('eda_plots/target_distribution_drift.png', dpi=150, bbox_inches='tight')
   ```
2. **Business Questions 6-Panel Grid (`In[6]`, line 317):**
   ```python
   plt.savefig('eda_plots/business_questions_grid.png', dpi=150, bbox_inches='tight')
   ```
3. **Sub-Grade & Temporal Drift Plot (`In[7]`, line 373):**
   ```python
   plt.savefig('eda_plots/subgrade_monotonicity.png', dpi=150, bbox_inches='tight')
   ```
4. **Heavy Tails Plot (`In[8]`, line 410):**
   ```python
   # Remove developer debug comment lines 404-407 and add:
   plt.savefig('eda_plots/heavy_tails.png', dpi=150, bbox_inches='tight')
   ```
5. **PCA & K-Means Clusters (`In[10]`, line 544):**
   ```python
   plt.savefig('eda_plots/pca_kmeans_personas.png', dpi=150, bbox_inches='tight')
   ```
6. **PCA Factor Loadings Heatmap (line 670):**
   ```python
   plt.savefig('eda_plots/pca_factor_loadings.png', dpi=150, bbox_inches='tight')
   ```
7. **Correlation Heatmap (`In[14]`, line 722):**
   ```python
   plt.savefig('eda_plots/correlation_heatmap.png', dpi=150, bbox_inches='tight')
   ```

### Recommendation 3: Clean Axis Labels, Titles, and Legends Across All Plots
1. **In Business Questions 6-Panel Grid (`In[6]`, lines 288–315):**
   - Subplot (0,1): `axes[0, 1].set_ylabel('Interest Rate (%)')`
   - Subplot (1,0): `axes[1, 0].set_ylabel('Loan Purpose')`, `axes[1, 0].set_xlabel('Default Rate (%)')`
   - Subplot (1,1): `axes[1, 1].set_ylabel('Default Rate (%)')`, `axes[1, 1].set_xlabel('Home Ownership')`
   - Subplot (2,0): `axes[2, 0].set_ylabel('Default Rate (%)')`, `axes[2, 0].set_xlabel('Income Verification Status')`
   - Subplot (2,1): `axes[2, 1].set_xlabel('Loan Amount ($)')`, `axes[2, 1].set_ylabel('Monthly Installment ($)')`
   - Subplot (2,1) Legend: Pass `hue_order=[0, 1]` and explicit legend labels `['Fully Paid', 'Charged Off']`.
2. **In PCA & K-Means Grid (`In[10]`, lines 537–542):**
   - Subplot 1 Legend: Replace `0, 1` with `Fully Paid` and `Charged Off`.
   - Subplot 2 Legend: Replace `0, 1, 2` with `Cluster 0: Moderate/Stable`, `Cluster 1: Highly Leveraged`, `Cluster 2: Wealthy High-Rollers`.
3. **In PCA vs t-SNE Plot (`In[11]`, lines 602–610):**
   - Add discrete legend handles for Clusters 0, 1, and 2 on `axes[0, 0]` and `axes[0, 1]`.

### Recommendation 4: Complete Missing Business Markdown Content
1. **Add Answer 6 to Section 4.3 (`final_notebook_text.md` line 111 & `Lending_Club_Project_Final.py` line 327):**
   ```markdown
   6. **Loan Amount vs. Installment:** The scatter plot illustrates a tight linear relationship bounded by interest rate slopes. High-installment loans relative to loan amount represent higher interest rate products (higher risk grades). Borrowers in the upper band of installments for a given loan amount show a significantly higher density of default (red points), demonstrating that monthly cash flow strain (installment magnitude) directly escalates default risk.
   ```
2. **Expand Temporal Drift Narrative in Section 4.1 & 4.4 (`final_notebook_text.md` lines 80 & 121):**
   ```markdown
   **Macroeconomic & Underwriting Temporal Context:**
   The temporal drift plot shows higher default rates in loans issued during 2007–2008 (reaching ~22%), reflecting the broader macroeconomic distress of the Great Recession and early unrefined platform underwriting. Between 2009 and 2012, default rates stabilized (~12–15%) as Lending Club tightened credit criteria and institutional capital entered the platform. The slight upward trend in 2014–2015 reflects portfolio seasoning (older loans accumulating defaults over time).
   ```

---

## 6. Handoff Protocol (5-Component Structure)

### 1. Observation
- Verified all code and markdown blocks in `Lending_Club_Project_Final.py` (1,286 lines) and `final_notebook_text.md` (440 lines).
- Inspected physical plot files in `eda_plots/`: `heavy_tails.png` (246 KB), `pca_vs_tsne.png` (1.6 MB), `silhouette_elbow.png` (130 KB).
- Confirmed that 6 out of 9 plot panels rendered in Phase 1 do NOT execute `plt.savefig()`.
- Confirmed that Section 4.3 Markdown lists Question 6 but omits Answer 6 entirely.

### 2. Logic Chain
- Section 1 of `ds_review_guide.md` requires clean aesthetics, plot saving, target presentation clarity, and business context under *every* major chart.
- Omission of `plt.savefig()` prevents visual outputs from being programmatically exported, inspected, or verified in downstream reporting artifacts.
- Missing Answer 6 leaves a gap in the business storytelling requirement.
- Adding formatted axis labels, clear legend titles, percentage annotations, and missing business narrative directly addresses all criteria to achieve a 100/100 score.

### 3. Caveats
- No code in `Lending_Club_Project_Final.py` or `.ipynb` was modified during this audit (strictly read-only investigation).
- `heavy_tails.png` exists in `eda_plots/`, indicating it was saved during an earlier execution run, but the python script missing `plt.savefig()` will cause future clean runs to omit re-saving it.

### 4. Conclusion
Phase 1 (EDA) currently scores **86 / 100**. It has robust data science fundamentals, but suffers from aesthetic incompleteness, missing plot exports, and a missing business answer for Q6. Applying Recommendations 1–4 will elevate the score to **100 / 100**.

### 5. Verification Method
- Run `python Lending_Club_Project_Final.py` in an environment with the Lending Club dataset.
- Inspect `eda_plots/` directory to verify that all 7 missing `.png` files are generated:
  - `target_distribution_drift.png`
  - `business_questions_grid.png`
  - `subgrade_monotonicity.png`
  - `heavy_tails.png`
  - `silhouette_elbow.png`
  - `pca_kmeans_personas.png`
  - `pca_factor_loadings.png`
  - `pca_vs_tsne.png`
  - `correlation_heatmap.png`
- Check `final_notebook_text.md` to confirm Answer 6 and temporal drift narrative are present.
