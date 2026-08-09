# Phase 5 QA Verification & Forensic Audit Report: Evaluation & Explainability

**Project:** Lending Club Data Science Project (`c:\Users\yonat\.gemini\antigravity\scratch\lending_club_ds`)  
**QA Reviewer:** QA Reviewer Agent 5 (`reviewer_phase5`)  
**Investigator:** Investigator Agent 5 (`explorer_phase5`)  
**Date:** July 27, 2026  
**Audit Standard:** `ds_review_guide.md` Section 5 & `ORIGINAL_REQUEST.md` R5  

---

## 1. Executive Summary & Verdict

### Final Verdict: REQUEST_CHANGES (Current Grade: 78 / 100)

An independent, zero-trust QA cross-verification of the Phase 5 evaluation artifacts produced by `explorer_phase5` confirms that while the project establishes a sound methodological baseline in credit risk metric selection (rejecting accuracy in favor of ROC-AUC and PR-AUC), it suffers from **severe forensic inconsistencies** between free-text markdown narrative, printed execution outputs in code cells, and displayed plot graphics. Additionally, key local explainability artifacts (SHAP dependence and waterfall plots) and FCRA regulatory compliance frameworks are missing.

| Evaluation Dimension | Weight | Status | Score | Core Finding |
|---|---|---|---|---|
| **1. Accurate Metric Selection** | 30% | **PASS** | 95 / 100 | Relies on ROC-AUC & PR-AUC for class-imbalanced data (18.0% default rate). Explicitly rejects 82.0% accuracy baseline. |
| **2. 100% Forensic Consistency** | 40% | **CRITICAL FLAW** | 60 / 100 | Severe numerical & qualitative contradictions between text claims, cell printed stdout, and plot images (AUC claims 0.7272 vs printed 0.7140 vs plot 0.7111; MLP claim contradiction; SHAP rank hallucination; threshold shift direction error). |
| **3. Model Explainability** | 30% | **PARTIAL PASS** | 78 / 100 | Global SHAP Beeswarm and Bar plots present. Missing SHAP Dependence, SHAP Waterfall plots, and FCRA Adverse Action compliance narrative. |
| **OVERALL GRADE** | **100%** | **REQUEST_CHANGES** | **78 / 100** | **Pass with Critical Forensic Gaps. Remediation required to achieve 100/100.** |

---

## 2. Detailed Forensic Cross-Verification Breakdown

### Requirement 1: Accurate Metric Selection (Score: 95/100)

#### Verified Successes
1. **Class Imbalance Identification**:
   - `final_notebook_text.md` (lines 74–79) & `Lending_Club_Project_Final.py` (lines 185–190) correctly identify the 82.0% Fully Paid vs. 18.0% Charged Off target distribution (imbalance ratio 4.55:1).
   - Text explicitly notes: *"Baseline accuracy of 82.0% achieved by predicting all loans as 'Fully Paid'... misleading accuracy."*
2. **Imbalance-Robust Evaluation Metrics**:
   - `Lending_Club_Project_Final.py` lines 850, 920, 962, 1111–1112 verify that LightGBM, Optuna tuning, Walk-Forward CV (`TimeSeriesSplit`), and MLP evaluation rely on **ROC-AUC** (`roc_auc_score`) and **PR-AUC** (`average_precision_score`).

#### Minor Ambiguities
1. **Threshold Metric Ambiguity**:
   - `Lending_Club_Project_Final.py` line 857 states threshold optimization maximizes F1-Score, whereas Cell 19 (`model_plots/threshold_optimization.png`) displays **F2-Score ($\beta=2$)** (peak at 0.30) and **Net Expected Profit** (peak at 0.59).

---

### Requirement 2: 100% Forensic Consistency (Score: 60/100 - CRITICAL FLAW)

#### Flaw 2.1: Discrepancy in Reported LightGBM OOT ROC-AUC Scores (0.7272 vs 0.7140 vs 0.7111)
- **Locations**:
  - `final_notebook_text.md`: Lines 369, 391, 399
  - `Lending_Club_Project_Final.py`: Lines 1009, 1017, 1196, 1199, 1244, 1252
  - `Lending_Club_Project_Final.ipynb`: Cells 23, 24, 25 and `model_plots/model_comparison.png`
- **Verified Evidence**:
  - **Text Claim (Markdown & Comments)**: Claims Standalone LightGBM achieves ROC-AUC of **`0.7272`** (e.g., text line 369: *"Standalone LightGBM model achieves an ROC-AUC of 0.7272"*).
  - **Printed Execution Stdout (Cells 23 & 24)**: Printed stdout in Cell 23 (line 3963) and Cell 24 (line 7988) reports **`LightGBM Test ROC-AUC: 0.7140`**.
  - **Plot Graphic (`model_plots/model_comparison.png`)**: Labels LightGBM Optuna as **`AUC = 0.7111`** and **`PR-AUC = 0.3932`**.
  - **Printed Execution Stdout (Cell 25)**: Cell 25 stdout (line 8181) reports **`Final LightGBM ROC-AUC on Held-out OOT Test Set: 0.7272`**.
- **Forensic Assessment**: Presenting three distinct AUC figures (`0.7111`, `0.7140`, `0.7272`) across plots, intermediate prints, and final text summary without explicit narrative reconciliation violates `ds_review_guide.md` Section 5.

#### Flaw 2.2: Contradictory Claim Regarding Deep Learning (MLP) Performance
- **Locations**:
  - `Lending_Club_Project_Final.py`: Lines 1114–1118, 1143–1148, 1150–1172
  - `final_notebook_text.md`: Lines 330–360
  - `Lending_Club_Project_Final.ipynb`: Cell 24 printed stdout (lines 7985–7989) & `model_plots/mlp_training.png`
- **Verified Evidence**:
  - **Printed Execution Stdout of Cell 24**:
    ```text
    ==================================================
    MLP Test ROC-AUC:     0.7303
    MLP Test PR-AUC:      0.4069
    LightGBM Test ROC-AUC: 0.7140
    ==================================================
    ```
  - **Text Claim in Markdown directly below Cell 24 (`final_notebook_text.md` lines 336–358)**:
    *"The MLP's underperformance is not a bug — it's a feature of the problem domain... Why Tree-Based Models Outperform Deep Learning on Tabular Data... This MLP experiment validates our architectural choice..."*
- **Forensic Assessment**: Direct integrity contradiction. Printed output proves MLP (**0.7303**) outperformed LightGBM (**0.7140** / **0.7272**), yet free-text markdown claims MLP underperformed and provides a 25-line theoretical justification citing Grinsztajn et al. (2022).

#### Flaw 2.3: Hallucinated SHAP Feature Importance Ranking in Text
- **Locations**:
  - `Lending_Club_Project_Final.py`: Line 900
  - `final_notebook_text.md`: Line 300
  - `model_plots/shap_beeswarm.png` & `model_plots/shap_importance.png`
- **Verified Evidence**:
  - **Text Claim**: *"The SHAP summary clearly shows that `fico_range_low` and `dti` (Debt-to-Income) are the most dominant predictors."*
  - **Actual Plotted Data (`shap_importance.png`)**:
    - **Rank #1**: `sub_grade_num` (Mean \|SHAP Value\| $\approx 0.395$, double any other feature).
    - **Rank #2**: `term_num` (Mean \|SHAP Value\| $\approx 0.205$).
    - **Rank #3**: `adjusted_dti` (Mean \|SHAP Value\| $\approx 0.175$).
    - **Rank #4**: `fico_avg` (Mean \|SHAP Value\| $\approx 0.130$).
- **Forensic Assessment**: Text narrative completely omits rank #1 (`sub_grade_num`) and rank #2 (`term_num`), hallucinating that `fico_range_low` and `dti` are top 1 & 2.

#### Flaw 2.4: Directional Mismatch in Threshold Optimization Text
- **Locations**:
  - `Lending_Club_Project_Final.py`: Line 867
  - `final_notebook_text.md`: Line 280
  - `model_plots/threshold_optimization.png`
- **Verified Evidence**:
  - **Text Claim**: *"By shifting the decision threshold from the default 0.50 down to the optimal point, we correctly identify more potential defaults, drastically increasing the net expected profit..."*
  - **Actual Plotted Graphic (`threshold_optimization.png`)**:
    - Left Subplot ("Threshold vs. Net Profit"): Optimal threshold for Net Profit is **0.59** (shifted **UP** from 0.50).
    - Middle Subplot ("Threshold vs. F2 Score"): Optimal threshold for F2 is **0.30** (shifted **DOWN** from 0.50).
- **Forensic Assessment**: Shifting threshold down to 0.30 maximizes recall/F2, but reduces net profit (~$180M vs. ~$600M at 0.59). Claiming that shifting threshold *down* increases *net profit* contradicts the plotted curve.

#### Flaw 2.5: Detached Static Plot Files & Missing Pre-Computation Scripts
- **Locations**: `Lending_Club_Project_Final.py` lines 836–897.
- **Verified Evidence**: Script comments state `# Results are pre-computed in modeling.py and saved to model_results.json`. Plots are loaded statically via `display(Image(filename='model_plots/...'))`. Neither `modeling.py` nor `model_results.json` exists in the repository.

#### Flaw 2.6: Omitted Brier Score Calibration Metrics in Text
- **Locations**: `Lending_Club_Project_Final.py` line 879, `final_notebook_text.md` line 288, `model_plots/calibration_curves.png`.
- **Verified Evidence**: `calibration_curves.png` displays explicit Brier score improvements: Uncalibrated Brier = **0.1978**, Calibrated (Isotonic) Brier = **0.1555**. Markdown text ignores Brier scores completely.

---

### Requirement 3: Model Explainability & SHAP Analysis (Score: 78/100 - GAP)

#### Verified Successes
1. **Global SHAP Beeswarm (`shap_beeswarm.png`)**: Clean visualization of feature directionality (lower FICO scores push risk higher).
2. **Global SHAP Importance Bar (`shap_importance.png`)**: Ranks features by mean absolute SHAP value.

#### Verified Gaps
1. **Missing SHAP Dependence Plots**: No interaction plots exist for non-linear feature pairs (e.g., `adjusted_dti` vs. `fico_avg`).
2. **Missing SHAP Waterfall / Force Plots**: No local explainability plots exist for individual borrower credit decisions.
3. **Missing FCRA Regulatory Compliance Narrative**: No framework is provided for mapping SHAP feature attributions into Adverse Action reason codes under the Fair Credit Reporting Act (FCRA).

---

## 3. Actionable Recommendations for 100/100 Forensic Grade

To elevate the Phase 5 evaluation to a flawless **100 / 100** Kaggle-grade audit, implement the following line-by-line modifications in `Lending_Club_Project_Final.py` and `final_notebook_text.md`:

### Recommendation 1: Reconcile ROC-AUC Metric Claims Across Text & Outputs
- **Target**: `Lending_Club_Project_Final.py` (lines 1009, 1146, 1195–1199, 1244) & `final_notebook_text.md` (lines 369, 391, 399).
- **Modification**: Update text to present the step-by-step metric progression across experimental stages:
  - Baseline Logistic Regression: **0.6945** (PR-AUC: 0.3736)
  - LightGBM Optuna CV Mean: **0.6983 $\pm$ 0.0178** (OOT initial plot: **0.7111** / PR-AUC: 0.3932)
  - LightGBM Intermediate OOT (fast train): **0.7140**
  - Final Standalone LightGBM (full retrain): **0.7272** (OOT test)
  - MLP Neural Network: **0.7303** (PR-AUC: 0.4069)

### Recommendation 2: Correct Deep Learning (MLP) Performance Narrative
- **Target**: `Lending_Club_Project_Final.py` (lines 1143–1172) & `final_notebook_text.md` (lines 330–360).
- **Modification**: Replace the underperformance narrative with:
  > *"Empirical execution results demonstrate that the MLP Neural Network achieves a raw Test ROC-AUC of 0.7303 (PR-AUC 0.4069), slightly outperforming Standalone LightGBM (0.7272). However, Standalone LightGBM is selected as the primary production model due to: (1) Native handling of categorical features and missing values without data imputation leakage, (2) Exact, tree-based SHAP explainability required for FCRA credit compliance, and (3) Operational efficiency and lower inference latency."*

### Recommendation 3: Align SHAP Feature Importance Text with Plot Data
- **Target**: `Lending_Club_Project_Final.py` (line 900) & `final_notebook_text.md` (line 300).
- **Modification**: Replace hallucinated text with:
  > *"The SHAP summary reveals that `sub_grade_num` (Mean |SHAP| = 0.395) and `term_num` (Mean |SHAP| = 0.205) are the primary global drivers of default risk prediction, followed by `adjusted_dti` (0.175) and `fico_avg` (0.130). This confirms that Lending Club's pre-origination credit rating and loan duration carry the strongest predictive weight, while financial leverage (`adjusted_dti`) and credit history (`fico_avg`) provide critical non-linear risk adjustments."*

### Recommendation 4: Fix Threshold Optimization Terminology & Directionality
- **Target**: `Lending_Club_Project_Final.py` (line 867) & `final_notebook_text.md` (lines 276–280).
- **Modification**: Update text to state:
  > *"Decision threshold optimization is evaluated under two distinct business objectives: (1) Maximizing F2-Score ($\beta=2$) to prioritize default recall, which lowers the decision threshold to 0.30 (Recall = 91.2%), and (2) Maximizing Net Portfolio Profit, which raises the decision threshold to 0.59 (Expected Net Profit = $600M)."*

### Recommendation 5: Live Reproducible SHAP, Waterfall, Dependence & Brier Calibration Code
- **Target**: `Lending_Club_Project_Final.py` (Section 6.2).
- **Modification**: Add the following executable Python block:

```python
# --- Dynamic SHAP & Probability Calibration Code ---
import shap
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import brier_score_loss

# 1. SHAP Tree Explainer
explainer = shap.TreeExplainer(final_lgbm)
shap_values = explainer.shap_values(X_test.iloc[:2000])

# 2. SHAP Dependence Plot (adjusted_dti vs fico_avg)
plt.figure(figsize=(8, 5))
shap.dependence_plot("adjusted_dti", shap_values[1] if isinstance(shap_values, list) else shap_values, 
                    X_test.iloc[:2000], display_features=X_test.iloc[:2000], show=False)
plt.title("SHAP Dependence: Adjusted DTI vs. FICO Score", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('model_plots/shap_dependence.png', dpi=150, bbox_inches='tight')
plt.close()

# 3. SHAP Waterfall Plot (Individual Applicant Decision)
plt.figure(figsize=(8, 6))
shap.plots.waterfall(shap.Explanation(
    values=shap_values[1][0] if isinstance(shap_values, list) else shap_values[0],
    base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
    data=X_test.iloc[0],
    feature_names=X_test.columns
), show=False)
plt.title("SHAP Waterfall: Borrower #1 Adverse Action Explanation", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('model_plots/shap_waterfall.png', dpi=150, bbox_inches='tight')
plt.close()

# 4. Probability Calibration & Brier Score Calculation
brier_raw = brier_score_loss(y_test, lgbm_preds)

iso_cal = CalibratedClassifierCV(final_lgbm, cv='prefit', method='isotonic')
iso_cal.fit(X_test, y_test)
cal_preds = iso_cal.predict_proba(X_test)[:, 1]
brier_cal = brier_score_loss(y_test, cal_preds)

print(f"Uncalibrated Brier Score: {brier_raw:.4f}")
print(f"Calibrated (Isotonic) Brier Score: {brier_cal:.4f}")
```

### Recommendation 6: FCRA Regulatory Adverse Action Compliance Framework
- **Target**: `final_notebook_text.md` (Section 6.4).
- **Modification**: Include an explicit FCRA compliance narrative:
  > *"Under the Fair Credit Reporting Act (15 U.S.C. § 1681 et seq.), lenders are legally required to provide consumers with principal reason codes when taking adverse credit action. By extracting top negative SHAP feature contributions for individual rejected applicants (e.g., Applicant #1), the model automatically generates FCRA-compliant Adverse Action notices (e.g., Reason 1: High Sub-Grade Risk [`sub_grade_num`], Reason 2: High Debt-to-Income [`adjusted_dti`], Reason 3: Insufficient FICO Score [`fico_avg`])."*

---

## 4. Verification Method

To independently verify these audit findings and proposed fixes:

1. **Verify Cell Stdout & Text Claims**:
   - Inspect `Lending_Club_Project_Final.ipynb` Cell 24 stdout (lines 7985–7989) for MLP AUC = 0.7303 vs LightGBM AUC = 0.7140.
   - Inspect `Lending_Club_Project_Final.ipynb` Cell 25 stdout (line 8181) for Final LightGBM AUC = 0.7272.
2. **Inspect Graphic Image Files**:
   - `model_plots/model_comparison.png` for LightGBM Optuna AUC = 0.7111.
   - `model_plots/shap_importance.png` for `sub_grade_num` rank #1 (0.395) and `term_num` rank #2 (0.205).
   - `model_plots/threshold_optimization.png` for Net Profit optimal threshold = 0.59 and F2 optimal threshold = 0.30.
   - `model_plots/calibration_curves.png` for Brier scores = 0.1978 & 0.1555.
3. **Execute Live Code**:
   - Run the dynamic SHAP snippet in Recommendation 5 to verify reproduction of `shap_dependence.png` and `shap_waterfall.png`.
