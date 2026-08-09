# Phase 5 Audit Report: Forensic Evaluation & Explainability

**Project:** Lending Club Data Science Project (`c:\Users\yonat\.gemini\antigravity\scratch\lending_club_ds`)  
**Auditor:** Investigator Agent 5 (Phase 5 Auditor)  
**Date:** July 27, 2026  
**Audit Standard:** `ds_review_guide.md` Section 5 & `ORIGINAL_REQUEST.md` R5  

---

## 1. Executive Summary & Audit Score

### Overall Grade: 78 / 100 (Pass with Critical Forensic Gaps)

The Phase 5 evaluation of the Lending Club Data Science project demonstrates a strong foundation in credit risk metric selection—correctly rejecting dummy baseline accuracy (82%) in favor of **ROC-AUC** and **PR-AUC** suitable for class-imbalanced data (18.0% default rate). However, an exhaustive audit against `ds_review_guide.md` Section 5 reveals **critical forensic inconsistencies** between text claims in Markdown/comments and printed execution outputs/plots, as well as missing explainability artifacts (SHAP dependence and waterfall plots).

| Evaluation Category | Status | Score | Core Summary |
|---|---|---|---|
| **1. Accurate Metric Selection** | **PASS / SUCCESS** | 95 / 100 | Relies on ROC-AUC & PR-AUC for class imbalance. Explicitly documents why Accuracy (82%) is misleading. Minor ambiguity between F1, F2, and Profit threshold metrics. |
| **2. 100% Forensic Consistency** | **CRITICAL FLAW** | 60 / 100 | Severe numerical and qualitative contradictions between Markdown text claims, printed cell outputs, and displayed plot labels (e.g., LightGBM AUC claims 0.7272 vs plot 0.7111 vs cell output 0.7140; MLP AUC claims vs printed 0.7303; SHAP feature ranking claims vs actual plot ranks). |
| **3. Model Explainability** | **PARTIAL PASS / GAP** | 78 / 100 | Implements clean SHAP Beeswarm and Bar Importance plots. However, lacks SHAP Dependence plots, SHAP Waterfall plots for individual credit decisions, and omits FCRA/business compliance context. |

---

## 2. Detailed Forensic Audit Findings

### Requirement 1: Accurate Metric Selection

#### Pass / Success Items
1. **Class Imbalance Recognition (`final_notebook_text.md` lines 74–79, `Lending_Club_Project_Final.py` lines 185–190)**:
   - The project explicitly identifies the 82.0% Fully Paid vs. 18.0% Charged Off target distribution (class imbalance ratio 4.55:1).
   - Text explicitly warns: *"Baseline accuracy of 82.0% achieved by predicting all loans as 'Fully Paid'... misleading accuracy."*
2. **Imbalance-Robust Evaluation Metrics (`Lending_Club_Project_Final.py` lines 850–852, 920, 962, 1111–1112)**:
   - Primary evaluation across LightGBM, Optuna tuning, Walk-Forward CV (`TimeSeriesSplit`), and MLP Neural Network relies on **ROC-AUC** (`roc_auc_score`) and **PR-AUC** (`average_precision_score`).

#### Flaws & Ambiguities
1. **Threshold Metric Contradiction (`Lending_Club_Project_Final.py` line 857 vs line 864)**:
   - Line 857 states: *"To translate probabilities into hard predictions, we optimize the decision threshold by maximizing the F1-Score."*
   - However, cell 19 (`model_plots/threshold_optimization.png`) displays **F2-Score ($\beta=2$)** (Optimal = 0.30) and **Net Expected Profit** (Optimal = 0.59), without presenting an F1-Score curve.

---

### Requirement 2: 100% Forensic Consistency (CRITICAL AUDIT)

#### Flaw 2.1: Discrepancy in Reported LightGBM OOT ROC-AUC Scores (0.7272 vs 0.7140 vs 0.7111)
* **Location in Code/Notebook**: 
  - `Lending_Club_Project_Final.py`: Lines 1009, 1017, 1117, 1146, 1196, 1199, 1244, 1252
  - `final_notebook_text.md`: Lines 369, 391, 399
  - `Lending_Club_Project_Final.ipynb`: Cells 23, 24, 25 and `model_plots/model_comparison.png`
* **Observation**:
  - **Text Claim (Markdown & Comments)**: Claims Standalone LightGBM achieves an ROC-AUC of **`0.7272`** (e.g., text.md line 369: *"Standalone LightGBM model achieves an ROC-AUC of 0.7272"*).
  - **Printed Execution Output in Cell 23 & 24**: Printed output in Cell 23 (line 3963) and Cell 24 (line 7988) reports **`LightGBM Test ROC-AUC: 0.7140`**.
  - **Displayed Plot Image (`model_plots/model_comparison.png`)**: Labels LightGBM Optuna as **`AUC = 0.7111`** and **`PR-AUC = 0.3932`**.
  - **Printed Execution Output in Cell 25**: Printed output in Cell 25 (line 8181) reports **`Final LightGBM ROC-AUC on Held-out OOT Test Set: 0.7272`**.
* **Logic Chain**:
  - In Cell 23 and 24, `oot_auc` defaulted to `0.7140`.
  - In `model_comparison.png`, the curve shows `0.7111`.
  - In Cell 25, the final model was retrained on full `X_train`, producing `0.7272`.
  - Presenting three different AUC numbers (`0.7111`, `0.7140`, `0.7272`) across plots, intermediate prints, and final summary text without explicitly reconciling them violates the 100% Forensic Consistency rule.

#### Flaw 2.2: Contradictory Claim Regarding Deep Learning (MLP) Performance
* **Location in Code/Notebook**: 
  - `Lending_Club_Project_Final.py`: Lines 1114–1118, 1143–1148, 1150–1172
  - `final_notebook_text.md`: Lines 330–360
  - `Lending_Club_Project_Final.ipynb`: Cell 24 printed stdout (lines 7985–7989) & `model_plots/mlp_training.png`
* **Observation**:
  - **Printed Execution Output in Cell 24**:
    ```text
    ==================================================
    MLP Test ROC-AUC:     0.7303
    MLP Test PR-AUC:      0.4069
    LightGBM Test ROC-AUC: 0.7140
    ==================================================
    ```
  - **Text Claim in Markdown right below Cell 24 (`final_notebook_text.md` lines 336–358)**:
    *"The MLP's underperformance is not a bug — it's a feature of the problem domain... Why Tree-Based Models Outperform Deep Learning on Tabular Data... This MLP experiment validates our architectural choice..."*
* **Logic Chain**:
  - The actual printed output of Cell 24 shows MLP Test ROC-AUC (**0.7303**) and PR-AUC (**0.4069**) **outperforming** LightGBM Test ROC-AUC (**0.7140** / **0.7272**).
  - The text in Markdown immediately asserts that the MLP underperformed LightGBM and writes a 25-line theoretical justification citing Grinsztajn et al. (2022) for why trees beat deep learning.
  - This is a direct forensic contradiction between printed empirical execution results and free-text narrative.

#### Flaw 2.3: Hallucinated SHAP Feature Importance Ranking in Text
* **Location in Code/Notebook**: 
  - `Lending_Club_Project_Final.py`: Lines 887–900
  - `final_notebook_text.md`: Lines 296–301
  - `model_plots/shap_beeswarm.png` & `model_plots/shap_importance.png`
* **Observation**:
  - **Text Claim (`final_notebook_text.md` line 300 & script line 900)**:
    *"The SHAP summary clearly shows that `fico_range_low` and `dti` (Debt-to-Income) are the most dominant predictors."*
  - **Actual Printed Plots (`shap_importance.png` & `shap_beeswarm.png`)**:
    - **#1 Dominant Feature**: `sub_grade_num` (Mean \|SHAP Value\| $\approx 0.395$, nearly double any other feature).
    - **#2 Dominant Feature**: `term_num` (Mean \|SHAP Value\| $\approx 0.205$).
    - **#3 Dominant Feature**: `adjusted_dti` (Mean \|SHAP Value\| $\approx 0.175$).
    - **#4 Dominant Feature**: `fico_avg` (Mean \|SHAP Value\| $\approx 0.130$).
* **Logic Chain**:
  - The text completely omits `sub_grade_num` (#1) and `term_num` (#2) from its explanation paragraph.
  - It falsely claims `fico_range_low` and `dti` are the top 2 features, whereas they are actually rank #4 and rank #3 (and named `fico_avg` and `adjusted_dti` in the plot).

#### Flaw 2.4: Directional Mismatch in Threshold Optimization Text
* **Location in Code/Notebook**: 
  - `Lending_Club_Project_Final.py`: Line 867
  - `final_notebook_text.md`: Line 280
  - `model_plots/threshold_optimization.png`
* **Observation**:
  - **Text Claim**: *"By shifting the decision threshold from the default 0.50 down to the optimal point, we correctly identify more potential defaults, drastically increasing the net expected profit..."*
  - **Actual Plot (`threshold_optimization.png`)**:
    - Left Subplot ("Threshold vs. Net Profit"): Optimal decision threshold for Net Profit is **0.59** (shifted **UP** from 0.50, not down).
    - Middle Subplot ("Threshold vs. F2 Score"): Optimal threshold for F2 is **0.30** (shifted **DOWN** from 0.50).
* **Logic Chain**:
  - Shifting down to 0.30 maximizes recall/F2, but yields a lower net profit (~$180M vs. ~$600M at 0.59).
  - Claiming that shifting threshold *down* increases *net profit* contradicts the plotted curve where profit peaks at 0.59.

#### Flaw 2.5: Detached Static Plot Files & Missing Pre-Computation Scripts
* **Location in Code/Notebook**: 
  - `Lending_Club_Project_Final.py`: Lines 836–897
  - Missing files: `modeling.py`, `model_results.json`
* **Observation**:
  - Script comments state: `# Results are pre-computed in modeling.py and saved to model_results.json.`
  - The notebook renders key evaluation figures using static image calls: `display(Image(filename='model_plots/model_comparison.png'))`.
  - Neither `modeling.py` nor `model_results.json` exists in the repository.
* **Logic Chain**:
  - Static display of pre-computed PNGs prevents independent verification and dynamic execution of Phase 5 plots inside the notebook pipeline.

#### Flaw 2.6: Omitted Brier Score Calibration Metrics in Text
* **Location in Code/Notebook**: 
  - `Lending_Club_Project_Final.py`: Line 879
  - `final_notebook_text.md`: Line 288
  - `model_plots/calibration_curves.png`
* **Observation**:
  - `calibration_curves.png` displays explicit Brier score improvements: Uncalibrated Brier = **0.1978**, Calibrated (Isotonic) Brier = **0.1555**.
  - The text markdown ignores Brier scores completely and merely provides a generic qualitative statement.

---

### Requirement 3: Model Explainability & SHAP Analysis

#### Pass / Success Items
1. **Global SHAP Beeswarm Plot (`model_plots/shap_beeswarm.png`)**:
   - High visual quality, clearly showing feature impact directionality (e.g., lower FICO / `fico_avg` blue dots push SHAP positive toward default).
2. **Global SHAP Feature Importance Bar Plot (`model_plots/shap_importance.png`)**:
   - Correctly ranks features by Mean \|SHAP Value\| across 20+ features.

#### Flaws & Gaps
1. **Missing SHAP Dependence Plots**:
   - No dependence plots exist to demonstrate non-linear interactions (e.g., `adjusted_dti` vs. `fico_avg`).
2. **Missing Individual SHAP Waterfall / Force Plots**:
   - No local explainability artifacts are generated for individual borrower credit decisioning (e.g., why Applicant #1042 was rejected).
3. **Missing Regulatory & Business Credit Decisioning Framework**:
   - The SHAP analysis does not explain how credit underwriters turn SHAP feature impacts into FCRA-compliant Adverse Action reason codes.
   - Does not address the business implications of `sub_grade_num` being the #1 predictor (i.e., whether the model adds incremental underwriting signal beyond Lending Club's legacy grade).

---

## 3. Actionable Recommendations for 100/100 Forensic Grade

To elevate the Phase 5 evaluation to a flawless 100/100 Kaggle-grade audit, implement the following line-by-line modifications:

### 1. Reconcile ROC-AUC Metric Claims Across Text & Notebook Outputs
- **Target File**: `Lending_Club_Project_Final.py` (lines 1009, 1146, 1195–1199, 1244) and `final_notebook_text.md` (lines 369, 391, 399).
- **Modification**:
  - Update `final_notebook_text.md` and script comments to explicitly present the progression of ROC-AUC scores across experimental stages:
    - Baseline Logistic Regression: **0.6945** (PR-AUC: 0.3736)
    - LightGBM Optuna (CV Mean): **0.6983 $\pm$ 0.0178** (OOT initial: **0.7111** / PR-AUC: 0.3932)
    - Final Standalone LightGBM (full retrain): **0.7272** (OOT test)
    - MLP Neural Network: **0.7303** (PR-AUC: 0.4069)

### 2. Correct the MLP vs. LightGBM Deep Learning Narrative
- **Target File**: `Lending_Club_Project_Final.py` (lines 1143–1172) and `final_notebook_text.md` (lines 330–360).
- **Modification**:
  - Rewrite the assessment section to reflect true execution outputs:
    > *"Empirical results show that the MLP Neural Network achieves a raw Test ROC-AUC of 0.7303 (PR-AUC 0.4069), slightly higher than Standalone LightGBM (0.7272). However, Standalone LightGBM is selected as the primary production model due to: (1) Native handling of missing values and categoricals without imputation leakage, (2) Exact, tree-based SHAP explainability required for FCRA credit compliance, and (3) Operational simplicity and lower latency."*

### 3. Align SHAP Feature Importance Text with Actual Plot Data
- **Target File**: `Lending_Club_Project_Final.py` (lines 891, 900) and `final_notebook_text.md` (lines 296, 300).
- **Modification**:
  - Update text to match `shap_importance.png`:
    > *"The SHAP summary reveals that `sub_grade_num` (Mean |SHAP| = 0.395) and `term_num` (Mean |SHAP| = 0.205) are the primary global drivers of risk prediction, followed by `adjusted_dti` (0.175) and `fico_avg` (0.130). This confirms that Lending Club's pre-origination credit rating and loan duration carry the strongest predictive weight, while financial stress indicators (`adjusted_dti`) provide critical non-linear adjustments."*

### 4. Fix Threshold Optimization Terminology & Directionality
- **Target File**: `Lending_Club_Project_Final.py` (lines 857, 867) and `final_notebook_text.md` (lines 276, 280).
- **Modification**:
  - Update text:
    > *"We evaluate decision thresholds under two objectives: (1) Maximizing F2-Score ($\beta=2$) to prioritize default recall, which lowers the optimal threshold to 0.30 (Recall = 91.2%), and (2) Maximizing Net Portfolio Profit, which raises the optimal decision threshold to 0.59 (Expected Profit = $600M)."*

### 5. Generate Live Reproducible SHAP & Evaluation Code
- **Target File**: `Lending_Club_Project_Final.py` (Section 6.2).
- **Modification**: Add dynamic SHAP generation, dependence, waterfall, and calibration code directly into the script execution flow:

```python
# --- Dynamic SHAP & Explainability Implementation ---
import shap
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import brier_score_loss

# 1. SHAP Explainer
explainer = shap.TreeExplainer(final_lgbm)
shap_values = explainer.shap_values(X_test.iloc[:2000])

# 2. SHAP Dependence Plot (adjusted_dti vs. fico_range_low)
plt.figure(figsize=(8, 5))
shap.dependence_plot("adjusted_dti", shap_values[1] if isinstance(shap_values, list) else shap_values, 
                    X_test.iloc[:2000], display_features=X_test.iloc[:2000], show=False)
plt.title("SHAP Dependence: Adjusted DTI vs. FICO", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('model_plots/shap_dependence.png', dpi=150, bbox_inches='tight')
plt.close()

# 3. SHAP Waterfall Plot (Individual Applicant Decision)
plt.figure(figsize=(8, 6))
shap.plots.waterfall(shap.Explanation(values=shap_values[1][0] if isinstance(shap_values, list) else shap_values[0],
                                       base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
                                       data=X_test.iloc[0],
                                       feature_names=X_test.columns), show=False)
plt.tight_layout()
plt.savefig('model_plots/shap_waterfall.png', dpi=150, bbox_inches='tight')
plt.close()

# 4. Probability Calibration & Brier Score Calculation
prob_true, prob_pred = calibration_curve(y_test, lgbm_preds, n_bins=10)
brier_raw = brier_score_loss(y_test, lgbm_preds)

iso_cal = CalibratedClassifierCV(final_lgbm, cv='prefit', method='isotonic')
iso_cal.fit(X_test, y_test)
cal_preds = iso_cal.predict_proba(X_test)[:, 1]
brier_cal = brier_score_loss(y_test, cal_preds)

print(f"Uncalibrated Brier Score: {brier_raw:.4f}")
print(f"Calibrated (Isotonic) Brier Score: {brier_cal:.4f}")
```

---

## 4. Verification Method

To verify the audit findings and proposed recommendations independently:

1. **Verify Metric Consistency**:
   - Inspect printed outputs of `Lending_Club_Project_Final.ipynb` Cell 23, 24, 25 and compare with `final_notebook_text.md` lines 330–400.
2. **Verify Plot Labels**:
   - Open `model_plots/model_comparison.png` to confirm LR AUC=0.6945 and LightGBM AUC=0.7111.
   - Open `model_plots/shap_importance.png` to confirm `sub_grade_num` is rank #1 (0.395) and `term_num` is rank #2 (0.205).
   - Open `model_plots/threshold_optimization.png` to confirm Net Profit optimal threshold = 0.59 and F2 optimal threshold = 0.30.
   - Open `model_plots/calibration_curves.png` to confirm Brier score values = 0.1978 and 0.1555.
3. **Verify Script Execution**:
   - Run the dynamic SHAP snippet above to confirm reproduction of `shap_dependence.png` and `shap_waterfall.png`.
