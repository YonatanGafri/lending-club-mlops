# Handoff Report: Phase 5 Forensic Evaluation & Explainability Audit

**Working Directory:** `c:\Users\yonat\.gemini\antigravity\scratch\lending_club_ds\.agents\explorer_phase5`  
**Target Project Directory:** `c:\Users\yonat\.gemini\antigravity\scratch\lending_club_ds`  
**Report File:** `c:\Users\yonat\.gemini\antigravity\scratch\lending_club_ds\.agents\explorer_phase5\report.md`  

---

## 1. Observation

Direct observations from auditing `Lending_Club_Project_Final.py`, `final_notebook_text.md`, `Lending_Club_Project_Final.ipynb`, `ds_review_guide.md` Section 5, and files in `model_plots/`:

1. **Metric Selection**:
   - Class imbalance: 82.0% Fully Paid vs 18.0% Charged Off (`Lending_Club_Project_Final.py` line 186, `final_notebook_text.md` line 75).
   - Standard accuracy (82%) is explicitly flagged as misleading in text markdown (`final_notebook_text.md` lines 75-78).
   - Evaluation across models utilizes ROC-AUC (`roc_auc_score`) and PR-AUC (`average_precision_score`) (`Lending_Club_Project_Final.py` lines 850, 920, 962, 1111-1112).

2. **Forensic Inconsistencies Between Text & Code Outputs**:
   - **LightGBM OOT ROC-AUC Mismatch**:
     - Text claim (`final_notebook_text.md` lines 369, 391, 399; script lines 1199, 1244, 1252) claims Standalone LightGBM achieves ROC-AUC of **0.7272**.
     - Notebook cell 23 printed output (`Lending_Club_Project_Final.ipynb` line 3963) & cell 24 printed output (line 7988) reports `LightGBM Test ROC-AUC: 0.7140`.
     - Displayed plot image `model_plots/model_comparison.png` labels `LightGBM Optuna (AUC=0.7111)` and `PR-AUC=0.3932`.
   - **MLP Performance Narrative Contradiction**:
     - Printed execution output of Cell 24 (`Lending_Club_Project_Final.ipynb` lines 7985-7989) shows:
       `MLP Test ROC-AUC: 0.7303`, `MLP Test PR-AUC: 0.4069`, `LightGBM Test ROC-AUC: 0.7140`.
     - Text in Markdown directly below Cell 24 (`final_notebook_text.md` lines 336-358) claims that MLP underperformed LightGBM and writes a 25-line theoretical justification for why tree models beat deep learning.
   - **SHAP Feature Importance Hallucination**:
     - Text claim (`final_notebook_text.md` line 300, script line 900) states: *"The SHAP summary clearly shows that `fico_range_low` and `dti` are the most dominant predictors."*
     - Actual plots `model_plots/shap_importance.png` and `model_plots/shap_beeswarm.png` show:
       - Rank #1: `sub_grade_num` (Mean |SHAP| = 0.395)
       - Rank #2: `term_num` (Mean |SHAP| = 0.205)
       - Rank #3: `adjusted_dti` (Mean |SHAP| = 0.175)
       - Rank #4: `fico_avg` (Mean |SHAP| = 0.130)
   - **Threshold Direction Mismatch**:
     - Text (`final_notebook_text.md` line 280, script line 867) claims shifting threshold "down from 0.50 to optimal point" drastically increases net profit.
     - Plot `model_plots/threshold_optimization.png` shows Net Profit peaks at threshold **0.59** (shifted UP from 0.50), while threshold 0.30 maximizes F2 score.
   - **Detached Pre-Computed Plots & Missing Scripts**:
     - Script lines 836-897 load pre-rendered PNGs via `display(Image(filename='model_plots/...'))`.
     - Referenced files `modeling.py` and `model_results.json` do not exist in the project repository.
   - **Omitted Calibration Metrics**:
     - `model_plots/calibration_curves.png` displays Brier score improvement from 0.1978 to 0.1555 (Isotonic), but text markdown ignores Brier scores completely.

3. **Model Explainability & SHAP Coverage**:
   - `model_plots/shap_beeswarm.png` and `model_plots/shap_importance.png` are present.
   - SHAP Dependence Plots (non-linear interactions) and SHAP Waterfall Plots (individual applicant decisions) are MISSING.
   - Business compliance context (FCRA adverse action reason codes) is omitted.

---

## 2. Logic Chain

1. **Evaluating Requirement 1 (Metric Selection)**:
   - The project correctly identifies class imbalance (18.0% default) and rejects simple Accuracy (82%) in favor of ROC-AUC and PR-AUC. This fulfills the metric selection requirement of Section 5.

2. **Evaluating Requirement 2 (Forensic Consistency)**:
   - Cross-referencing every free-text claim against printed execution outputs reveals multiple critical discrepancies:
     - Claiming AUC 0.7272 when plot shows 0.7111 and intermediate prints show 0.7140.
     - Claiming MLP underperformed LightGBM when printed test output shows MLP AUC = 0.7303 > LightGBM AUC = 0.7140.
     - Claiming FICO and DTI are the top 2 SHAP features when the plot clearly shows `sub_grade_num` and `term_num` as the top 2 features.
   - These mismatches violate the 100% Forensic Consistency standard of `ds_review_guide.md` Section 5.

3. **Evaluating Requirement 3 (Model Explainability)**:
   - Global SHAP plots exist, but local explainability (waterfall plots) and interaction analysis (dependence plots) are missing, limiting the business utility for credit underwriters.

---

## 3. Caveats

- **Scope**: Audit focused on Phase 5 (Evaluation & Explainability). Feature engineering (Phase 2), Clustering (Phase 3), and Model Training (Phase 4) were evaluated strictly in the context of their impact on Phase 5 evaluation outputs.
- **Assumptions**: We assumed `Lending_Club_Project_Final.ipynb` represents the executed notebook state.
- **Read-Only Constraint**: We did not modify any python script or notebook files in the project directory, ensuring strict read-only compliance.

---

## 4. Conclusion

Phase 5 earns a grade of **78 / 100**. While metric selection (ROC-AUC & PR-AUC) is methodologically sound, critical forensic mismatches exist between text claims and printed cell outputs/plots, along with missing SHAP dependence and waterfall plots. Following the concrete, line-by-line recommendations in `report.md` will elevate the score to **100 / 100**.

---

## 5. Verification Method

1. **Inspect Report File**: Read `c:\Users\yonat\.gemini\antigravity\scratch\lending_club_ds\.agents\explorer_phase5\report.md`.
2. **Cross-Reference Notebook Outputs**:
   - `Lending_Club_Project_Final.ipynb` Cell 24 stdout (lines 7985-7989) for MLP AUC = 0.7303 vs LightGBM AUC = 0.7140.
   - `Lending_Club_Project_Final.ipynb` Cell 25 stdout (line 8181) for Final LightGBM AUC = 0.7272.
3. **Inspect Image Files**:
   - View `model_plots/model_comparison.png` for LightGBM Optuna AUC = 0.7111.
   - View `model_plots/shap_importance.png` for `sub_grade_num` rank #1 (0.395) and `term_num` rank #2 (0.205).
   - View `model_plots/threshold_optimization.png` for Net Profit optimal threshold = 0.59.
   - View `model_plots/calibration_curves.png` for Brier scores = 0.1978 & 0.1555.
