# Handoff Report — Phase 2: Feature Engineering & Preprocessing

**Agent**: Investigator Agent 2 (Phase 2 Explorer)  
**Target Path**: `c:\Users\yonat\.gemini\antigravity\scratch\lending_club_ds\.agents\explorer_phase2\handoff.md`  
**Report Path**: `c:\Users\yonat\.gemini\antigravity\scratch\lending_club_ds\.agents\explorer_phase2\report.md`  
**Date**: 2026-07-27  

---

## 1. Observation

1. **Data Leakage Removal**:
   - `Lending_Club_Project_Final.py` lines 81–96 (`round1_leakage`): 22 post-origination columns explicitly dropped (`total_pymnt`, `recoveries`, `last_pymnt_d`, `total_rec_prncp`, `last_pymnt_amnt`, etc.).
   - `Lending_Club_Project_Final.py` lines 114–142 (`round2_leakage`): Drops `funded_amnt`, `funded_amnt_inv`, `num_tl_30dpd`, `num_tl_120dpd_2m`, `chargeoff_within_12_mths`, 15 `hardship_*` columns, 13 `sec_app_*` columns, constants, free-text, and >75% null columns.
2. **`sub_grade` & `int_rate` Handling Discrepancy**:
   - `Lending_Club_Project_Final.py` line 61: Claims `sub_grade` is NOT leakage and is kept/encoded ordinals.
   - `Lending_Club_Project_Final.py` lines 824–829: Ordinally encodes `sub_grade` into `sub_grade_num`.
   - `Lending_Club_Project_Final.py` lines 928–933: `sub_grade_num` and `int_rate` are **excluded** from `features` list used for model training.
   - `Lending_Club_Project_Final.py` line 1247: Narrative claims `sub_grade` and `int_rate` were removed to prevent "Target Proxy Leakage".
3. **Survivorship Bias & Embargo Period**:
   - `Lending_Club_Project_Final.py` line 177: `df = df.filter(pl.col('issue_date').dt.year() < 2016)`. Explicitly drops 2016–2018 loans to avoid right-censoring for 36-month loans.
   - 60-month loans from 2014–2015 remain un-embargoed (dataset ends 2018Q4), leaving slight residual bias.
4. **Custom Financial Features**:
   - `Lending_Club_Project_Final.py` lines 100–111: Consolidates joint income/DTI into `total_annual_inc` and `total_dti`.
   - `Lending_Club_Project_Final.py` lines 770–796: Creates `installment_to_income`, `adjusted_dti`, `loan_to_income`, `estimated_revol_limit`.
   - Missing: `credit_history_length` (calculated from `earliest_cr_line`) is never created or used.
5. **NaNs & Structural Missingness Flags Omission**:
   - `Lending_Club_Project_Final.py` lines 747–759: Creates `mths_since_*_never` flags and `emp_length_missing` flag, and imputes 999.
   - `Lending_Club_Project_Final.py` line 928: **Omites** all `_never` flags, `emp_length_missing`, and 999-imputed `mths_since_*` columns from `features` array.
   - Tree vs NN Prep (lines 939, 1080): Uses `category` dtypes for LightGBM; One-Hot Encoding + median fill + scaling for MLP.

---

## 2. Logic Chain

1. *From Observation 1*: The two-round leakage cleanup is rigorous and successfully drops all 50+ post-origination columns and target proxies.
2. *From Observation 2*: The code and narrative conflict directly regarding `sub_grade` and `int_rate`. While line 61 and line 828 state and implement ordinal encoding for `sub_grade`, line 928 leaves `sub_grade_num` out of the model `features` list, and line 1247 erroneously claims it was dropped as leakage. Reconciling this mismatch is essential for 100/100 scientific consistency.
3. *From Observation 3*: The embargo period filter `< 2016` effectively eliminates right-censoring for 36-month loans. However, 60-month loans from 2014-2015 have a 5-year duration extending past 2018Q4, meaning non-matured active loans were filtered out on line 159, creating minor residual survivorship bias.
4. *From Observation 4 & 5*: The domain feature engineering (`adjusted_dti`, joint consolidation) and structural missingness analysis (`_never` flags) are conceptually strong. However, omitting all `_never` flags and `credit_history_length` from line 928 `features` list prevents the model from utilizing these signals.

---

## 3. Caveats

- **Scope Limit**: Audit was strictly read-only per system prompt guidelines. No project source files were modified.
- **Model Training Execution**: We verified feature arrays by code inspection of `Lending_Club_Project_Final.py`. Re-running model fitting with restored features requires execution approval.

---

## 4. Conclusion

Phase 2 exhibits strong credit domain engineering, but suffers from two major implementation gaps:
1. **Omission of Structural Missingness & Ordinal Features**: `_never` flags and `sub_grade_num` were engineered in Section 5 but left out of line 928 `features`.
2. **Text-Code Mismatch**: Narrative claims regarding `sub_grade` / `int_rate` leakage conflict between section 1.1 and section 6.4.

Addressing these issues with the provided 5 actionable recommendations will bring Phase 2 to a 100/100 score.

---

## 5. Verification Method

1. **Verify Line Numbers & Code Blocks**:
   - Inspect `Lending_Club_Project_Final.py` lines 81–96 (R1 leakage), lines 114–142 (R2 leakage), line 177 (embargo filter), lines 747–759 (structural missingness), and line 928 (`features` list definition).
2. **Verify Detailed Audit Report**:
   - Read full findings and line-by-line recommendations in `c:\Users\yonat\.gemini\antigravity\scratch\lending_club_ds\.agents\explorer_phase2\report.md`.
3. **Verify Plot Artifacts**:
   - Check file existence of `eda_plots/silhouette_elbow.png`, `eda_plots/pca_vs_tsne.png`, `fe_plots/collinearity_heatmap.png`.
