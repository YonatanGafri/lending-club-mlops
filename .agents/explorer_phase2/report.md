# Phase 2 Audit Report: Feature Engineering & Preprocessing

**Project**: Lending Club Data Science Risk Modeling  
**Target Codebase**: `c:\Users\yonat\.gemini\antigravity\scratch\lending_club_ds`  
**Audited Files**: `Lending_Club_Project_Final.py`, `final_notebook_text.md`, `ds_review_guide.md`  
**Auditor**: Investigator Agent 2 (Phase 2: Feature Engineering & Preprocessing)  
**Date**: July 27, 2026  

---

## Executive Summary

Phase 2 (Feature Engineering & Preprocessing) represents the architectural backbone of credit risk prediction. A rigorous audit was conducted against **Section 2 of `ds_review_guide.md`** and **`ORIGINAL_REQUEST.md` requirement R3**. 

Overall, the project demonstrates advanced financial domain awareness, strong data cleaning hygiene, and sound methodological principles (such as out-of-time chronological splitting and structural missingness modeling). However, the audit revealed **critical code-narrative contradictions** (e.g., `sub_grade` handling), **omitted engineered features** during actual model training (e.g., structural `_never` flags), and **residual survivorship bias** for 60-month loans in the embargo filter.

---

## Exhaustive Section-by-Section Audit

### 1. Data Leakage Prevention

#### Audit Focus & Criteria
- Explicit removal of post-origination variables known only after loan issuance (`total_pymnt`, `recoveries`, `last_pymnt_d`, `total_rec_prncp`, `last_pymnt_amnt`, etc.).
- Proper handling and consistency regarding pre-assigned initial loan attributes like `sub_grade` and `int_rate`.

#### Findings & Evidence

1. **Pass / Success Items**:
   - **Two-Round Leakage Cleanup Pipeline**:
     - **Round 1 Cleanup** (`Lending_Club_Project_Final.py` lines 81–96): Explicitly identifies and drops 22 post-origination columns:
       ```python
       round1_leakage = [
           'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int',
           'total_rec_late_fee', 'recoveries', 'collection_recovery_fee',
           'last_pymnt_d', 'last_pymnt_amnt', 'next_pymnt_d', 'debt_settlement_flag',
           'settlement_status', 'settlement_date', 'settlement_amount',
           'settlement_percentage', 'settlement_term', 'hardship_flag',
           'out_prncp', 'out_prncp_inv',
           'last_fico_range_high', 'last_fico_range_low', 'last_credit_pull_d'
       ]
       ```
       Execution: `cols_to_drop_r1 = list(set([c for c in round1_leakage + useless_ids + constant_cols if c in df.columns]))` and `df = df.drop(cols_to_drop_r1)` (lines 95–96).
     - **Round 2 Cleanup** (`Lending_Club_Project_Final.py` lines 114–142): Drops additional leakage and post-origination attributes:
       - `funded_amnt`, `funded_amnt_inv` (determined post-application).
       - `num_tl_30dpd`, `num_tl_120dpd_2m`, `chargeoff_within_12_mths` (target proxies).
       - 15 post-origination hardship columns (`hardship_type`, `hardship_reason`, etc.).
       - 13 `sec_app_*` columns (100% null in training set).
       - High-missing (>75%) columns (`open_acc_6m`, `il_util`, `open_il_12m`, etc.).
       Execution: `df = df.drop(r2_to_drop)` (lines 140–141).

2. **Flaws, Contradictions & Gaps**:
   - 🔴 **Direct Contradiction on `sub_grade` Handling**:
     - *Section 1 Narrative (lines 48–61)*: States: `"Our Decision: Since sub_grade is assigned strictly pre-origination, it is NOT leakage. It is a powerful proxy for creditworthiness, so we keep it as a feature and encode it ordinally (A1=1, G5=35)."`
     - *Feature Preprocessing (lines 824–829)*: Ordinally encodes `sub_grade` to `sub_grade_num`.
     - *Modeling Feature List (lines 928–933)*: `sub_grade_num` is **EXCLUDED** from `features = ["loan_amnt", "total_annual_inc", "total_dti", "revol_util", "fico_range_low", "adjusted_dti", "installment_to_income", "loan_to_income", "estimated_revol_limit", "emp_length_num", "term_num", "purpose", "home_ownership", "verification_status"]`.
     - *Executive Summary Narrative (line 1247)*: States: `"1. Target Proxy Leakage Prevented: Removed internal sub_grade and interest rate assignments."`
     - **Impact**: Severe narrative-code mismatch. The text contradicts itself across sections (keeping vs dropping `sub_grade`), while the code encodes `sub_grade` on line 828 but silently drops it from model input on line 928.
   - 🔴 **Omission & Narrative Mismatch on `int_rate`**:
     - `int_rate` is pre-assigned by Lending Club prior to origination based on initial risk assessment.
     - `int_rate` is heavily analyzed in EDA (lines 293, 447, 511, 659, 716) and used in PCA factor loading analysis (line 659).
     - However, `int_rate` is omitted from `features` (lines 928–933), and line 1247 falsely claims `int_rate` was explicitly removed as leakage.

---

### 2. Survivorship Bias & Embargo Period

#### Audit Focus & Criteria
- Embargo period implementation to filter out non-matured recent loans (e.g. 2016+).
- Verification of exact code lines and mitigation of right-censoring / survivorship bias.

#### Findings & Evidence

1. **Pass / Success Items**:
   - **Explicit Embargo Period Implementation**:
     - `Lending_Club_Project_Final.py` line 177:
       ```python
       # --- Implement Embargo Period to prevent right-censoring ---
       df = df.filter(pl.col('issue_date').dt.year() < 2016)
       ```
     - **Rationale**: The dataset ends in 2018Q4. Loans issued in 2016 or later have not reached full 36-month maturity. Dropping loans issued $\ge 2016$ ensures that all 36-month loans in the dataset have reached their terminal status window.
   - **Chronological Out-of-Time (OOT) Split**:
     - `Lending_Club_Project_Final.py` lines 179–185:
       ```python
       df = df.sort('issue_date')
       total_rows = df.shape[0]
       train_size = int(total_rows * 0.8)
       train_df = df.head(train_size)
       test_df = df.tail(total_rows - train_size)
       ```
     - Replaces invalid random K-Fold splits with strict chronological order, mimicking real-world production scoring.

2. **Flaws, Contradictions & Gaps**:
   - ⚠️ **Incomplete Embargo for 60-Month Loans**:
     - While `< 2016` provides 3+ years of maturity (suitable for 36-month loans), 60-month loans issued in 2014 and 2015 require 5 full years to mature (reaching 2019 / 2020), which exceeds the 2018Q4 dataset boundary.
     - Consequently, 60-month loans issued in 2014–2015 that were still active (`Current`) were dropped during target filtering (`loan_status.is_in(['Fully Paid', 'Charged Off'])`, line 159), while 60-month loans from those years that defaulted early or paid off early were retained.
     - **Impact**: Introduces a residual survivorship bias in the 60-month loan subset.

---

### 3. Custom Financial Features

#### Audit Focus & Criteria
- Creation of domain-specific financial features (joint application income adjustment, DTI ratios, loan-to-income, credit history length).

#### Findings & Evidence

1. **Pass / Success Items**:
   - **Joint Application Consolidation**:
     - `Lending_Club_Project_Final.py` lines 100–111:
       ```python
       df = df.with_columns(
           pl.when(pl.col('application_type') == 'Joint App')
           .then(pl.col('annual_inc_joint'))
           .otherwise(pl.col('annual_inc'))
           .alias('total_annual_inc')
       )
       df = df.with_columns(
           pl.when(pl.col('application_type') == 'Joint App')
           .then(pl.col('dti_joint'))
           .otherwise(pl.col('dti'))
           .alias('total_dti')
       )
       ```
     - Consolidates individual and joint applicant income and DTI prior to dropping joint-specific columns in Round 2 cleanup.
   - **Domain-Specific Financial Ratios**:
     - `Lending_Club_Project_Final.py` lines 770–796:
       - `installment_to_income`: `installment / (total_annual_inc / 12 + EPSILON)` (Monthly payment burden ratio).
       - `adjusted_dti`: `((total_dti/100 * total_annual_inc/12 + installment) / (total_annual_inc/12 + EPSILON)) * 100` (Recalculates DTI incorporating the new loan installment).
       - `loan_to_income`: `loan_amnt / (total_annual_inc + EPSILON)` (Principal relative to annual income).
       - `estimated_revol_limit`: `revol_bal / (revol_util/100 + EPSILON)` (Back-calculated credit limit).
       - Protected with `EPSILON = 1e-6` against zero division.
   - **Ordinal Transformations**:
     - `emp_length_num`: Mapped `< 1 year` $\rightarrow 0$, ..., `10+ years` $\rightarrow 10$ (`Lending_Club_Project_Final.py` lines 808–817).
     - `term_num`: Encoded 36 vs 60 as integer (lines 819–822).

2. **Flaws, Contradictions & Gaps**:
   - 🔴 **Missing Credit History Length (`credit_history_length`)**:
     - The Lending Club column `earliest_cr_line` (date of borrower's first credit line) is a cornerstone of credit scoring (represents 15% of FICO score calculation).
     - `earliest_cr_line` was never converted into a numeric age duration feature (`credit_history_length`) and is omitted entirely.
   - 🔴 **Orphaned / Unused Engineered Features**:
     - `sub_grade_num` is created on line 828, but omitted from `features` on line 928.
     - `estimated_revol_limit` is created on line 795 and included in `features`, but `revol_bal` itself is left out of `features`.
     - `fico_range_low` is included, but `fico_range_high` or `fico_average` is absent.

---

### 4. NaNs & Categorical Preprocessing

#### Audit Focus & Criteria
- Missing value handling strategies (avoiding blind `fillna(0)`).
- Missingness indicators (`_never` flags for credit history).
- Preprocessing distinction between tree-based models and neural networks.
- Verification of visual artifacts (`fe_plots/collinearity_heatmap.png`, etc.).

#### Findings & Evidence

1. **Pass / Success Items**:
   - **Structural Missingness Strategy**:
     - EDA (`Lending_Club_Project_Final.py` lines 240–270) demonstrates that missing values in credit history columns (`mths_since_last_delinq`, `mths_since_last_record`, `mths_since_last_major_derog`, `mths_since_recent_bc_dlq`, `mths_since_recent_revol_delinq`, `mths_since_recent_inq`) indicate "event never occurred" (protective factor with ~16.5% default rate vs ~21% when present).
     - EDA also identifies `emp_length` missingness as anomalous (higher default rate: 20.3% vs 17.6%).
     - Pipeline implementation (`Lending_Club_Project_Final.py` lines 740–761):
       ```python
       for col in structural_miss_cols:
           flag_name = f'{col}_never'
           train_df = train_df.with_columns(pl.col(col).is_null().cast(pl.Int8).alias(flag_name))
           test_df = test_df.with_columns(pl.col(col).is_null().cast(pl.Int8).alias(flag_name))
           train_df = train_df.with_columns(pl.col(col).fill_null(999))
           test_df = test_df.with_columns(pl.col(col).fill_null(999))
       ```
   - **Tree Models vs Neural Network Preprocessing Differentiation**:
     - **LightGBM (Tree Model)**: Nominal categorical features (`purpose`, `home_ownership`, `verification_status`) are set to pandas `category` dtype (`Lending_Club_Project_Final.py` lines 939–943). Preserves native categorical splits and avoids blind `fillna(0)`.
     - **MLP (Neural Network)**: Categorical features are One-Hot Encoded via `pd.get_dummies` (lines 1080–1082), continuous missing values are median-imputed (lines 1083–1085), and input features are standardized via `StandardScaler` (lines 1089–1092).
     - Includes academic literature citation (Grinsztajn et al., NeurIPS 2022) explaining tabular performance differentials (lines 1150–1172).

2. **Flaws, Contradictions & Gaps**:
   - 🚨 **CRITICAL OMISSION — Structural Missingness Flags Omitted from Model Training!**:
     - Lines 747–759 create `mths_since_last_delinq_never`, `mths_since_last_record_never`, `mths_since_last_major_derog_never`, `mths_since_recent_bc_dlq_never`, `mths_since_recent_revol_delinq_never`, `mths_since_recent_inq_never`, `emp_length_missing`, and impute 999 into the original columns.
     - **However, on Line 928**, when defining the feature list for model training:
       ```python
       features = [
           "loan_amnt", "total_annual_inc", "total_dti", "revol_util", 
           "fico_range_low", "adjusted_dti", "installment_to_income", 
           "loan_to_income", "estimated_revol_limit", "emp_length_num", 
           "term_num", "purpose", "home_ownership", "verification_status"
       ]
       ```
     - **NONE** of the `_never` flags, `emp_length_missing` flag, or 999-imputed `mths_since_*` features are included in `features`!
     - **Impact**: All structural missingness engineering performed in Section 5.1 is silently discarded during model training.
   - ⚠️ **Missing Plot Save Call for `fe_plots/collinearity_heatmap.png`**:
     - While `fe_plots/collinearity_heatmap.png` exists on disk, `Lending_Club_Project_Final.py` lines 715–723 call `plt.show()` without calling `plt.savefig('fe_plots/collinearity_heatmap.png')`.

---

## Comprehensive Summary Table of Findings

| Section | Topic | Status | Key Code Location | Description |
| :--- | :--- | :---: | :--- | :--- |
| **2.1** | Leakage Removal (R1 & R2) | ✅ Pass | `Lending_Club_Project_Final.py:81-142` | Explicitly drops 22 post-origination columns in R1 and 30+ leakage/null columns in R2. |
| **2.1** | `sub_grade` & `int_rate` Logic | 🔴 Flaw | `Lending_Club_Project_Final.py:61,828,928,1247` | Text states `sub_grade` kept (L61), code encodes it (L828), code drops it (L928), conclusion text claims dropped as leakage (L1247). |
| **3.0** | Embargo Period | ✅ Pass | `Lending_Club_Project_Final.py:177` | Filter `issue_date < 2016` applied to ensure 36-month loan maturity. |
| **3.0** | 60-Month Loan Embargo | ⚠️ Gap | `Lending_Club_Project_Final.py:177` | 60-month loans from 2014-2015 remain non-matured by 2018Q4 cutoff, leaving mild residual bias. |
| **5.1** | Joint App Consolidation | ✅ Pass | `Lending_Club_Project_Final.py:100-111` | Consolidates joint income and DTI into `total_annual_inc` and `total_dti`. |
| **5.2** | Domain Financial Features | ✅ Pass | `Lending_Club_Project_Final.py:770-796` | Creates `installment_to_income`, `adjusted_dti`, `loan_to_income`, `estimated_revol_limit`. |
| **5.2** | Credit History Length | 🔴 Flaw | Missing in `Lending_Club_Project_Final.py` | `earliest_cr_line` is never parsed into a numeric duration feature (`credit_history_length`). |
| **5.1** | Structural Missingness Flags | 🚨 Critical Omission | `Lending_Club_Project_Final.py:747-759, 928` | Flags (`_never`) created in Sec 5.1 but omitted from `features` array on line 928. |
| **5.3** | Categorical & Model Prep | ✅ Pass | `Lending_Club_Project_Final.py:939, 1080` | Native `category` dtypes for LightGBM; One-Hot Encoding + median fill + scaling for MLP. |
| **4.8** | Visual Artifact Save | ⚠️ Gap | `Lending_Club_Project_Final.py:722` | `fe_plots/collinearity_heatmap.png` generated in notebook but lacks `plt.savefig` call in `.py` script. |

---

## Actionable Recommendations for a 100/100 Grade

To elevate Phase 2 to perfection, implement the following line-by-line code changes:

### 1. Fix Feature List Omissions (`Lending_Club_Project_Final.py` Line 928)
Update the `features` list to include engineered structural missingness flags, `sub_grade_num`, `int_rate`, and `credit_history_length`:

```python
# Lines 928-933 in Lending_Club_Project_Final.py
features = [
    # Baseline Core Features
    "loan_amnt", "int_rate", "sub_grade_num", "total_annual_inc", "total_dti", 
    "revol_util", "revol_bal", "fico_range_low", "emp_length_num", "term_num",
    
    # Custom Domain Ratios
    "adjusted_dti", "installment_to_income", "loan_to_income", "estimated_revol_limit",
    "credit_history_length",
    
    # Structural Missingness & Credit History Flags
    "mths_since_last_delinq_never", "mths_since_last_record_never", 
    "mths_since_last_major_derog_never", "mths_since_recent_bc_dlq_never", 
    "mths_since_recent_revol_delinq_never", "mths_since_recent_inq_never", 
    "emp_length_missing", "mths_since_last_delinq", "mths_since_last_record",
    
    # Categoricals
    "purpose", "home_ownership", "verification_status"
]
```

### 2. Feature Engineer `credit_history_length` (`Lending_Club_Project_Final.py` Section 5.2)
Add the following code block around line 798:

```python
# Calculate credit history length in years from earliest_cr_line
if 'earliest_cr_line' in train_df.columns:
    train_df = train_df.with_columns(
        pl.col('earliest_cr_line').str.strptime(pl.Date, format='%b-%Y', strict=False).alias('earliest_cr_date')
    )
    test_df = test_df.with_columns(
        pl.col('earliest_cr_line').str.strptime(pl.Date, format='%b-%Y', strict=False).alias('earliest_cr_date')
    )
    train_df = train_df.with_columns(
        ((pl.col('issue_date') - pl.col('earliest_cr_date')).dt.total_days() / 365.25).alias('credit_history_length')
    )
    test_df = test_df.with_columns(
        ((pl.col('issue_date') - pl.col('earliest_cr_date')).dt.total_days() / 365.25).alias('credit_history_length')
    )
```

### 3. Reconcile Narrative Claims Regarding `sub_grade` & `int_rate`
- Align line 61, line 828, line 928, and line 1247.
- Clarify in the narrative: `sub_grade` and `int_rate` are pre-origination attributes assigned by Lending Club. Including them allows the model to benchmark against LC's internal rating, whereas excluding them tests raw algorithmic underwriting capability. Whichever choice is made, ensure code and text are 100% consistent across all sections.

### 4. Enhance Embargo Logic for 60-Month Loans (`Lending_Club_Project_Final.py` Line 177)
Refine the embargo filter to handle 60-month loan maturity explicitly:

```python
# Line 177: Filter 36-month loans < 2016 and 60-month loans < 2014
df = df.filter(
    ((pl.col('term').str.contains('36')) & (pl.col('issue_date').dt.year() < 2016)) |
    ((pl.col('term').str.contains('60')) & (pl.col('issue_date').dt.year() < 2014))
)
```

### 5. Save Heatmap Figure Explicitly (`Lending_Club_Project_Final.py` Line 722)
Add image export line:

```python
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title('Correlation Heatmap of Financial Features')
plt.savefig('fe_plots/collinearity_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
```

---
*Report compiled by Investigator Agent 2 (Phase 2 Explorer).*
