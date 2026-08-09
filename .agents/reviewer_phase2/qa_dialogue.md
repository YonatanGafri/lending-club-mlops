# Phase 2 QA Cross-Verification Dialogue & Technical Transcript

**Participants**:
- **`explorer_phase2`** (Investigator Agent 2 — Phase 2 Lead)
- **`reviewer_phase2`** (QA Reviewer & Adversarial Critic — Phase 2 Lead)

**Date**: July 27, 2026  
**Subject**: Comprehensive Audit & Cross-Verification of Phase 2 (Feature Engineering & Preprocessing) Findings  
**Target Codebase**: `c:\Users\yonat\.gemini\antigravity\scratch\lending_club_ds` (`Lending_Club_Project_Final.py`, `final_notebook_text.md`, `ds_review_guide.md`)

---

## Session 1: Opening & Scope Verification

**`reviewer_phase2`**: Welcome, `explorer_phase2`. I have received your Phase 2 Audit Report (`report.md`) and Handoff Report (`handoff.md`). As QA Reviewer and Adversarial Critic, my goal is to independently cross-verify every line reference, variable name, mathematical logic, and narrative claim in `Lending_Club_Project_Final.py`, `final_notebook_text.md`, and `ds_review_guide.md`. We will examine whether your findings hold up under rigorous scrutiny, test your assumptions against potential edge cases, and verify codebase integrity. Let's begin with the overall structure of your audit.

**`explorer_phase2`**: Thank you, `reviewer_phase2`. In our Phase 2 investigation, we audited the feature engineering and preprocessing pipeline against Section 2 of `ds_review_guide.md` and Requirement R3 of `ORIGINAL_REQUEST.md`. Our audit evaluated four primary pillars:
1. Data Leakage Prevention (Round 1 & Round 2 cleanups, `sub_grade`/`int_rate` handling).
2. Survivorship Bias & Embargo Period (OOT split, right-censoring filters).
3. Custom Financial Features (Joint app consolidation, domain financial ratios, credit history length).
4. NaNs & Categorical Preprocessing (Structural missingness `_never` flags, Tree vs Neural Net preprocessing paths).

Our central finding is that while the domain logic and exploratory analysis in Phase 2 are exemplary, there are **critical implementation gaps**—most notably, structural missingness flags created in Section 5.1 are silently omitted from model training on line 928, and a three-way contradiction exists regarding `sub_grade` handling.

**`reviewer_phase2`**: Excellent summary. Let's drill into each section systematically and verify the evidence.

---

## Session 2: Deep-Dive into Data Leakage Prevention

**`reviewer_phase2`**: Let's start with Data Leakage Prevention. In your report, you highlighted a two-round leakage cleanup pipeline. Let's verify the exact code lines.

**`explorer_phase2`**: On lines 81–96 of `Lending_Club_Project_Final.py`, Round 1 cleanup explicitly defines `round1_leakage` with 22 post-origination columns:
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
These are combined with `useless_ids` (`['id', 'member_id', 'url', 'desc']`) and single-value `constant_cols`, and dropped on line 96 (`df = df.drop(cols_to_drop_r1)`).

In Round 2 (lines 114–142), another 30+ columns are dropped, including:
- Definite post-application leakage: `funded_amnt`, `funded_amnt_inv`, `num_tl_30dpd`, `num_tl_120dpd_2m`, `chargeoff_within_12_mths`.
- Hardship columns (15 post-origination attributes).
- Secondary applicant columns (13 `sec_app_*` attributes that are 100% null in training).
- High missing (>75%) columns like `open_acc_6m`, `il_util`, `open_il_12m`, etc.

**`reviewer_phase2`**: I have independently inspected `Lending_Club_Project_Final.py` lines 81–142, and I confirm these line numbers and lists are exact. That part of the pipeline is clean. 

However, let's scrutinize your claim regarding **`sub_grade`** and **`int_rate`**. You alleged a major narrative-code contradiction. Can you lay out the exact evidence from both `.py` and `.md` files?

**`explorer_phase2`**: Absolutely. Here is the exact paper trail of `sub_grade`:
1. **`Lending_Club_Project_Final.py` line 61** (and `final_notebook_text.md` line 36):
   > *"Our Decision: Since `sub_grade` is assigned strictly pre-origination, it is NOT leakage. It is a powerful proxy for creditworthiness, so we keep it as a feature and encode it ordinally (A1=1, G5=35)."*
2. **`Lending_Club_Project_Final.py` lines 824–829**:
   The script actually executes the ordinal mapping, creating `sub_grade_num`:
   ```python
   sub_grades = sorted(train_df['sub_grade'].unique().drop_nulls().to_list())
   sg_map = {sg: i+1 for i, sg in enumerate(sub_grades)}
   train_df = train_df.with_columns(pl.col('sub_grade').replace_strict(sg_map, default=0).cast(pl.Int32).alias('sub_grade_num'))
   ```
3. **`Lending_Club_Project_Final.py` lines 928–933**:
   When `features` is populated for LightGBM training, `sub_grade_num` is **EXCLUDED**:
   ```python
   features = [
       "loan_amnt", "total_annual_inc", "total_dti", "revol_util", 
       "fico_range_low", "adjusted_dti", "installment_to_income", 
       "loan_to_income", "estimated_revol_limit", "emp_length_num", 
       "term_num", "purpose", "home_ownership", "verification_status"
   ]
   ```
4. **`Lending_Club_Project_Final.py` line 1247** (Executive Summary):
   > *"1. Target Proxy Leakage Prevented: Removed internal `sub_grade` and interest rate assignments."*
5. **`final_notebook_text.md` line 394** (Executive Summary in Notebook):
   > *"1. Target Proxy Leakage Prevented: Removed strict future variables (e.g., payments, recoveries). Decided to **keep** `sub_grade` as a valid pre-origination risk proxy (expert baseline)."*

**`reviewer_phase2`**: That is a stark, triple contradiction:
- Line 61 text says **keep** `sub_grade`.
- Line 828 code **encodes** `sub_grade_num`.
- Line 928 code **drops** `sub_grade_num` from model input.
- `.py` Line 1247 text claims `sub_grade` was **removed as leakage**.
- `.md` Line 394 text claims `sub_grade` was **kept as expert baseline**.

This is a classic integrity and narrative consistency failure where the text author and the code author diverged. In credit risk modeling, keeping vs dropping `sub_grade` is a key methodological decision (keeping it measures value-add over Lending Club's grade; dropping it tests raw underwriting). But claiming both simultaneously while silently omitting it from line 928 is unacceptable for a Kaggle-grade project.

---

## Session 3: Survivorship Bias & Embargo Period Stress-Test

**`reviewer_phase2`**: Moving to Section 3: Survivorship Bias and Embargo Period. You verified that `Lending_Club_Project_Final.py` line 177 contains `df = df.filter(pl.col('issue_date').dt.year() < 2016)`. 

Let's stress-test your challenge regarding 60-month loans. Why do you claim that `< 2016` leaves residual survivorship bias?

**`explorer_phase2`**: Consider the timeline:
- The dataset cutoff is **2018Q4** (December 31, 2018).
- A **36-month loan** issued in December 2015 matures by December 2018 (3 full years). Thus, filtering `issue_date < 2016` guarantees that all 36-month loans issued prior to 2016 have reached terminal status (`Fully Paid` or `Charged Off`).
- However, a **60-month loan** issued in 2014 or 2015 requires **5 full years** to mature—reaching terminal status in 2019 or 2020.
- Because the dataset ends in 2018Q4, any 60-month loan issued in 2014–2015 that was still performing on schedule as of late 2018 was labeled as `Current`.
- On line 159, the pipeline filters for terminal status only: `df = df.filter(pl.col('loan_status').is_in(['Fully Paid', 'Charged Off']))`.
- As a result, 60-month loans from 2014–2015 that were `Current` were dropped. But 60-month loans from those same years that **defaulted early** (e.g. within 2–3 years) or were **prepaid early** (`Fully Paid`) were retained!
- This creates an artificial inflation of default rates in the 60-month cohort issued in 2014–2015, introducing residual survivorship bias.

**`reviewer_phase2`**: That logic is airtight. The math is simple: $2014.0 + 5.0\text{ years} = 2019.0 > 2018.75$. 60-month loans issued in 2014 and 2015 cannot achieve 100% maturity within a dataset terminating in 2018Q4.

To achieve complete embargo rigor, the code should apply term-differentiated embargo cutoffs:
```python
df = df.filter(
    ((pl.col('term').str.contains('36')) & (pl.col('issue_date').dt.year() < 2016)) |
    ((pl.col('term').str.contains('60')) & (pl.col('issue_date').dt.year() < 2014))
)
```
This ensures both 36-month and 60-month loans in the filtered dataset have fully elapsed term windows.

---

## Session 4: Custom Financial Features & Missing Signals

**`reviewer_phase2`**: Now let's examine Custom Financial Features. You noted that joint application consolidation is performed on lines 100–111 and financial ratios are constructed on lines 770–796.

Let's check the code:
```python
# Lines 770-796
EPSILON = 1e-6
train_df = train_df.with_columns((pl.col('installment') / (pl.col('total_annual_inc') / 12 + EPSILON)).alias('installment_to_income'))
train_df = train_df.with_columns(((pl.col('total_dti')/100 * pl.col('total_annual_inc')/12 + pl.col('installment')) / (pl.col('total_annual_inc')/12 + EPSILON) * 100).alias('adjusted_dti'))
train_df = train_df.with_columns((pl.col('loan_amnt') / (pl.col('total_annual_inc') + EPSILON)).alias('loan_to_income'))
train_df = train_df.with_columns((pl.col('revol_bal') / (pl.col('revol_util')/100 + EPSILON)).alias('estimated_revol_limit'))
```
All four features use zero-division protection via `EPSILON = 1e-6`.

However, you raised two significant omissions:
1. `credit_history_length` was never calculated from `earliest_cr_line`.
2. Feature selection discrepancies.

Let's examine `earliest_cr_line`. In credit bureau scoring (such as FICO or VantageScore), the age of the oldest credit line is a primary predictor of default probability. Why was this omitted?

**`explorer_phase2`**: `earliest_cr_line` is present in the raw Lending Club schema as a string formatted like `'Jan-2001'`. To convert it into a valid continuous feature, one must parse it to a date and compute the delta relative to `issue_date`:
$$\text{credit\_history\_length} = \frac{\text{issue\_date} - \text{earliest\_cr\_date}}{365.25}$$
In the current code, `earliest_cr_line` is left as an unparsed string and dropped during preprocessing, missing a critical domain signal.

**`reviewer_phase2`**: Agreed. Furthermore, I noticed that `estimated_revol_limit` is included in `features` on line 928, but `revol_bal` itself (the raw revolving balance) is omitted from `features`! Similarly, `fico_range_low` is included, but `fico_range_high` is omitted without taking their average (`fico_average`). These are minor but noteworthy inconsistencies.

---

## Session 5: The Critical Finding — Structural Missingness Flags Omission

**`reviewer_phase2`**: Let's now address what is unquestionably the most critical finding of Phase 2: **Structural Missingness Flags Omission**.

Walk me through lines 240–270, lines 740–761, and line 928.

**`explorer_phase2`**: In Section 4.2 (lines 240–270), the author conducts EDA on structural missingness, demonstrating that missing values in credit history columns (`mths_since_last_delinq`, `mths_since_last_record`, `mths_since_last_major_derog`, `mths_since_recent_bc_dlq`, `mths_since_recent_revol_delinq`, `mths_since_recent_inq`) represent "event never occurred," which is a protective factor (16.5% default rate vs 21.0% when present). Conversely, missing `emp_length` is an anomaly with higher risk (20.3% vs 17.6%).

In Section 5.1 (lines 740–761), the script builds structural missingness indicators and imputes `999`:
```python
structural_miss_cols = [
    'mths_since_last_delinq', 'mths_since_last_record',
    'mths_since_last_major_derog', 'mths_since_recent_bc_dlq',
    'mths_since_recent_revol_delinq', 'mths_since_recent_inq',
]

for col in structural_miss_cols:
    flag_name = f'{col}_never'
    train_df = train_df.with_columns(pl.col(col).is_null().cast(pl.Int8).alias(flag_name))
    test_df = test_df.with_columns(pl.col(col).is_null().cast(pl.Int8).alias(flag_name))
    train_df = train_df.with_columns(pl.col(col).fill_null(999))
    test_df = test_df.with_columns(pl.col(col).fill_null(999))

if 'emp_length' in train_df.columns:
    train_df = train_df.with_columns(pl.col('emp_length').is_null().cast(pl.Int8).alias('emp_length_missing'))
    test_df = test_df.with_columns(pl.col('emp_length').is_null().cast(pl.Int8).alias('emp_length_missing'))
```
This adds 7 new binary columns: `mths_since_last_delinq_never`, `mths_since_last_record_never`, `mths_since_last_major_derog_never`, `mths_since_recent_bc_dlq_never`, `mths_since_recent_revol_delinq_never`, `mths_since_recent_inq_never`, and `emp_length_missing`.

However, on line 928, when defining the `features` array for LightGBM and MLP training:
```python
features = [
    "loan_amnt", "total_annual_inc", "total_dti", "revol_util", 
    "fico_range_low", "adjusted_dti", "installment_to_income", 
    "loan_to_income", "estimated_revol_limit", "emp_length_num", 
    "term_num", "purpose", "home_ownership", "verification_status"
]
```
`features` consists of **ONLY 14 COLUMNS**. 
**Every single one of the 7 `_never` flags, the `emp_length_missing` flag, and the 6 `999`-imputed `mths_since_*` continuous features is completely excluded from the model!**

**`reviewer_phase2`**: This is a monumental finding. The entire sophisticated feature engineering effort for structural missingness in Section 5.1 is rendered non-functional in Section 6 because the model selection array explicitly ignores those columns.

Let's double-check: Did LightGBM or MLP use any automatic column selector like `df.drop(['target', 'issue_date'])`?

**`explorer_phase2`**: No! Line 934 explicitly executes: `X_train = train_df.select(features).to_pandas()`. It selects only the 14 columns explicitly named in `features`.

**`reviewer_phase2`**: Verified. This confirms an **INTEGRITY & DESIGN OMISSION**. The feature engineering pipeline was executed in memory, but the model training pipeline hardcoded a 14-feature subset that omitted all structural missingness flags, `sub_grade_num`, `int_rate`, and `credit_history_length`.

---

## Session 6: Model Preprocessing Distinction & Artifact Verification

**`reviewer_phase2`**: Let's review the dual preprocessing pathways for Tree models vs Neural Networks in lines 939–943 and lines 1080–1092.

**`explorer_phase2`**:
1. **LightGBM (Tree Model)**:
   Lines 939–943 set categorical columns to pandas `category` dtype:
   ```python
   cat_cols = ["purpose", "home_ownership", "verification_status"]
   for col in cat_cols:
       X_train[col] = X_train[col].astype('category')
       X_test[col] = X_test[col].astype('category')
   ```
   This leverages LightGBM's native categorical splitting algorithm (Fisher's exact test-based optimal splits) without artificial numerical ranking or high-cardinality dummy expansion.

2. **MLP (Neural Network)**:
   Lines 1080–1092 implement proper neural network pipeline requirements:
   - One-Hot Encoding via `pd.get_dummies` (lines 1080–1081).
   - Column alignment between train and test using `.align(..., join='left', fill_value=0)` (line 1082).
   - Median imputation using training set medians only (lines 1083–1085).
   - Standardization via `StandardScaler()` fit on train and transformed on test (lines 1089–1092).

**`reviewer_phase2`**: This dual-path architecture is fully compliant with best practices as outlined in `ds_review_guide.md` Section 4 ("Tree models vs Neural Networks").

Finally, what about visual plot artifacts?

**`explorer_phase2`**: On lines 719–722 of `Lending_Club_Project_Final.py`:
```python
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title('Correlation Heatmap of Financial Features')
plt.show()
```
The script renders the heatmap on screen with `plt.show()`, but fails to save it to disk via `plt.savefig('fe_plots/collinearity_heatmap.png')`. However, the image file `fe_plots/collinearity_heatmap.png` exists in the repository (likely generated from the Jupyter notebook run). To maintain script parity, `plt.savefig` must be added.

---

## Session 7: QA Verdict & Unanimous Remediation Agreement

**`reviewer_phase2`**: Based on our independent line-by-line verification, we have reached unanimous consensus on the Phase 2 audit findings:

1. **Data Leakage (R1 & R2)**: PASS on dropping 50+ post-origination columns. FLAW on `sub_grade`/`int_rate` narrative-code 3-way contradiction.
2. **Survivorship Bias & Embargo**: PASS on applying `< 2016` filter. GAP on 60-month loan 5-year maturation horizon relative to 2018Q4 dataset boundary.
3. **Custom Domain Features**: PASS on joint income/DTI consolidation and 4 domain financial ratios. FLAW on omitting `credit_history_length` (`earliest_cr_line`).
4. **NaNs & Categoricals**: PASS on EDA structural missingness insights and LightGBM/MLP dual pipeline. CRITICAL OMISSION on excluding all 7 `_never` flags and 999-imputed columns from line 928 `features` list.
5. **Script Artifact Export**: GAP on missing `plt.savefig` for collinearity heatmap.

**Verdict**: **APPROVE WITH REQUIRED REMEDIATION**.

Let's define the exact code replacements required to elevate Phase 2 to a flawless 100/100 score:

1. **Restore Complete Feature List on Line 928**:
   Expand `features` from 14 to 26 columns, incorporating `int_rate`, `sub_grade_num`, `credit_history_length`, all `_never` missingness flags, and 999-imputed `mths_since_*` columns.
2. **Feature Engineer `credit_history_length`**:
   Add Polars date parsing for `earliest_cr_line` and compute year duration delta against `issue_date`.
3. **Harmonize Narrative Claims**:
   Update `Lending_Club_Project_Final.py` line 61, line 828, line 928, line 1247, and `final_notebook_text.md` line 394 to present a unified narrative regarding `sub_grade` and `int_rate`.
4. **Refine Embargo Filter for 60-Month Loans**:
   Update line 177 to filter 36-month loans `< 2016` and 60-month loans `< 2014`.
5. **Save Visual Artifact Explicitly**:
   Add `plt.savefig('fe_plots/collinearity_heatmap.png', dpi=150, bbox_inches='tight')` before `plt.show()` on line 722.

**`explorer_phase2`**: Fully agreed. This dialogue captures all observations, mathematical rationales, and exact code locations with complete forensic accuracy.

---
*Transcript verified and approved by QA Reviewer Agent 2 (`reviewer_phase2`).*
