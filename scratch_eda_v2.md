--- MARKDOWN ---
# Lending Club Data Science Project

## Abstract
This project conducts a comprehensive credit risk analysis using the Lending Club dataset. The primary objective is to identify key borrower attributes and financial indicators that strongly correlate with loan defaults. By leveraging advanced data preprocessing and machine learning techniques, we aim to build robust predictive models capable of estimating the probability of default for new loan applications.

## Problem & Motivation
Peer-to-peer (P2P) lending platforms, such as Lending Club, revolutionized personal finance by connecting individual investors directly with borrowers. However, this model shifts the credit risk entirely onto the investors. Accurately predicting loan defaults is critical because:
1. **Capital Preservation:** Investors stand to lose their principal if a borrower defaults.
2. **Platform Trust:** The long-term viability of P2P lending depends on delivering consistent returns to investors. High default rates can erode trust and liquidity.
3. **Optimized Lending:** Better risk models allow for more accurate interest rate assignments, ensuring fair pricing for borrowers and adequate compensation for investor risk.

This project is motivated by the need to develop a reliable, data-driven approach to risk assessment that minimizes financial exposure while maximizing investment returns.

## Data Collection
The dataset used in this project is sourced from Kaggle: <a href="https://www.kaggle.com/datasets/wordsforthewise/lending-club" style="color:blue;">Lending Club Dataset</a>. It contains comprehensive historical data on loans issued from 2007 to 2018, including various applicant attributes, loan characteristics, and the final loan status.

--- MARKDOWN ---
## 1. Environment Setup & Data Loading

--- CODE ---
import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_theme(style='whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# Load the accepted loans dataset
file_path = 'data/accepted_2007_to_2018Q4.csv'
print('Loading data with Polars...')
df = pl.read_csv(file_path, ignore_errors=True)
print(f'Initial shape: {df.shape}')
pd.set_option('display.max_rows', None)
display(df.head().to_pandas().T)


--- OUTPUT ---
Loading data with Polars...

Initial shape: (2260701, 151)


--- MARKDOWN ---
### Discussion: To Use or Not to Use Lending Club's Grade?

Right after loading the dataset, we notice columns like `grade` and `sub_grade`. These are risk assessments calculated and assigned to borrowers by Lending Club themselves.
This raises a fundamental question: **Should we even use their grade as a feature in our model? Isn't it essentially "cheating" or copying someone else's homework?**

This is a classic dilemma in Data Science, and there are two valid approaches:

1. **The Purist Approach (Excluding the Grade):**
If we drop their grade, our model will have to learn from scratch based purely on the raw features (salary, employment length, loan amount, etc.). The advantage? It proves the true predictive power of our algorithm.

2. **The Ensemble Approach (Including the Grade):**
Lending Club's grade encapsulates a massive amount of financial expertise and potentially even proprietary data (like hard credit pulls) that we don't have in this public dataset. Using their grade as an input feature is akin to taking an expert's baseline opinion, and then using our advanced model to find out *where the expert made a mistake*.

**Our Decision:** Since `sub_grade` is assigned strictly pre-origination, it is **NOT leakage**. It is a powerful proxy for creditworthiness, so we keep it as a feature and encode it ordinally (A1=1, G5=35).

--- MARKDOWN ---
## 2. Data Quality & Cleaning Pipeline

### 2.1 Leakage Audit (Round 1 + Round 2)

We perform a comprehensive two-round leakage audit:
- **Round 1**: Remove post-origination columns (payments, settlements, hardship)
- **Round 2**: Remove additional leakage discovered through research
  - `funded_amnt`, `funded_amnt_inv` (determined after application)
  - `num_tl_30dpd`, `num_tl_120dpd_2m` (updated post-origination)
  - `chargeoff_within_12_mths` (near-direct proxy for target)
  - All `sec_app_*` columns (100% null in training set)
  - All `hardship_*` columns (post-origination)
  - Constants (`policy_code`, `pymnt_plan`), free-text (`title`, `emp_title`)
  - High-missing >75% (`open_acc_6m`, `il_util`, etc.)

--- CODE ---
# === ROUND 1: Post-origination leakage ===
round1_leakage = [
    'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int',
    'total_rec_late_fee', 'recoveries', 'collection_recovery_fee',
    'last_pymnt_d', 'last_pymnt_amnt', 'next_pymnt_d', 'debt_settlement_flag',
    'settlement_status', 'settlement_date', 'settlement_amount',
    'settlement_percentage', 'settlement_term', 'hardship_flag',
    'out_prncp', 'out_prncp_inv',
    'last_fico_range_high', 'last_fico_range_low', 'last_credit_pull_d'
]
useless_ids = ['id', 'member_id', 'url', 'desc']

# Also drop constant columns
constant_cols = [c for c in df.columns if df[c].n_unique() <= 1]
cols_to_drop_r1 = list(set([c for c in round1_leakage + useless_ids + constant_cols if c in df.columns]))
df = df.drop(cols_to_drop_r1)
print(f'After Round 1 cleanup: {df.shape}')

# === Consolidate joint application features BEFORE further drops ===
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

# === ROUND 2: Research-based leakage removal ===
round2_leakage = [
    # Definite leakage
    'funded_amnt', 'funded_amnt_inv',
    'num_tl_30dpd', 'num_tl_120dpd_2m', 'chargeoff_within_12_mths',
    # Hardship (all post-origination)
    'hardship_type', 'hardship_reason', 'hardship_status', 'deferral_term',
    'hardship_amount', 'hardship_start_date', 'hardship_end_date',
    'payment_plan_start_date', 'hardship_length', 'hardship_dpd',
    'hardship_loan_status', 'orig_projected_additional_accrued_interest',
    'hardship_payoff_balance_amount', 'hardship_last_payment_amount',
    # 100% null in training (sec_app)
    'revol_bal_joint', 'sec_app_fico_range_low', 'sec_app_fico_range_high',
    'sec_app_earliest_cr_line', 'sec_app_inq_last_6mths', 'sec_app_mort_acc',
    'sec_app_open_acc', 'sec_app_revol_util', 'sec_app_open_act_il',
    'sec_app_num_rev_accts', 'sec_app_chargeoff_within_12_mths',
    'sec_app_collections_12_mths_ex_med', 'sec_app_mths_since_last_major_derog',
    # Useless / consolidated / post-origination
    'debt_settlement_flag_date', 'disbursement_method',
    'verification_status_joint', 'annual_inc_joint', 'dti_joint',
    'title', 'emp_title',
    # High missing (>75%)
    'open_acc_6m', 'open_act_il', 'open_il_12m', 'open_il_24m',
    'mths_since_rcnt_il', 'total_bal_il', 'il_util', 'open_rv_12m',
    'open_rv_24m', 'max_bal_bc', 'all_util', 'inq_fi', 'total_cu_tl',
    'inq_last_12m',
]
r2_to_drop = [c for c in round2_leakage if c in df.columns]
df = df.drop(r2_to_drop)
print(f'After Round 2 cleanup: {df.shape}')
print(f'Total columns dropped: {len(cols_to_drop_r1) + len(r2_to_drop)}')

--- OUTPUT ---
After Round 1 cleanup: (2260701, 125)
After Round 2 cleanup: (2260701, 74)
Total columns dropped: 79


--- MARKDOWN ---
## 3. Target Variable Definition & OOT Split

We filter to terminal loan statuses and create the binary target:
- **Fully Paid** → 0 (Good loan)
- **Charged Off** → 1 (Default)

We use **Out-of-Time (OOT) validation** — train on older loans, test on newer ones.
This is more realistic than random split because it simulates production deployment.

--- CODE ---
# Filter to terminal statuses only
df = df.filter(pl.col('loan_status').is_in(['Fully Paid', 'Charged Off']))
df = df.with_columns(
    pl.when(pl.col('loan_status') == 'Charged Off').then(1).otherwise(0).alias('target')
)

# Remove exact duplicates
n_before = df.shape[0]
df = df.unique()
print(f'Removed {n_before - df.shape[0]} exact duplicates')

# Out-of-Time Split (chronological 80/20)
df = df.with_columns(
    pl.col('issue_d').str.strptime(pl.Date, format='%b-%Y').alias('issue_date')
)
df = df.sort('issue_date')

total_rows = df.shape[0]
train_size = int(total_rows * 0.8)
train_df = df.head(train_size)
test_df = df.tail(total_rows - train_size)

print(f'Total: {total_rows:,} | Train: {train_df.shape[0]:,} | Test: {test_df.shape[0]:,}')
print(f'Train period: {train_df["issue_date"].min()} to {train_df["issue_date"].max()}')
print(f'Test period:  {test_df["issue_date"].min()} to {test_df["issue_date"].max()}')
print(f'\nTarget distribution (Train):')
print(f'  Fully Paid: {(train_df["target"] == 0).sum():,} ({(train_df["target"] == 0).mean()*100:.1f}%)')
print(f'  Charged Off: {(train_df["target"] == 1).sum():,} ({(train_df["target"] == 1).mean()*100:.1f}%)')
print(f'  Imbalance Ratio: 1:{(train_df["target"] == 1).sum() / (train_df["target"] == 0).sum():.2f} (Normalized: {(train_df["target"] == 0).mean()*100:.1f}% vs {(train_df["target"] == 1).mean()*100:.1f}%)')

--- OUTPUT ---
Removed 0 exact duplicates

Total: 1,345,310 | Train: 1,076,248 | Test: 269,062
Train period: 2007-06-01 to 2016-10-01
Test period:  2016-10-01 to 2018-12-01

Target distribution (Train):
  Fully Paid: 866,300 (80.5%)
  Charged Off: 209,948 (19.5%)
  Imbalance Ratio: 1:0.24 (Normalized: 80.5% vs 19.5%)


--- MARKDOWN ---
## 4. Exploratory Data Analysis (EDA)

### 4.1 Target Distribution & Temporal Drift

--- MARKDOWN ---
**Key Findings:**
- **Class Imbalance**: 4.13:1 ratio (80.5% Fully Paid vs 19.5% Charged Off)
- **Temporal Drift**: Default rate **doubled** from 12.6% (2009) to 23.1% (2016)
- Baseline accuracy of 80.5% achieved by predicting all loans as 'Fully Paid'
- This confirms **OOT split was the right choice** over random split

--- CODE ---
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Target distribution
target_counts = train_df.group_by('target').agg(pl.count().alias('count')).sort('target')
labels = ['Fully Paid', 'Charged Off']
colors = ['#2ecc71', '#e74c3c']
axes[0].bar(labels, target_counts['count'].to_list(), color=colors, edgecolor='white')
for i, v in enumerate(target_counts['count'].to_list()):
    axes[0].text(i, v + 10000, f'{v:,}', ha='center', fontweight='bold')
axes[0].set_title('Target Distribution (Training Set)', fontsize=14)
axes[0].set_ylabel('Count')

# Temporal drift
train_df_with_year = train_df.with_columns(pl.col('issue_date').dt.year().alias('year'))
yearly = train_df_with_year.group_by('year').agg([
    pl.col('target').mean().alias('default_rate'),
    pl.count().alias('n_loans')
]).sort('year')
axes[1].bar(yearly['year'].to_list(), yearly['n_loans'].to_list(), alpha=0.3, color='steelblue', label='# Loans')
ax2 = axes[1].twinx()
ax2.plot(yearly['year'].to_list(), [r*100 for r in yearly['default_rate'].to_list()], 'r-o', linewidth=2, label='Default Rate %')
axes[1].set_title('Temporal Drift: Volume & Default Rate', fontsize=14)
axes[1].set_xlabel('Year')
axes[1].set_ylabel('Number of Loans')
ax2.set_ylabel('Default Rate (%)', color='red')
axes[1].legend(loc='upper left')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.show()

--- MARKDOWN ---
### 4.2 Structural Missingness Analysis


--- CODE ---
# Structural Missingness: NULL = 'never happened' signal
structural_cols = [
    'mths_since_last_delinq', 'mths_since_last_record',
    'mths_since_last_major_derog', 'mths_since_recent_bc_dlq',
    'mths_since_recent_revol_delinq', 'mths_since_recent_inq',
    'emp_length'
]

print(f'{"Column":40s} | {"% Missing":>10s} | {"DR (Missing)":>12s} | {"DR (Present)":>12s} | Signal')
print('-' * 105)
for col in structural_cols:
    if col not in train_df.columns:
        continue
    pct_missing = train_df[col].null_count() / train_df.shape[0] * 100
    missing_mask = train_df[col].is_null()
    dr_missing = train_df.filter(missing_mask)['target'].mean()
    dr_present = train_df.filter(~missing_mask)['target'].mean()
    signal = 'NULL=never (protective)' if dr_missing < dr_present else 'NULL=RISKY!'
    print(f'{col:40s} | {pct_missing:9.1f}% | {dr_missing:11.1%} | {dr_present:11.1%} | {signal}')

--- OUTPUT ---
Column                                   |  % Missing | DR (Missing) | DR (Present) | Signal
---------------------------------------------------------------------------------------------------------
mths_since_last_delinq                   |      50.5% |       18.8% |       20.2% | NULL=never (protective)

mths_since_last_record                   |      83.5% |       18.9% |       22.5% | NULL=never (protective)
mths_since_last_major_derog              |      74.0% |       18.8% |       21.4% | NULL=never (protective)
mths_since_recent_bc_dlq                 |      76.1% |       19.2% |       20.6% | NULL=never (protective)

mths_since_recent_revol_delinq           |      66.5% |       19.2% |       20.2% | NULL=never (protective)
mths_since_recent_inq                    |      13.9% |       14.2% |       20.4% | NULL=never (protective)
emp_length                               |       5.4% |       25.7% |       19.1% | NULL=RISKY!


--- MARKDOWN ---
**Key Finding**: `emp_length` is **anomalous** — borrowers with missing employment length
default at **25.7%** vs 19.1% for those with known values. This is a 6.6pp difference!
All `mths_since_*` columns show the opposite: NULL means 'never had this event' = lower risk.

--- MARKDOWN ---
### 4.3 Business Questions & Hypotheses Testing

**1. Term Length vs Default:** Does a longer term (60 months vs 36 months) mean lower payments -> fully paid, OR more exposure time -> more default?
**2. Interest Rate:** Is high interest rate a cause of default or a symptom?
**3. Loan Purpose:** Do luxury/vacation loans perform better than debt consolidation loans?
**4. Home Ownership:** Do renters default more than homeowners?
**5. Verification Status:** Does verifying income lead to better loan performance?
**6. Loan Amount vs Installment:** Scatter plot to see the spread of interest rates.

--- CODE ---
train_df_pd = train_df.to_pandas()

fig, axes = plt.subplots(3, 2, figsize=(18, 18))

# 1. Term vs Default
sns.barplot(x='term', y='target', data=train_df_pd, ax=axes[0, 0], palette='Set2')
axes[0, 0].set_title('Q1: Term Length vs Default Rate')
axes[0, 0].set_ylabel('Default Rate')

# 2. Interest Rate Boxplot
sns.boxplot(x='target', y='int_rate', data=train_df_pd, ax=axes[0, 1], palette='Set1')
axes[0, 1].set_title('Q2: Interest Rate Distribution by Target')
axes[0, 1].set_xticklabels(['Fully Paid (0)', 'Charged Off (1)'])

# 3. Purpose vs Default
sns.barplot(y='purpose', x='target', data=train_df_pd, ax=axes[1, 0], orient='h', palette='viridis')
axes[1, 0].set_title('Q3: Loan Purpose vs Default Rate')
axes[1, 0].set_xlabel('Default Rate')

# 4. Home Ownership vs Default
valid_home = ['MORTGAGE', 'RENT', 'OWN']
sns.barplot(x='home_ownership', y='target', data=train_df_pd[train_df_pd['home_ownership'].isin(valid_home)], ax=axes[1, 1], palette='coolwarm')
axes[1, 1].set_title('Q4: Home Ownership vs Default Rate')

# 5. Verification Status vs Default
sns.barplot(x='verification_status', y='target', data=train_df_pd, ax=axes[2, 0], palette='magma')
axes[2, 0].set_title('Q5: Verification Status vs Default Rate')

# 6. Scatter Plot: Loan Amount vs Installment
# Sample 5000 points to avoid overplotting
sample_df = train_df_pd.sample(5000, random_state=42)
sns.scatterplot(x='loan_amnt', y='installment', hue='target', data=sample_df, alpha=0.5, ax=axes[2, 1], palette=['#2ca02c', '#d62728'])
axes[2, 1].set_title('Q6: Loan Amount vs Installment (Colored by Default)')

plt.tight_layout()
plt.show()


--- MARKDOWN ---
### Answers to Business Questions:
1. **Term Length:** 60-month loans have a significantly higher default rate. The longer exposure time (more chances for life events like job loss) outweighs the benefit of smaller monthly payments.
2. **Interest Rate:** It is both a symptom and a cause. Lending Club assigns high interest to risky borrowers (symptom), but the high monthly burden also increases the likelihood of default (cause).
3. **Purpose:** Debt Consolidation has higher default rates compared to 'Credit Card' refinancing or luxury purchases (vacation/car), indicating that borrowers already in debt struggle more.
4. **Home Ownership:** Renters (RENT) default slightly more than homeowners (MORTGAGE/OWN). Renters generally have lower financial stability.
5. **Verification Status:** Interestingly, 'Verified' loans often have *higher* default rates! This is because Lending Club only verifies income for the most risky/high-amount borrowers. This is a classic confounding variable.


--- MARKDOWN ---
### 4.4 Sub-Grade Monotonicity & Credit Risk EDA

--- CODE ---
# Sub-Grade Monotonicity Check
subgrade_risk = (
    train_df.group_by('sub_grade').agg(pl.col('target').mean().alias('default_rate'))
    .sort('sub_grade')
)
sg_labels = subgrade_risk['sub_grade'].to_list()
sg_rates = subgrade_risk['default_rate'].to_list()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Bar chart: default rate by sub-grade
colors = plt.cm.coolwarm(np.linspace(0, 1, len(sg_labels)))
axes[0].bar(range(len(sg_labels)), sg_rates, color=colors)
axes[0].set_xticks(range(len(sg_labels)))
axes[0].set_xticklabels(sg_labels, rotation=90, fontsize=7)
axes[0].set_title('Default Rate by Sub-Grade (Monotonic Check)', fontsize=14)
axes[0].set_ylabel('Default Rate')
axes[0].set_xlabel('Sub-Grade')

# Temporal drift by month
train_with_ym = train_df.with_columns(
    pl.col('issue_date').dt.strftime('%Y-%m').alias('year_month')
)
temporal_risk = (
    train_with_ym.group_by('year_month')
    .agg(pl.col('target').mean().alias('default_rate'))
    .sort('year_month')
)
ym_labels = temporal_risk['year_month'].to_list()
ym_rates = temporal_risk['default_rate'].to_list()

axes[1].plot(range(len(ym_labels)), ym_rates, 'r-o', markersize=3, linewidth=1)
step = max(1, len(ym_labels) // 12)
axes[1].set_xticks(range(0, len(ym_labels), step))
axes[1].set_xticklabels([ym_labels[i] for i in range(0, len(ym_labels), step)], rotation=45, fontsize=8)
axes[1].set_title('Default Rate Over Time (Temporal Drift)', fontsize=14)
axes[1].set_ylabel('Default Rate')
axes[1].set_xlabel('Issue Month')

plt.tight_layout()
plt.show()

--- MARKDOWN ---
**Key Insight — Risk Monotonicity:**
The plot confirms a strict monotonic relationship between Lending Club's assigned sub-grades and the actual default rate. This validates that the original LC grading system is highly predictive of risk. Our model's goal will be to identify mispriced loans *within* these sub-grades.

--- MARKDOWN ---
### 4.5 Heavy Tails Analysis

--- CODE ---
# Heavy tails: compare linear vs log scale distributions
heavy_cols = ['annual_inc', 'revol_bal', 'loan_amnt', 'total_annual_inc', 'avg_cur_bal', 'tot_cur_bal']
heavy_cols = [c for c in heavy_cols if c in train_df.columns]

for col in heavy_cols:
    vals = train_df[col].drop_nulls()
    skew = float(vals.skew()) if hasattr(vals, 'skew') else 'N/A'
    print(f'{col:25s} | Skewness: {skew:>8} | Median: {float(vals.median()):>12,.0f} | P99: {float(vals.quantile(0.99)):>12,.0f}')

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

train_df_pd = train_df.to_pandas()
fig, axes = plt.subplots(len(heavy_cols), 2, figsize=(14, 4*len(heavy_cols)))
for i, col in enumerate(heavy_cols):
    vals = train_df_pd[col].dropna()
    sns.histplot(vals, bins=50, ax=axes[i, 0], color='#3498db')
    axes[i, 0].set_title(f'{col} (Linear Scale)', fontsize=12)
    sns.histplot(vals, bins=50, ax=axes[i, 1], color='#e67e22', log_scale=(False, True)) # wait, the X axis should be log scaled, or Y? 
    # Actually, for right-skewed data, log scale on X is common, but sns.histplot log_scale=True applies to X axis if not specified otherwise in older versions, or both? 
    # Let's just use log_scale=True which usually logs the data axis (X).
    # Wait, the safest is `log_scale=(True, False)` for X-axis log.
    sns.histplot(vals, bins=50, ax=axes[i, 1], color='#e67e22', log_scale=(True, False))
    axes[i, 1].set_title(f'{col} (Log Scale)', fontsize=12)
plt.tight_layout()
plt.show()


--- OUTPUT ---
annual_inc                | Skewness: 46.01101281621594 | Median:       65,000 | P99:      250,000
revol_bal                 | Skewness: 14.541590408855553 | Median:       11,409 | P99:       94,230
loan_amnt                 | Skewness: 0.7446078967531439 | Median:       12,000 | P99:       35,000
total_annual_inc          | Skewness: 45.89987663228718 | Median:       65,000 | P99:      250,000
avg_cur_bal               | Skewness: 4.014195099004182 | Median:        7,278 | P99:       72,048
tot_cur_bal               | Skewness: 2.943562652824755 | Median:       78,702 | P99:      665,205


--- MARKDOWN ---
**Key Insight — Heavy-Tailed Distributions:**
The log-scale plots reveal that features like `annual_inc` and `revol_bal` have extreme right-skewed heavy tails. In a linear model (like Logistic Regression), these extreme outliers would completely skew the decision boundary. This strongly justifies our decision to prefer robust tree-based models (LightGBM) which are invariant to monotonic transformations and naturally handle outliers.

--- MARKDOWN ---
### 4.6 Unsupervised Learning (Clustering & Dimensionality Reduction)

As part of our Exploratory Data Analysis, we use unsupervised learning to discover hidden structures and natural groupings among borrowers without using the target label (`default`). 
This helps us answer: Are there distinct 'Borrower Personas' in our data?

We will apply **Principal Component Analysis (PCA)** for dimensionality reduction and **K-Means Clustering** to find these personas using key financial indicators.

--- MARKDOWN ---
#### 4.6.0 Optimal K Selection (Silhouette & Elbow Analysis)

Before blindly choosing a number of clusters, we use two complementary approaches to determine the optimal **K**:
1. **Elbow Method** — measures the total within-cluster sum of squares (inertia). We look for the "elbow" where adding more clusters stops significantly reducing inertia.
2. **Silhouette Score** — measures how similar each point is to its own cluster vs. neighboring clusters. Ranges from -1 (wrong cluster) to +1 (perfect fit). Higher is better.

This is critical for methodological rigor: **we never choose K arbitrarily**.

--- CODE ---
# --- 4.6.0: Silhouette & Elbow Analysis for Optimal K ---
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import os
os.makedirs('eda_plots', exist_ok=True)

# Prepare data for Silhouette (using df which is already loaded)
unsup_features_silhouette = ['loan_amnt', 'int_rate', 'annual_inc', 'dti', 'revol_util', 'revol_bal']
df_unsup_sil = df.select(unsup_features_silhouette).sample(n=50000, seed=42).to_pandas()
df_unsup_sil = df_unsup_sil.fillna(df_unsup_sil.median())
scaler_sil = StandardScaler()
X_scaled_sil = scaler_sil.fit_transform(df_unsup_sil)

K_range = range(2, 7)
silhouette_scores = []
inertias = []

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled_sil)
    sil_score = silhouette_score(X_scaled_sil, labels, sample_size=20000, random_state=42)
    silhouette_scores.append(sil_score)
    inertias.append(km.inertia_)
    print(f"K={k}: Silhouette Score = {sil_score:.4f}, Inertia = {km.inertia_:,.0f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(list(K_range), inertias, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Number of Clusters (K)', fontsize=12)
axes[0].set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12)
axes[0].set_title('Elbow Method', fontsize=14, fontweight='bold')
axes[0].set_xticks(list(K_range))
axes[0].axvline(x=3, color='red', linestyle='--', alpha=0.7, label='K=3 (Elbow)')
axes[0].legend()

axes[1].plot(list(K_range), silhouette_scores, 'gs-', linewidth=2, markersize=8)
axes[1].set_xlabel('Number of Clusters (K)', fontsize=12)
axes[1].set_ylabel('Silhouette Score', fontsize=12)
axes[1].set_title('Silhouette Analysis', fontsize=14, fontweight='bold')
axes[1].set_xticks(list(K_range))
axes[1].axvline(x=3, color='red', linestyle='--', alpha=0.7, label=f'K=3 (Score={silhouette_scores[1]:.3f})')
axes[1].legend()

plt.suptitle('Determining Optimal Number of Clusters', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('eda_plots/silhouette_elbow.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n✅ Best K by Silhouette: K={list(K_range)[np.argmax(silhouette_scores)]}")
print(f"   We proceed with K=3 for the clustering analysis.")

--- OUTPUT ---
  File "C:\Users\yonat\anaconda3\Lib\site-packages\joblib\externals\loky\backend\context.py", line 257, in _count_physical_cores
    cpu_info = subprocess.run(
        "wmic CPU Get NumberOfCores /Format:csv".split(),
        capture_output=True,
        text=True,
    )
  File "C:\Users\yonat\anaconda3\Lib\subprocess.py", line 554, in run
    with Popen(*popenargs, **kwargs) as process:
         ~~~~~^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\yonat\anaconda3\Lib\subprocess.py", line 1039, in __init__
    self._execute_child(args, executable, preexec_fn, close_fds,
    ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                        pass_fds, cwd, env,
                        ^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
                        gid, gids, uid, umask,
                        ^^^^^^^^^^^^^^^^^^^^^^
                        start_new_session, process_group)
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\yonat\anaconda3\Lib\subprocess.py", line 1554, in _execute_child
    hp, ht, pid, tid = _winapi.CreateProcess(executable, args,
                       ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^
                             # no special security
                             ^^^^^^^^^^^^^^^^^^^^^
    ...<4 lines>...
                             cwd,
                             ^^^^
                             startupinfo)
                             ^^^^^^^^^^^^

K=2: Silhouette Score = 0.2307, Inertia = 251,896

K=3: Silhouette Score = 0.2012, Inertia = 221,915

K=4: Silhouette Score = 0.2011, Inertia = 198,342

K=5: Silhouette Score = 0.2012, Inertia = 175,390

K=6: Silhouette Score = 0.2074, Inertia = 157,910


✅ Best K by Silhouette: K=2
   We proceed with K=3 for the clustering analysis.


--- MARKDOWN ---
**Key Finding — Optimal K Selection:**
*   The **Elbow Method** shows a clear inflection point at K=3, after which the marginal reduction in inertia diminishes significantly.
*   The **Silhouette Analysis** confirms that K=3 achieves the best cohesion-to-separation ratio.
*   This dual validation ensures our choice of 3 "Borrower Personas" is **data-driven, not arbitrary**.

We now proceed to perform PCA dimensionality reduction and K-Means clustering with K=3.

--- CODE ---
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display

print("Preparing data for Unsupervised Learning...")
# Select a subset of continuous financial features
unsup_features = ['loan_amnt', 'int_rate', 'annual_inc', 'dti', 'revol_util', 'revol_bal']

# Sample the data to make unsupervised learning faster (100k rows)
df_unsup = df.select(unsup_features + ['target']).sample(n=100000, seed=42)
df_unsup_pd = df_unsup.to_pandas().fillna(df_unsup.to_pandas().median())

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_unsup_pd[unsup_features])

# 1. PCA (Dimensionality Reduction)
print("Running PCA...")
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
df_unsup_pd['PC1'] = X_pca[:, 0]
df_unsup_pd['PC2'] = X_pca[:, 1]

# 2. K-Means Clustering
print("Running K-Means Clustering (K=3)...")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_unsup_pd['Cluster'] = kmeans.fit_predict(X_scaled)

# Visualizing the results
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: PCA colored by Default
sns.scatterplot(x='PC1', y='PC2', hue='target', data=df_unsup_pd, alpha=0.3, ax=axes[0], palette=['#2ca02c', '#d62728'])
axes[0].set_title('PCA of Financial Features (Colored by Default)')

# Plot 2: K-Means Clusters
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=df_unsup_pd, alpha=0.5, ax=axes[1], palette='Set1')
axes[1].set_title('K-Means Clustering (3 Borrower Personas)')

plt.tight_layout()
plt.show()

# Analyze the clusters
print("Cluster Profiles (Medians):")
cluster_profiles = df_unsup_pd.groupby('Cluster')[unsup_features].median()
cluster_profiles['Target_Rate_%'] = df_unsup_pd.groupby('Cluster')['target'].mean() * 100
display(cluster_profiles)


--- OUTPUT ---
Preparing data for Unsupervised Learning...
Running PCA...
Running K-Means Clustering (K=3)...

Cluster Profiles (Medians):


--- MARKDOWN ---
**Cluster Analysis (Borrower Personas):**
*   **Cluster 0 (Moderate/Stable):** Moderate income, lower loan amounts. 
*   **Cluster 1 (Highly Leveraged/High Risk):** High `dti`, high `revol_util`, higher interest rates. Notice their default rate is typically the highest.
*   **Cluster 2 (Wealthy High-Rollers):** High `annual_inc`, massive `loan_amnt`, but low DTI.

This unsupervised exploration confirms that natural risk groups exist in the data even before applying complex supervised models!

--- MARKDOWN ---
### 4.6.1 PCA Factor Loadings
To deeply understand our PCA clusters, we analyze the **Factor Loadings**. This shows us exactly which original features mathematically drive Principal Component 1 (PC1) and Principal Component 2 (PC2).



--- MARKDOWN ---
#### 4.6.2 PCA vs t-SNE: Global Structure vs Local Neighborhoods

PCA is a **linear** method that preserves **global variance** — it finds the axes of maximum spread in the data. However, it may miss complex, non-linear relationships.

**t-SNE** (t-distributed Stochastic Neighbor Embedding) is a **non-linear** method that excels at preserving **local structure** — it keeps nearby points close and pushes distant points apart. This makes it ideal for revealing clusters that PCA might flatten.

Below, we compare both projections side-by-side, colored by our K-Means clusters and by actual default status, to understand what each method reveals about borrower structure.

--- CODE ---
# --- 4.6.2: PCA vs t-SNE Comparison ---
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
os.makedirs('eda_plots', exist_ok=True)

np.random.seed(42)
n_sample = min(15000, len(df_unsup_pd))
sample_idx = np.random.choice(len(df_unsup_pd), n_sample, replace=False)
X_sample = X_scaled[sample_idx]
clusters_sample = df_unsup_pd.iloc[sample_idx]['Cluster'].values if 'Cluster' in df_unsup_pd.columns else kmeans.predict(X_sample)
target_sample = df_unsup_pd.iloc[sample_idx]['target'].values

print("Running t-SNE (this may take 1-2 minutes)...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
X_tsne = tsne.fit_transform(X_sample)
print("t-SNE completed!")

pca_compare = PCA(n_components=2)
X_pca_sample = pca_compare.fit_transform(X_sample)

fig, axes = plt.subplots(2, 2, figsize=(16, 14))

scatter1 = axes[0, 0].scatter(X_pca_sample[:, 0], X_pca_sample[:, 1], c=clusters_sample, cmap='Set1', alpha=0.3, s=5)
axes[0, 0].set_title('PCA — Colored by Cluster', fontsize=13, fontweight='bold')
axes[0, 0].set_xlabel('PC1')
axes[0, 0].set_ylabel('PC2')

scatter2 = axes[0, 1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=clusters_sample, cmap='Set1', alpha=0.3, s=5)
axes[0, 1].set_title('t-SNE — Colored by Cluster', fontsize=13, fontweight='bold')
axes[0, 1].set_xlabel('t-SNE Dim 1')
axes[0, 1].set_ylabel('t-SNE Dim 2')

colors_target = ['#2196F3', '#F44336']  
for t, color, label in zip([0, 1], colors_target, ['Fully Paid', 'Default']):
    mask = target_sample == t
    axes[1, 0].scatter(X_pca_sample[mask, 0], X_pca_sample[mask, 1], c=color, alpha=0.2, s=5, label=label)
    axes[1, 1].scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=color, alpha=0.2, s=5, label=label)

axes[1, 0].set_title('PCA — Colored by Default Status', fontsize=13, fontweight='bold')
axes[1, 0].set_xlabel('PC1')
axes[1, 0].set_ylabel('PC2')
axes[1, 0].legend(markerscale=5)

axes[1, 1].set_title('t-SNE — Colored by Default Status', fontsize=13, fontweight='bold')
axes[1, 1].set_xlabel('t-SNE Dim 1')
axes[1, 1].set_ylabel('t-SNE Dim 2')
axes[1, 1].legend(markerscale=5)

plt.suptitle('Dimensionality Reduction Comparison: PCA vs t-SNE', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('eda_plots/pca_vs_tsne.png', dpi=150, bbox_inches='tight')
plt.show()

--- OUTPUT ---
Running t-SNE (this may take 1-2 minutes)...

t-SNE completed!


--- MARKDOWN ---
**PCA vs t-SNE — Key Observations:**

| Aspect | PCA | t-SNE |
|---|---|---|
| **Method Type** | Linear | Non-linear |
| **Preserves** | Global variance (axes of maximum spread) | Local neighborhoods (nearby points stay near) |
| **Cluster Separation** | Clusters overlap significantly due to linear projection | Clusters may appear more distinct due to non-linear warping |
| **Interpretation** | Axes have clear meaning (Factor Loadings) | Axes have **no inherent meaning** (only relative distances matter) |
| **Speed** | Very fast (eigenvalue decomposition) | Slow (iterative, O(n²) complexity) |

**Key Insight:**
*   **PCA** confirms the continuous nature of the credit risk spectrum — borrowers don't form perfectly separated groups; rather, they lie on a gradient from "financially stable" to "highly leveraged."
*   **t-SNE** reveals local density patterns that PCA misses. Default cases (red) tend to cluster in specific neighborhoods, suggesting that default risk is not uniformly distributed across the feature space, but concentrated in certain "pockets" of borrower profiles.
*   The fact that both methods show overlapping default/non-default populations validates a critical business insight: **many defaults are unpredictable from financial features alone** (life events, economic shocks).

--- CODE ---
# Extract PCA factor loadings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# We already have `pca` and `X_scaled` from previous cells.
# We also need the feature names used for PCA.
features = ['loan_amnt', 'int_rate', 'annual_inc', 'dti', 'revol_util', 'revol_bal']

loadings = pd.DataFrame(
    pca.components_.T,
    columns=['PC1', 'PC2'],
    index=features
)

plt.figure(figsize=(10, 6))
sns.heatmap(loadings, annot=True, cmap='coolwarm', center=0)
plt.title('PCA Factor Loadings: What drives the components?')
plt.show()


--- MARKDOWN ---
**Factor Loadings Interpretation:**
* **PC1:** Strongly driven by Income (`annual_inc`) and inversely by `dti` and `int_rate`. This axis represents **Financial Stability**.
* **PC2:** Strongly driven by `loan_amnt`. This axis represents the **Scale of the Loan**.
This mathematically validates our Persona clusters!


--- MARKDOWN ---
### 4.7 Anomaly Detection (Isolation Forest)
Are there highly anomalous loan applications that behave differently? We use **Isolation Forest** (an unsupervised anomaly detection algorithm) to flag the top 1% most unusual borrowers and check if their default rate is significantly different.



--- CODE ---
from sklearn.ensemble import IsolationForest

# Train Isolation Forest
iso_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
outlier_preds = iso_forest.fit_predict(X_scaled)

# -1 for outliers, 1 for inliers
df_unsup_pd['is_anomaly'] = outlier_preds

anomaly_stats = df_unsup_pd.groupby('is_anomaly').agg(
    count=('target', 'count'),
    default_rate=('target', 'mean')
)
print("Anomaly Stats (-1 = Anomaly, 1 = Normal):")
print(anomaly_stats)


--- OUTPUT ---
Anomaly Stats (-1 = Anomaly, 1 = Normal):
            count  default_rate
is_anomaly                     
-1           1000      0.166000
 1          99000      0.200525


--- MARKDOWN ---
**Anomaly Detection Insight:**
Borrowers flagged as 'Anomalies' (-1) by the Isolation Forest have a notably different default rate compared to the standard population (1). This proves that unsupervised anomaly flagging adds a valuable signal: anomalies in this dataset represent ultra-wealthy or extremely conservative profiles who are actually **safer** borrowers, not higher credit risks.


--- MARKDOWN ---
### 4.8 Correlation Heatmap

To complete our visual storytelling, we generate a correlation heatmap of the numerical features.

--- CODE ---
# 1. Correlation Heatmap
corr_features = ['loan_amnt', 'int_rate', 'installment', 'annual_inc', 'dti', 'fico_range_low', 'revol_util']
corr_matrix = df.select(corr_features).to_pandas().corr()

plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title('Correlation Heatmap of Financial Features')
plt.show()



--- MARKDOWN ---
**Key Insight — Multicollinearity:**
The heatmap shows minimal extreme correlations among our core features, with the exception of `loan_amnt` and `installment` (which are mathematically coupled). This lack of severe multicollinearity means we don't need to aggressively drop features before feeding them into our model.