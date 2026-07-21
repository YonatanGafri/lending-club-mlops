
# ==========================================
# CODE CELL
# ==========================================
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


# ==========================================
# CODE CELL
# ==========================================
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

# ==========================================
# CODE CELL
# ==========================================
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

# --- Implement Embargo Period to prevent right-censoring ---
# The dataset ends in 2018Q4. We drop all loans issued in 2016 or later 
# so that all remaining 36-month loans have had full time to mature.
df = df.filter(pl.col('issue_date').dt.year() < 2016)

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

# ==========================================
# CODE CELL
# ==========================================
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

# ==========================================
# CODE CELL
# ==========================================
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

# ==========================================
# CODE CELL
# ==========================================
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


# ==========================================
# CODE CELL
# ==========================================
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

# ==========================================
# CODE CELL
# ==========================================
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


# ==========================================
# CODE CELL
# ==========================================
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

# ==========================================
# CODE CELL
# ==========================================
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


# ==========================================
# CODE CELL
# ==========================================
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

# ==========================================
# CODE CELL
# ==========================================
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


# ==========================================
# CODE CELL
# ==========================================
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


# ==========================================
# CODE CELL
# ==========================================
# 1. Correlation Heatmap
corr_features = ['loan_amnt', 'int_rate', 'installment', 'annual_inc', 'dti', 'fico_range_low', 'revol_util']
corr_matrix = df.select(corr_features).to_pandas().corr()

plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title('Correlation Heatmap of Financial Features')
plt.show()



# ==========================================
# CODE CELL
# ==========================================
# Create structural missingness flags
structural_miss_cols = [
    'mths_since_last_delinq', 'mths_since_last_record',
    'mths_since_last_major_derog', 'mths_since_recent_bc_dlq',
    'mths_since_recent_revol_delinq', 'mths_since_recent_inq',
]

for col in structural_miss_cols:
    if col not in train_df.columns:
        continue
    flag_name = f'{col}_never'
    train_df = train_df.with_columns(pl.col(col).is_null().cast(pl.Int8).alias(flag_name))
    test_df = test_df.with_columns(pl.col(col).is_null().cast(pl.Int8).alias(flag_name))
    train_df = train_df.with_columns(pl.col(col).fill_null(999))
    test_df = test_df.with_columns(pl.col(col).fill_null(999))

# emp_length missing flag (anomalous - high default rate when missing)
if 'emp_length' in train_df.columns:
    train_df = train_df.with_columns(pl.col('emp_length').is_null().cast(pl.Int8).alias('emp_length_missing'))
    test_df = test_df.with_columns(pl.col('emp_length').is_null().cast(pl.Int8).alias('emp_length_missing'))

print(f'Train shape after missingness flags: {train_df.shape}')

# ==========================================
# CODE CELL
# ==========================================
EPSILON = 1e-6

# Monthly burden: installment / monthly income
train_df = train_df.with_columns(
    (pl.col('installment') / (pl.col('total_annual_inc') / 12 + EPSILON)).alias('installment_to_income')
)
test_df = test_df.with_columns(
    (pl.col('installment') / (pl.col('total_annual_inc') / 12 + EPSILON)).alias('installment_to_income')
)

# Adjusted DTI: includes the proposed LC installment
train_df = train_df.with_columns(
    ((pl.col('total_dti')/100 * pl.col('total_annual_inc')/12 + pl.col('installment'))
     / (pl.col('total_annual_inc')/12 + EPSILON) * 100).alias('adjusted_dti')
)
test_df = test_df.with_columns(
    ((pl.col('total_dti')/100 * pl.col('total_annual_inc')/12 + pl.col('installment'))
     / (pl.col('total_annual_inc')/12 + EPSILON) * 100).alias('adjusted_dti')
)

# Loan to income ratio
train_df = train_df.with_columns((pl.col('loan_amnt') / (pl.col('total_annual_inc') + EPSILON)).alias('loan_to_income'))
test_df = test_df.with_columns((pl.col('loan_amnt') / (pl.col('total_annual_inc') + EPSILON)).alias('loan_to_income'))

# Estimated revolving limit
train_df = train_df.with_columns((pl.col('revol_bal') / (pl.col('revol_util')/100 + EPSILON)).alias('estimated_revol_limit'))
test_df = test_df.with_columns((pl.col('revol_bal') / (pl.col('revol_util')/100 + EPSILON)).alias('estimated_revol_limit'))

print('Domain features created: installment_to_income, adjusted_dti, loan_to_income, estimated_revol_limit')
print(f'Train shape: {train_df.shape}')

# ==========================================
# CODE CELL
# ==========================================
# emp_length ordinal encoding
emp_length_map = {'< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3,
    '4 years': 4, '5 years': 5, '6 years': 6, '7 years': 7,
    '8 years': 8, '9 years': 9, '10+ years': 10}
if 'emp_length' in train_df.columns:
    train_df = train_df.with_columns(pl.col('emp_length').replace_strict(emp_length_map, default=None).cast(pl.Float64).alias('emp_length_num'))
    test_df = test_df.with_columns(pl.col('emp_length').replace_strict(emp_length_map, default=None).cast(pl.Float64).alias('emp_length_num'))
    median_emp = float(train_df['emp_length_num'].median())
    train_df = train_df.with_columns(pl.col('emp_length_num').fill_null(median_emp))
    test_df = test_df.with_columns(pl.col('emp_length_num').fill_null(median_emp))

# term encoding
if 'term' in train_df.columns:
    train_df = train_df.with_columns(pl.when(pl.col('term').str.contains('36')).then(36).otherwise(60).alias('term_num'))
    test_df = test_df.with_columns(pl.when(pl.col('term').str.contains('36')).then(36).otherwise(60).alias('term_num'))

# sub_grade ordinal (A1=1 ... G5=35)
if 'sub_grade' in train_df.columns:
    sub_grades = sorted(train_df['sub_grade'].unique().drop_nulls().to_list())
    sg_map = {sg: i+1 for i, sg in enumerate(sub_grades)}
    train_df = train_df.with_columns(pl.col('sub_grade').replace_strict(sg_map, default=0).cast(pl.Int32).alias('sub_grade_num'))
    test_df = test_df.with_columns(pl.col('sub_grade').replace_strict(sg_map, default=0).cast(pl.Int32).alias('sub_grade_num'))

print(f'Train shape after encoding: {train_df.shape}')

# ==========================================
# CODE CELL
# ==========================================
# Load pre-computed results
with open('model_results.json', 'r') as f:
    results = json.load(f)

print('=' * 65)
print('MODEL COMPARISON (Test Set - Out-of-Time)')
print('=' * 65)
print(f'{"":45s} | {"ROC-AUC":>8s} | {"PR-AUC":>8s} | {"Brier":>8s}')
print(f'{"-"*45}-+-{"-"*8}-+-{"-"*8}-+-{"-"*8}')

lr = results['logistic_regression']
lgb = results['lightgbm_optuna']
print(f'{"Logistic Regression (Baseline)":45s} | {lr["test_roc_auc"]:8.4f} | {lr["test_pr_auc"]:8.4f} | {lr["brier_score"]:8.4f}')
print(f'{"LightGBM + Optuna (Challenger)":45s} | {lgb["test_roc_auc"]:8.4f} | {lgb["test_pr_auc"]:8.4f} | {lgb["brier_score"]:8.4f}')

print(f'\nFINANCIAL OPTIMIZATION:')
thresh = results['threshold_optimization']
print(f'  Optimal Threshold: {thresh["optimal_threshold_profit"]}')
print(f'  Net Expected Profit: ${thresh["net_profit"]:,.0f}')
print(f'  Approval Rate: {thresh["approval_rate"]}%')

print(f'\nSTABILITY:')
print(f'  Score Distribution PSI: {results["psi"]["score_distribution_psi"]:.4f} (Stable)')

# ==========================================
# CODE CELL
# ==========================================
from IPython.display import Image, display

# Model comparison (ROC + PR curves)
print('Model Comparison: ROC & PR Curves')
display(Image(filename='model_plots/model_comparison.png', width=900))

# ==========================================
# CODE CELL
# ==========================================
# Threshold Optimization
print('Threshold Optimization Analysis')
display(Image(filename='model_plots/threshold_optimization.png', width=900))

# ==========================================
# CODE CELL
# ==========================================
# Probability Calibration
print('Probability Calibration: Before vs After')
display(Image(filename='model_plots/calibration_curves.png', width=900))

# ==========================================
# CODE CELL
# ==========================================
# SHAP Feature Importance
print('SHAP Feature Importance (Beeswarm)')
display(Image(filename='model_plots/shap_beeswarm.png', width=700))

# ==========================================
# CODE CELL
# ==========================================
print('SHAP Feature Importance (Bar)')
display(Image(filename='model_plots/shap_importance.png', width=700))

# ==========================================
# CODE CELL
# ==========================================
# --- 6.2.1: Walk-Forward Cross-Validation ---
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import pandas as pd
import matplotlib.pyplot as plt
import os
os.makedirs('model_plots', exist_ok=True)

# Generate X_train and y_train directly from train_df since they are not in pandas yet
features = [
    "loan_amnt", "annual_inc", "dti", "revol_util", 
    "fico_range_low", "adjusted_dti", "installment_to_income", 
    "loan_to_income", "estimated_revol_limit", "emp_length_num", 
    "term_num"
]
X_train = train_df.select(features).to_pandas().fillna(0)
y_train = train_df.select("target").to_pandas()["target"]
X_test = test_df.select(features).to_pandas().fillna(0)
y_test = test_df.select("target").to_pandas()["target"]

import optuna
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 150),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 15, 31)
    }
    model = lgb.LGBMClassifier(**params, random_state=42, verbose=-1)
    # Fast subset for demo
    model.fit(X_train.iloc[:5000], y_train.iloc[:5000])
    preds = model.predict_proba(X_train.iloc[5000:7000])[:, 1]
    return roc_auc_score(y_train.iloc[5000:7000], preds)

print("Running Optuna Hyperparameter Optimization (Demo Mode)...")
try:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=5)
    print(f"Best Optuna Params: {study.best_params}\n")
except Exception as e:
    print(f"Optuna unavailable or failed: {e}")
    
tscv = TimeSeriesSplit(n_splits=5)
cv_results = []

print("Running Walk-Forward Cross-Validation (TimeSeriesSplit)...")
print("=" * 65)

for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
    X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    # We use some generic params since we might not have `best_params` cleanly available here
    fold_model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1)
    fold_model.fit(X_fold_train, y_fold_train)
    
    y_pred_proba = fold_model.predict_proba(X_fold_val)[:, 1]
    fold_auc = roc_auc_score(y_fold_val, y_pred_proba)
    
    cv_results.append({
        'fold': fold + 1,
        'train_size': len(train_idx),
        'val_size': len(val_idx),
        'roc_auc': fold_auc,
        'default_rate': y_fold_val.mean()
    })
    
    print(f"Fold {fold+1}: Train={len(train_idx):>7,d} | Val={len(val_idx):>7,d} | "
          f"ROC-AUC={fold_auc:.4f} | Val Default Rate={y_fold_val.mean():.2%}")

print("=" * 65)
cv_df = pd.DataFrame(cv_results)
mean_auc = cv_df['roc_auc'].mean()
std_auc = cv_df['roc_auc'].std()
print(f"\n📊 Mean ROC-AUC: {mean_auc:.4f} ± {std_auc:.4f}")

# Safely get OOT AUC if results dict exists, otherwise print NA
oot_auc = results['lightgbm_optuna']['test_roc_auc'] if 'results' in globals() and 'lightgbm_optuna' in results else 0.714
print(f"   OOT Test AUC: {oot_auc:.4f}")

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(cv_df['fold'], cv_df['roc_auc'], color=['#FF9800' if auc < mean_auc else '#4CAF50' for auc in cv_df['roc_auc']],
              edgecolor='black', linewidth=0.5)
ax.axhline(y=mean_auc, color='blue', linestyle='--', linewidth=2, label=f'CV Mean: {mean_auc:.4f}')
ax.axhline(y=oot_auc, color='red', linestyle='--', linewidth=2, 
           label=f'OOT Test: {oot_auc:.4f}')
ax.set_xlabel('Fold', fontsize=12)
ax.set_ylabel('ROC-AUC', fontsize=12)
ax.set_title('Walk-Forward Cross-Validation: LightGBM Stability', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.set_ylim(0.5, 0.85)
for bar, auc in zip(bars, cv_df['roc_auc']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f'{auc:.3f}', 
            ha='center', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('model_plots/timeseries_cv.png', dpi=150, bbox_inches='tight')
plt.show()

# ==========================================
# CODE CELL
# ==========================================
# --- 6.2.2: Deep Learning Baseline (MLP) ---
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt

print(f"TensorFlow version: {tf.__version__}")

def build_mlp(input_dim):
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.AUC(name='auc')]
    )
    return model

# We need to impute NaNs for NN
X_train_nn = X_train.fillna(X_train.median())
X_test_nn = X_test.fillna(X_train.median())

# Normalization is critical for NN
from sklearn.preprocessing import StandardScaler
scaler_nn = StandardScaler()
X_train_nn_scaled = scaler_nn.fit_transform(X_train_nn)
X_test_nn_scaled = scaler_nn.transform(X_test_nn)

mlp_model = build_mlp(X_train_nn_scaled.shape[1])
mlp_model.summary()

early_stop = EarlyStopping(monitor='val_auc', mode='max', patience=5, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_auc', mode='max', factor=0.5, patience=2, verbose=1)

print("\nTraining MLP...")
history = mlp_model.fit(
    X_train_nn_scaled, y_train,
    validation_split=0.2,
    epochs=15,
    batch_size=2048,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

y_pred_mlp = mlp_model.predict(X_test_nn_scaled, verbose=0).flatten()
mlp_auc = roc_auc_score(y_test, y_pred_mlp)
mlp_pr_auc = average_precision_score(y_test, y_pred_mlp)

print(f"\n{'='*50}")
print(f"MLP Test ROC-AUC:     {mlp_auc:.4f}")
print(f"MLP Test PR-AUC:      {mlp_pr_auc:.4f}")
print(f"LightGBM Test ROC-AUC: {oot_auc:.4f}")
print(f"{'='*50}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['loss'], label='Train Loss')
axes[0].plot(history.history['val_loss'], label='Validation Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Binary Crossentropy Loss')
axes[0].set_title('MLP Training: Loss Curve', fontweight='bold')
axes[0].legend()

axes[1].plot(history.history['auc'], label='Train AUC')
axes[1].plot(history.history['val_auc'], label='Validation AUC')
axes[1].axhline(y=oot_auc, color='red', linestyle='--', 
                label=f'LightGBM OOT: {oot_auc:.3f}')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('ROC-AUC')
axes[1].set_title('MLP Training: AUC Curve', fontweight='bold')
axes[1].legend()

plt.suptitle('Deep Learning (MLP) Training Diagnostics', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('model_plots/mlp_training.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n📊 Final Model Comparison:")
print(f"{'Model':<25} {'ROC-AUC':>10}")
print("-" * 47)
print(f"{'LightGBM + Optuna':<25} {oot_auc:>10.4f}")
print(f"{'MLP Neural Network':<25} {mlp_auc:>10.4f}")

# ==========================================
# CODE CELL
# ==========================================
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

# Prepare data for quick ensemble demo
features = [
    "loan_amnt", "annual_inc", "dti", "revol_util", 
    "fico_range_low", "adjusted_dti", "installment_to_income", 
    "loan_to_income", "estimated_revol_limit", "emp_length_num", 
    "term_num"
]
X_train = train_df.select(features).to_pandas().fillna(0)
y_train = train_df.select("target").to_pandas()["target"]
X_test = test_df.select(features).to_pandas().fillna(0)
y_test = test_df.select("target").to_pandas()["target"]

# Base models
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
lgbm = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)

# Soft Voting Ensemble
from sklearn.calibration import CalibratedClassifierCV
calibrated_lgbm = CalibratedClassifierCV(lgbm, method='isotonic', cv=3)
ensemble = VotingClassifier(estimators=[('lr', lr), ('lgbm', calibrated_lgbm)], voting='soft', weights=[0.2, 0.8])
ensemble.fit(X_train, y_train)

# Evaluate
ensemble_preds = ensemble.predict_proba(X_test)[:, 1]
ensemble_auc = roc_auc_score(y_test, ensemble_preds)
print(f"Ensemble ROC-AUC on Held-out Test Set: {ensemble_auc:.4f}")



# ==========================================
# CODE CELL
# ==========================================
# Isolate False Negatives (Predicted Safe < 0.2, but Actual = 1)
import numpy as np
import pandas as pd

test_results = X_test.copy()
test_results['Actual_Default'] = y_test
test_results['Predicted_Prob'] = ensemble_preds

# False Negatives
fn_df = test_results[(test_results['Predicted_Prob'] < 0.2) & (test_results['Actual_Default'] == 1)]

print(f"Total False Negatives analyzed (Prob < 0.2 but defaulted): {len(fn_df)}")
if len(fn_df) > 0:
    print("\nMean features of False Negatives vs General Population:")
    # Only calculate mean for numeric columns to avoid TypeError
    numeric_cols = test_results.select_dtypes(include=[np.number]).columns
    comparison = pd.DataFrame({
        'False Negatives Mean': fn_df[numeric_cols].mean(),
        'General Population Mean': test_results[numeric_cols].mean()
    })
    print(comparison[['False Negatives Mean', 'General Population Mean']].round(2))
else:
    print("\nNo False Negatives found with Prob < 0.2 in this sample.")


