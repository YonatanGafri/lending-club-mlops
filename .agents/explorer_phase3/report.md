# Comprehensive Phase 3 Audit Report: Unsupervised Learning & Clustering

**Project:** Lending Club Data Science Project (`c:\Users\yonat\.gemini\antigravity\scratch\lending_club_ds`)  
**Auditor:** Investigator Agent 3 (Phase 3: Unsupervised Learning & Clustering)  
**Date:** July 27, 2026  
**Audit Criteria:** `ds_review_guide.md` Section 3 & `ORIGINAL_REQUEST.md` Requirement R4  

---

## Executive Summary

Phase 3 of the Lending Club project explores unsupervised learning techniques, specifically **Principal Component Analysis (PCA)**, **t-SNE (t-Distributed Stochastic Neighbor Embedding)**, **K-Means Clustering**, and **Isolation Forest Anomaly Detection**. 

While the phase contains high-quality side-by-side visualizations (e.g. PCA vs. t-SNE) and structured Markdown comparison tables, an exhaustive audit against the **Kaggle-Level 100/100 QA Guide** reveals several **critical flaws, logical contradictions, and theoretical omissions**.

### Overall Phase 3 Score: **82 / 100**

| Evaluation Sub-Category | Status | Score | Primary Reason for Score |
|---|:---:|:---:|---|
| **1. Smart Dimensionality Reduction (PCA)** | ⚠️ **PARTIAL FLAW** | 75/100 | Excellent PCA vs t-SNE side-by-side plot (`eda_plots/pca_vs_tsne.png`) and factor loadings analysis, but **`pca.explained_variance_ratio_` is completely absent** (never calculated, reported, or plotted). |
| **2. Mathematical Justification for $K$-Means ($K$ Selection)** | ❌ **CRITICAL FLAW** | 80/100 | Includes Elbow & Silhouette plots (`eda_plots/silhouette_elbow.png`), but contains a **logical contradiction**: Silhouette score peaks at $K=2$ (0.2307), yet text claims "dual validation" for selecting $K=3$ (0.2012). Feature set mismatch between $K$-selection (Cell 9) and $K$-Means execution (Cell 10). |
| **3. Cluster Profiling & Business Context** | ⚠️ **PARTIAL FLAW** | 90/100 | Profiles 3 distinct risk personas with median features and default rates in HTML output, but **omits all printed numerical metrics from Markdown text** (violating Forensic QA standards), and leaves clusters isolated without feeding them into Phase 4 models. |

---

## Detailed Audit Findings by Topic

---

### Topic 1: Smart Dimensionality Reduction (PCA) & t-SNE Comparison

#### A. Scope & Criteria
- **`ds_review_guide.md` Section 3**: *"האם יש הסבר מהי כמות השונות המוסברת (Explained Variance) שנשמרה לאחר ה-PCA?"* (Is there an explanation of the amount of explained variance ratio retained after PCA?)
- **`ORIGINAL_REQUEST.md` R4**: Check PCA variance explanation and t-SNE vs PCA comparisons.

#### B. Exact Code & File Locations
- **`Lending_Club_Project_Final.py`**:
  - Lines 523–526: PCA 2-component fit (`pca = PCA(n_components=2, random_state=42); X_pca = pca.fit_transform(X_scaled)`).
  - Lines 577–631 (Cell 11): t-SNE vs PCA side-by-side plot generation, saved to `eda_plots/pca_vs_tsne.png`.
  - Lines 653–670 (Cell 12): PCA Factor Loadings heatmap extraction (`pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=features)`).
- **`final_notebook_text.md`**:
  - Lines 171–173: PCA Factor Loadings section.
  - Lines 178–202: PCA vs t-SNE theoretical comparison table and key observations.
  - Lines 205–209: Factor Loadings interpretation.
- **Generated Plot Artifact**:
  - `eda_plots/pca_vs_tsne.png` (Size: 1,603,289 bytes).

---

#### C. Pass / Success Items
1. **PCA Implementation & Factor Loadings Analysis (Pass)**:
   - PCA is applied to continuous financial features in Cell 10 (`Lending_Club_Project_Final.py` lines 523–526).
   - Cell 12 (`Lending_Club_Project_Final.py` lines 661–665) explicitly extracts the eigenvector weights (`pca.components_.T`) into a pandas DataFrame and plots a Seaborn heatmap (`PCA Factor Loadings: What drives the components?`).
   - `final_notebook_text.md` (lines 205–209) provides high-level business interpretation of components:
     - **PC1**: Driven by Income (`annual_inc`) vs. DTI (`dti`) & Interest Rate (`int_rate`) $\rightarrow$ Represents **Financial Stability**.
     - **PC2**: Driven by Loan Amount (`loan_amnt`) $\rightarrow$ Represents **Scale of the Loan**.
2. **Side-by-Side PCA vs. t-SNE Visualization (Pass)**:
   - Cell 11 (`Lending_Club_Project_Final.py` lines 600–631) generates a 2x2 subplot comparison:
     - `axes[0, 0]`: PCA — Colored by Cluster
     - `axes[0, 1]`: t-SNE — Colored by Cluster
     - `axes[1, 0]`: PCA — Colored by Default Status
     - `axes[1, 1]`: t-SNE — Colored by Default Status
   - Saved properly as `eda_plots/pca_vs_tsne.png`.
3. **Structured Comparison Table (Pass)**:
   - `final_notebook_text.md` (lines 188–202) features a structured Markdown table comparing PCA and t-SNE across 5 dimensions:
     - *Method Type* (Linear vs. Non-linear)
     - *Preserved Structure* (Global variance vs. Local neighborhoods)
     - *Cluster Separation* (Linear overlap vs. Non-linear density warping)
     - *Interpretation* (Meaningful Factor Loadings vs. Arbitrary distance space)
     - *Computational Speed* ($O(N)$ SVD vs. $O(N^2)$ iterative optimization)

---

#### D. Flaws & Gaps
1. ❌ **CRITICAL FLAW — Total Absence of Explained Variance Ratio Calculation**:
   - `pca.explained_variance_ratio_` and `np.cumsum(pca.explained_variance_ratio_)` are **never calculated, printed, or plotted anywhere in the repository**.
   - Searching the entire codebase via `grep_search` for `explained_variance` yields **0 results**.
   - The student reduces 6 continuous financial variables down to 2 principal components without answering the fundamental mathematical question: **How much total variance in the original feature space is retained by PC1 and PC2?** (e.g. Does PC1 + PC2 explain 65%, 80%, or only 45% of total variance?).
   - This is a direct violation of `ds_review_guide.md` Section 3 ("האם יש הסבר מהי כמות השונות המוסברת שנשמרה לאחר ה-PCA?").
2. ⚠️ **Sample Size Inconsistency Between PCA Cells**:
   - Cell 10 runs PCA and $K$-Means on `n_sample = 100000` (`df_unsup_pd`).
   - Cell 11 runs t-SNE and `pca_compare` on `n_sample = 15000` (`X_sample`, line 586).
   - While sampling for t-SNE is necessary due to memory/time complexity, `pca_compare` in Cell 11 fits a separate `PCA(n_components=2)` instance on the 15k sample rather than transforming the existing fitted `pca` object from Cell 10.
3. ⚠️ **Omission of Exact Numerical Loading Values in Markdown Text**:
   - While Cell 12 displays the heatmap graphic, the exact numerical loading weights (e.g. `PC1(total_annual_inc) = +0.52`, `PC1(total_dti) = -0.44`) are missing from `final_notebook_text.md`.

---

#### E. Actionable Line-by-Line Recommendations for 100/100

1. **In `Lending_Club_Project_Final.py` (Cell 10 / Cell 12)**:
   - Add explicit calculation, printing, and plotting of cumulative explained variance ratio after line 526:
   ```python
   # --- Calculate and Report PCA Explained Variance Ratios ---
   var_exp = pca.explained_variance_ratio_
   cum_var_exp = np.cumsum(var_exp)

   print(f"PC1 Explained Variance: {var_exp[0]*100:.2f}%")
   print(f"PC2 Explained Variance: {var_exp[1]*100:.2f}%")
   print(f"Total Cumulative Variance Explained (PC1 + PC2): {cum_var_exp[1]*100:.2f}%")

   # Plot Scree / Cumulative Variance Plot
   plt.figure(figsize=(7, 4))
   plt.bar(range(1, 3), var_exp * 100, alpha=0.6, align='center', label='Individual Variance (%)', color='#1f77b4')
   plt.step(range(1, 3), cum_var_exp * 100, where='mid', label='Cumulative Variance (%)', color='#d62728', linewidth=2)
   plt.ylabel('Explained Variance Ratio (%)')
   plt.xlabel('Principal Component Index')
   plt.xticks([1, 2], ['PC1', 'PC2'])
   plt.title('PCA Explained Variance Ratio & Scree Plot')
   plt.legend(loc='best')
   plt.tight_layout()
   plt.savefig('eda_plots/pca_explained_variance.png', dpi=150, bbox_inches='tight')
   plt.show()
   ```

2. **In `final_notebook_text.md` (under Section 4.6.1)**:
   - Insert an explicit Markdown paragraph stating exact explained variance ratios:
   ```markdown
   **Mathematical Justification of Dimensionality Reduction (Explained Variance):**
   * **PC1** accounts for **XX.XX%** of the total variance across the financial feature space.
   * **PC2** accounts for **YY.YY%** of the total variance.
   * Together, the two principal components capture **ZZ.ZZ%** of the total variance, demonstrating that a 2D projection preserves the primary financial spread of borrower profiles while enabling clear visualization.
   ```

---

### Topic 2: Mathematical Justification for K-Means Clusters ($K$ Selection)

#### A. Scope & Criteria
- **`ds_review_guide.md` Section 3**: *"האם בחירת מספר האשכולות נעשתה בעזרת הצדקה מתמטית (Silhouette Score או Elbow Method) ולא כניחוש?"* (Is selection of number of clusters backed by mathematical metrics rather than guessing?)
- **`ORIGINAL_REQUEST.md` R4**: Mathematical justification for K-Means clusters (Silhouette Score analysis & Elbow Method plot).

#### B. Exact Code & File Locations
- **`Lending_Club_Project_Final.py`**:
  - Lines 438–489 (Cell 9): $K$-selection loop ($K \in [2, 6]$), inertia calculation, silhouette calculation (`silhouette_score(X_scaled_sil, labels, sample_size=20000)`), dual subplot creation.
  - Lines 492–496: Markdown text commentary on optimal $K$ selection.
- **`final_notebook_text.md`**:
  - Lines 143–158: Section 4.6.0 Optimal K Selection.
- **`Lending_Club_Project_Final.html`**:
  - Lines 9642–9670: Printed output showing exact Silhouette Scores and Inertia values.
- **Generated Plot Artifact**:
  - `eda_plots/silhouette_elbow.png` (Size: 130,516 bytes).

---

#### C. Pass / Success Items
1. **Automated Metric Evaluation Loop (Pass)**:
   - Cell 9 (`Lending_Club_Project_Final.py` lines 453–464) evaluates $K \in \{2, 3, 4, 5, 6\}$ using `KMeans(n_clusters=k, random_state=42, n_init=10)` and `silhouette_score`.
2. **Exact Printed Output Metrics (Pass)**:
   - Execution outputs exact metrics to terminal/notebook (`Lending_Club_Project_Final.html` lines 9642–9670):
     - **$K=2$**: Silhouette Score = **0.2307**, Inertia = **251,896**
     - **$K=3$**: Silhouette Score = **0.2012**, Inertia = **221,915**
     - **$K=4$**: Silhouette Score = **0.2011**, Inertia = **198,342**
     - **$K=5$**: Silhouette Score = **0.2012**, Inertia = **175,390**
     - **$K=6$**: Silhouette Score = **0.2074**, Inertia = **157,910**
3. **Dual Metric Visualization (Pass)**:
   - Plots side-by-side Elbow curve (left) and Silhouette Score curve (right), saved to `eda_plots/silhouette_elbow.png`.

---

#### D. Flaws & Gaps
1. ❌ **CRITICAL LOGICAL CONTRADICTION & METHODOLOGICAL FLAW**:
   - The code output explicitly prints: `Best K by Silhouette: K=2`.
   - $K=2$ achieves the highest global Silhouette Score (**0.2307**). $K=3$ drops to **0.2012** (a local minimum!).
   - On the Silhouette plot (`axes[1]`), the red vertical dashed line is drawn at **$K=3$** with label `K=3 (Score=0.201)`, misrepresenting $K=3$ as if it were the peak of the Silhouette curve!
   - `final_notebook_text.md` (lines 153–156) states:
     > *"The Elbow Method shows a clear inflection point at K=3... The Silhouette Analysis confirms that K=2 achieves the best cohesion-to-separation ratio. This dual validation ensures our choice of 3 Borrower Personas is data-driven, not arbitrary."*
   - **Flaw**: Calling $K=2$ peak silhouette and $K=3$ elbow a "dual validation" for choosing $K=3$ is logically incoherent. If Silhouette Score peaks at $K=2$ and drops at $K=3$, Silhouette Analysis actually **disagrees** with $K=3$. Choosing $K=3$ requires an explicit business/domain justification (balancing mathematical separation at $K=2$ with domain granularity of 3 risk tiers at $K=3$), rather than falsely claiming "dual validation".
2. ❌ **FEATURE SET MISMATCH BETWEEN $K$-SELECTION AND FINAL CLUSTERING**:
   - In Cell 9 ($K$-Selection, line 447):  
     `unsup_features_silhouette = ['loan_amnt', 'int_rate', 'annual_inc', 'dti', 'revol_util', 'revol_bal']`
   - In Cell 10 (Final $K$-Means, line 511):  
     `unsup_features = ['loan_amnt', 'int_rate', 'total_annual_inc', 'total_dti', 'revol_util', 'revol_bal']`
   - **Flaw**: Cell 9 uses raw single-applicant `annual_inc` and `dti`, while Cell 10 uses engineered joint/total features `total_annual_inc` and `total_dti`. The feature space evaluated during Silhouette/Elbow analysis is literally different from the feature space clustered in Cell 10!
3. ⚠️ **Absence of Individual Cluster Silhouette Coefficient Plots**:
   - The analysis only plots average silhouette scores across $K$. It does not generate individual cluster silhouette coefficient profiles (`silhouette_samples`), which would verify whether any cluster in $K=3$ suffers from negative silhouette values or severe cluster size imbalance.

---

#### E. Actionable Line-by-Line Recommendations for 100/100

1. **Fix Feature Mismatch in `Lending_Club_Project_Final.py` (Cell 9 line 447)**:
   - Align feature list in Cell 9 with Cell 10:
   ```python
   # Line 447 replacement:
   unsup_features_silhouette = ['loan_amnt', 'int_rate', 'total_annual_inc', 'total_dti', 'revol_util', 'revol_bal']
   ```

2. **Fix Plot Annotations & Selection Logic in Cell 9 (lines 475–481)**:
   - Draw vertical line at $K=2$ (Silhouette Peak) on Silhouette plot, and explain trade-off:
   ```python
   # Lines 475-481 update:
   axes[1].plot(list(K_range), silhouette_scores, 'gs-', linewidth=2, markersize=8)
   axes[1].set_xlabel('Number of Clusters (K)', fontsize=12)
   axes[1].set_ylabel('Silhouette Score', fontsize=12)
   axes[1].set_title('Silhouette Analysis', fontsize=14, fontweight='bold')
   axes[1].set_xticks(list(K_range))
   axes[1].axvline(x=2, color='green', linestyle=':', alpha=0.7, label='Peak Silhouette (K=2, Score=0.231)')
   axes[1].axvline(x=3, color='red', linestyle='--', alpha=0.7, label='Selected Persona K=3 (Score=0.201)')
   axes[1].legend()
   ```

3. **Rewrite Markdown Justification in `final_notebook_text.md` (lines 153–158)**:
   ```markdown
   **Mathematical Justification & Trade-Off Analysis for Cluster Selection ($K=3$):**
   * **Elbow Method**: Shows a distinct inflection point ("elbow") at $K=3$ (Inertia = 221,915), after which the marginal decrease in within-cluster sum of squares levels off.
   * **Silhouette Analysis**: Statistically peaks at $K=2$ (Silhouette Score = 0.2307), representing the coarsest binary split (High vs. Low Risk). However, $K=3$ maintains a strong score of 0.2012 while providing granular business utility.
   * **Trade-Off Rationale**: While $K=2$ is mathematically optimal for pure separation, $K=3$ is selected because it captures a critical intermediate risk profile ("Moderate/Stable Borrowers") between "High-Rollers" and "Highly Leveraged Borrowers", creating 3 actionable risk personas for credit underwriting.
   ```

4. **Add Individual Cluster Silhouette Profile Plot in Cell 9**:
   - Include standard Scikit-Learn cluster silhouette profile plot for $K=3$ to demonstrate no negative silhouette coefficients exist.

---

### Topic 3: Cluster Profiling & Business Context

#### A. Scope & Criteria
- **`ds_review_guide.md` Section 3**: *"פרופיילינג (Cluster Profiling): חפש פרשנות עסקית לאשכולות. האם הסטודנט הצליח לאפיין קבוצות של לווים מתוך הנתונים (למשל, 'לווים צעירים בסיכון גבוה')?"* (Look for business interpretation of clusters. Did student characterize borrower personas?)
- **`ORIGINAL_REQUEST.md` R4**: Business profiling for each cluster (characterizing borrower risk personas, mean financial features per cluster, business interpretation in Markdown).

#### B. Exact Code & File Locations
- **`Lending_Club_Project_Final.py`**:
  - Lines 548–552 (Cell 10): Cluster profiling code (`df_unsup_pd.groupby('Cluster')[unsup_features].median()`, target rate calculation `df_unsup_pd.groupby('Cluster')['target'].mean() * 100`, `display(cluster_profiles)`).
  - Lines 554–559: Markdown text on Cluster Analysis (Borrower Personas).
- **`final_notebook_text.md`**:
  - Lines 162–167: Cluster Analysis (Borrower Personas) text.
- **`Lending_Club_Project_Final.html`**:
  - Lines 9838–9867: HTML table output containing exact cluster median values and target default rates.

---

#### C. Pass / Success Items
1. **Persona Segmentation & Target Rate Calculation (Pass)**:
   - Cell 10 calculates cluster medians across all continuous financial features and calculates the default rate per cluster (`Target_Rate_%`).
2. **HTML Table Output Content (Pass)** (`Lending_Club_Project_Final.html` lines 9838–9867):
   - **Cluster 0 (Moderate / Stable Borrowers)**:
     - Median Loan Amount: **$10,000**
     - Interest Rate: **10.42%**
     - Annual Income: **$60,000**
     - DTI: **14.90**
     - Revolving Utilization: **32.6%**
     - Revolving Balance: **$7,261.50**
     - **Default Rate**: **13.81%** (Lowest Risk)
   - **Cluster 1 (Highly Leveraged / High Risk)**:
     - Median Loan Amount: **$11,550**
     - Interest Rate: **15.22%**
     - Annual Income: **$55,000**
     - DTI: **20.73**
     - Revolving Utilization: **68.9%** (Very High!)
     - Revolving Balance: **$11,648.00**
     - **Default Rate**: **26.33%** (Highest Risk — nearly DOUBLE Cluster 0!)
   - **Cluster 2 (Wealthy High-Rollers)**:
     - Median Loan Amount: **$28,000** (Large loans)
     - Interest Rate: **13.18%**
     - Annual Income: **$108,000** (High income)
     - DTI: **17.50**
     - Revolving Utilization: **60.3%**
     - Revolving Balance: **$26,659.00**
     - **Default Rate**: **20.09%** (Moderate-High Risk)
3. **Qualitative Persona Descriptions (Pass)**:
   - `final_notebook_text.md` (lines 162–167) identifies these 3 distinct borrower profiles.

---

#### D. Flaws & Gaps
1. ❌ **FORENSIC OMISSION — Complete Absence of Exact Numerical Metrics in Markdown Text**:
   - `final_notebook_text.md` (lines 162–167) contains only qualitative text:
     > *"Cluster 0 (Moderate/Stable): Moderate income, lower loan amounts. Cluster 1 (Highly Leveraged/High Risk): High dti, high revol_util, higher interest rates. Notice their default rate is typically the highest..."*
   - **Flaw**: None of the exact printed HTML table numbers (**13.81%**, **26.33%**, **20.09%**, **$108,000**, **68.9%**) appear in the Markdown summary text! This violates the Forensic Consistency rule (R5 & QA Guide Section 5), which mandates that textual conclusions must explicitly cite exact printed numerical metrics.
2. ❌ **ISOLATED EDA EXERCISE — Clusters Not Integrated Into Supervised Modeling**:
   - In Cell 10, cluster assignments are created on a temporary 100k pandas dataframe (`df_unsup_pd['Cluster']`).
   - These cluster labels are **never saved back into the main Polars dataset (`df`)** or exported to feature parquets (`train_fe.parquet`, `test_fe.parquet`).
   - Consequently, Phase 4 predictive models (LightGBM and MLP) do not receive `cluster_id` or OHE cluster personas as engineered features. This breaks the end-to-end integration between Phase 3 (Unsupervised) and Phase 4 (Supervised).
3. ⚠️ **Lack of Feature Distribution Visualizations Across Clusters**:
   - No radar charts, boxplots, or bar charts were created to visualize feature distributions across the 3 clusters.

---

#### E. Actionable Line-by-Line Recommendations for 100/100

1. **Rewrite Markdown Text in `final_notebook_text.md` (lines 162–167) with Forensic Numbers**:
   ```markdown
   **Quantitative Borrower Persona Profiles & Business Interpretation:**

   1. **Cluster 0: Moderate / Stable Borrowers (Low Risk — 13.81% Default Rate)**
      * **Financial Profile**: Median Income **$60,000**, Loan Amount **$10,000**, Interest Rate **10.42%**, DTI **14.90**, Revolving Utilization **32.6%**.
      * **Business Takeaway**: Represents low-risk, conservative borrowers with low credit utilization and moderate debt burden. Underwriting recommendation: Fast-track approval with standard prime rates.

   2. **Cluster 1: Highly Leveraged Borrowers (High Risk — 26.33% Default Rate)**
      * **Financial Profile**: Median Income **$55,000**, Loan Amount **$11,550**, Interest Rate **15.22%**, DTI **20.73**, Revolving Utilization **68.9%** (Extreme credit strain).
      * **Business Takeaway**: Default rate is nearly **DOUBLE** that of Cluster 0 (26.33% vs 13.81%). Driven by high debt-to-income and heavy credit card utilization. Underwriting recommendation: Require collateral, stricter DTI caps, or higher risk premiums.

   3. **Cluster 2: Wealthy High-Rollers (Moderate-High Risk — 20.09% Default Rate)**
      * **Financial Profile**: Median Income **$108,000**, Loan Amount **$28,000**, Interest Rate **13.18%**, DTI **17.50**, Revolving Utilization **60.3%**.
      * **Business Takeaway**: High-earning borrowers taking out massive loans ($28k median). Despite high income, elevated revolving utilization (60.3%) drives a 20.09% default rate. Underwriting recommendation: Cap total exposure per high-earning applicant.
   ```

2. **Integrate Cluster Labels into Main Dataset for Phase 4 (Cell 10 in `.py`)**:
   - Train $K$-Means model on scaled feature set and assign `cluster_id` back to main dataset:
   ```python
   # --- Save Cluster Feature back to Dataset for Supervised Modeling ---
   print("Assigning cluster personas to full dataset for Phase 4 feature engineering...")
   # Fit KMeans on scaled 100k sample, then predict on full dataset scaler transformation
   df_all_scaled = scaler.transform(df.select(unsup_features).to_pandas().fillna(df_unsup_pd[unsup_features].median()))
   df = df.with_columns(pl.Series('cluster_persona_id', kmeans.predict(df_all_scaled)))
   ```

3. **Add Feature Comparison Boxplots / Radar Chart**:
   - Add a 2x3 grid of Seaborn boxplots showing feature distribution (`loan_amnt`, `int_rate`, `total_dti`, `revol_util`, `total_annual_inc`, `revol_bal`) across Clusters 0, 1, and 2, saved to `eda_plots/cluster_profiles_distribution.png`.

---

## Actionable Checklist for a 100/100 Score

To elevate Phase 3 from **82/100 to 100/100**, execute the following concrete modifications:

- [ ] **Calculate & Plot PCA Explained Variance Ratio**:
  - Add `pca.explained_variance_ratio_` calculation to Cell 10 / 12.
  - Generate Scree plot saved to `eda_plots/pca_explained_variance.png`.
  - Report exact PC1 (XX.X%), PC2 (YY.Y%), and Cumulative (ZZ.Z%) variance ratios in `final_notebook_text.md`.
- [ ] **Resolve $K$-Selection Logical Contradiction & Align Features**:
  - Change Cell 9 feature list from `annual_inc`/`dti` to `total_annual_inc`/`total_dti`.
  - Update `eda_plots/silhouette_elbow.png` annotation to highlight $K=2$ (Peak Silhouette = 0.2307) and $K=3$ (Selected Persona Trade-Off = 0.2012).
  - Rewrite Markdown text to explicitly explain the trade-off between mathematical peak at $K=2$ and domain granularity at $K=3$.
- [ ] **Incorporate Forensic Numerical Metrics into Cluster Profiling Text**:
  - Transcribe exact HTML table numbers (Cluster 0: 13.81% default, Cluster 1: 26.33% default, Cluster 2: 20.09% default) into `final_notebook_text.md`.
- [ ] **Connect Unsupervised Clusters to Supervised Phase 4**:
  - Save `cluster_persona_id` to `df` and include as a categorical feature in `train_fe.parquet` / `test_fe.parquet` for LightGBM and MLP modeling.
