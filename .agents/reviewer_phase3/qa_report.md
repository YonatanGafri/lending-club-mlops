# Phase 3 QA Verification & Forensic Audit Report: Unsupervised Learning & Clustering

**Project:** Lending Club Data Science Project (`c:\Users\yonat\.gemini\antigravity\scratch\lending_club_ds`)  
**QA Reviewer:** QA Reviewer Agent 3 (`reviewer_phase3`) — Reviewer & Adversarial Critic  
**Investigator:** Investigator Agent 3 (`explorer_phase3`)  
**Audit Target:** Phase 3: Unsupervised Learning & Clustering  
**Date:** July 27, 2026  
**Reference Standards:** `ds_review_guide.md` Section 3, `ORIGINAL_REQUEST.md` Requirement R4, & Forensic Evaluation Rule R5  

---

## Executive Summary & QA Verdict

**Overall QA Verdict:** ⚠️ **REQUEST_CHANGES (Grade: 82 / 100)**

While Phase 3 demonstrates high visual presentation quality (such as the 2x2 PCA vs t-SNE comparison plot and factor loadings heatmap), an independent, objective QA audit reveals **critical theoretical omissions, logical contradictions, feature set inconsistencies, and forensic omissions**.

### Sub-Category Score Breakdown:

| Sub-Category | Status | Score | Primary Reason |
|---|:---:|:---:|---|
| **1. Smart Dimensionality Reduction (PCA & t-SNE)** | ⚠️ PARTIAL FLAW | **75 / 100** | Side-by-side PCA vs t-SNE plot (`eda_plots/pca_vs_tsne.png`) is excellent, but **`pca.explained_variance_ratio_` is 100% missing** (never calculated, printed, or plotted). |
| **2. Mathematical Justification for $K$-Means ($K$ Selection)** | ❌ CRITICAL FLAW | **80 / 100** | Logical contradiction: Output prints `Best K by Silhouette: K=2` (score 0.2307), yet text claims "dual validation" for $K=3$ (score 0.2012). Feature mismatch between Cell 9 (`annual_inc`/`dti`) and Cell 10 (`total_annual_inc`/`total_dti`). |
| **3. Cluster Profiling & Business Context** | ⚠️ PARTIAL FLAW | **90 / 100** | HTML table displays rich cluster metrics (e.g. 13.81% vs 26.33% default rates), but **Markdown text omits all exact numerical figures**. Cluster labels are isolated and not saved for Phase 4 models. |
| **Overall Phase 3 Score** | | **82 / 100** | **Requires concrete code & text updates to reach 100/100 perfection.** |

---

## Detailed Audit Findings & Forensic Verifications

---

### Finding 1: PCA Explained Variance Ratio is Completely Omitted [CRITICAL FLAW]

- **Location:** `Lending_Club_Project_Final.py` lines 523–526 & 661–665; `final_notebook_text.md` lines 171–209.
- **What was found:** 
  - PCA dimensionality reduction is fitted in Cell 10 (`pca = PCA(n_components=2, random_state=42)`), but `pca.explained_variance_ratio_` and `np.cumsum(...)` are **never called, printed, or plotted anywhere in the repository**.
  - Grep search for `explained_variance` across the codebase yielded **0 results**.
- **Why this is a problem:**
  - Dimensionality reduction without calculating explained variance fails to prove whether 2 principal components retain sufficient financial variance (e.g. 70%+) or discard vital information.
  - Direct violation of `ds_review_guide.md` Section 3: *"Is there an explanation of the amount of explained variance ratio retained after PCA?"*
- **Verified Claim:**
  - `eda_plots/pca_vs_tsne.png` exists (1,603,289 bytes) and correctly displays 2x2 subplots $\rightarrow$ **PASS (Visualization)**.
  - PCA Explained Variance Ratio calculation $\rightarrow$ **FAIL (Omitted)**.

---

### Finding 2: $K$-Selection Logical Contradiction & Feature Set Mismatch [CRITICAL FLAW]

- **Location:** `Lending_Club_Project_Final.py` lines 438–489 (Cell 9) & lines 502–532 (Cell 10); `final_notebook_text.md` lines 153–158; `Lending_Club_Project_Final.html` lines 9642–9671.
- **What was found:**
  1. **Logical Contradiction:** Code output prints `Best K by Silhouette: K=2` (score **0.2307**). $K=3$ drops to **0.2012**. However, `final_notebook_text.md` (line 156) claims: *"The Silhouette Analysis confirms that K=2 achieves the best cohesion-to-separation ratio. This dual validation ensures our choice of 3 Borrower Personas is data-driven, not arbitrary."*
  2. **Feature Set Mismatch:** Cell 9 line 447 defines:
     `unsup_features_silhouette = ['loan_amnt', 'int_rate', 'annual_inc', 'dti', 'revol_util', 'revol_bal']`
     Cell 10 line 511 defines:
     `unsup_features = ['loan_amnt', 'int_rate', 'total_annual_inc', 'total_dti', 'revol_util', 'revol_bal']`
- **Why this is a problem:**
  - Claiming "dual validation" for $K=3$ when Silhouette Score peaks at $K=2$ and drops at $K=3$ is mathematically invalid. Selecting $K=3$ requires framing it as a trade-off between mathematical peak separation ($K=2$) and business domain granularity ($K=3$ risk personas).
  - Cell 9 evaluates a different feature space (`annual_inc`/`dti`) than Cell 10 (`total_annual_inc`/`total_dti`), invalidating the $K$-selection process for Cell 10's clustering.
- **Verified Claim:**
  - `eda_plots/silhouette_elbow.png` generated $\rightarrow$ **PASS (Plot Artifact)**.
  - Mathematical coherence of $K$-selection $\rightarrow$ **FAIL (Contradiction & Feature Mismatch)**.

---

### Finding 3: Cluster Profiling Forensic Metric Omission & Model Isolation [MAJOR FLAW]

- **Location:** `Lending_Club_Project_Final.py` lines 548–559; `final_notebook_text.md` lines 162–167; `Lending_Club_Project_Final.html` lines 9838–9867.
- **What was found:**
  1. **Forensic Omission:** Printed HTML table shows exact numbers:
     - **Cluster 0 (Moderate/Stable):** Loan $10,000 | Interest Rate 10.42% | Income $60,000 | DTI 14.90 | Revol Util 32.6% | **Default Rate: 13.81%**
     - **Cluster 1 (Highly Leveraged):** Loan $11,550 | Interest Rate 15.22% | Income $55,000 | DTI 20.73 | Revol Util 68.9% | **Default Rate: 26.33%** (Highest Risk)
     - **Cluster 2 (High-Rollers):** Loan $28,000 | Interest Rate 13.18% | Income $108,000 | DTI 17.50 | Revol Util 60.3% | **Default Rate: 20.09%**
     However, `final_notebook_text.md` contains only qualitative bullets ("Moderate income", "highest default rate"), omitting all printed numbers.
  2. **Downstream Isolation:** Cluster labels (`df_unsup_pd['Cluster']`) are assigned to a temporary 100k pandas dataframe in Cell 10 and are **never saved back to `df` or exported to Phase 4 datasets (`train_fe.parquet`/`test_fe.parquet`)**.
- **Why this is a problem:**
  - Free text summaries must cite exact printed code metrics to meet Forensic Evaluation standards (Rule R5).
  - Unsupervised clustering is isolated from supervised modeling, missing an opportunity to enrich Phase 4 LightGBM and MLP models with `cluster_persona_id` features.
- **Verified Claim:**
  - Identification of 3 risk personas in HTML output $\rightarrow$ **PASS (Data Generation)**.
  - Forensic text consistency & model integration $\rightarrow$ **FAIL (Omitted Numbers & Isolated Pipeline)**.

---

## Verified Claims Matrix

| Claim in Audit Report / Codebase | Verification Method | Status | QA Notes |
|---|---|:---: |---|
| `pca.explained_variance_ratio_` missing | Grep search & code inspection of lines 523–670 | ✅ **CONFIRMED** | Zero occurrences in codebase or markdown text. |
| `eda_plots/pca_vs_tsne.png` exists | File system inspection | ✅ **CONFIRMED** | 1.6 MB PNG file present in `eda_plots/`. |
| Best K by Silhouette is K=2 (score 0.2307) | Inspection of HTML line 9642 | ✅ **CONFIRMED** | Output explicitly prints $K=2$ score 0.2307 vs $K=3$ score 0.2012. |
| Feature mismatch between Cell 9 & Cell 10 | Line 447 vs Line 511 comparison | ✅ **CONFIRMED** | `annual_inc`/`dti` in Cell 9 vs `total_annual_inc`/`total_dti` in Cell 10. |
| Printed HTML table contains exact persona metrics | Inspection of HTML lines 9838–9867 | ✅ **CONFIRMED** | Medians and default rates (13.81%, 26.33%, 20.09%) are printed in HTML. |
| `final_notebook_text.md` omits table numbers | Inspection of Markdown lines 162–167 | ✅ **CONFIRMED** | Text contains only qualitative descriptions. |
| Cluster labels isolated from Phase 4 | Code inspection of Cell 10 & Phase 4 | ✅ **CONFIRMED** | `df_unsup_pd['Cluster']` is never joined to main `df` or feature parquets. |

---

## Actionable Recommendations for a 100/100 Score

To elevate Phase 3 from **82/100 to 100/100**, implement the following concrete changes:

### 1. PCA Explained Variance Ratio (Cell 10 / Cell 12 & `final_notebook_text.md`)
- Add after line 526 in `Lending_Club_Project_Final.py`:
  ```python
  # --- Calculate & Report PCA Explained Variance ---
  var_exp = pca.explained_variance_ratio_
  cum_var_exp = np.cumsum(var_exp)

  print(f"PC1 Explained Variance: {var_exp[0]*100:.2f}%")
  print(f"PC2 Explained Variance: {var_exp[1]*100:.2f}%")
  print(f"Total Cumulative Variance (PC1 + PC2): {cum_var_exp[1]*100:.2f}%")

  # Generate Scree Plot
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
- Insert paragraph in `final_notebook_text.md` under Section 4.6.1 stating exact PC1, PC2, and cumulative variance values.

### 2. $K$-Selection Alignment & Trade-Off Justification (Cell 9 & `final_notebook_text.md`)
- Fix line 447 in `Lending_Club_Project_Final.py` to match Cell 10:
  ```python
  unsup_features_silhouette = ['loan_amnt', 'int_rate', 'total_annual_inc', 'total_dti', 'revol_util', 'revol_bal']
  ```
- Update plot annotations in Cell 9 (lines 475–481) to highlight $K=2$ as Peak Silhouette (0.231) and $K=3$ as Selected Persona Trade-Off (0.201).
- Rewrite `final_notebook_text.md` lines 153–158:
  ```markdown
  **Mathematical Justification & Trade-Off Analysis for Cluster Selection ($K=3$):**
  * **Elbow Method**: Inflection point at $K=3$ (Inertia = 221,915), after which marginal inertia reduction levels off.
  * **Silhouette Analysis**: Statistically peaks at $K=2$ (Silhouette Score = 0.2307), representing binary risk division. $K=3$ maintains a strong score of 0.2012.
  * **Trade-Off Rationale**: While $K=2$ is optimal for pure mathematical separation, $K=3$ is selected for domain utility—capturing an intermediate "Moderate/Stable" risk tier between low-risk and high-leveraged borrowers.
  ```

### 3. Quantitative Cluster Profiling & Model Integration (Cell 10 & `final_notebook_text.md`)
- Transcribe exact HTML table numbers into `final_notebook_text.md` lines 162–167:
  - **Cluster 0 (Moderate/Stable):** Median Income **$60,000**, Loan **$10,000**, Interest Rate **10.42%**, DTI **14.90**, Revol Util **32.6%**, **Default Rate: 13.81%**.
  - **Cluster 1 (Highly Leveraged):** Median Income **$55,000**, Loan **$11,550**, Interest Rate **15.22%**, DTI **20.73**, Revol Util **68.9%**, **Default Rate: 26.33%** (Nearly DOUBLE Cluster 0!).
  - **Cluster 2 (High-Rollers):** Median Income **$108,000**, Loan **$28,000**, Interest Rate **13.18%**, DTI **17.50**, Revol Util **60.3%**, **Default Rate: 20.09%**.
- Integrate cluster labels into main Polars dataset in Cell 10:
  ```python
  # Assign cluster persona feature for Phase 4 supervised modeling
  df_all_scaled = scaler.transform(df.select(unsup_features).to_pandas().fillna(df_unsup_pd[unsup_features].median()))
  df = df.with_columns(pl.Series('cluster_persona_id', kmeans.predict(df_all_scaled)))
  ```

---

## Verification Method for Subsequent Audits

1. Run `Lending_Club_Project_Final.py` and verify `eda_plots/pca_explained_variance.png` is generated.
2. Confirm printed output prints `PC1 Explained Variance` and `PC2 Explained Variance`.
3. Check `final_notebook_text.md` to ensure exact numbers (13.81%, 26.33%, 20.09%, $108,000, 68.9%) are quoted verbatim.
4. Verify `cluster_persona_id` is present in `df.columns` and saved in feature engineered parquets.
