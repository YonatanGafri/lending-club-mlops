# QA Dialogue & Technical Verification Transcript: Phase 3 (Unsupervised Learning & Clustering)

**Participants:**
- **Investigator Agent 3 (`explorer_phase3`)** — Original Audit Findings Author
- **QA Reviewer Agent 3 (`reviewer_phase3`)** — QA Reviewer & Adversarial Critic

**Date & Time:** July 27, 2026  
**Context:** Phase 3 Cross-Verification & Stress-Testing of Unsupervised Learning & Clustering Artifacts (`Lending_Club_Project_Final.py`, `final_notebook_text.md`, `Lending_Club_Project_Final.html`, `eda_plots/`).

---

## Session 1: Opening & Verification Scope

**`reviewer_phase3`**: Welcome, `explorer_phase3`. As QA Reviewer and Adversarial Critic for Phase 3, my objective is to independently cross-verify all claims, code references, line numbers, variable names, and mathematical logic in your Phase 3 audit report. We will evaluate Phase 3 against `ds_review_guide.md` Section 3 and `ORIGINAL_REQUEST.md` Requirement R4. 

Let me first verify the core files under review:
1. `Lending_Club_Project_Final.py` (Lines 438–730: Cells 9 through 14)
2. `final_notebook_text.md` (Lines 143–225)
3. `Lending_Club_Project_Final.html` (Lines 9642–9670 and 9838–9867)
4. Plot Artifacts: `eda_plots/pca_vs_tsne.png` and `eda_plots/silhouette_elbow.png`

Let's begin with Topic 1: Smart Dimensionality Reduction & PCA.

---

## Session 2: Challenge 1 — Dimensionality Reduction (PCA) & Explained Variance

**`explorer_phase3`**: In my audit of Topic 1, I evaluated PCA and t-SNE. The codebase implements a 2-component PCA in Cell 10 (lines 523–526) and generates a 2x2 side-by-side plot comparing PCA and t-SNE in Cell 11 (lines 600–631, saved as `eda_plots/pca_vs_tsne.png`). Cell 12 extracts factor loadings into a heatmap (`loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=features)`). 

However, I identified a critical flaw: `pca.explained_variance_ratio_` and cumulative variance calculations are completely missing. Searching the entire repository for `explained_variance` returned 0 results. As a result, I rated PCA at 75/100.

**`reviewer_phase3`**: Let me independently verify these line references and code structures.
1. Checking `Lending_Club_Project_Final.py`:
   - Line 523: `pca = PCA(n_components=2, random_state=42)`
   - Line 524: `X_pca = pca.fit_transform(X_scaled)`
   - Line 661–665: `loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=features)`
   - Lines 600–631: PCA vs t-SNE 2x2 plot saved to `eda_plots/pca_vs_tsne.png`.
2. I confirmed the file `eda_plots/pca_vs_tsne.png` exists in the filesystem (size: 1,603,289 bytes).
3. I ran an independent `grep_search` across all project files for `explained_variance`: **0 matches found**.

*Adversarial Challenge by `reviewer_phase3`*:
Why is the missing explained variance ratio so critical mathematically? If PC1 and PC2 only capture, say, 35% of total variance, reducing 6 financial dimensions to 2D discards 65% of the information! In credit risk, claiming that a 2D plot "validates" borrower risk spectrum without reporting retained variance is a major theoretical blind spot. The student assumes 2 components are sufficient without proof.

**`explorer_phase3`**: Exactly. `ds_review_guide.md` Section 3 explicitly asks: *"Is there an explanation of the amount of explained variance ratio retained after PCA?"* Without `pca.explained_variance_ratio_`, the student fails this mandatory Kaggle-level check.

**`reviewer_phase3`**: I also verified line 586 in Cell 11:
- `n_sample = min(15000, len(df_unsup_pd))`
- Line 597: `pca_compare = PCA(n_components=2)`
- Line 598: `X_pca_sample = pca_compare.fit_transform(X_sample)`
Notice that instead of transforming the already-fitted `pca` model from Cell 10, Cell 11 instantiates a new `PCA` object on the 15k sample. While the resulting components are visually similar, re-fitting PCA on a subsample introduces minor structural divergence.

**Consensus on Topic 1**: 
- **Score**: **75 / 100**
- **Verified Flaw**: Total omission of `pca.explained_variance_ratio_`.
- **Remediation**: Add variance ratio calculations, plot a Scree plot (`eda_plots/pca_explained_variance.png`), and report PC1, PC2, and cumulative variance percentages in `final_notebook_text.md`.

---

## Session 3: Challenge 2 — K-Means Mathematical Justification ($K$ Selection)

**`explorer_phase3`**: For Topic 2, I examined Cell 9 (`Lending_Club_Project_Final.py` lines 438–489) and `final_notebook_text.md` (lines 143–158). Cell 9 loops over $K \in [2, 6]$, calculates Inertia and Silhouette scores (using `sample_size=20000`), and generates `eda_plots/silhouette_elbow.png`.

I found two major flaws:
1. **Logical Contradiction**: Printed code output in `Lending_Club_Project_Final.html` line 9671 states: `Best K by Silhouette: K=2` (Score: **0.2307**). At $K=3$, the Silhouette score drops to **0.2012**. Yet, `final_notebook_text.md` line 156 claims: *"This dual validation ensures our choice of 3 Borrower Personas is data-driven, not arbitrary."*
2. **Feature Set Mismatch**: In Cell 9 line 447, `unsup_features_silhouette = ['loan_amnt', 'int_rate', 'annual_inc', 'dti', 'revol_util', 'revol_bal']`. But in Cell 10 line 511, `unsup_features = ['loan_amnt', 'int_rate', 'total_annual_inc', 'total_dti', 'revol_util', 'revol_bal']`. Cell 9 evaluates raw `annual_inc`/`dti`, whereas Cell 10 clusters on engineered `total_annual_inc`/`total_dti`.

**`reviewer_phase3`**: Let me stress-test both findings against the raw execution logs.
1. Inspection of `Lending_Club_Project_Final.html` lines 9642–9670 confirms the exact printed outputs:
   - `K=2: Silhouette Score = 0.2307, Inertia = 251,896`
   - `K=3: Silhouette Score = 0.2012, Inertia = 221,915`
   - `K=4: Silhouette Score = 0.2011, Inertia = 198,342`
   - `K=5: Silhouette Score = 0.2012, Inertia = 175,390`
   - `K=6: Silhouette Score = 0.2074, Inertia = 157,910`
   - Line 9671: `✅ Best K by Silhouette: K=2`
2. Inspection of `Lending_Club_Project_Final.py` lines 475–481 shows:
   - Line 480: `axes[1].axvline(x=3, color='red', linestyle='--', alpha=0.7, label=f'K=3 (Score={silhouette_scores[1]:.3f})')`
   The code forces a red dashed line at $K=3$ on the Silhouette plot and labels it as if $K=3$ were the chosen optimal point, ignoring that $K=2$ is the global peak!

*Adversarial Challenge by `reviewer_phase3`*:
Let's analyze the feature mismatch: `annual_inc` vs `total_annual_inc`. In joint applications, `total_annual_inc` combines primary and co-applicant income, significantly increasing variance and scale for joint borrowers. By running Silhouette analysis on single-applicant `annual_inc` in Cell 9 and then clustering on `total_annual_inc` in Cell 10, the mathematical topology of the feature space changes! This is a serious methodological oversight—$K$-selection must be performed on the exact feature matrix passed to the clustering algorithm.

Furthermore, claiming "dual validation" when Silhouette Analysis shows $K=2$ as optimal and $K=3$ as a lower local score is logically contradictory. In a peer-reviewed publication or production QA audit, this would be flagged as misrepresenting metric results. The student should have framed $K=3$ as a **business domain trade-off**: $K=2$ provides the highest mathematical separation (binary low vs high risk), but $K=3$ provides necessary business granularity (low, moderate, high risk personas).

**Consensus on Topic 2**:
- **Score**: **80 / 100**
- **Verified Flaw 1**: Logical contradiction claiming "dual validation" for $K=3$ when Silhouette Score peaks at $K=2$ (0.2307 vs 0.2012).
- **Verified Flaw 2**: Feature mismatch between Cell 9 (`annual_inc`/`dti`) and Cell 10 (`total_annual_inc`/`total_dti`).
- **Remediation**: Align Cell 9 features with Cell 10, update plot annotations, and rewrite Markdown text to explain the mathematical vs. business trade-off between $K=2$ and $K=3$.

---

## Session 4: Challenge 3 — Cluster Profiling, Forensic QA, & Downstream Model Integration

**`explorer_phase3`**: For Topic 3, I reviewed Cell 10 (`Lending_Club_Project_Final.py` lines 548–559) and `final_notebook_text.md` (lines 162–167). Cell 10 groups by cluster and displays median feature values and `Target_Rate_%`.

In `Lending_Club_Project_Final.html` lines 9838–9867, the printed table shows:
- **Cluster 0**: Loan $10,000 | Interest Rate 10.42% | Income $60,000 | DTI 14.90 | Revol Util 32.6% | **Default Rate: 13.81%**
- **Cluster 1**: Loan $11,550 | Interest Rate 15.22% | Income $55,000 | DTI 20.73 | Revol Util 68.9% | **Default Rate: 26.33%**
- **Cluster 2**: Loan $28,000 | Interest Rate 13.18% | Income $108,000 | DTI 17.50 | Revol Util 60.3% | **Default Rate: 20.09%**

My finding was that `final_notebook_text.md` omits every single printed number, providing only qualitative text ("Moderate income", "highest default rate"). Additionally, cluster labels (`df_unsup_pd['Cluster']`) are never saved back to the main Polars dataset `df` or passed to Phase 4 models. I rated Cluster Profiling at 90/100.

**`reviewer_phase3`**: I independently verified `final_notebook_text.md` lines 162–167:
- Line 163: `* Cluster 0 (Moderate/Stable): Moderate income, lower loan amounts.`
- Line 164: `* Cluster 1 (Highly Leveraged/High Risk): High dti, high revol_util, higher interest rates. Notice their default rate is typically the highest.`
- Line 165: `* Cluster 2 (Wealthy High-Rollers): High annual_inc, massive loan_amnt, but low DTI.`

*Adversarial Challenge by `reviewer_phase3`*:
The omission of exact numbers violates Forensic Evaluation Rule R5 and `ds_review_guide.md` Section 5: *"No rounding, no claiming numbers not actually printed by code."* Free text must explicitly quote printed numerical values. 

More importantly, let's address **downstream model integration**:
In machine learning pipelines, unsupervised clustering is most valuable when clusters are fed into downstream supervised models as engineered features (e.g. `cluster_persona_id`). In this codebase, Cell 10 creates `df_unsup_pd['Cluster']` on a temporary 100,000-row pandas dataframe, but never writes `Cluster` back to the main dataset `df`. Thus, Phase 4 supervised models (LightGBM and MLP) receive zero signal from the $K$-Means clustering exercise!

**Consensus on Topic 3**:
- **Score**: **90 / 100**
- **Verified Flaw 1**: Forensic omission of exact printed table numbers in Markdown text.
- **Verified Flaw 2**: Cluster assignments are isolated in Cell 10 and not integrated into Phase 4 datasets (`train_fe.parquet` / `test_fe.parquet`).
- **Remediation**: Transcribe exact median values and default rates into `final_notebook_text.md`, assign `cluster_persona_id` back to main `df`, and export as a feature for Phase 4.

---

## Session 5: Supplementary Audit — Isolation Forest & Correlation Analysis

**`reviewer_phase3`**: Let's review Cell 13 (Isolation Forest, lines 687–706) and Cell 14 (Correlation Heatmap, lines 715–728).
1. **Cell 13 (Isolation Forest)**:
   - `iso_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)`
   - Flags top 1% unusual borrowers.
   - Finding: Anomalies (-1) have a lower default rate than normal inliers (1), proving anomalies represent ultra-wealthy/conservative applicants rather than credit risks.
2. **Cell 14 (Correlation Heatmap)**:
   - Heatmap of core financial features shows minimal extreme multicollinearity (except `loan_amnt` vs `installment`).

Both supplementary sections are implemented correctly and add valuable domain insight.

---

## Session 6: Final Consensus & Grading Summary

**`reviewer_phase3`**: Based on our thorough cross-verification, we have reached full consensus on the Phase 3 evaluation:

### Final Category Scores:
1. **Smart Dimensionality Reduction (PCA & t-SNE)**: **75 / 100**
2. **Mathematical Justification for $K$-Means ($K$ Selection)**: **80 / 100**
3. **Cluster Profiling & Business Context**: **90 / 100**

### Overall Phase 3 Score: **82 / 100**

**`explorer_phase3`**: The findings are fully validated with 100% line-by-line evidence and zero discrepancies. We are ready to publish the final QA Report and Handoff.

**`reviewer_phase3`**: Excellent work. Dialogue session closed.
