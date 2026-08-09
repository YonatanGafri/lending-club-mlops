# 5-Component Handoff Report: Phase 3 Unsupervised Learning & Clustering

**Agent:** Investigator Agent 3 (Phase 3: Unsupervised Learning & Clustering)  
**Target Path:** `c:\Users\yonat\.gemini\antigravity\scratch\lending_club_ds\.agents\explorer_phase3\report.md`  
**Date:** July 27, 2026  

---

## 1. Observation

1. **PCA Implementation & Explained Variance**:
   - `Lending_Club_Project_Final.py` lines 523–524: `pca = PCA(n_components=2, random_state=42); X_pca = pca.fit_transform(X_scaled)`.
   - `Lending_Club_Project_Final.py` lines 661–665: `loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=features)`.
   - `grep_search` across `lending_club_ds` for `explained_variance` yielded **0 results**.
   - Neither `pca.explained_variance_ratio_` nor `np.cumsum(...)` is calculated, printed, or plotted in `.py` or reported in `final_notebook_text.md`.

2. **$K$-Means Selection ($K$ Silhouette & Elbow Analysis)**:
   - `Lending_Club_Project_Final.py` lines 438–485 (Cell 9) calculates inertia and silhouette scores for $K \in [2, 6]$, saving plot to `eda_plots/silhouette_elbow.png`.
   - `Lending_Club_Project_Final.html` lines 9642–9670 prints:
     - `K=2: Silhouette Score = 0.2307, Inertia = 251,896`
     - `K=3: Silhouette Score = 0.2012, Inertia = 221,915`
     - `K=4: Silhouette Score = 0.2011, Inertia = 198,342`
     - `K=5: Silhouette Score = 0.2012, Inertia = 175,390`
     - `K=6: Silhouette Score = 0.2074, Inertia = 157,910`
     - `Best K by Silhouette: K=2`
   - `final_notebook_text.md` lines 153–156 states: *"The Elbow Method shows a clear inflection point at K=3... The Silhouette Analysis confirms that K=2 achieves the best cohesion-to-separation ratio. This dual validation ensures our choice of 3 Borrower Personas is data-driven, not arbitrary."*
   - Feature mismatch: Cell 9 line 447 uses `['loan_amnt', 'int_rate', 'annual_inc', 'dti', 'revol_util', 'revol_bal']`, while Cell 10 line 511 uses `['loan_amnt', 'int_rate', 'total_annual_inc', 'total_dti', 'revol_util', 'revol_bal']`.

3. **Cluster Profiling & Business Context**:
   - `Lending_Club_Project_Final.py` lines 548–551 computes medians and `Target_Rate_%` per cluster.
   - `Lending_Club_Project_Final.html` lines 9838–9867 outputs HTML table:
     - Cluster 0: Loan $10,000, Int 10.42%, Inc $60,000, DTI 14.90, Revol Util 32.6%, Default Rate **13.81%**.
     - Cluster 1: Loan $11,550, Int 15.22%, Inc $55,000, DTI 20.73, Revol Util 68.9%, Default Rate **26.33%**.
     - Cluster 2: Loan $28,000, Int 13.18%, Inc $108,000, DTI 17.50, Revol Util 60.3%, Default Rate **20.09%**.
   - `final_notebook_text.md` lines 162–167 contains qualitative text only, omitting all exact numerical metrics (13.81%, 26.33%, 20.09%, $108k, 68.9%).
   - Cluster label `df_unsup_pd['Cluster']` is created on a temporary 100k pandas dataframe and not saved back to `df` or used as a feature in Phase 4.

---

## 2. Logic Chain

1. **PCA Deficiency**:
   - *Observation*: `pca.fit_transform(X_scaled)` is run in Cell 10 and Cell 11, but `pca.explained_variance_ratio_` is never called or printed.
   - *Reasoning*: Without explained variance ratios, the reader cannot verify how much information is retained in 2D space versus lost in dimensionality reduction. This violates `ds_review_guide.md` Section 3.
   - *Conclusion*: Score for PCA is capped at 75/100 despite excellent side-by-side t-SNE visualization and factor loadings heatmap.

2. **$K$-Selection Contradiction**:
   - *Observation*: Printed output shows $K=2$ has highest silhouette (0.2307) vs $K=3$ (0.2012). Plot draws red dashed line at $K=3$. Text claims "dual validation".
   - *Reasoning*: Silhouette score actually drops at $K=3$. Calling $K=2$ peak silhouette a "dual validation" for $K=3$ is logically false. Selecting $K=3$ requires framing it as a trade-off between mathematical peak at $K=2$ and domain granularity at $K=3$. Additionally, feature set in Cell 9 (`annual_inc`) differs from Cell 10 (`total_annual_inc`).
   - *Conclusion*: Score for $K$-Selection is capped at 80/100.

3. **Cluster Profiling Forensic Omission**:
   - *Observation*: Table output in HTML has exact numbers (13.81%, 26.33%, 20.09%), but Markdown text has only qualitative bullets ("Moderate income", "highest default rate").
   - *Reasoning*: Forensic evaluation standards (R5) require free text to explicitly state exact printed code metrics. Furthermore, clusters are left isolated without being fed into LightGBM/MLP models.
   - *Conclusion*: Score for Cluster Profiling is 90/100, bringing overall Phase 3 grade to 82/100.

---

## 3. Caveats

- **No Caveats**: All relevant files (`Lending_Club_Project_Final.py`, `final_notebook_text.md`, `Lending_Club_Project_Final.html`, `eda_plots/silhouette_elbow.png`, `eda_plots/pca_vs_tsne.png`) were inspected directly. Code logic and printed HTML table outputs were verified line by line.

---

## 4. Conclusion

Phase 3 achieves an overall grade of **82/100**. It demonstrates strong visual presentation (PCA vs t-SNE side-by-side plots and factor loading heatmaps), but suffers from 3 major addressable issues:
1. Complete omission of PCA explained variance ratio calculation/reporting.
2. A logical contradiction in $K$-selection justification ($K=2$ silhouette peak vs $K=3$ elbow) and feature set mismatch between Cell 9 and Cell 10.
3. Absence of printed numerical metrics in Markdown text and isolation of clusters from Phase 4 supervised modeling.

Following the concrete recommendations in `.agents/explorer_phase3/report.md` will elevate Phase 3 to a **100/100**.

---

## 5. Verification Method

To independently verify these findings:
1. Inspect `Lending_Club_Project_Final.py` lines 438–677 and confirm absence of `explained_variance_ratio_`.
2. Inspect `Lending_Club_Project_Final.html` lines 9642–9670 and 9838–9867 to verify exact printed Silhouette scores and Cluster median numbers.
3. Compare `final_notebook_text.md` lines 153–167 against HTML table outputs to confirm omission of numerical metrics in text.
4. Check `eda_plots/silhouette_elbow.png` and `eda_plots/pca_vs_tsne.png` in workspace.
