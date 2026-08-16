# Lending Club Credit Risk & Decision Intelligence Engine 🏦📊

## Executive Overview & Project Purpose
This project delivers a comprehensive, end-to-end Credit Risk Modeling and Decision Intelligence solution built on the historical Lending Club dataset (1.2M+ loans). The primary objective is to accurately predict borrower default risk at origination, optimize financial underwriting profitability under asymmetric risk economics, and establish an explainable, regulatory-compliant credit decision framework.

Rather than treating credit scoring merely as an academic classification problem, this research bridges machine learning with quantitative risk management—formulating custom loss functions, resolving circular data leakages, calibrating predictive default probabilities, and transforming statistical outputs into actionable, bottom-line financial gains.

---

## What the Research Addresses (Core Problems & Inquiries)
1. **The In-House vs. Platform Dilemma:** Can an independent, ground-up machine learning model outperform or enrich Lending Club's proprietary credit grades?
2. **Strict Pre-Origination Integrity:** Eliminating post-origination data leakage and right-censoring to simulate real-world underwriting conditions.
3. **Financial Feature Synthesis & Skewness:** Addressing heavy-tailed financial distributions (income, balances) and capturing non-linear interactions through custom solvency ratios and unsupervised clustering.
4. **Asymmetric Risk Optimization:** Balancing the trade-off between the upside of loan interest margins (e.g., +$2,800 on good loans) vs. the severe downside of loan charge-offs (e.g., -$10,500 on defaults).
5. **Model Explainability & Governance:** Generating compliant Adverse Action notice reason codes (FCRA/ECOA) to explain individual loan denials with Shapley additive explanations (TreeSHAP).

---

## End-to-End Workflow & Methodology

```
┌─────────────────────────┐     ┌───────────────────────────┐     ┌──────────────────────────┐
│  Data Audit & Cleaning  │ ──> │ Feature Engineering & EDA │ ──> │   Model Benchmarking     │
│  - 2-Round Leakage Drop │     │ - Solvency Ratios         │     │   - 5 Architectures      │
│  - OOT Chrono Split     │     │ - K-Means Personas (6D)   │     │   - Optuna Optimization  │
│  - Embargo Window       │     │ - Structural Imputation   │     │   - Isotonic Calibration │
└─────────────────────────┘     └───────────────────────────┘     └──────────────────────────┘
                                                                               │
                                                                               ▼
┌─────────────────────────┐     ┌───────────────────────────┐     ┌──────────────────────────┐
│ Production Decision API │ <── │  Explainability (TreeSHAP)│ <── │  Financial Optimization  │
│ - Real-time Payload API │     │  - Global Attributions    │     │  - Cost Matrix & Cutoffs │
│ - Adverse Action Codes  │     │  - Error / FN Analysis    │     │  - Architecture Showdown │
│ - Risk-Based Pricing    │     │  - Waterfall Local Codes  │     │  - +$10.53M Net Lift     │
└─────────────────────────┘     └───────────────────────────┘     └──────────────────────────┘
```

1. **Environment Setup & High-Performance Ingestion:** Rapid data wrangling with Polars and seamless conversion to Pandas/Scikit-Learn.
2. **Two-Stage Leakage Audit & Deduplication:**
   * *Round 1 (Trivial Drops):* Elimination of post-origination payment tracking, recoveries, settlement flags, and uninformative IDs.
   * *Feature Consolidation:* Pre-drop aggregation of joint borrower assets (`total_annual_inc`, `total_dti`).
   * *Round 2 (Domain & Data Quality Pruning):* Removal of hidden dynamic attributes (`funded_amnt`, `num_tl_*dpd`, `hardship_*`, >75% null features).
3. **Temporal Out-of-Time (OOT) Splitting & Embargo Period:**
   * Chronological 80/20 train/test partition.
   * Embargo filter on loans issued $\ge 2016$ to eliminate survivorship bias and right-censoring on 36-month loan terms.
4. **Exploratory Data Analysis (EDA) & Unsupervised Discovery:**
   * Class imbalance profiling (82.0% Paid vs 18.0% Charged Off).
   * Structural missingness discovery (e.g., `emp_length` missingness correlates with higher risk; `mths_since_*` nulls represent clean credit histories).
   * K-Means clustering across a 6D independent financial feature space (validated via Elbow and Silhouette analysis) projecting 3 distinct borrower personas (*Prime/Safe*, *High Exposure*, *Strained/High Risk*).
   * Dimensionality reduction via PCA (55.1% variance explained in 2D) and non-linear t-SNE neighborhood analysis.
   * Isolation Forest anomaly detection to isolate multi-dimensional extreme-income / ultra-wealthy outliers without distorting tree splits.
5. **Dual Feature Space Pipelines:**
   * *Tree-based Universe (17 features):* Raw non-linear signals, recency transformations ($1/(x+1)$), and native categories.
   * *Linear/Neural Universe (20 dimensions):* P99 log1p transformations, one-hot encodings, and strict train-only `StandardScaler`.
6. **Multi-Model Benchmark & Hyperparameter Tuning:**
   * Systematic comparison across 5 algorithms: Logistic Regression, Random Forest, XGBoost, MLP Neural Network, and LightGBM.
   * Bayesian hyperparameter optimization with Optuna driving LightGBM ROC-AUC from $0.7188 \rightarrow 0.7232$.
   * Probability calibration via Isotonic Regression, improving Brier Score from $0.210 \rightarrow 0.145$.
7. **Business Threshold Optimization & Architectural Showdown:**
   * Tailored expected profit matrix replacing the naive $T = 0.50$ cutoff with the empirical optimal threshold $T^* = 0.20$.
   * Head-to-head comparison of (1) Legacy LC Grades, (2) In-House Purist Model, and (3) Combined Ensemble Model.
8. **Explainable AI (XAI) & Production Decision Engine:**
   * Global and local TreeSHAP attribution analysis identifying engineered solvency ratios (`installment_to_income`, `loan_to_income`) as dominant risk drivers.
   * FCRA-compliant Adverse Action code generation for automated loan rejection reasoning.
   * Deployment of an end-to-end decision function (`credit_decision_engine`) processing raw JSON application payloads into automated underwriting decisions and risk-adjusted pricing.

---

## Tools, Frameworks & Libraries
* **Data Processing & Analytics:** Polars, Pandas, NumPy
* **Machine Learning & Modeling:** Scikit-Learn, LightGBM, XGBoost
* **Hyperparameter Optimization:** Optuna
* **Unsupervised Learning & Anomaly Detection:** K-Means, PCA, t-SNE, Isolation Forest
* **Explainability & Interpretability:** SHAP (TreeSHAP)
* **Visualization & Reporting:** Matplotlib, Seaborn

---

## Business Takeaways, Financial Profit & Value Generated
* **Statistical Performance:**
  * In-House Purist Model achieved **$0.7287$ ROC-AUC** and **$0.4042$ PR-AUC** strictly from raw applicant fundamentals (no reliance on external legacy grades).
  * Combined Ensemble Model achieved **$0.7363$ ROC-AUC** and **$0.4080$ PR-AUC**.
* **Financial Monetization (On Held-Out 165K Test Portfolio):**
  * Optimizing the approval cutoff to $T^* = 0.20$ generated **+$10.53 Million in incremental net profit** (+8.5% lift over Lending Club's historical underwriting baseline).
  * Reduced portfolio default rate from 18.0% down to **11.02%** among approved loans.
* **Domain & Risk Insights:**
  * *Solvency Ratios Dominate Raw Totals:* A borrower's debt-to-income and payment-to-income ratios carry significantly more predictive weight than raw income or loan amounts.
  * *Verification Paradox:* Borrowers with verified income historically defaulted at higher rates because verifications were disproportionately triggered for high-risk, borderline applicants.
  * *Unsupervised Risk Stratification:* Geometric clustering without target labels naturally separated borrowers into distinct default risk tiers ranging from 12.1% to 23.7%.

---

## Strategic Recommendations & Future Roadmap
1. **Reject Inference on Rejected Applications:** Incorporate the 1.7GB Lending Club rejected loans dataset via Heckman two-stage selection or semi-supervised labeling to eliminate applicant survivorship bias.
2. **Geospatial & Macroeconomic Enrichment:** Link 3-digit zip codes to U.S. Census and Bureau of Labor Statistics (BLS) data (regional unemployment, median county income, local home price index).
3. **Dynamic Macro Rate Stress Testing:** Integrate real-time Federal Reserve interest rates and CPI trends to model credit portfolio resilience under shifting interest rate environments.
4. **Continuous Monitoring & Drift Detection:** Implement Population Stability Index (PSI) and Concept Drift monitoring to trigger automated retraining when economic shifts occur.
