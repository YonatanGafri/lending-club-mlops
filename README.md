# Lending Club - End-to-End Credit Risk Machine Learning Engine 🏦📊

An enterprise-grade, end-to-end Machine Learning pipeline that predicts personal loan defaults, optimizes underwriting profit, and delivers regulatory-compliant decision intelligence on the Lending Club portfolio (1.2M+ loans).

---

## 🚀 Executive Highlights & Key Business Results
* **Empirical Generalization on Held-Out Test (165,321 loans):**
  * **In-House Purist Model:** `0.7287 ROC-AUC` | `0.4042 PR-AUC` (Independent credit engine built strictly from borrower fundamentals).
  * **Combined Ensemble Champion:** `0.7363 ROC-AUC` | `0.4080 PR-AUC` (Augmenting legacy ratings with domain machine learning).
* **Multi-Million Dollar Profit Monetization:** 
  * Calibrated default probability with Isotonic Regression (Brier Score: `0.145`).
  * Optimized underwriting decision threshold to $T^* = 0.20$ based on asymmetric loan economics ($1:4$ default loss ratio).
  * **Generated +$10.53 Million in incremental profit on the unseen Test Set** (+8.5% lift over Lending Club's legacy baseline) while slashing portfolio default rate down to **11.02%**.
* **Zero-Leakage & Real-Time Production Engine:**
  * Strict Out-of-Time (OOT) chronological partitioning.
  * Resolved the *Pricing Circularity Challenge* via automated 6D unsupervised K-Means Personas.
  * Real-time automated decision function (`credit_decision_engine`) returning approvals, risk-based pricing, and FCRA-compliant Adverse Action reason codes via TreeSHAP.

---

## 📁 Repository Structure
* `Lending_Club_Project_Semifinal.ipynb`: The primary monolithic Jupyter Notebook with visualizations and analysis.
* `Lending_Club_Project_Final.py`: The standalone Python implementation of the entire pipeline.

---

## 🗺️ High-Level Project Pipeline

### 1. Data Integrity & Leakage Prevention (Pre-Origination Audit)
* **Target Proxy & Circularity Audit:** Dropped post-origination repayment trackers (`total_pymnt`, `recoveries`, `settlement_status`) and proxy variables to guarantee realistic pre-loan conditions.
* **Structural Missingness Encoding:** Converted non-random missingness in credit history (e.g., `mths_since_last_delinq`) into recency velocity features ($1 / (x + 1)$), mapping clean borrowers to `0.0`.

### 2. Domain Feature Engineering & Unsupervised Clustering
* **Financial Leverage Ratios:** Engineered critical solvency metrics: `installment_to_income`, `loan_to_income`, and `adjusted_dti` (debt-to-income augmented with revolving credit utilization).
* **Unsupervised Borrower Personas:** Clustered applicants into 4 distinct risk archetypes using K-Means across a 6D financial feature space mapped via PCA.
* **Anomaly Detection (Isolation Forest):** Flagged multi-dimensional outliers (`is_anomaly`), capturing high-wealth / extreme-income profiles without distorting model distributions.

### 3. Dual Feature Space & Leakage-Free Scaling
* **Tree Feature Universe (17 features):** Maintained native raw distributions, categorical types, and structural missingness for gradient-boosted trees.
* **Linear / Neural Universe (20 dimensions):** Applied P99 Log-Capping on heavy-tailed dollar amounts, consolidated categoricals into 3 macro risk tiers, and fitted `StandardScaler` strictly on the training partition.

### 4. Supervised Model Benchmarking (Algorithmic Parity)
Evaluated 5 distinct architectures under regularized conditions:
* `Logistic Regression` (Linear Baseline): $0.7113$ ROC-AUC
* `Random Forest` (Tree Bagging): $0.7023$ ROC-AUC
* `LightGBM` (Gradient Boosting): $0.7188$ ROC-AUC
* `XGBoost` (Gradient Boosting): $0.7199$ ROC-AUC
* `MLP Neural Network` (Deep Learning): $0.7252$ ROC-AUC (demonstrating the power of clean feature engineering on tabular data).

### 5. Bayesian Optimization & Probability Calibration (LightGBM Champion)
* **Optuna Tuning:** Explored hyperparameter space, driving LightGBM ROC-AUC from $0.7188 \rightarrow \mathbf{0.7232}$ with heavy $L_1$ regularization (`reg_alpha = 9.31`).
* **Isotonic Calibration:** Aligned raw tree probabilities onto the empirical 45° line, dropping the Brier Score from $0.210 \rightarrow \mathbf{0.145}$.

### 6. Financial Cost Matrix & Threshold Optimization
* Replaced the naive $T = 0.50$ cutoff with an empirical cost matrix ($+\$2,800$ interest margin on good loans vs. $-\$10,500$ principal loss on defaults).
* Optimized cutoff at $T^* = 0.20-0.21$, nearly tripling net portfolio profit.

### 7. Explainable AI (TreeSHAP) & Error Analysis
* **Global Attributions:** Proved `installment_to_income` and `loan_to_income` are the **#2 and #3 most impactful features in the entire model**, exceeding raw loan amount and income.
* **Adverse Action Reason Codes:** Built local Waterfall decompositions explaining individual credit denials.
* **False Negative Error Analysis:** Demonstrated that approved defaulting borrowers possessed strong origination fundamentals (median FICO 690, DTI 23.8%), proving their default was driven by post-origination exogenous life shocks.

### 8. Architectural Showdown & The Multi-Million Dollar Lift
* **Lending Club Baseline Alone:** $0.7162$ Test ROC-AUC ($\$124.34\text{M}$ profit)
* **In-House Purist Model:** $0.7287$ Test ROC-AUC ($\$130.90\text{M}$ profit, **$+\$6.55\text{M}$ lift**)
* **Combined Ensemble Model:** $\mathbf{0.7363}$ Test ROC-AUC ($\$134.88\text{M}$ profit, **$+\$10.53\text{M}$ lift**)

### 9. Production Decision Engine (`credit_decision_engine`)
Operationalized a real-time underwriting function that consumes raw applicant JSON payloads, dynamically computes engineered ratios, maps borrower personas, verifies anomalies, applies optimal cutoffs, and returns risk-based interest rates and FCRA denial reasons.

---

## 🔮 Future Work & Continuous Improvement Roadmap

1. **Socio-Economic Zip Code Census Mapping:**
   Currently, `zip_code` is a 3-digit prefix. Future iterations will map zip codes to external U.S. Census & Bureau of Labor Statistics (BLS) data (median county income, regional unemployment, home price index) to inject real-time macroeconomic context.
2. **Reject Inference on the 1.7GB Rejected Loan Dataset:**
   Our model is trained on *accepted* loans (creating potential survivorship bias). By implementing semi-supervised **Reject Inference** (e.g., Fuzzy Augmentation / Two-Stage Heckman Selection Models), we can infer the outcomes of rejected applicants and refine borderline cutoffs.
3. **Macroeconomic Climate & Dynamic Fed Rate Integration:**
   Integrate real-time Fed Funds interest rates and Consumer Price Index (CPI) trends to enable dynamic stress-testing under inflation and interest-rate hike environments.
4. **Data Drift & Monitoring (Population Stability Index - PSI):**
   Deploy automated drift monitors on production payloads. When population distribution drift exceeds critical thresholds ($\text{PSI} > 0.25$), trigger automated retraining pipelines.
