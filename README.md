# Lending Club - ML Project 🏦📊

This repository contains a comprehensive Data Science project predicting loan defaults using the Lending Club dataset. The current phase of the project focuses heavily on robust data preparation, strict prevention of data leakage, and advanced feature engineering designed for production-level ML architectures.

## 📁 Repository Structure
* `Lending_Club_Project_Semifinal.ipynb`: The main Jupyter Notebook containing the pipeline.
* `Lending_Club_Project_Final.py`: The core Python script that powers the notebook.

## 🌟 Project Pipeline

### 1. Data Loading & Sub-Grade Encoding
We ingest the raw dataset and explicitly encode the `sub_grade` feature. While assigned by Lending Club, it is evaluated strictly *pre-origination*, meaning it does not constitute data leakage. We convert this ordinal string (e.g., 'A1', 'G5') into an integer scale (1-35) to give the model a powerful baseline risk proxy.

### 2. Data Quality & Cleaning (Leakage Audit)
We perform a rigorous two-round audit to eliminate **Data Leakage**:
* **Round 1 (Post-Origination):** Removed columns detailing post-funding events such as `total_pymnt`, `recoveries`, and `settlement_status`.
* **Round 2 (Implicit Proxies):** Removed columns identified through domain research as implicit post-origination trackers, such as `funded_amnt` and `chargeoff_within_12_mths`.

We also drop columns with >75% missing data, constant columns (like `policy_code`), and free-text NLP inputs that are outside the scope of tabular analysis.

### 3. Exploratory Data Analysis (EDA)
We conduct deep statistical analysis to uncover the drivers of default:
* **Out-of-Time (OOT) Split:** We enforce chronological integrity by splitting chronologically to simulate real-world deployment.
* **Univariate & Bivariate Analysis:** We map out density distributions, calculate Weight of Evidence (WOE), and map Information Value (IV) across key features.
* **Multicollinearity Audit:** Evaluated Pearson correlations and dropped highly collinear redundant variables.
* **K-Means Borrower Personas:** Clustered borrowers into three distinct profiles ("Prime/Safe", "High Exposure", and "Highly Leveraged/Strained") using a 6D financial feature space mapped via PCA.

### 4. Advanced Feature Engineering
We transform raw signals into ML-ready inputs, ensuring fairness across different model architectures:
* **Domain Financial Features:** Created powerful ratio interactions such as `adjusted_dti` (debt-to-income factoring in revolving utilization) and `installment_to_income`.
* **Recency Transformation:** Resolved structural missingness in temporal columns (e.g., `mths_since_last_delinq`). Instead of arbitrary high scalars (e.g., `999`), we applied `1 / (x + 1)`, imputing NULLs (events that never happened) to exactly `0.0` for linear stability.
* **Heavy-Tailed Log Capping:** Computed P99 thresholds to cap extreme financial outliers (like $10M incomes) and applied `.log1p()`. This prevents extreme gradients in distance-based algorithms, allowing us to build explicit `tree_features` and `linear_features` lists.

---

## 🔮 Future Roadmap & Next Steps

### 5. Modeling & Evaluation Strategy
We plan to benchmark multiple architectures—**Logistic Regression, Tree-based Ensembles (LightGBM/XGBoost), and Neural Networks**—using our meticulously engineered features.
* **Handling Class Imbalance:** For models sensitive to imbalance (like Logistic Regression), we plan to apply `class_weight='balanced'` instead of destructive techniques like SMOTE or undersampling, preserving the native distributions while penalizing minority class misclassifications.
* **Evaluation Metrics:** ROC AUC, F1-Score, and Precision-Recall Curves.
* **Business Translation:** Optimizing the probability threshold by applying an **Expected Profit / Cost-Benefit Matrix** to maximize ROI.
* **Explainable AI & Error Analysis:** Deep-diving into model decisions using **SHAP** values, followed by rigorous Error Analysis (investigating False Negatives and False Positives) to identify blind spots.

### 6. Production & Real-Time Inference (Decision Engine)
We aim to deploy the best model as a Real-Time (RT) Inference API tool that accepts JSON payloads and returns predictions at low latency. 
* **Architecture Thoughts:** To hit low-latency SLAs, we may need a high-performance **Feature Store** (e.g., Redis or DynamoDB) to cache pre-computed borrower aggregates and socio-economic variables, avoiding expensive on-the-fly SQL joins.

#### Production Inference & The Pricing Circularity Challenge

**The Production Challenge:**
In historical credit datasets, `int_rate` is a post-decision feature assigned by lenders *after* risk assessment. However, when a new applicant submits a JSON request in real-time, the interest rate is naturally unknown beforehand. Using `int_rate` in model scoring creates a potential circular dependency during real-time inference.

**Architectural Solutions Evaluated:**

1. **Proposal 1: Two-Stage Pipeline Architecture**
   • *Concept:* Decouple risk assessment into two distinct models. Stage 1 predicts baseline $PD$ using purely organic applicant features (excluding `int_rate`). Stage 2 assigns a risk grade and interest rate based on $PD$, followed by a final capacity check on `adjusted_dti`.
   • *Pros/Cons:* Highly modular, but adds computational latency and requires maintaining two separate model pipelines.

2. **Proposal 2: Persona-Based Benchmark Pricing (Selected Implementation)**
   • *Concept:* Leverage our $K$-Means unsupervised clustering topology. Upon receiving a new JSON payload, the engine calculates $6D$ Euclidean centroid distances to assign the applicant to their geometric `borrower_persona`. The engine then imputes the benchmark interest rate of that specific persona as a realistic baseline input for model scoring.
   • *Pros/Cons:* Computationally efficient, fast real-time scoring, and seamless integration with our existing feature space.

**Decision & Implementation Strategy:**
We elected to implement **Proposal 2 (Persona-Based Benchmark Pricing)** for our deployment engine. This approach provides an elegant, production-ready solution that eliminates pipeline complexity while effectively grounding interest rate estimates in our empirically validated borrower personas.

### 7. Data Drift & Monitoring
Financial landscapes evolve rapidly. We plan to implement monitoring using the **Population Stability Index (PSI)** to detect Data Drift (e.g., inflation shifting income distributions). If PSI crosses a critical threshold, the model will trigger automated retraining pipelines.

### 8. Socio-Economic Zip Code Engineering (Idea)
Currently, `zip_code` is just a high-cardinality prefix. In the future, we plan to engineer this by mapping zip codes to regional socio-economic tiers (e.g., average household income, unemployment rate) using external statistical bureau data, injecting macroeconomic context directly into the model.

### 9. Isolation Forest for Anomaly Detection
We have explored anomaly detection using **Isolation Forest** during EDA. Moving forward, we plan to fully integrate this into the main feature engineering pipeline. We will train the Isolation Forest on the scaled training set, generate an `is_anomaly` column for both the train and test sets, and explicitly add this column to our `tree_features` list so downstream models can natively leverage the anomaly signal. This decision is driven by our observation from the existing notebook run that anomalous borrowers actually exhibit a lower default rate compared to normal borrowers. Furthermore, generating an explicit anomaly flag allows the model to capture individuals subjected to extreme value capping (e.g., $10M incomes) during preprocessing. This preserves their unique "outlier" status as a powerful signal without breaking linear models' stability.
