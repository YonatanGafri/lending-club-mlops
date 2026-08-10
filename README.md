# Lending Club Credit Risk Modeling 🏦📊

This repository contains a comprehensive Data Science project predicting loan defaults using the Lending Club dataset. The current phase of the project focuses heavily on robust data preparation, strict prevention of data leakage, and advanced feature engineering designed for production-level ML architectures.

## 📁 Repository Structure
* `Lending_Club_Project_Semifinal.ipynb`: The main Jupyter Notebook containing the pipeline.
* `Lending_Club_Project_Final.py`: The core Python script that powers the notebook.
* `/eda_plots`, `/fe_plots`: Directories containing high-resolution visual outputs of our data analysis.

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
* **Evaluation Metrics:** ROC AUC, F1-Score, and Precision-Recall Curves.
* **Business Translation:** Optimizing the probability threshold by applying an **Expected Profit / Cost-Benefit Matrix** to maximize ROI.
* **Explainable AI & Error Analysis:** Deep-diving into model decisions using **SHAP** values, followed by rigorous Error Analysis (investigating False Negatives and False Positives) to identify blind spots.

### 6. Production & Real-Time Inference (RT)
We aim to deploy the best model as a Real-Time (RT) Inference API tool that accepts JSON payloads and returns predictions at low latency. 
* **Architecture Thoughts:** To hit low-latency SLAs, we may need a high-performance **Feature Store** (e.g., Redis or DynamoDB) to cache pre-computed borrower aggregates and socio-economic variables, avoiding expensive on-the-fly SQL joins.

### 7. Data Drift & Monitoring
Financial landscapes evolve rapidly. We plan to implement monitoring using the **Population Stability Index (PSI)** to detect Data Drift (e.g., inflation shifting income distributions). If PSI crosses a critical threshold, the model will trigger automated retraining pipelines.

### 8. Socio-Economic Zip Code Engineering (Idea)
Currently, `zip_code` is just a high-cardinality prefix. In the future, we plan to engineer this by mapping zip codes to regional socio-economic tiers (e.g., average household income, unemployment rate) using external statistical bureau data, injecting macroeconomic context directly into the model.

---
עד כאן הפרוייקט מוכן
