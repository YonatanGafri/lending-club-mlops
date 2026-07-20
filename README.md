# Lending Club Credit Risk Modeling 🏦📊

This repository contains a comprehensive, end-to-end Data Science project predicting loan defaults using the Lending Club dataset. Designed with production-level standards, it tackles extreme class imbalance, strictly guards against data leakage, and optimizes models for actual financial impact.

## 🌟 Key Highlights & Advanced Features

* **Rigorous Data Preprocessing & Leakage Prevention:** Built with strict separation of train/test data to prevent information leakage, simulating real-world Out-Of-Time (OOT) validation.
* **Advanced Feature Engineering (WOE & IV):** Implemented Weight of Evidence (WOE) and Information Value (IV) analysis to interpret categorical variables strictly through their predictive power, standardizing risk signals.
* **Unsupervised Learning & Anomaly Detection:**
  * **PCA Factor Loadings:** Dimensionality reduction to uncover latent "borrower personas".
  * **Isolation Forest:** Proactive anomaly detection to identify and flag outliers before they skew the model.
* **State-of-the-Art Modeling & Ensembling:** 
  * Baseline Logistic Regression vs. optimized LightGBM (via Optuna).
  * **Soft Voting Ensemble:** Combining the interpretability of linear models with the non-linear predictive power of tree-based models.
* **Financial Threshold Optimization:** Moved beyond standard ML metrics (AUC, F1) to optimize the classification threshold directly for **Net Expected Profit**, maximizing ROI for the lending institution.
* **Explainable AI (XAI):** Integrated SHAP (SHapley Additive exPlanations) values to provide deep interpretability into the drivers of default, down to the individual borrower level.
* **Deep Error Analysis:** Targeted investigation into False Negatives (high-risk borrowers who "looked good on paper") to understand model blind spots.

## 📁 Repository Structure

* `Lending_Club_Project.ipynb`: The main Jupyter Notebook containing the full pipeline, from EDA to Model Deployment considerations.
* `Lending_Club_Project.html`: Exported HTML version of the notebook for easy viewing in the browser.
* `model_results.json`: Cached evaluation metrics and financial projections.
* `/eda_plots`, `/fe_plots`, `/model_plots`: Saved high-resolution visualizations from the analysis (SHAP, Calibration curves, colinearity heatmaps, etc.).

## 🚀 Getting Started

To run the notebook locally:
1. Clone this repository.
2. Ensure you have Python installed, along with libraries: `pandas`, `polars`, `scikit-learn`, `lightgbm`, `optuna`, `shap`, `seaborn`, `matplotlib`.
3. Open `Lending_Club_Project.ipynb` in Jupyter Notebook or VS Code.

## 📈 Results summary

The final Ensemble model successfully isolates high-risk loans, dramatically increasing the theoretical net profit compared to the baseline approval strategy. By leveraging Threshold Optimization, the model strategically balances risk and reward, proving the immense business value of advanced data science in the credit domain.
