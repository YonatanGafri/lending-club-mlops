# Original User Request

## 2026-07-27T20:53:49Z

# Teamwork Project Prompt — Draft

Review the Lending Club Data Science project against the Ultimate QA Guide to ensure it meets the high standards of an excellent student aiming for a perfect score (100). The review process should be highly collaborative, with specialized agents handling specific topics and cross-reviewing each other's findings.

Working directory: c:\Users\yonat\.gemini\antigravity\scratch\lending_club_ds
Integrity mode: development

## Requirements

### R1. Establish an Agent Team for Code Review
You must act as the manager of this review process. Delegate the review of the 5 key project phases to specialized sub-agents. For each topic, assign an investigating agent and a QA agent to cross-verify the findings. All discussions between these agents must be documented and visible in your final output.

### R2. Phase 1: Exploratory Data Analysis (EDA)
Verify aesthetics, clear presentation of the target variable (`Fully Paid` vs. `Charged Off`) and class imbalance awareness. Ensure there is business context (Markdown explanations) under every major chart.

### R3. Phase 2: Feature Engineering & Preprocessing
Check for data leakage (future variables like `total_pymnt`, `recoveries`, `int_rate`, `sub_grade` must be removed). Ensure an embargo period is applied to prevent survivorship bias (filtering out recent loans). Evaluate the creation of custom financial features and smart handling of NaNs/categories (avoiding blind `fillna(0)`).

### R4. Phase 3 & 4: Unsupervised and Supervised Learning
Check PCA variance explanation and mathematical justification for K-Means clusters, alongside business profiling. For supervised learning, ensure `TimeSeriesSplit` (Walk-Forward CV) is used instead of random K-Fold, and that hyperparameter tuning uses early stopping and chronological splits. Verify model-specific categorical handling (LightGBM vs. MLP).

### R5. Phase 5: Forensic Evaluation & Explainability
Verify the use of ROC-AUC and PR-AUC instead of Accuracy. **Critically**, cross-reference textual conclusions with actual printed code outputs to ensure 100% forensic consistency (no rounded or hallucinated metrics). Verify the presence of SHAP plots for explainability.

## Acceptance Criteria

### Review Quality
- [ ] The final report contains a detailed breakdown of all 5 phases based on the Ultimate QA Guide.
- [ ] The report includes transcripts or clear summaries of the dialogue between the reviewing agents and their QA counterparts.
- [ ] Every identified flaw or success references specific files or code blocks in the working directory.
- [ ] The final assessment provides actionable feedback for a student aiming for a 100.
