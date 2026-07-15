# Loan Approval Prediction

This repository contains a loan approval classification project built from the `loan_sanction_train.csv` dataset. The goal is to predict whether a loan application will be approved based on customer details and financial information.

## Prediction Website UI
Link : [website](https://home-loan-approval.vercel.app/)

<img width="1853" height="967" alt="image" src="https://github.com/user-attachments/assets/0f344de1-d971-4bb7-9007-0ad8cd0286ad" />


## Project Overview

The notebook performs the following steps:

- Loaded and explored the loan sanction dataset.
- Cleaned and preprocessed the data:
  - Removed `Loan_ID` as an identifier.
  - Imputed missing values for categorical fields such as `Gender`, `Married`, `Dependents`, and `Self_Employed`.
  - Dropped rows missing key numeric values like `LoanAmount`, `Loan_Amount_Term`, and `Credit_History`.
  - Converted numeric credit history values into readable categories.
- Conducted exploratory data analysis (EDA) on numeric and categorical features.
- Performed correlation and association analysis to identify key predictors.
- Encoded categorical features and scaled numeric features using `RobustScaler`.
- Selected top features using an `ExtraTreesClassifier` importance ranking.
- Trained and evaluated multiple models:
  - Logistic Regression
  - XGBoost
  - Logistic Regression + SMOTETomek
  - XGBoost + SMOTETomek
  - Bayesian hyperparameter tuning for best model performance

## Key Findings

- The dataset is imbalanced: around 69% of loans were approved and 31% were rejected.
- `Credit_History` is the strongest predictor of loan approval.
- Graduated applicants and married applicants showed higher approval rates.
- Applicants in semi-urban areas had the highest approval ratio.
- Numeric features like `ApplicantIncome` and `LoanAmount` are skewed and benefit from robust scaling.

## Results

- EDA findings:
    - Class imbalance: The target is imbalanced, motivating oversampling/cost-sensitive strategies.
    - Missingness & cleaning: Several features contained missing values handled with imputation (simple or model-based) during preprocessing.
    - Feature distributions: Numeric features show skew (e.g., income/amount), requiring log or power transforms and scaling.
    - Correlations: Strong predictors identified include credit history, applicant income, and loan amount (and engineered ratios like debt-to-income), with some multicollinearity among related financial features.
    - Categorical patterns: Certain categorical levels (employment, dependents, property area) align with target differences and benefit from careful encoding.
    - Outliers: A small number of extreme values were present and handled (capping or removal) to stabilize training.

- Results:
    - Specificity: 
        - LR — Ordinary 0.67 / SMOTETomek 0.61 / Bayesian 0.73. 
        - XGB — Ordinary 0.73 / SMOTETomek 0.73 / Bayesian 0.73.
    - AUC: 
        - LR — Ordinary 0.74 / SMOTETomek 0.71 / Bayesian 0.77. 
        - XGB — Ordinary 0.82 / SMOTETomek 0.77 / Bayesian 0.82.
    - Summary: 
        - LR Bayesian and XGB-SMOTETomek, XGB-Bayesian gives highest specificity (0.73). 
        - XGB-Ordinary and XGB-Bayesian gives highest AUC (0.82). 

- Key insights:
    - AUC improvements do not guarantee higher specificity—trade-offs exist between discrimination and negative-class recall.
    - SMOTETomek did not consistently increase specificity; model- and feature-level.

## Conclusion

- The best discrimination performance was achieved by `XGBoost` with SMOTETomek.
- `Credit_History` remains the most important feature for loan approval decisions.
- Data preprocessing, feature engineering, and handling imbalance are crucial for improving model reliability.
- There is a trade-off between specificity and AUC, so model selection should align with the business objective (e.g., minimizing false loan approvals vs. maximizing overall discrimination).
