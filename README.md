# Loan Approval Prediction

This repository contains a loan approval classification project built from the `loan_sanction_train.csv` dataset. The goal is to predict whether a loan application will be approved based on customer details and financial information.

## Prediction Website UI
Link : [website](https://home-loan-approval-brian-maxwell-ketaren.streamlit.app/)

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

The notebook compares model performance using specificity and AUC metrics:

- Logistic Regression:
  - Ordinary: specificity ~0.70, AUC ~0.73
  - SMOTETomek: specificity ~0.67, AUC ~0.71
  - Bayesian tuning: specificity ~0.67, AUC ~0.72
- XGBoost:
  - Ordinary: specificity ~0.67, AUC ~0.72
  - SMOTETomek: specificity ~0.70, AUC ~0.74
  - Bayesian tuning: specificity ~0.67, AUC ~0.72

## Conclusion

- The best discrimination performance was achieved by `XGBoost` with SMOTETomek.
- `Credit_History` remains the most important feature for loan approval decisions.
- Data preprocessing, feature engineering, and handling imbalance are crucial for improving model reliability.
- There is a trade-off between specificity and AUC, so model selection should align with the business objective (e.g., minimizing false loan approvals vs. maximizing overall discrimination).

## Files

- `LOAN APPROVAL.ipynb`: Main Jupyter notebook containing the full analysis, visualizations, and model training.

## Tools and Libraries

- Python
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- imbalanced-learn
- xgboost
- scikit-optimize
