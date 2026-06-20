# Loan Approval Prediction System

An automated machine learning system designed to predict loan eligibility based on customer information using KNN and Artificial Neural Networks (ANN).

## 📋 Project Overview

The purpose of this project is to automate the loan eligibility process by predicting whether a loan applicant will be approved or rejected based on their personal and financial information. The system analyzes customer details such as:

- **Personal Information**: Gender, Marital Status, Number of Dependents
- **Financial Profile**: Income, Co-applicant Income, Loan Amount, Loan Term
- **Credit History**: Credit score eligibility
- **Property Details**: Property location area (Urban, Semi-Urban, Rural)

The prediction system utilizes two different machine learning algorithms:
1. **K-Nearest Neighbors (KNN)** - A distance-based classification algorithm
2. **Artificial Neural Networks (ANN)** - A deep learning approach using TensorFlow/Keras

---

## 📊 Dataset

**Source**: [Kaggle - Home Loan Approval Dataset](https://www.kaggle.com/datasets/rishikeshkonapure/home-loan-approval)

**Data File**: `loan_sanction_train.csv`

### Data Dictionary

| Feature | Description | Type |
|---------|-------------|------|
| Loan_ID | Applicant Identifier | Unique ID |
| Gender | Applicant Gender | Female / Male |
| Married | Marital Status | Y (Yes) / N (No) |
| Dependents | Number of Dependents | Integer |
| Education | Education Level | Graduate / Undergraduate |
| Self_Employed | Employment Status | Y (Yes) / N (No) |
| ApplicantIncome | Applicant's Income | Numeric ($) |
| CoapplicantIncome | Co-applicant's Income | Numeric ($) |
| LoanAmount | Requested Loan Amount | Numeric (Thousand $) |
| Loan_Amount_Term | Loan Duration | Numeric (Months) |
| Credit_History | Credit History Status | Y (Meets Guidelines) / N (Does Not Meet) |
| Property_Area | Location Type | Urban / Semi-Urban / Rural |
| **Loan_Status** | **Loan Approval Status (Target)** | **Y (Approved) / N (Rejected)** |

### Class Distribution
- **Approved (Y)**: 69.2%
- **Rejected (N)**: 30.8%

> Note: The dataset exhibits class imbalance, which is handled using SMOTE (Synthetic Minority Over-sampling Technique).

---

## 🔍 Exploratory Data Analysis (EDA)

### Key Findings

#### Numerical Features Distribution:
1. **ApplicantIncome**: Majority in range $2,900-$5,815
2. **CoapplicantIncome**: Majority in range $0-$2,232
3. **LoanAmount**: Majority in range 100-167 (thousand $)
4. **Loan_Amount_Term**: Most applicants request 360-month terms

#### Correlation Analysis:
- **Spearman Correlation**: Used for numerical features (data is non-normal)
- **Strong Correlation**: LoanAmount ↔ ApplicantIncome (ρ = 0.5)
- **Weak Correlations**: Most other feature pairs

#### Categorical Associations (Cramér's V):
- **Strong Association**: 
  - Gender ↔ Married (0.4)
  - Credit_History ↔ Loan_Status (0.5)
- **Weak Associations**: Other categorical features

#### Mixed Analysis:
- Target variable (Loan_Status) shows strongest relationship with **Credit_History (0.5)**
- Weak relationships with most other features

---

## 🛠️ Data Preprocessing

### 1. Data Cleaning
- **Dropped**: Loan_ID column (not predictive)
- **Missing Values Handling**:
  - `Gender`: Filled with mode ("Male")
  - `Married`: Filled with "unknown"
  - `Dependents`: Filled with "unknown"
  - `Self_Employed`: Filled with "No"
  - `LoanAmount`, `Loan_Amount_Term`, `Credit_History`: Rows removed

### 2. Feature Engineering
- **Encoding**: One-hot encoding for all categorical variables
  - Categorical Features: Gender, Education, Married, Dependents, Self_Employed, Credit_History, Property_Area
- **Feature Scaling**: RobustScaler
  - Applied to numerical features to minimize outlier effects
  - Robust Scaler used because data contains outliers

### 3. Train-Test Split
- **Training Set**: 80%
- **Testing Set**: 20%
- **Random State**: 42 (for reproducibility)

### 4. Feature Selection
- **Method**: ExtraTreesClassifier Feature Importance
- **Selected Features**: Top 10 most important features
- Reduces dimensionality and improves model efficiency

---

## 🤖 Machine Learning Models

### Model 1: K-Nearest Neighbors (KNN)

**Algorithm**: Distance-based classification using Euclidean distance

**Configuration**:
- K-value: Tuned via GridSearchCV
- Features: Top 10 selected features
- Scaling: RobustScaler applied

**Performance Metrics**:
- Classification Report: Precision, Recall, F1-Score
- Confusion Matrix: Shows True Positives, False Positives, True Negatives, False Negatives
- ROC-AUC Score: Measures model's ability to distinguish between classes

### Model 2: Artificial Neural Network (ANN)

**Architecture**:
- **Framework**: TensorFlow/Keras
- **Layers**: Dense layers with Dropout regularization
- **Activation Functions**: ReLU (hidden layers), Sigmoid (output layer)
- **Optimization**: Adam optimizer
- **Loss Function**: Binary Crossentropy

**Training Strategy**:
- **Early Stopping**: Custom callback stops training when accuracy > 90%
- **Class Imbalance Handling**: SMOTE applied to training data
- **Features**: Top 10 selected features
- **Batch Size & Epochs**: Configured for optimal convergence

**Performance Evaluation**:
- ROC Curve & AUC Score
- Classification Report
- Training/Validation Accuracy Curves

---

## 📈 Results

### Key Insights

1. **Target Variable Imbalance**: Loans are approved 2.2x more frequently than rejected
   - Solution: SMOTE oversampling applied to training data

2. **Feature Importance**: Credit history is the strongest predictor of loan approval

3. **Numerical Features**: 
   - Non-normal distribution → Spearman correlation used
   - Outliers present → RobustScaler applied

4. **Model Comparison**:
   - KNN: Fast inference, sensitive to feature scaling
   - ANN: Higher capacity, better for capturing complex patterns

---

## 📊 Visualizations

The notebook includes comprehensive visualizations:

- **Distribution Analysis**: Histograms and box plots for numerical features
- **Heatmaps**: Correlation matrices (Spearman, Cramér's V, Mixed)
- **Categorical Analysis**: Stacked bar charts with percentages
- **Feature Importance**: Top 10 features bar plot
- **Model Evaluation**: 
  - Confusion matrices (heatmaps)
  - ROC curves with AUC scores
  - Classification report plots

---

## 🔧 Model Hyperparameter Tuning

### KNN Tuning
```python
GridSearchCV(
    KNeighborsClassifier(),
    param_grid={'n_neighbors': [3, 5, 7, 9, 11]},
    cv=5
)
```

### ANN Architecture Example
```python
model = Sequential([
    Dense(64, activation='relu', input_dim=10),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

---

## 💡 Key Techniques Used

| Technique | Purpose |
|-----------|---------|
| **SMOTE** | Handle class imbalance in training data |
| **RobustScaler** | Scale numerical features, resistant to outliers |
| **ExtraTreesClassifier** | Feature importance extraction |
| **GridSearchCV** | Hyperparameter optimization |
| **Spearman Correlation** | Analyze non-normal numerical data |
| **Cramér's V** | Measure categorical association |
| **Kruskal-Wallis Test** | Test categorical vs numerical relationships |
| **Early Stopping Callback** | Prevent ANN overfitting |

---

## 🎯 Future Improvements

- [ ] Hyperparameter tuning for KNN (n_neighbors, weights)
- [ ] Deep ANN architecture optimization
- [ ] Cross-validation implementation
- [ ] Feature interaction analysis
- [ ] Model deployment using Flask/FastAPI
- [ ] Real-time prediction API
- [ ] Additional ensemble methods (Random Forest, Gradient Boosting, XGBoost)
- [ ] Handling missing values using imputation techniques

---

## 📝 License

This project is provided as-is for educational and learning purposes.

---

## 🙋 Questions & Contribution

For questions or suggestions, feel free to open an issue or contribute to this project.

---

## 📚 References

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [TensorFlow/Keras Documentation](https://www.tensorflow.org/)
- [Imbalanced-learn SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)
- [Statistical Methods](https://www.scipy-lectures.org/)

---

**Last Updated**: 2026-06-20  
**Author**: Data Science Enthusiast  
**Dataset Source**: [Kaggle](https://www.kaggle.com/datasets/rishikeshkonapure/home-loan-approval)

