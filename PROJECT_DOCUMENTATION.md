# CreditPathAI - Loan Default Prediction System

---

## Name
**Rajath Raj K T**

## Organization
**Springboard Infosys Virtual Internship Program**



---

## Abstract

CreditPathAI is an end-to-end machine learning system to predict loan defaults using two datasets (Kaggle and Microsoft) with 24+ features. Seven models were evaluated, with XGBoost achieving an F1-Score of 0.8595 and 96.41% recall. It includes a Streamlit web app for real-time risk assessment, with a novel preprocessing pipeline using ColumnTransformer and class balancing.

---

## Introduction

Loan defaults risk financial losses for institutions. Traditional methods fail to capture complex borrower patterns, necessitating an automated prediction system.

---

## Literature

Existing systems use logistic regression, Random Forest, Gradient Boosting, and neural networks. A gap exists in handling diverse features and class imbalance with comprehensive preprocessing.

---

## Methodology

The project involves data collection, EDA, advanced preprocessing, model training/evaluation, and web deployment using Python, Scikit-learn, XGBoost, and Streamlit.

---

## Database

Datasets include Kaggle (148,000+ records, 24 features) and Microsoft Loan Credit Risk. Features cover demographics, financials, and loan details, with an imbalanced binary target (Default/No Default).

**Dataset Sources:**
- **Kaggle:** https://www.kaggle.com/datasets/yasserh/loan-default-dataset
- **Microsoft:** https://microsoft.github.io/r-server-loan-credit-risk/dba.html

---

## Preprocessing

Missing values were handled with mean imputation for numerical and most frequent for categorical features. A ColumnTransformer integrated StandardScaler and OneHotEncoder pipelines, with class balancing via `class_weight` and `scale_pos_weight`.

---

## Novelty

The innovation lies in a dual-pipeline preprocessing approach:
- **Numerical:** mean imputation + StandardScaler
- **Categorical:** mode imputation + OneHotEncoder

This uses ColumnTransformer, serialized with Joblib, and validated across two datasets.

---

## Proposed Method

The method includes:
1. Data loading
2. Preprocessing with ColumnTransformer
3. Training seven models
4. Evaluating performance
5. Deploying a Streamlit app for real-time predictions

---

## Results

### EDA

Histograms show loan amount is right-skewed (<$500K), Credit Score is normal (700-750), and Status is imbalanced. Correlations include loan_amount-property_value (0.73).

![Feature Distributions](images/feature_distributions.png)

### Visualizations

Key visualizations include feature distribution histograms, a correlation heatmap, and a box plot of interest rate vs. loan status.

![Correlation Matrix](images/correlation_matrix.png)

![Rate of Interest vs Loan Status](images/interest_rate_boxplot.png)

### Confusion Matrices

XGBoost has the highest true positives, Random Forest the lowest false positives, and Naive Bayes models show balanced but lower performance.

### Results

Models were compared, with XGBoost offering the best recall and Random Forest the highest accuracy.

### Accuracy, Recall, Precision, F1-score

| Model | Accuracy | Precision | Recall | F1-score |
|-------|----------|-----------|--------|----------|
| **XGBoost** | **0.922326** | **0.775397** | **0.964083** | **0.859507** |
| **Random Forest** | **0.933115** | **0.909247** | **0.809389** | **0.856417** |
| **Decision Tree** | 0.920469 | 0.839610 | 0.837227 | 0.838417 |
| **Gaussian Naive Bayes** | 0.879520 | 0.914043 | 0.564192 | 0.697718 |
| **Bernoulli Naive Bayes** | 0.854149 | 0.736257 | 0.636026 | 0.682481 |
| **K-Nearest Neighbors** | 0.876991 | 0.943371 | 0.532860 | 0.681038 |
| **Logistic Regression** | 0.832894 | 0.663815 | 0.652293 | 0.658003 |

### ROC-AUC Curves

ROC curves show Random Forest (AUC ≈ 0.98) and XGBoost (AUC ≈ 0.99) outperforming others, with all models exceeding random guessing.

![ROC-AUC Curves](images/roc_curves.png)

---

## Discussion

**Problems faced:**
- **Class Imbalance:** Addressed with balancing techniques (`class_weight='balanced'` and `scale_pos_weight`)
- **Missing Data:** Imputed strategically (mean for numerical, mode for categorical)
- **Feature Scaling Issues:** Resolved with StandardScaler in preprocessing pipeline
- **Categorical Encoding:** Handled high cardinality with OneHotEncoder
- **Model File Size:** Large files excluded from version control (added to .gitignore)
- **Package Conflicts:** Resolved scikit-learn version mismatch (upgraded to 1.7.2)
- **XGBoost Module:** Added to requirements.txt for production deployment

---

## Conclusion

CreditPathAI delivers a robust loan default prediction system with XGBoost's 0.8595 F1-Score and a functional Streamlit app, enhancing risk management for financial institutions.

**Key Achievements:**
- 96.41% recall ensuring minimal missed defaults
- Production-ready web application
- Comprehensive preprocessing pipeline
- Multi-dataset validation

---

## Future Works

**Future plans include:**

### ONNX Integration
Convert models to ONNX (Open Neural Network Exchange) format for:
- Cross-platform deployment (Python, Java, C++)
- Faster inference times
- Smaller model file sizes
- Better interoperability

### Model Quantization
Reduce model precision (float32 → int8) to:
- Achieve 4x smaller model size
- Improve inference speed on edge devices
- Lower memory footprint
- Maintain accuracy with minimal loss

### Additional Enhancements
- Hyperparameter tuning with GridSearchCV
- Deep learning models (LSTM, neural networks)
- RESTful API development
- Model explainability (SHAP, LIME)
- Cloud deployment (AWS/Azure/GCP)

---

**Document Prepared by:** Rajath Raj K T  
**Organization:** Springboard Infosys Virtual Internship Program  
**Date:** October 9, 2025

---