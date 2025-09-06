# CreditPathAI
Springboard Infosys
CreditPathAI – Loan Recovery Prediction and Optimization
Project Objective

CreditPathAI automates and optimizes the loan recovery lifecycle by:

Predicting borrower default risk.

Modeling repayment behavior using diverse data.

Recommending personalized recovery actions.

The project leverages open-source machine learning tools to ensure scalability, cost-effectiveness, and actionable insights for improving delinquency recovery efficiency.

Project Workflow

Data Ingestion

Datasets: Kaggle Loan Default Dataset, Microsoft R Server Loan Credit Risk.

Tools: Pandas, CSV/Excel, SQLite/PostgreSQL.

Feature Engineering

Creation of derived features (e.g., Total Income).

Transformation of skewed variables (e.g., log transformation of Loan Amount).

Encoding categorical variables.

Handling missing values with median/mode imputation.

Model Training

Logistic Regression.

Decision Tree.

Random Forest.

Support Vector Machine (SVM).

XGBoost.

Evaluation

Accuracy, Precision, Recall, F1-score.

Confusion Matrix.

Model comparison visualizations.

Deployment (Future Work)

REST API using Flask/Django.

Dashboard for loan officers.

Results
Model	Accuracy
Logistic Regression	86.1%
Random Forest	85.3%
SVM	85.3%
XGBoost	82.1%
Decision Tree	78.0%

Best Model: Logistic Regression with 86.1% accuracy.
This enables early identification of default risks, supporting targeted recovery strategies.

Tech Stack

Programming Language: Python

Libraries: Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn

Tools: Jupyter Notebook/Google Colab, GitHub, Kaggle
