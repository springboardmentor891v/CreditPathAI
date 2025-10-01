# app_safe_optimized.py

import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# --- App Config ---
st.set_page_config(page_title="💳 Loan Default Prediction", layout="wide")
st.title("💳 Loan Default Prediction")
st.write("Enter applicant details in the sidebar, select a model, and predict default risk.")

# --- Available models ---
models = {
    "Logistic Regression": "saved_models/logistic_regression.pkl",
    "Decision Tree": "saved_models/decision_tree.pkl",
    "Random Forest": "saved_models/random_forest.pkl",
    "Gradient Boosting": "saved_models/gradient_boosting.pkl",
    "KNN": "saved_models/knn.pkl",
    "Naive Bayes": "saved_models/naive_bayes.pkl",
    "XGBoost": "saved_models/xgboost.pkl"
}

# --- Sidebar Inputs ---
st.sidebar.header("Enter Applicant Details")

loan_id = st.sidebar.text_input("Loan ID", "L001")
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0, step=1000)
income = st.sidebar.number_input("Annual Income", min_value=0, step=1000)
credit_score = st.sidebar.number_input("Credit Score", min_value=300, max_value=850, step=1)
age = st.sidebar.number_input("Age", min_value=18, max_value=100, step=1)
months_employed = st.sidebar.number_input("Months Employed", min_value=0, step=1)
num_credit_lines = st.sidebar.number_input("Number of Credit Lines", min_value=0, step=1)
interest_rate = st.sidebar.number_input("Interest Rate (%)", min_value=0.0, max_value=100.0, step=0.1)
loan_term = st.sidebar.number_input("Loan Term (months)", min_value=6, max_value=360, step=6)
dti_ratio = st.sidebar.number_input("Debt-to-Income Ratio (%)", min_value=0.0, max_value=100.0, step=0.1)

education = st.sidebar.selectbox("Education", ["High School", "Bachelor", "Master", "PhD"])
employment_type = st.sidebar.selectbox("Employment Type", ["Salaried", "Self-Employed", "Unemployed", "Other"])
marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced"])
has_mortgage = st.sidebar.selectbox("Has Mortgage", ["Yes", "No"])
has_dependents = st.sidebar.selectbox("Has Dependents", ["Yes", "No"])
loan_purpose = st.sidebar.selectbox("Loan Purpose", ["Personal", "Education", "Home", "Car", "Business", "Other"])
has_cosigner = st.sidebar.selectbox("Has Co-Signer", ["Yes", "No"])

# --- Cache pipeline and expected columns ---
@st.cache_resource
def load_pipeline(model_path):
    pipeline = joblib.load(model_path)
    expected_cols = pipeline.named_steps["preprocessor"].get_feature_names_out()
    return pipeline, expected_cols

# --- Main Page ---
st.subheader("Select Model & Predict")
model_choice = st.selectbox("Select Model", list(models.keys()))

if st.button("🔮 Predict"):
    model_path = models[model_choice]
    
    if os.path.exists(model_path):
        try:
            pipeline, expected_cols = load_pipeline(model_path)
            
            # Create input DataFrame
            input_dict = {
                "LoanID": loan_id,
                "Age": age,
                "Income": income,
                "LoanAmount": loan_amount,
                "CreditScore": credit_score,
                "MonthsEmployed": months_employed,
                "NumCreditLines": num_credit_lines,
                "InterestRate": interest_rate,
                "LoanTerm": loan_term,
                "DTIRatio": dti_ratio,
                "Education": education,
                "EmploymentType": employment_type,
                "MaritalStatus": marital_status,
                "HasMortgage": has_mortgage,
                "HasDependents": has_dependents,
                "LoanPurpose": loan_purpose,
                "HasCoSigner": has_cosigner
            }
            input_data = pd.DataFrame([input_dict])
            
            # Fill missing columns efficiently
            missing_cols = [c for c in expected_cols if c not in input_data.columns]
            if missing_cols:
                filler = pd.DataFrame([{c: 0 if input_data.select_dtypes(include="number").shape[1] else "Unknown" for c in missing_cols}])
                input_data = pd.concat([input_data, filler], axis=1)
            
            # Prediction
            prediction = pipeline.predict(input_data)[0]
            prediction_proba = pipeline.predict_proba(input_data)[0][1] if hasattr(pipeline, "predict_proba") else None
            
            if prediction == 1:
                st.error("🚨 Applicant is likely a **Defaulter**.")
            else:
                st.success("✅ Applicant is a **Non-Defaulter**.")
            
            if prediction_proba is not None:
                st.info(f"Prediction Confidence (Defaulter probability): {prediction_proba:.2f}")
            
        except Exception as e:
            st.error(f"⚠️ Prediction failed: {e}")
    else:
        st.error(f"⚠️ Model file not found: {model_path}")

# --- Optional: Evaluate on a Test Dataset ---
st.subheader("Evaluate Model on Test Dataset (Optional)")
uploaded_file = st.file_uploader("Upload CSV with actual defaults", type="csv")

if uploaded_file is not None:
    try:
        test_df = pd.read_csv(uploaded_file)
        y_true = test_df["Default"]
        X_test = test_df.drop(columns=["Default"])
        
        pipeline, _ = load_pipeline(models[model_choice])
        y_pred = pipeline.predict(X_test)
        
        acc = accuracy_score(y_true, y_pred)
        st.info(f"Accuracy on test dataset: {acc:.2f}")
        
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"⚠️ Evaluation failed: {e}")
