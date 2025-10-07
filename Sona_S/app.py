import streamlit as st
import pandas as pd
import joblib

# Load the trained Random Forest model
rf_model = joblib.load("model.pkl", mmap_mode='r')  

st.title(" Loan Default Prediction 🚀")
st.write("Enter applicant details to predict loan default:")

age = st.number_input("Age", min_value=18, max_value=100, value=30)
employment_type = st.selectbox("Employment Type", ["Salaried", "Self-Employed", "Unemployed", "Other"])
education = st.selectbox("Education", ["High School", "Bachelor", "Master", "PhD", "Other"])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
loan_purpose = st.selectbox("Loan Purpose", ["Home", "Car", "Education", "Business", "Other"])

loan_id = st.text_input("Loan ID (Optional)")
income = st.number_input("Income (USD)", min_value=0)
loan_amount = st.number_input("Loan Amount (USD)", min_value=0)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
months_employed = st.number_input("Months Employed", min_value=0)
num_credit_lines = st.number_input("Number of Credit Lines", min_value=0)
interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=100.0, value=10.0)
loan_term = st.number_input("Loan Term (months)", min_value=1)
dti_ratio = st.number_input("Debt-to-Income Ratio (%)", min_value=0.0, max_value=100.0, value=20.0)
has_mortgage = st.selectbox("Has Mortgage?", ["Yes", "No"])
has_dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
has_co_signer = st.selectbox("Has Co-signer?", ["Yes", "No"])


def encode_input(df):
    df['Education'] = df['Education'].map({"High School":0, "Bachelor":1, "Master":2, "PhD":3, "Other":4})
    df['EmploymentType'] = df['EmploymentType'].map({"Salaried":0, "Self-Employed":1, "Unemployed":2, "Other":3})
    df['MaritalStatus'] = df['MaritalStatus'].map({"Single":0, "Married":1, "Divorced":2, "Widowed":3})
    df['HasMortgage'] = df['HasMortgage'].map({"No":0, "Yes":1})
    df['HasDependents'] = df['HasDependents'].map({"No":0, "Yes":1})
    df['HasCoSigner'] = df['HasCoSigner'].map({"No":0, "Yes":1})
    df['LoanPurpose'] = df['LoanPurpose'].map({"Home":0, "Car":1, "Education":2, "Business":3, "Other":4})
    return df

if st.button("🔍 Predict Default"):
   
    input_df = pd.DataFrame([[
        loan_id, age, income, loan_amount, credit_score, months_employed,
        num_credit_lines, interest_rate, loan_term, dti_ratio,
        education, employment_type, marital_status, has_mortgage,
        has_dependents, loan_purpose, has_co_signer
    ]], columns=[
        "LoanID", "Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed",
        "NumCreditLines", "InterestRate", "LoanTerm", "DTIRatio",
        "Education", "EmploymentType", "MaritalStatus", "HasMortgage",
        "HasDependents", "LoanPurpose", "HasCoSigner"
    ])

   
    input_df_encoded = encode_input(input_df)

    try:
        
        prediction = rf_model.predict(input_df_encoded)[0]
        probability = rf_model.predict_proba(input_df_encoded)[0][1]

        
        st.subheader("📊 Loan Default Risk Prediction")

        if prediction == 1:
            st.markdown(f"⚠️ **High Risk!** This loan is likely to **DEFAULT**.")
            st.progress(min(probability, 1.0)) 
        else:
            st.markdown(f"✅ **Low Risk!** This loan is likely to be **REPAID** safely.")
            st.progress(min(1-probability, 1.0))  

        st.info(f"**Probability of Default:** {probability:.2f}")

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
