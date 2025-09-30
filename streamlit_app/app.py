# streamlit_app/app.py
import streamlit as st
import pandas as pd
# The utils import remains the same
from utils import load_model, predict_loan_default, get_available_models, get_model_performance

# --- Function to display the input form in the sidebar ---
def show_input_form():
    """Displays the input form in the sidebar for user to enter data."""
    st.header("Applicant Information")
    st.write("Enter the applicant's details to generate a prediction.")

    # --- Create input fields for all features ---
    
    # Using columns for better layout
    col1, col2 = st.columns(2)

    with col1:
        # --- Categorical Features ---
        gender = st.selectbox("Gender", ['Sex Not Available', 'Male', 'Joint', 'Female'])
        approv_in_adv = st.selectbox("Pre-approved?", ['nopre', 'pre', 'Not Provided'])
        loan_type = st.selectbox("Loan Type", ['type1', 'type2', 'type3'])
        loan_purpose = st.selectbox("Loan Purpose", ['p1', 'p4', 'p3', 'p2', 'Not Provided'])
        credit_worthiness = st.selectbox("Credit Worthiness", ['l1', 'l2'])
        business_or_commercial = st.selectbox("Business or Commercial", ['nob/c', 'b/c'])
        neg_ammortization = st.selectbox("Negative Amortization", ['not_neg', 'neg_amm', 'Not Provided'])
        interest_only = st.selectbox("Interest Only", ['not_int', 'int_only'])
        lump_sum_payment = st.selectbox("Lump Sum Payment", ['not_lpsm', 'lpsm'])
        submission_of_application = st.selectbox("Application Submitted To", ['to_inst', 'not_inst', 'Not Provided'])
        region = st.selectbox("Region", ['south', 'North', 'central', 'North-East'])
        
    with col2:
        occupancy_type = st.selectbox("Occupancy Type", ['pr', 'sr', 'ir'])
        total_units = st.selectbox("Total Units", ['1U', '2U', '3U', '4U'])
        credit_type = st.selectbox("Credit Type", ['EXP', 'EQUI', 'CRIF', 'CIB'])
        age = st.selectbox("Age Bracket", ['25-34', '55-64', '35-44', '45-54', '65-74', '>74', '<25', 'Not Provided'])
        loan_limit = st.selectbox("Loan Limit", ['cf', 'ncf', 'Not Provided'])

    st.markdown("---") # Visual separator

    # --- Numerical Features ---
    # Use sliders for a more interactive feel
    loan_amount = st.slider("Loan Amount ($)", min_value=16500, max_value=3576500, value=150000, step=1000)
    rate_of_interest = st.slider("Rate of Interest (%)", min_value=0.0, max_value=8.0, value=4.5, step=0.1)
    term = st.slider("Loan Term (months)", min_value=96, max_value=360, value=360, step=12)
    property_value = st.slider("Property Value ($)", min_value=8000, max_value=16508000, value=250000, step=5000)
    income = st.slider("Applicant Income ($/month)", min_value=0, max_value=578580, value=5000, step=100)
    credit_score = st.slider("Credit Score", min_value=500, max_value=900, value=700, step=1)
    ltv = st.slider("Loan to Value (LTV)", min_value=0.9, max_value=100.0, value=80.0, step=0.1) # Assuming LTV should be %
    dtir1 = st.slider("Debt-to-Income Ratio (DTI)", min_value=5.0, max_value=61.0, value=36.0, step=0.5)

    # --- Collect all inputs into a dictionary ---
    input_data = {
        'loan_limit': None if loan_limit == 'Not Provided' else loan_limit,
        'Gender': gender,
        'approv_in_adv': None if approv_in_adv == 'Not Provided' else approv_in_adv,
        'loan_type': loan_type,
        'loan_purpose': None if loan_purpose == 'Not Provided' else loan_purpose,
        'Credit_Worthiness': credit_worthiness,
        'business_or_commercial': business_or_commercial,
        'loan_amount': loan_amount,
        'rate_of_interest': rate_of_interest,
        'term': term,
        'Neg_ammortization': None if neg_ammortization == 'Not Provided' else neg_ammortization,
        'interest_only': interest_only,
        'lump_sum_payment': lump_sum_payment,
        'property_value': property_value,
        'occupancy_type': occupancy_type,
        'total_units': total_units,
        'income': income,
        'credit_type': credit_type,
        'Credit_Score': credit_score,
        'age': None if age == 'Not Provided' else age,
        'submission_of_application': None if submission_of_application == 'Not Provided' else submission_of_application,
        'LTV': ltv,
        'Region': region,
        'dtir1': dtir1
    }
    return input_data

def main():
    st.set_page_config(page_title="CreditPathAI", layout="wide") # Use wide layout
    
    st.title("🏦 CreditPathAI - Loan Default Prediction")
    st.write("An interactive tool to predict loan default risk using various machine learning models. "
             "Adjust the applicant's information in the sidebar to see how it affects the prediction.")

    # --- Sidebar ---
    with st.sidebar:
        st.header("Model Controls")
        available_models = get_available_models()
        if not available_models:
            st.error("No trained models found! Please run train_models.py first.")
            return
        
        selected_model = st.selectbox("Choose a model:", available_models, index=0) # Default to first model
        
        performance_df = get_model_performance()
        if performance_df is not None:
            st.subheader("Selected Model Performance")
            model_perf = performance_df[performance_df['Model'] == selected_model]
            if not model_perf.empty:
                perf = model_perf.iloc[0]
                st.metric("Recall", f"{perf['Recall']:.3f}")
                st.metric("F1-Score", f"{perf['F1-score']:.3f}")
                st.metric("Precision", f"{perf['Precision']:.3f}")

        st.markdown("---")
        # Display the input form and get the data
        input_data = show_input_form()

    # --- Main Content Area ---
    col1, col2 = st.columns([1.5, 1]) # Create two columns, one larger than the other

    with col1:
        st.header("🔮 Prediction Results")
        
        # Add a prediction button
        if st.button("🎯 Predict Default Risk", type="primary"):
            try:
                model = load_model(selected_model)
                result = predict_loan_default(model, input_data)
                
                st.write("---")
                pred_col1, pred_col2 = st.columns(2)
                with pred_col1:
                    if result['prediction_label'] == 'Default':
                        st.error(f"## Prediction: {result['prediction_label']}")
                    else:
                        st.success(f"## Prediction: {result['prediction_label']}")
                
                with pred_col2:
                    st.metric("Default Probability", f"{result['probability_default']:.2%}")

                # Risk assessment
                if result['probability_default'] > 0.7:
                    st.error("**Risk Assessment: 🔴 High Risk**")
                elif result['probability_default'] > 0.3:
                    st.warning("**Risk Assessment: 🟡 Medium Risk**")
                else:
                    st.success("**Risk Assessment: 🟢 Low Risk**")

            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")
        else:
            st.info("Adjust the parameters in the sidebar and click 'Predict Default Risk'.")

    with col2:
        st.header("📊 Model Performance Overview")
        if performance_df is not None:
            st.dataframe(performance_df.sort_values('F1-score', ascending=False).set_index('Model'))
        else:
            st.warning("Performance summary not found.")
            
    st.markdown("---")
    st.markdown("*CreditPathAI - Powered by Machine Learning*")

if __name__ == "__main__":
    main()