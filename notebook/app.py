import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import shap
import lime
import lime.lime_tabular
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from streamlit_theme import set_theme
from streamlit_shap import st_shap
from xgboost import XGBClassifier

from database import init_db, log_prediction_to_db, get_prediction_history
st.set_page_config(page_title="CreditPath AI", layout="wide")

def check_password():
    """Returns `True` if the user has entered the correct password."""
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"] 
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        return True

def get_recovery_actions(risk_level, default_prob):
    """Provides suggested actions based on the calculated risk level."""
    if risk_level == "High Risk":
        return f"🔴 **High Risk** (Default Probability: {default_prob:.2%}): Suggest collateral verification, shorter loan term, or decline application."
    elif risk_level == "Medium Risk":
        return f"🟠 **Medium Risk** (Default Probability: {default_prob:.2%}): Suggest reduced loan amount, higher interest rate, or an additional guarantor."
    else: # Safe
        return f"🟢 **Safe** (Default Probability: {default_prob:.2%}): Standard processing recommended. Proceed with application."

def determine_risk_level(prediction, probability):
    """Determines risk level and default probability."""
    default_prob = probability if prediction == 1 else 1 - probability
    if prediction == 1 or (prediction == 0 and default_prob >= 0.20):
        return ("High Risk" if prediction == 1 else "Medium Risk"), default_prob
    return "Safe", default_prob
@st.cache_resource
def setup_model_and_data():
    """Loads the trained model, scaler, data, and initializes explainers."""
    try:
        model = joblib.load('models/xgboost_model.joblib')
        scaler = joblib.load('models/scaler.joblib')
    except FileNotFoundError:
        st.error("Model files not found. Please run `python save_models.py` first.")
        st.stop()
        
    try:
        df = pd.read_csv('data/loan_data.csv')
        X = df.drop('default', axis=1)
        X_train = X 
    except FileNotFoundError:
        st.error("Training data ('data/loan_data.csv') not found. Please run `python save_models.py` first.")
        st.stop()
        
    shap_explainer = shap.TreeExplainer(model)

    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=['Not Default', 'Default'], # Corresponds to 0 and 1
        mode='classification'
    )
    
    return model, scaler, shap_explainer, lime_explainer, X_train.columns

model, scaler, shap_explainer, lime_explainer, feature_names = setup_model_and_data()

init_db(feature_names)


st.title("CreditPath AI: Loan Application Analysis")
st.markdown("Enter applicant details to get a loan eligibility prediction and explanation.")

st.sidebar.header("Applicant Information")
input_data = {}
with st.sidebar.form("input_form"):
    input_data['loan_amount'] = st.number_input("Loan Amount ($)", 1000, 100000, 15000, 500, help="The total amount of money requested for the loan.")
    input_data['annual_income'] = st.number_input("Annual Income ($)", 10000, 200000, 60000, 1000)
    input_data['credit_history_good'] = st.selectbox("Credit History", [0, 1], format_func=lambda x: "Good" if x == 1 else "Not Good")
    input_data['loan_term_months'] = st.selectbox("Loan Term (Months)", [36, 60], index=0)
    input_data['employment_length_years'] = st.slider("Employment Length (Years)", 0, 40, 5)
    
    predict_button = st.form_submit_button(label="Analyze Application")


if predict_button:
    input_df = pd.DataFrame([input_data], columns=feature_names)
    
    input_scaled = scaler.transform(input_df)

    # Get probabilities for both classes: [prob_approved, prob_denied]
    probabilities = model.predict_proba(input_scaled)[0]
    prediction = model.predict(input_scaled)[0]
    confidence = probabilities[prediction] # Confidence in the predicted outcome
    default_prob = probabilities[1] # The specific probability of default (class 1)

    log_prediction_to_db(input_df, prediction, confidence)
    
    st.header("Prediction Result")
    risk_level, default_prob_for_actions = determine_risk_level(prediction, confidence)

    if prediction == 0:
        st.success(f"**Loan Approved** (Confidence: {confidence:.2%})")
    else:
        st.error(f"**Loan Denied (Default Risk)** (Confidence: {confidence:.2%})")

    st.subheader("Recommended Actions")
    recovery_message = get_recovery_actions(risk_level, default_prob_for_actions)
    st.info(recovery_message)

    st.header("Prediction Explanation")
    tab1, tab2 = st.tabs(["🔹 SHAP", "🔹 LIME"])

    with tab1:
        st.subheader("SHAP (SHapley Additive exPlanations)")
        st.markdown("SHAP values show the impact of each feature on the model's output for this specific prediction.")
        
        shap_values = shap_explainer.shap_values(input_scaled)
        
        st.write("**SHAP Force Plot**")
        st.markdown("This plot shows features pushing the prediction higher (red) or lower (blue).")
        force_plot = shap.force_plot(shap_explainer.expected_value, shap_values[0,:], input_df.iloc[0,:], link="logit")
        st_shap(force_plot, height=200)

        st.write("**SHAP Waterfall Plot**")
        st.markdown("This plot breaks down the contribution of each feature from the base value to the final prediction.")
        shap_explanation = shap.Explanation(values=shap_values[0], 
                                            base_values=shap_explainer.expected_value, 
                                            data=input_df.iloc[0],
                                            feature_names=feature_names)
        shap.plots.waterfall(shap_explanation, show=False)
        st.pyplot(plt.gcf())

    with tab2:
        st.subheader("LIME (Local Interpretable Model-agnostic Explanations)")
        st.markdown("LIME explains the prediction by creating a simpler, interpretable local model around the prediction.")
        
        lime_explanation = lime_explainer.explain_instance(
            data_row=input_df.iloc[0].values,
            predict_fn=lambda x: model.predict_proba(scaler.transform(x)),
            num_features=len(feature_names)
        )
        
        st.write("**LIME Explanation Plot**")
        fig = lime_explanation.as_pyplot_figure()
        st.pyplot(fig)
        
        st.write("**LIME Explanation Details**")
        st.write(pd.DataFrame(lime_explanation.as_list(), columns=['Feature Condition', 'Weight']))


st.divider()

main_tab1, main_tab2 = st.tabs(["🔍 Individual Prediction Analysis", "📊 Portfolio Dashboard"])

history_df = get_prediction_history()

with main_tab1:
    st.header("Global Feature Importance & History")
    if history_df.empty:
        st.info("No prediction history found. Make a prediction to see the dashboard.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Global Feature Importance")
            st.markdown("Based on the trained XGBoost model.")
            
            feature_importances = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)

            fig = go.Figure(go.Bar(
                x=feature_importances['importance'],
                y=feature_importances['feature'],
                orientation='h'
            ))
            fig.update_layout(title="Model Feature Importance", yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Prediction Outcomes")
            st.markdown("Distribution of historical loan application outcomes.")
            outcome_counts = history_df['prediction'].value_counts().rename({0: 'Approved', 1: 'Denied'})
            st.bar_chart(outcome_counts)

        st.subheader("Prediction History")
        st.dataframe(history_df.sort_values('timestamp', ascending=False), use_container_width=True)

with main_tab2:
    st.header("Portfolio Risk Dashboard")
    if history_df.empty:
        st.info("No prediction history found. Make a prediction to see the dashboard.")
    else:
        st.sidebar.divider()
        st.sidebar.header("Dashboard Filters")
        history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
        min_date = history_df['timestamp'].min().date()
        max_date = history_df['timestamp'].max().date()

        date_range = st.sidebar.date_input(
            "Filter by Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )

        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]).replace(hour=23, minute=59, second=59)
        filtered_df = history_df[(history_df['timestamp'] >= start_date) & (history_df['timestamp'] <= end_date)].copy()

        if filtered_df.empty:
            st.warning("No data available for the selected filters.")
        else:
            col1, col2 = st.columns([0.4, 0.6])
            with col1:
                st.subheader("Prediction Distribution")
                outcome_counts = filtered_df['prediction'].value_counts().rename({0: 'Approved', 1: 'Denied'})
                fig = go.Figure(data=[go.Pie(labels=outcome_counts.index, values=outcome_counts.values, hole=.3)])
                fig.update_layout(title_text="Safe vs. Default Predictions")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("Application Volume Over Time")
                usage_by_day = filtered_df.set_index('timestamp').resample('D').size().rename("Number of Applications")
                st.bar_chart(usage_by_day)

            st.subheader("Default Probability Over Time")
            # Use .loc to safely modify the DataFrame copy
            filtered_df['default_probability'] = np.where(filtered_df['prediction'] == 1, filtered_df['probability'], 1 - filtered_df['probability'])
            prob_over_time = filtered_df.set_index('timestamp')[['default_probability']]
            st.line_chart(prob_over_time)

            st.divider()
            st.subheader("Download Filtered Data")
            report_data = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Report as CSV",
                data=report_data,
                file_name=f"creditpath_report_{date_range[0]}_to_{date_range[1]}.csv",
                mime="text/csv",
            )

st.sidebar.divider()
st.sidebar.header("UI Settings")

light_theme = {
    'primary': '#1c83e1',
    'base': 'light'
}
dark_theme = {
    'primary': '#83c9ff', 
    'base': 'dark'
}

theme_choice = st.sidebar.selectbox(
    "Choose a Theme",
    ["light", "dark"]
)

selected_theme = dark_theme if theme_choice == 'dark' else light_theme
set_theme(selected_theme)
