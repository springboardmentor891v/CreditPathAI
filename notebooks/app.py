import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# -------------------------
# Available models
# -------------------------
models = {
    "Logistic Regression": "Logistic_Regression.pkl",
    "Random Forest": "Random_Forest.pkl",
    "XGBoost": "XGBoost.pkl",
    "Gradient Boosting": "Gradient_Boosting.pkl",
    "K-Nearest Neighbors (KNN)": "K-Nearest_Neighbors_KNN.pkl",
    "AdaBoost": "AdaBoost.pkl",
    "Gaussian Naive Bayes": "Gaussian_Naive_Bayes.pkl",
    "LightGBM": "LightGBM.pkl"
}

# -------------------------
# Page Config & Custom CSS
# -------------------------
st.set_page_config(page_title="Loan Default Prediction", layout="wide")

st.markdown("""
    <style>
    /* Move main container slightly up */
    .block-container {
        padding-top: 2rem;
    }

    /* Style Predict Button */
    .stButton>button {
        background-color: #4B0082;
        color: white;
        font-size:16px;
        font-weight:bold;
        border-radius:12px;
        padding:10px 20px;
        transition:0.3s;
    }
    .stButton>button:hover {
        background-color: #6A0DAD;
        color: #f2f2f2;
        transform: scale(1.05);
    }

    /* Info Box Styling */
    .info-box {
        background-color: #f9f5ff;
        border-radius: 15px;
        padding: 25px;
        margin: 40px auto;
        width: 80%;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.2);
        font-size: 16px;
        line-height: 1.6;
        color: #333;
    }

    /* Sidebar headings */
    .sidebar-title {
        font-size: 19px !important;
        font-weight: bold !important;
        color: #6A0DAD !important;
        margin-bottom: 8px !important;
        border-bottom: 2px solid #ddd;
        padding-bottom: 4px;
    }

    /* Section headings */
    .section-heading {
        text-align: center;
        font-size: 22px;
        color: #4B0082;
        margin-bottom: 12px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------
# App Heading
# -------------------------
st.markdown(
    """
    <h1 style='text-align: center; color: #4B0082; margin-bottom:0;'>Loan Default Prediction</h1>
    <p style='text-align: center; color: #555; margin-top:0;'>An AI-powered app to assess loan risks with multiple ML models</p>
    """, unsafe_allow_html=True
)

# -------------------------
# Sidebar with styled titles
# -------------------------
st.sidebar.markdown("<p class='sidebar-title'>Choose Model</p>", unsafe_allow_html=True)
model_choice = st.sidebar.selectbox("Select ML Model", list(models.keys()))

st.sidebar.markdown("<p class='sidebar-title'>Enter Loan Details</p>", unsafe_allow_html=True)

# Integer inputs
loan_amount = st.sidebar.number_input("Loan Amount", 1000, 10000000, 100000, step=1000)
term = st.sidebar.number_input("Term", 1, 1000, 360, step=1)
submission_of_application_to_inst = st.sidebar.number_input("Submission of Application", 0, 365, 10, step=1)
loan_limit_ncf = st.sidebar.number_input("Loan Limit NCF", 0, 10000000, 50000, step=1000)

# Float inputs
rate_of_interest = st.sidebar.number_input("Rate of Interest", 0.0, 20.0, 4.0, step=0.1)
interest_rate_spread = st.sidebar.number_input("Interest Rate Spread", 0.0, 10.0, 0.5, step=0.1)
upfront_charges = st.sidebar.number_input("Upfront Charges", 0.0, 100000.0, 1000.0, step=10.0)
property_value = st.sidebar.number_input("Property Value", 1000.0, 10000000.0, 300000.0, step=1000.0)
income = st.sidebar.number_input("Income", 0.0, 10000000.0, 50000.0, step=1000.0)
credit_score = st.sidebar.number_input("Credit Score", 0.0, 1000.0, 700.0, step=1.0)
ltv = st.sidebar.number_input("Loan to Value (LTV)", 0.0, 200.0, 80.0, step=0.1)
dtir1 = st.sidebar.number_input("Debt-to-Income Ratio", 0.0, 100.0, 30.0, step=0.1)

# -------------------------
# Binary Inputs
# -------------------------
has_co_applicant = st.sidebar.selectbox("Has Co-Applicant", [0, 1])
approv_in_adv_pre = st.sidebar.selectbox("Approved in Advance", [0, 1])
credit_worthiness_l2 = st.sidebar.selectbox("Credit Worthiness L2", [0, 1])
business_or_commercial_nob_c = st.sidebar.selectbox("Business or Commercial", [0, 1])
neg_ammortization_not_neg = st.sidebar.selectbox("Negative Amortization Not Negative", [0, 1])
interest_only_not_int = st.sidebar.selectbox("Interest Only Not Interest", [0, 1])
lump_sum_payment_not_lpsm = st.sidebar.selectbox("Lump Sum Payment Not LPSM", [0, 1])

# -------------------------
# Categorical Inputs (one-hot)
# -------------------------
# Gender
gender = st.sidebar.selectbox("Gender", ["Female", "Joint", "Male", "Not Available"])
gender_Female = 1 if gender == "Female" else 0
gender_Joint = 1 if gender == "Joint" else 0
gender_Male = 1 if gender == "Male" else 0
gender_Sex_Not_Available = 1 if gender == "Not Available" else 0


# Loan Type
loan_type = st.sidebar.selectbox("Loan Type", ["Type1", "Type2", "Type3"])
loan_type_type2 = 1 if loan_type == "Type2" else 0
loan_type_type3 = 1 if loan_type == "Type3" else 0

# Loan Purpose
loan_purpose = st.sidebar.selectbox("Loan Purpose", ["Purpose1", "Purpose2", "Purpose3", "Purpose4"])
loan_purpose_p2 = 1 if loan_purpose == "Purpose2" else 0
loan_purpose_p3 = 1 if loan_purpose == "Purpose3" else 0
loan_purpose_p4 = 1 if loan_purpose == "Purpose4" else 0

# Occupancy Type
occupancy_type = st.sidebar.selectbox("Occupancy Type", ["PR", "SR"])
occupancy_type_pr = 1 if occupancy_type == "PR" else 0
occupancy_type_sr = 1 if occupancy_type == "SR" else 0

# Total Units
total_units = st.sidebar.selectbox("Total Units", ["1U", "2U", "3U", "4U"])
total_units_2U = 1 if total_units == "2U" else 0
total_units_3U = 1 if total_units == "3U" else 0
total_units_4U = 1 if total_units == "4U" else 0

# Credit Type
credit_type = st.sidebar.selectbox("Credit Type", ["CRIF", "EQUI", "EXP"])
credit_type_CRIF = 1 if credit_type == "CRIF" else 0
credit_type_EQUI = 1 if credit_type == "EQUI" else 0
credit_type_EXP = 1 if credit_type == "EXP" else 0

# Age Group
age_group = st.sidebar.selectbox("Age Group", ["<25", "35-44", "45-54", "55-64", "65-74", ">74"])
age__25 = 1 if age_group == "<25" else 0
age_35_44 = 1 if age_group == "35-44" else 0
age_45_54 = 1 if age_group == "45-54" else 0
age_55_64 = 1 if age_group == "55-64" else 0
age_65_74 = 1 if age_group == "65-74" else 0
age__74 = 1 if age_group == ">74" else 0

# Region
region = st.sidebar.selectbox("Region", ["North East", "Central", "South"])
region_North_East = 1 if region == "North East" else 0
region_central = 1 if region == "Central" else 0
region_south = 1 if region == "South" else 0

# -------------------------
# Create DataFrame
# -------------------------
input_data = pd.DataFrame([{
    'loan_amount': loan_amount,
    'rate_of_interest': rate_of_interest,
    'term': term,
    'property_value': property_value,
    'income': income,
    'credit_score': credit_score,
    'ltv': ltv,
    'dtir1': dtir1,
    'has_co_applicant': has_co_applicant,
    'loan_limit_ncf': loan_limit_ncf,
    'gender_Joint': gender_Joint,
    'gender_Male': gender_Male,
    'gender_Sex_Not_Available': gender_Sex_Not_Available,
    'approv_in_adv_pre': approv_in_adv_pre,
    'loan_type_type2': loan_type_type2,
    'loan_type_type3': loan_type_type3,
    'loan_purpose_p2': loan_purpose_p2,
    'loan_purpose_p3': loan_purpose_p3,
    'loan_purpose_p4': loan_purpose_p4,
    'credit_worthiness_l2': credit_worthiness_l2,
    'business_or_commercial_nob_c': business_or_commercial_nob_c,
    'neg_ammortization_not_neg': neg_ammortization_not_neg,
    'interest_only_not_int': interest_only_not_int,
    'lump_sum_payment_not_lpsm': lump_sum_payment_not_lpsm,
    'occupancy_type_pr': occupancy_type_pr,
    'occupancy_type_sr': occupancy_type_sr,
    'total_units_2U': total_units_2U,
    'total_units_3U': total_units_3U,
    'total_units_4U': total_units_4U,
    'credit_type_CRIF': credit_type_CRIF,
    'credit_type_EQUI': credit_type_EQUI,
    'credit_type_EXP': credit_type_EXP,
    'age_35_44': age_35_44,
    'age_45_54': age_45_54,
    'age_55_64': age_55_64,
    'age_65_74': age_65_74,
    'age__25': age__25,
    'age__74': age__74,
    'submission_of_application_to_inst': submission_of_application_to_inst,
    'region_North_East': region_North_East,
    'region_central': region_central,
    'region_south': region_south
}])

# -------------------------
# Middle Info Section
# -------------------------
st.markdown("""
    <div class="info-box">
        <h3 style="color:#6A0DAD; text-align:center;">Project Overview</h3>
        <p>
        This system leverages multiple ML models like Logistic Regression, Random Forest, XGBoost, and LightGBM 
        to predict loan defaults based on borrower details such as income, loan amount, credit score, and more.  
        <br><br>
        <b style="color:#4B0082;">Goal:</b> Help banks and financial institutions make smarter, data-driven lending decisions.
        </p>
    </div>
""", unsafe_allow_html=True)

# -------------------------
# Preprocessing (Scaling)
# -------------------------
numeric_features = ['loan_amount', 'rate_of_interest', 'term', 'property_value',
                    'income', 'credit_score', 'ltv', 'dtir1', 'loan_limit_ncf',
                    'submission_of_application_to_inst']
scaler = StandardScaler()
input_data[numeric_features] = scaler.fit_transform(input_data[numeric_features])

# -------------------------
# Predict Button
# -------------------------
if st.sidebar.button("Predict"):
    with open(models[model_choice], "rb") as f:
        model = pickle.load(f)

    prediction = model.predict(input_data)[0]

    st.markdown("<h3 class='section-heading'>Prediction Results</h3>", unsafe_allow_html=True)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_data)[0]
        st.write(f"**Probability of NOT Default:** {proba[0]*100:.2f}%")
        st.write(f"**Probability of Default:** {proba[1]*100:.2f}%")

    if prediction == 1:
        st.error("Loan will DEFAULT")
    else:
        st.success("Loan will NOT Default")
