import os
import pandas as pd
import pickle

# --- Load trained model safely ---
base_dir = os.path.dirname(os.path.abspath(__file__))  # folder of this script
model_path = os.path.join(base_dir, "xgb_model.pkl")   # ensure model is in the same folder
model = pickle.load(open(model_path, "rb"))

# --- Mapping dictionaries for categorical features ---
gender_map = {"Male": 0, "Female": 1, "Joint": 2, "Not Available": 3}
loan_limit_map = {"Not Limited": 0, "Limited": 1}
approv_in_adv_map = {"No": 0, "Yes": 1}
loan_type_map = {"Type 1": 0, "Type 2": 1, "Type 3": 2}
loan_purpose_map = {"Home Purchase": 0, "Refinance": 1, "Improvement": 2, "Other": 3}
credit_worthiness_map = {"Good": 0, "Bad": 1}
open_credit_map = {"No": 0, "Yes": 1}
business_or_commercial_map = {"No": 0, "Yes": 1}
neg_amort_map = {"No": 0, "Yes": 1}
interest_only_map = {"No": 0, "Yes": 1}
lump_sum_map = {"No": 0, "Yes": 1}
construction_type_map = {"Type 1": 0, "Type 2": 1}
occupancy_type_map = {"Owner": 0, "Co-Owner": 1, "Tenant": 2}
secured_by_map = {"Home": 0, "Other": 1}
total_units_map = {"1 Unit": 0, "2 Units": 1, "3 Units": 2, "4 Units": 3}
credit_type_map = {"Conventional": 0, "FHA": 1, "VA": 2, "Other": 3}
co_applicant_credit_map = {"No": 0, "Yes": 1}
age_map = {"<25": 0, "25-34": 1, "35-44": 2, "45-54": 3, "55-64": 4, "65-74": 5, "75+": 6}
submission_map = {"Not Submitted": 0, "Submitted": 1}
region_map = {"North": 0, "South": 1, "East": 2, "West": 3}
security_type_map = {"Type 1": 0, "Type 2": 1}

# --- Collect user inputs ---
print("Enter Applicant Details:")

loan_limit = loan_limit_map[input("Loan Limit (Not Limited/Limited): ")]
gender = gender_map[input("Gender (Male/Female/Joint/Not Available): ")]
approv_in_adv = approv_in_adv_map[input("Approval in Advance (Yes/No): ")]
loan_type = loan_type_map[input("Loan Type (Type 1/2/3): ")]
loan_purpose = loan_purpose_map[input("Loan Purpose (Home Purchase/Refinance/Improvement/Other): ")]
credit_worthiness = credit_worthiness_map[input("Credit Worthiness (Good/Bad): ")]
open_credit = open_credit_map[input("Open Credit (Yes/No): ")]
business_or_commercial = business_or_commercial_map[input("Business/Commercial (Yes/No): ")]

loan_amount = float(input("Loan Amount (₹): "))
rate_of_interest = float(input("Rate of Interest (%): "))
interest_rate_spread = float(input("Interest Rate Spread: "))
upfront_charges = float(input("Upfront Charges: "))
term = int(input("Loan Term (months): "))

neg_amort = neg_amort_map[input("Negative Amortization (Yes/No): ")]
interest_only = interest_only_map[input("Interest Only (Yes/No): ")]
lump_sum_payment = lump_sum_map[input("Lump Sum Payment (Yes/No): ")]

property_value = float(input("Property Value (₹): "))
construction_type = construction_type_map[input("Construction Type (Type 1/2): ")]
occupancy_type = occupancy_type_map[input("Occupancy Type (Owner/Co-Owner/Tenant): ")]
secured_by = secured_by_map[input("Secured By (Home/Other): ")]
total_units = total_units_map[input("Total Units (1 Unit/2 Units/3 Units/4 Units): ")]

income = float(input("Applicant Income (₹): "))
credit_type = credit_type_map[input("Credit Type (Conventional/FHA/VA/Other): ")]
credit_score = int(input("Credit Score (300-900): "))
co_applicant_credit_type = co_applicant_credit_map[input("Co-Applicant Credit Type (Yes/No): ")]
age = age_map[input("Age Group (<25/25-34/35-44/45-54/55-64/65-74/75+): ")]
submission_of_application = submission_map[input("Submission of Application (Submitted/Not Submitted): ")]

ltv = float(input("Loan-to-Value Ratio (LTV): "))
region = region_map[input("Region (North/South/East/West): ")]
security_type = security_type_map[input("Security Type (Type 1/2): ")]
dtir1 = float(input("DTI Ratio (%): "))

# --- Prepare input DataFrame ---
input_data = pd.DataFrame([[  
    loan_limit, gender, approv_in_adv, loan_type, loan_purpose,
    credit_worthiness, open_credit, business_or_commercial,
    loan_amount, rate_of_interest, interest_rate_spread, upfront_charges, term,
    neg_amort, interest_only, lump_sum_payment,
    property_value, construction_type, occupancy_type, secured_by, total_units,
    income, credit_type, credit_score, co_applicant_credit_type, age, submission_of_application,
    ltv, region, security_type, dtir1
]], columns=[
    'loan_limit','Gender','approv_in_adv','loan_type','loan_purpose',
    'Credit_Worthiness','open_credit','business_or_commercial',
    'loan_amount','rate_of_interest','Interest_rate_spread','Upfront_charges','term',
    'Neg_ammortization','interest_only','lump_sum_payment',
    'property_value','construction_type','occupancy_type','Secured_by','total_units',
    'income','credit_type','Credit_Score','co-applicant_credit_type','age','submission_of_application',
    'LTV','Region','Security_Type','dtir1'
])

# --- Prediction ---
prediction = model.predict(input_data)[0]
if prediction == 0:
    print("\n✅ Prediction: Applicant is Non-Defaulter")
else:
    print("\n❌ Prediction: Applicant is Defaulter")
