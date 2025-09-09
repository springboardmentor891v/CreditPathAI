dataset_overview = """
# Loan Default Prediction Dataset Overview

## Source
The dataset is sourced from [Loan Default Dataset (Kaggle)](https://www.kaggle.com/datasets/yasserh/loan-default-dataset/data).

---

## Dataset Information
- **Total Rows:** 148,670
- **Total Columns:** 34
- **File Size (approx):** ~45 MB
- **Target Variable:** `Status`
  - `1` → Loan Defaulted
  - `0` → Loan Not Defaulted

---

## Dataset Schema

| Column Name              | Data Type | Description |
|---------------------------|-----------|-------------|
| **ID**                   | int64     | Unique identifier for each loan |
| **year**                 | int64     | Year of loan application |
| **loan_limit**           | object    | Loan limit category (e.g., cf, ncf) |
| **Gender**               | object    | Gender of applicant |
| **approv_in_adv**        | object    | Whether approval was obtained in advance |
| **loan_type**            | object    | Type of loan applied |
| **loan_purpose**         | object    | Purpose of the loan |
| **Credit_Worthiness**    | object    | Creditworthiness rating of the borrower |
| **open_credit**          | object    | Open credit information |
| **business_or_commercial** | object  | Whether loan is for business/commercial purpose |
| **loan_amount**          | int64     | Total loan amount requested |
| **rate_of_interest**     | float64   | Rate of interest (%) applied |
| **Interest_rate_spread** | float64   | Spread over base interest rate |
| **Upfront_charges**      | float64   | Upfront charges for the loan |
| **term**                 | float64   | Loan term (in months/years) |
| **Neg_ammortization**    | object    | Negative amortization status |
| **interest_only**        | object    | Whether interest-only payments allowed |
| **lump_sum_payment**     | object    | Whether lump sum payment allowed |
| **property_value**       | float64   | Value of property |
| **construction_type**    | object    | Type of property construction |
| **occupancy_type**       | object    | Occupancy type of property |
| **Secured_by**           | object    | What the loan is secured by (collateral) |
| **total_units**          | object    | Number of total housing units |
| **income**               | float64   | Borrower's income |
| **credit_type**          | object    | Credit bureau type (EXP, EQUI, CRIF, etc.) |
| **Credit_Score**         | int64     | Borrower's credit score |
| **co-applicant_credit_type** | object | Co-applicant’s credit type |
| **age**                  | object    | Age group of borrower (e.g., 25–34, 45–54) |
| **submission_of_application** | object | How loan application was submitted (to institution/not) |
| **LTV**                  | float64   | Loan-to-Value ratio |
| **Region**               | object    | Geographical region of applicant |
| **Security_Type**        | object    | Type of loan security |
| **Status**               | int64     | Loan status (1 = Defaulted, 0 = Not Defaulted) |
| **dtir1**                | float64   | Debt-to-Income ratio |

---

## Sample Data

| ID    | year | loan_limit | Gender | approv_in_adv | loan_type | loan_purpose | Credit_Worthiness | open_credit | business_or_commercial | credit_type | Credit_Score | co-applicant_credit_type | age   | submission_of_application | LTV      | Region | Security_Type | Status | dtir1 |
|-------|------|------------|--------|---------------|-----------|--------------|-------------------|-------------|------------------------|-------------|--------------|--------------------------|-------|---------------------------|----------|--------|---------------|--------|-------|
| 24890 | 2019 | cf         | Sex Not Available | nopre | type1 | p1 | l1 | nopc | nob/c | EXP  | 758 | CIB | 25-34 | to_inst   | 98.72 | south | direct | 1 | 45.0 |
| 24891 | 2019 | cf         | Male   | nopre         | type2     | p1 | l1 | nopc | b/c   | EQUI | 552 | EXP | 55-64 | to_inst   | NaN   | North | direct | 1 | NaN |
| 24892 | 2019 | cf         | Male   | pre           | type1     | p1 | l1 | nopc | nob/c | EXP  | 834 | CIB | 35-44 | to_inst   | 80.02 | south | direct | 0 | 46.0 |
| 24893 | 2019 | cf         | Male   | nopre         | type1     | p4 | l1 | nopc | nob/c | EXP  | 587 | CIB | 45-54 | not_inst  | 69.38 | North | direct | 0 | 42.0 |


---

## Notes
- **Target variable:** `Status` is binary (0 = Not Defaulted, 1 = Defaulted).  
- **Categorical columns:** (`loan_limit`, `Gender`, `loan_type`, `loan_purpose`, `Credit_Worthiness`, `Region`, etc.) → Require encoding before modeling.  
- **Numerical columns:** (`Credit_Score`, `LTV`, `dtir1`) → May need scaling/normalization depending on ML algorithms.  
- **Missing values:** Some columns (e.g., `LTV`, `dtir1`) contain `NaN` values that require handling.
