#  Loan Default Prediction Dataset Overview

## Source
The dataset is sourced from Kaggle:  
[Loan Default Dataset](https://www.kaggle.com/datasets/nikhil1e9/loan-default?resource=download)

---

## Dataset Information
- **Total Rows:** 255,347  
- **Total Columns:** 18  
- **File Size (approx):** 35 MB  
- **Target Variable:** `Default`  
  - `1` → Loan Defaulted  
  - `0` → Loan Not Defaulted  

---

## Dataset Schema

| Column Name      | Data Type | Description |
|------------------|-----------|-------------|
| **LoanID**       | object    | Unique identifier for each loan |
| **Age**          | int64     | Age of borrower (in years) |
| **Income**       | int64     | Annual income of borrower |
| **LoanAmount**   | int64     | Total loan amount applied for |
| **CreditScore**  | int64     | Borrower's credit score |
| **MonthsEmployed** | int64   | Number of months the borrower has been employed |
| **NumCreditLines** | int64   | Number of active credit lines borrower holds |
| **InterestRate** | float64   | Interest rate (%) applied on the loan |
| **LoanTerm**     | int64     | Duration of loan (in months) |
| **DTIRatio**     | float64   | Debt-to-Income ratio |
| **Education**    | object    | Education level (e.g., High School, Bachelor's, Master's) |
| **EmploymentType** | object  | Type of employment (e.g., Full-time, Unemployed) |
| **MaritalStatus** | object   | Marital status of borrower (Married, Divorced, etc.) |
| **HasMortgage**  | object    | Whether borrower has an existing mortgage (Yes/No) |
| **HasDependents** | object   | Whether borrower has dependents (Yes/No) |
| **LoanPurpose**  | object    | Purpose of loan (Auto, Business, Other, etc.) |
| **HasCoSigner**  | object    | Whether loan has a co-signer (Yes/No) |
| **Default**      | int64     | Loan default status (1 = Defaulted, 0 = Not Defaulted) |

---

## Sample Data

| LoanID     | Age | Income | LoanAmount | CreditScore | MonthsEmployed | NumCreditLines | InterestRate | LoanTerm | DTIRatio | Education   | EmploymentType | MaritalStatus | HasMortgage | HasDependents | LoanPurpose | HasCoSigner | Default |
|------------|-----|--------|------------|-------------|----------------|----------------|--------------|----------|----------|-------------|----------------|---------------|-------------|---------------|-------------|-------------|---------|
| I38PQUQS96 | 56  | 85994  | 50587      | 520         | 80             | 4              | 15.23        | 36       | 0.44     | Bachelor's  | Full-time      | Divorced      | Yes         | Yes           | Other       | Yes         | 0       |
| HPSK72WA7R | 69  | 50432  | 124440     | 458         | 15             | 1              | 4.81         | 60       | 0.68     | Master's    | Full-time      | Married       | No          | No            | Other       | Yes         | 0       |
| C1OZ6DPJ8Y | 46  | 84208  | 129188     | 451         | 26             | 3              | 21.17        | 24       | 0.31     | Master's    | Unemployed     | Divorced      | Yes         | Yes           | Auto        | No          | 1       |

---

## Notes
- **Target variable:** `Default` is binary (0 = Not Defaulted, 1 = Defaulted).  
- **Categorical columns:** (`Education`, `EmploymentType`, `MaritalStatus`, etc.) → Require encoding before modeling.  
- **Numerical columns:** (`Income`, `LoanAmount`, `CreditScore`, `InterestRate`, `DTIRatio`) → May need scaling/normalization depending on the ML algorithm.  
---
