# Data Notes

## Schema Insights
- Kaggle and Microsoft datasets share common fields like loan_amount, interest_rate, grade.
- Microsoft dataset has additional columns like LoanDuration, EmploymentType.
- Some fields differ in naming but represent the same concept.

## Missing Fields
- Kaggle dataset missing EmploymentType.
- Microsoft dataset missing purpose field.

## Inconsistencies
- Interest rate stored as percentage in Kaggle, as decimal in Microsoft.
- Default target labeled as {0,1} in Kaggle vs {"Charged Off", "Fully Paid"} in Microsoft.

## Assumptions
- Will standardize all target labels to 0/1 (default vs non-default).
- Interest rates converted to percentage format.
