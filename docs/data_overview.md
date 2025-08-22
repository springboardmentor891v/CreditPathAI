import pandas as pd
import os

# Load your dataset (update path if needed)
df = pd.read_csv("Loan_default.csv")

# Create docs folder if not exists
os.makedirs("docs", exist_ok=True)

# Prepare dataset documentation content
data_doc = f"""
# Dataset Overview

- **File Name**: Loan_default.csv
- **Number of Rows**: {df.shape[0]}
- **Number of Columns**: {df.shape[1]}

## Column Information
{df.dtypes.to_string()}

## Missing Values
{df.isna().sum().to_string()}

## Sample Data (first 5 rows)
{df.head().to_string()}

## Basic Statistics
{df.describe().to_string()}

## Notes
- Source: Local dataset
- Inspected with pandas `.head()`, `.info()`, `.describe()`
- Missing value summary included
"""

# Save into docs/data_overview.md
with open("docs/data_overview.md", "w", encoding="utf-8") as f:
    f.write(data_doc)

print("✅ Dataset documentation saved as docs/data_overview.md")
