# Data Manipulation and Analysis
import pandas as pd
import numpy as np

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('/content/drive/MyDrive/Loan_Default.csv')

# Display the first few rows
print("First 5 rows of the dataset:")
print(df.head())

# Get a concise summary of the dataframe
print("\nDataset Information:")
df.info()

# Get a concise summary of the dataframe
print("\nDataset Information:")
df.info()

# Check for missing values in each column
print("\nMissing Values per Column:")
print(df.isnull().sum())
