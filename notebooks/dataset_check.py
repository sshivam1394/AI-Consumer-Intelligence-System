import pandas as pd

# Load dataset
df = pd.read_csv("../data/1429_1.csv")

# Dataset shape
print("Dataset Shape:")
print(df.shape)

# Column names
print("\nColumn Names:")
print(df.columns)

# First 5 rows
print("\nFirst 5 Rows:")
print(df.head())

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())