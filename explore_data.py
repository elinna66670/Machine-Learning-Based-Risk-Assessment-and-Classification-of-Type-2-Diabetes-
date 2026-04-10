import pandas as pd

df = pd.read_csv(r"d:\study\BMI 5109\diabetes_012_health_indicators_BRFSS2015.csv")

print("=== Shape ===")
print(df.shape)
print()

print("=== Columns & Dtypes ===")
print(df.dtypes)
print()

print("=== Missing Values ===")
print(df.isnull().sum())
print()

print("=== Target Distribution (Diabetes_012) ===")
print(df["Diabetes_012"].value_counts().sort_index())
print()

print("=== Descriptive Statistics ===")
pd.set_option("display.max_columns", 25)
pd.set_option("display.width", 200)
print(df.describe().T)
print()

print("=== First 5 Rows ===")
print(df.head())
