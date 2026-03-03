import pandas as pd


#Load Dataset
dataset = pd.read_csv("../../data/raw/50_Startups.csv")

print("Dataset Shape:", dataset.shape)
print("\nColumn Data Types:\n", dataset.dtypes)

#Detect Categorical Features
categorical_cols = dataset.select_dtypes(include=['object', 'category']).columns

print("\nCategorical Columns Detected:")
print(list(categorical_cols))
print("Number of Categorical Features:", len(categorical_cols))


#Cardinality Check
print("\nCardinality of Categorical Features:")
for col in categorical_cols:
    unique_count = dataset[col].nunique()
    print(f"{col}: {unique_count} unique categories")


# Print Actual Unique Values of Categorical Features
print("\nUnique Values in Categorical Features:")
for col in categorical_cols:
    unique_values = dataset[col].unique()
    print(f"{col} ({dataset[col].nunique()} unique): {unique_values}")


#Detect Possible Mis-Typed Categorical (Numeric but Low Unique Values)
print("\nChecking for Potential Numeric-Categorical Columns:")
for col in dataset.select_dtypes(include=['int64', 'float64']).columns:
    if dataset[col].nunique() < 10:
        print(f"{col} might be categorical (only {dataset[col].nunique()} unique values)")


#High Cardinality Warning
print("\nHigh Cardinality Warning (more than 15 unique values):")
for col in categorical_cols:
    if dataset[col].nunique() > 15:
        print(f"{col} has high cardinality!")


#Quick Summary
print("\nCategorical Summary:")
print(dataset.describe(include=['object']))