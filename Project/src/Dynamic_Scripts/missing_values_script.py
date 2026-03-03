import pandas as pd

#Load Dataset
dataset = pd.read_csv("../../data/raw/50_Startups.csv")

print("Dataset Shape:", dataset.shape)
print("\nColumn Data Types:\n", dataset.dtypes)


#Check Missing Values
missing_count = dataset.isnull().sum()  # Number of missing per column
missing_percent = (missing_count / len(dataset)) * 100  # Percent missing


#Combine into a Table
missing_table = pd.DataFrame({
    'Missing Values': missing_count,
    'Percentage': missing_percent
})

# Only keep columns with missing values
missing_table = missing_table[missing_table['Missing Values'] > 0]

#Print Results
if missing_table.empty:
    print("\nNo missing values found in the dataset!")
else:
    print("\nColumns with Missing Values:")
    print(missing_table)


#Optional: Print Rows with Missing Values
if not missing_table.empty:
    print("\nExample rows with missing values:")
    print(dataset[dataset.isnull().any(axis=1)].head())