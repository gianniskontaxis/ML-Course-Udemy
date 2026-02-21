# Coding Exercise 4: Dataset Splitting and Feature Scaling
#
# 1: Import necessary Python libraries:
#    - pandas for data manipulation
#    - train_test_split from sklearn.model_selection for splitting the dataset
#    - StandardScaler from sklearn.preprocessing for feature scaling
#
# 2: Load the Iris dataset using Pandas read_csv function.
#    The dataset name is 'iris.csv'.
#
# 3: Use train_test_split to split the dataset into an 80-20
#    training-test split.
#
# 4: Apply random_state with value 42 in the train_test_split
#    function to ensure reproducible results.
#
# 5: Print X_train, X_test, y_train, and y_test to understand
#    how the dataset has been split.
#
# 6: Use StandardScaler to apply feature scaling on the
#    training and test sets.
#
# 7: Print the scaled training and test sets to verify that
#    feature scaling has been applied correctly.



# Import necessary libraries
import pandas as pd 
import numpy as np 
import matplotlib as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing  import StandardScaler

# Load the Iris dataset
df = pd.read_csv('iris.csv')

# Separate features and target
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Split the dataset into an 80-20 training-test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=42)
print(X_train)
print(X_test)
print(y_train)
print(y_test)

# Apply feature scaling on the training and test sets

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Print the scaled training and test setsprint("X_train:\n", X_train)
print("X_test:\n", X_test)
print("y_train:\n", y_train)
print("y_test:\n", y_test)