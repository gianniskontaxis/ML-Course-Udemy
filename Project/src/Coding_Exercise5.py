# Coding Exercise 5: Feature Scaling for Machine Learning
#
# 1: Import the necessary libraries for data preprocessing,
#    including StandardScaler from sklearn.preprocessing and
#    train_test_split from sklearn.model_selection.
#
# 2: Load the "Wine Quality Red" dataset into a pandas DataFrame.
#    Use the pd.read_csv function and make sure to set the correct
#    delimiter for the file (semicolon ';').
#
# 3: Split the dataset into an 80-20 training-test set.
#    Set random_state to 42 to ensure reproducible results.
#
# 4: Create an instance of the StandardScaler class.
#
# 5: Fit the StandardScaler on the features from the training set,
#    excluding the target variable 'quality'.
#
# 6: Use the fit_transform method of the StandardScaler object
#    on the training dataset to scale the features.
#
# 7: Apply the transform method of the StandardScaler object
#    on the test dataset to scale the features.
#
# 8: Print the scaled training and test datasets to verify
#    the feature scaling process.


# Import necessary libraries
import numpy as np 
import pandas as pd 
import matplotlib as plt 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the Wine Quality Red dataset
df = pd.read_csv('winequality-red.csv', delimiter=';')

# Separate features and target
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Split the dataset into an 80-20 training-test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
print(X_train)
print(X_test)
print(y_train)
print(y_test)

# Create an instance of the StandardScaler class
sc = StandardScaler()

# Fit the StandardScaler on the features from the training set and transform it
X_train = sc.fit_transform(X_train)

# Apply the transform to the test set
X_test = sc.transform(X_test)

# Print the scaled training and test datasets
print(X_train)
print(X_test)
