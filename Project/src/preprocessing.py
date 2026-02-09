import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#In supervised machine learning, your data is split into:
#X → the input features (what the model learns from)
#y → the target / label (what the model tries to predict)

dataset = pd.read_csv('../data/raw/Data.csv')   
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print(X)
#print(y)

#Taking care of missing data(replace the missing data with the average of the values of the columns)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')#we want to replace every nan value, every empty value with the mean(the average value of the column)

#Now we have to apply this imputer to the matrixs of features
imputer.fit(X[:, 1:3]) #Only for the columns 2-3, and why 1:3? because the uppern bound of a range in python is excluded so if we exclude 2 we will not have it at all
X[:, 1:3] = imputer.transform(X[:, 1:3])#This method returns the new replaced matrix, without the missing values

print(X)

#Encoding Categorical Data

#Encoding the the Independent Variable
#Turning this country column into 3 columns because we have 3 countries, France, Germany and Spain
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])]  , remainder='passthrough')#we wrote passthrogh because we want to keep the columns that wont be applied one hot encoding 
X = np.array(ct. fit_transform(X))

print(X)

#Encoding the the Dependent Variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
 
print(y)

#Splitting the datasest into the Training Set and the Testing Set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=1)

print(X_train)
print(X_test)
print(y_train)
print(y_test)


#Feature Scaling

#Standardisation
#xstand = x - mean(x) / standard deviation(x)

#Normalization
#xnorm = x - min(x) / max(x) - min(x)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])




#----Another case-----#

# ============================================================
# Coding Exercise 2: Handling Missing Data in a Dataset for
# Machine Learning
# ============================================================

# 1. Import the necessary Python libraries for data preprocessing,
#    including the SimpleImputer class from the scikit-learn library.

# 2. Load the dataset into a pandas DataFrame using the read_csv
#    function from the pandas library.
#    Dataset name: 'pima-indians-diabetes.csv'

# 3. Identify missing data in the dataset.
#    Print out the number of missing entries in each column.
#    Analyze its potential impact on machine learning model training.
#    This step is crucial because missing data can lead to inaccurate
#    and misleading results if not handled properly.

# 4. Implement a strategy for handling missing data.
#    In this exercise, the strategy is to replace missing values
#    with the mean value of the column, based on the nature of the dataset.
#    Other possible strategies include:
#      - Dropping rows with missing values
#      - Dropping entire columns with missing values
#      - Replacing missing values with the median or a constant value

# 5. Configure an instance of the SimpleImputer class to replace
#    missing values with the mean value of the column.

# 6. Apply the fit method of the SimpleImputer class on the numerical
#    columns of the matrix of features to learn the replacement values.

# 7. Use the transform method of the SimpleImputer class to replace
#    missing data in the specified numerical columns.

# 8. Update the matrix of features by assigning the result of the
#    transform method to the correct columns.

# 9. Print the updated matrix of features to verify the success of
#    the missing data replacement.




#----Solution-----#

# ==============================
# Importing the necessary libraries
# ==============================

# import pandas as pd
# import numpy as np 
# import matplotlib as plt 


# ==============================
# Load the dataset
# ==============================

# Read the CSV file and store it as a pandas DataFrame
# dataset = pd.read_csv('pima-indians-diabetes.csv')


# ==============================
# Separate features (X) and target variable (y)
# ==============================

# X contains all columns except the last one (input features)
# X = dataset.iloc[:, :-1].values

# y contains only the last column (output/label)
# y = dataset.iloc[:, -1].values


# ==============================
# Inspect the dataset
# ==============================

# Print the feature matrix to visually inspect the data
# print(X)

# Print the target vector
# print(y)


# ==============================
# Identify missing data
# ==============================

# Print the number of missing (NaN) values in each column
# This helps us understand which features contain incomplete data
# print(dataset.isnull().sum())


# ==============================
# Handle missing data using SimpleImputer
# ==============================

# Import SimpleImputer to replace missing values
# from sklearn.impute import SimpleImputer

# Create an imputer instance
# missing_values = np.NaN  → specifies what value is considered missing
# strategy = 'mean'        → replace missing values with the column mean
# imputer = SimpleImputer(missing_values=np.NaN, strategy='mean')


# ==============================
# Fit and transform the dataset
# ==============================

# Learn the mean value of each column (ignoring NaNs)
# imputer.fit(dataset)

# Replace missing values in the dataset with the learned means
# dataset = imputer.transform(dataset)


# ==============================
# Verify the result
# ==============================

# Print the updated dataset after imputing missing values
# print(dataset)



