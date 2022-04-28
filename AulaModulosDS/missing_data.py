# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
print(X)
y = dataset.iloc[:, 3].values
print(y)
print(X[:,0])


# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.NAN, strategy= 'mean')
imputer.fit(X[:, 1:3])
X[:,1:3] = imputer.transform(X[:, 1:3])
print(X)