# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data


# New Version
# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([("State",OneHotEncoder(),[3])], remainder = 'passthrough')
X = ct.fit_transform(X)

'''
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()
'''

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
print('MSE_Multiple_Regression: ', np.mean((y_test - y_pred)**2))


X_opt_train = np.append(arr = np.ones((X_train.shape[0],1)), values = X_train[:,3].reshape(-1,1), axis = 1)
reg_otimo = LinearRegression()
reg_otimo.fit(X_opt_train, y_train)
X_opt_test = np.append(arr = np.ones((X_test.shape[0],1)), values = X_test[:,3].reshape(-1,1), axis = 1)

y_pred_otimo = reg_otimo.predict(X_opt_test)
print('MSE_Optimum_Feature: ', np.mean((y_test - y_pred_otimo)**2))
print('Is the error by using the optimal feature less than the original one? ',
      np.mean((y_test - y_pred)**2) > np.mean((y_test - y_pred_otimo)**2) )

# Building the optimal model using Backward Elimination
# import statsmodels.formula.api as sm
import statsmodels.regression.linear_model as sm

#Your X_opt array has a dtype of object and this may be causing an error. Try changing it to float. For example you can use this:
X= np.append(arr = np.ones((50,1)).astype(int), values = X, axis =1)
X_opt = X[:,[0,1,2,3,4,5]]
X_opt = np.array(X_opt, dtype=float)

#X = np.append(arr = np.ones((50, 1)), values = X, axis = 1)
#X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print(regressor_OLS.summary())

X_opt = X[:, [0, 1, 3, 4, 5]]
X_opt = np.array(X_opt, dtype=float)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print(regressor_OLS.summary())

X_opt = X[:, [0, 3, 4, 5]]
X_opt = np.array(X_opt, dtype=float)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print(regressor_OLS.summary())

X_opt = X[:, [0, 3, 5]]
X_opt = np.array(X_opt, dtype=float)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print(regressor_OLS.summary())

X_opt = X[:, [0, 3]]
X_opt = np.array(X_opt, dtype=float)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print(regressor_OLS.summary())

