# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 21:31:51 2021

@author: israe
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('DatasetHorasEstudoNota.csv', sep=";")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/4, random_state = 0)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, y)

X_test = np.array([6, 9, 12, 15, 16, 4])
y_test = regressor.predict(X_test.reshape(-1,1))
print("Notas Alunos: {0}".format(y_test))



import matplotlib.pyplot as plt

plt.scatter(X_train, y_train, color = "red")
plt.plot(X_train, regressor.predict(X_train), color = "blue")


