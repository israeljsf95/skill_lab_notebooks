# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd 
import numpy as np
import sqlite3 as sq
import matplotlib.pyplot as plt

dataset = pd.read_csv('dado1.csv', sep =";")
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 1].values


from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression

x_train, x_test, y_train, y_test = tts(x, y, 
                                       test_size = 1/3, random_state = 0)

modelo_linear = LinearRegression()
modelo_linear.fit(x_train, y_train)
y_pred = modelo_linear.predict(x_test)
print("Erro Quadratico Medio: {0:.2f}".format(np.mean((y_pred-y_test)**2)))
print("R2: {}".format(modelo_linear.score(x_test, y_test)))
x_min = np.min(x)
x_max = np.max(x)
xx = np.linspace(x_min, x_max, 100)
yy = modelo_linear.predict(xx.reshape(-1, 1))


eixos = dataset.columns
plt.plot(x, y, label = 'data')
plt.plot(xx, yy, label = 'modelo')
plt.xlabel(eixos[0].upper())
plt.ylabel(eixos[1].upper())
plt.legend(loc = "lower right")
plt.show()
























