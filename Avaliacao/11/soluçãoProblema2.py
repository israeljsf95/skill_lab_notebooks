# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 21:18:52 2021

@author: israe
"""

import pandas as pd
import numpy as np

def oneHotEncoder(x):
    unicos = np.unique(x)
    codificacao = np.zeros((x.shape[0], len(unicos)))
    for index in range(len(unicos)):
        codificacao[np.where(x == unicos[index]), index] = 1 
    return codificacao            

dataset = pd.read_csv('dados2.csv', sep = ';')

X = dataset.iloc[:,:-1].values

#Codificando as caracteristicas categoricas
a = oneHotEncoder(X[:,1])
b = oneHotEncoder(X[:,3])
X = np.delete(X, [1, 3], axis = 1)

#Estas linhas evitam a armadilha da dummy variable
X = np.insert(X, [1], a[:,0].reshape(-1,1), axis = 1)
X = np.insert(X,[2], b[:, :-1], axis = 1)

y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

print(X)
sc = StandardScaler()
X = sc.fit_transform(X)
print(X)


x_train, x_test, y_train, y_test = tts(X, y,
                                       test_size = 0.2, random_state = 0)
modelo_linear = LinearRegression()
modelo_linear.fit(x_train, y_train)
y_pred = modelo_linear.predict(x_test)
print("Erro Quadratico Medio: {0:.2f}".format(np.mean((y_pred-y_test)**2)))
print("R2: {}".format(modelo_linear.score(x_test, y_test)))













