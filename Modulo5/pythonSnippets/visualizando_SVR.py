# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 20:26:04 2021

@author: israe
"""
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


#numero de pontos
N = 300

#gerando o dataset
r = np.sqrt(np.random.rand(1,N))
theta = np.random.rand(1,N)*2*np.pi
#alocando memoria
x = np.empty((2,N))
y = np.zeros((1,N))

#calculando os pontos
x[0,:] = r*np.sin(theta)
x[1,:] = r*np.cos(theta)

#visualizando os dados gerados
plt.scatter(x[0,:], x[1,:])
plt.show()
raio = np.sqrt(np.sum(x**2,0))
y = (raio < .5)*1

#visualizando as duas classes
fig = plt.figure()
plt.scatter(x[0, y==0], x[1, y==0], color = 'b') 
plt.scatter(x[0, y==1], x[1 ,y==1], color = 'r')
plt.show()
#aplicando o kernel
r = np.exp(-(x ** 2).sum(0)) #kernel gaussiano

#visualizando a transformacao ao aplicar uma funcao da base radial 
fig = plt.figure()
ax = fig.gca(projection = '3d')
ax.scatter(x[0, y==0], x[1, y==0], r[y==0], color = 'b') 
ax.scatter(x[0, y==1], x[1 ,y==1], r[y==1], color = 'r')
ax.view_init(elev = 19, azim = -71)
plt.show()
#import o regressor
from sklearn.svm import SVR

svr_rbf = SVR(kernel='rbf', C = 100, gamma = 0.1, epsilon = .1)
svr_rbf.fit(x.T, y)

xlim = ax.get_xlim()
ylim = ax.get_ylim()
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
yy, xx = np.meshgrid(yy, xx)
xy = np.vstack([xx.ravel(), yy.ravel()]).T
avaliando_no_grid = svr_rbf.predict(xy)


fig = plt.figure()
ax = fig.gca(projection = '3d')
ax.scatter(x[0, y==0], x[1, y==0], color = 'b') 
ax.scatter(x[0, y==1], x[1 ,y==1], color = 'r')

ax.plot_surface(xx, yy, avaliando_no_grid.reshape(xx.shape), alpha = 0.5)
plt.show()






