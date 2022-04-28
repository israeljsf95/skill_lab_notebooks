# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 11:38:53 2021

@author: israe
"""

def var_to_z(x, mean, sigma):
    return (x-mean)/sigma

def z_to_var(z, mean, sigma):
    return (z*sigma + mean)

from scipy import stats


mu, sigma = 30, 6
x = 31
z = var_to_z(x, mu, sigma)

p = stats.norm.cdf(z)
pp = stats.norm.ppf(p)
x2 = z_to_var(pp, mu, sigma)
print("x: {0}\nmu:{1}\nsigma:{2}".format(x, mu, sigma))
print("z: %f.2", z)
print("pz: %f.2", p)
print("X2: %.f2", x2)







