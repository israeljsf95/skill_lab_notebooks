# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd

dataset = pd.read_csv('Data.csv')

X = dataset.iloc[:,0:-1].values
Y = dataset.iloc[:,-1].values

from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values = "Nan", strategy = 'mean', axis = 0)
imputer = imputer.fit(X)