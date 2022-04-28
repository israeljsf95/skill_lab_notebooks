# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 19:19:15 2021

@author: israe
"""

import pandas as pd

# Uncomment this line if using this notebook locally
# insurance = pd.read_csv('./data/insurance/insurance.csv') 

file_name = "https://raw.githubusercontent.com/rajeevratan84/datascienceforbusiness/master/insurance.csv"
insurance = pd.read_csv(file_name)

# Preview our data
print(insurance.head())
print(insurance.info())
print(insurance.describe())


print("Number of Rows of Insurance: ", insurance.shape[0])
print("Number of Columns of Insurance: ", insurance.shape[1])
print("Features: ", insurance.columns.tolist())
print("Missing Values: ", insurance.isnull().sum().values.sum())
print("Unique Values:\n", insurance.nunique())



