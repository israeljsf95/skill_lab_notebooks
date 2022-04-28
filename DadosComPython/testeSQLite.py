# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 19:26:13 2021

@author: israe
"""


import sqlite3 as sq3


con = sq3.connect('movies.db')    

print(con.execute("Select * FROM sqlite_master").fetchall())
print('\n')
print(con.execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER by name").fetchall())
print('\n')
con.close()



import pandas as pd
import json

with open("some_movies.json") as f:
    data = json.load(f)

print(data)
print('\n')
df_data = pd.json_normalize(data, sep = '_')
print(df_data.head())


movies = df_data[["id","title","revenue","budget","belongs_to_collection_name","release_date"]].copy()
print(movies.head())


movies.release_date = pd.to_datetime(df_data.release_date)
movies.revenue = df_data.revenue/1e6
movies.budget = df_data.budget/1e6
print(movies.info())

movies_test = movies.copy()
movies_test = movies_test.set_index('id')
print(movies_test.head())
print('\n')
movies_test = movies_test.sort_values(by = 'id')
print(movies_test.head())

