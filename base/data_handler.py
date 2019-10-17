# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 09:29:16 2019

@author: P900017
"""
    
import pandas as pd
import numpy as np
    
#Primer archivo, base de datos total con ratings S&P
df = pd.read_csv('data/data_em_1212_1218.csv', sep=',', index_col = ["Fecha", 'Ticker'], encoding = "latin1")
#print(df.describe())

#Segundo archivo, ratings de 3 calificadoras
df2 = pd.read_csv('data/database_allratings.csv')

tickers = list(df2.columns)[1:]
dates = list(df2['Ticker CIQ'])[1:]
df2 = df2.set_index('Ticker CIQ')

# Creando nuevo df para reemplazar valores inexistentes
dates2df = []
tickers2df = []
for i in range(len(tickers)):
    for j in range(len(dates)): 
        dates2df.append(dates[j])
        tickers2df.append(tickers[i])

newdf = pd.DataFrame(index=[tickers2df, dates2df], columns=["Rating"])

# Extrayendo ratings como "NaN" en archivo original y reemplazando por 
# ratings de nuevo archivo
print(df.IssuerRating.isna().sum())
for ticker in tickers:
    for date in dates:
        newdf.loc[(ticker, date), "Rating"] = df2.loc[date][ticker]
        try:
            if np.isnan(df.loc[(ticker, date), "IssuerRating"]):
                df.loc[(ticker, date), "IssuerRating"] = newdf.loc[(ticker, date), "Rating"]
        except TypeError:
            pass

# Ratings aún inexistentes
print("After missing ratings:")
print(df.IssuerRating.isna().sum())

# Archivo para training ML, hay que reemplazar aun el orden de las columnas y
# rellenar vacíos con "NaN"
df.to_csv('data/research_data_em_all.csv')
