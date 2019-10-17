# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 09:54:30 2019

@author: P900017
"""
import pandas as pd
import numpy as np
from rating_functions import model_training, feat_elim
#import matplotlib.pyplot as plt


## Cargando archivosy bases de datos necesarias
# Rutas, nombres de ratios / nombres de ratios pure
folder = 'C:/Users/P900017/Documents/Python Scripts/credit_rating_clf/base/'
feat_key = pd.read_csv(folder + 'data/features.csv', sep=',', index_col = ["Feature"], encoding = "latin1")
# Encoder para calificaciones:
le = pd.read_csv(folder + 'data/lab_encoder.csv', sep=',', index_col = 0, encoding = "latin1")
# Datos de entrenamiento:
data_em = pd.read_csv(folder + 'data/data_em_1212_0119.csv', sep=',', index_col = ["Fecha", 'Ticker'], encoding = "latin1")

## Visualización de clases para mayor insight (histograma y estadisticos basicos)
#print(data_em.describe())
#data_em['IssuerRating'].value_counts().plot(kind='bar')
#plt.show()

## Reduccion de variables
# Variables disponibles para agregar a to_del
#   "Ratio1", "Ratio2", "Ratio3", "Ratio4", "Ratio5", "Ratio6", 
#   "Ratio7", "Ratio8", "Ratio9", "Ratio10", "Ratio11", "Ratio12", "Ratio13"
# Automáticamente toma los LTM +13
#Para no reducir variables, to_del =[] 
to_del = ["Ratio7", "Ratio13"]
if to_del:
    data_em, feat_key = feat_elim(data_em, feat_key, to_del)

# Eliminar NaN desde ya
#data_em['NA'] = False
#for i in range(len(data_em)):
#    for obs in list(data_em.iloc[i]):
#        if pd.isnull(obs):
#            data_em.iat[i, data_em.columns.get_loc('NA')] = True
#data_em = data_em[data_em.NA == False]
#del data_em["NA"]

## Remover desviaciones atípicas para algunos ratios
# Lista de ratios solo comprueba el if
#Usar | para varias condiciones
#q = 98
#critical_feat = []
#if critical_feat:
#    data_em = data_em[data_em.Ratio3 < np.percentile(data_em.Ratio3,q)]
    
# inputs para las funciones de training
remove_nan = True # Remover filas con datos faltantes.
n_estimators = 500 # Número de árboles de entrenamiento
min_samples_leaf = 2
model_file = folder + 'model/actual_rf_em.sav' # Modelo.
sov_encoder_file = folder + 'model/sov_lab_encoder_em.sav' # Encoder de rating soberano.
output_test = folder + 'output/pred_test.csv' # Archivo de salida con prediciones.
#LIME train set
train_set = folder + 'explainer/X_train_actual.sav' # training set, depende del modelo utilizado

# Training original de rating_functions
model_training(data_em, feat_key, le, remove_nan, output_test, 
               model_file, train_set, sov_encoder_file,
               n_estimators = n_estimators, min_samples_leaf = min_samples_leaf,
               permut=True, shuffle_sample=False, conf_matrix = True)
