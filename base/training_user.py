# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 09:54:30 2019

@author: P900017
"""
import pandas as pd
from rating_functions import model_training

# Rutas, nombres de ratios / nombres de ratios pure
folder = 'C:/Users/P900017/Documents/Python Scripts/credit_rating_clf/base/'
feat_key = pd.read_csv(folder + 'data/features.csv', sep=',', index_col = ["Feature"], encoding = "latin1")

# Encoder para calificaciones:
le = pd.read_csv(folder + 'data/lab_encoder.csv', sep=',', index_col = 0, encoding = "latin1")

# Datos de entrenamiento:
data_em = pd.read_csv(folder + 'data/original_data_em.csv', sep=',', index_col = ["Fecha", 'Ticker'], encoding = "latin1")
# data_dm = pd.read_csv(folder + 'research_data_dm.csv', sep=',', index_col = ["Fecha", 'Ticker'], encoding = "latin1")

# inputs para las funciones de training
remove_nan = True # Remover filas con datos faltantes.
perc_train_size = 0.8 # Porcentaje de observaciones para entrenamiento.
n_estimators = 500 # Número de árboles de entrenamiento
min_samples_leaf = 1
model_file = folder + 'model/original_rf_em.sav' # Modelo.
sov_encoder_file = folder + 'model/sov_lab_encoder_em.sav' # Encoder de rating soberano.
output_test = folder + 'output/pred_test.csv' # Archivo de salida con prediciones.

# Training original de rating_functions
model_training(data_em, feat_key, le, remove_nan, perc_train_size,
               output_test, model_file, sov_encoder_file, n_estimators = n_estimators,
               min_samples_leaf = min_samples_leaf, permut=True, shuffle_sample=False)