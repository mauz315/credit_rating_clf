# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 15:43:18 2019

@author: P900017
"""

# LIME explainer

import sklearn
import sklearn.ensemble
import numpy as np
import pandas as pd
from sklearn.externals import joblib
import lime
import lime.lime_tabular

# Specify files to import
folder = 'C:/Users/P900017/Documents/Python Scripts/credit_rating_clf/base/'
model_file = folder + 'model/original_rf_em.sav'
train_set = 'X_train.sav'
#test_set = 'X_test.sav'
feat_names = folder + 'data/features.csv'
cat_names = folder + 'data/lab_encoder.csv'
pred_value = '...'

# Load model, train set and predictions
model = joblib.load(folder + model_file)
X_train = joblib.load(folder + train_set)
#X_test = joblib.load(folder + test_set)
sov_lab_encoder = joblib.load(folder + 'sov_lab_encoder_em.sav')
# Line-up the feature and categories names
feat_key = pd.read_csv(folder[:-6] + feat_names, sep=',', index_col = ["Feature"], encoding = "latin1")
feature_names = list(feat_key.Key)

data = pd.read_csv('pred_testing/avh/avh_rating_pred.csv', sep=',', index_col=["Indicadores"], encoding="latin1")
X_new = np.array(data.loc[feat_key.index].T)
if sov_lab_encoder is not None:
    pos_sr = feat_key.index.get_loc(feat_key[feat_key["Key"] == 'SovereignRating'].index[0])
    sob_rating = X_new[:, pos_sr].copy()
    X_new[:, pos_sr] = sov_lab_encoder.transform(X_new[:, pos_sr])
X_new = X_new.astype('float')

# features_names = sum([feature_names_key], [])
# print(features_names)
# Encoder para calificaciones:
le = pd.read_csv(folder[:-6] + cat_names, sep=',', index_col=0, encoding="latin1")
class_names = list(le.index)

# Create the lambda function and the Lime explainer
explainer = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=feature_names, class_names=class_names,
                                                   discretize_continuous=True)
# predict_fn_rf = lambda x: model.predict_proba(x).astype(float)
# explainer = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=feature_names,
#                                                   class_names=class_names, categorical_features=columns,
#                                                   categorical_names=feature_names_cat, kernel_width=3)

# Explaining prediction with Lime
i = 4
exp = explainer.explain_instance(X_new[i], model.predict_proba, num_features=5, top_labels=1)
# exp.show_in_notebook(show_table=True, show_all=False)
exp.save_to_file('output/lime_test1.html')