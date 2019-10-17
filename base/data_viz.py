# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 09:43:02 2019

@author: P900017
"""
import pandas as pd
#import numpy as np
import seaborn as sns

# Data explorer: pairplot
# Ratings a evaluar
search_ratings = ["B-", "B+", "BB-","BBB+", "BB+", "BBB-", "BBB"]
ratios = ['Ratio3', 'Ratio4', 'Ratio5']

# Read in data as a dataframe
feat_key = pd.read_csv('data/features.csv', sep=',', index_col = ["Feature"], encoding = "latin1")
features = pd.read_csv('data/data_em_1212_1218.csv')
features.describe()
print(features.head(5))

# Create columns of categories for pair plotting colors
categories = []
for IssuerRat in features['IssuerRating']:
    if IssuerRat in search_ratings:
        categories.append(1)
    else:
        categories.append(0)

features["Target"] = categories
reduced_features = features[features["Target"] != 0]
ratios.append('IssuerRating')  
reduced_features = reduced_features[ratios]

# Use seaborn for pair plots
sns.set(style="ticks", color_codes=True);
# Create a custom color palete
palette = sns.xkcd_palette(['dark blue', 'dark green', 'gold', 'orange', "black", "dark red", "silver"])
# Make the pair plot with a some aesthetic changes
sns.pairplot(reduced_features, hue = 'IssuerRating', diag_kind = 'kde', palette= palette, plot_kws=dict(alpha = 0.7),
                   diag_kws=dict(shade=True)).savefig("variables.png")
#sns.boxplot(x=features["Ratio3"])
