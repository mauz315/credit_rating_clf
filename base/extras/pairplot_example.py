# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 09:43:02 2019

@author: P900017
"""
import pandas as pd
import seaborn as sns

# Data explorer: pairplot

# Pandas is used for data manipulation

# Read in data as a dataframe
features = pd.read_csv('extras/temps_extended.csv')
print(features.head(5))
# Create columns of categories for pair plotting colors
seasons = []
for month in features['month']:
    if month in [1, 2, 12]:
        seasons.append('winter')
    elif month in [3, 4, 5]:
        seasons.append('spring')
    elif month in [6, 7, 8]:
        seasons.append('summer')
    elif month in [9, 10, 11]:
        seasons.append('fall')
# Will only use six variables for plotting pairs
reduced_features = features[['temp_1', 'prcp_1', 'ws_1', 'average', 'friend', 'actual']]
reduced_features['season'] = seasons
# Use seaborn for pair plots
sns.set(style="ticks", color_codes=True);
# Create a custom color palete
palette = sns.xkcd_palette(['dark blue', 'dark green', 'gold', 'orange'])
# Make the pair plot with a some aesthetic changes
sns.pairplot(reduced_features, hue = 'season', diag_kind = 'kde', palette= palette, plot_kws=dict(alpha = 0.7),
                   diag_kws=dict(shade=True))
