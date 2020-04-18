#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 14:04:24 2020

@author: anu
"""


# Import libraries
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# ------------------- PREVIOUS OPERATIONS -------------------
# Import dataset
d = pd.read_csv('./data/dataset_mood_smartphone.csv')
# Drop column 'Unnamed: 0'
d = d.drop(columns='Unnamed: 0')

# Convert time values to datetime
d['time'] = pd.to_datetime(d['time'])
# Create pivot table with index id & time
d = d.pivot_table(index=['id', 'time'], columns='variable', values='value')
# Rename columns
d.rename(columns={
    'circumplex.arousal': 'arousal',
    'circumplex.valence': 'valence',
    'appCat.builtin': 'builtin',
    'appCat.communication': 'communication',
    'appCat.entertainment': 'entertainment',
    'appCat.finance': 'finance',
    'appCat.game': 'game',
    'appCat.office': 'office',
    'appCat.other': 'other',
    'appCat.social': 'social',
    'appCat.travel': 'travel',
    'appCat.unknown': 'unknown',
    'appCat.utilities': 'utilities',
    'appCat.weather': 'weather'}, inplace=True)

# Set another frame for daily tracking
daily = d.reset_index().set_index('time').groupby('id').resample('D')

# Delete unnecessary frames
del d

# Define aggregation dict
aggs = {
    'mood': 'mean',
    'arousal': 'mean',
    'valence': 'mean',
    'activity': 'mean',
    'screen': 'sum',
    'call': 'sum',
    'sms': 'sum',
    'builtin': 'sum',
    'communication': 'sum',
    'entertainment': 'sum',
    'finance': 'sum',
    'game': 'sum',
    'office': 'sum',
    'other': 'sum',
    'social': 'sum',
    'travel': 'sum',
    'unknown': 'sum',
    'utilities': 'sum',
    'weather': 'sum'
}

# Apply aggregation
daily = daily.agg(aggs)

# Get rid of rows which does not have mood variable
daily = daily[daily.mood.notnull()]
# ------------------- PREVIOUS OPERATIONS -------------------

# ------------------- FILLING MISSING VALUES -------------------
# Import aggregated data
daily = pd.read_csv('./data/agg.csv')
# Set indexes as id and time
daily = daily.set_index(['id', 'time'])

# For filling missing values with regards to its user
# set variables & empty lists
levels = daily.index.levels
ids = levels[0]  # Index of id
colId = daily.reset_index()['id'].to_numpy()  # List of column id
idList = list()  # Empty list for ids

# Iterate through ids to fill the missing values
for id in ids:
    # Set frame to relevant user daily information
    dailyId = daily.loc[id]
    # Set a dictionary to pass it into fillna method
    values = {
            'arousal': dailyId.arousal.mean(),
            'valence': dailyId.valence.mean(),
            'activity': dailyId.activity.mean()
        }
    # Fill columns by the mean
    dailyId = dailyId.fillna(value=values)
    # Append frame to the id list
    idList.append(dailyId)
    
# Concat ids again into one frame
daily = pd.concat(idList)
# Add id column back
daily.insert(loc=0, column='id', value=colId)
# Set index as id and time
daily = daily.reset_index().set_index(['id', 'time'])
# ------------------- FILLING MISSING VALUES -------------------

# ------------------- ADDING NEW FEATURES -------------------
# Import aggregated and filled missing values data
daily = pd.read_csv('./data/aggFilled.csv')
# Set time as index
daily = daily.set_index('time')

# Add sum of entertainment, social, and game to new column
daily['sumOfEntSocGa'] = daily['entertainment'] + daily['social'] + daily['game']
# Add sum of call and sms to new column
daily['sumofCallSMS'] = daily['call'] + daily['sms']
# Add binary column to indicate whether it is a weekday or weekend
daily['weekday'] = ((pd.DatetimeIndex(daily.index).dayofweek) // 5 == 1).astype(int)
# Add previous day of mood
daily['prevMood'] = daily.mood.shift(1)

# Set new columns for average from past days
newColumns = ['avgMood', 'avgValence', 'avgActivity', 'avgArousal',
              'avgScreen', 'avgAppSum', 'avgCallSMS']
# Set lookup columns for newColumns
lookupColumns = ['mood', 'valence', 'activity', 'arousal',
                 'screen', 'sumOfEntSocGa', 'sumofCallSMS']
# Put them into the frame
for column in newColumns:
    daily[column] = np.nan

# Set ids list
ids = daily.id_label.unique()
# Iterate through newColumns to fill them
# according to the previous 3 & 5 days
# Roll over the frame to fill the newColumns
for index, col in enumerate(newColumns):
    # Apply roller to the related column
    daily[col] = daily[lookupColumns[index]].rolling(5).mean()
    
# Delete the first 5 rows of each user
for id in ids:
    daily[daily['id_label'] == id] = daily[daily['id_label'] == id].iloc[5:]
    
# Drop NaN columns
daily = daily.dropna()
# ------------------- ADDING NEW FEATURES -------------------

# ------------------- FEATURE SELECTION -------------------
# Split the data by independent and dependent columns
X = daily.loc[:, daily.columns != 'mood']  # independent columns
y = daily.loc[:, 'mood']  # target column

# Create LabelEncoder to be able to fit the model later
lab_enc = preprocessing.LabelEncoder()
# Encode the target
y = lab_enc.fit_transform(y)

# Create model
model = ExtraTreesClassifier()
# Fit the model
model.fit(X, y)

# Print the importance of features
print(model.feature_importances_)  # use inbuilt class feature_importances of tree based classifiers
# Plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()
# ------------------- FEATURE SELECTION -------------------
