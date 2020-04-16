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
    # Fill activity column by the mean
    dailyId = dailyId.fillna(dailyId.activity.mean())
    # Append frame to the id list
    idList.append(dailyId)

# Concat ids again into one frame
daily = pd.concat(idList)
# Add id column back
daily.insert(loc=0, column='id', value=colId)
# Set index as id and time
daily = daily.reset_index().set_index(['id', 'time'])
# Set new columns
newColumns = ['avgValence', 'avgActivity', 'avgArousal', 'avgScreen']
# Set lookup columns for newColumns
lookupColumns = ['valence', 'activity', 'arousal', 'screen']
# Put them into the frame
for column in newColumns:
    daily[column] = np.nan

# Refresh idList
idList = list()

# Iterate through ids to fill the newColumns
# according to the previous 5 days
for id in ids:
    # Set frame to relevant user daily information
    dailyId = daily.loc[id]
    # Roll over the frame to fill the newColumns
    for index, col in enumerate(newColumns):
        # Apply roller to the related column
        dailyId[col] = dailyId[lookupColumns[index]].rolling('5D',
                                                             min_periods=5).mean()
    # Append frame to the idList
    idList.append(dailyId)

# Concat ids again into one frame
daily = pd.concat(idList)
# Drop NaN rows from avgColumns
# values from the beginning of rolling
daily.dropna(inplace=True)

# Sort frame according to time
daily = daily.sort_values(by='time')
# Drop the time column
daily = daily.reset_index().drop(['time'], axis=1)

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
